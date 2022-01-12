# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 11:19:45 2021

@author: Fiona
"""

import the_game
import numpy as np
import matplotlib.pyplot as plt
import game_logging
import os
# importing cProfile
import cProfile #to check what takes longest

#%% make plots pretty
NUMBER_OF_ATTEMPTS = 100
MARKER = 'o'
COLOR= 'w'
PLOTTING_COLORS=['g', 'b', 'r', 'm', 'c']
BACKGROUND_COLOR = 'k'
EDGE_COLOR='w'
FONT_COLOR='w'
FIG_SIZE = [6, 4]
FONTSIZE = 18
LINEWIDTH = 3
TITLE_SIZE = 25
TITLE_DICT={'fontsize': TITLE_SIZE, 'color': FONT_COLOR}


#%% running tests
class TestStrategy():
    def run_one_test(number_of_attempts=100, cards_in_hand=6,
                 number_of_players=4, number_of_piles=4, cards_per_turn=2,
                 number_of_cards=100, strategy=the_game.PlayWithDistanceCutoffStrategy):
        win_array = np.zeros(number_of_attempts)
        for i in range(number_of_attempts):
            my_strategy = strategy(
                cards_in_hand=cards_in_hand,
                number_of_players=number_of_players,
                number_of_piles=number_of_piles, cards_per_turn=cards_per_turn,
                number_of_cards=number_of_cards)
            my_strategy.start_game(my_strategy.game.players[0])
            my_strategy.play()
            game_logging.GameLoggers.strategy_logger.info(
                "Game won: {}".format(my_strategy.game.game_won()))
            game_logging.GameLoggers.strategy_logger.info(
                "players: {}".format(my_strategy.game.print_hands()))
            game_logging.GameLoggers.strategy_logger.info(
                "piles: {}".format(my_strategy.game.print_piles()))
            if my_strategy.game.game_won():
                win_array[i] = 1
        #print(np.sum(win_array)/number_of_attempts)
        return np.sum(win_array)/number_of_attempts

    def run_tests(strategies, number_of_cards_array=[*range(10, 110, 10)], number_of_players_array=[*range(1, 10, 1)], number_of_piles_array=[*range(2, 8, 2)]):
        with game_logging.ContextManager() as manager:

            winning_percentage = np.zeros((len(strategies), len(number_of_cards_array)))

            for i, j in enumerate(number_of_cards_array):
                for s, strategy in enumerate(strategies):
                    winning_percentage[ s, i] = TestStrategy.run_one_test(
                        number_of_attempts=NUMBER_OF_ATTEMPTS, number_of_cards=j, strategy=strategy)


            return winning_percentage
#%% plotting helper

def draw_plot(x_data, y_data, labels, x_label, y_label, title, position=111):
    """


    Parameters
    ----------
    x_data : numpy array (one-dimensional)
        the x-values of the data points
    y_data : numpy array (can be two-dimensional if plotting more than line in one figure).
        Each row represents one line, the row-length must match that of the x-data
    labels : list
        list of labels for the y-data. The length must match the number of rows in the y-data array
    x_label : String
        x axis label
    y_label : String
        y-axis-label
    title : String
        Title of the graph
    position : int
        position of subplot (use 111 if only plotting one subplot)

    Returns
    -------
    None.

    """
    ax = fig.add_subplot(position, facecolor=BACKGROUND_COLOR)
    ax.tick_params(color=FONT_COLOR, labelcolor=FONT_COLOR)
    for i in range(len(labels)):
        ax.plot(x_data, y_data[:,i], marker=MARKER, color=PLOTTING_COLORS[i], label=labels[i], linewidth=LINEWIDTH)

    ax.legend(labelcolor=PLOTTING_COLORS[:len(labels)])
    for spine in ax.spines.values():
        spine.set_edgecolor(EDGE_COLOR)
    ax.set_ylabel(y_label, color=COLOR)
    ax.set_xlabel(x_label, color=COLOR)
    ax.set_title(label, color=COLOR)
    plt.title(title)

#%% main

if __name__ == "__main__":
    strategies=[the_game.PlayWithMetricStrategy, the_game.PlayWithDistanceCutoffStrategy]
    number_of_cards_array=[*range(10, 110, 10)]
    winning_percentage = TestStrategy.run_tests(strategies, number_of_cards_array=[*range(10, 110, 10)])



    #%% plotting
    label =['number_of_cards', 'number_of_players', 'number_of_piles']
    x_label = ['number_of_cards (in the game)' , 'number_of_players', 'number_of_piles']
    y_label = 3*['success percentage']

    title="comparison between always playing 2 cards and potentially playing more cards"
    fig = plt.figure(figsize=FIG_SIZE, facecolor=BACKGROUND_COLOR, edgecolor=EDGE_COLOR)
    i=0
    draw_plot(number_of_cards_array, np.transpose(winning_percentage), ["PlayWithMetric", "PlayWithDistanceCutoff"], x_label[i], y_label[i], title)

    plt.show()

    my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
    my_figure_folder="figures"
    my_file = 'graph.png'
    fig.savefig(os.path.join(my_path, my_figure_folder, my_file))