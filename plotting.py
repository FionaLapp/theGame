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


    def run_one_test (number_of_attempts=100, cards_in_hand=6,
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
        print(np.sum(win_array)/number_of_attempts)
        return np.sum(win_array)/number_of_attempts

    def run_tests(strategies, x_data, variable="number_of_cards"):

        with game_logging.ContextManager() as manager:

            winning_percentage = np.zeros((len(strategies), len(x_data)))
            variable_dictionary=dict(number_of_attempts=100, cards_in_hand=6,number_of_players=4,
                                     number_of_piles=4, cards_per_turn=2,number_of_cards=100, strategy=the_game.PlayWithDistanceCutoffStrategy)
            variable_dictionary[variable]=None
            if len(strategies)>1:
                variable_dictionary['strategy']=None
            if variable=="number_of_cards":

                for i, j in enumerate(x_data):
                    for s, strategy in enumerate(strategies):
                        winning_percentage[ s, i] = TestStrategy.run_one_test(
                            number_of_attempts=NUMBER_OF_ATTEMPTS, number_of_cards=j, strategy=strategy)

            elif variable=="number_of_players":

                for i, j in enumerate(x_data):
                    for s, strategy in enumerate(strategies):
                        winning_percentage[ s, i] = TestStrategy.run_one_test(
                            number_of_attempts=NUMBER_OF_ATTEMPTS, number_of_players=j, strategy=strategy, number_of_cards=70)
                variable_dictionary['number_of_cards']=70

            elif variable=="number_of_piles":

                for i, j in enumerate(x_data):
                    for s, strategy in enumerate(strategies):
                        winning_percentage[ s, i] = TestStrategy.run_one_test(
                            number_of_attempts=NUMBER_OF_ATTEMPTS, number_of_piles=j, strategy=strategy)

            return winning_percentage, variable_dictionary
#%% plotting helper

def draw_plot(x_data, y_data, labels, x_label, y_label, title, variable_dictionary, position=111):
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
    ax.legend(labels, labelcolor=PLOTTING_COLORS[:len(labels)])

    for spine in ax.spines.values():
        spine.set_edgecolor(EDGE_COLOR)
    ax.set_ylabel(y_label, color=COLOR)
    ax.set_xlabel(x_label, color=COLOR)
    ax.set_title(title, color=COLOR)
    plt.title(title)
    #box_text_vars=["number_of_attempts","cards_in_hand", "number_of_players", "number_of_piles", "cards_per_turn", "number_of_cards"]
    #box_text_vals=[test_strategy.number_of_attempts, test_strategy.cards_in_hand, test_strategy.number_of_players, test_strategy.number_of_piles, test_strategy.number_of_cards, test_strategy.cards_per_turn]
    box_text=""
    for key, val in variable_dictionary.items():
        if not val is None:
            box_text = box_text + key +": "+ str(val) + ", \n"
    box_text=box_text.strip(",")
    box_style=dict(boxstyle='square', facecolor=BACKGROUND_COLOR, alpha=0.5)
    ax.text(1.05, 0.75,box_text,
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes, color='w', bbox=box_style)

#%% main

if __name__ == "__main__":
    #%% prepare plotting
    x_label = ['number_of_cards (in the game)' , 'number_of_players', 'number_of_piles']
    y_label = 3*['success proportion']
    title_all="comparison between always playing 2 cards and potentially playing more cards \n variable: "
    titles=['number_of_cards', 'number_of_players', 'number_of_piles']
    #for the number of cards plot, it makes sense to change the number of cards in the game to a lower number, since wedo not have enough samples otherwise
    strategies=[the_game.PlayWithMetricStrategy, the_game.PlayWithDistanceCutoffStrategy]
    x_array=[[*range(10, 110, 10)],[*range(1, 10, 1)], [*range(2, 8, 2)] ]
    #%% calculate
    for i in range (1,2):

        winning_percentage, variable_dictionary = TestStrategy.run_tests(strategies, x_data=x_array[i], variable= titles[i])

        #%% plot
        print(x_array[i])
        print(winning_percentage)
        fig = plt.figure(figsize=FIG_SIZE, facecolor=BACKGROUND_COLOR, edgecolor=EDGE_COLOR)
        draw_plot(x_array[i], np.transpose(winning_percentage), ["PlayWithMetric", "PlayWithDistanceCutoff"], x_label[i], y_label[i], title_all+titles[i], variable_dictionary)

        plt.show()

        my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
        my_figure_folder="figures"
        file_name=titles[i].replace(" ", "_") + ".png"
        file_name.replace("\n", "_")
        fig.savefig(os.path.join(my_path, my_figure_folder, file_name))