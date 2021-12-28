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

NUMBER_OF_ATTEMPTS = 100
MARKER = 'o'
COLOR= 'w'
COLOR_1 = 'g'
COLOR_2 = 'b'
BACKGROUND_COLOR = 'k'
EDGE_COLOR='w'
FONT_COLOR='w'
FIG_SIZE = [6, 4]
FONTSIZE = 18
LINEWIDTH = 3
TITLE_SIZE = 25
TITLE_DICT={'fontsize': TITLE_SIZE, 'color': FONT_COLOR}


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
        print(np.sum(win_array)/number_of_attempts)
        return np.sum(win_array)/number_of_attempts

    def run_tests(number_of_cards_array):
        with game_logging.ContextManager() as manager:

            number_of_players_array=[*range(1, 10, 1)]
            number_of_piles_array=[*range(2, 8, 2)]
            number_of_cards_winning_percentage_first_strategy = np.zeros(len(number_of_cards_array))
            number_of_cards_winning_percentage_second_strategy = np.zeros(len(number_of_cards_array))

            number_of_players_winning_percentage = np.zeros(len(number_of_players_array))
            number_of_piles_winning_percentage = np.zeros(len(number_of_piles_array))
            for i, j in enumerate(number_of_cards_array):
                number_of_cards_winning_percentage_first_strategy[i] = TestStrategy.run_one_test(
                    number_of_attempts=NUMBER_OF_ATTEMPTS, number_of_cards=j, strategy=the_game.PlayWithMetricStrategy)
                number_of_cards_winning_percentage_second_strategy[i] = TestStrategy.run_one_test(
                    number_of_attempts=NUMBER_OF_ATTEMPTS, number_of_cards=j, strategy=the_game.PlayWithDistanceCutoffStrategy)

            # for i, j in enumerate(number_of_players_array):
            #     number_of_players_winning_percentage[i] = TestStrategy.run_test(
            #         number_of_attempts=NUMBER_OF_ATTEMPTS, number_of_players=j, number_of_cards=50)
            # for i, j in enumerate(number_of_piles_array):
            #     number_of_cards_winning_percentage[i] = TestStrategy.run_test(
            #         number_of_attempts=NUMBER_OF_ATTEMPTS, number_of_piles=j, number_of_cards=100)
            return number_of_cards_winning_percentage_first_strategy, number_of_cards_winning_percentage_second_strategy

#%% plotting helper

def draw_plot(x_data, y_data, label, x_label, y_label, position):
    ax = fig.add_subplot(position, facecolor=BACKGROUND_COLOR)
    ax.tick_params(color=FONT_COLOR, labelcolor=FONT_COLOR)
    ax.plot(x_data, y_data[:,0], marker=MARKER, color=COLOR_1, label=label, linewidth=LINEWIDTH)
    ax.plot(x_data, y_data[:,1], marker=MARKER, color=COLOR_2, label=label, linewidth=LINEWIDTH)

    ax.legend(labelcolor=[COLOR_1, COLOR_2])
    for spine in ax.spines.values():
        spine.set_edgecolor(EDGE_COLOR)
    ax.set_ylabel(y_label, color=COLOR)
    ax.set_ylabel(x_label, color=COLOR)
    ax.set_title(label, color=COLOR)
    plt.title(label)

#%% main

if __name__ == "__main__":
    number_of_cards_array=[*range(10, 110, 10)]
    number_of_cards_winning_percentage_first_strategy, number_of_cards_winning_percentage_second_strategy = TestStrategy.run_tests(number_of_cards_array)



    #%% plotting
    label =['number_of_cards', 'number_of_players', 'number_of_piles']
    x_label = ['number_of_cards', 'number_of_players', 'number_of_piles']
    y_label = ['success percentage','success percentage','success percentage']
    #x_data= [number_of_cards_winning_percentage_first_strategy, number_of_players_winning_percentage, number_of_piles_winning_percentage]
    position=[111, 111,111]# [311, 312, 313]
    fig = plt.figure(figsize=FIG_SIZE, facecolor=BACKGROUND_COLOR, edgecolor=EDGE_COLOR)
    i=0
    draw_plot(number_of_cards_array, np.transpose(np.array([number_of_cards_winning_percentage_first_strategy, number_of_cards_winning_percentage_second_strategy])), ["PlayWithMetric", "PlayWithDistanceCutoff"], x_label[i], y_label[i], position[i])

    plt.show()

    my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
    my_figure_folder="figures"
    my_file = 'graph.png'
    fig.savefig(os.path.join(my_path, my_figure_folder, my_file))