# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 11:19:45 2021

@author: Fiona
"""

import game
import numpy as np
import matplotlib.pyplot as plt
import game_logging

NUMBER_OF_ATTEMPTS = 100
MARKER = 'o'
COLOR = 'm'
BACKGROUND_COLOR = 'k'
EDGE_COLOR='w'
FONT_COLOR='w'
FIG_SIZE = [6, 4]
FONTSIZE = 18
LINEWIDTH = 3
TITLE_SIZE = 25
TITLE_DICT={'fontsize': TITLE_SIZE, 'color': FONT_COLOR}


class TestStrategy():
    def run_test(number_of_attempts=100, cards_in_hand=6,
                 number_of_players=4, number_of_piles=4, cards_per_turn=2,
                 number_of_cards=100):
        win_array = np.zeros(number_of_attempts)
        for i in range(number_of_attempts):
            my_strategy = game.PlayWithMetricStrategy(
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
            print(np.sum(win_array))
            print(np.sum(win_array)/number_of_attempts)
        return np.sum(win_array)/number_of_attempts
#%% plotting helper

def draw_plot(data, label, x_label, y_label, position):
    ax = fig.add_subplot(position, facecolor=BACKGROUND_COLOR)
    ax.tick_params(color=FONT_COLOR, labelcolor=FONT_COLOR)
    ax.plot(data, marker=MARKER, color=COLOR, label=label, linewidth=LINEWIDTH)
    ax.legend(labelcolor=COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(EDGE_COLOR)
    ax.set_ylabel(y_label)
    ax.set_ylabel(x_label)
    ax.set_title(label)

#%% main

if __name__ == "__main__":
    with game_logging.ContextManager() as manager:
        number_of_cards_array=[*range(10, 110, 10)]
        number_of_players_array=[*range(1, 11, 1)]
        number_of_piles_array=[*range(2, 22, 2)]
        number_of_cards_winning_percentage = np.zeros(10)
        number_of_players_winning_percentage = np.zeros(10)
        number_of_piles_winning_percentage = np.zeros(10)
        for i, j in enumerate(number_of_cards_array):
            number_of_cards_winning_percentage[i-1] = TestStrategy.run_test(
                number_of_attempts=NUMBER_OF_ATTEMPTS, number_of_cards=j)
        # for i, j in enumerate(number_of_players_array):
        #     number_of_players_winning_percentage[i-1] = TestStrategy.run_test(
        #         number_of_attempts=NUMBER_OF_ATTEMPTS, number_of_players=j)
        # for i, j in enumerate(number_of_piles_array):
        #     number_of_cards_winning_percentage[i-1] = TestStrategy.run_test(
        #         number_of_attempts=NUMBER_OF_ATTEMPTS, number_of_piles=j)




    #%% plotting
    label =['number_of_cards', 'number_of_players', 'number_of_piles']
    x_label = ['number_of_cards', 'number_of_players', 'number_of_piles']
    y_label = ['success percentage','success percentage','success percentage']
    data= [number_of_cards_winning_percentage, number_of_players_winning_percentage, number_of_piles_winning_percentage]
    position=[111]# [311, 312, 313]
    fig = plt.figure(figsize=FIG_SIZE, facecolor=BACKGROUND_COLOR, edgecolor=EDGE_COLOR)
    i=0
    draw_plot(data[i], label[i], x_label[i], y_label[i], position[i])

    plt.show()
