# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 11:19:45 2021

@author: Fiona
"""

import the_game
import numpy as np
import matplotlib.pyplot as plt
import game_logging
import plotting_constants as c
import os
# importing cProfile
import cProfile #to check what takes longest

NUMBER_OF_GAMES = 100

# #%% make plots pretty
# NUMBER_OF_GAMES = 100
# MARKER = 'o'
# COLOR= 'w'
# PLOTTING_COLORS=['g', 'b', 'r', 'm', 'c']
# BACKGROUND_COLOR = 'k'
# EDGE_COLOR='w'
# FONT_COLOR='w'
# FIG_SIZE = [6, 4]
# FONTSIZE = 18
# LINEWIDTH = 3
# TITLE_SIZE = 25
# TITLE_DICT={'fontsize': TITLE_SIZE, 'color': FONT_COLOR}


#%% running tests
class TestStrategy():


    def run_one_test (number_of_games=100, cards_in_hand=6,
                     number_of_players=4, number_of_piles=4, cards_per_turn=2,
                     number_of_cards=100, strategy=the_game.PlayWithDistanceCutoffStrategy, return_jump_count=True):

        win_array = np.zeros(number_of_games)
        jump_array= np.zeros(number_of_games)
        for i in range(number_of_games):
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
            jump_array[i]=my_strategy.game.jump_counter

        number_of_wins=np.sum(win_array)/number_of_games

        average_jumps=np.sum(jump_array)/number_of_games
        print(average_jumps)
        # if return_jump_count==True:
        #     return average_jumps#jump_array
        return number_of_wins, jump_array

    def run_tests(strategies, x_data, variable="number_of_cards", points_for_error_calculation=4):

        with game_logging.ContextManager() as manager:
            jump_count=np.zeros((len(strategies), len(x_data), points_for_error_calculation*NUMBER_OF_GAMES))
            winning_percentage = np.zeros((len(strategies), len(x_data), points_for_error_calculation))
            variable_dictionary=dict(number_of_games=100, cards_in_hand=6,number_of_players=4,
                                     number_of_piles=4, cards_per_turn=2,number_of_cards=100, strategy=the_game.PlayWithDistanceCutoffStrategy, points_for_error_calculation=points_for_error_calculation)
            variable_dictionary[variable]=None
            if len(strategies)>1:
                variable_dictionary['strategy']=None
            if variable=="number_of_cards":
                for k in range(points_for_error_calculation):
                    for i, j in enumerate(x_data):
                        for s, strategy in enumerate(strategies):
                            winning_percentage[ s, i, k], jump_count[s,i,NUMBER_OF_GAMES*k:NUMBER_OF_GAMES*(k+1)] = TestStrategy.run_one_test(
                                number_of_games=NUMBER_OF_GAMES, number_of_cards=j, strategy=strategy)

            elif variable=="number_of_players":
                for k in range(points_for_error_calculation):
                    for i, j in enumerate(x_data):
                        for s, strategy in enumerate(strategies):
                            winning_percentage[ s, i, k] ,jump_count[s,i,NUMBER_OF_GAMES*k:NUMBER_OF_GAMES*(k+1)]= TestStrategy.run_one_test(
                                number_of_games=NUMBER_OF_GAMES, number_of_players=j, strategy=strategy, number_of_cards=70)
                variable_dictionary['number_of_cards']=70

            elif variable=="number_of_piles":
                for k in range(points_for_error_calculation):
                    for i, j in enumerate(x_data):
                        for s, strategy in enumerate(strategies):
                            winning_percentage[ s, i, k],jump_count[s,i,NUMBER_OF_GAMES*k:NUMBER_OF_GAMES*(k+1)] = TestStrategy.run_one_test(
                                number_of_games=NUMBER_OF_GAMES, number_of_piles=j, strategy=strategy)

            elif variable=="cards_in_hand":
                for k in range(points_for_error_calculation):
                    for i, j in enumerate(x_data):
                        for s, strategy in enumerate(strategies):
                            winning_percentage[ s, i, k],jump_count[s,i,NUMBER_OF_GAMES*k:NUMBER_OF_GAMES*(k+1)] = TestStrategy.run_one_test(
                                number_of_games=NUMBER_OF_GAMES, cards_in_hand=j, strategy=strategy)

            elif variable=="cards_per_turn":
                for k in range(points_for_error_calculation):
                    for i, j in enumerate(x_data):
                        for s, strategy in enumerate(strategies):
                            winning_percentage[ s, i, k],jump_count[s,i,NUMBER_OF_GAMES*k:NUMBER_OF_GAMES*(k+1)] = TestStrategy.run_one_test(
                                number_of_games=NUMBER_OF_GAMES, cards_per_turn=j, number_of_cards=50, strategy=strategy)


            return winning_percentage, jump_count, variable_dictionary
#%% plotting helper

def draw_plot(fig, x_data, y_data, labels, x_label, y_label, title, variable_dictionary, jump_plot, position=111):
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
    ax = fig.add_subplot(position, facecolor=c.BACKGROUND_COLOR)
    ax.tick_params(color=c.FONT_COLOR, labelcolor=c.FONT_COLOR)
    std_error=np.std(y_data, axis=2) / np.sqrt(np.size(y_data[0,0,:]))
    y_data=y_data.mean(axis=2)
    for i in range(len(labels)):

        ax.plot(x_data, y_data[i,:], marker=c.MARKER, color=c.PLOTTING_COLORS[i], label=labels[i], linewidth=c.LINEWIDTH)
        ax.errorbar(x_data, y_data[i, :], yerr=std_error[i], capsize=4, color=c.PLOTTING_COLORS[i])
    ax.legend(labels, labelcolor=c.PLOTTING_COLORS[:len(labels)])

    for spine in ax.spines.values():
        spine.set_edgecolor(c.EDGE_COLOR)
    ax.set_ylabel(y_label, color=c.COLOR)
    ax.set_xlabel(x_label, color=c.COLOR)
    ax.set_ylim(bottom=0)
    ax.set_title(title, color=c.COLOR)
    plt.title(title)
    #box_text_vars=["number_of_games","cards_in_hand", "number_of_players", "number_of_piles", "cards_per_turn", "number_of_cards"]
    #box_text_vals=[test_strategy.number_of_games, test_strategy.cards_in_hand, test_strategy.number_of_players, test_strategy.number_of_piles, test_strategy.number_of_cards, test_strategy.cards_per_turn]
    box_text=""
    for key, val in variable_dictionary.items():
        if not val is None:
            box_text = box_text + key +": "+ str(val) + ", \n"
    if jump_plot==False:
        #if plotting wins, can't do error bars for number of games, so we have split the number of games games up into points_for_error_calculation subsets
        box_text=box_text+ "\n error bars show standard error \n for n="+str(variable_dictionary['points_for_error_calculation']) + " data points"
    else:
        #if plotting jumps, error bars are for number of games
        box_text=box_text+ "\n error bars show standard error \n for n="+str(variable_dictionary['points_for_error_calculation']* NUMBER_OF_GAMES) + " games per point"
    box_style=dict(boxstyle='square', facecolor=c.BACKGROUND_COLOR, alpha=0.5)
    ax.text(1.05, 0.75,box_text,
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes, color='w', bbox=box_style)

#%% main

if __name__ == "__main__":
    #%% prepare plotting
    x_label = ['number_of_cards (in the game)' , 'number_of_players', 'number_of_piles', 'cards_in_hand', 'cards_per_turn']
    y_label_wins = 'success proportion'
    y_label_jumps='average number of jumps per game'
    title_all="comparison between only playing minimum number of cards and potentially playing more cards \n variable: "
    titles=['number_of_cards', 'number_of_players', 'number_of_piles', 'cards_in_hand', 'cards_per_turn']
    #for the number of cards plot, it makes sense to change the number of cards in the game to a lower number, since wedo not have enough samples otherwise
    strategies=[the_game.PlayWithMetricStrategy, the_game.PlayWithDistanceCutoffStrategy]
    x_array=[[*range(10, 150, 10)],[*range(1, 10, 1)], [*range(2, 8, 2)], [*range(4,10,1)], [*range(1,6,1)] ]
    #%% calculate
    for i in range (5):
        winning_percentage, jump_count, variable_dictionary = TestStrategy.run_tests(strategies, x_data=x_array[i], variable= titles[i])

        #%% plot
        print(x_array[i])
        print(winning_percentage)
        fig_wins = plt.figure(figsize=c.FIG_SIZE, facecolor=c.BACKGROUND_COLOR, edgecolor=c.EDGE_COLOR)
        draw_plot(fig_wins, x_array[i], winning_percentage, ["Minimum", "Potentially more"], x_label[i], y_label_wins, title_all+titles[i], variable_dictionary, jump_plot=False)

        plt.show()
        fig_jumps = plt.figure(figsize=c.FIG_SIZE, facecolor=c.BACKGROUND_COLOR, edgecolor=c.EDGE_COLOR)
        draw_plot(fig_jumps, x_array[i], jump_count, ["Minimum", "Potentially more"], x_label[i], y_label_jumps, title_all+titles[i], variable_dictionary, jump_plot=True)

        plt.show()


        my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
        my_figure_folder=os.path.join("figures", "stats")
        file_name=titles[i].replace(" ", "_") + ".png"
        file_name.replace("\n", "_")
        fig_wins.savefig(os.path.join(my_path, my_figure_folder, "wins_"+file_name), bbox_inches='tight')
        fig_jumps.savefig(os.path.join(my_path, my_figure_folder, "jumps_"+file_name), bbox_inches='tight')
