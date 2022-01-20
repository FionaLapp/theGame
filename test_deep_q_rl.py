# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:10:06 2022

@author: Fiona

Doing some plotting for the Deep-Q-RL-model from attempt 3.


"""


from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import tensorflow as tf
import random
from tqdm import tqdm
import time
import pdb

import the_game
import matplotlib.pyplot as plt
import os
import plotting_constants as c
import pandas as pd
from attempt_3_deep_q_rl import Environment
import attempt_3_deep_q_rl

#%%
CARDS_IN_HAND = 6
NUMBER_OF_PLAYERS = 2
NUMBER_OF_PILES = 4
CARDS_PER_TURN = 1
NUMBER_OF_CARDS = 70
NUMBER_OF_ACTIONS=6
variable_dictionary=dict(cards_in_hand=CARDS_IN_HAND,
number_of_players=NUMBER_OF_PLAYERS,
number_of_piles=NUMBER_OF_PILES,
cards_per_turn=CARDS_PER_TURN,
number_of_cards=NUMBER_OF_CARDS)
model_name="models/no_of_cards_70___147.00max__110.12avg__-74.00min__1642510245.model"#"no_of_cards_70___100.00max___14.00avg____0.00min__1642446958.model"
#model_name="models/no_of_cards_100___100.00max____4.00avg____0.00min__1642444639.model"
#model_name="models/model___100.00max__100.00avg__100.00min__1642437186.model"
#%%

class DRLStrategy(the_game.Strategy):
    def __init__(self, cards_in_hand=CARDS_IN_HAND, number_of_players=NUMBER_OF_PLAYERS, number_of_piles=NUMBER_OF_PILES,
                   cards_per_turn=CARDS_PER_TURN, number_of_cards=NUMBER_OF_CARDS):
          self.env=Environment()
          self.model=tf.keras.models.load_model(model_name)

    def predict_action(self, state):
        return np.argmax(self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0][0])


    def play(self):
        self.actions=np.zeros(6)
        while True:
              if self.env.game.game_finished():
                  return self.actions
              else:

                  state=attempt_3_deep_q_rl.get_state(self.env.game)
                  action=self.predict_action(state)
                  self.actions[action]+=1
                  self.env.step(action)



    def start_game(self):
        self.env.game.current_player = self.env.game.players[0]
        self.env.game.calculate_basic_metric()

def run_tests(number_of_tests, strategy=DRLStrategy):
    strategy=DRLStrategy()
    win_array=np.zeros(number_of_tests)
    jump_array=np.zeros(number_of_tests)
    actions_array=np.zeros((number_of_tests, 6))

    for i in range(number_of_tests):

        strategy.env.reset()
        strategy.start_game()
        actions=strategy.play()
        print(actions)
        actions_array[i,:]=actions
        jump_array[i]=strategy.env.game.jump_counter
        if strategy.env.game.game_won():
            win_array[i]=1
        print("{}% done".format((i+1)/number_of_tests*100))
    return win_array, jump_array, actions_array

number_of_tests=50
win_array=np.zeros((number_of_tests,4))
jump_array=np.zeros((number_of_tests,4))
actions_array=np.zeros((number_of_tests, NUMBER_OF_ACTIONS, 4))

for i in range(4):
    win_array[:,i], jump_array[:,i], actions_array[:,:,i]= run_tests(number_of_tests)
#%%
def format_plot(y_data, x_label, y_label, title, variable_dictionary):
        position=111

        fig=plt.figure(figsize=c.FIG_SIZE, facecolor=c.BACKGROUND_COLOR, edgecolor=c.EDGE_COLOR)
        ax = fig.add_subplot(position, facecolor=c.BACKGROUND_COLOR)
        ax.tick_params(color=c.FONT_COLOR, labelcolor=c.FONT_COLOR)
        ax.legend(labelcolor=c.PLOTTING_COLORS[0])

        for spine in ax.spines.values():
            spine.set_edgecolor(c.EDGE_COLOR)
        ax.set_ylabel(y_label, color=c.COLOR)
        ax.set_xlabel(x_label, color=c.COLOR)
        ax.set_ylim(bottom=0)
        ax.set_title(title, color=c.COLOR)
        plt.title(title)
        variable_dictionary=dict(cards_in_hand=CARDS_IN_HAND,
        number_of_players=NUMBER_OF_PLAYERS,
        number_of_piles=NUMBER_OF_PILES,
        cards_per_turn=CARDS_PER_TURN,
        number_of_cards=NUMBER_OF_CARDS)
        box_text= "Test of Reinforcement learning model using deep q learning: \n"
        for key, val in variable_dictionary.items():
            if not val is None:
                box_text = box_text + key +": "+ str(val) + ", \n"

        box_style=dict(boxstyle='square', facecolor=c.BACKGROUND_COLOR, alpha=0.5)
        ax.text(1.05, 0.75,box_text,
         horizontalalignment='left',
         verticalalignment='center',
         transform = ax.transAxes, color='w', bbox=box_style)


        #plt.plot(aggregate_episode_wins['ep'], aggregate_episode_wins['max'], label="max rewards")
        #plt.plot(aggregate_episode_wins['ep'], aggregate_episode_wins['min'], label="min rewards")
        plt.legend(loc=4)
        plt.bar(1, np.mean(y_data), ecolor=c.FONT_COLOR, color=c.PLOTTING_COLORS[0])
        ax.errorbar(1, np.mean(y_data), yerr=get_std_error(np.mean(y_data)), capsize=4, color=c.FONT_COLOR)

        plt.show()

        my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
        my_figure_folder=(os.path.join("figures", "q_table_graphs"))
        file_name  ="deep_q_test_attempt_3.png"
        plt.show()
        fig.savefig(os.path.join(my_path, my_figure_folder, ""+file_name), bbox_inches='tight')


y_labels=["success_proportion", "jump_count", "actions"]
x_label="Deep Q RL, model trained on 100 episodes"
titles=["Average wins using q-table reinforcement learning","Average jumps using q-table reinforcement learning"
"Average actions using q-table reinforcement learning"
]

def get_std_error(y_data):
    return np.std(y_data) / np.sqrt(np.size(y_data))


win_proportion=win_array.sum(axis=0)/number_of_tests



format_plot(jump_array, x_label, y_labels[1], titles[1], variable_dictionary)



