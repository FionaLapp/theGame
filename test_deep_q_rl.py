# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:10:06 2022

@author: Fiona
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

CARDS_IN_HAND = 6
NUMBER_OF_PLAYERS = 2
NUMBER_OF_PILES = 4
CARDS_PER_TURN = 1
NUMBER_OF_CARDS = 70
model_name="models/no_of_cards_70___100.00max___14.00avg____0.00min__1642446958.model"
#model_name="models/no_of_cards_100___100.00max____4.00avg____0.00min__1642444639.model"
#model_name="models/model___100.00max__100.00avg__100.00min__1642437186.model"
#%%

class DRLStrategy(the_game.Strategy):
    def __init__(self, cards_in_hand=6, number_of_players=4, number_of_piles=4,
                   cards_per_turn=1, number_of_cards=NUMBER_OF_CARDS):
          self.env=Environment()
          self.model=tf.keras.models.load_model(model_name)

    def predict_action(self, state):
        return np.argmax(self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0])


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

def run_tests(number_of_tests):
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

number_of_tests=25
win_array=np.zeros((number_of_tests,4))
jump_array=np.zeros((number_of_tests,4))
actions_array=np.zeros((number_of_tests, 6, 4))

for i in range(4):
    win_array[:,i], jump_array[:,i], actions_array[:,:,i]= run_tests(number_of_tests)



