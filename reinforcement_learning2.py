# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 16:35:19 2022

@author: Fiona
"""

#https://github.com/H3OX/dqnagent

import the_game
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import LeakyReLU
import tensorflow as tf

import numpy as np
import random
from collections import deque



def get_tensor(self, game):
    tensor_list=[]
    for player in game.players:
        hand=player.hand
        while len(hand)<game.cards_in_hand:
            hand.append(0)
        tensor_list.extend(hand)
    for pile in game.piles:
        tensor_list.append(pile.top_card)
    input_tensor=tf.convert_to_tensor(tensor_list)
    return input_tensor


class Environment():
    def __init__(self):
        CARDS_IN_HAND = 2
        NUMBER_OF_PLAYERS = 1
        NUMBER_OF_PILES = 2
        CARDS_PER_TURN = 1
        NUMBER_OF_CARDS = 10
        self.game=the_game.Game(cards_in_hand=CARDS_IN_HAND,
        number_of_players=NUMBER_OF_PLAYERS,
        number_of_piles=NUMBER_OF_PILES,
        cards_per_turn=CARDS_PER_TURN,
        number_of_cards=NUMBER_OF_CARDS)
        self.game.current_player=self.game.players[0]
        self.STATE_SPACE_SIZE=(1, self.game.number_of_players*self.game.cards_in_hand+self.game.number_of_piles, 1)
        self.ACTION_SPACE_SIZE=3 # 3 neurons, one each for pile, card, want_to_draw
        self.MOST_NEGATIVE_REWARD=-(self.game.number_of_cards+1)
        self.game_over=False


    def reset(self):
        self.__init__()
    def step(self, action_vector):
        """

        Parameters
        ----------
        action_vector : tensor
            first element is the pile, second the card, third the boolean want_to_draw.
        Takes the action if the card is playable, otherwise does nothing and returns a large negative reward

        Returns
        -------
        reward : int
            the reward for having taken this action

        """
        pile_number=action_vector[0]
        card=action_vector[1]
        want_to_draw=action_vector[2]
        if pile_number<0 or pile_number>len(self.game.piles):
            return self.game , self.MOST_NEGATIVE_REWARD, False
        elif not self.game.card_playable(self.game.current_player, self.game.piles[pile_number], card):
            return self.game , self.MOST_NEGATIVE_REWARD, False
        else:
            reward=self.reward(card, self.game.piles[pile_number])
            self.game.play_card(self.game.piles[pile_number], card, want_to_draw)
            if self.game.finished:
                self.game_over=True
            return self.game , reward, self.game_over

    def reward(card, pile): #this will be negative except for jumps
        if isinstance(pile, the_game.IncreasingPile):
            return -(card-pile.top_card)
        else:
            return -(pile.top_card-card)

train_env=Environment()


class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size
		self.action_size = 3 # hold, buy, sell
		self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

		self.model = load_model(model_name) if is_eval else self._model()

	def _model(self):
		model = Sequential()
		model.add(Dense(units=32, input_dim=self.state_size, activation='relu'))
		model.add(Dense(units=16, activation='relu'))
		model.add(Dense(units=8, activation='relu'))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.005))
		return model

	def act(self, state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)

		options = self.model.predict(state)
		return np.argmax(options[0])

	def expReplay(self, batch_size):
		mini_batch = random.sample(self.memory, batch_size)

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay


import sys
import numpy as np
import os
import agent

print(os.getcwd())

def formatPrice(n):
	return ("-" if n < 0 else "") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))
	return vec

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])



#stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = agent.Agent(train_env.STATE_SPACE_SIZE)
#data = getStockDataVec(stock_name)
#l = len(data) - 1
batch_size = 32

for e in range(100 + 1):
#	print("Эпоха " + str(e))
	state=get_tensor(train_env.game)

	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		# sit
		next_state = get_tensor(train_env.game)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			print("Покупка: " + formatPrice(data[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			print("Продажа: " + formatPrice(data[t]) + " | Прибыль: " + formatPrice(data[t] - bought_price))

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print ("Суммарная прибыль: " + formatPrice(total_profit))

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

		agent.model.save("model_ep" + str(e))