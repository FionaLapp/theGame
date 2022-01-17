# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 16:10:28 2022

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

CARDS_IN_HAND = 6
NUMBER_OF_PLAYERS = 2
NUMBER_OF_PILES = 4
CARDS_PER_TURN = 1
NUMBER_OF_CARDS = 70



DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'no_of_cards_100'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 100


EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes


RESULTS=[]


class Environment():
    def __init__(self):
        self.game=the_game.Game(cards_in_hand=CARDS_IN_HAND,
        number_of_players=NUMBER_OF_PLAYERS,
        number_of_piles=NUMBER_OF_PILES,
        cards_per_turn=CARDS_PER_TURN,
        number_of_cards=NUMBER_OF_CARDS)
        self.game.current_player=self.game.players[0]
        self.STATE_SPACE_SIZE=NUMBER_OF_CARDS*[6]# for each card, state can be: in drawing_pile, been_played, in my hand, in someone else's hand, top card
        self.ACTION_SPACE_SIZE=6
        self.MOST_NEGATIVE_REWARD=-1 #-(self.game.number_of_cards+1)
        self.game_over=False


    def reset(self):
        self.__init__()
    def step(self, action_number):
        want_to_draw=action_number%2
        action_number=int(action_number/2)
        self.game.calculate_basic_metric()
        metric_matrix=self.game.basic_metric
        less_actions_available=False
        index=None
        for i in range(action_number): #e.g. if action =1 (play second best card), then set best action to impossible

            temp_index = np.unravel_index(np.argmin(metric_matrix, axis=None),
                                     metric_matrix.shape)
            metric_matrix[temp_index]=NUMBER_OF_CARDS+1
            if metric_matrix[temp_index]==np.max(metric_matrix):
                index=temp_index
                less_actions_available=True
                break
        if not less_actions_available:
            index = np.unravel_index(np.argmin(metric_matrix, axis=None),
                                 metric_matrix.shape)

        pile_number = index[1]
        pile = self.game.piles[pile_number]
        card_position = index[0]
        card = self.game.current_player.hand[card_position]


        if not self.game.card_playable(self.game.current_player, self.game.piles[pile_number], card):
            return self.game , self.MOST_NEGATIVE_REWARD, False
        else:
            #print("card played:", card, " on pile", pile_number, " by player ", self.game.players[0])
            reward=0#self.reward(card, self.game.piles[pile_number])
            self.game.play_card(self.game.piles[pile_number], card, want_to_draw)
            #print(self.game.print_hands())
            #print(self.game.print_piles())
            if self.game.game_finished():
                self.game_over=True
                if self.game.game_won():
                    reward=100
                    RESULTS.append(1)
                else:
                    RESULTS.append(0)
            return self.game , reward, self.game_over

    def reward(self, card, pile): #this will be negative except for jumps
        if isinstance(pile, the_game.IncreasingPile):
            return -(card-pile.top_card)
        else:
            return -(pile.top_card-card)

    def reward_if_won(self, card, pile):
        #TODO write a function that returns 0 if card playable and game not finished, 1 if game 1, -1 if game lost, whatever negative if card not playable
        print("not implemented")




def get_state(game):
    state=np.int_(np.zeros(NUMBER_OF_CARDS)) #0: we assume everything is in the drawing pile
    for player in game.players:
        if player==game.current_player:
            for card in player.hand:
                state[card]=1 #1: in my hand
        else:
            for card in player.hand:
                state[card]=2 #2: in someone else's hand
    for pile in game.piles:
        for card in pile.cards[1:]:

            state[card]=4 # card has already been played
        if pile.top_card!=NUMBER_OF_CARDS and pile.top_card!=0:
            state[pile.top_card]=4-pile.pile_multiplier #5 if decreasing, 3 if increasing

    return state


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self._train_dir="/models"
        self._train_step=None
#        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        pass
        #print(stats)
#        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self):
        env=Environment()
        # main model  # gets trained every step
        self.model = self.create_model(env)

        # Target model this is what we .predict against every step
        self.target_model = self.create_model(env)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        print(self.replay_memory)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def create_model(self, env):
        model = Sequential()
        model.add(Input(shape=NUMBER_OF_CARDS))
        model.add(Dense(32, activation='relu'))


        # model.add(Conv2D(3, (3, 3), input_shape=env.STATE_SPACE_SIZE))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(2, 2))
        # model.add(Dropout(0.2))

        # model.add(Conv2D(256, (3, 3)))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(2, 2))
        # model.add(Dropout(0.2))

        # model.add(Flatten())
        # model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        if transition[0] is None:
            print(transition)
        else:
            self.replay_memory.append(transition)

    def get_qs(self, game, step):
        print(step)
        print(game)
        if game is None:
            pdb.set_trace()
        state=get_state(game)
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    # Trains main network every step during episode
    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        try:
            current_states = np.array([get_state(transition[0]) for transition in minibatch])/255
        except Exception:
            pdb.set_trace()
            self.train(terminal_state, step)

        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([get_state(transition[3]) for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        try:
            self.model.fit(np.array([get_state(game) for game in X])/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        except Exception:
            pdb.set_trace()
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, game):
        state=get_state(game)

        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]


env = Environment()

def build_model():
    # Exploration settings
    epsilon = 1  # not a constant, going to be decayed
    aggregate_episode_wins=[]

    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    #tf.set_random_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    #backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    agent = DQNAgent()

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        env.reset()
        current_state = env.game

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)

            new_state, reward, done = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward



            # Every step we update replay memory and train main network
            if not current_state is None and not new_state is None:
                agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            aggregate_episode_wins.append(average_reward)
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    return aggregate_episode_wins


def do_plotting(aggregate_episode_wins, episodes):
    position=111
    y_label="success_proportion"
    x_label="number_of_games"
    title="Average wins using q-table reinforcement learning"

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
    box_text= "Reinforcement learning model using deep q learning: \n"
    for key, val in variable_dictionary.items():
        if not val is None:
            box_text = box_text + key +": "+ str(val) + ", \n"

    box_style=dict(boxstyle='square', facecolor=c.BACKGROUND_COLOR, alpha=0.5)
    ax.text(1.05, 0.75,box_text,
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes, color='w', bbox=box_style)


    plt.plot(np.array(aggregate_episode_wins), label="average winning proportion")
    #plt.plot(aggregate_episode_wins['ep'], aggregate_episode_wins['max'], label="max rewards")
    #plt.plot(aggregate_episode_wins['ep'], aggregate_episode_wins['min'], label="min rewards")
    plt.legend(loc=4)
    plt.show()

    my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
    my_figure_folder=(os.path.join("figures", "q_table_graphs"))
    file_name  ="q_tables{}_attempt_3.png".format(episodes)
    fig.savefig(os.path.join(my_path, my_figure_folder, "wins_"+file_name), bbox_inches='tight')

if __name__=="__main__":
    aggregate_episode_wins=build_model()
    print(aggregate_episode_wins)
#%%
    do_plotting(RESULTS, EPISODES)


