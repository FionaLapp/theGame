# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 18:41:05 2021

@author: Fiona
"""
#https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
# from gym import Env, spaces
# import numpy as np
# import random
# import tensorflow as tf
# import matplotlib.pyplot as plt

# https://github.com/ModMaamari/reinforcement-learning-using-python
import the_game


from random import randint, choice
from collections import deque
from time import sleep
import time
import numpy as np
import pandas as pd

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.layers import Input, BatchNormalization, GlobalMaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint
#import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, Model
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm
import random
import os



# For more repetitive results
random.seed(1)
np.random.seed(1)
#tf.random.set_seed(1)


PATH = ""
# Create models folder
if not os.path.isdir(f'{PATH}models'):
    os.makedirs(f'{PATH}models')
# Create results folder
if not os.path.isdir(f'{PATH}results'):
    os.makedirs(f'{PATH}results')

TstartTime = time.time()


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



# #These lines establish the feed-forward part of the network used to choose actions
# inputs1 = tf.zeros(shape=[1,16],dtype=tf.float32)
# W = tf.Variable(tf.random_uniform([16,4],0,0.01))
# Qout = tf.matmul(inputs1,W)
# predict = tf.argmax(Qout,1)

# #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
# nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
# loss = tf.reduce_sum(tf.square(nextQ - Qout))
# trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# updateModel = trainer.minimize(loss)


# init = tf.initialize_all_variables()

# # Set learning parameters
# y = .99
# e = 0.1
# num_episodes = 2000
# #create lists to contain total rewards and steps per episode
# jList = []
# rList = []
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(num_episodes):
#         #Reset environment and get first new observation
#         s = Environment.reset()
#         rAll = 0
#         d = False
#         j = 0
#         #The Q-Network
#         while j < 99:
#             j+=1
#             #Choose an action by greedily (with e chance of random action) from the Q-network
#             a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
#             if np.random.rand(1) < e:
#                 a[0] = Environment.action_space.sample()
#             #Get new state and reward from environment
#             s1,r,d,_ = env.step(a[0])
#             #Obtain the Q' values by feeding the new state through our network
#             Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
#             #Obtain maxQ' and set our target value for chosen action.
#             maxQ1 = np.max(Q1)
#             targetQ = allQ
#             targetQ[0,a[0]] = r + y*maxQ1
#             #Train our network using target and predicted Q values
#             _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
#             rAll += r
#             s = s1
#             if d == True:
#                 #Reduce chance of random action as we train the model.
#                 e = 1./((i/50) + 10)
#                 break
#         jList.append(j)
#         rList.append(rAll)


class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, name)

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

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()
######################################################################################
# Agent class
class DQNAgent:
    def __init__(self, name, env, conv_list, dense_list, util_list):
        self.env = env
        self.conv_list  = conv_list
        self.dense_list = dense_list
        self.name = [str(name) +" | " + "".join(str(c)+"C | " for c in conv_list) + "".join(str(d) + "D | " for d in dense_list) + "".join(u + " | " for u in util_list) ][0]

        # Main model
        self.model = self.create_model(self.conv_list, self.dense_list)

        # Target network
        self.target_model = self.create_model(self.conv_list, self.dense_list)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(name, log_dir="{}logs/{}-{}".format(PATH, name, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0


    # Creates a convolutional block given (filters) number of filters, (dropout) dropout rate,
    # (bn) a boolean variable indecating the use of BatchNormalization,
    # (pool) a boolean variable indecating the use of MaxPooling2D
    def conv_block(self, inp, filters=64, bn=True, pool=True, dropout = 0.2):
        _ = Conv2D(filters=filters, kernel_size=3, activation='relu')(inp)
        if bn:
            _ = BatchNormalization()(_)
        if pool:
            _ = MaxPooling2D(pool_size=(2, 2))(_)
        if dropout > 0:
            _ = Dropout(0.2)(_)
        return _
    # Creates the model with the given specifications:
    def create_model(self, conv_list, dense_list):
        # Defines the input layer with shape = ENVIRONMENT_SHAPE
        input_layer = Input(shape=self.env.STATE_SPACE_SIZE)
        # Defines the first convolutional block:
        print(input_layer)
        # _ = self.conv_block(input_layer, filters=conv_list[0], bn=False, pool=False)
        # # If number of convolutional layers is 2 or more, use a loop to create them.
        # if len(conv_list)>1:
        #     for c in conv_list[1:]:
        #         _ = self.conv_block(_, filters=c)
        # # Flatten the output of the last convolutional layer.
        # _  = Flatten()(_)
        _=input_layer
        # Creating the dense layers:
        for d in dense_list:
            _ = Dense(units=d, activation='relu')(_)
        # The output layer has 5 nodes (one node per action)
        output = Dense(units=self.env.ACTION_SPACE_SIZE,
                          activation='linear', name='output')(_)

        # Put it all together:
        model = Model(inputs=input_layer, outputs=[output])
        model.compile(optimizer=Adam(lr=0.001),
                      loss={'output': 'mse'},
                      metrics={'output': 'accuracy'})

        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states.reshape(-1, *env.STATE_SPACE_SIZE))


        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states.reshape(-1, *env.STATE_SPACE_SIZE))

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
        self.model.fit(x = np.array(X).reshape(-1, *env.STATE_SPACE_SIZE),
                        y = np.array(y),
                        batch_size = MINIBATCH_SIZE, verbose = 0,
                        shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, *env.STATE_SPACE_SIZE))
######################################################################################
def save_model_and_weights(agent, model_name, episode, max_reward, average_reward, min_reward):
    checkpoint_name = f"{model_name}| Eps({episode}) | max({max_reward:_>7.2f}) | avg({average_reward:_>7.2f}) | min({min_reward:_>7.2f}).model"
    agent.model.save(f'{PATH}models/{checkpoint_name}')
    best_weights = agent.model.get_weights()
    return best_weights
######################################################################################
# ## Constants:
# RL Constants:
DISCOUNT               = 0.99
REPLAY_MEMORY_SIZE     = 3_000   # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000   # Minimum number of steps in a memory to start training
UPDATE_TARGET_EVERY    = 20      # Terminal states (end of episodes)
MIN_REWARD             = 1000    # For model save
SAVE_MODEL_EVERY       = 1000    # Episodes
SHOW_EVERY             = 20      # Episodes
EPISODES               = 100  # Number of episodes
#  Stats settings
AGGREGATE_STATS_EVERY = 20  # episodes
SHOW_PREVIEW          = False
######################################################################################
# Models Arch :
  # [{[conv_list], [dense_list], [util_list], MINIBATCH_SIZE, {EF_Settings}, {ECC_Settings}} ]

models_arch = [  {"conv_list":[32], "dense_list":[32,32], "util_list":["ECC2", "1A-5Ac"],
                  "MINIBATCH_SIZE":128, "best_only":False,
                  "EF_Settings":{"EF_Enabled":False}, "ECC_Settings":{"ECC_Enabled":False}},

                {"conv_list":[32], "dense_list":[32,32,32], "util_list":["ECC2", "1A-5Ac"],
                  "MINIBATCH_SIZE":128, "best_only":False,
                  "EF_Settings":{"EF_Enabled":False}, "ECC_Settings":{"ECC_Enabled":False}},

                {"conv_list":[32], "dense_list":[32,32], "util_list":["ECC2", "1A-5Ac"],
                "MINIBATCH_SIZE":128, "best_only":False,
                  "EF_Settings":{"EF_Enabled":True, "FLUCTUATIONS":2},
                "ECC_Settings":{"ECC_Enabled":True, "MAX_EPS_NO_INC":int(EPISODES*0.2)}}]

# A dataframe used to store grid search results
res = pd.DataFrame(columns = ["Model Name","Convolution Layers", "Dense Layers", "Batch Size", "ECC", "EF",
                                "Best Only" , "Average Reward", "Best Average", "Epsilon 4 Best Average",
                                "Best Average On", "Max Reward", "Epsilon 4 Max Reward", "Max Reward On",
                                "Total Training Time (min)", "Time Per Episode (sec)"])
######################################################################################
# Grid Search:
for i, m in enumerate(models_arch):
    startTime = time.time() # Used to count episode training time
    MINIBATCH_SIZE = m["MINIBATCH_SIZE"]

    # Exploration settings :
    # Epsilon Fluctuation (EF):
    EF_Enabled          = m["EF_Settings"]["EF_Enabled"]  # Enable Epsilon Fluctuation
    MAX_EPSILON         = 1      # Maximum epsilon value
    MIN_EPSILON         = 0.001    # Minimum epsilon value
    if EF_Enabled:
        FLUCTUATIONS    = m["EF_Settings"]["FLUCTUATIONS"]     # How many times epsilon will fluctuate
        FLUCTUATE_EVERY = int(EPISODES/FLUCTUATIONS) # Episodes
        EPSILON_DECAY   = MAX_EPSILON - (MAX_EPSILON/FLUCTUATE_EVERY)
        epsilon         = 1  # not a constant, going to be decayed
    else:
        EPSILON_DECAY   = MAX_EPSILON - (MAX_EPSILON/(0.8*EPISODES))
        epsilon         = 1  # not a constant, going to be decayed

    # Initialize some variables:
    best_average        = -100
    best_score          = -100

    # Epsilon Conditional Constantation (ECC):
    ECC_Enabled         = m["ECC_Settings"]["ECC_Enabled"]
    avg_reward_info     = [[1, best_average, epsilon]] # [[episode1, reward1 , epsilon1] ... [episode_n, reward_n , epsilon_n]]
    max_reward_info     = [[1, best_score ,  epsilon]]
    if ECC_Enabled : MAX_EPS_NO_INC      = m["ECC_Settings"]["MAX_EPS_NO_INC"] # Maximum number of episodes without any increment in reward average
    eps_no_inc_counter  = 0 # Counts episodes with no increment in reward


    # For stats
    ep_rewards = [best_average]



    env = Environment()
    # env.MOVE_WALL_EVERY = 1 # Every how many frames the wall moves.

    agent = DQNAgent(f"M{i}", env, m["conv_list"], m["dense_list"], m["util_list"])
    MODEL_NAME = agent.name

    best_weights = [agent.model.get_weights()]

    # Uncomment these two lines if you want to show preview on your screen
    # WINDOW          = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
    # clock           = pygame.time.Clock()

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        if m["best_only"]: agent.model.set_weights(best_weights[0])
        # agent.target_model.set_weights(best_weights[0])

        score_increased = False
        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step    = 1
        action = 0
        # Reset environment and get initial state
        current_state = env.reset()
        game_over     = env.game_over
        while not game_over:
            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))

            else:
                # Get random action
                # action = choice(env.ACTION_SPACE)
                random_pile=randint(0, env.game.number_of_piles-1)
                random_card= randint(0, env.game.cards_in_hand-1)
                want_to_draw=randint(0, 1)
                action_vector= np.array([random_pile, random_card, want_to_draw])

            new_state, reward, game_over = env.step(action)

            # Transform new continuous state to new discrete state and count reward
            episode_reward += reward

            # Uncomment the next block if you want to show preview on your screen
            # if SHOW_PREVIEW and not episode % SHOW_EVERY:
            #     clock.tick(27)
            #     env.render(WINDOW)

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, game_over))
            agent.train(game_over, step)

            current_state = new_state
            step += 1

        if ECC_Enabled : eps_no_inc_counter += 1
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)

        if not episode % AGGREGATE_STATS_EVERY:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save models, but only when avg reward is greater or equal a set value
            if not episode % SAVE_MODEL_EVERY:
                # Save Agent :
                _ = save_model_and_weights(agent, MODEL_NAME, episode, max_reward, average_reward, min_reward)


            if average_reward > best_average:
                best_average = average_reward
                # update ECC variables:
                avg_reward_info.append([episode, best_average, epsilon])
                eps_no_inc_counter = 0
                # Save Agent :
                best_weights[0] = save_model_and_weights(agent, MODEL_NAME, episode, max_reward, average_reward, min_reward)

            if ECC_Enabled and eps_no_inc_counter >= MAX_EPS_NO_INC:
                epsilon = avg_reward_info[-1][2] # Get epsilon value of the last best reward
                eps_no_inc_counter = 0

        if episode_reward > best_score:
            try:
                best_score = episode_reward
                max_reward_info.append([episode, best_score, epsilon])

                # Save Agent :
                best_weights[0] = save_model_and_weights(agent, MODEL_NAME, episode, max_reward, average_reward, min_reward)

            except:
                pass

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        # Epsilon Fluctuation:
        if EF_Enabled:
            if not episode % FLUCTUATE_EVERY:
                epsilon = MAX_EPSILON

    endTime = time.time()
    total_train_time_sec = round((endTime - startTime))
    total_train_time_min = round((endTime - startTime)/60,2)
    time_per_episode_sec = round((total_train_time_sec)/EPISODES,3)

    # Get Average reward:
    average_reward = round(sum(ep_rewards)/len(ep_rewards), 2)

    # Update Results DataFrames:
    res = res.append({"Model Name":MODEL_NAME, "Convolution Layers":m["conv_list"], "Dense Layers":m["dense_list"],
                      "Batch Size":m["MINIBATCH_SIZE"], "ECC":m["ECC_Settings"], "EF":m["EF_Settings"],
                                "Best Only":m["best_only"], "Average Reward":average_reward,
                                "Best Average":avg_reward_info[-1][1], "Epsilon 4 Best Average":avg_reward_info[-1][2],
                                "Best Average On":avg_reward_info[-1][0], "Max Reward":max_reward_info[-1][1],
                                "Epsilon 4 Max Reward":max_reward_info[-1][2], "Max Reward On":max_reward_info[-1][0],
                                "Total Training Time (min)":total_train_time_min, "Time Per Episode (sec)":time_per_episode_sec}
                    , ignore_index=True)
    res = res.sort_values(by = 'Best Average')
    avg_df = pd.DataFrame(data = avg_reward_info, columns=["Episode", "Average Reward", "Epsilon"])
    max_df = pd.DataFrame(data = max_reward_info, columns=["Episode", "Max Reward", "Epsilon"])

    # Save dataFrames
    res.to_csv(f"{PATH}results/Results.csv")
    avg_df.to_csv(f"{PATH}results/{MODEL_NAME}-Results-Avg.csv")
    max_df.to_csv(f"{PATH}results/{MODEL_NAME}-Results-Max.csv")

TendTime = time.time()
######################################################################################
print( f"Training took {round((TendTime - TstartTime)/60)  } Minutes ")
print( f"Training took {round((TendTime - TstartTime)/3600)  } Hours ")
######################################################################################