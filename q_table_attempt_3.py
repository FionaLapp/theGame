# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:08:04 2022

@author: Fiona


Here we go again...
Second attempt at producing a q-table reinforcement model for the game.
I've changed both state space and action space:
For the state space, we have a vector of length number_of_cards. Each card
has a value: 0 if it's in the drawing pile, 1 if it's in the current
player's hand, 2 in someone else's hand, 3 and 5 for top_cards of increasing
and decreasing piles, and 4 for cards that have been played already but
aren't top cards.
For the action space, we only consider the best 3 card-pile combinations,
and we only play as many cards as necessary (want_to_draw always true).
Overall, i had this winning about 30% at one point but then I changed the
hyperparameters and now I can't even get that back, but if there's only 10
cards in the game, 30% is not a particularly respectable score anyways, so
let's bury this and move on to bigger and better things.
"""

# based on this tutorial series: https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/



import numpy as np
import the_game
import matplotlib.pyplot as plt
import os
import plotting_constants as c
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

CARDS_IN_HAND = 2
NUMBER_OF_PLAYERS = 1
NUMBER_OF_PILES = 2
CARDS_PER_TURN = 1
NUMBER_OF_CARDS = 10

NUMBER_OF_DIMENSIONS_PER_CARD=6
#learning constants
LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 300

START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2

STATS_EVERY=25


class Environment():
    def __init__(self):
        self.game=the_game.Game(cards_in_hand=CARDS_IN_HAND,
        number_of_players=NUMBER_OF_PLAYERS,
        number_of_piles=NUMBER_OF_PILES,
        cards_per_turn=CARDS_PER_TURN,
        number_of_cards=NUMBER_OF_CARDS)
        self.game.current_player=self.game.players[0]
        #self.STATE_SPACE_SIZE=(1, self.game.number_of_players*self.game.cards_in_hand+self.game.number_of_piles, 1)
        self.ACTION_SPACE_SIZE=3
        self.MOST_NEGATIVE_REWARD=-1 #-(self.game.number_of_cards+1)
        self.game_over=False


    def reset(self):
        self.__init__()
    def step(self, action_number):
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
            want_to_draw=True
            #print("card played:", card, " on pile", pile_number, " by player ", self.game.players[0])
            reward=0#self.reward(card, self.game.piles[pile_number])
            self.game.play_card(self.game.piles[pile_number], card, want_to_draw)
            #print(self.game.print_hands())
            #print(self.game.print_piles())
            if self.game.game_finished():
                self.game_over=True
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
    state=np.int_(np.zeros((NUMBER_OF_CARDS, NUMBER_OF_DIMENSIONS_PER_CARD))) #0: we assume everything is in the drawing pile
    for player in game.players:
        if player==game.current_player:
            for card in player.hand:
                state[card, 1]=1 #1: in my hand
        else:
            for card in player.hand:
                state[card, 2]=1 #2: in someone else's hand
    for pile in game.piles:
        for card in pile.cards[1:]:

            state[card, 3]=0 # card has already been played
        if pile.top_card!=NUMBER_OF_CARDS and pile.top_card!=0:
            if pile.pile_multiplier==-1:
                state[pile.top_card, 4]=1#-pile.pile_multiplier #5 if decreasing, 3 if increasing
            else:
                state[pile.top_card, 5]=1
    return state



def build_model():
    env= Environment()
    env.reset()


    DISCRETE_OS_SIZE = NUMBER_OF_CARDS*[3]# for each card, state can be: in drawing_pile, been_played, in my hand, in someone else's hand, top card

    #[env.game.number_of_cards, env.game.number_of_cards, env.game.number_of_cards+1, env.game.number_of_cards]
    #discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE



    # Exploration settings
    epsilon = 1  # not a constant, qoing to be decayed
    epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

    # For stats
    episode_wins = []
    aggregate_episode_wins = pd.DataFrame({'episode': [], 'average_wins': [], })#'max': [], 'min': []})

    #initialise new table:
    q_table = np.zeros(shape=(DISCRETE_OS_SIZE + [env.ACTION_SPACE_SIZE]))
    global_episode=EPISODES

    # # use previous table:
    # start_count=50000
    # q_table=np.load(f"q_tables/{start_count}-qtable.npy")
    # global_episode=start_count+EPISODES

    done = False

    for episode in range(EPISODES):
        episode_win = 0
        env.reset()
        #print("new game")
        if episode%STATS_EVERY==0:
            print(episode)
        discrete_state = get_state(env.game)
        done = False
        while not done:
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(q_table[discrete_state])
            else:
                # Get random action
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)
            new_state, reward, done = env.step(action)
            new_discrete_state = get_state(env.game)

            #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # If simulation did not end yet after last step - update Q table
            if not done:

                # Maximum possible Q value in next step (for new state)
                max_future_q = np.max(q_table[new_discrete_state])

                # Current Q value (for current state and performed action)
                current_q = q_table[discrete_state + (action,)]

                # And here's our equation for a new Q value for current state and action
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                # Update Q table with new Q value
                q_table[tuple(discrete_state) + (action,)] = new_q


                # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
            elif env.game_over:

                #q_table[discrete_state + (action,)] = reward
                if new_state.game_lost():
                    q_table[discrete_state + (action,)] = -1000
                else:
                    episode_win=1
                    #print("Won a game")
                    q_table[discrete_state + (action,)] = 1000

            discrete_state = new_discrete_state

            # Decaying is being done every episode if episode number is within decaying range
            if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
                epsilon -= epsilon_decay_value
        episode_wins.append(episode_win)
        if not episode % STATS_EVERY:
            average_win = sum(episode_wins[-STATS_EVERY:])/STATS_EVERY
            aggregate_episode_wins.loc[len(aggregate_episode_wins.index)] = [episode, average_win]
            print(f'Episode: {episode:>5d}, average reward: {average_win:>4.4f}, current epsilon: {epsilon:>1.2f}')
    np.save(f"q_tables/{global_episode}-qtable_attempt_2.npy", q_table)
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
    box_text= "Reinforcement learning model using a q-table: \n"
    for key, val in variable_dictionary.items():
        if not val is None:
            box_text = box_text + key +": "+ str(val) + ", \n"

    box_style=dict(boxstyle='square', facecolor=c.BACKGROUND_COLOR, alpha=0.5)
    ax.text(1.05, 0.75,box_text,
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes, color='w', bbox=box_style)


    plt.plot(aggregate_episode_wins['episode'], aggregate_episode_wins['average_wins'], label="average winning proportion")
    #plt.plot(aggregate_episode_wins['ep'], aggregate_episode_wins['max'], label="max rewards")
    #plt.plot(aggregate_episode_wins['ep'], aggregate_episode_wins['min'], label="min rewards")
    plt.legend(loc=4)
    plt.show()

    my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
    my_figure_folder=(os.path.join("figures", "q_table_graphs"))
    file_name  ="q_tables{}_attempt_2.png".format(episodes)
    fig.savefig(os.path.join(my_path, my_figure_folder, "wins_"+file_name), bbox_inches='tight')

if __name__=="__main__":
    aggregate_episode_wins=build_model()
    print(aggregate_episode_wins)
#%%
    do_plotting(aggregate_episode_wins, EPISODES)


