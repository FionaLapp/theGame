# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:28:52 2022

@author: Fiona
"""

# based on this tutorial series: https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/


import numpy as np
import the_game
import matplotlib.pyplot as plt
import os
from plotting_constants import *

CARDS_IN_HAND = 2
NUMBER_OF_PLAYERS = 1
NUMBER_OF_PILES = 2
CARDS_PER_TURN = 1
NUMBER_OF_CARDS = 10


class Environment():
    def __init__(self):
        self.game=the_game.Game(cards_in_hand=CARDS_IN_HAND,
        number_of_players=NUMBER_OF_PLAYERS,
        number_of_piles=NUMBER_OF_PILES,
        cards_per_turn=CARDS_PER_TURN,
        number_of_cards=NUMBER_OF_CARDS)
        self.game.current_player=self.game.players[0]
        self.STATE_SPACE_SIZE=(1, self.game.number_of_players*self.game.cards_in_hand+self.game.number_of_piles, 1)
        self.ACTION_SPACE_SIZE=2*(NUMBER_OF_CARDS-2)*NUMBER_OF_PILES # 3 neurons, one each for pile, card, want_to_draw
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
        card_index=action_vector[1]
        try:
            card=self.game.current_player.hand[card_index]
        except IndexError:
            return self.game , self.MOST_NEGATIVE_REWARD, False
        want_to_draw=action_vector[2]
        if pile_number<0 or pile_number>len(self.game.piles):
            return self.game , self.MOST_NEGATIVE_REWARD, False
        elif not self.game.card_playable(self.game.current_player, self.game.piles[pile_number], card):
            return self.game , self.MOST_NEGATIVE_REWARD, False
        else:
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

    def action_number_to_vector(self, action_number):
        number_of_playable_cards=self.game.cards_in_hand
        pile_number=action_number%self.game.number_of_piles
        card_index=int((action_number-pile_number)/self.game.number_of_piles)%number_of_playable_cards
        want_to_draw=int(int((action_number-pile_number)/self.game.number_of_piles)/number_of_playable_cards)%2
        action_vector= np.array([pile_number, card_index, want_to_draw])
        return action_vector


def get_vector(game):
    vector_list=[]
    for player in game.players:
        hand=player.hand.copy()
        while len(hand)<game.cards_in_hand:
            hand.append(0)
        vector_list.extend(hand)
    for pile in game.piles:
        vector_list.append(pile.top_card)
    vector=np.array(vector_list)
    return tuple(vector)



def build_model():
    env= Environment()
    env.reset()


    LEARNING_RATE = 0.1

    DISCOUNT = 0.95
    EPISODES = 50000
    DISCRETE_OS_SIZE = [env.game.number_of_cards, env.game.number_of_cards, env.game.number_of_cards+1, env.game.number_of_cards]
    #discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE


    # Exploration settings
    epsilon = 1  # not a constant, qoing to be decayed
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES//2
    epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

    # For stats
    episode_wins = []
    episode_reward =[]
    aggr_episode_wins = {'ep': [], 'avg': [], 'max': [], 'min': []}
    STATS_EVERY=1000

    #initialise new table:
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.ACTION_SPACE_SIZE]))
    global_episode=EPISODES
    # # use previous table:
    # start_count=50000
    # q_table=np.load(f"q_tables/{start_count}-qtable.npy")
    # global_episode=start_count+EPISODES

    done = False




    for episode in range(EPISODES):
        episode_win = 0
        episode_reward=0
        env.reset()
        #print("new game")
        if episode%STATS_EVERY==0:
            print(episode)
        discrete_state = get_vector(env.game)
        done = False
        while not done:
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(q_table[discrete_state])
            else:
                # Get random action
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)
            new_state, reward, done = env.step(env.action_number_to_vector(action))
            episode_reward += reward
            new_discrete_state = get_vector(env.game)

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
                    q_table[discrete_state + (action,)] = 1000

            discrete_state = new_discrete_state

            # Decaying is being done every episode if episode number is within decaying range
            if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
                epsilon -= epsilon_decay_value
        episode_wins.append(episode_win)
        if not episode % STATS_EVERY:
            average_win = sum(episode_wins[-STATS_EVERY:])/STATS_EVERY
            aggr_episode_wins['ep'].append(episode)
            aggr_episode_wins['avg'].append(average_win)
            #aggr_episode_wins['max'].append(max(episode_wins[-STATS_EVERY:]))
            #aggr_episode_wins['min'].append(min(episode_wins[-STATS_EVERY:]))
            print(f'Episode: {episode:>5d}, average reward: {average_win:>4.1f}, current epsilon: {epsilon:>1.2f}')
    return aggr_episode_wins
    #%% save model
    np.save(f"q_tables/{global_episode}-qtable.npy", q_table)
#%%

def do_plotting(aggr_episode_wins):
    position=111
    y_label="success_proportion"
    x_label="number_of_games"
    title="Average wins using q-table reinforcement learning"

    fig=plt.figure(figsize=FIG_SIZE, facecolor=BACKGROUND_COLOR, edgecolor=EDGE_COLOR)
    ax = fig.add_subplot(position, facecolor=BACKGROUND_COLOR)
    ax.tick_params(color=FONT_COLOR, labelcolor=FONT_COLOR)
    ax.legend(labelcolor=PLOTTING_COLORS[0])

    for spine in ax.spines.values():
        spine.set_edgecolor(EDGE_COLOR)
    ax.set_ylabel(y_label, color=COLOR)
    ax.set_xlabel(x_label, color=COLOR)
    ax.set_ylim(bottom=0)
    ax.set_title(title, color=COLOR)
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

    box_style=dict(boxstyle='square', facecolor=BACKGROUND_COLOR, alpha=0.5)
    ax.text(1.05, 0.75,box_text,
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes, color='w', bbox=box_style)


    plt.plot(aggr_episode_wins['ep'], aggr_episode_wins['avg'], label="average winning proportion")
    #plt.plot(aggr_episode_wins['ep'], aggr_episode_wins['max'], label="max rewards")
    #plt.plot(aggr_episode_wins['ep'], aggr_episode_wins['min'], label="min rewards")
    plt.legend(loc=4)
    plt.show()

    my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
    my_figure_folder=(os.path.join("figures", "q_table_graphs"))
    file_name  ="q_tables{}.png".format(global_episode)
    fig.savefig(os.path.join(my_path, my_figure_folder, "wins_"+file_name), bbox_inches='tight')


aggr_episode_wins=build_model()
do_plotting(aggr_episode_wins)


