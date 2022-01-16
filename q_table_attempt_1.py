# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:28:52 2022

@author: Fiona

This is my first ("working"? if you use a very generous definition of the word)
attempt at producing a q-table reinforcement model for the game.
Here, the input space consists of the player's hand and the top cards on the
piles. However, since each of these can take a value from 0 to 100, I had to
use a ridiculously small game.
The action space size was cards_in_hand*number_of_piles*2 (want to draw or
not), which turned out to be a bit too big as well, especially since a lot
of the time, not all moves are possible.
One issue I was faced was to set a reward that would punish for imposssible
actions, so we wouldn't get stuck playing moves that aren't possible, but
still reward for winning rather than just finishing a game (although, now
that I think about it, simply giving no reward should work)
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

CARDS_IN_HAND = 2
NUMBER_OF_PLAYERS = 1
NUMBER_OF_PILES = 2
CARDS_PER_TURN = 1
NUMBER_OF_CARDS = 10


#learning constants
LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 25000

START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2

STATS_EVERY=100


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
        self.MOST_NEGATIVE_REWARD=-1 #-(self.game.number_of_cards+1)
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

    DISCRETE_OS_SIZE = [env.game.number_of_cards, env.game.number_of_cards, env.game.number_of_cards+1, env.game.number_of_cards]
    #discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE



    # Exploration settings
    epsilon = 1  # not a constant, qoing to be decayed
    epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

    # For stats
    episode_wins = []
    aggregate_episode_wins = pd.DataFrame({'episode': [], 'avgerage_wins': [], })#'max': [], 'min': []})

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
            aggregate_episode_wins.loc[len(aggregate_episode_wins.index)] = [episode, average_win]
            print(f'Episode: {episode:>5d}, average reward: {average_win:>4.4f}, current epsilon: {epsilon:>1.2f}')
    np.save(f"q_tables/{global_episode}-qtable.npy", q_table)
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
    file_name  ="q_tables{}.png".format(episodes)
    fig.savefig(os.path.join(my_path, my_figure_folder, "wins_"+file_name), bbox_inches='tight')


aggregate_episode_wins=build_model()
do_plotting(aggregate_episode_wins, EPISODES)


