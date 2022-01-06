# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:28:52 2022

@author: Fiona
"""

# based on this tutorial series: https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/


import numpy as np
import the_game


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
        self.ACTION_SPACE_SIZE=2*NUMBER_OF_CARDS*NUMBER_OF_PILES # 3 neurons, one each for pile, card, want_to_draw
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
        if card==0:
            print(action_vector)
        if pile_number<0 or pile_number>len(self.game.piles):
            return self.game , self.MOST_NEGATIVE_REWARD, False
        elif not self.game.card_playable(self.game.current_player, self.game.piles[pile_number], card):
            return self.game , self.MOST_NEGATIVE_REWARD, False
        else:
            print("card played:", card, pile_number)
            reward=self.reward(card, self.game.piles[pile_number])
            self.game.play_card(self.game.piles[pile_number], card, want_to_draw)
            if self.game.finished:
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
        number_of_playable_cards=self.game.number_of_cards-2
        pile_number=action_number%self.game.number_of_piles
        card=int((action_number-pile_number)/self.game.number_of_piles)%number_of_playable_cards +2
        want_to_draw=int(int((action_number-pile_number)/self.game.number_of_piles)/number_of_playable_cards)%2
        action_vector= np.array([pile_number, card, want_to_draw])
        return action_vector


def get_vector(game):
    vector_list=[]
    for player in game.players:
        hand=player.hand
        while len(hand)<game.cards_in_hand:
            hand.append(0)
        vector_list.extend(hand)
    for pile in game.piles:
        vector_list.append(pile.top_card)
    vector=np.array(vector_list)
    return tuple(vector)




env= Environment()
env.reset()


LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 20
DISCRETE_OS_SIZE = [env.game.number_of_cards, env.game.number_of_cards, env.game.number_of_cards+1, env.game.number_of_cards]
#discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.ACTION_SPACE_SIZE]))

done = False





env.reset()
discrete_state = get_vector(env.game)
done = False
while not done:

    action = np.argmax(q_table[discrete_state])
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
    elif new_state[0] >= env.goal_position:
        #q_table[discrete_state + (action,)] = reward
        q_table[discrete_state + (action,)] = 0

    discrete_state = new_discrete_state

