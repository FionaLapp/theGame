# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 18:09:48 2021

@author: Fiona
"""
from abc import ABCMeta, abstractmethod
from random import randint
import logging
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import game_logging


LOG_LEVEL = logging.INFO
# CARDS_IN_HAND=6
# NUMBER_OF_PLAYERS=4
# NUMBER_OF_PILES=4
# CARDS_PER_TURN=2
# NUMBER_OF_CARDS=100
LOWEST_PLAYABLE_NUMBER = 2
DECREASING = "DECREASING"
INCREASING = "INCREASING"
NUMBER_OF_ATTEMPTS = 100

# %% errors


class InputValidationError(Exception):
    """Exception raised for invalid inputs.

    Attributes:
        input_value -- invalid input value
        message -- explanation of the error
    """

    def __init__(self, input_value, message="input not valid"):
        game_logging.GameLoggers.debug_logger.handlers.clear()
        self.input_value = input_value
        self.message = message
        super().__init__(self.message)


class CardNotPlayableError(Exception):
    """Exception raised when card cannot be played.

    Attributes:
        card -- invalid card
        message -- explanation of the error
    """

    def __init__(self, card, message="card cannot be played"):
        game_logging.GameLoggers.debug_logger.handlers.clear()
        self.card = card
        self.message = message
        super().__init__(self.message)

# %% player


class Player:
    def __init__(self, id_number, drawing_pile, game, cards_in_hand,
                 cards_per_turn):
        self.id_number = id_number
        self.hand = []
        # draw as many cards as a hand needs
        for i in range(cards_in_hand):
            game.draw_card(self, drawing_pile)
        self.is_my_turn = False
        self.number_of_cards_i_need_to_play = cards_per_turn
        game_logging.GameLoggers.debug_logger.debug("{} initialised successfully".format(
            self))

    def card_playable(self, card):
        # check if playable
        game_logging.GameLoggers.debug_logger.debug("checking if {} playable".format(card))
        if card in self.hand:
            return True
        else:
            return False

    def remove_card_from_hand(self, card):
        """
        This function does everything associated with playing a card on the
        player-side: removing it from their hand, decreasing the number of
        cards they need to play
        inside the game object, remove_card_from_hand must be called for pile
        and player
        This needs to throw an exception if the card is not playable
        """
        if not self.card_playable(card):
            # throw exception
            game_logging.GameLoggers.debug_logger.debug("not possible")
            raise CardNotPlayableError(
                card, ("Cannot play {} because it is not contained in hand "
                       "{}").format(card, self.hand))

        else:
            # remove card from hand
            self.hand.remove(card)
            # decrease number of cards I need to play
            if self.number_of_cards_i_need_to_play > 0:
                self.number_of_cards_i_need_to_play -= 1
            else:
                self.number_of_cards_i_need_to_play = 0

    def add_card_to_hand(self, card):
        """
        This function does everything associated with drawing a card on the
        player-side: adding it to their hand
        inside the game object, draw_card must be called for pile and player
        """
        # add card
        game_logging.GameLoggers.debug_logger.debug("adding card {} to hand".format(card))
        self.hand.append(card)

    def __str__(self):
        return "Player {} with cards {}".format(self.id_number, self.hand)

# %% pile


class Pile(metaclass=ABCMeta):

    def __init__(self):
        self.cards = []


class DrawingPile(Pile):

    def __init__(self, size):
        # if size=100, this will go up to 99
        self.cards = [*range(LOWEST_PLAYABLE_NUMBER, size)]

    def remove_card(self):
        """
        This function does everything associated with drawing a card on the
        pile-side: removing it from the pile
        inside the game object, draw_card must be called for pile and player
        """
        # randomly pick a card
        random_index = randint(0, len(self.cards) - 1)
        # remove card from pile
        try:
            card = self.cards.pop(random_index)
        except Exception:
            # this is only in here because I coded it without the -1 before;
            # thought I might as well leave it in
            raise InputValidationError(
                random_index, ("Something went wrong with drawing the cards:"
                               " tried to draw card at index {} but there are "
                               "only {} cards available").format(
                    random_index, len(
                        self.cards)))

        game_logging.GameLoggers.debug_logger.debug("drew random card: {}".format(card))

        return card

    def __str__(self):
        return "DrawingPile"


class PlayingPile(Pile):

    def __init__(self, id_number):
        super().__init__()
        self.id_number = id_number

    @abstractmethod
    def card_playable(self, card):
        """
        An abstract method that needs to be implemented by increasing and
        decreasing piles separately
        """
        # return False

    def play_card(self, card):
        """
        This function does everything associated with playing a card on the
        pile-side: adding it to the pile
        inside the game object, play_card must be called for pile and player
        This needs to throw an exception if the card is not playable
        """
        if not self.card_playable(card):
            # throw exception
            game_logging.GameLoggers.debug_logger.debug("not possible")
            raise CardNotPlayableError(
                card, "Cannot play {} ({}) on pile with top card {}".format(
                    card, self.pile_type, self.get_top_card()))

        else:
            # add card to pile
            self.cards.append(card)
            game_logging.GameLoggers.debug_logger.debug("adding card {} to pile".format(card
                                                                           ))

    def get_top_card(self):
        # get top card
        top_card = self.cards[-1]
        game_logging.GameLoggers.debug_logger.debug("getting top card: {}".format(top_card))
        return top_card

    def __str__(self):
        return "Playing Pile {}".format(self.id_number)


class DecreasingPile(PlayingPile):
    def __init__(self, id_number, number_of_cards):
        super().__init__(id_number)
        self.pile_type = DECREASING
        self.cards.append(number_of_cards)

    def card_playable(self, card):
        # check if playable
        top_card = self.get_top_card()
        if card < top_card or card == top_card + 10:
            game_logging.GameLoggers.debug_logger.debug(("card {} playable on decreasing "
                                            "pile with top card {}").format(
                                                card, top_card))
            return True
        else:
            game_logging.GameLoggers.debug_logger.debug(("card {} not playable on "
                                            "decreasing pilewith top card {}"
                                            ).format(card, top_card))
            return False

    def __str__(self):
        return "Decreasing Pile {} with top card {}".format(
            self.id_number, self.get_top_card())


class IncreasingPile(PlayingPile):
    def __init__(self, id_number):
        super().__init__(id_number)
        self.pile_type = INCREASING
        self.cards.append(LOWEST_PLAYABLE_NUMBER - 1)

    def card_playable(self, card):
        # check if playable
        top_card = self.get_top_card()
        if card > top_card or card == top_card - 10:
            game_logging.GameLoggers.debug_logger.debug(("card {} playable on increasing"
                                            "pile with top card {}").format(
                                          card, top_card))
            return True
        else:
            game_logging.GameLoggers.debug_logger.debug(("card {} not playable on "
                                            "increasing pile with top card {}"
                                            ).format(card, top_card))
            return False

    def __str__(self):
        return "Increasing Pile {} with top card {}".format(
            self.id_number, self.get_top_card())

# %% Game


class Game():
    def __init__(
            self,
            cards_in_hand=6,
            number_of_players=4,
            number_of_piles=4,
            cards_per_turn=2,
            number_of_cards=100):
        self.cards_in_hand = cards_in_hand
        self.number_of_players = number_of_players
        self.number_of_piles = number_of_piles
        self.cards_per_turn = cards_per_turn
        self.number_of_cards = number_of_cards
        self.drawing_pile = DrawingPile(number_of_cards)
        self.piles = []
        # add piles
        self._create_piles(self.number_of_piles, self.number_of_cards)

        self.players = []
        # add players
        self._create_players(
            self.number_of_players,
            self.cards_in_hand,
            self.cards_per_turn)

        self.current_player = None
        self.finished = False

    def game_won(self):
        if self.drawing_pile.cards != []:
            # there are still cards on the drawing pile
            return False
        else:
            for player in self.players:
                if player.hand != []:
                    # a player's hand isn't empty
                    return False
            # no cards in drawing pile and all hands empty
            return True

    def game_lost(self):
        if not self.game_won() and not self.current_player.hand == []:
            # there are still cards that haven't been played
            # true if player can play, false otherwise
            can_play_matrix = self.can_play(self.current_player)
            return np.all(np.invert(can_play_matrix))  # all have to be true

    def set_next_player(self):
        next_player_number = (self.current_player.id_number + 1
                              ) % self.number_of_players
        self.current_player = self.players[next_player_number]
        if self.current_player.hand == []:
            self.set_next_player()  # yay recursion

    def game_finished(self):
        # check if finished
        game_logging.GameLoggers.debug_logger.debug("checking if finished")
        if self.game_won() or self.game_lost():
            self.finished = True
            return True
        return False

    def can_play(self, player):
        can_play_matrix = np.zeros(
            (len(
                player.hand), len(
                self.piles)), dtype=bool)
        for i, card in enumerate(player.hand):
            for j, pile in enumerate(self.piles):
                can_play_matrix[i, j] = self.card_playable(player, pile, card)
        return can_play_matrix

    def card_playable(self, player, pile, card):
        return (player.card_playable(card) and pile.card_playable(card))

    def play_card(self, pile, card, want_to_draw):
        if not self.game_finished():
            player = self.current_player
            if player.number_of_cards_i_need_to_play > 1:
                # if I only have one mandatory turn left, it's this turn
                want_to_draw = False
            if not want_to_draw and len(player.hand) == 1:
                # if hand empty then draw
                want_to_draw = True
            # play card
            if self.card_playable(player, pile, card):
                player.remove_card_from_hand(card)
                pile.play_card(card)
                game_logging.GameLoggers.debug_logger.debug("{} plays {} on {} ".format(
                    player, card, pile))

                # check if game finished
                if self.game_finished():
                    return
            else:
                game_logging.GameLoggers.debug_logger.debug(CardNotPlayableError)
                raise CardNotPlayableError(card)

            if want_to_draw:
                player.number_of_cards_i_need_to_play = self.cards_per_turn
                player.is_my_turn = False
                if self.drawing_pile == []:
                    print("empty pile")
                while ((len(player.hand) < self.cards_in_hand) and (
                        len(self.drawing_pile.cards) != 0)):
                    self.draw_card(player, self.drawing_pile)

                self.set_next_player()

    def draw_card(self, player, drawing_pile):
        # draw card
        if drawing_pile.cards != []:
            card = drawing_pile.remove_card()
            player.add_card_to_hand(card)
            game_logging.GameLoggers.debug_logger.debug("{} drew {} from {} ".format(
                player, card, drawing_pile))

    def _create_piles(self, number_of_piles, number_of_cards):
        if number_of_piles % 2 != 0:
            raise InputValidationError(
                number_of_piles,
                ("I'd really prefer an even number of piles, and {} might be a"
                 " lot of things but it ain't that").format(number_of_piles))
        for i in range(int(number_of_piles / 2)):
            self.piles.append(DecreasingPile(2 * i, number_of_cards))
            self.piles.append(IncreasingPile((2 * i) + 1))
        game_logging.GameLoggers.debug_logger.debug("creating {} piles".format(
            number_of_piles))

    def _create_players(
            self,
            number_of_players,
            cards_in_hand,
            cards_per_turn):
        for i in range(int(number_of_players)):
            self.players.append(
                Player(
                    i,
                    self.drawing_pile,
                    self,
                    cards_in_hand,
                    cards_per_turn))
        game_logging.GameLoggers.debug_logger.debug("creating {} players".format(
            number_of_players))

    def print_hands(self):
        hand_string = ""
        for player in self.players:
            hand_string += "{}".format(player.__str__())
        return hand_string

    def print_piles(self):
        pile_string = ""
        for pile in self.piles:
            pile_string += "{} with cards {}".format(
                pile.__str__(), pile.cards)
        return pile_string

    def __str__(self):
        return "Game"

# %% strategy


class Strategy(metaclass=ABCMeta):

    def __init__(self, cards_in_hand=6, number_of_players=4, number_of_piles=4,
                 cards_per_turn=2, number_of_cards=100):
        self.game = Game(cards_in_hand=cards_in_hand,
                         number_of_players=number_of_players,
                         number_of_piles=number_of_piles,
                         cards_per_turn=cards_per_turn,
                         number_of_cards=number_of_cards)

    @abstractmethod
    def play(self):
        return

    def start_game(self, player):
        if player not in self.game.players:
            raise InputValidationError(
                player, "Player does not participate in game")
        self.game.current_player = player


class PlayFirstTwoStrategy(Strategy):
    def play(self):
        while not self.game.finished:
            want_to_draw = True
            for pile in self.game.piles:
                for card in self.game.current_player.hand:
                    if self.game.card_playable(self.game.current_player, pile,
                                               card):
                        game_logging.GameLoggers.strategy_logger.info(
                            "{} playing card {} on pile {}".format(
                                self.game.current_player, card, pile))

                        self.game.play_card(pile, card, want_to_draw)
                        continue
                continue


class PlayWithMetricStrategy(Strategy):
    def play(self):
        while not self.game.finished:
            want_to_draw = True
            self.metric_matrix = self.basic_metric()
            metric_matrix = self.metric_matrix
            min_index = np.unravel_index(np.argmin(metric_matrix, axis=None),
                                         metric_matrix.shape)
            pile_number = min_index[1]
            pile = self.game.piles[pile_number]
            card_position = min_index[0]
            card = self.game.current_player.hand[card_position]
            if self.game.card_playable(self.game.current_player, pile, card):
                game_logging.GameLoggers.strategy_logger.info(
                    "{} playing card {} on pile {}".format(
                                self.game.current_player, card, pile))
                self.game.play_card(pile, card, want_to_draw)
            else:
                if self.game.game_finished():
                    return
                raise CardNotPlayableError(
                    card, "{} cannot play {} on pile {}".format(
                        self.game.current_player, card, pile))
                self.game.play_card(pile, card, want_to_draw)

    def basic_metric(self):
        """
        Creates a distance matrix for the strategy's game's current player with
        maximal value (number of cards +1) for impossible moves, distances
        otherwise (and -10 for jumps).

        Parameters
        ----------
        None

        Returns
        -------
        metric_matrix : numpy.array
            A distance matrix with higher values for worse cards (jumps have
            negative values) and the highest value for impossible moves.

        """
        game = self.game
        player = game.current_player
        # create matrix and fill it with highest possible metric so that
        # impossible moves are never played
        metric_matrix = (game.number_of_cards + 1) * np.ones(
            (len(player.hand), len(game.piles)), dtype=int)
        for i, card in enumerate(player.hand):
            for j, pile in enumerate(game.piles):
                if game.card_playable(player, pile, card):
                    if isinstance(pile, DecreasingPile):
                        metric_matrix[i, j] = pile.get_top_card()-card
                    elif isinstance(pile, IncreasingPile):
                        metric_matrix[i, j] = card-pile.get_top_card()
        return metric_matrix


if __name__ == "__main__":
    with game_logging.ContextManager() as manager:
        my_strategy = PlayWithMetricStrategy()
        my_strategy.start_game(my_strategy.game.players[0])
        my_strategy.play()
        print(
            "Game won: {}".format(my_strategy.game.game_won()))
        print(
            "players: {}".format(my_strategy.game.print_hands()))
        print(
            "piles: {}".format(my_strategy.game.print_piles()))
