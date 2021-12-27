# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 11:55:06 2021

@author: Fiona
"""
import game


CARDS_IN_HAND = 2
NUMBER_OF_PLAYERS = 1
NUMBER_OF_PILES = 2
CARDS_PER_TURN = 1
NUMBER_OF_CARDS = 10


def test_player_creation():
    """
    Verify player creation

    Returns
    -------
    None.

    """
    my_game = game.Game(
        cards_in_hand=CARDS_IN_HAND,
        number_of_players=NUMBER_OF_PLAYERS,
        number_of_piles=NUMBER_OF_PILES,
        cards_per_turn=CARDS_PER_TURN,
        number_of_cards=NUMBER_OF_CARDS)
    my_player = my_game.players[0]
    assert not my_player.is_my_turn
    assert len(my_player.hand) == CARDS_IN_HAND
    assert my_player.number_of_cards_i_need_to_play == CARDS_PER_TURN


def test_pile_creation():
    """
    Verify pile creation

    Returns
    -------
    None.

    """
    my_game = game.Game(
        cards_in_hand=CARDS_IN_HAND,
        number_of_players=NUMBER_OF_PLAYERS,
        number_of_piles=NUMBER_OF_PILES,
        cards_per_turn=CARDS_PER_TURN,
        number_of_cards=NUMBER_OF_CARDS)
    my_decreasing_pile = my_game.piles[0]
    my_increasing_pile = my_game.piles[1]
    my_drawing_pile = my_game.drawing_pile
    assert my_increasing_pile.cards == [game.LOWEST_PLAYABLE_NUMBER - 1]
    assert my_increasing_pile.id_number == 1
    assert my_increasing_pile.get_top_card() == game.LOWEST_PLAYABLE_NUMBER - 1

    assert my_decreasing_pile.cards == [NUMBER_OF_CARDS]
    assert my_decreasing_pile.id_number == 0
    assert my_decreasing_pile.get_top_card() == NUMBER_OF_CARDS

    assert len(my_drawing_pile.cards) == NUMBER_OF_CARDS - \
        2 - CARDS_IN_HAND * NUMBER_OF_PLAYERS


def test_game_creation():
    """
    Verify game creation

    Returns
    -------
    None.

    """
    my_game = game.Game(
        cards_in_hand=CARDS_IN_HAND,
        number_of_players=NUMBER_OF_PLAYERS,
        number_of_piles=NUMBER_OF_PILES,
        cards_per_turn=CARDS_PER_TURN,
        number_of_cards=NUMBER_OF_CARDS)
    assert len(my_game.piles) == NUMBER_OF_PILES
    assert len(my_game.players) == NUMBER_OF_PLAYERS