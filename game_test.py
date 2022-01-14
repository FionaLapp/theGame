# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 11:55:06 2021

@author: Fiona
"""
import the_game
import numpy as np


CARDS_IN_HAND = 2
NUMBER_OF_PLAYERS = 1
NUMBER_OF_PILES = 2
CARDS_PER_TURN = 1
NUMBER_OF_CARDS = 10
STRATEGIES=[the_game.PlayWithMetricStrategy, the_game.PlayWithDistanceCutoffStrategy]



def test_player_creation():
    """
    Verify player creation

    Returns
    -------
    None.

    """
    my_game = the_game.Game(
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
    my_game = the_game.Game(
        cards_in_hand=CARDS_IN_HAND,
        number_of_players=NUMBER_OF_PLAYERS,
        number_of_piles=NUMBER_OF_PILES,
        cards_per_turn=CARDS_PER_TURN,
        number_of_cards=NUMBER_OF_CARDS)
    my_decreasing_pile = my_game.piles[0]
    my_increasing_pile = my_game.piles[1]
    my_drawing_pile = my_game.drawing_pile
    assert my_increasing_pile.cards == [the_game.LOWEST_PLAYABLE_NUMBER - 1]
    assert my_increasing_pile.id_number == 1
    assert my_increasing_pile.top_card == the_game.LOWEST_PLAYABLE_NUMBER - 1

    assert my_decreasing_pile.cards == [NUMBER_OF_CARDS]
    assert my_decreasing_pile.id_number == 0
    assert my_decreasing_pile.top_card == NUMBER_OF_CARDS

    assert len(my_drawing_pile.cards) == NUMBER_OF_CARDS - \
        2 - CARDS_IN_HAND * NUMBER_OF_PLAYERS


def test_game_creation():
    """
    Verify game creation

    Returns
    -------
    None.

    """
    my_game = the_game.Game(
        cards_in_hand=CARDS_IN_HAND,
        number_of_players=NUMBER_OF_PLAYERS,
        number_of_piles=NUMBER_OF_PILES,
        cards_per_turn=CARDS_PER_TURN,
        number_of_cards=NUMBER_OF_CARDS)
    assert len(my_game.piles) == NUMBER_OF_PILES
    assert len(my_game.players) == NUMBER_OF_PLAYERS

def test_game_result_with_metric_win():
    """
    Play a game

    Returns
    -------
    None.

    """
    #as many piles as there are cards --> guaranteed win
    for strategy in STRATEGIES:
        my_strategy=strategy(
            cards_in_hand=CARDS_IN_HAND,
            number_of_players=NUMBER_OF_PLAYERS,
            number_of_piles=NUMBER_OF_CARDS,
            cards_per_turn=CARDS_PER_TURN,
            number_of_cards=NUMBER_OF_CARDS)
        my_strategy.start_game(my_strategy.game.players[0])
        my_strategy.play()
        my_game=my_strategy.game
        assert my_game.game_finished()
        assert my_game.game_won()
        print(my_game.current_player)
        print(my_game.current_player.hand)
        print(my_game.piles[0])
        assert (not my_game.game_lost())
        assert len(my_game.drawing_pile.cards)==0
        for player in my_game.players:
            assert len(player.hand)==0

def test_game_result_with_metric_lost():
    """
    Play a game

    Returns
    -------
    None.

    """
    #as many piles as there are cards --> guaranteed win
    for strategy in STRATEGIES:
        my_strategy=strategy(
            cards_in_hand=CARDS_IN_HAND,
            number_of_players=NUMBER_OF_PLAYERS,
            number_of_piles=2,
            cards_per_turn=CARDS_PER_TURN,
            number_of_cards=NUMBER_OF_CARDS)
        new_drawing_pile=my_strategy.game.drawing_pile.cards
        new_drawing_pile.extend(new_drawing_pile)
        my_strategy.game.drawing_pile.cards.extend(new_drawing_pile)
        my_strategy.start_game(my_strategy.game.players[0])
        my_strategy.play()
        my_game=my_strategy.game
        assert my_game.game_finished()
        assert (not my_game.game_won())
        assert (my_game.game_lost())

def test_card_playable():
    my_game = the_game.Game(
        cards_in_hand=CARDS_IN_HAND,
        number_of_players=NUMBER_OF_PLAYERS,
        number_of_piles=NUMBER_OF_PILES,
        cards_per_turn=CARDS_PER_TURN,
        number_of_cards=NUMBER_OF_CARDS)
    my_player=my_game.players[0]
    my_player.hand.append(NUMBER_OF_CARDS+1)
    my_player.hand.append(0)
    decreasing_piles, increasing_piles=my_game.separate_piles()
    #pile0 should be decreasing, pile1 increasing
    assert my_game.card_playable(my_player, decreasing_piles[0], 0)
    assert not my_game.card_playable(my_player, decreasing_piles[0], NUMBER_OF_CARDS +1)
    assert my_game.card_playable(my_player, increasing_piles[0], NUMBER_OF_CARDS +1)
    assert not my_game.card_playable(my_player, increasing_piles[0], 0)

def test_drawing_for_empty_hand():
    """
    Check that even if a player does not want to draw, they eventually will once their hand is empty

    Returns
    -------
    None.

    """
    my_game = the_game.Game(number_of_players=2,
        cards_in_hand=2, cards_per_turn=1)
    my_player=my_game.players[0]
    my_game.start_game()
    #print(my_game.piles[0], my_player.hand[0])
    my_game.play_card(my_game.piles[0], my_player.hand[0], False)
    #print(my_game.piles[1], my_player.hand[0])
    my_game.play_card(my_game.piles[1], my_player.hand[0], False)
    #print(my_game.print_hands())
    #print(my_game.current_player)
    assert isinstance(my_game.current_player, the_game.Player)
    assert len(my_player.hand)==2
    assert my_game.current_player==my_game.players[1]

def test_basic_metric():
    my_game=the_game.Game(number_of_players=2,
        cards_in_hand=2, cards_per_turn=1, number_of_cards=100, number_of_piles=2)
    my_player=my_game.players[0]
    decreasing_piles, increasing_piles=my_game.separate_piles()
    increasing_pile=increasing_piles[0]
    my_game.start_game()
    my_player.hand=[2,12]
    my_game.play_card(increasing_pile, 12, False)
    my_player.hand=[2,3]
    my_game.current_player=my_player
    my_game.calculate_basic_metric()

    assert np.array_equal(my_game.basic_metric, np.array([[98, -10], [97, 101]]))

