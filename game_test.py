# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 11:55:06 2021

@author: Fiona
"""
import game

def test_calc_addition():
    """
    Verify player creation

    Returns
    -------
    None.

    """
    player= game.Player()
    assert player.is_my_turn==False
    assert len(player.cards)==game.CARDS_IN_HAND
    assert player.number_of_cards_i_need_to_play==game.CARDS_PER_TURN