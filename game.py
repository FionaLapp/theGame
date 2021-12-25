# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 18:09:48 2021

@author: Fiona
"""

from abc import ABCMeta, abstractmethod
from random import randint
import logging
import sys
#TODO add toString methods

LOG_LEVEL=logging.INFO
CARDS_IN_HAND=6
NUMBER_OF_PLAYERS=4
NUMBER_OF_PILES=4
CARDS_PER_TURN=2
NUMBER_OF_CARDS=100
LOWEST_PLAYABLE_NUMBER=2


def configure_logging():
    # create logger
    logger = logging.getLogger('debug_logger')
    logger.setLevel(LOG_LEVEL)
    # create console handler and set level to debug
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(LOG_LEVEL)
    logger.addHandler(console)
    return logger
logger=configure_logging()

class Player:
    def __init__(self, drawing_pile, game):
        self.hand=[]
        #draw as many cards as a hand needs
        for i in range(CARDS_IN_HAND):
            game.draw_card(self, drawing_pile)
        self.is_my_turn= False
        self.number_of_cards_i_need_to_play=CARDS_PER_TURN
        
    def card_playable(self, card):
        #TODO check if playable
        logger.info("checking if {} playable".format(card))
        return False
        
    def play_card(self, card):
        """
        This function does everything associated with playing a card on the player-side: removing it from their hand, decreasing the number of cards they need to play
        inside the game object, play_card must be called for pile and player
        This needs to throw an exception if the card is not playable
        """
        if not self.card_playable(card):
            #TODO throw exception
            logger.info("not possible")
        else:
            #TODO add card to pile
            logger.info("removing card {} from hand".format(card))
        #TODO decrease number of cards I need to play
        #TODO remove card from  my cards
        
    def add_card_to_hand(self, card):
        """
        This function does everything associated with drawing a card on the player-side: adding it to their hand
        inside the game object, draw_card must be called for pile and player
        """
        #TODO draw card
        logger.info("adding card {} to hand".format(card))
        self.hand.append(card)
        
    def __str__(self):
        return "Player with cards {}".format(self.hand)
        
    
    
    
    
    
class Pile(metaclass= ABCMeta):
    
    def __init__(self):
        self.cards=[]

        
class DrawingPile(Pile):
    
    def __init__(self, size):
        self.cards=[*range(LOWEST_PLAYABLE_NUMBER, size)] #if size=100, this will go up to 99
    
    def remove_card(self):
        """
        This function does everything associated with drawing a card on the pile-side: removing it from the pile
        inside the game object, draw_card must be called for pile and player
        """
        #TODO randomly pick a card
        random_index=randint(0,len(self.cards))
        card=self.cards.pop(random_index)
        
        #TODO remove card from pile
        logger.info("drawing random card: ".format(card))
        
        return card


class PlayingPile(Pile):
    
    @abstractmethod
    def card_playable(self, card):
        """
        An abstract method that needs to be implemented by increasing and decreasing piles separately
        """
        return False
    
    def play_card(self, card):
        """
        This function does everything associated with playing a card on the pile-side: adding it to the pile
        inside the game object, play_card must be called for pile and player
        This needs to throw an exception if the card is not playable
        """
        if not self.card_playable(card):
            #TODO throw exception
            logger.info("not possible")
        else:
            #TODO add card to pile
            logger.info("adding card {} to pile".format( card))
    
    def get_top_card(self):
        #TODO get top card
        logger.info("getting top card")
    
class DecreasingPile(PlayingPile):
        
    def card_playable(self,card):
        #TODO check if playable
        logger.info("checking if {} playable".format(card))
        return False
    
class IncreasingPile(PlayingPile):
        
    def card_playable(self,card):
        #TODO check if playable
        logger.info("checking if {} playable".format(card))
        return False
    
        
class Game:
    def __init__(self):
        self.drawing_pile= DrawingPile(NUMBER_OF_CARDS)
        self.piles=[]
        #TODO add piles
        self.players=[]
        #TODO add players
        self.current_player=None
        self.next_player=None
       
    #TODO add game logic  
    def play(self, first_player):
        #TODO start game 
        #set current_player, next_player
        #play card
        #check if finished
        #draw card if player change
        logger.info("{} starts game".format(first_player))
        
    def check_if_finished(self):
        #TODO check if finished
        logger.info("checking if finished")
        game_finished=False
        return game_finished

    def play_card(self, player, pile, card):
        #TODO play card
        logger.info("{} plays {} on {} ".format(player, card, pile))
    
    def draw_card(self, player, drawing_pile):
        #TODO draw card
        card=drawing_pile.remove_card()
        player.add_card_to_hand(card)
        logger.info("{} draws {} from {} ".format(player, card, drawing_pile))
    
    def _create_piles(self, number_of_piles):
        logger.info("creating {} piles".format(number_of_piles))
    
    def _create_players(self, number_of_players):
        logger.info("creating {} players".format(number_of_players))
        
        
if __name__ == "__main__":
    #logging.basicConfig(filename='debug.log', encoding='utf-8', level=LOG_LEVEL)
    logger.info("HI")
    game=Game()
    player=Player(game.drawing_pile, game)
    # pile1=DecreasingPile()
    # pile1.card_playable(1)
    # pile2= DrawingPile(100)
    # player=Player(pile2, game)
    # logger.info(player.hand)
    
    # logger.info(pile2.remove_card())
    # logger.info(pile2.cards)
    
    
logger.handlers.clear()
    