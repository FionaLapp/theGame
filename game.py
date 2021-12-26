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


LOG_LEVEL=logging.INFO
# CARDS_IN_HAND=6
# NUMBER_OF_PLAYERS=4
# NUMBER_OF_PILES=4
# CARDS_PER_TURN=2
# NUMBER_OF_CARDS=100
LOWEST_PLAYABLE_NUMBER=2
DECREASING="DECREASING"
INCREASING="INCREASING"

#%% configuration

def configure_logging():
    # create logger
    logger = logging.getLogger('debug_logger')
    logger.setLevel(LOG_LEVEL)
    # create console handler and set level to debug
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(LOG_LEVEL)
    logger.addHandler(console)
    return logger

#%% errors

class InputValidationError(Exception):
    """Exception raised for invalid inputs.

    Attributes:
        input_value -- invalid input value
        message -- explanation of the error
    """

    def __init__(self, input_value, message="input not valid"):
        logger.handlers.clear()
        self.input_value=input_value
        self.message = message
        super().__init__(self.message)

class CardNotPlayableError(Exception):
    """Exception raised when card cannot be played.

    Attributes:
        card -- invalid card
        message -- explanation of the error
    """

    def __init__(self, card, message="card cannot be played"):
        logger.handlers.clear()
        self.card=card
        self.message = message
        super().__init__(self.message)

#%% player

class Player:
    def __init__(self, id_number, drawing_pile, game, cards_in_hand, cards_per_turn):
        self.id_number=id_number
        self.hand=[]
        #draw as many cards as a hand needs
        for i in range(cards_in_hand):
            game.draw_card(self, drawing_pile)
        self.is_my_turn= False
        self.number_of_cards_i_need_to_play=cards_per_turn
        logger.debug("{} initialised successfully".format(self))
        
    def card_playable(self, card):
        #check if playable
        logger.debug("checking if {} playable".format(card))
        if card in self.hand:
            return True
        else:
            return False
        
    def play_card(self, card):
        """
        This function does everything associated with playing a card on the player-side: removing it from their hand, decreasing the number of cards they need to play
        inside the game object, remove_card_from_hand must be called for pile and player
        This needs to throw an exception if the card is not playable
        """
        if not self.card_playable(card):
            #throw exception
            logger.debug("not possible")
            raise CardNotPlayableError(card, "Cannot play {} because it is not contained in hand {}".format(card, self.hand))
            
        else:
            #remove card from hand
            self.hand.remove(card)
            #decrease number of cards I need to play
            if self.number_of_cards_i_need_to_play>0:
                self.number_of_cards_i_need_to_play-=1
            else:
                self.number_of_cards_i_need_to_play=0
            
        
    def add_card_to_hand(self, card):
        """
        This function does everything associated with drawing a card on the player-side: adding it to their hand
        inside the game object, draw_card must be called for pile and player
        """
        #add card
        logger.debug("adding card {} to hand".format(card))
        self.hand.append(card)
        
    def __str__(self):
        return "Player {} with cards {}".format(self.id_number, self.hand)
        
#%% pile    
      
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
        #randomly pick a card
        random_index=randint(0,len(self.cards)-1)
        
        #remove card from pile
        try:
            card=self.cards.pop(random_index)
        except Exception:
            #this is only in here because I coded it without the -1 before; thought I might as well leave it in
            raise InputValidationError(random_index, "Something went wrong with drawing the cards: tried to draw card at index {} but there are only {} cards available".format(random_index, len(self.cards)))
        
        logger.debug("drew random card: {}".format(card))
        
        return card
    
    def __str__(self):
        return "DrawingPile"
     


class PlayingPile(Pile):
    
    def __init__(self, id_number):
        super().__init__()
        self.id_number= id_number
    
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
            #throw exception
            logger.debug("not possible")
            raise CardNotPlayableError(card, "Cannot play {} ({}) on pile with top card {}".format(card, self.pile_type, self.get_top_card()))
            
            
        else:
            #add card to pile
            self.cards.append(card)
            logger.debug("adding card {} to pile".format( card))
    
    def get_top_card(self):
        #get top card
        top_card=self.cards[-1]
        logger.debug("getting top card: {}".format(top_card))
        return top_card 
    
    def __str__(self):
        return "Playing Pile {}".format(self.id_number)
        
    
class DecreasingPile(PlayingPile):
    def __init__(self, id_number, number_of_cards):
        super().__init__(id_number)
        self.pile_type=DECREASING
        self.cards.append(number_of_cards)
        
    def card_playable(self,card):
        #check if playable
        top_card=self.get_top_card()
        if card<top_card or card==top_card+10:
            logger.debug("card {} playable on decreasing pile with top card {}".format(card, top_card))
            return True
        else:
            logger.debug("card {} not playable on decreasing pile with top card {}".format(card, top_card))
            return False
    
    def __str__(self):
        return "Decreasing Pile {}".format(self.id_number)
     
    
class IncreasingPile(PlayingPile):
    def __init__(self, id_number):
        super().__init__(id_number)
        self.pile_type=INCREASING
        self.cards.append(LOWEST_PLAYABLE_NUMBER-1)
        
    def card_playable(self,card):
        #check if playable
        top_card=self.get_top_card()
        if card>top_card or card==top_card-10:
            logger.debug("card {} playable on increasing pile with top card {}".format(card, top_card))
            return True
        else:
            logger.debug("card {} not playable on increasing pile with top card {}".format(card, top_card))
            return False
    
    def __str__(self):
        return "Increasing Pile {}".format(self.id_number)

#%% Game    
        
class Game():
    def __init__(self, cards_in_hand=6, number_of_players=4, number_of_piles=4, cards_per_turn=2, number_of_cards=100):
        self.cards_in_hand= cards_in_hand
        self.number_of_players=number_of_players
        self.number_of_piles=number_of_piles
        self.cards_per_turn=cards_per_turn
        self.number_of_cards=number_of_cards
        self.drawing_pile= DrawingPile(number_of_cards)
        self.piles=[]
        #add piles
        self._create_piles(self.number_of_piles, self.number_of_cards)
        
        self.players=[]
        #add players
        self._create_players(self.number_of_players, self.cards_in_hand, self.cards_per_turn)
           
        self.current_player=None
        self.next_player=None
       
    #TODO add game logic  
    def play(self, first_player):
        #TODO start game 
        #set current_player, next_player
        #play card
        #check if finished
        #draw card if player change
        logger.debug("{} starts game".format(first_player))
    
    def game_won(self):
        if self.drawing_pile.cards!=[]: 
            #there are still cards on the drawing pile
            return False
        else:
            for player in self.players:
                if player.hand!=[]:
                    #a player's hand isn't empty
                    return False
            #no cards in drawing pile and all hands empty
            return True
    def game_lost(self):
        if not self.game_won():
            #there are still cards that haven't been played
            can_play_matrix=self.can_play(self.current_player) #true if player can play, false otherwise
            return np.all(np.invert(can_play_matrix))#all have to be true
                        
        
    def check_if_finished(self):
        #check if finished
        logger.debug("checking if finished")
        if self.game_won() or self.game_lost():
            return True
        return False
    
    def can_play(self, player):
        can_play_matrix=np.zeros(len(player.hand), len(self.piles))
        for i, card in enumerate(player.hand):
            for j, pile in enumerate(self.piles):
                can_play_matrix[i, j]=self.card_playable(player, pile, card)
        return can_play_matrix        
    
    def card_playable(self, player, pile, card):
        return player.card_playable(card) and pile.card_playable(card)
    
    def play_card(self, player, pile, card):
        #play card
        if self.card_playable(player, pile, card):
            player.remove_card_from_hand(card)
            pile.play_card(card)
            logger.debug("{} plays {} on {} ".format(player, card, pile))
        else:
            logger.debug(CardNotPlayableError)
            raise CardNotPlayableError(card)
            
        
    def draw_card(self, player, drawing_pile):
        #draw card
        card=drawing_pile.remove_card()
        player.add_card_to_hand(card)
        logger.debug("{} drew {} from {} ".format(player, card, drawing_pile))
    
    def _create_piles(self, number_of_piles, number_of_cards):
        if number_of_piles%2!=0:
            raise InputValidationError(number_of_piles, "I'd really prefer an even number of piles, and {} might be a lot of things but it ain't that".format(number_of_piles))
        for i in range(int(number_of_piles/2)):
            self.piles.append(DecreasingPile(2*i, number_of_cards))
            self.piles.append(IncreasingPile((2*i)+1))
        logger.debug("creating {} piles".format(number_of_piles))
    
    def _create_players(self, number_of_players, cards_in_hand, cards_per_turn):
        for i in range(int(number_of_players)):
            self.players.append(Player(i, self.drawing_pile, self, cards_in_hand, cards_per_turn))
        logger.debug("creating {} players".format(number_of_players))
    
    def __str__(self):
        return "Game"
        
if __name__ == "__main__":
    logger=configure_logging()
    #logging.basicConfig(handler=logging.StreamHandler(sys.stdout), level=LOG_LEVEL)
    game=Game()
    print(game.piles[1])
    player=Player(1, game.drawing_pile, game, 6, 2)
    pile=DecreasingPile(1, 100)
    #player.play_card(1)
    logger.handlers.clear()
    # pile1=DecreasingPile()
    # pile1.card_playable(1)
    # pile2= DrawingPile(100)
    # player=Player(pile2, game)
    # logger.debug(player.hand)
    
    # logger.debug(pile2.remove_card())
    # logger.debug(pile2.cards)
    
    

    