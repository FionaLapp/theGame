# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 18:09:48 2021

@author: Fiona
"""

from abc import ABC, abstractmethod
#TODO add toString methods

CARDS_IN_HAND=6
NUMBER_OF_PLAYERS=4
NUMBER_OF_PILES=4
CARDS_PER_TURN=2
NUMBER_OF_CARDS=100
LOWEST_PLAYABLE_NUMBER=2


class Player:
    def __init__(self):
        self.cards=[]
        #TODO draw 6 cards       
        self.is_my_turn= False
        self.number_of_cards_i_need_to_play=CARDS_PER_TURN
        
    def card_playable(self, card):
        #TODO check if playable
        print("checking if {} playable".format(card))
        return False
        
    def play_card(self, card):
        """
        This function does everything associated with playing a card on the player-side: removing it from their hand, decreasing the number of cards they need to play
        inside the game object, play_card must be called for pile and player
        This needs to throw an exception if the card is not playable
        """
        if not self.card_playable(card):
            #TODO throw exception
            print("not possible")
        else:
            #TODO add card to pile
            print("removing card {} from hand".format( card))
        #TODO decrease number of cards I need to play
        #TODO remove card from  my cards
        
    def draw_card(self):
        """
        This function does everything associated with drawing a card on the player-side: adding it to their hand
        inside the game object, draw_card must be called for pile and player
        """
        #TODO draw card
        print("drawing card")
    def __str__(self):
        return "Player with cards {}".format(self.cards)
        
    
    
    
    
    
class Pile(ABC):
    
    def __init__(self):
        self.cards=[]

        
class DrawingPile(Pile):
    
    def __init__(self, size):
        self.cards=[*range(LOWEST_PLAYABLE_NUMBER, size)] #if size=100, this will go up to 99
    
    def draw_random_card():
        """
        This function does everything associated with drawing a card on the pile-side: removing it from the pile
        inside the game object, draw_card must be called for pile and player
        """
        #TODO randomly pick a card
        card="dummy"
        
        #TODO remove card from pile
        print("drawing random card: ", card)
        
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
            print("not possible")
        else:
            #TODO add card to pile
            print("adding card {} to pile".format( card))
    
    def get_top_card(self):
        #TODO get top card
        print("getting top card")
    
class DecreasingPile(PlayingPile):
        
    def card_playable(self,card):
        #TODO check if playable
        print("checking if {} playable".format(card))
        return False
    
class IncreasingPile(PlayingPile):
        
    def card_playable(self,card):
        #TODO check if playable
        print("checking if {} playable".format(card))
        return False
    
        
class Game:
    def __init__(self):
        self.drawing_pile= DrawingPile()
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
        print("{} starts game".format(first_player))
        
    def check_if_finished(self):
        #TODO check if finished
        print("checking if finished")
        game_finished=False
        return game_finished

    def play_card(self, player, pile, card):
        #TODO play card
        print("{} plays {} on {} ".format(player, card, pile))
    
    def draw_card(self, player, pile, card):
        #TODO draw card
        print("{} draws {} from {} ".format(player, card, pile))
    
    def _create_piles(self, number_of_piles):
        print("creating {} piles".format(number_of_piles))
    
    def _create_players(self, number_of_players):
        print("creating {} players".format(number_of_players))
        
        
if __name__ == "__main__":
    player1=Player()
    player1.draw_card()
    pile1=DecreasingPile()
    pile1.card_playable(1)
    pile2= DrawingPile(100)
    print(pile2.cards)
    
    