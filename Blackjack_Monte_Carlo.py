# We are going to be implementing first-visit Monte Carlo Prediction on the Black Jack Problem
# First we will need to define the game

import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib

# We will create a Card class to make the entire game more Object oriented and simpler to implement
class Card:

    conversion = {'2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10,
                  'J' : 10, 'Q' : 10, 'K' : 10, 'A' : 1}
    def __init__(self, suit, name):
        self.suit = suit
        self.name = name
        self.value = Card.conversion[name]
        self.is_ace = name == 'A'

    def __str__(self):
        return "{0} of {1}".format(self.name, self.suit)

# We will also create a Deck card for the dealer (house) to use. For our version of BJ the deck will be delt
# with replacement, so that each card is equally likely to be delt at any given time.
class Deck:
    suits = ['Clubs', 'Diamonds', 'Hearts', "Spades"]
    values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    def __init__(self):
        self.deck = []
        for suit in Deck.suits:
            for value in Deck.values:
                self.deck.append(Card(suit, value))

    def draw(self):
        return random.choice(self.deck)


class Blackjack:
    # class variables
    GAME_DRAW = 0
    GAME_LOSS = -1
    GAME_WON = 1

    def __init__(self):
        self.dealer = []
        self.player = []
        self.deck = Deck()

    def count(self, hand="player"):
        total = 0
        usable_ace = False
        if hand == 'player':
            for card in self.player:
                total += card.value
                if card.is_ace:
                    usable_ace = True
        else:
            for card in self.dealer:
                total += card.value
                if card.is_ace:
                    usable_ace = True

        if usable_ace and total <= 11:
            total += 10

        return total, usable_ace

    def player_hit(self):
        self.player.append(self.deck.draw())
        if self.count()[0] > 21:
            return False
        return True

    def dealer_hit(self):
        self.dealer.append(self.deck.draw())
        if self.count(hand='dealer')[0] > 21:
            return False
        return True

    def actions(self):
        return ['hit', 'stick']

    def get_action_policy(self):
        if self.count()[0] < 20:
            return 'hit'
        return 'stick'

    def get_state(self):
        return self.count()[0], self.dealer[0].value, self.count()[1]

    def game_init(self):
        self.dealer.clear()
        self.player.clear()

        self.dealer.append(self.deck.draw())
        self.dealer.append(self.deck.draw())

        self.player.append(self.deck.draw())
        self.player.append(self.deck.draw())

        # First check if the game is a natural
        player_value, usable_ace = self.count(hand='player')
        dealer_value, _ = self.count(hand='dealer')
        if player_value == 21 and dealer_value == 21:
            return Blackjack.GAME_DRAW
        elif player_value == 21:
            return Blackjack.GAME_WON
        elif dealer_value == 21:
            return Blackjack.GAME_LOSS

        while self.count(hand='player')[0] < 12:
            self.player_hit()

        return None

    def game_loop(self):
        still_playing = True
        while still_playing:
            action = self.get_action_policy()
            if action == 'hit':
                still_playing = self.player_hit()
            elif action == 'stick':
                still_playing = False

        if self.count(hand='player')[0] > 21:
            return Blackjack.GAME_LOSS

        while self.count(hand='dealer')[0] < 17:
            self.dealer_hit()

        if self.count(hand='player')[0] > self.count(hand='dealer')[0]:
            return Blackjack.GAME_WON
        elif self.count(hand='player')[0] < self.count(hand='dealer')[0]:
            return Blackjack.GAME_LOSS
        else:
            return Blackjack.GAME_DRAW

    def play_blackjack(self, times):
        player_wins = 0
        for i in range(times):
            ret = self.game_init()
            if ret == Blackjack.GAME_WON:
                player_wins += 1

            if ret is not None:
                continue

            ret = self.game_loop()
            if ret == Blackjack.GAME_WON:
                player_wins += 1

        return player_wins

    def simulate_blackjack(self):
        ret = self.game_init()
        states = []
        if ret is not None:
            return states, ret

        still_playing = True
        while still_playing:
            states.append(self.get_state())
            action = self.get_action_policy()
            if action == 'hit':
                still_playing = self.player_hit()
            elif action == 'stick':
                still_playing = False

        if self.count(hand='player')[0] > 21:
            return states, Blackjack.GAME_LOSS

        while self.count(hand='dealer')[0] < 17:
            self.dealer_hit()

        if self.count(hand='player')[0] > self.count(hand='dealer')[0]:
            return states, Blackjack.GAME_WON
        elif self.count(hand='player')[0] < self.count(hand='dealer')[0]:
            return states, Blackjack.GAME_LOSS
        else:
            return states, Blackjack.GAME_DRAW


    def monte_carlo_prediction(self, times):
        '''
        :param times: number of iterations to perform monte_carlo
        :return: the predicted value function from monte carlo simulation
        '''

        # pseudo code for first visit MC method
        # Initialize:
        # π ← policy to be evaluated
        # V ← an arbitrary state-value function
        # Returns(s) ← an empty list, for all s ∈ S
        # Repeat forever:
        # Generate an episode using π
        # For each state s appearing in the episode:
        # G ← return following the first occurrence of s
        # Append G to Returns(s)
        # V (s) ← average(Returns(s))

        # In blackjack, it is impossible to visit a state more than once during one episode so first-visit MC
        # and every-visit MC are the same for this application

        returns = {}
        visited = {}
        value_function = {}
        for episode in range(times):
            states, ret = self.simulate_blackjack()
            for s in states:
                if s in returns:
                    returns[s] += ret
                    visited[s] += 1
                else:
                    returns[s] = ret
                    visited[s] = 1

        for s in returns:
            value_function[s] = returns[s]/visited[s]

        return value_function

    def display_value_function(self, value_function):
        ax1 = plt.figure().add_subplot(projection='3d')
        X = np.linspace(12, 21, 10)
        Y = np.linspace(2, 11, 10)
        Z = np.zeros(shape=(10, 10))
        for i in range(12, 22):
            for j in range(2, 12):
                if (i, j, False) in value_function:
                    Z[i-12][j-2] = value_function[(i, j, False)]
                else:
                    Z[i - 12][j - 2] = 0
        ax1.contour(X, Y, Z, extend3d=True, cmap=cm.coolwarm)
        ax1.set_xlabel("Player Sum")
        ax1.set_ylabel("Dealer Showing")
        ax1.set_title("Predicted Value Function No Usable Ace")

        ax2 = plt.figure().add_subplot(projection='3d')
        X = np.linspace(12, 21, 10)
        Y = np.linspace(2, 11, 10)
        Z = np.zeros(shape=(10, 10))
        for i in range(12, 22):
            for j in range(2, 12):
                if (i, j, True) in value_function:
                    Z[i - 12][j - 2] = value_function[(i, j, True)]
                else:
                    Z[i - 12][j - 2] = 0
        ax2.contour(X, Y, Z, extend3d=True, cmap=cm.coolwarm)
        ax2.set_xlabel("Player Sum")
        ax2.set_ylabel("Dealer Showing")
        ax2.set_title("Predicted Value Function Usable Ace")

        plt.show()


# demo
blackjack = Blackjack()
val_fnc = blackjack.monte_carlo_prediction(10000000)
blackjack.display_value_function(val_fnc)







