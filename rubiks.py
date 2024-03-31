import numpy as np
import torch
import random

# Define the rubik's cube environment
class Rubiks:
    R = 0
    R_PRIME = 1
    U = 2
    U_PRIME = 3
    L = 4
    L_PRIME = 5
    F = 6
    F_PRIME = 7
    B = 8
    B_PRIME = 9
    D = 10
    D_PRIME = 11

    ORANGE = 0
    WHITE = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    RED = 5

    def __init__(self):
        self.states = np.zeros((6,3,3), dtype=int) # first array is the orange face, second array is white face, then green, then yellow, then blue, then red
        self.actions = [move for move in range(12)]

        # initialize rubik's cube to solved state
        for i in range(1, 6):
            self.states[i] = np.full((3, 3), i)

    def transition(self, s, a):
        pass


if __name__ == '__main__':
    print(Rubiks().states)
