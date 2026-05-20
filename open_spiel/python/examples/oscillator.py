import numpy as np
import random
from collections import defaultdict
from open_spiel.python import rl_agent
import pyspiel

class Oscillator(rl_agent.AbstractAgent):
    def __init__(self, game: pyspiel.Game, player_id: int):
        # Initialize using only player_id per PySpiel Bot API
        self.game = game
        self.player_id = player_id
        self.mul = 1
        self.states = {
                '0'   : np.random.random(),
                '0p'  : np.random.random(),
                '0b'  : np.random.random(),
                '0pb' : np.random.random(),
                '1'   : np.random.random(),
                '1p'  : np.random.random(),
                '1b'  : np.random.random(),
                '1pb' : np.random.random(),
                '2'   : np.random.random(),
                '2p'  : np.random.random(),
                '2b'  : np.random.random(),
                '2pb' : np.random.random(),
            }

    def restart(self): pass
    
    def step(self, state):
        iss = state.information_state_string(self.player_id)
        move = np.random.random()
        shift = np.random.random() / 40
        move = bool(move < self.states[iss])
        self.states[iss] += self.mul * shift
        if self.states[iss] > 1: self.states[iss] = 2 - self.states[iss] ; self.mul = -1
        if self.states[iss] < 0: self.states[iss] *= -1; self.mul = 1
        return move