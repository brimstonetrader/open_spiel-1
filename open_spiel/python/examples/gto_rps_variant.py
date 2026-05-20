import numpy as np
import random
from collections import defaultdict
from open_spiel.python import rl_agent
from open_spiel.python import policy as policy_lib

import pyspiel

class GTO_RPS(rl_agent.AbstractAgent):
    def __init__(self, game: pyspiel.Game, player_id: int, i: int, name='rps_agent'):
        # Initialize using only player_id per PySpiel Bot API
        self.game = game
        self.player_id = player_id
        self.i = i
    
    def step(self, state):
        i = self.i
        r = np.random.random()
        if self.player_id == 1:
            p_r = 2*i*i/(1+2*i)**2
            p_p = i/(1+2*i)
            p_s = (1+3*i)/(1+2*i)**2
        else:
            p_r = 2*i*(i+1)/(1+2*i)**2
            p_p = i/(1+2*i)
            p_s = (1+i)/(1+2*i)**2
        if r<p_r: return 0
        if r<p_r+p_p: return 1
        return 2
    
    def policy(self, state):
        i = self.i
        if self.player_id == 0:
            p_r = 2*i*i/(1+2*i)**2
            p_p = i/(1+2*i)
            p_s = (1+3*i)/(1+2*i)**2
        else:
            p_r = 2*i*(i+1)/(1+2*i)**2
            p_p = i/(1+2*i)
            p_s = (1+i)/(1+2*i)**2

        r1 = np.random.random()
        r2 = np.random.random()
        r3 = np.random.random()
        sm = r1 + r2 + r3 
        return {0: r1/sm, 1: r2/sm, 2: r3/sm}
    

class GTOPolicy(policy_lib.Policy):
    def __init__(self, game, gto_agent):
        super().__init__(game, list(range(game.num_players())))
        self.gto_agent = gto_agent
        
    def action_probabilities(self, state, player_id=None):
        # Use the GTO agent's policy method
        return self.gto_agent.policy(state)