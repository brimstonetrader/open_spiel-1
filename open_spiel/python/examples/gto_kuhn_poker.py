import numpy as np
import random
from collections import defaultdict
from open_spiel.python import rl_agent
import pyspiel
from open_spiel.python import policy as policy_lib

class GTOKuhnPoker(rl_agent.AbstractAgent):
    def __init__(self, game: pyspiel.Game, player_id: int, name='rwywe_agent'):
        # Initialize using only player_id per PySpiel Bot API
        self.game = game
        self.player_id = player_id
    
    def step(self, state):
        alpha = 0
        iss = state.information_state_string(self.player_id)
        card = int(iss[0])
        r = np.random.random()
        turn = len(iss)
        if self.player_id == 0:
            if turn==1:
                if card==2: 
                    if turn==1: return int(r<3*alpha)
                    else: return 1
                if card==1:
                    if turn==1: return 0
                    else: return int(r<1/3+alpha)
                if card==0:
                    if turn>1: return 0
                    else: return int(r<alpha)
        else:
            r = np.random.random()
            if card==0:
                if iss[1]=='p': return int(r<1/3)
                else: return 0
            if card==1:
                if iss[1]=='p': return 0
                else: return int(r<1/3)
            if card==2: return 1
        

class GTOKuhnPolicy(policy_lib.Policy):
    def __init__(self, game, alpha):
        super().__init__(game, list(range(game.num_players())))
        self.alpha = alpha

    def action_probabilities(self, state, player_id=None):
        # Kuhn equilibrium parameter (OpenSpiel default)
        alpha = self.alpha
        if state.is_chance_node():
            return dict(state.chance_outcomes())

        acts = state.legal_actions()
        out = {a: 0.0 for a in acts}

        player = state.current_player()
        iss = state.information_state_string(player)
        card = int(iss[0])
        history = iss[1:]    # "": root, "p": opponent checked, "b": opponent bet

        # Player 0 (first to act)
        if player == 0:
            # Root
            if history == "":
                if card == 0:        # J
                    out[1] = alpha
                    out[0] = 1 - alpha
                elif card == 1:      # Q
                    out[0] = 1.0
                else:                # K
                    out[1] = 3*alpha
                    out[0] = 1-3*alpha
                return out
            # if history == "b":
            #  ^--- this was the issue! this history never happens
            else:
                if card == 0:        # J
                    out[0] = 1.0     # fold
                elif card == 1:      # Q
                    call = alpha + 1/3
                    out[0] = 1 - call
                    out[1] = call
                else:                # K
                    out[1] = 1.0     # call
                return out
        # if player == 0: out = {0:0.5,1:0.5}
        
        # Player 1 (second to act)
        if player == 1:
            if history == "p":
                if card == 0:        # J
                    out[1] = 1/3
                    out[0] = 2/3
                elif card == 1:      # Q
                    out[0] = 1.0     # check
                else:                # K
                    out[1] = 1.0     # bet
                return out

            if history == "b":
                if card == 0:        # J
                    out[0] = 1.0     # fold
                elif card == 1:      # Q
                    out[1] = 1/3
                    out[0] = 2/3
                else:                # K
                    out[1] = 1.0     # call
                return out

        # p = 1.0 / len(acts)
        # for a in acts:
        #     out[a] = p
        return out
