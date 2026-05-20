import numpy as np
import pyspiel
from open_spiel.python import policy as policy_lib

import warnings
warnings.filterwarnings("ignore")

class BRSPlus(pyspiel.Bot):

    def std_value_function(state, p, c=25):
        returns = 0 
        for _ in range(c):
            clone = state.clone()
            while not clone.is_terminal():
                la = clone.legal_actions()
                clone.apply_action(la[np.random.randint(len(la))])
            returns += clone.returns()[p]
        return returns / c

    def __init__(self, game, player_id, depth=2, value_function=std_value_function):
        super().__init__()
        self.game = game
        self.player_id = player_id
        self.default_depth = depth
        self.value_function = value_function  
        
    def restart_at(self, state): pass

    def simulate_default_sequence(self, state, target_player):
        seq_state = state.clone()
        # Loop until it's target_player's turn or terminal
        while not seq_state.is_terminal() and seq_state.current_player() != target_player:
            legal = seq_state.legal_actions()
            move = legal[np.random.randint(len(legal))]
            seq_state.apply_action(move)
        return seq_state

    def evaluate(self, state, depth):
        num_players = state.num_players()
        if state.is_terminal():
            return list(state.returns())

        if depth <= 0:
            if self.value_function is None:
                raise ValueError("Reached depth 0 but no value_function provided for non-terminal node.")
            # return vector [value_function(state,p) for p in range(num_players)]
            return [ self.value_function(state, p) for p in range(num_players) ]

        if state.is_chance_node():
            values = [0.0] * num_players
            for action, prob in state.chance_outcomes():
                child = state.clone()
                child.apply_action(action)
                child_vals = self.evaluate(child, depth-1)
                for p in range(num_players):
                    values[p] += prob * child_vals[p]
            return values

        current = state.current_player()
        if current == self.player_id:
            best_vec = None
            for action in state.legal_actions():
                # apply action
                child = state.clone()
                child.apply_action(action)
                vec = self.brs_response(child, depth-1)
                if best_vec is None or vec[self.player_id] > best_vec[self.player_id]:
                    best_vec = vec
            return best_vec if best_vec is not None else list(state.returns())

        else:
            return self.brs_response(state, depth)

    def brs_response(self, state, depth):
        if state.is_terminal():
            return list(state.returns())

        if depth <= 0:
            if self.value_function is None:
                raise ValueError("Reached depth 0 but no value_function provided for non-terminal node.")
            return [ self.value_function(state, p) for p in range(state.num_players()) ]

        current = state.current_player()
        if current == self.player_id:
            return self.evaluate(state, depth)

        best_vec = None
        root_val_best = float('inf')
        for action in state.legal_actions():
            child = state.clone()
            child.apply_action(action)
            next_state = self.simulate_default_sequence(child, self.player_id)
            vec = self.evaluate(next_state, depth-1)
            root_val = vec[self.player_id]
            if best_vec is None or root_val < root_val_best:
                best_vec = vec
                root_val_best = root_val
        return best_vec if best_vec is not None else list(state.returns())

    def step(self, state):

        if state.is_terminal(): return None  

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            r = np.random.random()
            cum = 0.0
            for action, prob in outcomes:
                cum += prob
                if r < cum:
                    return action
            return outcomes[-1][0]

        best_action = None
        best_val = -float('inf')
        for action in state.legal_actions():
            child = state.clone()
            child.apply_action(action)
            vec = self.brs_response(child, self.default_depth - 1)
            root_val = vec[self.player_id]
            if best_action is None or root_val > best_val:
                best_action = action
                best_val = root_val
        return best_action


# import numpy as np
# import pyspiel
# from open_spiel.python import policy as policy_lib

# import warnings
# warnings.filterwarnings("ignore")

# class BRSPlus(pyspiel.Bot):
#     def __init__(self, game, player_id, alpha=-0.3, beta=0.3, delta=0.5):
#         super().__init__()
#         self.game = game
#         self.player_id = player_id

#     def restart_at(self, state): pass

#     def simulate_default_sequence(self, state, target_player):
#         seq_state = state.clone()
#         while seq_state.current_player() != target_player and not seq_state.is_terminal():
#             move = seq_state.legal_actions()[np.random.randint(len(seq_state.legal_actions()))] 
#             seq_state = seq_state.apply_action(move)
#         return seq_state

#     def brs_opponent_response(self, state, player_id, depth, value_function=None):
#         if state.is_terminal():
#             return list(state.returns())
#         if depth == 0:
#             if value_function is None:  raise ValueError("Reached depth 0 but no value_function provided for non-terminal node.")
#             return [value_function(state, p) for p in range(state.num_players)]
#         min_root_val = 1000000
#         selected_opponent = None 
#         for a in state.legal_actions():
#             vec = list(state.returns())
#             if vec[player_id] < min_root_val:
#                 min_root_val = vec[player_id]
#                 selected_opponent = state.current_player()
#         seq_state = state.clone() 
#         moves_simulated = 0
#         while state.current_player() != selected_opponent:
#             moves = state.legal_actions()
#             move = moves[np.random.randint(len(moves))]
#             moves_simulated += 1
#             seq_state = seq_state.apply_move(move)
#             if seq_state.is_terminal(): return list(state.returns())
#         best_vec = None 
#         for action in state.legal_actions():
#             child = seq_state
#             child.apply_action(action)
#             next_state = self.simulate_default_sequence(child, player_id)
#             vec = self.brs_opponent_response(next_state, player_id, depth - 1 -  moves_simulated)
#             if best_vec is None or vec[player_id] < best_vec[player_id]: best_vec = vec 
#         return best_vec
        

        


    
#     def step(self, state, player_id, depth=100, value_function=None):

#         num_players = state.num_players()
#         if state.is_terminal():
#             return list(state.returns())
#         if depth == 0:
#             if value_function is None:  raise ValueError("Reached depth 0 but no value_function provided for non-terminal node.")
#             return [value_function(state, p) for p in range(num_players)]

#         current_player = state.current_player()

#         # Chance node: expectation over chance outcomes
#         if state.is_chance_node():
#             # initialize sum of values
#             values = [0.0] * num_players
#             for action, prob in state.chance_outcomes():
#                 child = state.clone()
#                 child.apply_action(action)
#                 child_vals = self.step(child, depth, value_function, None)
#                 for p in range(num_players):
#                     values[p] += prob * child_vals[p]
#             return values

#         if current_player == player_id:
#             best_vec = None 
#             for action in state.legal_actions(): 
#                 child = state.clone()
#                 child.apply_action(action)
#                 vec = self.brs_opponent_response(child, player_id, depth-1)
#                 if best_vec is None or vec[player_id] > best_vec[player_id]:
#                     best_vec = vec 
#             return best_vec 
#         else: return self.brs_opponent_response(state, depth)
