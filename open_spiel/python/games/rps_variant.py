import pyspiel
import numpy as np

_GAME_TYPE = pyspiel.GameType(
    short_name="extended_rps",
    long_name="Extended Rock Paper Scissors",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,  # Changed to SEQUENTIAL for RCFR
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=3,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-2,
    max_utility=2,
    utility_sum=0,
    max_game_length=3  # Player 1 -> Player 2 -> Terminal
)

class ExtendedRPSGame(pyspiel.Game):
    
    def __init__(self, alpha, params=None):
        if params is None:
            params = {}
        super().__init__(_GAME_TYPE, _GAME_INFO, dict())
        self.num_actions = 3
        self.alpha = alpha
        
    def num_distinct_actions(self):
        return self.num_actions
    
    def max_chance_outcomes(self):
        return 0
    
    def num_players(self):
        return 2
    
    def min_utility(self):
        return -self.alpha
    
    def max_utility(self):
        return self.alpha
    
    def utility_sum(self):
        return 0
    
    def new_initial_state(self):
        return ExtendedRPSState(self)
    
    def max_game_length(self):
        return 2
    
    def action_to_string(self, player, action):
        actions = {0: "Rock", 1: "Paper", 2: "Scissors"}
        return actions.get(action, f"Unknown({action})")
    
    def state_to_string(self, state):
        return str(state)
    
    def information_state_tensor_size(self):
        return 4  # [player_to_move, rock_legal, paper_legal, scissors_legal]
    
    def information_state_tensor_shape(self):
        return (self.information_state_tensor_size(),)

class ExtendedRPSState(pyspiel.State):
    
    def __init__(self, game):
        super().__init__(game)
        self._game = game
        self._alpha = game.alpha
        self._current_player = 0  
        self._actions = []        
        self._is_terminal = False
        self._returns = [0, 0]
    
    def current_player(self):
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        else:
            return self._current_player
    
    def _legal_actions(self, player):
        if self._is_terminal:
            return []
        return [0, 1, 2]
    
    def _apply_action(self, action):
        if self._is_terminal:
            raise ValueError("Cannot apply action to terminal state")
        
        self._actions.append(action)
        
        if len(self._actions) == 2:  
            self._resolve_game()
        else:
            self._current_player = 1  
    
    def _resolve_game(self):
        scissors_parameter = self._alpha
        p1_move, p2_move = self._actions
        
        if p1_move == p2_move == 1: 
            self._returns = [1, -1]
        elif p1_move == p2_move:
            # Tie
            self._returns = [0, 0]
        elif (p1_move == 0 and p2_move == 2) or \
             (p1_move == 1 and p2_move == 0) or \
             (p1_move == 2 and p2_move == 1):
            multiplier = scissors_parameter if 2 in [p1_move, p2_move] else 1
            self._returns = [multiplier, -multiplier]
        else:
            multiplier = scissors_parameter if 2 in [p1_move, p2_move] else 1
            self._returns = [-multiplier, multiplier]
        
        self._is_terminal = True
    
    def returns(self):
        return self._returns
    
    def is_terminal(self):
        return self._is_terminal
    
    def information_state_string(self, player=0):
        player = self.current_player()
        legal_actions = self._legal_actions(player)
        legal_actions_mask = [1 if i in legal_actions else 0 for i in range(3)]
        player_to_move = 1 if self._current_player == player else 0
        return f"Player:{player},ToMove:{player_to_move},Legal:{legal_actions}"
    
    def information_state_tensor(self, player):
        legal_actions = self._legal_actions(player)
        legal_actions_mask = [1.0 if i in legal_actions else 0.0 for i in range(3)]
        # [is_current_player, rock_legal, paper_legal, scissors_legal]
        is_current_player = 1.0 if self.current_player() == player else 0.0
        tensor = [is_current_player] + legal_actions_mask
        return np.array(tensor, dtype=np.float32)
    
    def observation_string(self, player):
        return self.information_state_string(player)
    
    def observation_tensor(self, player):
        return self.information_state_tensor(player)
    
    def __str__(self):
        if self.is_terminal():
            p1_action = self._game.action_to_string(0, self._actions[0])
            p2_action = self._game.action_to_string(1, self._actions[1])
            return f"Terminal: P1={p1_action}, P2={p2_action}, Returns={self.returns()}"
        elif len(self._actions) == 0:
            return f"Current: Player 0 to move"
        else:
            p1_action = self._game.action_to_string(0, self._actions[0])
            return f"Current: P1={p1_action}, Player 1 to move"

pyspiel.register_game(_GAME_TYPE, ExtendedRPSGame)