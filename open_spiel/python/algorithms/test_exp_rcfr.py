from open_spiel.python.algorithms import exploitability
from open_spiel.python.examples import gto_rps_variant
from open_spiel.python.games import rps_variant
from open_spiel.python.policy import policy as policy_lib
import numpy as np

p1_strategy = [0.4898,0.42857,0.08163]
p2_strategy = [0.36735,0.42857,0.20408]


class StaticPolicy(policy_lib.Policy):
    def __init__(self, game, strategies):
        """
        Args:
            game: The game
            strategies: List of strategies, one per player
        """
        super().__init__(game, list(range(game.num_players())))
        self.strategies = strategies
    
    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
        
        legal_actions = state.legal_actions()
        strategy = self.strategies[player_id]
        
        # Return dict mapping actions to probabilities
        return {action: strategy[i] for i, action in enumerate(legal_actions)}

# Create the game
game = rps_variant.ExtendedRPSGame(alpha=3.0)

# Create the policy
static_policy = StaticPolicy(game, [p1_strategy, p2_strategy])

# Compute Nash convergence
nash_conv_value = exploitability.nash_conv(game, static_policy)
print(f"Nash Convergence of GTO: {nash_conv_value}")
