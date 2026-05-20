import numpy as np
import pyspiel
import time
from open_spiel.python.algorithms import lazy_cfr as cfr
from open_spiel.python.algorithms import expected_game_score

res1 = [0,0]
board_size = 5
game_name = 'connect_four'
game = pyspiel.load_game(game_name)
cum_regrets = []
iterations = 100

# Train CFR solver first
cfr_solver = cfr.CFRSolver(game)
print("Training CFR solver...")
for i in range(300):
    cfr_solver.evaluate_and_update_policy()
    if (i + 1) % 50 == 0:
        print(f"Completed {i + 1} iterations")

average_policy = cfr_solver.average_policy()
average_policy_values = expected_game_score.policy_value(
    game.new_initial_state(), [average_policy] * 2)
print(f"Average policy values: {average_policy_values}")

# Create a bot that uses the CFR policy
class CFRPolicyBot:
    def __init__(self, game, policy):
        self.game = game
        self.policy = policy
        
    def step(self, state):
        # Get current information state
        info_state = state.information_state_string(state.current_player())
        
        # Get action probabilities from the policy
        action_probs = self.policy.policy_for_key(info_state)
        
        # Choose action with highest probability
        legal_actions = state.legal_actions()
        
        # Filter probabilities for legal actions only
        legal_probs = []
        for action in legal_actions:
            if action < len(action_probs):
                legal_probs.append(action_probs[action])
            else:
                legal_probs.append(0.0)
        
        # Normalize probabilities
        total_prob = sum(legal_probs)
        if total_prob > 0:
            legal_probs = [p / total_prob for p in legal_probs]
        else:
            # If all probabilities are zero, use uniform distribution
            legal_probs = [1.0 / len(legal_actions)] * len(legal_actions)
        
        # Choose action based on probabilities
        chosen_action = np.random.choice(legal_actions, p=legal_probs)
        return chosen_action

# Create the CFR bot
cfr_bot = CFRPolicyBot(game, average_policy)

start = time.perf_counter()
for k in range(iterations):
    game_instance = pyspiel.load_game(game_name)
    state = game_instance.new_initial_state()

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np.random.choice(actions, p=probs)
            state.apply_action(action)
        else:
            pid = state.current_player()
            legal_actions = state.legal_actions()
            
            if pid == 1:
                # Player 2 uses random policy
                action = legal_actions[np.random.randint(len(legal_actions))]
                state.apply_action(action)        
            else:
                # Player 1 uses CFR policy
                best_action = cfr_bot.step(state)
                state.apply_action(best_action)
    
    returns = state.returns()
    res1 = [res1[0] + returns[0], res1[1] + returns[1]]
    
    if (k + 1) % 10 == 0:
        print(f"Completed {k + 1} games")

print("AgentP1:", res1[0]/iterations, "RandomP2:", res1[1]/iterations)
end = time.perf_counter()
print((end - start) * 1000 / iterations, 'milliseconds per iteration on average')