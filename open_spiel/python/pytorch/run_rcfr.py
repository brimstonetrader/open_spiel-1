# run_rcfr.py
import pyspiel
import torch
import matplotlib.pyplot as plt
import numpy as np
from open_spiel.python.pytorch import rcfr, rcfr_with_alpha
from open_spiel.python.algorithms import coxucb, coxucb2, coxucb3
from open_spiel.python.algorithms import exploitability
from open_spiel.python.examples import gto_rps_variant
from open_spiel.python.games import rps_variant
from open_spiel.python import policy as policy_lib


i = 3.14 #parameter from 2-4

row_payoffs_variant = [
    [0, -1, 1*i],
    [1, -1, -1*i],
    [-1*i, 1*i, 0*i]
]

col_payoffs_variant = [
    [0, 1, -1*i],
    [-1, 1, 1*i],
    [1*i, -1*i, 0*i]
]

alpha = np.random.random() * 2 + 2

def train_fn(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = model(x_batch)           
        loss = torch.nn.functional.mse_loss(preds, y_batch) 
        loss.backward()
        optimizer.step()


def _new_model():
  return rcfr.DeepRcfrModel(
      rps_variant.ExtendedRPSGame(alpha),
      num_hidden_layers=1,
      num_hidden_units=14,
      num_hidden_factors=2,
      use_skip_connections=True)


game = rps_variant.ExtendedRPSGame(alpha)
res  = [[0], [0], [0]]
solver1 = rcfr.RcfrSolver(game, [_new_model(), _new_model()])
solver2 = rcfr.RcfrSolver(game, [_new_model(), _new_model()])
gto1 = gto_rps_variant.GTO_RPS(game, 0, game.alpha)
gto2 = gto_rps_variant.GTO_RPS(game, 1, game.alpha)
equilibs = []

wrapper1 = solver1._root_wrapper
wrapper2 = solver2._root_wrapper

num_players = 2
dummy = [solver1._cumulative_seq_probs[0], solver2._cumulative_seq_probs[1]]

def best_response(i, player, policy, j): 
    row_payoffs_variant = [[0, -1, 1*i], [1, -1, -1*i], [-1*i, 1*i, 0*i]]
    col_payoffs_variant = [[0, 1, -1*i], [-1, 1, 1*i], [1*i, -1*i, 0*i]]
    game = row_payoffs_variant if player == 0 else col_payoffs_variant
    mx = 0,-10000
    for i in range(3):
        ex = 0
        for j in range(3):
            ex += policy[i] * game[i][j]
        if ex > mx[1]:
            mx = i,ex
    return mx[0]


def wrapped(s):
    dummy = [solver1._cumulative_seq_probs[0], solver2._cumulative_seq_probs[1]]
    raw_fn = wrapper1.sequence_weights_to_policy_fn(dummy)
    probs = raw_fn(s)
    legal_actions = [0,1,2]
    return {action: probs[i] for i, action in enumerate(legal_actions)}


game = rps_variant.ExtendedRPSGame(alpha=3.0)
moves = [0, 1, 2]  # Rock, Paper, Scissors

print("=== ACTUAL GAME PAYOFFS ===")
for p1_move in moves:
    for p2_move in moves:
        state = game.new_initial_state()
        state.apply_action(p1_move)
        state.apply_action(p2_move)
        returns = state.returns()
        p1_action = game.action_to_string(0, p1_move)
        p2_action = game.action_to_string(1, p2_move)
        print(f"P1 {p1_action} vs P2 {p2_action}: {returns}")

iterations = 1000
exs = []

for i in range(iterations):
    gto1.i = alpha
    gto2.i = alpha
    rs = [2 + i*0.025 for i in range(80)]
    xs = []
    rst = []
    pst = []
    sst = []
    np.random.shuffle(rs)
    alpha = 3
    solver1.evaluate_and_update_policy(train_fn)
    solver2.evaluate_and_update_policy(train_fn)
   
    game = rps_variant.ExtendedRPSGame(alpha)
    state = game.new_initial_state()
    pf0 = [0,0,0]
    pf1 = [0,0,0]
    
    while not state.is_terminal():
        if state.is_chance_node():
            a, _ = state.chance_outcomes()[np.random.randint(len(state.chance_outcomes()))]
            state.apply_action(a)
        else:
            pid = state.current_player()
            dummy = [solver1._cumulative_seq_probs[0], solver2._cumulative_seq_probs[1]]
            policy_fn = wrapper1.sequence_weights_to_policy_fn(dummy)                    
            pf = policy_fn(state)
            
            # Store policies for both players
            if pid == 0:
                pf0 = pf
            else:
                pf1 = pf
            
            if i % 1000 == 0 and i > 0 and state.current_player() == 1: 
                # Get both player strategies
                state_p0 = game.new_initial_state()
                state_p1 = game.new_initial_state()
                state_p1.apply_action(0)  # Advance to player 1's turn
                
                dummy = [solver1._cumulative_seq_probs[0], solver2._cumulative_seq_probs[1]]
                policy_fn = wrapper1.sequence_weights_to_policy_fn(dummy)
                
                p0_strategy = policy_fn(state_p0)
                p1_strategy = policy_fn(state_p1)
                
                # Create policy for exploitability calculation
                class LearnedPolicy(policy_lib.Policy):
                    def __init__(self, game, solver1, solver2, wrapper):
                        super().__init__(game, list(range(game.num_players())))
                        self.solver1 = solver1
                        self.solver2 = solver2
                        self.wrapper = wrapper
                    
                    def action_probabilities(self, state, player_id=None):
                        if player_id is None:
                            player_id = state.current_player()
                        
                        dummy = [self.solver1._cumulative_seq_probs[0], 
                                self.solver2._cumulative_seq_probs[1]]
                        policy_fn = self.wrapper.sequence_weights_to_policy_fn(dummy)
                        probs = policy_fn(state)
                        
                        # Convert array to dictionary
                        legal_actions = state.legal_actions()
                        return {action: probs[i] for i, action in enumerate(legal_actions)}
                
                learned_policy = LearnedPolicy(game, solver1, solver2, wrapper1)
                nash_conv = exploitability.nash_conv(game, learned_policy)
                
                print(f'\nIteration {i}')
                print(f'Player 0 strategy: {p0_strategy}')
                print(f'Player 1 strategy: {p1_strategy}')
                print(f'Nash Convergence: {nash_conv}')
                
                exs.append(nash_conv)
                
                # Compare to GTO
                p1_strategy_gto = [0.4898,0.42857,0.08163]
                p2_strategy_gto = [0.36735, 0.42857, 0.20408]
                
                class StaticPolicy(policy_lib.Policy):
                    def __init__(self, game, strategies):
                        super().__init__(game, list(range(game.num_players())))
                        self.strategies = strategies
                    
                    def action_probabilities(self, state, player_id=None):
                        if player_id is None:
                            player_id = state.current_player()
                        
                        legal_actions = state.legal_actions()
                        strategy = self.strategies[player_id]
                        
                        return {action: strategy[i] for i, action in enumerate(legal_actions)}

                static_policy = StaticPolicy(game, [p1_strategy_gto, p2_strategy_gto])
                nash_conv_value = exploitability.nash_conv(game, static_policy)
                # print(f"Nash Convergence of GTO: {nash_conv_value}\n")
            
            # Sample action from learned policy
            action = np.random.random()
            action_idx = 0 if action < pf[0] else (1 if action < pf[0] + pf[1] else 2)
            state.apply_action(action_idx)
            
        if i > 0 and i % (iterations//80) == 0: 
            xs.append(alpha)
            rst.append(pf0[0])
            pst.append(pf0[1])
            sst.append(pf0[2])
        if i < 500:
            res[0].append(res[0][-1] + state.returns()[0])
            res[1].append(res[1][-1] + state.returns()[1])


print("\n=== Final Results ===")
print(f"Solver 1 sequence weights: {solver1._sequence_weights()}")
print(f"Solver 2 sequence weights: {solver2._sequence_weights()}")

x = list(range(len(exs)))
print(f"\nExploitability over time: {exs}")
print(f"Average return player 0: {res[0][-1] / len(res[0])}")
print(f"Average return player 1: {res[1][-1] / len(res[1])}")

plt.figure()
plt.plot(x, exs, marker='o', label='Exploitability of RCFR')
plt.xlabel('Iteration (x1000)')
plt.ylabel('Nash Convergence')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()