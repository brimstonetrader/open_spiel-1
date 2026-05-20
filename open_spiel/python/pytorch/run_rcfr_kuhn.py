# run_rcfr.py
import pyspiel
import torch
import matplotlib.pyplot as plt
import numpy as np
from open_spiel.python.pytorch import rcfr, rcfr_with_alpha
from open_spiel.python.algorithms import coxucb, coxucb2, coxucb3
from open_spiel.python.algorithms import exploitability
from open_spiel.python.examples import gto_rps_variant, gto_kuhn_poker
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

game = pyspiel.load_game('kuhn_poker')

# uniform_policy = policy_lib.UniformRandomPolicy(game)
# uniform_nash_conv = exploitability.nash_conv(game, uniform_policy)
# print(f"Uniform random NashConv: {uniform_nash_conv}")

agent = gto_kuhn_poker.GTOKuhnPoker(game, player_id=1)

for i in range(50):
    value = exploitability.nash_conv(game, gto_kuhn_poker.GTOKuhnPolicy(game, i/99))
    print('alpha = ', i, '/99. exploitability = ', value)


# game = pyspiel.load_game("kuhn_poker")

# Replace this with the policy object you passed previously:
# pi = gto_kuhn_poker.GTOKuhnPolicy(game, agent)   # or GTOKuhnPolicy(game, agent) if that's your ctor

# # 1) Print action probabilities for every infoset
# def dump_policy(policy):
#     print("INFOSTATE -> {action:prob}")
#     # iterate all possible states by traversing the game tree
#     stack = [game.new_initial_state()]
#     seen = []
#     while stack:
#         s = stack.pop()
#         key = (s.history(), s.current_player())
#         if key in seen: continue
#         seen.append(key)
#         if s.is_terminal(): continue
#         if s.is_chance_node():
#             for a, _ in s.chance_outcomes():
#                 stack.append(policy_lib.child(s, a))
#             continue
#         probs = policy.action_probabilities(s)
#         # show only infosets (states with a player)
#         if not s.is_chance_node():
#             info = s.information_state_string(s.current_player())
#             print(info, "->", probs, "legal:", s.legal_actions())
#         for a in s.legal_actions():
#             stack.append(policy_lib.child(s, a))

# dump_policy(pi)

# # 2) Compare to OpenSpiel canonical kuhn equilibrium
# kuhn_equil = data.kuhn_nash_equilibrium(alpha=0.2)  # exact equilibrium used in tests
# print("\nDifferences vs canonical kuhn equilibrium (non-zero entries):")
# for player in range(game.num_players()):
#     stack = [game.new_initial_state()]
#     seen = set()
#     while stack:
#         s = stack.pop()
#         if s.is_terminal(): continue
#         if s.is_chance_node():
#             for a, _ in s.chance_outcomes():
#                 stack.append(policy_lib.child(s, a))
#             continue
#         p = s.current_player()
#         info = s.information_state_string(p)
#         p_my = pi.action_probabilities(s)
#         p_ref = kuhn_equil.action_probabilities(s)
#         # compare only legal actions
#         for a in s.legal_actions():
#             myv = float(p_my.get(a, 0.0))
#             refv = float(p_ref.get(a, 0.0))
#             if abs(myv - refv) > 1e-9:
#                 print(f"{info} (player{p}) action {a}: mine={myv:.6f} ref={refv:.6f}")
#         for a in s.legal_actions():
#             stack.append(policy_lib.child(s, a))

# 3) Print per-player on-policy and best-response values
# conv = exploitability.nash_conv(game, pi, return_only_nash_conv=False, use_cpp_br=False)
# print("\nNashConv:", conv.nash_conv)
# print("Player improvements (best_response - on_policy):", conv.player_improvements)
# # on-policy values:
# root = game.new_initial_state()
# # use exploitability.best_response to get on-policy and BR specifics
# from open_spiel.python.algorithms import exploitability as ex
# for pid in range(game.num_players()):
#     br_info = ex.best_response(game, pi, player_id=pid)
#     print(f"\nPlayer {pid} on_policy_value = {br_info['on_policy_value']}")
#     print(f"Player {pid} best_response_value = {br_info['best_response_value']}")
#     print("Best response actions (infosets):", br_info["best_response_action"])


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
    pyspiel.load_game("kuhn_poker"),
      num_hidden_layers=1,
      num_hidden_units=14,
      num_hidden_factors=2,
      use_skip_connections=True)

game = pyspiel.load_game("kuhn_poker")
res  = [[0], [0], [0]]
solver = rcfr.RcfrSolver(game, [_new_model(), _new_model()])
gto2 = gto_kuhn_poker.GTOKuhnPoker(game, 1)
transformer_player_id = 0
equilibs = []

wrapper = solver._root_wrapper
player_idx = transformer_player_id

num_players = 2#solver._game.num_players()
dummy = [None]*num_players
dummy[player_idx] = solver._cumulative_seq_probs[player_idx]

policy_fn = wrapper.sequence_weights_to_policy_fn(dummy)

for state in rcfr.all_states(solver._game.new_initial_state(),
                            depth_limit=-1,
                            include_terminals=False,
                            include_chance_states=False):
    if state.current_player() == player_idx:
        print(state.information_state_string(),
            policy_fn(state))
print(wrapper.sequence_features)

print(wrapper._walk_descendants(game.new_initial_state()))
# print(solver.alpha)
print() ; print() ; print()

def best_response(i, player, policy, j): 
    row_payoffs_variant = [[0, -1, 1*i], [1, -1, -1*i], [-1*i, 1*i, 0*i]]
    col_payoffs_variant = [[0, 1, -1*i], [-1, 1, 1*i], [1*i, -1*i, 0*i]]
    game = row_payoffs_variant if player == 0 else col_payoffs_variant
    mx = 0,-10000
    equilibs.append((alpha, pf))
    for i in range(3):
        ex = 0
        for j in range(3):
            ex += policy[i] * game[i][j]
        if ex > mx[1]:
            mx = i,ex
    # return gto2.step(state)
    return mx[0]



def wrapped(s):
    probs = raw_fn(s)
    legal_actions = [0,1,2]
    return {action: probs[i] for i, action in enumerate(legal_actions)}

iterations = 100000
for i in range(iterations):
    solver.evaluate_and_update_policy(train_fn)
    while not state.is_terminal():
        if state.is_chance_node():
            a, _ = state.chance_outcomes()[np.random.randint(len(state.chance_outcomes()))]
            state.apply_action(a)
        else:
            pid = state.current_player()
            dummy = [None]*2
            for p in range(2): dummy[p] = solver._cumulative_seq_probs[p]
            raw_fn = wrapper.sequence_weights_to_policy_fn(dummy)
            policy_fn = solver._root_wrapper.sequence_weights_to_policy_fn(dummy)                    
            pf = policy_fn(state)
            if i%50==0: print(i)
            if i%1000==0 and state.current_player()==transformer_player_id: 
                raw_fn = wrapper.sequence_weights_to_policy_fn(dummy)
                policy_f = policy_lib.tabular_policy_from_callable(game, wrapped)
                print('[', i, ',', exploitability.nash_conv(game, policy_f), ',', wrapped(state), ']')
                current_policy = solver.current_policy()
                nash_conv = exploitability.nash_conv(game, current_policy)
                print(f'[{i}, {nash_conv}]')


            if pid==transformer_player_id:
                action = np.random.random()
                state.apply_action(1 if pf[0] < action else 2 if pf[0]+pf[1] < action else 0)
            else:                   
                state.apply_action(gto2.step(state))
        if i<500:
            res[0].append(res[0][-1] + state.returns()[transformer_player_id])
            res[1].append(res[1][-1] + state.returns()[1-transformer_player_id])
            # res[2].append(solver.alpha)
    # if i%100==0:
    #     wrapper = solver._root_wrapper
    #     num_players = solver._game.num_players()
    #     dummy = [None]*num_players
    #     dummy[player_idx] = solver._cumulative_seq_probs[player_idx]

    #     # build a policy‐function for only that player
    #     policy_fn = wrapper.sequence_weights_to_policy_fn(dummy)
    
    #     for state in rcfr.all_states(solver._game.new_initial_state(),
    #                                 depth_limit=-1,
    #                                 include_terminals=False,
    #                                 include_chance_states=False):
    #         if state.current_player() == player_idx:
    #             print(state.information_state_string(),
    #                 policy_fn(state))
    #     print(wrapper.sequence_features)
    #     # print(solver.alpha)
    #     print() ; print() ; print()

    #     avg_policy = solver.average_policy()
    #     avg_policy = sorted(avg_policy.items(), key=lambda x : x[0])
    #     # nash_conv = exploitability.nash_conv(game, avg_policy)  # :contentReference[oaicite:0]{index=0}
    #     # print(f"Iter {i:4d} → NashConv (exploitability):")
        # for p in avg_policy: print(p)



    #     avg_policy = solver.average_policy()
    #     avg_policy = sorted(avg_policy.items(), key=lambda x : x[0])
    #     # nash_conv = exploitability.nash_conv(game, avg_policy)  # :contentReference[oaicite:0]{index=0}
    #     # print(f"Iter {i:4d} → NashConv (exploitability):")
    #     # for p in avg_policy: print(p)

# print("Transformer:", res[0]/100, "Random", res[1]/100)

print(solver._sequence_weights())

x = list(range(len(exs)))
print(exs)
print(res[0][-1] / len(res[0]))
# print(equilibs)
# for e in equilibs: print(e)
plt.figure()
plt.plot(x, exs, marker='o', label='Exploitability of RCFR')
# plt.plot(x, res[0], marker='o', label='RCFR With Alpha')
# plt.plot(x, res[2], marker='o', label='Alpha')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 