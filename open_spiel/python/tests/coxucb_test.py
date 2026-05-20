import numpy as np
import pyspiel
import matplotlib.pyplot as plt
from open_spiel.python.algorithms import coxucb, coxucb2, coxucb3
from open_spiel.python.examples import oscillator
from open_spiel.python.pytorch import rcfr


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
      pyspiel.load_game('kuhn_poker'),
      num_hidden_layers=1,
      num_hidden_units=13,
      num_hidden_factors=2,
      use_skip_connections=True)



res1 = [0,0]
res2 = [0,0]
res3 = [0,0]
game = pyspiel.load_game("kuhn_poker")
cum_regrets = []
cum_returns1 = [0]
cum_returns2 = [0]
cum_returns3 = [0]
iterations = 10000
coxucb0   = coxucb.CoxUCBBot(game , 0)
coxucb0_2 = coxucb2.CoxUCBBot(game, 0)
coxucb0_3 = coxucb3.CoxUCBBot(game, 0)
oscillator1 = oscillator.Oscillator(game, 1)
oscillator2 = oscillator.Oscillator(game, 1)
oscillator3 = oscillator.Oscillator(game, 1)
solver1 = rcfr.RcfrSolver(game, [_new_model(), _new_model()])
solver2 = rcfr.RcfrSolver(game, [_new_model(), _new_model()])
solver3 = rcfr.RcfrSolver(game, [_new_model(), _new_model()])
wrapper1 = solver1._root_wrapper
wrapper2 = solver2._root_wrapper
wrapper3 = solver3._root_wrapper




for k in range(iterations):
    game1 = pyspiel.load_game("kuhn_poker")
    game2 = pyspiel.load_game("kuhn_poker")
    game3 = pyspiel.load_game("kuhn_poker")
    state1 = game1.new_initial_state()
    state2 = game2.new_initial_state()
    state3 = game3.new_initial_state()
    while not state1.is_terminal():
        if state1.is_chance_node():
            a, _ = state1.chance_outcomes()[np.random.randint(len(state1.chance_outcomes()))]
            state1.apply_action(a)
        else:
            pid = state1.current_player()
            if pid==1:
                dummy = [None]*2
                dummy[1] = solver1._cumulative_seq_probs[0]
                policy_fn = solver1._root_wrapper.sequence_weights_to_policy_fn(dummy)                    
                pf = policy_fn(state1)
                action = np.random.random()
                state1.apply_action(0 if pf[0] < action else 1)
            else:
                best_action = coxucb0.step(state1)
                state1.apply_action(best_action)
    while not state2.is_terminal():
        if state2.is_chance_node():
            a, _ = state2.chance_outcomes()[np.random.randint(len(state2.chance_outcomes()))]
            state2.apply_action(a)
        else:
            pid = state2.current_player()
            if pid==1:
                dummy = [None]*2
                dummy[1] = solver2._cumulative_seq_probs[0]
                policy_fn = solver2._root_wrapper.sequence_weights_to_policy_fn(dummy)                    
                pf = policy_fn(state2)
                action = np.random.random()
                state2.apply_action(0 if pf[0] < action else 1)
            else:
                best_action = coxucb0_2.step(state2)
                state2.apply_action(best_action)
    while not state3.is_terminal():
        if state3.is_chance_node():
            a, _ = state3.chance_outcomes()[np.random.randint(len(state3.chance_outcomes()))]
            state3.apply_action(a)
        else:
            pid = state3.current_player()
            if pid==1:
                dummy = [None]*2
                dummy[1] = solver3._cumulative_seq_probs[0]
                policy_fn = solver3._root_wrapper.sequence_weights_to_policy_fn(dummy)                    
                pf = policy_fn(state3)
                action = np.random.random()
                state3.apply_action(0 if pf[0] < action else 1)
            else:
                best_action = coxucb0_3.step(state3)
                state3.apply_action(best_action)



    iss_agent = state1.information_state_string(0)
    iss_opp = state2.information_state_string(1)
    best_value = 0 if iss_opp[0] > iss_agent[1] else 2
    regret = best_value
    cum_regrets.append(regret + (0 if cum_regrets==[] else cum_regrets[-1]))    
    cum_returns1.append(state1.returns()[0] + cum_returns1[-1])     
    cum_returns2.append(state2.returns()[0] + cum_returns2[-1])     
    cum_returns3.append(state3.returns()[0] + cum_returns3[-1])     
    res1 = [res1[0] + state1.returns()[0], res1[1] + state1.returns()[1]]
    res2 = [res2[0] + state2.returns()[0], res2[1] + state2.returns()[1]]
    
x = list(range(-9+iterations//100))
plt.figure()
plt.plot(x, cum_returns1[1000::100], marker='o', label='COX-UCB')
plt.plot(x, cum_returns2[1000::100], marker='o', label='COX-UCB with sliding window')
plt.plot(x, cum_returns3[1000::100], marker='o', label='COX-UCB with exponential decay')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 
print("AgentP1:", res1[0]/iterations, "RandomP2:", res1[1]/iterations)
print('P1Regret:', cum_regrets[-1] / ((iterations * np.log(iterations))**0.5))
print("AgentP2:", res2[1]/iterations, "RandomP1:", res2[0]/iterations)


# import numpy as np
# import pyspiel
# from open_spiel.python.algorithms import coxucb
# from open_spiel.python.examples import gto_kuhn_poker

# # game = pyspiel.load_game("kuhn_poker")     

# def random_eval(state, player):
#     return 0.0

# # values, best_action = maxn.maxn_search(
# #     game,
# #     state=None,              # None means “start from initial state”
# #     value_function=random_eval,
# #     depth_limit=3            # tune depth to your liking
# # )

# # print("Estimated returns per player:", values)
# # print("Best root action for player 0:", best_action)

# res = [0,0]
# game = pyspiel.load_game("kuhn_poker")
# agent_player_id = 0
# cum_regrets = []
# cox_ucb = coxucb.CoxUCBBot(game, agent_player_id)
# for k in range(10000):
#     game = pyspiel.load_game("kuhn_poker")
#     state = game.new_initial_state()
#     while not state.is_terminal():
#         if state.is_chance_node():
#             a, _ = state.chance_outcomes()[np.random.randint(len(state.chance_outcomes()))]
#             state.apply_action(a)
#         else:
#             pid = state.current_player()
#             if pid==0:
#                 legal_actions = state.legal_actions()
#                 action = np.random.choice(legal_actions)
#                 state.apply_action(action)
#             else:
#                 # iss = state.information_state_string(cox_ucb.player_id)
#                 # print(iss)
#                 # la = state.legal_actions()
#                 # pv = cox_ucb.compute_payoff_vector(state)
#                 # cr = cox_ucb.compute_confidence_region(iss, la)
#                 # ucs = cox_ucb.construct_utility_constrained_set(iss, la, pv)
#                 # strat = cox_ucb.select_strategy(iss, la, pv)
#                 best_action = cox_ucb.step(state=state)
#                 state.apply_action(best_action)
#                 # print(state.legal_actions(), state.information_state_string(cox_ucb.player_id))
#     iss_agent = state.information_state_string(agent_player_id)
#     iss_opp = state.information_state_string(1 - agent_player_id)
#     best_value = 0 if iss_opp[0] > iss_agent[1] else 2
#     regret = best_value

#     res = [res[0] + state.returns()[agent_player_id], res[1] + state.returns()[1-agent_player_id]]
# print("Agent:", res[0]/10000, "Random", res[1]/10000)
# # print(cox_ucb.counts)
# # print(cox_ucb.visits)
# # print('confidence region: ', cr)
# # print('payoff vector: ', pv)
# # print('utility constrained set', ucs)
# # print('strategy', strat)