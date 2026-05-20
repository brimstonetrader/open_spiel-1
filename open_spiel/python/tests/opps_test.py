import numpy as np
import pyspiel
import time
from open_spiel.python.algorithms import opps, dynamic_opps, dynamic_opps2, maxn, brsplus

def std_value_function(state, p, c=100):
        returns = 0 
        for _ in range(c):
            clone = state.clone()
            while not clone.is_terminal():
                la = clone.legal_actions()
                clone.apply_action(la[np.random.randint(len(la))])
            returns += clone.returns()[p]
        return returns / c

res = [0,0]
agent_pid = 3
num_players = 8
game_name = 'kuhn_poker'
cum_regrets = []
iterations = 5000
start = time.perf_counter()
for k in range(iterations):
    # if k%10==0: 
    # print(k)
    agent_pid = (agent_pid+1) % num_players
    game = pyspiel.load_game(game_name + "(players=" + str(num_players) + ')')
    state = game.new_initial_state()
    opps_bot = opps.OPPSBot(game, agent_pid)
    while not state.is_terminal():
        if state.is_chance_node():
            a, _ = state.chance_outcomes()[np.random.randint(len(state.chance_outcomes()))]
            state.apply_action(a)
        else:
            pid = state.current_player()
            if pid!=agent_pid:
                legal_actions = state.legal_actions()
                action = np.random.choice(legal_actions)
                state.apply_action(action)
            else:
                best_action = opps_bot.step(state)
                state.apply_action(best_action)    
                # values, best_action = maxn.maxn_search(
                #     game,
                #     state=state,              
                #     value_function=std_value_function,
                #     depth_limit=3            
                # )
    # print(state.returns())
    res[0] += state.returns()[agent_pid]
    res[1] += (sum(state.returns()) - state.returns()[agent_pid]) / (num_players - 1)
print('Agent: ', res[0] / iterations)
print('Random_Avg: ', res[1] / iterations)
end = time.perf_counter()
print((end - start) * 1000 / iterations, 'milliseconds per iteration on average')