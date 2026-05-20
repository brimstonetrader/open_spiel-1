import numpy as np
import pyspiel
from open_spiel.python.examples import gto_kuhn_poker

res1 = [0,0]
res2 = [0,0]
game = pyspiel.load_game("kuhn_poker")
cum_regrets = []
iterations = 100000
gto1 = gto_kuhn_poker.GTOKuhnPoker(game, 0)
gto2 = gto_kuhn_poker.GTOKuhnPoker(game, 1)
for k in range(iterations):
    game1 = pyspiel.load_game("kuhn_poker")
    game2 = pyspiel.load_game("kuhn_poker")
    state1 = game1.new_initial_state()
    state2 = game2.new_initial_state()
    while not state1.is_terminal():
        if state1.is_chance_node():
            a, _ = state1.chance_outcomes()[np.random.randint(len(state1.chance_outcomes()))]
            state1.apply_action(a)
        else:
            pid = state1.current_player()
            if pid==1:
                legal_actions = state1.legal_actions()
                action = np.random.choice(legal_actions)
                state1.apply_action(action)
            else:
                best_action = gto1.step(state1)
                state1.apply_action(best_action)    
    while not state2.is_terminal():
        if state2.is_chance_node():
            a, _ = state2.chance_outcomes()[np.random.randint(len(state2.chance_outcomes()))]
            state2.apply_action(a)
        else:
            pid = state2.current_player()
            if pid==0:
                legal_actions = state2.legal_actions()
                action = np.random.choice(legal_actions)
                state2.apply_action(action)
            else:
                best_action = gto2.step(state=state2)
                state2.apply_action(best_action)

    res1 = [res1[0] + state1.returns()[0], res1[1] + state1.returns()[1]]
    res2 = [res2[0] + state2.returns()[0], res2[1] + state2.returns()[1]]
print("AgentP1:", res1[0]/iterations, "RandomP2:", res1[1]/iterations)
print("AgentP2:", res2[1]/iterations, "RandomP1:", res2[0]/iterations)