import numpy as np
import pyspiel
from open_spiel.python.examples import rwywe
from open_spiel.python.algorithms import random_agent, maxn


# game = pyspiel.load_game("kuhn_poker")       # or "kuhn_poker", "go(9)", etc.

def random_eval(state, player):
    return 0.0

# values, best_action = maxn.maxn_search(
#     game,
#     state=None,              # None means “start from initial state”
#     value_function=random_eval,
#     depth_limit=3            # tune depth to your liking
# )

# print("Estimated returns per player:", values)
# print("Best root action for player 0:", best_action)

res = [0,0]
for k in range(10000):
    game = pyspiel.load_game("kuhn_poker")
    # bot1 = rwywe.RWYWEAgent(game, 1)
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            a, _ = state.chance_outcomes()[np.random.randint(len(state.chance_outcomes()))]
            state.apply_action(a)
        else:
            # legal_actions = state.legal_actions()
            # action = np.random.choice(legal_actions)
            # state.apply_action(action)
            pid = state.current_player()
            if pid==k%2:
                legal_actions = state.legal_actions()
                action = np.random.choice(legal_actions)
                state.apply_action(action)
            else:
                # a = bot1.step(state)
                # state.apply_action(a)
                values, best_action = maxn.maxn_search(
                    game,
                    state=state,              
                    value_function=random_eval,
                    depth_limit=3            
                )
                state.apply_action(best_action)
    res = [res[0] + state.returns()[k%2], res[1] + state.returns()[1-k%2]]
print("Agent:", res[1]/10000, "Random", res[0]/10000)
