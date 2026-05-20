import numpy as np
import pyspiel
import time
from open_spiel.python.examples import rwywe3
from open_spiel.python.algorithms import outcome_sampling_mccfr

res1 = [0,0]
board_size = 5
kuhn = pyspiel.load_game('kuhn_poker')
leduc = pyspiel.load_game('leduc_poker')
# dice = pyspiel.load_game('liars_dice')
ttt  = pyspiel.load_game('tic_tac_toe')
# c4 = pyspiel.load_game('connect_four')
# hex = pyspiel.load_game('hex')
iterations = 5000

# Initialize RWYWE agents
print("Initializing RWYWE agents...")
rwywek   = rwywe3.RWYWEAgent(kuhn, 0)
rwyweled = rwywe3.RWYWEAgent(leduc, 0)
# rwywet = rwywe3.RWYWEAgent(ttt,  0)
# rwywec = rwywe3.RWYWEAgent(c4,   0)
# rwyweh = rwywe3.RWYWEAgent(hex,  0)

# Initialize OPPS bots
print("Initializing OPPS bots...")
# oppsk = opps.OPPSBot(game=kuhn, player_id=0, depth=10)
# oppsd = opps.OPPSBot(game=dice, player_id=0, depth=10)
# oppst = opps.OPPSBot(game=ttt, player_id=0, depth=10)
# oppsc = opps.OPPSBot(game=c4, player_id=0, depth=10)
# oppsh = opps.OPPSBot(game=hex, player_id=0, depth=10)

# Initialize maxn bots
# print("Initializing maxn bots...")
# class MaxnBot:
#     def __init__(self, game, player_id, depth=10):
#         self.game = game
#         self.player_id = player_id
#         self.depth = depth
    
#     def step(self, state):
#         if state.is_terminal():
#             return None
#         if state.is_chance_node():
#             outcomes = state.chance_outcomes()
#             return outcomes[np.random.randint(len(outcomes))][0]
#         try:
#             values, best_action = maxn.maxn_search(self.game, state=state, depth_limit=self.depth)
#             return best_action if best_action is not None else state.legal_actions()[0]
#         except:
#             return state.legal_actions()[0]

# maxnk = MaxnBot(kuhn, 0)
# maxnd = MaxnBot(dice, 0)
# maxnt = MaxnBot(ttt, 0)
# maxnc = MaxnBot(c4, 0)
# maxnh = MaxnBot(hex, 0)

# Train MCCFR solvers
print("Training MCCFR solvers...")
mctsk_100 = outcome_sampling_mccfr.OutcomeSamplingSolver(kuhn)
mctsk_500 = outcome_sampling_mccfr.OutcomeSamplingSolver(kuhn)
mctsk_2000 = outcome_sampling_mccfr.OutcomeSamplingSolver(kuhn)
mctsk_5000 = outcome_sampling_mccfr.OutcomeSamplingSolver(kuhn)
mctsd_100  = outcome_sampling_mccfr.OutcomeSamplingSolver(leduc)
mctsd_500  = outcome_sampling_mccfr.OutcomeSamplingSolver(leduc)
mctsd_2000 = outcome_sampling_mccfr.OutcomeSamplingSolver(leduc)
mctsd_5000 = outcome_sampling_mccfr.OutcomeSamplingSolver(leduc)

print("Training MCCFR(100)...")
for _ in range(100):
    mctsk_100.iteration()
    mctsd_100.iteration()

print("Training MCCFR(500)...")
for _ in range(500):
    mctsk_500.iteration()
    mctsd_500.iteration()

print("Training MCCFR(2000)...")
for _ in range(2000):
    mctsk_2000.iteration()
    mctsd_2000.iteration()

print("Training MCCFR(5000)...")
for _ in range(5000):
    mctsk_5000.iteration()
    mctsd_5000.iteration()

print("Training complete.\n")

# Test function
def test_agent_vs_mccfr(game, agent, mccfr_policy, iterations):
    returns = [0, 0]
    for i in range(iterations):
        if iterations==50 or (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{iterations} games...")
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                a, _ = state.chance_outcomes()[np.random.randint(len(state.chance_outcomes()))]
                state.apply_action(a)
            else:
                pid = state.current_player()
                if pid == 0:
                    # Agent plays
                    action = agent.step(state)
                else:
                    # MCCFR plays
                    dist = mccfr_policy.action_probabilities(state)
                    legal = state.legal_actions()
                    probs = np.array([dist.get(a, 0) for a in legal])
                    if probs.sum() > 0:
                        probs /= probs.sum()
                    else:
                        probs = np.ones(len(legal)) / len(legal)
                    action = np.random.choice(legal, p=probs)
                state.apply_action(action)
        returns[0] += state.returns()[0]
        returns[1] += state.returns()[1]
    return returns[0]/iterations, returns[1]/iterations

# Run tests
games = [
    ('Kuhn Poker', kuhn, rwywek,  mctsk_100, mctsk_500, mctsk_2000, mctsk_5000),
    ("Leduc Poker", leduc, rwyweled, mctsd_100, mctsd_500, mctsd_2000, mctsd_5000),
]

opponent_configs = [
    ('MCCFR(100)', 0),
    ('MCCFR(500)', 1),
    ('MCCFR(2000)', 2),
    ('MCCFR(5000)', 3)
]

for opp_name, opp_idx in opponent_configs:
    print("\n" + "=" * 80)
    print(f"AVERAGE RETURN VS {opp_name} - Performance Against Opponent")
    print("=" * 80)
    print(f"{'Game':<15} {'RWYWEp2':<12} {'OPPS':<12} {'maxn':<12}")
    print("-" * 80)
    
    for game_name, game, rwywe_agent, mccfr_100, mccfr_500, mccfr_2000, mccfr_5000 in games:
        print(f"\nTesting {game_name}...")
        
        # Select the opponent
        opponent_solvers = [mccfr_100, mccfr_500, mccfr_2000, mccfr_5000]
        opponent_policy = opponent_solvers[opp_idx].average_policy()
        
        print("  Testing RWYWEp2...")
        if game_name=='Hex': iterations=50
        rwywe_ret, _ = test_agent_vs_mccfr(game, rwywe_agent, opponent_policy, iterations)
        iterations=5
        
        
        print(f"\n{game_name:<15} {rwywe_ret:<12.6f} {opponent_configs[opp_idx]}")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)