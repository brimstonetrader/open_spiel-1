# rwywe_exp_test.py
#
# Compares exploitability of four strategies on Goofspiel(4):
#   - Random
#   - MCCFR (1000 iterations)
#   - MCCFR (5000 iterations)
#   - RWYWE
#
# Fixes vs previous version:
#   - mccfr_policies lookup keys now match the training iteration counts
#   - warmup_rwywe no longer rebuilds MCCFR on every call — uses passed-in policy
#   - goofspiel uses num_cards=4 (default 13 is far too large for tabularization)

import numpy as np
import pyspiel

from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import best_response
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import outcome_sampling_mccfr
from open_spiel.python.examples import rwywe3

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
NUM_SAMPLES     = 100    # samples of agent.step() to build RWYWE distribution
WARMUP_EPISODES = 1000  # episodes RWYWE plays vs MCCFR before tabularizing

GAMES = [
    ("Goofspiel5", pyspiel.load_game(
        "turn_based_simultaneous_game(game=goofspiel(num_cards=5,points_order=random))")),
]

MCCFR_ITERS = [1000, 5000]  # single source of truth for iteration counts


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def rwywe_action_probs(agent, state, num_samples=NUM_SAMPLES):
    legal  = state.legal_actions()
    counts = {a: 0 for a in legal}
    for _ in range(num_samples):
        a = agent.step(state)
        if a in counts:
            counts[a] += 1
    total = sum(counts.values())
    if total == 0:
        u = 1.0 / len(legal)
        return {a: u for a in legal}
    return {a: c / total for a, c in counts.items()}


def warmup_rwywe(game, agent, mccfr_policy, n_episodes):
    """Run n_episodes of self-play vs the provided mccfr_policy."""
    for ep in range(n_episodes):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes      = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                probs          = np.array(probs, dtype=np.float64)
                a              = np.random.choice(actions, p=probs / probs.sum())
                state.apply_action(a)
            else:
                pid = state.current_player()
                if pid == agent.player_id:
                    action = agent.step(state)
                else:
                    dist  = mccfr_policy.action_probabilities(state)
                    legal = state.legal_actions()
                    probs = np.array([dist.get(a, 0.0) for a in legal])
                    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(legal)) / len(legal)
                    action = np.random.choice(legal, p=probs)
                    agent.inform_action(state, pid, action)
                state.apply_action(action)
        agent.on_terminal(state)


def build_rwywe_tabular(game, player_id, warmup_policy):
    """Warm up a RWYWE agent, then walk the game tree to tabularize its policy.

    Returns (tabular, owned_indices) where owned_indices is the set of row
    indices that belong to player_id, so the merge step never lets one player
    overwrite the other's rows.
    """
    agent = rwywe3.RWYWEAgent(game, player_id)
    print(f"    Warming up p{player_id} ({WARMUP_EPISODES} episodes vs MCCFR)...")
    warmup_rwywe(game, agent, warmup_policy, WARMUP_EPISODES)
    print(f"    k after warmup: {agent.k:.6f}")

    tabular       = policy_lib.TabularPolicy(game)
    visited       = set()
    owned_indices = set()

    def _walk(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                _walk(state.child(a))
            return
        if state.current_player() == player_id:
            info_str = state.information_state_string(player_id)
            if info_str not in visited:
                visited.add(info_str)
                idx = tabular.state_lookup.get(info_str)
                if idx is not None:
                    owned_indices.add(idx)
                    probs = agent.getdist(state)
                    mask  = tabular.legal_actions_mask[idx]
                    legal = [a for a, m in enumerate(mask) if m]
                    total = sum(probs.get(a, 0.0) for a in legal)
                    if total > 0:
                        for i, a in enumerate(legal):
                            tabular.action_probability_array[idx][i] = (
                                probs.get(a, 0.0) / total)
        for a in state.legal_actions():
            _walk(state.child(a))

    _walk(game.new_initial_state())
    return tabular, owned_indices


def build_mccfr_tabular(game, iterations):
    """Train OutcomeSamplingSolver and return its average TabularPolicy."""
    solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
    for _ in range(iterations):
        solver.iteration()
    return solver.average_policy().to_tabular()


def build_random_tabular(game):
    """Uniform-random policy (TabularPolicy default)."""
    return policy_lib.TabularPolicy(game)


def eval_value(game, br_policy, opp_policy, br_player):
    policies = [None, None]
    policies[br_player]     = br_policy
    policies[1 - br_player] = opp_policy
    return expected_game_score.policy_value(
        game.new_initial_state(), policies)[br_player]


def compute_exploitability(game, agent_name, opp_tabular):
    br_values = {}
    for br_player in range(game.num_players()):
        br_value  = best_response.BestResponsePolicy(game, br_player, opp_tabular)
        final_val = eval_value(game, br_value, opp_tabular, br_player)
        br_values[br_player] = final_val
        print(f"      BR value p{br_player}: {final_val:.6f}")
    return sum(br_values.values()) / 2.0


# --------------------------------------------------------------------------- #
# Pre-train MCCFR policies once per game
# --------------------------------------------------------------------------- #

print("=" * 70)
print("Pre-training MCCFR policies...")
print("=" * 70)

mccfr_policies = {}
for game_name, game in GAMES:
    for iters in MCCFR_ITERS:
        print(f"  MCCFR({iters}) on {game_name}...")
        mccfr_policies[(game_name, iters)] = build_mccfr_tabular(game, iters)

print()


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #

results = {}

for game_name, game in GAMES:
    print(f"\n{'=' * 70}")
    print(f"Game: {game_name}")
    print(f"{'=' * 70}")

    agents = {
        "Random":               build_random_tabular(game),
        f"MCCFR({MCCFR_ITERS[0]})": mccfr_policies[(game_name, MCCFR_ITERS[0])],
        f"MCCFR({MCCFR_ITERS[1]})": mccfr_policies[(game_name, MCCFR_ITERS[1])],
    }

    # Build RWYWE: warm up one agent per player vs the stronger MCCFR policy,
    # tabularize, then merge using owned_indices so players don't overwrite each other.
    print(f"\n  Building RWYWE tabular policy...")
    warmup_pol          = mccfr_policies[(game_name, MCCFR_ITERS[1])]
    rwywe_p0, owned_p0  = build_rwywe_tabular(game, 0, warmup_pol)
    rwywe_p1, owned_p1  = build_rwywe_tabular(game, 1, warmup_pol)
    rwywe_joint         = policy_lib.TabularPolicy(game)

    print("  Merging RWYWE player policies...")
    print(f"    pre-merge row sum min: {min(rwywe_joint.action_probability_array.sum(axis=1)):.4f}")
    print(f"    pre-merge row sum max: {max(rwywe_joint.action_probability_array.sum(axis=1)):.4f}")

    for tab, owned in [(rwywe_p0, owned_p0), (rwywe_p1, owned_p1)]:
        for idx in owned:
            rwywe_joint.action_probability_array[idx] = (
                tab.action_probability_array[idx].copy())

    agents["RWYWE"] = rwywe_joint

    for agent_name, opp_tabular in agents.items():
        print(f"\n  --- {agent_name} ---")
        expl = compute_exploitability(game, agent_name, opp_tabular)
        results[(game_name, agent_name)] = expl
        print(f"  Exploitability ({agent_name}): {expl:.6f}")


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #

agent_names = ["Random", f"MCCFR({MCCFR_ITERS[0]})", f"MCCFR({MCCFR_ITERS[1]})", "RWYWE"]

print("\n\n" + "=" * 70)
print("EXPLOITABILITY SUMMARY")
print("=" * 70)
print(f"{'Game':<15}", end="")
for a in agent_names:
    print(f"  {a:>12}", end="")
print()
print("-" * 70)

for game_name, _ in GAMES:
    print(f"{game_name:<15}", end="")
    for a in agent_names:
        val = results.get((game_name, a), float("nan"))
        print(f"  {val:>12.6f}", end="")
    print()

print("=" * 70)
print("\nLower exploitability = closer to Nash equilibrium.")
print(f"Random >> MCCFR({MCCFR_ITERS[0]}) > MCCFR({MCCFR_ITERS[1]}) >> Nash (0.0) is expected ordering.")