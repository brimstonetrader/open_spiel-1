# rwywe_goofspiel_test.py
#
# Tests RWYWE against a Best Response policy on Goofspiel.
#
# Why Goofspiel:
#   - Pure card-bidding game: each player holds identical hands and bids on
#     prize cards revealed one at a time. Highest bid wins the prize.
#   - Payoffs accumulate across all tricks (not just -1/0/+1 at terminal),
#     giving RWYWE's safety budget k real signal to learn from.
#   - Sequential, zero-sum, stochastic (prize card order is random),
#     2-player — ideal for RWYWE's design assumptions.
#   - num_cards=4 keeps the game tree small so MCCFR converges quickly.
#
# Three design choices vs the old TTT test:
#   1. v_star is initialized from MCCFR's expected value, not hardcoded 0.
#   2. RWYWE plays ONLINE during games (no tabularization) so it keeps adapting.
#   3. Exploitability is measured pre- and post-warmup via a tabular snapshot.
#
# Usage:
#   python rwywe_goofspiel_test.py

import numpy as np
import pyspiel

from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import best_response
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import outcome_sampling_mccfr
from open_spiel.python.examples import rwywe3

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
NUM_CARDS          = 4        # goofspiel hand size — 4 is compact but meaningful
MCCFR_WARMUP_ITERS = 20_000  # iterations for the MCCFR warmup opponent
RWYWE_WARMUP_EPS   = 20_000  # warmup episodes RWYWE plays before the match
NUM_GAMES          = 10       # live games after warmup
ONLINE_SAMPLES     = 50       # step() calls per move when playing online


# ------------------------------------------------------------------ #
# MCCFR helpers
# ------------------------------------------------------------------ #
def build_mccfr_tabular(game, iterations):
    print(f"  Training MCCFR({iterations:,})...", flush=True)
    solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
    for _ in range(iterations):
        solver.iteration()
    return solver.average_policy().to_tabular()


def mccfr_expected_value(game, tabular, player_id):
    """Compute expected value of a tabular policy for player_id via self-play."""
    policies = [tabular, tabular]
    return expected_game_score.policy_value(
        game.new_initial_state(), policies)[player_id]


# ------------------------------------------------------------------ #
# RWYWE warmup
# ------------------------------------------------------------------ #
def warmup_rwywe(game, agent, mccfr_policy, n_episodes):
    print(f"  Warming up RWYWE p{agent.player_id} "
          f"({n_episodes:,} episodes vs MCCFR)...", flush=True)
    report_every = max(1, n_episodes // 5)

    for ep in range(n_episodes):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, weights = zip(*outcomes)
                action = np.random.choice(actions, p=np.array(weights) / sum(weights))
                state.apply_action(action)
                continue

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

        if (ep + 1) % report_every == 0:
            print(f"    ep {ep+1:>6,} / {n_episodes:,}  k = {agent.k:.6f}", flush=True)

    print(f"  Warmup done. Final k = {agent.k:.6f}")


# ------------------------------------------------------------------ #
# Tabularize RWYWE — snapshot for exploitability measurement only
# ------------------------------------------------------------------ #
def tabularize_rwywe(game, agent):
    tabular  = policy_lib.TabularPolicy(game)
    visited  = set()

    def _walk(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                _walk(state.child(a))
            return

        pid = state.current_player()
        if pid == agent.player_id:
            info_str = state.information_state_string(pid)
            if info_str not in visited:
                visited.add(info_str)
                idx = tabular.state_lookup.get(info_str)
                if idx is not None:
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
    return tabular


# ------------------------------------------------------------------ #
# Exploitability
# ------------------------------------------------------------------ #
def measure_exploitability(game, tabular):
    """
    Average best-response gain across both players.
    Nash equilibrium = 0. Random policy = ~max_utility.
    """
    total = 0.0
    for br_player in range(game.num_players()):
        br = best_response.BestResponsePolicy(game, br_player, tabular)
        policies = [None, None]
        policies[br_player]     = br
        policies[1 - br_player] = tabular
        val = expected_game_score.policy_value(
            game.new_initial_state(), policies)[br_player]
        total += val
    return total / game.num_players()


# ------------------------------------------------------------------ #
# Online RWYWE action — agent keeps learning, no frozen tabular lookup
# ------------------------------------------------------------------ #
def rwywe_online_action(agent, state, num_samples=ONLINE_SAMPLES):
    legal  = state.legal_actions()
    counts = {a: 0 for a in legal}
    for _ in range(num_samples):
        a = agent.step(state)
        if a in counts:
            counts[a] += 1
    total = sum(counts.values())
    probs = np.array([counts[a] / total for a in legal])
    return np.random.choice(legal, p=probs)


# ------------------------------------------------------------------ #
# Pretty-print game state
# ------------------------------------------------------------------ #
def print_state(state, game_name="goofspiel"):
    print(f"\n  {str(state).strip()}\n")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    print("=" * 60)
    print(f"RWYWE vs Best Response — Goofspiel (num_cards={NUM_CARDS})")
    print("=" * 60)

    # Goofspiel is simultaneous-move; wrap it so MCCFR and BestResponsePolicy
    # (which both require sequential dynamics) can work on it.
    # OpenSpiel accepts nested game specs as a string parameter.
    game = pyspiel.load_game(
        f"turn_based_simultaneous_game(game=goofspiel(num_cards={NUM_CARDS},points_order=random))"
    )

    # ---- 1. Build MCCFR warmup opponent ----
    print("\n[Setup] Building MCCFR warmup policy...")
    warmup_pol = build_mccfr_tabular(game, MCCFR_WARMUP_ITERS)

    # ---- 2. Compute v_star from MCCFR self-play ----
    # This is the baseline expected value for player 0 under MCCFR.
    # Used to initialize RWYWE's safety budget correctly instead of 0.
    v_star = mccfr_expected_value(game, warmup_pol, player_id=0)
    print(f"\n[Setup] MCCFR baseline value (v_star) for player 0: {v_star:.6f}")
    print(f"  (Goofspiel is symmetric so this should be near 0.0)")

    # ---- 3. Warm up RWYWE ----
    print("\n[Setup] Warming up RWYWE (player 0)...")
    rwywe_agent = rwywe3.RWYWEAgent(game, player_id=0)
    rwywe_agent.v_star = v_star   # fix the hardcoded 0.0
    warmup_rwywe(game, rwywe_agent, warmup_pol, RWYWE_WARMUP_EPS)

    # ---- 4. Measure exploitability post-warmup ----
    print("\n[Setup] Measuring exploitability post-warmup...")
    snapshot_pre = tabularize_rwywe(game, rwywe_agent)
    expl_pre     = measure_exploitability(game, snapshot_pre)
    mccfr_expl   = measure_exploitability(game, warmup_pol)
    print(f"  MCCFR({MCCFR_WARMUP_ITERS:,}) exploitability : {mccfr_expl:.6f}")
    print(f"  RWYWE post-warmup exploitability : {expl_pre:.6f}")

    # ---- 5. Build Best Response (player 1) against the RWYWE snapshot ----
    print("\n[Setup] Computing Best Response (player 1) vs RWYWE snapshot...")
    br_policy = best_response.BestResponsePolicy(game, 1, snapshot_pre)
    print("[Setup] Ready.\n")

    # ---- 6. Play live games ----
    record  = {"RWYWE (p0)": 0, "BR (p1)": 0, "Draw": 0}
    returns_log = []

    for game_num in range(1, NUM_GAMES + 1):
        print("=" * 60)
        print(f"  GAME {game_num} of {NUM_GAMES}  [RWYWE adapting online]")
        print("=" * 60)

        state    = game.new_initial_state()
        move_num = 0

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, weights = zip(*outcomes)
                action = np.random.choice(actions, p=np.array(weights) / sum(weights))
                state.apply_action(action)
                print(f"  [Chance] Prize card revealed")
                print_state(state)
                continue

            pid      = state.current_player()
            move_num += 1

            if pid == 0:
                action = rwywe_online_action(rwywe_agent, state)
                print(f"  Move {move_num}: RWYWE (p0) plays action {action}")
            else:
                dist  = br_policy.action_probabilities(state)
                legal = state.legal_actions()
                probs = np.array([dist.get(a, 0.0) for a in legal])
                probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(legal)) / len(legal)
                action = np.random.choice(legal, p=probs)
                print(f"  Move {move_num}: BR    (p1) plays action {action}")

            state.apply_action(action)
            print_state(state)

        ret = state.returns()
        returns_log.append(ret[0])

        if ret[0] > ret[1]:
            outcome = "RWYWE (p0) wins!"
            record["RWYWE (p0)"] += 1
        elif ret[1] > ret[0]:
            outcome = "BR (p1) wins!"
            record["BR (p1)"] += 1
        else:
            outcome = "Draw!"
            record["Draw"] += 1

        print(f"  >> {outcome}  returns: p0={ret[0]:+.2f}  p1={ret[1]:+.2f}")
        print(f"     Running k = {rwywe_agent.k:.6f}\n")

    # ---- 7. Measure exploitability post-games ----
    print("[Post-game] Re-measuring exploitability...")
    snapshot_post = tabularize_rwywe(game, rwywe_agent)
    expl_post     = measure_exploitability(game, snapshot_post)
    delta         = expl_post - expl_pre
    trend         = "▼ improved" if delta < -0.001 else ("▲ worsened" if delta > 0.001 else "→ unchanged")

    # ---- 8. Summary ----
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for label, count in record.items():
        pct = 100.0 * count / NUM_GAMES
        bar = "█" * count
        print(f"  {label:<15}: {count:>2}/{NUM_GAMES}  ({pct:4.0f}%)  {bar}")

    avg_return = np.mean(returns_log)
    print(f"\n  RWYWE avg return per game : {avg_return:+.4f}")
    print(f"  v_star (MCCFR baseline)   : {v_star:+.6f}")
    print(f"  Final k                   : {rwywe_agent.k:.6f}")
    print()
    print(f"  Exploitability — MCCFR({MCCFR_WARMUP_ITERS:,})  : {mccfr_expl:.6f}")
    print(f"  Exploitability — RWYWE pre-games  : {expl_pre:.6f}")
    print(f"  Exploitability — RWYWE post-games : {expl_post:.6f}  ({trend})")
    print(f"  Nash target                       : 0.000000")
    print("=" * 60)


if __name__ == "__main__":
    main()