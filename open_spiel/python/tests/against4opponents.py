"""
rwywe_test_suite.py
--------------------
Tests RWYWEAgent from rwywe3.py over 1000 hands against four opponent types,
repeated 10 times with results averaged across runs.

Opponents:
  1. Fully uniform random
  2. Static near-GTO (±0.2 of GTO, fixed mixed strategy chosen at outset)
  3. Uniform random for first 100 hands, then exact best response
  4. Full GTO
"""

import sys, os, random
import numpy as np
import pyspiel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rwywe_exploitabilityvsk import (
    agent_v_star,
    enumerate_opponent_pure_strategies,
    exact_agent_ev,
    get_agent_played_policy,
    compute_strategy_exploitability,
    compute_step_info,
    gto_action_probs,
)

NUM_HANDS = 1000
NUM_RUNS  = 10


# ---------------------------------------------------------------------------
# Opponent factories — rng created internally, no external rng args
# ---------------------------------------------------------------------------

def make_opp1(seed):
    """
    Fully uniform random — samples a fixed mixed strategy at outset by
    drawing action probabilities uniformly at random at each ISS, then
    plays that same fixed strategy for the entire match.
    """
    rng = random.Random(seed)

    # Build the fixed strategy lazily on first call to act(), since we
    # need a game state to know the legal actions at each ISS.
    # We'll populate it on the first hand via a flag.
    strategy = {}

    def act(state, hand_num):
        iss   = state.information_state_string(state.current_player())
        if iss not in strategy:
            legal = state.legal_actions()
            # Draw weights uniformly at random and normalise
            raw   = np.array([rng.random() for _ in legal])
            raw  /= raw.sum()
            strategy[iss] = list(zip(legal, raw.tolist()))
        acts, wts = zip(*strategy[iss])
        return rng.choices(acts, weights=wts)[0]

    return act


def make_opp2(game, opp_player, seed, noise=0.2):
    """
    Static near-GTO: sample a fixed mixed strategy at outset (±noise of GTO),
    play it for the entire match.
    """
    rng = random.Random(seed)

    strategy = {}
    def walk(s):
        if s.is_terminal(): return
        if s.current_player() == pyspiel.PlayerId.CHANCE:
            for a, _ in s.chance_outcomes():
                s2 = s.clone(); s2.apply_action(a); walk(s2)
        elif s.current_player() == opp_player:
            iss = s.information_state_string(opp_player)
            if iss not in strategy:
                gto   = gto_action_probs(s, opp_player, alpha=0.0)
                legal = s.legal_actions()
                raw   = np.array([
                    max(0.0, gto.get(a, 0.0) + rng.uniform(-noise, noise))
                    for a in legal
                ])
                total = raw.sum()
                raw   = raw / total if total > 0 else np.ones(len(legal)) / len(legal)
                strategy[iss] = list(zip(legal, raw.tolist()))
            for a in s.legal_actions():
                s2 = s.clone(); s2.apply_action(a); walk(s2)
        else:
            for a in s.legal_actions():
                s2 = s.clone(); s2.apply_action(a); walk(s2)
    walk(game.new_initial_state())

    def act(state, hand_num):
        iss       = state.information_state_string(opp_player)
        acts, wts = zip(*strategy[iss])
        return rng.choices(acts, weights=wts)[0]
    return act


def make_opp3(game, agent, opp_player, seed):
    """
    Uniform random for hands 1-100, then exact best response recomputed
    every hand from 101 onwards against agent's current policy.
    Works with any agent that has either get_policy() or compute_step_info
    compatibility (rwywe3-style with solver).
    """
    rng         = random.Random(seed)
    pure_strats, iss_list = enumerate_opponent_pure_strategies(game, opp_player)

    def probe_policy():
        """Build full agent policy by probing each ISS via getdist() or solver."""
        agent_policy = {}
        def walk(s):
            if s.is_terminal(): return
            if s.current_player() == pyspiel.PlayerId.CHANCE:
                for a, _ in s.chance_outcomes():
                    s2 = s.clone(); s2.apply_action(a); walk(s2)
            elif s.current_player() == agent.player_id:
                iss = s.information_state_string(agent.player_id)
                if iss not in agent_policy:
                    legal = s.legal_actions(agent.player_id)
                    if hasattr(agent, 'getdist'):
                        dist = agent.getdist(s.clone()) or {}
                    else:
                        avg  = agent.solver.average_policy()
                        dist = avg.action_probabilities(s, agent.player_id)
                    norm = sum(dist.get(a, 0.0) for a in legal) or 1.0
                    agent_policy[iss] = {a: dist.get(a, 0.0) / norm for a in legal}
                for a in s.legal_actions(agent.player_id):
                    s2 = s.clone(); s2.apply_action(a); walk(s2)
            else:
                for a in s.legal_actions():
                    s2 = s.clone(); s2.apply_action(a); walk(s2)
        walk(game.new_initial_state())
        return agent_policy

    def compute_br():
        if hasattr(agent, 'get_policy'):
            agent_policy = agent.get_policy()
        else:
            agent_policy = probe_policy()
        ev_func = exact_agent_ev(agent, game, agent_policy)
        return min(pure_strats, key=ev_func)

    def act(state, hand_num):
        if hand_num <= 100:
            return rng.choice(state.legal_actions())
        br  = compute_br()
        iss = state.information_state_string(opp_player)
        return br.get(iss, state.legal_actions()[0])
    return act


def make_opp4(opp_player, seed):
    """Full GTO (alpha=0)."""
    rng = random.Random(seed)
    def act(state, hand_num):
        probs = gto_action_probs(state, opp_player, alpha=0.0)
        acts  = list(probs.keys())
        wts   = [probs[a] for a in acts]
        return rng.choices(acts, weights=wts)[0]
    return act


# ---------------------------------------------------------------------------
# Agent step adaptors
# ---------------------------------------------------------------------------

def make_agent_step_exact(game):
    """
    Adaptor for RWYWEExactAgent.
    agent.step() returns an int action directly.
    agent.get_policy() returns the full ISS->dist policy dict for this hand.
    """
    def step(agent, state):
        action = agent.step(state)
        policy = agent.get_policy()   # already computed by step()
        return action, policy
    return step


def make_agent_step_rwywe(game):
    """
    Adaptor for rwywe3-style agents which expose getdist() for the policy.
    """
    def step(agent, state):
        action = agent.step(state)

        legal = state.legal_actions(agent.player_id)
        try:
            action = int(action)
            if action not in legal:
                raise ValueError
        except (TypeError, ValueError):
            action = legal[0]

        # rwywe3 exposes getdist() which returns {action: prob} at this state
        dist = agent.getdist(state) or {}
        norm = sum(dist.get(a, 0.0) for a in legal) or 1.0
        iss  = state.information_state_string(agent.player_id)

        agent_policy = {}
        def walk(s):
            if s.is_terminal(): return
            if s.current_player() == pyspiel.PlayerId.CHANCE:
                for a, _ in s.chance_outcomes():
                    s2 = s.clone(); s2.apply_action(a); walk(s2)
            elif s.current_player() == agent.player_id:
                cur_iss = s.information_state_string(agent.player_id)
                if cur_iss not in agent_policy:
                    lgl = s.legal_actions(agent.player_id)
                    if cur_iss == iss:
                        agent_policy[cur_iss] = {
                            a: dist.get(a, 0.0) / norm for a in lgl
                        }
                    else:
                        avg = agent.solver.average_policy()
                        d   = avg.action_probabilities(s, agent.player_id)
                        t   = sum(d.get(a, 0.0) for a in lgl) or 1.0
                        agent_policy[cur_iss] = {a: d.get(a, 0.0) / t for a in lgl}
                for a in s.legal_actions(agent.player_id):
                    s2 = s.clone(); s2.apply_action(a); walk(s2)
            else:
                for a in s.legal_actions():
                    s2 = s.clone(); s2.apply_action(a); walk(s2)
        walk(game.new_initial_state())
        return action, agent_policy
    return step


# ---------------------------------------------------------------------------
# Single match runner
# ---------------------------------------------------------------------------

def run_match(game, agent, opp_player, pure_strats, iss_list,
              get_opp_action, num_hands, agent_step_fn, verbose=True):
    """
    Run one match. agent_step_fn(agent, state) -> (action, played_policy).
    played_policy is a dict: agent ISS -> {action: prob}.
    """
    rng        = random.Random()
    v_star     = agent_v_star(agent.player_id)
    cumulative = 0.0
    violations = 0
    k_history, expl_history, payoff_history = [], [], []

    if verbose:
        hdr = (
            f"{'Hand':>5} | {'k_before':>9} | "
            f"{'expl':>8} | {'<=k':>6} | {'payoff':>7} | {'cum_pay':>9}"
        )
        print(hdr)
        print("-" * len(hdr))

    for hand_num in range(1, num_hands + 1):
        state    = game.new_initial_state()
        k_before = agent.k

        played_policy = None

        while not state.is_terminal():
            cur = state.current_player()
            if cur == pyspiel.PlayerId.CHANCE:
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(outcomes, weights=probs)[0])
            elif cur == agent.player_id:
                action, played_policy = agent_step_fn(agent, state)
                state.apply_action(action)
            else:
                opp_action = get_opp_action(state, hand_num)
                agent.inform_action(state, cur, opp_action)
                state.apply_action(opp_action)

        agent.on_terminal(state)

        hand_payoff  = state.returns()[agent.player_id]
        cumulative  += hand_payoff

        expl, _, _, _, _ = compute_strategy_exploitability(
            agent, game, played_policy or {}
        )

        safe_ok = expl <= k_before + 1e-9 or k_before < 0
        if not safe_ok:
            violations += 1

        if verbose:
            flag = "OK" if safe_ok else "VIOL"
            print(
                f"{hand_num:>5} | {k_before:>9.4f} | "
                f"{expl:>8.5f} | {flag:>6} | "
                f"{hand_payoff:>7.3f} | {cumulative:>9.3f}"
            )

        k_history.append(agent.k)
        expl_history.append(expl)
        payoff_history.append(hand_payoff)

    return {
        "mean_payoff":    cumulative / num_hands,
        "final_k":        agent.k,
        "max_expl":       max(expl_history),
        "min_k":          min(k_history),
        "max_k":          max(k_history),
        "violations":     violations,
        "k_history":      k_history,
        "expl_history":   expl_history,
        "payoff_history": payoff_history,
    }


# ---------------------------------------------------------------------------
# One full suite (all 4 opponents, one agent type, one seed)
# ---------------------------------------------------------------------------

def run_suite_once(game, opp_player, pure_strats, iss_list,
                   agent_factory, agent_step_fn,
                   seed, verbose=False):
    v_star = agent_v_star(0)
    results = []

    for opp_idx, (label, opp_factory) in enumerate([
        ("Opponent 1: Uniform random",              lambda s: make_opp1(s)),
        ("Opponent 2: Static near-GTO (±0.2)",      lambda s: make_opp2(game, opp_player, s)),
        ("Opponent 3: Uniform (1-100) → best resp", None),   # needs agent ref, handled below
        ("Opponent 4: Full GTO",                    lambda s: make_opp4(opp_player, s)),
    ]):
        agent = agent_factory()
        opp_seed = seed * 10 + opp_idx + 1

        if opp_idx == 2:   # opp3 needs the agent reference
            opp_fn = make_opp3(game, agent, opp_player, opp_seed)
        else:
            opp_fn = opp_factory(opp_seed)

        r = run_match(game, agent, opp_player, pure_strats, iss_list,
                      opp_fn, NUM_HANDS, agent_step_fn, verbose=verbose)
        r["label"] = label
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Full multi-run averaged suite — runs both agent versions
# ---------------------------------------------------------------------------

def run_test_suite(num_runs=NUM_RUNS, base_seed=None):
    if base_seed is None:
        base_seed = random.randint(0, 1_000_000)
    print(f"Base seed: {base_seed}  |  {num_runs} runs × {NUM_HANDS} hands\n")

    game        = pyspiel.load_game("kuhn_poker")
    opp_player  = 1
    pure_strats, iss_list = enumerate_opponent_pure_strategies(game, opp_player)
    v_star      = agent_v_star(0)

    # Import exact RWYWE agent
    from open_spiel.python.examples.rwywe3 import RWYWEAgent

    def agent_factory():
        return RWYWEAgent(game, player_id=0)

    agent_step_fn = make_agent_step_rwywe(game)

    runs = []
    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        print(f"Run {run_idx + 1}/{num_runs}  (seed {seed})")
        suite = run_suite_once(
            game, opp_player, pure_strats, iss_list,
            agent_factory,
            agent_step_fn,
            seed=seed, verbose=False,
        )
        for r in suite:
            print(f"  {r['label']:<44}  "
                  f"pay={r['mean_payoff']:+.4f}  "
                  f"k_max={r['max_k']:.3f}  "
                  f"viol={r['violations']}")
        runs.append(suite)
        print()

    # -----------------------------------------------------------------------
    # Averaged summary — one block per agent
    # -----------------------------------------------------------------------
    w = 78
    print(f"\n{'='*w}")
    print(f"  RESULTS  ({num_runs} runs × {NUM_HANDS} hands each)  —  rwywe3, player 0")
    print(f"  v* = {v_star:+.6f}")
    print(f"{'='*w}")
    print(f"  {'Opponent':<44} {'Mean pay':>9} {'± std':>7} "
          f"{'vs v*=−1/18':>11} {'k_max':>7} {'Viol/run':>9}")
    print(f"  {'-'*44} {'-'*9} {'-'*7} {'-'*11} {'-'*7} {'-'*9}")

    num_opps = len(runs[0])
    for opp_idx in range(num_opps):
        runs_for_opp = [runs[r][opp_idx] for r in range(num_runs)]
        label     = runs_for_opp[0]["label"]
        pays      = [r["mean_payoff"] for r in runs_for_opp]
        k_maxes   = [r["max_k"]       for r in runs_for_opp]
        viols     = [r["violations"]  for r in runs_for_opp]
        mean_pay  = np.mean(pays)
        std_pay   = np.std(pays)
        above_v   = mean_pay - v_star   # how much above the GTO floor
        mean_kmax = np.mean(k_maxes)
        mean_viol = np.mean(viols)
        # Flag if result is suspiciously far from v* for GTO opponent
        flag = "  ← ~correct" if "GTO" in label and abs(mean_pay - v_star) < 0.01 else ""
        print(f"  {label:<44} {mean_pay:>+9.5f} {std_pay:>7.5f} "
              f"{above_v:>+11.5f} {mean_kmax:>7.3f} {mean_viol:>9.2f}{flag}")

    print()
    return runs


def _make_agent(cls, game, player_id, v_star):
    agent = cls(game, player_id=player_id)
    agent.v_star = v_star
    return agent


if __name__ == "__main__":
    run_test_suite(num_runs=NUM_RUNS, base_seed=random.randint(0, 1_000_000))