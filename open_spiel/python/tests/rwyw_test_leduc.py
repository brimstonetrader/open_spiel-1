"""
test_rwyw_exploitability_leduc.py
----------------------------------
Shows that RWYW's expected exploitability shrinks toward zero as the number
of hands per match grows large — for Leduc poker.

For each round size T in [108, 1080, 1998, 5004, 9990]:
  - Run NUM_RUNS independent matches of T hands each
  - Each match: RWYW agent vs dynamic approximate best response
  - Record mean exploitability across all hands in the match
  - Average across NUM_RUNS matches

Exploitability is approximated via outcome-sampling MCCFR best response
(exact enumeration is infeasible for Leduc's large strategy space).

BR computation is skipped when k <= 0 (agent plays equilibrium).
"""

import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pyspiel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from open_spiel.python.algorithms import outcome_sampling_mccfr, exploitability as expl_lib
from open_spiel.python.examples.rwyw_leduc import (
    RWYWAgent, solve_equilibrium, _ev
)

ROUND_SIZES  = [108, 1080, 1998, 5004, 9990]
NUM_RUNS     = 5
BR_ITERS     = 2000    # MCCFR iterations to approximate best response
GAME_NAME    = "leduc_poker"


# ---------------------------------------------------------------------------
# Exploitability via tabular policy + OpenSpiel's exploitability module
# ---------------------------------------------------------------------------

def policy_exploitability(agent, game):
    """
    Approximate exploitability of the agent's current policy using
    OpenSpiel's exploitability module, which computes the exact best
    response value via a full tree traversal.

    Returns expl(π) = v* - min_{opp} EV(π, opp).
    """
    # Build a TabularPolicy-compatible dict for player 0
    from open_spiel.python.policy import TabularPolicy

    current = agent.get_policy()

    # Walk the tree and build a full joint policy (agent + equilibrium for opp)
    joint = {}
    def walk(s):
        if s.is_terminal(): return
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                s2 = s.clone(); s2.apply_action(a); walk(s2)
        else:
            pid  = s.current_player()
            iss  = s.information_state_string(pid)
            lgl  = s.legal_actions(pid)
            if pid == agent.player_id:
                dist = current.get(iss, {a: 1.0/len(lgl) for a in lgl})
            else:
                dist = agent._eq_opp_policy.get(iss, {a: 1.0/len(lgl) for a in lgl})
            norm = sum(dist.get(a, 0.0) for a in lgl) or 1.0
            joint[(pid, iss)] = {a: dist.get(a, 0.0)/norm for a in lgl}
            for a in lgl:
                s2 = s.clone(); s2.apply_action(a); walk(s2)
    walk(game.new_initial_state())

    # Use _ev with a best-response oracle to compute exploitability.
    # For Leduc, we compute BR by finding the opponent strategy that
    # minimises agent EV via tree traversal — exact but expensive.
    # We cache it since the agent's policy only changes once per hand.
    opp_br = _compute_br_policy(agent, game, current)
    min_ev = _ev(game, agent.player_id, current, opp_br)
    return agent.v_star - min_ev


def _compute_br_policy(agent, game, agent_policy):
    """
    Compute exact best response for the opponent against agent_policy
    by tree traversal — fills in best-responding action at each opp ISS.
    """
    opp_id  = agent.opp_id
    br_pol  = {}

    def best_response_value(s):
        """Returns (br_value, br_action_at_this_node) for opp nodes."""
        if s.is_terminal():
            return s.returns()[agent.player_id], None
        cur = s.current_player()
        if cur == pyspiel.PlayerId.CHANCE:
            val = 0.0
            for a, p in s.chance_outcomes():
                s2 = s.clone(); s2.apply_action(a)
                val += p * best_response_value(s2)[0]
            return val, None
        elif cur == agent.player_id:
            iss  = s.information_state_string(agent.player_id)
            dist = agent_policy.get(iss, {})
            lgl  = s.legal_actions(agent.player_id)
            norm = sum(dist.get(a, 0.0) for a in lgl) or 1.0
            val  = 0.0
            for a in lgl:
                s2 = s.clone(); s2.apply_action(a)
                val += (dist.get(a, 0.0)/norm) * best_response_value(s2)[0]
            return val, None
        else:
            # Opponent: pick action minimising agent EV
            lgl  = s.legal_actions(cur)
            best_a, best_v = None, float("inf")
            for a in lgl:
                s2 = s.clone(); s2.apply_action(a)
                v, _ = best_response_value(s2)
                if v < best_v:
                    best_v, best_a = v, a
            iss = s.information_state_string(cur)
            br_pol[iss] = best_a
            return best_v, best_a

    best_response_value(game.new_initial_state())
    return br_pol


def opp_best_response(agent, game):
    """
    When k <= 0 the agent plays equilibrium — return equilibrium for opp.
    When k > 0 compute exact BR.
    """
    if agent.k <= 0:
        return agent._eq_opp_policy
    return _compute_br_policy(agent, game, agent.get_policy())


def exploitability_approx(agent, game):
    """expl(π) = v* - EV(π, BR(π))"""
    br     = _compute_br_policy(agent, game, agent.get_policy())
    min_ev = _ev(game, agent.player_id, agent.get_policy(), br)
    return agent.v_star - min_ev


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

def play_hand(game, agent, opp_policy):
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            state.apply_action(random.choices(actions, weights=probs)[0])
        elif state.current_player() == agent.player_id:
            action = agent.step(state)
            if action is None:
                action = random.choice(state.legal_actions(agent.player_id))
            state.apply_action(action)
        else:
            iss    = state.information_state_string(state.current_player())
            lgl    = state.legal_actions()
            action = opp_policy.get(iss, lgl[0])
            if isinstance(action, dict):
                # Mixed policy
                norm = sum(action.get(a, 0.0) for a in lgl) or 1.0
                action = random.choices(lgl,
                    weights=[action.get(a, 0.0)/norm for a in lgl])[0]
            agent.inform_action(state, state.current_player(), action)
            state.apply_action(action)
    agent.on_terminal(state)
    return state.returns()[agent.player_id]


def run_match(game, shared_eq, num_hands, seed):
    random.seed(seed)
    np.random.seed(seed)
    agent   = RWYWAgent(game, player_id=0, _shared_eq=shared_eq)
    expls   = []
    payoffs = []
    for h in range(num_hands):
        br = opp_best_response(agent, game)
        if h % max(1, num_hands // 10) == 0:
            expls.append(exploitability_approx(agent, game))
        payoffs.append(play_hand(game, agent, br))
    # Fill remaining expl slots with last known value
    mean_expl = float(np.mean(expls)) if expls else 0.0
    return mean_expl, float(np.mean(payoffs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_suite(base_seed=None):
    if base_seed is None:
        base_seed = random.randint(0, 1_000_000)
    print(f"Base seed: {base_seed}  |  {NUM_RUNS} runs per round size")
    print(f"Round sizes: {ROUND_SIZES}\n")

    game      = pyspiel.load_game(GAME_NAME)
    shared_eq = solve_equilibrium(game, iterations=5000)
    v_star    = shared_eq[0]
    print(f"v* = {v_star:.6f}\n")

    mean_expls   = []
    std_expls    = []
    mean_payoffs = []
    std_payoffs  = []

    for T in ROUND_SIZES:
        run_expls   = []
        run_payoffs = []
        for run_idx in range(NUM_RUNS):
            seed   = base_seed + T + run_idx
            me, mp = run_match(game, shared_eq, T, seed)
            run_expls.append(me)
            run_payoffs.append(mp)
            print(f"  T={T:>5}  run {run_idx+1}/{NUM_RUNS}  "
                  f"mean_expl={me:.5f}  mean_pay={mp:+.5f}")

        mean_expls.append(float(np.mean(run_expls)))
        std_expls.append(float(np.std(run_expls)))
        mean_payoffs.append(float(np.mean(run_payoffs)))
        std_payoffs.append(float(np.std(run_payoffs)))
        print(f"  T={T:>5}  avg_expl={mean_expls[-1]:.5f}  "
              f"avg_pay={mean_payoffs[-1]:+.5f}\n")

    plot(ROUND_SIZES, mean_expls, std_expls,
         mean_payoffs, std_payoffs, v_star, base_seed)
    return ROUND_SIZES, mean_expls, std_expls, mean_payoffs, std_payoffs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(round_sizes, mean_expls, std_expls,
         mean_payoffs, std_payoffs, v_star, seed):
    BG, PANEL = "#ffffff", "#111827"
    GRID, MUTED, TEXT = "#1e293b", "#64748b", "#000000"
    GREEN, BLUE, RED = "#34d399", "#38bdf8", "#ef4444"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.13, wspace=0.35)

    xs    = list(range(len(round_sizes)))
    xlbls = [f"T={T:,}" for T in round_sizes]

    for ax, means, stds, colour, ylabel, title, hline, hlabel in [
        (axes[0],
         mean_expls, std_expls, GREEN,
         "Mean exploitability  expl(π_t)",
         "Mean Exploitability vs Match Length",
         0.0, "Unexploitable (0)"),
        (axes[1],
         mean_payoffs, std_payoffs, BLUE,
         "Mean payoff per hand",
         "Mean Payoff per Hand vs Match Length",
         v_star, f"Equilibrium baseline (v*={v_star:.4f})"),
    ]:
        ax.set_facecolor(PANEL)
        means_arr = np.array(means)
        stds_arr  = np.array(stds)

        ax.bar(xs, means_arr, color=colour, alpha=0.7, width=0.5, zorder=3)
        ax.errorbar(xs, means_arr, yerr=stds_arr, fmt='none',
                    color=colour, capsize=6, capthick=1.5,
                    elinewidth=1.5, zorder=4)

        for i, (m, s) in enumerate(zip(means_arr, stds_arr)):
            offset = (s + abs(m) * 0.02 + 0.002) * (1 if m >= 0 else -1)
            ax.text(i, m + offset, f"{m:+.4f}", ha='center',
                    va='bottom' if m >= 0 else 'top',
                    color=TEXT, fontsize=8, fontfamily='monospace')

        ax.axhline(hline, color=RED, lw=1.2, ls='--', alpha=0.7,
                   zorder=2, label=hlabel)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlbls, color=MUTED, fontsize=9)
        ax.set_xlabel("Hands per match (T)", color=MUTED, fontsize=10, labelpad=6)
        ax.set_ylabel(ylabel, color=colour, fontsize=9)
        ax.tick_params(axis="y", colors=colour, labelsize=8)
        ax.tick_params(axis="x", colors=MUTED, labelsize=8)
        ax.set_title(title, color=TEXT, fontsize=10, fontweight='bold', pad=8)
        ax.grid(True, axis='y', color=GRID, lw=0.5, zorder=0)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.legend(fontsize=8, framealpha=0.3, facecolor=PANEL,
                  edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle(
        f"RWYW vs Dynamic Best Response  ·  Leduc Poker  ·  "
        f"{NUM_RUNS} runs per T  ·  seed {seed}",
        color=TEXT, fontsize=11, fontweight="bold",
    )

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "rwyw_exploitability_leduc.png")
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {path}")


if __name__ == "__main__":
    run_suite()