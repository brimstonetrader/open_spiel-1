"""
test_rwyw_exploitability.py
----------------------------
Shows that RWYW's expected exploitability shrinks toward zero as the number
of hands per match grows large.

For each round size T in [108, 1080, 1998, 5004, 9990]:
  - Run NUM_RUNS independent matches of T hands each
  - Each match: RWYW agent vs dynamic best response (recomputed every hand)
  - Record mean exploitability across all hands in the match
  - Average across NUM_RUNS matches

Plots mean exploitability vs T, showing convergence to zero.
"""

import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pyspiel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from open_spiel.python.examples.rwyw_kuhn import RWYWAgent, _ev, _pure_strategies, _all_infosets

ROUND_SIZES = [108, 1080, 1998, 5004, 9990]
NUM_RUNS    = 5
V_STAR      = -1.0 / 18.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent(game):
    agent = RWYWAgent(game, player_id=0)
    opp_iss = _all_infosets(game, agent.opp_id)
    agent._opp_pure_strats, _ = _pure_strategies(opp_iss)
    return agent


def exploitability(agent):
    """expl(pi) = v* - min_{opp pure} EV(pi, opp_pure)"""
    return agent.v_star - min(
        _ev(agent.game, agent.player_id, agent.get_policy(), op)
        for op in agent._opp_pure_strats
    )


def opp_best_response(agent):
    # if agent.k <= 0:
    #     # Agent plays GTO — uniform random is fine, skip expensive BR
    #     return {iss: random.choice([p[iss] for p in agent._opp_pure_strats])
    #             for iss in agent._opp_pure_strats[0]}
    return min(
        agent._opp_pure_strats,
        key=lambda op: _ev(agent.game, agent.player_id, agent.get_policy(), op)
    )


def play_hand(game, agent, opp_pure):
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            state.apply_action(random.choices(actions, weights=probs)[0])
        elif state.current_player() == agent.player_id:
            state.apply_action(agent.step(state))
        else:
            iss    = state.information_state_string(state.current_player())
            action = opp_pure.get(iss, state.legal_actions()[0])
            agent.inform_action(state, state.current_player(), action)
            state.apply_action(action)
    agent.on_terminal(state)
    return state.returns()[agent.player_id]


# ---------------------------------------------------------------------------
# Run one match of T hands, return mean exploitability over the match
# ---------------------------------------------------------------------------

def run_match(game, num_hands, seed):
    random.seed(seed)
    np.random.seed(seed)
    agent   = make_agent(game)
    expls   = []
    payoffs = []
    for _ in range(num_hands):
        br = opp_best_response(agent)
        expls.append(exploitability(agent))
        payoffs.append(play_hand(game, agent, br))
    return float(np.mean(expls)), float(np.mean(payoffs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_suite(base_seed=None):
    if base_seed is None:
        base_seed = random.randint(0, 1_000_000)
    print(f"Base seed: {base_seed}  |  {NUM_RUNS} runs per round size")
    print(f"Round sizes: {ROUND_SIZES}\n")

    game = pyspiel.load_game("kuhn_poker")

    mean_expls  = []
    std_expls   = []
    mean_payoffs = []
    std_payoffs  = []

    for T in ROUND_SIZES:
        run_expls   = []
        run_payoffs = []
        for run_idx in range(NUM_RUNS):
            seed   = base_seed + T + run_idx
            me, mp = run_match(game, T, seed)
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

    plot(ROUND_SIZES, mean_expls, std_expls, mean_payoffs, std_payoffs, base_seed)
    return ROUND_SIZES, mean_expls, std_expls, mean_payoffs, std_payoffs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(round_sizes, mean_expls, std_expls, mean_payoffs, std_payoffs, seed):
    BG, PANEL = "#ffffff", "#111827"
    GRID, MUTED, TEXT = "#1e293b", "#64748b", "#000000"
    GREEN, BLUE, RED = "#34d399", "#38bdf8", "#ef4444"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.13, wspace=0.35)

    xs     = list(range(len(round_sizes)))
    xlbls  = [f"T={T:,}" for T in round_sizes]
    gto_pay = V_STAR   # expected pay per hand at GTO

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
         gto_pay, f"GTO baseline (v*={V_STAR:.4f})"),
    ]:
        ax.set_facecolor(PANEL)
        means_arr = np.array(means)
        stds_arr  = np.array(stds)

        ax.bar(xs, means_arr, color=colour, alpha=0.7, width=0.5, zorder=3)
        ax.errorbar(xs, means_arr, yerr=stds_arr, fmt='none',
                    color=colour, capsize=6, capthick=1.5, elinewidth=1.5, zorder=4)

        for i, (m, s) in enumerate(zip(means_arr, stds_arr)):
            ax.text(i, m + (s if m >= 0 else -s) + 0.0005 * (1 if m >= 0 else -1),
                    f"{m:+.4f}", ha='center',
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
        f"RWYW vs Dynamic Best Response  ·  Kuhn Poker  ·  "
        f"{NUM_RUNS} runs per T  ·  seed {seed}",
        color=TEXT, fontsize=11, fontweight="bold",
    )

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "rwyw_exploitability.png")
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {path}")


if __name__ == "__main__":
    run_suite()