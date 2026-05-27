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

from open_spiel.python.examples.rwywe import RWYWEAgent, _ev, _pure_strategies, _all_infosets

ROUND_SIZES = [108, 1080, 1998, 5004, 9990]
NUM_RUNS    = 5
V_STAR      = -1.0 / 18.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent(game):
    agent = RWYWEAgent(game, player_id=0)
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
    agent  = make_agent(game)
    expls  = []
    for _ in range(num_hands):
        br   = opp_best_response(agent)
        expls.append(exploitability(agent))
        play_hand(game, agent, br)
    return float(np.mean(expls))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_suite(base_seed=None):
    if base_seed is None:
        base_seed = random.randint(0, 1_000_000)
    print(f"Base seed: {base_seed}  |  {NUM_RUNS} runs per round size")
    print(f"Round sizes: {ROUND_SIZES}\n")

    game = pyspiel.load_game("kuhn_poker")

    mean_expls = []   # one entry per round size
    std_expls  = []

    for T in ROUND_SIZES:
        run_means = []
        for run_idx in range(NUM_RUNS):
            seed     = base_seed + T + run_idx
            mean_e   = run_match(game, T, seed)
            run_means.append(mean_e)
            print(f"  T={T:>5}  run {run_idx+1}/{NUM_RUNS}  mean_expl={mean_e:.5f}")

        m = float(np.mean(run_means))
        s = float(np.std(run_means))
        mean_expls.append(m)
        std_expls.append(s)
        print(f"  T={T:>5}  avg={m:.5f}  std={s:.5f}\n")

    plot(ROUND_SIZES, mean_expls, std_expls, base_seed)
    return ROUND_SIZES, mean_expls, std_expls


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(round_sizes, mean_expls, std_expls, seed):
    BG, PANEL = "#ffffff", "#111827"
    GRID, MUTED, TEXT = "#1e293b", "#64748b", "#000000"
    GREEN = "#34d399"

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(PANEL)

    xs    = list(range(len(round_sizes)))
    means = np.array(mean_expls)
    stds  = np.array(std_expls)

    # Bar chart
    bars = ax.bar(xs, means, color=GREEN, alpha=0.7, width=0.5, zorder=3)

    # Error bars
    ax.errorbar(xs, means, yerr=stds, fmt='none',
                color=GREEN, capsize=6, capthick=1.5, elinewidth=1.5, zorder=4)

    # Value labels on bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.001, f"{m:.4f}",
                ha='center', va='bottom', color=TEXT, fontsize=8,
                fontfamily='monospace')

    # Zero line
    ax.axhline(0, color=TEXT, lw=0.6, alpha=0.4, zorder=2)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"T={T:,}" for T in round_sizes], color=MUTED, fontsize=9)
    ax.set_xlabel("Hands per match (T)", color=MUTED, fontsize=10, labelpad=6)
    ax.set_ylabel("Mean exploitability  expl(π_t)", color=GREEN, fontsize=10)
    ax.tick_params(axis="y", colors=GREEN, labelsize=8)
    ax.grid(True, axis='y', color=GRID, lw=0.5, zorder=0)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)

    ax.set_title(
        f"RWYW Expected Exploitability vs Match Length  ·  Kuhn Poker\n"
        f"Dynamic best response opponent  ·  {NUM_RUNS} runs per T  ·  seed {seed}",
        color=TEXT, fontsize=11, fontweight="bold", pad=10,
    )

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "rwyw_exploitability.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {path}")


if __name__ == "__main__":
    run_suite()