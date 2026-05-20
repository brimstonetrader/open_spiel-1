"""
run_rwywe_warmup_leduc.py
--------------------------
Plots cumulative profit and safety budget k_t for four RWYWEAgent variants
(0, 20, 2000, 20000 warmup hands) against a sophisticated static opponent
in Leduc poker: fixed mixed strategy within ±20% of uniform at each ISS.

Leduc poker: 2-player, zero-sum, imperfect information. 6-card deck
(J/Q/K × 2 suits), public flop card, two betting rounds. Substantially
larger than Kuhn poker — good test of RWYWE scaling.

Run from the directory containing rwywe3.py:
    python run_rwywe_warmup_leduc.py
"""

import sys, random
sys.path.insert(0, ".")

import numpy as np
import pyspiel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

from open_spiel.python.examples.rwywe3 import RWYWEAgent


GAME_NAME = "leduc_poker"
OPP_NOISE = 0.20
NUM_HANDS = 1000
NUM_SEEDS = 5
WARMUPS   = [0, 20, 2000, 20000]
LABELS    = ["0 warmup", "20 warmup", "2 000 warmup", "20 000 warmup"]


# ── Opponent: fixed mixed strategy ±noise of uniform ─────────────────────────

def build_static_opponent(game, noise=0.20, seed=0, mccfr_iters=10000):
    """
    Build a near-equilibrium static opponent for player 1 by:
      1. Running MCCFR for mccfr_iters iterations to get an approximate
         equilibrium strategy.
      2. Perturbing each ISS's action probabilities by ±noise and renormalising.
      3. Fixing that perturbed strategy for the entire match.

    This gives a genuinely suboptimal but strategically coherent opponent —
    not one that folds randomly 33% of the time.
    """
    from open_spiel.python.algorithms import outcome_sampling_mccfr

    rng = random.Random(seed)

    # Solve for approximate equilibrium
    print(f"    Solving Leduc equilibrium ({mccfr_iters} MCCFR iters)...",
          flush=True)
    solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
    for _ in range(mccfr_iters):
        solver.iteration()
    avg_policy = solver.average_policy()

    # Walk tree, perturb equilibrium probs by ±noise at each player-1 ISS
    strategy = {}

    def walk(s):
        if s.is_terminal(): return
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                s2 = s.clone(); s2.apply_action(a); walk(s2)
        elif s.current_player() == 1:
            iss = s.information_state_string(1)
            if iss not in strategy:
                legal = s.legal_actions()
                dist  = avg_policy.action_probabilities(s, 1)
                raw   = np.array([
                    max(0.0, dist.get(a, 0.0) + rng.uniform(-noise, noise))
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

    def act(state):
        iss = state.information_state_string(1)
        if iss not in strategy:
            return rng.choice(state.legal_actions())
        acts, ws = zip(*strategy[iss])
        return rng.choices(acts, weights=ws)[0]

    return act


# ── Single hand ───────────────────────────────────────────────────────────────

def play_hand(game, agent, opp_act):
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            state.apply_action(random.choices(actions, probs)[0])
        elif state.current_player() == agent.player_id:
            action = agent.step(state)
            if action is None:
                action = random.choice(state.legal_actions(agent.player_id))
            state.apply_action(action)
        else:
            a = opp_act(state)
            agent.inform_action(state, state.current_player(), a)
            state.apply_action(a)
    agent.on_terminal(state)
    return state.returns()[agent.player_id]


# ── Warmup ────────────────────────────────────────────────────────────────────

def warmup_nn(game, agent, warmup_hands, opp_act, rng):
    """
    Pre-play warmup_hands hands against the real match opponent so the NN
    opponent model accumulates training data before the match begins.
    Agent acts randomly during warmup — we only care about NN training.
    """
    for _ in range(warmup_hands):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(actions, probs)[0])
            elif state.current_player() == agent.player_id:
                state.apply_action(rng.choice(state.legal_actions()))
            else:
                a = opp_act(state)
                agent.inform_action(state, state.current_player(), a)
                state.apply_action(a)
        agent.on_terminal(state)


# ── Variant runner ────────────────────────────────────────────────────────────

def run_variant(warmup_hands, num_hands=NUM_HANDS, opp_seed=7,
                base_hand_seed=42, shared_opp=None):
    all_profits = np.zeros(num_hands)
    all_ks      = np.zeros(num_hands)

    # Use shared opponent if provided, otherwise build (slow)
    if shared_opp is None:
        game_for_opp = pyspiel.load_game(GAME_NAME)
        opp_act      = build_static_opponent(game_for_opp, noise=OPP_NOISE, seed=opp_seed)
    else:
        opp_act = shared_opp

    for seed_offset in range(NUM_SEEDS):
        hand_seed = base_hand_seed + warmup_hands + seed_offset * 999983
        rng       = random.Random(hand_seed)
        random.seed(hand_seed)
        np.random.seed(hand_seed)

        game  = pyspiel.load_game(GAME_NAME)
        agent = RWYWEAgent(game, player_id=0)

        if warmup_hands > 0:
            warmup_nn(game, agent, warmup_hands, opp_act, rng)

        cumulative = 0.0
        for h in range(num_hands):
            cumulative     += play_hand(game, agent, opp_act)
            all_profits[h] += cumulative
            all_ks[h]      += agent.k

    profits = (all_profits / NUM_SEEDS).tolist()
    ks      = (all_ks      / NUM_SEEDS).tolist()

    print(f"  warmup={warmup_hands:>6} hands  |  "
          f"avg final profit={profits[-1]:+.2f}  |  avg final k={ks[-1]:.2f}")
    return profits, ks


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(all_profits, all_ks, path="rwywe_warmup_leduc.png"):
    hands  = list(range(1, NUM_HANDS + 1))
    BG     = "#FFFFFF"
    PANEL  = "#111827"
    GRID   = "#1e293b"
    TEXT   = "#000000"
    MUTED  = "#64748b"
    ACCENT = "#f59e0b"
    SERIES = ["#38bdf8", "#34d399", "#fb923c", "#f472b6"]

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = GridSpec(2, 2, figure=fig,
                   hspace=0.38, wspace=0.30,
                   left=0.07, right=0.96, top=0.88, bottom=0.08)

    for idx, (warmup, label) in enumerate(zip(WARMUPS, LABELS)):
        row, col = divmod(idx, 2)
        ax1 = fig.add_subplot(gs[row, col])
        ax2 = ax1.twinx()

        profits = all_profits[idx]
        ks      = all_ks[idx]
        colour  = SERIES[idx]

        ax1.plot(hands, profits, color=colour, lw=2.0, label="Cumulative profit")
        ax1.axhline(0, color=TEXT, lw=0.35, alpha=0.25)

        ax2.fill_between(hands, ks, alpha=0.07, color=ACCENT, linewidth=0)
        ax2.plot(hands, ks, color=ACCENT, lw=1.2, alpha=0.85, label=r"$k_t$")

        ax1.set_facecolor(PANEL)
        for sp in ax1.spines.values():
            sp.set_edgecolor(GRID)
        ax1.grid(True, color=GRID, lw=0.5, zorder=0)
        ax1.set_xlim(1, NUM_HANDS)
        ax1.tick_params(axis="y", colors=colour, labelsize=8)
        ax1.tick_params(axis="x", colors=MUTED, labelsize=8)
        ax2.tick_params(axis="y", colors=ACCENT, labelsize=8)
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%+.0f"))
        ax1.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax1.set_xlabel("Hand", color=MUTED, fontsize=9, labelpad=4)
        ax1.set_ylabel("Cumulative profit", color=colour, fontsize=9)
        ax2.set_ylabel(r"$k_t$", color=ACCENT, fontsize=9)
        ax1.set_title(
            f"{label}\nprofit {profits[-1]:+.1f}  ·  k_final {ks[-1]:.1f}",
            color=TEXT, fontsize=10, pad=8, loc="left", fontfamily="monospace")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.5,
                   framealpha=0.3, facecolor=PANEL,
                   edgecolor=GRID, labelcolor=TEXT)

    fig.text(0.515, 0.945,
             f"RWYWE  ·  NN Warmup vs Static Opponent "
             f"(±{OPP_NOISE*100:.0f}% uniform)  ·  Leduc Poker",
             ha="center", color=TEXT, fontsize=13, fontweight="bold")
    fig.text(0.515, 0.920,
             f"Warmup = pre-match hands to train NN opponent model  ·  "
             f"{NUM_HANDS:,} match hands  ·  averaged over {NUM_SEEDS} seeds",
             ha="center", color=MUTED, fontsize=9)

    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build the opponent once — MCCFR solve is expensive
    print("Building opponent strategy (solve once)...", flush=True)
    _game_for_opp = pyspiel.load_game(GAME_NAME)
    _shared_opp   = build_static_opponent(_game_for_opp, noise=OPP_NOISE, seed=7)

    all_profits, all_ks = [], []
    for warmup in WARMUPS:
        print(f"Running warmup={warmup:>6} hands ...", flush=True)
        profits, ks = run_variant(warmup, num_hands=NUM_HANDS, shared_opp=_shared_opp)
        all_profits.append(profits)
        all_ks.append(ks)

    plot(all_profits, all_ks,
         path="/home/nmorr/open_spiel/open_spiel/python/tests/rwywe_warmup_leduc.png")