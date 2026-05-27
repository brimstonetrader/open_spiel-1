"""
test_intrahand_budget.py
------------------------
Compares two variants of RWYWEAgent in Leduc poker:

  1. Standard RWYWE  — k is updated once per hand at on_terminal(),
     using the full realized return. The safety gate uses the full k
     at every decision node within the hand.

  2. Intra-hand Budgeted RWYWE — the available k budget is split across
     decision nodes within a hand. Before each decision node, the agent
     sees only a fraction of k proportional to how many decision nodes
     remain. After each opponent action is observed mid-hand, a partial
     k-update is applied (gift detected → more budget available for the
     next node). This implements the within-game-iteration exploitation
     idea from Section 8.3 of Ganzfried & Sandholm (2015).

     Concretely for Leduc (up to 2 agent decision nodes per hand):
       - Node 1 (pre-flop):  gate threshold = k / num_remaining_nodes
       - After observing opp's pre-flop action, compute partial delta
         = realized_ev_if_terminal_here - v_star and add to remaining budget
       - Node 2 (post-flop): gate threshold = updated remaining budget

Results are averaged over NUM_RUNS × NUM_HANDS and plotted as cumulative
profit curves with k_t on a secondary axis.
"""

import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pyspiel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

import torch
import torch.nn.functional as F
from collections import deque

from open_spiel.python.algorithms import outcome_sampling_mccfr
from open_spiel.python.examples.rwywe3 import RWYWEAgent, OpponentMLP


# ---------------------------------------------------------------------------
# Intra-hand budgeted agent (subclass of RWYWEAgent)
# ---------------------------------------------------------------------------

class RWYWEBudgetedAgent(RWYWEAgent):
    """
    Variant of RWYWEAgent that splits the k budget across decision nodes
    within a hand, allowing mid-hand gift detection and exploitation.

    Key differences from standard RWYWEAgent:
      - Tracks how many agent decision nodes have fired this hand
      - Gate threshold at node i = remaining_k / remaining_nodes
      - After each opponent action, estimates a partial gift value and
        adds it to remaining_k for subsequent nodes
      - on_terminal() only handles NN training; k is updated mid-hand
    """

    def __init__(self, game, player_id, name="rwywe_budgeted"):
        super().__init__(game, player_id, name)
        # Per-hand intra-hand state
        self._remaining_k      = 0.0
        self._nodes_this_hand  = 0
        self._opp_actions_seen = []   # list of (state_clone, action) pairs

    def restart(self):
        super().restart()
        self._remaining_k      = 0.0
        self._nodes_this_hand  = 0
        self._opp_actions_seen = []

    def _estimate_remaining_nodes(self, state):
        """
        Estimate how many more agent decision nodes remain this hand.
        For Leduc: pre-flop node + (possibly) post-flop node = max 2.
        We use history length as a proxy: short history → early in hand.
        """
        hist_len = len(state.history())
        # Leduc: chance deals 2 cards (hist 0-1), then betting begins.
        # Pre-flop betting: hist ~2-6. Flop card: hist ~7. Post-flop: ~8+.
        # Conservative estimate: always assume at least 1 more node remains.
        if hist_len < 6:
            return 2   # pre-flop: both nodes remain
        else:
            return 1   # post-flop: only this node remains

    def _partial_gift_value(self, state, opp_action):
        """
        Estimate the gift value from the opponent's observed action at *state*.
        This is the difference between what we'd earn if the game ended here
        vs v_star — a rough mid-hand gift signal.

        We use the MC rollout value of our current MCCFR policy from *state*
        after the opponent plays opp_action, minus v_star.
        """
        total = 0.0
        for _ in range(self.num_rollouts):
            s = state.clone()
            s.apply_action(opp_action)
            total += self._simulate_rollout(s)
        mean_ev = total / self.num_rollouts
        return max(0.0, mean_ev - self.v_star)

    def step(self, state):
        if state.is_terminal(): return None
        if state.current_player() != self.player_id: return None
        legal = state.legal_actions(self.player_id)
        if not legal: return None

        self._nodes_this_hand += 1

        # Budget: split remaining k by estimated remaining decision nodes
        remaining_nodes = max(1, self._estimate_remaining_nodes(state))
        gate_k          = self._remaining_k / remaining_nodes

        # Estimate action values and score with opponent model
        evs = {}
        for a in random.sample(legal, min(len(legal), 10)):
            evs[a] = self._estimate_action_value(state, a)
        for a in legal:
            if a not in evs: evs[a] = float("-inf")

        opp_probs = self._predict_opponent_action_probs(state)
        scored    = [(a, evs[a] + 0.05 * (opp_probs[a] if a < len(opp_probs) else 0.0))
                     for a in legal]
        best_a, _ = max(scored, key=lambda x: x[1])
        best_ev   = evs[best_a]

        # Gate uses per-node budget fraction
        self._last_evs  = dict(evs)
        if best_ev - self.v_star >= gate_k:
            action = best_a
            self._last_dist = {a: 1.0 if a == best_a else 0.0 for a in legal}
            # Consume the budget used by this deviation
            self._remaining_k -= max(0.0, best_ev - self.v_star)
            self._remaining_k  = max(0.0, self._remaining_k)
        else:
            dist  = self.solver.average_policy().action_probabilities(state, self.player_id)
            probs = np.array([dist.get(a, 0.0) for a in legal], dtype=np.float64)
            total = probs.sum()
            probs = probs / total if total > 0 else np.ones(len(legal)) / len(legal)
            action = np.random.choice(legal, p=probs)
            self._last_dist = {a: float(dist.get(a, 0.0)) for a in legal}

        return action

    def inform_action(self, state, player, action):
        # Standard NN training
        super().inform_action(state, player, action)
        if player == self.opp_id:
            # Mid-hand gift detection: add partial gift value to remaining budget
            gift = self._partial_gift_value(state, action)
            self._remaining_k += gift

    def on_terminal(self, state):
        realized_return = state.player_return(self.player_id)

        # End-of-hand: full k update from realized return
        self.k = max(0.0, self.k + (realized_return - self.v_star))

        # Carry remaining_k into next hand (unused budget is preserved)
        self._remaining_k  = self.k
        self._nodes_this_hand = 0

        # NN training
        if len(self.replay) >= self.batch_size:
            batch   = random.sample(self.replay, self.batch_size)
            inputs  = torch.stack([b[0] for b in batch])
            targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
            logits  = self.model(inputs)
            loss    = F.cross_entropy(logits, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self._last_opp_action = None
        self._last_dist       = None
        self._last_evs        = None
        self._opp_actions_seen = []


# ---------------------------------------------------------------------------
# Opponent
# ---------------------------------------------------------------------------

def build_static_opponent(game, noise=0.20, seed=0, mccfr_iters=7500):
    from open_spiel.python.algorithms import outcome_sampling_mccfr as mccfr_mod
    rng    = random.Random(seed)
    solver = mccfr_mod.OutcomeSamplingSolver(game)
    print(f"  Solving opponent equilibrium ({mccfr_iters} iters)...", flush=True)
    for _ in range(mccfr_iters):
        solver.iteration()
    avg_policy = solver.average_policy()

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
                raw   = np.array([max(0.0, dist.get(a, 0.0) + rng.uniform(-noise, noise))
                                  for a in legal])
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


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

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


def run_match(agent_cls, game, opp_act, num_hands, seed):
    random.seed(seed); np.random.seed(seed)
    agent      = agent_cls(game, player_id=0)
    cumulative = 0.0
    profits, ks = [], []
    for _ in range(num_hands):
        cumulative += play_hand(game, agent, opp_act)
        profits.append(cumulative)
        ks.append(agent.k)
    return profits, ks


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

NUM_HANDS = 1000
NUM_RUNS  = 10
GAME_NAME = "leduc_poker"
OPP_NOISE = 0.20

AGENTS = [
    ("Standard RWYWE",          RWYWEAgent),
    ("Intra-hand Budgeted RWYWE", RWYWEBudgetedAgent),
]


def run_suite(base_seed=None):
    if base_seed is None:
        base_seed = random.randint(0, 1_000_000)
    print(f"Base seed: {base_seed}  |  {NUM_RUNS} runs × {NUM_HANDS} hands\n")

    game    = pyspiel.load_game(GAME_NAME)
    opp_act = build_static_opponent(game, noise=OPP_NOISE, seed=42)

    results = {name: {"profits": [], "ks": []} for name, _ in AGENTS}

    for run_idx in range(NUM_RUNS):
        seed = base_seed + run_idx
        print(f"Run {run_idx + 1}/{NUM_RUNS}  (seed {seed})")
        for name, cls in AGENTS:
            profits, ks = run_match(cls, game, opp_act, NUM_HANDS, seed)
            results[name]["profits"].append(profits)
            results[name]["ks"].append(ks)
            print(f"  {name:<30}  profit={profits[-1]:+.3f}  k={ks[-1]:.3f}")
        print()

    # Average across runs
    averaged = {}
    for name, _ in AGENTS:
        p_arr = np.array(results[name]["profits"])
        k_arr = np.array(results[name]["ks"])
        averaged[name] = {
            "mean_profit": p_arr.mean(axis=0).tolist(),
            "std_profit":  p_arr.std(axis=0).tolist(),
            "mean_k":      k_arr.mean(axis=0).tolist(),
            "final_mean":  float(p_arr[:, -1].mean()),
            "final_std":   float(p_arr[:, -1].std()),
        }

    # Summary table
    print(f"\n{'='*60}")
    print(f"  RESULTS  ({NUM_RUNS} runs × {NUM_HANDS} hands)  —  {GAME_NAME}")
    print(f"{'='*60}")
    print(f"  {'Agent':<32} {'Mean profit':>12} {'± std':>8}")
    print(f"  {'-'*32} {'-'*12} {'-'*8}")
    for name, _ in AGENTS:
        r = averaged[name]
        print(f"  {name:<32} {r['final_mean']:>+12.3f} {r['final_std']:>8.3f}")

    plot(averaged, base_seed)
    return averaged


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(averaged, seed):
    hands  = list(range(1, NUM_HANDS + 1))
    COLORS = {"Standard RWYWE": "#38bdf8",
               "Intra-hand Budgeted RWYWE": "#f472b6"}
    K_COLORS = {"Standard RWYWE": "#34d399",
                 "Intra-hand Budgeted RWYWE": "#fb923c"}

    BG, PANEL, GRID = "#ffffff", "#111827", "#1e293b"
    TEXT, MUTED     = "#000000", "#64748b"

    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35,
                   left=0.07, right=0.96, top=0.86, bottom=0.12)

    # Left: cumulative profit comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL)
    ax1.axhline(0, color=TEXT, lw=0.4, alpha=0.3)
    for name, _ in AGENTS:
        r   = averaged[name]
        mp  = r["mean_profit"]
        std = r["std_profit"]
        c   = COLORS[name]
        ax1.plot(hands, mp, color=c, lw=2.0, label=name)
        ax1.fill_between(hands,
                         [m - s for m, s in zip(mp, std)],
                         [m + s for m, s in zip(mp, std)],
                         color=c, alpha=0.12)
    ax1.set_xlim(1, NUM_HANDS)
    ax1.set_xlabel("Hand", color=MUTED, fontsize=9)
    ax1.set_ylabel("Cumulative profit", color=TEXT, fontsize=9)
    ax1.set_title("Cumulative Profit", color=TEXT, fontsize=11, pad=8)
    ax1.tick_params(colors=MUTED, labelsize=8)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%+.0f"))
    ax1.grid(True, color=GRID, lw=0.5)
    for sp in ax1.spines.values(): sp.set_edgecolor(GRID)
    ax1.legend(fontsize=8, framealpha=0.3, facecolor=PANEL,
               edgecolor=GRID, labelcolor=TEXT)

    # Right: k_t comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL)
    ax2.axhline(0, color=TEXT, lw=0.4, alpha=0.3)
    for name, _ in AGENTS:
        r  = averaged[name]
        mk = r["mean_k"]
        c  = K_COLORS[name]
        ax2.plot(hands, mk, color=c, lw=2.0, label=name)
    ax2.set_xlim(1, NUM_HANDS)
    ax2.set_xlabel("Hand", color=MUTED, fontsize=9)
    ax2.set_ylabel(r"Safety budget $k_t$", color=TEXT, fontsize=9)
    ax2.set_title(r"Safety Budget $k_t$", color=TEXT, fontsize=11, pad=8)
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax2.grid(True, color=GRID, lw=0.5)
    for sp in ax2.spines.values(): sp.set_edgecolor(GRID)
    ax2.legend(fontsize=8, framealpha=0.3, facecolor=PANEL,
               edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle(
        f"Standard vs Intra-hand Budgeted RWYWE  ·  Leduc Poker  ·  "
        f"Static Opponent ±{OPP_NOISE*100:.0f}% GTO  ·  "
        f"{NUM_RUNS} runs × {NUM_HANDS} hands",
        color=TEXT, fontsize=11, fontweight="bold", y=0.97
    )

    path = "/home/nmorr/open_spiel/open_spiel/python/tests/rwywe_intrahand_budget.png"
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    run_suite(base_seed=random.randint(0, 1_000_000))