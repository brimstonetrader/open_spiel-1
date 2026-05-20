import numpy as np
import random
from collections import defaultdict
import pyspiel
from open_spiel.python import rl_agent
from open_spiel.python.examples import gto_kuhn_poker

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


class RWYWEAgent(rl_agent.AbstractAgent):
    def __init__(self, game: pyspiel.Game, player_id: int, name='rwywe_agent'):
        super().__init__(game, player_id, name)
        self.game = game
        self.player_id = player_id
        self.opp_id = 1 - player_id
        self.v_star = -1.0 / 18.0
        self.k = 0.0
        self.opp_model = defaultdict(lambda: defaultdict(int))
        self.gto_policy = gto_kuhn_poker.GTOKuhnPolicy(game, alpha=0.2) 
        self.num_rollouts = 100
        self._pending_mixed_ev = None   # set in step(), consumed in on_terminal()
        self._pending_action_probs = None  # mixed strategy used this hand


    def _simulate_rollout(self, state):
        s = state.clone()
        while not s.is_terminal():
            legal = s.legal_actions()
            a = legal[random.randint(0, len(legal) - 1)]
            s.apply_action(a)
        return s.returns()[self.player_id]

    def _opp_action_probs(self, info_str):
        counts = self.opp_model[info_str]
        if not counts:
            return None  # unseen infoset → caller should use uniform / GTO
        total = sum(counts.values())
        return {a: 0.5 for a, c in counts.items()}

    def _estimate_action_ev(self, state, action):
        total = 0.0
        for _ in range(self.num_rollouts):
            s = state.clone()
            s.apply_action(action)
            total += self._simulate_rollout(s)
        return total / self.num_rollouts

    # ------------------------------------------------------------------
    # Core decision logic
    # ------------------------------------------------------------------

    def _exploitability_of_pure(self, state, action):
        ev = self._estimate_action_ev(state, action)
        worst_case_ev = ev - np.sqrt(1.0 / self.num_rollouts)  # >= true min w.h.p.
        expl_estimate = max(0.0, self.v_star - worst_case_ev)
        return ev, expl_estimate

    def step(self, state):
        legal = state.legal_actions()

        action_stats = {}  # action -> (ev, expl_estimate)
        for a in legal:
            ev, expl = self._exploitability_of_pure(state, a)
            action_stats[a] = (ev, expl)
        safe_actions = [(a, ev) for a, (ev, expl) in action_stats.items() if expl <= self.k]
        if safe_actions:
            best_action, best_ev = max(safe_actions, key=lambda x: x[1])
            self._pending_action_probs = {a: 1.0 if a == best_action else 0.0 for a in legal}
            self._pending_action_evs = {a: action_stats[a][0] for a in legal}
            chosen_action = best_action
        else:
            gto_probs = self.gto_policy.action_probabilities(state)
            actions = list(gto_probs.keys())
            weights = [gto_probs[a] for a in actions]
            chosen_action = random.choices(actions, weights)[0]
            self._pending_action_probs = gto_probs
            self._pending_action_evs = {a: action_stats[a][0] for a in legal}
        return chosen_action

    def inform_action(self, state, player, action):
        if player == self.opp_id:
            info = state.information_state_string(self.opp_id)
            self.opp_model[info][action] += 1
            self._last_opp_action = action

    def on_terminal(self, state):
        if self._pending_action_probs is None:
            return
        opp_action = getattr(self, '_last_opp_action', None)
        if opp_action is not None and self._pending_action_evs is not None:
            mixed_ev = sum(
                self._pending_action_probs.get(a, 0.0) * ev
                for a, ev in self._pending_action_evs.items()
            )
        else:
            mixed_ev = self.v_star

        self.k = max(0.0, self.k + mixed_ev - self.v_star)
        self._pending_action_probs = None
        self._pending_action_evs = None
        self._last_opp_action = None

    def restart(self):
        self._pending_action_probs = None
        self._pending_action_evs = None
        self._last_opp_action = None


def run_simulation(num_hands: int = 1000, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    game = pyspiel.load_game("kuhn_poker")
    agent = RWYWEAgent(game, player_id=0)
    gto = gto_kuhn_poker.GTOKuhnPoker(game, 1)
    def random_policy(state):
        return random.choice(state.legal_actions())
    k_history = []
    gain_history = []
    cumulative_gain = 0.0
    for _ in range(num_hands):
        state = game.new_initial_state()
        while not state.is_terminal():
            cur = state.current_player()
            if cur == pyspiel.PlayerId.CHANCE:
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(np.random.choice(outcomes, p=probs))
            elif cur == agent.player_id:
                action = agent.step(state)
                state.apply_action(action)
            else:
                opp_action = random_policy(state)
                agent.inform_action(state, cur, opp_action)
                state.apply_action(opp_action)

        agent.on_terminal(state)

        hand_payoff = state.returns()[agent.player_id]
        cumulative_gain += hand_payoff

        k_history.append(agent.k)
        gain_history.append(cumulative_gain)

    return k_history, gain_history


def plot_results(k_history, gain_history):
    hands = range(1, len(k_history) + 1)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    # --- Cumulative gain (left axis) ---
    color_gain = "#16a34a"
    ax1.plot(hands, gain_history, color=color_gain, linewidth=1.4,
             alpha=0.85, label="Cumulative gain")
    ax1.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Hand", color="black", fontsize=11)
    ax1.set_ylabel("Cumulative gain (chips)", color=color_gain, fontsize=11)
    ax1.tick_params(axis="y", colors=color_gain)
    ax1.tick_params(axis="x", colors="black")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#cccccc")

    # --- k budget (right axis) ---
    ax2 = ax1.twinx()
    ax2.set_facecolor("white")
    color_k = "#c2410c"
    ax2.plot(hands, k_history, color=color_k, linewidth=1.4,
             alpha=0.75, linestyle="--", label="k (exploit budget)")
    ax2.set_ylabel("k  (exploit budget)", color=color_k, fontsize=11)
    ax2.tick_params(axis="y", colors=color_k)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#cccccc")

    # --- Legend ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2, labels1 + labels2,
                        loc="upper left", framealpha=0.25,
                        facecolor="white", edgecolor="#cccccc",
                        labelcolor="black", fontsize=10)

    fig.suptitle("RWYWE Agent — 1 000 hands of Kuhn Poker\nvs. random opponent",
                 color="black", fontsize=13, y=1.01)
    fig.tight_layout()
    plt.savefig("rwywe_simulation.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.show()
    print("Chart saved to rwywe_simulation.png")


if __name__ == "__main__":
    k_hist, gain_hist = run_simulation(num_hands=1000)
    plot_results(k_hist, gain_hist)