import numpy as np
import random

import pyspiel
from open_spiel.python import rl_agent


# ---------------------------------------------------------------------------
# Kuhn poker GTO opponent model (hardcoded Nash equilibrium, alpha = 1/3)
#
# Info-state string format used by OpenSpiel's kuhn_poker:
#   card digit (0=J, 1=Q, 2=K) followed by action history ('p'=Pass, 'b'=Bet)
#
# Actions: 0 = Pass / Check / Fold
#          1 = Bet  / Call
#
# The unique fully-mixed Nash equilibrium is parameterised by alpha = 1/3.
# All 12 reachable info-states are listed explicitly.
# ---------------------------------------------------------------------------
_ALPHA = 1.0 / 3.0

# Maps info_state_string -> {action: probability}
KUHN_GTO: dict[str, dict[int, float]] = {
    # ---- Player 0 (acts first) ----
    "0":    {0: 1 - _ALPHA, 1: _ALPHA},   # J:  pass 2/3, bet 1/3
    "1":    {0: 1.0,        1: 0.0},       # Q:  always pass
    "2":    {0: 0.0,        1: 1.0},       # K:  always bet
    "0pb":  {0: 1.0,        1: 0.0},       # J facing bet after pass: fold
    "1pb":  {0: 1 - _ALPHA, 1: _ALPHA},   # Q facing bet after pass: call 1/3
    "2pb":  {0: 0.0,        1: 1.0},       # K facing bet after pass: always call
    # ---- Player 1 (acts second) ----
    "0p":   {0: 1.0,        1: 0.0},       # J after opponent check: check behind
    "1p":   {0: 1 - _ALPHA, 1: _ALPHA},   # Q after opponent check: bet 1/3
    "2p":   {0: 0.0,        1: 1.0},       # K after opponent check: always bet
    "0b":   {0: 1.0,        1: 0.0},       # J facing bet: fold
    "1b":   {0: 1 - _ALPHA, 1: _ALPHA},   # Q facing bet: call 1/3
    "2b":   {0: 0.0,        1: 1.0},       # K facing bet: always call
}

_UNIFORM_2 = {0: 0.5, 1: 0.5}  # fallback for unseen info states


class RWYWEAgent(rl_agent.AbstractAgent):
    """
    Risk What You've Won in Expectation agent.

    High-level behaviour:
    1. Use the hardcoded Kuhn poker GTO strategy as both the baseline and
       the rollout policy.
    2. For each legal action, estimate its value with a few GTO rollouts.
    3. Use the hardcoded GTO opponent model to bias toward actions that
       exploit the opponent's known strategy.
    4. Only deviate from the GTO baseline if the estimated value is "safe"
       relative to the running safety budget k.
    """

    def __init__(self, game: pyspiel.Game, player_id: int, name: str = "rwywe_agent"):
        super().__init__(game, player_id, name)

        self.game = game
        self.player_id = player_id
        self.opp_id = 1 - player_id

        # RWYWE safety budget.
        self.k = -0.018
        self.v_star = -0.056

        # Number of rollout continuations used to estimate the value of an action.
        self.num_rollouts = 3

        self.num_actions = game.num_distinct_actions()

        # Per-hand state
        self.pending_opp_obs  = None
        self._last_opp_action = None
        self._last_dist       = None
        self._last_evs        = None

    # ------------------------------------------------------------------
    # Opponent model
    # ------------------------------------------------------------------

    def _predict_opponent_action_probs(self, state) -> dict[int, float]:
        """
        Return a probability distribution over the opponent's legal actions
        using the hardcoded Kuhn poker GTO table.

        Keys are action IDs; values are probabilities that sum to 1.
        Falls back to uniform if the info state is not in the table
        (should never happen in standard Kuhn poker).
        """
        info_str = state.information_state_string(self.opp_id)
        dist = KUHN_GTO.get(info_str, _UNIFORM_2)

        # Restrict to actually-legal actions and renormalise (safety).
        legal = set(state.legal_actions(self.opp_id))
        filtered = {a: p for a, p in dist.items() if a in legal}
        total = sum(filtered.values())
        if total <= 0:
            return {a: 1.0 / len(legal) for a in legal}
        return {a: p / total for a, p in filtered.items()}

    # ------------------------------------------------------------------
    # Rollout helpers
    # ------------------------------------------------------------------

    def _gto_action_probs(self, state, player) -> dict[int, float]:
        """
        Return GTO probabilities for `player` at `state`, restricted to legal
        actions and renormalised. Falls back to uniform for unseen info states.
        """
        info_str = state.information_state_string(player)
        dist = KUHN_GTO.get(info_str, _UNIFORM_2)
        legal = set(state.legal_actions(player))
        filtered = {a: p for a, p in dist.items() if a in legal}
        total = sum(filtered.values())
        if total <= 0:
            return {a: 1.0 / len(legal) for a in legal}
        return {a: p / total for a, p in filtered.items()}

    def _rollout_policy_action(self, state, cp):
        """Sample one action from the GTO policy at the given state."""
        legal = state.legal_actions(cp)
        if not legal:
            return None
        dist = self._gto_action_probs(state, cp)
        actions = list(dist.keys())
        probs = np.array([dist[a] for a in actions], dtype=np.float64)
        return np.random.choice(actions, p=probs)

    def _simulate_rollout(self, state):
        """Roll the game forward using the baseline policy; return this agent's payoff."""
        s = state.clone()
        depth = 0
        while not s.is_terminal() and depth < 40:
            cp = s.current_player()
            if cp < 0:
                break
            a = self._rollout_policy_action(s, cp)
            if a is None:
                break
            s.apply_action(a)
            depth += 1
        return s.returns()[self.player_id] if s.is_terminal() else 0.0

    def _estimate_lower_confidence_bound(self, mean_value, iters, confidence=0.95):
        """Hoeffding-style lower confidence bound."""
        eps = np.sqrt(np.log(2.0 / (1.0 - confidence)) / (2.0 * max(iters, 1)))
        return mean_value - eps

    def _estimate_action_value(self, state, action):
        """
        Force 'action', then roll out with the baseline policy.
        Returns a conservative lower-confidence estimate.
        """
        total = sum(
            self._simulate_rollout(self._apply(state, action))
            for _ in range(self.num_rollouts)
        )
        avg = total / self.num_rollouts
        return self._estimate_lower_confidence_bound(avg, self.num_rollouts, confidence=0.95)

    @staticmethod
    def _apply(state, action):
        s = state.clone()
        s.apply_action(action)
        return s

    # ------------------------------------------------------------------
    # Core decision methods
    # ------------------------------------------------------------------

    def _exact_pessimistic_value(
        self,
        state,
        observed_trajectory,
        traj_index=0,
    ):
        """
        Exact pessimistic continuation value.

        Rules:
            - our agent follows baseline average policy
            - opponent must follow observed trajectory while possible
            - once trajectory consistency breaks, opponent becomes nemesis
        """

        if state.is_terminal():
            return state.returns()[self.player_id]

        cp = state.current_player()

        if cp < 0:
            return 0.0

        legal = state.legal_actions(cp)

        if cp == self.player_id:
            # expectation under our baseline policy
            dist = self.solver.average_policy().action_probabilities(
                state,
                self.player_id,
            )

            total = 0.0

            for a in legal:
                p = dist.get(a, 0.0)

                if p <= 0:
                    continue

                ns = state.clone()
                ns.apply_action(a)

                total += p * self._exact_pessimistic_value(
                    ns,
                    observed_trajectory,
                    traj_index,
                )

            return total

        # ----------------------------------------------------------
        # opponent node
        # ----------------------------------------------------------

        forced_action = None

        if traj_index < len(observed_trajectory):
            forced_action = observed_trajectory[traj_index]

        # if observed action still legal, force it
        if forced_action in legal:
            ns = state.clone()
            ns.apply_action(forced_action)

            return self._exact_pessimistic_value(
                ns,
                observed_trajectory,
                traj_index + 1,
            )

        # otherwise pessimistic nemesis continuation
        best = float("inf")

        for a in legal:
            ns = state.clone()
            ns.apply_action(a)

            v = self._exact_pessimistic_value(
                ns,
                observed_trajectory,
                traj_index,
            )

            if v < best:
                best = v

        return best
    def step(self, state):
        """
        Choose an action at the current state.

        - Estimate values for legal actions via rollouts.
        - Score each action: rollout EV + small bonus for actions that
          are less likely under the GTO opponent model (exploit deviations).
        - If the best candidate clears the RWYWE safety threshold, take it.
        - Otherwise fall back to the GTO baseline.
        """
        if state.is_terminal() or state.current_player() != self.player_id:
            return None
        legal = state.legal_actions(self.player_id)
        if not legal:
            return None

        # Conservative value estimates (sample up to 10 actions).
        evs = {}
        for a in random.sample(legal, min(len(legal), 10)):
            evs[a] = self._estimate_action_value(state, a)
        for a in legal:
            if a not in evs:
                evs[a] = float("-inf")

        # GTO opponent probs: favour actions where the opponent is LESS likely
        # to have the best response (i.e. where our EV gain is largest).
        opp_probs = self._predict_opponent_action_probs(state)

        scored_actions = []
        for a in legal:
            # Bonus is inversely proportional to how often the GTO opponent
            # takes the action that would beat us — use max opp prob as a
            # proxy for how "covered" we are; small coefficient keeps it tiebreaker-scale.
            max_opp_p = max(opp_probs.values()) if opp_probs else 0.5
            model_bonus = (1.0 - max_opp_p) * 0.05
            score = evs[a] + model_bonus
            scored_actions.append((a, score))

        best_a, _ = max(scored_actions, key=lambda x: x[1])
        best_ev = evs[best_a]

        self._last_evs = dict(evs)

        if best_ev - self.v_star >= self.k:
            action = best_a
            self._last_dist = {a: (1.0 if a == best_a else 0.0) for a in legal}
        else:
            dist = self._gto_action_probs(state, self.player_id)
            actions = list(dist.keys())
            probs = np.array([dist[a] for a in actions], dtype=np.float64)
            action = np.random.choice(actions, p=probs)
            self._last_dist = dist

        return action

    def getdist(self, state):
        if state.is_terminal() or state.current_player() != self.player_id:
            return None
        legal = state.legal_actions(self.player_id)
        if not legal:
            return None

        evs = {}
        for a in random.sample(legal, min(len(legal), 10)):
            evs[a] = self._estimate_action_value(state, a)
        for a in legal:
            if a not in evs:
                evs[a] = float("-inf")

        opp_probs = self._predict_opponent_action_probs(state)
        max_opp_p = max(opp_probs.values()) if opp_probs else 0.5

        scored_actions = [(a, evs[a] + (1.0 - max_opp_p) * 0.05) for a in legal]
        best_a, _ = max(scored_actions, key=lambda x: x[1])
        best_ev = evs[best_a]

        if best_ev - self.v_star >= self.k:
            return {a: (np.float64(1) if a == best_a else np.float64(0)) for a in legal}
        else:
            return self._gto_action_probs(state, self.player_id)

    def inform_action(self, state, player, action):
        """Record the opponent's action for the k-update in on_terminal."""
        if player == self.opp_id:
            self._last_opp_action = action

    def restart(self):
        self.pending_opp_obs  = None
        self._last_opp_action = None
        self._last_dist       = None
        self._last_evs        = None

    def on_terminal(self, state):
        realized_return = state.player_return(self.player_id)
        self.k = self.k + (realized_return - self.v_star)

        # reset episodic state
        self.pending_opp_obs  = None
        self._last_opp_action = None
        self._last_dist       = None
        self._last_evs        = None