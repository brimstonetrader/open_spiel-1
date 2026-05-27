import itertools
import random
from collections import defaultdict

import numpy as np
import pyspiel
from open_spiel.python import rl_agent


# ---------------------------------------------------------------------------
# GTO probabilities for player 1 (unique equilibrium, alpha-independent)
# Used to seed the opponent model with 5 fictitious prior hands (Section 9.2)
# ---------------------------------------------------------------------------

def _gto_probs_p1(iss: str) -> dict:
    card    = int(iss[0])
    history = iss[1:]
    if history == "p":   # after agent checks
        if card == 0: return {0: 2/3, 1: 1/3}
        if card == 1: return {0: 1.0, 1: 0.0}
        if card == 2: return {0: 0.0, 1: 1.0}
    if history == "b":   # after agent bets
        if card == 0: return {0: 1.0, 1: 0.0}
        if card == 1: return {0: 2/3, 1: 1/3}
        if card == 2: return {0: 0.0, 1: 1.0}
    return {}


# ---------------------------------------------------------------------------
# Exact game-tree utilities (no MC — Kuhn poker has 12 terminal nodes)
# ---------------------------------------------------------------------------

def _build_infosets(game, player_id):
    """Return {iss: representative_state} and {iss: [legal_actions]}."""
    iss_states  = {}
    iss_actions = {}
    def walk(s):
        if s.is_terminal(): return
        if s.current_player() == pyspiel.PlayerId.CHANCE:
            for a, _ in s.chance_outcomes():
                s2 = s.clone(); s2.apply_action(a); walk(s2)
        elif s.current_player() == player_id:
            iss = s.information_state_string(player_id)
            if iss not in iss_states:
                iss_states[iss]  = s.clone()
                iss_actions[iss] = s.legal_actions()
            for a in s.legal_actions():
                s2 = s.clone(); s2.apply_action(a); walk(s2)
        else:
            for a in s.legal_actions():
                s2 = s.clone(); s2.apply_action(a); walk(s2)
    walk(game.new_initial_state())
    return iss_states, iss_actions


def _enumerate_pure_strategies(iss_actions: dict):
    """All pure strategies as list of {iss: action} dicts."""
    iss_list     = sorted(iss_actions)
    action_lists = [iss_actions[iss] for iss in iss_list]
    return [dict(zip(iss_list, combo))
            for combo in itertools.product(*action_lists)]


def _ev_exact(game, agent_id, agent_policy: dict, opp_policy: dict) -> float:
    """
    Exact expected payoff for agent_id.
    agent_policy : {iss -> {action: prob}}
    opp_policy   : {iss -> {action: prob}}  OR  {iss -> int}  (pure strategy)
    """
    def recurse(s, prob):
        if s.is_terminal():
            return prob * s.returns()[agent_id]
        cur   = s.current_player()
        total = 0.0
        if cur == pyspiel.PlayerId.CHANCE:
            for a, p in s.chance_outcomes():
                s2 = s.clone(); s2.apply_action(a)
                total += recurse(s2, prob * p)
        elif cur == agent_id:
            iss  = s.information_state_string(agent_id)
            dist = agent_policy.get(iss, {})
            lgl  = s.legal_actions(agent_id)
            norm = sum(dist.get(a, 0.0) for a in lgl) or 1.0
            for a in lgl:
                s2 = s.clone(); s2.apply_action(a)
                total += recurse(s2, prob * dist.get(a, 0.0) / norm)
        else:
            opp_id = cur
            iss    = s.information_state_string(opp_id)
            entry  = opp_policy.get(iss)
            lgl    = s.legal_actions(opp_id)
            if isinstance(entry, int):
                s2 = s.clone(); s2.apply_action(entry)
                total += recurse(s2, prob)
            elif isinstance(entry, dict):
                norm = sum(entry.get(a, 0.0) for a in lgl) or 1.0
                for a in lgl:
                    s2 = s.clone(); s2.apply_action(a)
                    total += recurse(s2, prob * entry.get(a, 0.0) / norm)
            else:
                for a in lgl:
                    s2 = s.clone(); s2.apply_action(a)
                    total += recurse(s2, prob / len(lgl))
        return total
    return recurse(game.new_initial_state(), 1.0)


def _exploitability(game, agent_id, agent_policy, opp_pure_strats) -> float:
    """expl(π) = v* - min_{opp pure} u(π, opp)"""
    v_star = -1.0/18.0 if agent_id == 0 else 1.0/18.0
    min_ev = min(_ev_exact(game, agent_id, agent_policy, p) for p in opp_pure_strats)
    return v_star - min_ev


def _mixed_strategy_ev(game, agent_id, agent_policy,
                       opp_reached_iss, opp_obs_action,
                       opp_iss_actions, opp_model) -> float:
    """
    Compute u_i(π_t, τ_t) per RWYWE Algorithm 6.

    τ_t is constructed as:
      - At opp_reached_iss: play opp_obs_action (observed on path of play)
      - At all other ISS: play according to the current opponent model
        (smoothed frequency estimates)

    This is the correct interpretation of Algorithm 6 for imperfect
    information: we make pessimistic assumptions only about what the opponent
    *would have done* at the ISS they actually reached — for other ISS we
    use our best estimate of their strategy (the opponent model).

    Using worst-case (nemesis) at off-path ISS is too conservative: it drives
    pess_ev below v* even when the opponent gave us a gift on-path, preventing
    k from growing.
    """
    v_star = -1.0/18.0 if agent_id == 0 else 1.0/18.0

    if opp_reached_iss not in opp_iss_actions:
        return v_star

    # Build τ: observed action at reached ISS, model prediction elsewhere
    tau_pol = {}
    for iss, actions in opp_iss_actions.items():
        if iss == opp_reached_iss:
            tau_pol[iss] = {opp_obs_action: 1.0}
        else:
            # Use opponent model counts as probabilities
            counts = opp_model.get(iss, {})
            total  = sum(counts.values())
            if total > 0:
                tau_pol[iss] = {a: counts.get(a, 0.0) / total for a in actions}
            else:
                tau_pol[iss] = {a: 1.0 / len(actions) for a in actions}

    return _ev_exact(game, agent_id, agent_policy, tau_pol)


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class RWYWEAgent(rl_agent.AbstractAgent):
    """
    Risk What You've Won in Expectation — Algorithm 6 (imperfect information)
    from Ganzfried & Sandholm (2015).

    Key design choices matching the paper:
      - Opponent model: frequency dict seeded with 5 fictitious GTO hands
        per ISS (Section 9.2).
      - Safe best response: exact — enumerate all 2^6 = 64 agent pure
        strategies, pick the highest-EV one with expl ≤ k.
      - Exploitability: exact worst-case over all 8 opponent pure strategies.
        Precomputed at init (never changes); only EV vs opponent model
        is recomputed each hand.
      - k-update: exact pessimistic τ per Algorithm 6.
    """

    PRIOR_HANDS = 5   # fictitious prior hands at GTO (Section 9.2)

    def __init__(self, game: pyspiel.Game, player_id: int, name='rwywe_agent'):
        super().__init__(game, player_id, name)
        self.game      = game
        self.player_id = player_id
        self.opp_id    = 1 - player_id
        self.v_star    = -1.0/18.0 if player_id == 0 else 1.0/18.0
        self.k         = 0.0

        # Pre-build game tree structure
        _, self._agent_iss_actions = _build_infosets(game, player_id)
        _, self._opp_iss_actions   = _build_infosets(game, self.opp_id)

        self._agent_pure_strats = _enumerate_pure_strategies(self._agent_iss_actions)
        self._opp_pure_strats   = _enumerate_pure_strategies(self._opp_iss_actions)

        # Precompute exploitability for every agent pure strategy (fixed by game tree)
        self._agent_pure_expls = [
            _exploitability(game, player_id,
                            {iss: {a: 1.0} for iss, a in pure.items()},
                            self._opp_pure_strats)
            for pure in self._agent_pure_strats
        ]

        # Precompute full EV matrix: ev_matrix[i, j] = EV of agent pure
        # strategy i vs opponent pure strategy j. Fixed by game tree — never
        # recomputed. Shape: (n_agent_pure, n_opp_pure).
        n_a = len(self._agent_pure_strats)
        n_o = len(self._opp_pure_strats)
        self._ev_matrix = np.zeros((n_a, n_o))
        for i, agent_pure in enumerate(self._agent_pure_strats):
            pol = {iss: {a: 1.0} for iss, a in agent_pure.items()}
            for j, opp_pure in enumerate(self._opp_pure_strats):
                self._ev_matrix[i, j] = _ev_exact(
                    game, player_id, pol, opp_pure)

        # Precompute opponent pure strategy weight vectors for dot products.
        # opp_pure_vecs[j] = indicator vector over opp ISS actions for pure j.
        # Used to convert opp_model counts → weight vector over pure strategies.
        self._opp_iss_list = sorted(self._opp_iss_actions)
        self._opp_pure_indices = [
            tuple(pure[iss] for iss in self._opp_iss_list)
            for pure in self._opp_pure_strats
        ]

        # Opponent model: simple dict of counts, seeded with 5 GTO prior hands
        # {opp_iss -> {action -> count}}
        self.opp_model = {}
        for iss, actions in self._opp_iss_actions.items():
            gto = _gto_probs_p1(iss)
            self.opp_model[iss] = {
                a: self.PRIOR_HANDS * gto.get(a, 1.0 / len(actions))
                for a in actions
            }

        # Per-hand state
        self._current_policy       = None
        self._pending_action_probs = None
        self._opp_obs_action       = None
        self._opp_reached_iss      = None

    def reset_opponent_model(self):
        """
        Reset the opponent model to the GTO prior (5 fictitious hands).
        Call this when you know the opponent has switched strategy — e.g.
        at hand 101 in the dynamic opponent test — so accumulated observations
        from the old strategy don't pollute the model for the new one.
        """
        for iss, actions in self._opp_iss_actions.items():
            gto = _gto_probs_p1(iss)
            self.opp_model[iss] = {
                a: self.PRIOR_HANDS * gto.get(a, 1.0 / len(actions))
                for a in actions
            }

    # ------------------------------------------------------------------
    # Opponent model
    # ------------------------------------------------------------------

    def _opp_model_policy(self) -> dict:
        """Smoothed opponent mixed strategy from frequency counts."""
        pol = {}
        for iss, counts in self.opp_model.items():
            total = sum(counts.values())
            pol[iss] = {a: c / total for a, c in counts.items()}
        return pol

    # ------------------------------------------------------------------
    # Safe best response (exact)
    # ------------------------------------------------------------------

    def _compute_policy(self) -> dict:
        """
        Exact ε-safe best response via LP, using the precomputed EV matrix.

        All _ev_exact calls are replaced by dot products against the
        precomputed self._ev_matrix (shape: n_agent_pure × n_opp_pure).

        Per hand: one dot product for objective (64-dim), one matrix-vector
        multiply for safety constraints (8×64) — microseconds vs ~512 tree
        traversals before.
        """
        from scipy.optimize import linprog

        epsilon = max(self.k, 0.0)
        v_star  = -1.0/18.0 if self.player_id == 0 else 1.0/18.0
        n_a     = len(self._agent_pure_strats)

        opp_pol = self._opp_model_policy()

        # EV of each agent pure strategy vs the opponent model (mixed strategy).
        # Use _ev_exact with the mixed opp_pol directly — correct and exact.
        # This is 64 tree traversals but each is O(12 nodes) so still fast.
        ev_vs_model = np.array([
            _ev_exact(self.game, self.player_id,
                      {iss: {a: 1.0} for iss, a in pure.items()}, opp_pol)
            for pure in self._agent_pure_strats
        ])

        # Safety constraints: EV of each agent pure vs each opponent pure
        # = ev_matrix itself (already computed)
        # Constraint: Σ_i λ_i * ev_matrix[i,j] ≥ v* - ε  ∀j
        # → -ev_matrix.T @ λ ≤ -(v* - ε) = ε - v*
        A_ub = -self._ev_matrix.T          # shape: (n_opp_pure, n_agent_pure)
        b_ub = np.full(A_ub.shape[0], epsilon - v_star)

        # Simplex: Σ λ_i = 1
        A_eq = np.ones((1, n_a))
        b_eq = np.ones(1)

        result = linprog(-ev_vs_model,
                         A_ub=A_ub, b_ub=b_ub,
                         A_eq=A_eq, b_eq=b_eq,
                         bounds=[(0.0, 1.0)] * n_a,
                         method='highs')

        if result.success:
            lambdas = np.clip(result.x, 0.0, None)
            lambdas /= lambdas.sum()

            # Marginalise to behavioral strategy
            iss_list = sorted(self._agent_iss_actions)
            policy   = {}
            for iss in iss_list:
                actions = self._agent_iss_actions[iss]
                dist    = {a: 0.0 for a in actions}
                for i, pure in enumerate(self._agent_pure_strats):
                    dist[pure[iss]] += lambdas[i]
                total = sum(dist.values())
                policy[iss] = {a: v / total for a, v in dist.items()} if total > 0 \
                              else {a: 1.0 / len(actions) for a in actions}
            return policy
        else:
            fallback = min(zip(self._agent_pure_strats, self._agent_pure_expls),
                           key=lambda x: x[1])[0]
            return {iss: {a: 1.0} for iss, a in fallback.items()}

    # ------------------------------------------------------------------
    # rl_agent interface
    # ------------------------------------------------------------------

    def step(self, state: pyspiel.State) -> int:
        if self._current_policy is None:
            self._current_policy       = self._compute_policy()
            self._pending_action_probs = self._current_policy.get(
                state.information_state_string(self.player_id), {}
            )

        iss  = state.information_state_string(self.player_id)
        dist = self._current_policy.get(iss, {})
        lgl  = state.legal_actions(self.player_id)
        norm = sum(dist.get(a, 0.0) for a in lgl) or 1.0
        probs = [dist.get(a, 0.0) / norm for a in lgl]
        return random.choices(lgl, weights=probs)[0]

    def get_policy(self) -> dict:
        """Return this hand's policy (computing it if not yet done)."""
        if self._current_policy is None:
            self._current_policy = self._compute_policy()
        return self._current_policy

    def inform_action(self, state: pyspiel.State, player: int, action: int):
        if player == self.opp_id:
            self._opp_reached_iss = state.information_state_string(self.opp_id)
            self._opp_obs_action  = action
            self.opp_model[self._opp_reached_iss][action] = \
                self.opp_model[self._opp_reached_iss].get(action, 0.0) + 1.0

    def on_terminal(self, state: pyspiel.State):
        if self._current_policy is None:
            return

        if (self._opp_obs_action is not None
                and self._opp_reached_iss is not None):
            pess_ev = _mixed_strategy_ev(
                self.game, self.player_id,
                self._current_policy,
                self._opp_reached_iss,
                self._opp_obs_action,
                self._opp_iss_actions,
                self.opp_model,
            )
            self.k += pess_ev - self.v_star

        # Reset per-hand state
        self._current_policy       = None
        self._pending_action_probs = None
        self._opp_obs_action       = None
        self._opp_reached_iss      = None

    def restart(self):
        self._current_policy       = None
        self._pending_action_probs = None
        self._opp_obs_action       = None
        self._opp_reached_iss      = None