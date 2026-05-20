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
                       opp_iss_actions) -> float:
    """
    Compute u_i(π_t, τ_t) per RWYWE Algorithm 6:

        u_i(π_t_i, a^t_{-i})

    This is the expected payoff of the agent's FULL mixed strategy π_t
    against the opponent's observed action a^t_{-i}, averaged over all
    chance outcomes (card deals).

    For the ISS the opponent actually reached, τ plays the observed action.
    For all other opponent ISS (off the path of play for this specific hand,
    but reachable under other card deals), we make the pessimistic assumption
    that τ plays a best response to π_t — i.e. minimises agent EV.

    This is the key difference from RWYW:
      - RWYW: uses the realized payoff of the action actually taken
      - RWYWE: uses the expected payoff of π_t against τ, integrated over
               all card deals (the expectation over agent's randomisation)

    Implementation:
      - At opp_reached_iss: fix action = opp_obs_action
      - At all other opp ISS: for each ISS independently, pick the action
        that minimises agent EV given π_t (pessimistic off-path assumption)
    """
    v_star = -1.0/18.0 if agent_id == 0 else 1.0/18.0

    if opp_reached_iss not in opp_iss_actions:
        return v_star

    # Build τ: observed action on-path, pessimistic best response off-path.
    # For each off-path ISS, find the opponent action that minimises agent EV
    # when the opponent plays deterministically at that ISS (all else uniform).
    opp_id  = 1 - agent_id
    tau_pol = {}

    for iss, actions in opp_iss_actions.items():
        if iss == opp_reached_iss:
            tau_pol[iss] = {opp_obs_action: 1.0}
        else:
            # Pessimistic: pick the action minimising agent EV at this ISS,
            # holding all other ISS at uniform (we'll fill in properly below)
            tau_pol[iss] = {a: 1.0 / len(actions) for a in actions}

    # Now refine off-path ISS actions pessimistically:
    # For each off-path ISS, try each action and pick the one giving lowest
    # agent EV when substituted into the full tau_pol.
    for iss in opp_iss_actions:
        if iss == opp_reached_iss:
            continue
        actions = opp_iss_actions[iss]
        if len(actions) == 1:
            tau_pol[iss] = {actions[0]: 1.0}
            continue
        best_a_for_opp = None
        best_ev_for_opp = float("inf")
        for a in actions:
            tau_pol[iss] = {a: 1.0}
            ev = _ev_exact(game, agent_id, agent_policy, tau_pol)
            if ev < best_ev_for_opp:
                best_ev_for_opp = ev
                best_a_for_opp  = a
        tau_pol[iss] = {best_a_for_opp: 1.0}

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
        ε-safe best response to the current opponent model.
        Enumerates all 64 agent pure strategies; picks highest EV among
        those with precomputed expl ≤ max(k, 0).
        Returns policy as {iss -> {action: prob}}.
        """
        epsilon       = max(self.k, 0.0)
        opp_pol       = self._opp_model_policy()
        best_pure     = None
        best_ev       = -float("inf")
        fallback      = None
        fallback_expl = float("inf")

        for pure, expl in zip(self._agent_pure_strats, self._agent_pure_expls):
            policy = {iss: {a: 1.0} for iss, a in pure.items()}
            ev     = _ev_exact(self.game, self.player_id, policy, opp_pol)
            if expl <= epsilon + 1e-9:
                if ev > best_ev:
                    best_ev   = ev
                    best_pure = pure
            if expl < fallback_expl:
                fallback_expl = expl
                fallback      = pure

        chosen = best_pure if best_pure is not None else fallback
        return {iss: {a: 1.0} for iss, a in chosen.items()}

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