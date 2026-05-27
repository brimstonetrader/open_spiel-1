"""
rwyw.py — Risk What You've Won (Algorithm 1, Ganzfried & Sandholm 2015)

  k_1 = 0
  for each hand t:
    pi_t = argmax_{pi in SAFE(max(k_t,0))} M(pi)
    play action a_t according to pi_t
    update M with opponent's observed action
    k_{t+1} = k_t + u(a_t, opp_action) - v*
"""

import itertools, random
import numpy as np
import pyspiel
from open_spiel.python import rl_agent
from scipy.optimize import linprog

V_STAR = -1.0 / 18.0


def _gto_probs_p1(iss):
    card, hist = int(iss[0]), iss[1:]
    if hist == "p": return [{0:2/3,1:1/3},{0:1.0,1:0.0},{0:0.0,1:1.0}][card]
    if hist == "b": return [{0:1.0,1:0.0},{0:2/3,1:1/3},{0:0.0,1:1.0}][card]
    return {}


def _all_infosets(game, player_id):
    result = {}
    def walk(s):
        if s.is_terminal(): return
        if s.current_player() == pyspiel.PlayerId.CHANCE:
            for a, _ in s.chance_outcomes():
                s2 = s.clone(); s2.apply_action(a); walk(s2)
        elif s.current_player() == player_id:
            iss = s.information_state_string(player_id)
            if iss not in result:
                result[iss] = s.legal_actions()
            for a in s.legal_actions():
                s2 = s.clone(); s2.apply_action(a); walk(s2)
        else:
            for a in s.legal_actions():
                s2 = s.clone(); s2.apply_action(a); walk(s2)
    walk(game.new_initial_state())
    return result


def _pure_strategies(iss_actions):
    iss_list = sorted(iss_actions)
    return [dict(zip(iss_list, c))
            for c in itertools.product(*[iss_actions[i] for i in iss_list])], iss_list


def _ev(game, agent_id, agent_policy, opp_policy):
    def rec(s, p):
        if s.is_terminal(): return p * s.returns()[agent_id]
        cur, total = s.current_player(), 0.0
        if cur == pyspiel.PlayerId.CHANCE:
            for a, pr in s.chance_outcomes():
                s2 = s.clone(); s2.apply_action(a); total += rec(s2, p * pr)
        elif cur == agent_id:
            iss  = s.information_state_string(agent_id)
            dist = agent_policy.get(iss, {})
            lgl  = s.legal_actions(agent_id)
            norm = sum(dist.get(a, 0.0) for a in lgl) or 1.0
            for a in lgl:
                s2 = s.clone(); s2.apply_action(a)
                total += rec(s2, p * dist.get(a, 0.0) / norm)
        else:
            iss   = s.information_state_string(cur)
            entry = opp_policy.get(iss)
            lgl   = s.legal_actions(cur)
            if isinstance(entry, int):
                s2 = s.clone(); s2.apply_action(entry); total += rec(s2, p)
            elif isinstance(entry, dict):
                norm = sum(entry.get(a, 0.0) for a in lgl) or 1.0
                for a in lgl:
                    s2 = s.clone(); s2.apply_action(a)
                    total += rec(s2, p * entry.get(a, 0.0) / norm)
            else:
                for a in lgl:
                    s2 = s.clone(); s2.apply_action(a)
                    total += rec(s2, p / len(lgl))
        return total
    return rec(game.new_initial_state(), 1.0)


class RWYWAgent(rl_agent.AbstractAgent):
    """Risk What You've Won agent."""

    PRIOR = 5

    def __init__(self, game, player_id, name="rwyw_agent"):
        super().__init__(game, player_id, name)
        self.game, self.player_id = game, player_id
        self.opp_id = 1 - player_id
        self.v_star = V_STAR if player_id == 0 else -V_STAR
        self.k      = 0.0

        self._agent_iss = _all_infosets(game, player_id)
        self._opp_iss   = _all_infosets(game, self.opp_id)
        self._opp_pure_strats, _ = _pure_strategies(self._opp_iss)
        self._iss_list = sorted(self._agent_iss)
        self.n = len(self._iss_list)   # number of agent ISS

        # Precompute EV linear coefficients for the LP.
        # For each (ISS i, opponent pure j): ev0[j] and grad[i,j] where
        #   EV(p, opp_j) = ev0[j] + sum_i grad[i,j] * p[i]
        # p[i] = prob(action=1) at ISS i  (action index, not action value)
        # This holds exactly because EV is linear in behavioral strategy probs.
        n, m = self.n, len(self._opp_pure_strats)

        def pol(probs):
            return {iss: {self._agent_iss[iss][0]: 1.0 - p,
                          self._agent_iss[iss][1]: p}
                    for iss, p in zip(self._iss_list, probs)}

        p0 = [0.0] * n
        self._ev0 = np.array([_ev(game, player_id, pol(p0), op)
                               for op in self._opp_pure_strats])   # (m,)
        self._grad = np.zeros((n, m))
        for i in range(n):
            ei = p0[:]; ei[i] = 1.0
            ev1 = np.array([_ev(game, player_id, pol(ei), op)
                             for op in self._opp_pure_strats])
            self._grad[i] = ev1 - self._ev0   # (m,)

        # Opponent model seeded with GTO prior
        self.M = {iss: {a: self.PRIOR * _gto_probs_p1(iss).get(a, 1.0/len(acts))
                        for a in acts}
                  for iss, acts in self._opp_iss.items()}

        self._policy               = None
        self._pending_action_probs = None
        self._opp_action           = None
        self._opp_reached_iss      = None

    def reset_opponent_model(self):
        self.M = {iss: {a: self.PRIOR * _gto_probs_p1(iss).get(a, 1.0/len(acts))
                        for a in acts}
                  for iss, acts in self._opp_iss.items()}

    def _M_policy(self):
        return {iss: {a: c/sum(cnt.values()) for a, c in cnt.items()}
                for iss, cnt in self.M.items()}

    def _gto_policy(self, alpha=1/3):
        """Player 0 GTO strategy parameterised by alpha (Kuhn 1950)."""
        policy = {}
        for iss in self._iss_list:
            card, hist = int(iss[0]), iss[1:]
            if hist == "":
                p_bet = [alpha/3, 0.0, alpha][card]
            else:   # "pb" — facing a bet after check
                p_bet = [0.0, alpha/3 + 1/3, 1.0][card]
            policy[iss] = {self._agent_iss[iss][0]: 1.0-p_bet,
                           self._agent_iss[iss][1]: p_bet}
        return policy

    def _safe_br(self):
        """
        LP over behavioral strategy variables p[i] = prob(action=1) at ISS i.

        EV(p, opp_j) = ev0[j] + grad[:,j] @ p   (precomputed, exactly linear)

        Maximise EV vs M subject to EV vs each opp pure >= v* - epsilon.
        """
        epsilon = max(self.k, 0.0)
        if epsilon == 0.0:
            return self._gto_policy()

        M_pol    = self._M_policy()
        # Opponent model as a weight vector over pure strategies (product dist)
        # EV vs M = ev0 @ w + grad @ p @ w  where w[j] = prod_iss M[iss][pure[j][iss]]
        # But easier: compute EV(p=0, M) and gradient directly
        p0       = [0.0] * self.n
        def pol(probs):
            return {iss: {self._agent_iss[iss][0]: 1.0-p, self._agent_iss[iss][1]: p}
                    for iss, p in zip(self._iss_list, probs)}

        ev0_M = _ev(self.game, self.player_id, pol(p0), M_pol)
        c     = np.zeros(self.n)
        for i in range(self.n):
            ei    = p0[:]; ei[i] = 1.0
            c[i]  = -(_ev(self.game, self.player_id, pol(ei), M_pol) - ev0_M)

        # Safety: EV(p, opp_j) >= v* - epsilon
        # ev0[j] + grad[:,j]@p >= v* - epsilon
        # => -grad[:,j]@p <= ev0[j] - v* + epsilon
        A_ub = -self._grad.T                                      # (m, n)
        b_ub = self._ev0 - self.v_star + epsilon                  # (m,)

        result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                         bounds=[(0.0, 1.0)] * self.n, method='highs')

        if not result.success:
            return self._gto_policy()

        return pol(np.clip(result.x, 0.0, 1.0).tolist())

    def get_policy(self):
        if self._policy is None:
            self._policy = self._safe_br()
        return self._policy

    def step(self, state):
        if self._policy is None:
            self._policy = self._safe_br()
        iss  = state.information_state_string(self.player_id)
        dist = self._policy.get(iss, {})
        lgl  = state.legal_actions(self.player_id)
        norm = sum(dist.get(a, 0.0) for a in lgl) or 1.0
        action = random.choices(lgl, weights=[dist.get(a, 0.0)/norm for a in lgl])[0]
        self._pending_action_probs = {a: dist.get(a, 0.0)/norm for a in lgl}
        return action

    def inform_action(self, state, player, action):
        if player == self.opp_id:
            iss = state.information_state_string(self.opp_id)
            self._opp_action = action
            self._opp_reached_iss = iss
            self.M[iss][action] = self.M[iss].get(action, 0.0) + 1.0

    def on_terminal(self, state):
        if self._policy is None:
            return
        self.k += state.returns()[self.player_id] - self.v_star
        self._policy               = None
        self._pending_action_probs = None
        self._opp_action           = None
        self._opp_reached_iss      = None

    def restart(self):
        self._policy               = None
        self._pending_action_probs = None
        self._opp_action           = None
        self._opp_reached_iss      = None