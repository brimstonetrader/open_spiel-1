"""
rwyw_leduc.py — Risk What You've Won (Algorithm 1, Ganzfried & Sandholm 2015)
             adapted for Leduc poker.

  k_1 = 0
  for each hand t:
    pi_t = argmax_{pi in SAFE(max(k_t,0))} M(pi)
    play action a_t according to pi_t
    update M with opponent's observed action
    k_{t+1} = k_t + u(a_t, opp_action) - v*

Key differences from Kuhn version:
  - Equilibrium solved via 10000 iterations of CFR+ (no closed form)
  - GTO prior for opponent model read from CFR+ solution
  - LP variables = one prob per agent ISS (same structure, more variables)
  - EV linearity in behavioral probs still holds exactly in Leduc
"""

import itertools, random
import numpy as np
import pyspiel
from open_spiel.python import rl_agent
from open_spiel.python.algorithms import cfr
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Game tree utilities  (identical to Kuhn version)
# ---------------------------------------------------------------------------

def _all_infosets(game, player_id):
    """Return {iss: [legal_actions]} for player_id."""
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
    """Exact EV by full game-tree traversal."""
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


# ---------------------------------------------------------------------------
# CFR+ equilibrium solver
# ---------------------------------------------------------------------------

def solve_equilibrium(game, iterations=10000):
    """
    Run CFR+ for *iterations* steps and return:
      - v_star     : game value for player 0
      - policy_p0  : {iss -> {action -> prob}} for player 0
      - policy_p1  : {iss -> {action -> prob}} for player 1
    """
    print(f"  Solving Leduc equilibrium ({iterations} CFR+ iterations)...",
          flush=True)
    solver = cfr.CFRPlusSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()

    avg = solver.average_policy()

    def extract(player_id, iss_actions):
        pol = {}
        for iss, acts in iss_actions.items():
            # Build a representative state to query the policy
            # We use action_probabilities via the tabular policy directly
            probs = {}
            for a in acts:
                probs[a] = avg.action_probability(iss, a) if hasattr(avg, 'action_probability') \
                           else avg.policy_for_key(iss).get(a, 1.0/len(acts))
            norm = sum(probs.values()) or 1.0
            pol[iss] = {a: v/norm for a, v in probs.items()}
        return pol

    # Compute v_star by evaluating equilibrium policies against each other
    agent_iss = _all_infosets(game, 0)
    opp_iss   = _all_infosets(game, 1)

    # Extract policies by walking the tree and querying the average policy
    pol0, pol1 = {}, {}
    def walk(s):
        if s.is_terminal(): return
        if s.current_player() == pyspiel.PlayerId.CHANCE:
            for a, _ in s.chance_outcomes():
                s2 = s.clone(); s2.apply_action(a); walk(s2)
        else:
            pid  = s.current_player()
            iss  = s.information_state_string(pid)
            dist = avg.action_probabilities(s, pid)
            (pol0 if pid == 0 else pol1)[iss] = dict(dist)
            for a in s.legal_actions():
                s2 = s.clone(); s2.apply_action(a); walk(s2)
    walk(game.new_initial_state())

    v_star = _ev(game, 0, pol0, pol1)
    print(f"  Leduc equilibrium value (player 0): {v_star:.6f}")
    return v_star, pol0, pol1


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RWYWAgent(rl_agent.AbstractAgent):
    """
    Risk What You've Won agent for Leduc poker.

    Equilibrium computed via CFR+ at construction time.
    LP safe best response over behavioral strategy variables (one per ISS).
    Opponent model seeded with PRIOR fictitious hands at CFR+ equilibrium.
    """

    PRIOR = 5

    def __init__(self, game, player_id, name="rwyw_leduc",
                 cfr_iters=10000, _shared_eq=None):
        super().__init__(game, player_id, name)
        self.game, self.player_id = game, player_id
        self.opp_id = 1 - player_id

        # Solve equilibrium (or reuse shared solution for speed)
        if _shared_eq is not None:
            self.v_star, pol0, pol1 = _shared_eq
        else:
            self.v_star, pol0, pol1 = solve_equilibrium(game, cfr_iters)

        self._eq_policy     = pol0 if player_id == 0 else pol1
        self._eq_opp_policy = pol1 if player_id == 0 else pol0
        self.k = 0.0

        self._agent_iss = _all_infosets(game, player_id)
        self._opp_iss   = _all_infosets(game, self.opp_id)
        self._opp_pure_strats, _ = _pure_strategies(self._opp_iss)
        self._iss_list = sorted(self._agent_iss)
        self.n = len(self._iss_list)

        print(f"  Agent ISS: {self.n},  Opp pure strategies: {len(self._opp_pure_strats)}")

        # Precompute EV linear coefficients:
        #   EV(p, opp_j) = ev0[j] + grad[:,j] @ p
        # where p[i] = prob(action index 1) at ISS i.
        # Assumes exactly 2 legal actions per agent ISS.
        # For Leduc: some ISS have 3 actions (fold/call/raise).
        # We generalise: p is a flat vector of ALL action probs,
        # with one free variable per (ISS, action) pair minus one per ISS
        # (the last action's prob = 1 - sum of others).
        # Simpler: use the full prob vector with a sum-to-1 equality per ISS.
        self._build_lp_coefficients()

        # Opponent model seeded with CFR+ equilibrium prior
        self.M = {}
        for iss, acts in self._opp_iss.items():
            eq_dist = self._eq_opp_policy.get(iss, {a: 1.0/len(acts) for a in acts})
            self.M[iss] = {a: self.PRIOR * eq_dist.get(a, 1.0/len(acts))
                           for a in acts}

        self._policy               = None
        self._pending_action_probs = None
        self._opp_action           = None
        self._opp_reached_iss      = None

    def _build_lp_coefficients(self):
        """
        Build flat LP variable index and precompute EV0 + gradient.

        Variables: x[k] = prob of action actions[k] at ISS var_iss[k].
        For each ISS with n_a actions, we have n_a free variables subject
        to sum-to-1 (equality constraint) and x >= 0.

        EV(x, opp_j) = ev0[j] + sum_k grad[k,j] * x[k]  (linear in x).
        """
        # Build variable index: flat list of (iss, action) pairs
        self._var_index = []   # list of (iss, action)
        self._iss_var_range = {}  # iss -> (start, end) indices into _var_index
        for iss in self._iss_list:
            start = len(self._var_index)
            for a in self._agent_iss[iss]:
                self._var_index.append((iss, a))
            self._iss_var_range[iss] = (start, len(self._var_index))
        self.n_vars = len(self._var_index)
        m = len(self._opp_pure_strats)

        def pol_from_x(x):
            policy = {}
            for iss in self._iss_list:
                start, end = self._iss_var_range[iss]
                acts = self._agent_iss[iss]
                vals = x[start:end]
                norm = sum(vals) or 1.0
                policy[iss] = {a: v/norm for a, v in zip(acts, vals)}
            return policy

        # All-zero baseline
        x0 = [0.0] * self.n_vars
        self._ev0 = np.array([
            _ev(self.game, self.player_id, pol_from_x(x0), op)
            for op in self._opp_pure_strats
        ])  # (m,)

        # Gradient: dEV/dx[k] for each variable k vs each opp pure j
        self._grad = np.zeros((self.n_vars, m))
        for k in range(self.n_vars):
            ek = x0[:]; ek[k] = 1.0
            ev1 = np.array([
                _ev(self.game, self.player_id, pol_from_x(ek), op)
                for op in self._opp_pure_strats
            ])
            self._grad[k] = ev1 - self._ev0

        self._pol_from_x = pol_from_x

    def reset_opponent_model(self):
        for iss, acts in self._opp_iss.items():
            eq_dist = self._eq_opp_policy.get(iss, {a: 1.0/len(acts) for a in acts})
            self.M[iss] = {a: self.PRIOR * eq_dist.get(a, 1.0/len(acts))
                           for a in acts}

    def _M_policy(self):
        return {iss: {a: c/sum(cnt.values()) for a, c in cnt.items()}
                for iss, cnt in self.M.items()}

    def _safe_br(self):
        """
        LP: maximise EV(x, M) subject to EV(x, opp_j) >= v* - epsilon for all j,
        sum-to-1 per ISS, x >= 0.
        """
        epsilon = max(self.k, 0.0)

        # If k <= 0, play equilibrium
        if epsilon == 0.0:
            return dict(self._eq_policy)

        M_pol = self._M_policy()
        x0    = [0.0] * self.n_vars

        # Objective gradient vs M
        ev0_M = _ev(self.game, self.player_id, self._pol_from_x(x0), M_pol)
        c     = np.zeros(self.n_vars)
        for k in range(self.n_vars):
            ek    = x0[:]; ek[k] = 1.0
            c[k]  = -(_ev(self.game, self.player_id, self._pol_from_x(ek), M_pol) - ev0_M)

        # Safety constraints: ev0[j] + grad[:,j]@x >= v* - epsilon
        A_ub = -self._grad.T                            # (m, n_vars)
        b_ub = self._ev0 - self.v_star + epsilon        # (m,)

        # Sum-to-1 equality per ISS
        n_iss = len(self._iss_list)
        A_eq  = np.zeros((n_iss, self.n_vars))
        b_eq  = np.ones(n_iss)
        for i, iss in enumerate(self._iss_list):
            start, end = self._iss_var_range[iss]
            A_eq[i, start:end] = 1.0

        result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                         A_eq=A_eq, b_eq=b_eq,
                         bounds=[(0.0, 1.0)] * self.n_vars,
                         method='highs')

        if not result.success:
            return dict(self._eq_policy)

        return self._pol_from_x(np.clip(result.x, 0.0, None).tolist())

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
            self._opp_action      = action
            self._opp_reached_iss = iss
            self.M[iss][action]   = self.M[iss].get(action, 0.0) + 1.0

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