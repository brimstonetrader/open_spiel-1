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
  - Equilibrium solved via CFR+ (no closed form for Leduc)
  - LP formulated in SEQUENCE FORM, not behavioral strategy space.
    Behavioral strategies make EV multilinear (up to degree 4 in Leduc);
    sequence-form strategies make EV bilinear, so the LP is exact.
  - Safety constraint: min_{q1} EV(q0, q1) >= v* - k, enforced via
    LP duality — dual variables y are added to the joint LP so that
    strong duality converts the inner min into a linear constraint.
    This avoids enumerating opponent pure strategies (3^468 in Leduc).
  - LP size: ~1093 agent + ~469 dual variables, ~1093 inequality + ~469
    equality constraints. HiGHS solves it in ~70ms.
  - Opponent model: M[iss][action] = pseudo-count; seeded from CFR+ prior.
"""

import random
import numpy as np
import pyspiel
from open_spiel.python import rl_agent
from open_spiel.python.algorithms import cfr
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Game tree utilities
# ---------------------------------------------------------------------------

def _ev(game, agent_id, agent_policy, opp_policy):
    """
    Exact EV by full game-tree traversal. Used only for v_star computation.

    agent_policy : {iss -> {action -> prob}}
    opp_policy   : {iss -> {action -> prob}}
    """
    def _dist_probs(policy, iss, lgl):
        """Handle both {action: prob} dicts and pure int actions."""
        entry = policy.get(iss)
        if isinstance(entry, int):
            return {a: (1.0 if a == entry else 0.0) for a in lgl}
        dist = entry or {}
        norm = sum(dist.get(a, 0.0) for a in lgl) or 1.0
        return {a: dist.get(a, 0.0) / norm for a in lgl}

    def rec(s, p):
        if s.is_terminal(): return p * s.returns()[agent_id]
        cur, total = s.current_player(), 0.0
        if cur == pyspiel.PlayerId.CHANCE:
            for a, pr in s.chance_outcomes():
                s2 = s.clone(); s2.apply_action(a); total += rec(s2, p * pr)
        elif cur == agent_id:
            iss = s.information_state_string(agent_id)
            lgl = s.legal_actions(agent_id)
            for a, prob in _dist_probs(agent_policy, iss, lgl).items():
                if prob > 0:
                    s2 = s.clone(); s2.apply_action(a)
                    total += rec(s2, p * prob)
        else:
            iss = s.information_state_string(cur)
            lgl = s.legal_actions(cur)
            for a, prob in _dist_probs(opp_policy, iss, lgl).items():
                if prob > 0:
                    s2 = s.clone(); s2.apply_action(a)
                    total += rec(s2, p * prob)
        return total
    return rec(game.new_initial_state(), 1.0)


# ---------------------------------------------------------------------------
# CFR+ equilibrium solver
# ---------------------------------------------------------------------------

def solve_equilibrium(game, iterations=200):
    """
    Run CFR+ for *iterations* steps and return:
      v_star   : game value for player 0
      policy_p0: {iss -> {action -> prob}} for player 0
      policy_p1: {iss -> {action -> prob}} for player 1
    """
    print(f"  Solving Leduc equilibrium ({iterations} CFR+ iterations)...",
          flush=True)
    solver = cfr.CFRPlusSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()

    avg  = solver.average_policy()
    pol0, pol1 = {}, {}

    def walk(s):
        if s.is_terminal(): return
        if s.current_player() == pyspiel.PlayerId.CHANCE:
            for a, _ in s.chance_outcomes():
                s2 = s.clone(); s2.apply_action(a); walk(s2)
        else:
            pid = s.current_player()
            iss = s.information_state_string(pid)
            dist = avg.action_probabilities(s, pid)
            (pol0 if pid == 0 else pol1)[iss] = dict(dist)
            for a in s.legal_actions():
                s2 = s.clone(); s2.apply_action(a); walk(s2)

    walk(game.new_initial_state())
    v_star = _ev(game, 0, pol0, pol1)
    print(f"  Leduc equilibrium value (player 0): {v_star:.6f}")
    return v_star, pol0, pol1


# ---------------------------------------------------------------------------
# Sequence form data
# ---------------------------------------------------------------------------

def build_sequence_form(game, agent_id):
    """
    Build the sequence-form representation for agent_id.

    A sequence is a root-to-node path of (iss, action) pairs for agent_id.
    The empty tuple () is the root sequence with q[root] = 1.

    Returns a dict with:
      agent_seqs     : list of sequences (each a tuple of (iss,action) pairs)
      agent_seq_map  : {(iss, action) -> child_seq_index}
      agent_iss_par  : {iss -> parent_seq_index}
      agent_iss_acts : {iss -> [legal_actions]}
      opp_seqs, opp_seq_map, opp_iss_par, opp_iss_acts : same for opponent
      A              : EV matrix (n_agent_seqs, n_opp_seqs), A[s0,s1] = sum_terminals pc*r
      F_a, f_a       : sequence constraint matrix/rhs for agent   (F_a q0 = f_a)
      F_o, f_o       : sequence constraint matrix/rhs for opponent (F_o q1 = f_o)
    """
    ag_seqs = [()];  ag_map = {(): 0};  ag_par = {};  ag_acts = {}
    op_seqs = [()];  op_map = {(): 0};  op_par = {};  op_acts = {}
    ev_entries = []

    def walk(s, pc, aseq, oseq):
        if s.is_terminal():
            r = s.returns()[agent_id]
            if r != 0:
                ev_entries.append((aseq, oseq, pc, r))
            return
        cur = s.current_player()
        if cur == pyspiel.PlayerId.CHANCE:
            for a, pr in s.chance_outcomes():
                s2 = s.clone(); s2.apply_action(a)
                walk(s2, pc * pr, aseq, oseq)
        elif cur == agent_id:
            iss = s.information_state_string(agent_id)
            if iss not in ag_par:
                ag_par[iss] = aseq
                ag_acts[iss] = s.legal_actions()
            for a in s.legal_actions():
                key = (iss, a)
                ns  = ag_map.get(key)
                if ns is None:
                    ns = len(ag_seqs)
                    ag_seqs.append(ag_seqs[aseq] + (key,))
                    ag_map[key] = ns
                s2 = s.clone(); s2.apply_action(a)
                walk(s2, pc, ns, oseq)
        else:
            iss = s.information_state_string(cur)
            if iss not in op_par:
                op_par[iss] = oseq
                op_acts[iss] = s.legal_actions()
            for a in s.legal_actions():
                key = (iss, a)
                ns  = op_map.get(key)
                if ns is None:
                    ns = len(op_seqs)
                    op_seqs.append(op_seqs[oseq] + (key,))
                    op_map[key] = ns
                s2 = s.clone(); s2.apply_action(a)
                walk(s2, pc, aseq, ns)

    walk(game.new_initial_state(), 1.0, 0, 0)

    n_a, n_o = len(ag_seqs), len(op_seqs)

    A = np.zeros((n_a, n_o))
    for s0, s1, pc, r in ev_entries:
        A[s0, s1] += pc * r

    def make_F(iss_par, iss_acts, seq_map, n):
        rows = len(iss_par) + 1
        F = np.zeros((rows, n))
        f = np.zeros(rows)
        F[0, 0] = 1.0;  f[0] = 1.0          # root: q[0] = 1
        for row, (iss, par) in enumerate(iss_par.items(), 1):
            F[row, par] = 1.0                 # q[parent] - sum q[children] = 0
            for a in iss_acts[iss]:
                c = seq_map.get((iss, a))
                if c is not None:
                    F[row, c] = -1.0
        return F, f

    F_a, f_a = make_F(ag_par, ag_acts, ag_map, n_a)
    F_o, f_o = make_F(op_par, op_acts, op_map, n_o)

    return dict(
        agent_seqs=ag_seqs, agent_seq_map=ag_map,
        agent_iss_par=ag_par, agent_iss_acts=ag_acts,
        opp_seqs=op_seqs, opp_seq_map=op_map,
        opp_iss_par=op_par, opp_iss_acts=op_acts,
        A=A, F_a=F_a, f_a=f_a, F_o=F_o, f_o=f_o,
    )


def pol_to_seqform(pol, iss_par, iss_acts, seq_map, n):
    """
    Convert a behavioral policy {iss -> {action -> prob}} to sequence-form
    vector q of length n.  q[s] = product of action probs along the sequence s.
    Processes info states in their sequence-index order, which is topological.
    """
    q = np.zeros(n)
    q[0] = 1.0
    for iss, par_seq in iss_par.items():
        dist = pol.get(iss, {})
        acts = iss_acts[iss]
        norm = sum(dist.get(a, 0.0) for a in acts) or 1.0
        for a in acts:
            c = seq_map.get((iss, a))
            if c is not None:
                q[c] = q[par_seq] * dist.get(a, 0.0) / norm
    return q


def seqform_to_pol(q, iss_par, iss_acts, seq_map):
    """
    Convert sequence-form vector back to behavioral policy.
    prob(iss, a) = q[child(iss,a)] / q[par(iss)]  (0 if parent unreached).
    """
    pol = {}
    for iss, par_seq in iss_par.items():
        acts = iss_acts[iss]
        q_par = q[par_seq]
        if q_par <= 0:
            pol[iss] = {a: 1.0/len(acts) for a in acts}
        else:
            pol[iss] = {}
            for a in acts:
                c = seq_map.get((iss, a))
                pol[iss][a] = (q[c] / q_par) if c is not None else 0.0
    return pol


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RWYWAgent(rl_agent.AbstractAgent):
    """
    Risk What You've Won agent for Leduc poker.

    Safe best-response LP in sequence form:

      max  (A q0)^T m           [EV vs opponent model m]
      s.t.
        F_a q0        = f_a     [agent sequence constraints]
        A^T q0 - F_o^T y >= 0  [dual feasibility for safety inner LP]
        f_o^T y       >= v*-k   [safety: min EV >= v* - k via strong duality]
        q0 >= 0
        y  unconstrained

    Variables: q0 (1093), y (469). Solved with HiGHS in ~70 ms.

    When k <= 0, fall back to the CFR+ equilibrium policy directly.
    """

    PRIOR = 5   # pseudo-count weight for equilibrium prior in opponent model

    def __init__(self, game, player_id, name="rwyw_leduc",
                 cfr_iters=200, _shared_eq=None):
        super().__init__(game, player_id, name)
        self.game, self.player_id = game, player_id
        self.opp_id = 1 - player_id

        if _shared_eq is not None:
            self.v_star, pol0, pol1 = _shared_eq
        else:
            self.v_star, pol0, pol1 = solve_equilibrium(game, cfr_iters)

        self._eq_pol_agent  = pol0 if player_id == 0 else pol1
        self._eq_pol_opp    = pol1 if player_id == 0 else pol0
        self._eq_opp_policy = self._eq_pol_opp   # alias expected by test harness
        self.k = 0.0

        # Build sequence form
        print(f"  Building sequence form...", flush=True)
        sf = build_sequence_form(game, player_id)
        self._sf = sf
        self.A   = sf['A']
        self.F_a = sf['F_a'];  self.f_a = sf['f_a']
        self.F_o = sf['F_o'];  self.f_o = sf['f_o']
        n_a = len(sf['agent_seqs'])
        n_o = len(sf['opp_seqs'])
        self._n_a    = n_a
        self._n_o    = n_o
        self._n_dual = len(self.f_o)

        # Equilibrium in sequence form (for fallback)
        self._q0_eq = pol_to_seqform(
            self._eq_pol_agent,
            sf['agent_iss_par'], sf['agent_iss_acts'], sf['agent_seq_map'], n_a)
        self._q1_eq = pol_to_seqform(
            self._eq_pol_opp,
            sf['opp_iss_par'], sf['opp_iss_acts'], sf['opp_seq_map'], n_o)

        print(f"  Agent seqs: {n_a}, Opp seqs: {n_o}, Dual vars: {self._n_dual}",
              flush=True)

        # Opponent model: M[iss][action] = pseudo-count
        opp_iss_acts = sf['opp_iss_acts']
        self.M = {}
        for iss, acts in opp_iss_acts.items():
            eq_dist = self._eq_pol_opp.get(iss, {a: 1.0/len(acts) for a in acts})
            self.M[iss] = {a: self.PRIOR * eq_dist.get(a, 1.0/len(acts))
                           for a in acts}

        self._policy               = None
        self._pending_action_probs = None
        self._opp_action           = None
        self._opp_reached_iss      = None

    # ------------------------------------------------------------------
    # Opponent model
    # ------------------------------------------------------------------

    def _M_seqform(self):
        """
        Convert opponent model M (pseudo-counts) to a sequence-form vector m.
        m is a normalised probability distribution over opponent sequences.
        """
        sf  = self._sf
        pol = {}
        for iss, cnt in self.M.items():
            total = sum(cnt.values()) or 1.0
            pol[iss] = {a: c/total for a, c in cnt.items()}
        return pol_to_seqform(
            pol,
            sf['opp_iss_par'], sf['opp_iss_acts'], sf['opp_seq_map'], self._n_o)

    def reset_opponent_model(self):
        sf = self._sf
        for iss, acts in sf['opp_iss_acts'].items():
            eq_dist = self._eq_pol_opp.get(iss, {a: 1.0/len(acts) for a in acts})
            self.M[iss] = {a: self.PRIOR * eq_dist.get(a, 1.0/len(acts))
                           for a in acts}

    # ------------------------------------------------------------------
    # LP solver
    # ------------------------------------------------------------------

    def _safe_br(self):
        """
        Solve the joint sequence-form LP and return a behavioral policy.
        Falls back to equilibrium when k <= 0.
        """
        epsilon = max(self.k, 0.0)
        if epsilon == 0.0:
            return dict(self._eq_pol_agent)

        m      = self._M_seqform()
        n_a    = self._n_a
        n_dual = self._n_dual
        n_vars = n_a + n_dual

        # Objective: max q0^T (A m) => minimise -(A m)^T q0
        c = np.empty(n_vars)
        c[:n_a]  = -(self.A @ m)
        c[n_a:]  = 0.0

        # Inequality constraints (A_ub x <= b_ub):
        #   F_o^T y - A^T q0 <= 0   (dual feasibility)
        #   -f_o^T y          <= -(v* - epsilon)  (safety)
        n_ineq = self._n_o + 1
        A_ub = np.zeros((n_ineq, n_vars))
        A_ub[:self._n_o, :n_a]  = -self.A.T
        A_ub[:self._n_o, n_a:]  =  self.F_o.T
        A_ub[self._n_o,  n_a:]  = -self.f_o
        b_ub = np.zeros(n_ineq)
        b_ub[self._n_o] = -(self.v_star - epsilon)

        # Equality constraints: F_a q0 = f_a
        A_eq = np.zeros((len(self.f_a), n_vars))
        A_eq[:, :n_a] = self.F_a
        b_eq = self.f_a

        bounds = [(0.0, None)] * n_a + [(None, None)] * n_dual

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')

        if not result.success:
            return dict(self._eq_pol_agent)

        q0_opt = np.clip(result.x[:n_a], 0.0, None)
        sf = self._sf
        return seqform_to_pol(
            q0_opt,
            sf['agent_iss_par'], sf['agent_iss_acts'], sf['agent_seq_map'])

    # ------------------------------------------------------------------
    # rl_agent interface
    # ------------------------------------------------------------------

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