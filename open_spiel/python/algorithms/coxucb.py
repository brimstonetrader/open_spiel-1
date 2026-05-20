import numpy as np
import cvxpy as cp
import pyspiel
from open_spiel.python import policy as policy_lib

import warnings
warnings.filterwarnings("ignore")

class CoxUCBBot(pyspiel.Bot):
    def __init__(self, game, player_id, alpha=-0.3, beta=0.3, delta=0.5):
        super().__init__()
        self.game = game
        self.iterator = 0
        self.player_id = player_id
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.counts = {}      # infoset -> {action: count}
        self.counts_w_weights = {}
        self.visits = {}      # infoset -> total count

    def restart_at(self, state): pass

    def step_with_policy(self, state):
        if self.player_id is None:
            self.player_id = state.current_player()
        info_str = state.information_state_string(self.player_id)
        legal = state.legal_actions()
        if info_str not in self.counts:
            self.counts[info_str] = {a: 0 for a in legal}
            self.visits[info_str] = 0
        u = np.zeros(len(legal))
        for i, a in enumerate(legal):
            nxt = state.clone()
            nxt.apply_action(a)
            while not nxt.is_terminal():
                act = np.random.choice(nxt.legal_actions())
                nxt.apply_action(act)
            u[i] = nxt.returns()[self.player_id]
            
        self.iterator += 1
        x = cp.Variable(len(legal))
        constraints = [x >= 0, cp.sum(x) == 1]
        obj = cp.Maximize(u @ x)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS)

        strat = np.array(x.value).flatten()
        strat = np.nan_to_num(strat, nan=1.0/len(legal))
        strat = np.clip(strat, 0.0, None)
        if strat.sum() <= 0:
            strat = np.ones(len(legal)) / len(legal)
        else:
            strat /= strat.sum()

        action = np.random.choice(legal, p=strat)
        policy = [(a, float(strat[i])) for i, a in enumerate(legal)]
        return policy, action

    def step(self, state):
        _, action = self.step_with_policy(state)
        self.last_infoset = state.information_state_string(self.player_id)
        return action

    def compute_confidence_region(self, info_str, legal_actions):
        total = max(1, self.visits.get(info_str, 0))
        counts = self.counts.get(info_str, {a: 0 for a in legal_actions})
        p_hat = np.array([counts.get(a, 0) for a in legal_actions], dtype=float) / total
        radius = np.sqrt(np.log(2.0 / self.delta) / (2 * total))

        lower = np.clip(p_hat - radius, 0.0, 1.0)
        upper = np.clip(p_hat + radius, 0.0, 1.0)
        return list(zip(lower.tolist(), upper.tolist()))


    def compute_payoff_vector(self, state):
        return np.array([self._expected_value(state.child(a)) for a in state.legal_actions()])

    def _expected_value(self, state):
        if state.is_terminal(): return state.returns()[self.player_id]
        if state.is_chance_node():
            ev = 0.0
            for a, prob in state.chance_outcomes():
                ev += prob * self._expected_value(state.child(a))
            return ev
        # for decision nodes (including opponent’s), assume uniform random
        ev = 0.0
        legal = state.legal_actions()
        p = 1.0 / len(legal)
        for a in legal:
            ev += p * self._expected_value(state.child(a))
        return ev

    def construct_utility_constrained_set(self, info_str, legal_actions, payoff_vector):
      #  Constructs cvxpy variables and constraints for the utility-constrained strategy set X:
      #      X = {x in Delta: alpha \leq x^T payoff_vector * p \leq beta for some p in confidence region Y}
      #  Implements constraints:
      #    - x in Delta (prob. dist over legal_actions)
      #    - There exists p in Y (confidence region) with alpha \leq x^T payoff_vector @ p \leq beta
      #  Returns (x, p, constraints) where x, p are cvxpy Variables and constraints is a list.
      #  payoff_vector: numpy array of shape (len(legal_actions),) giving utility for each action.

        n = len(legal_actions)
        x = cp.Variable(n)
        p = cp.Variable(n)
        bounds = self.compute_confidence_region(info_str, legal_actions)
        lower, upper = zip(*bounds)
        constraints = []
        constraints += [p >= np.array(lower), p <= np.array(upper), cp.sum(p) == 1]
        constraints += [x >= 0, cp.sum(x) == 1]
        constraints += [self.alpha <= cp.multiply(cp.multiply(x, payoff_vector), p), cp.multiply(cp.multiply(x, payoff_vector), p) <= self.beta]
        return x, p, constraints


    def select_strategy(self, info_str, legal_actions, payoff_vector, iterations=5):
        n = len(legal_actions)
        bounds = self.compute_confidence_region(info_str, legal_actions)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        p = (lower + upper) / 2.0
        p /= p.sum()

        x = np.ones(n) / n
        for _ in range(iterations):
            x_var = cp.Variable(n)
            expr_x = cp.multiply(x_var, payoff_vector) * p
            cons_x = [x_var >= 0, cp.sum(x_var) == 1, expr_x >= self.alpha, expr_x <= self.beta]
            prob_x = cp.Problem(cp.Maximize(expr_x), cons_x)
            prob_x.solve(solver=cp.SCS)
            raw_x = x_var.value
            if raw_x is None:
                x = np.ones(n) / n
            else:
                x = np.array([v if v is not None else 1.0/n for v in raw_x], dtype=float)
                x = np.clip(x, 0.0, None)
                x = x / x.sum() if x.sum() > 0 else np.ones(n) / n

            p_var = cp.Variable(n)
            expr_p = cp.multiply(x, payoff_vector) * p_var
            cons_p = [p_var >= lower, p_var <= upper, cp.sum(p_var) == 1]
            prob_p = cp.Problem(cp.Minimize(expr_p), cons_p)
            prob_p.solve(solver=cp.SCS)
            raw_p = p_var.value
            if raw_p is None:
                p = (lower + upper) / 2.0
                p = p / p.sum()
            else:
                p = np.array([v if v is not None else (lower[i]+upper[i])/2 for i, v in enumerate(raw_p)], dtype=float)
                p = np.clip(p, lower, upper)
                p = p / p.sum() if p.sum() > 0 else (lower + upper) / 2.0

        return x
