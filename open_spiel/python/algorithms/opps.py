import numpy as np
import pyspiel
import copy
import warnings
warnings.filterwarnings("ignore")

class OPPSBot(pyspiel.Bot):

    def std_value_function(state, p, c=100):
        returns = 0 
        for _ in range(c):
            clone = state.clone()
            while not clone.is_terminal():
                la = clone.legal_actions()
                clone.apply_action(la[np.random.randint(len(la))])
            returns += clone.returns()[p]
        return returns / c

    def __init__(self, game, player_id, depth=10, n1=1, l1=1000000000, l2=1,
                 value_function=std_value_function,
                 move_ordering_fn=None):
        super().__init__()
        self.game = game
        self.player_id = player_id  # root player index
        self.default_depth = depth
        self.n1 = n1
        self.l1 = l1
        self.l2 = l2
        self.value_function = value_function
        self.move_ordering_fn = move_ordering_fn

    def restart_at(self, state): pass

    def _static_move_order(self, state):
        legal = state.legal_actions()
        if self.move_ordering_fn is not None:
            ordered = self.move_ordering_fn(state, self.player_id, self.value_function)
            return ordered
        else:
            vals = []
            for a in legal:
                child = state.clone()
                child.apply_action(a)
                h = self.value_function(child, self.player_id)
                vals.append((a, h))
            vals_sorted = sorted(vals, key=lambda x: x[1])
            ordered = [a for (a, _) in vals_sorted]
            return ordered

    def _opps_search(self, state, depth, m_count):
        if state.is_terminal():
            returns = state.returns() 
            return returns[self.player_id]
        if depth <= 0:
            return self.value_function(state, self.player_id)

        current = state.current_player()
        ordered = self._static_move_order(state)

        if current == self.player_id:
            m_count = 0
            A_prime = ordered 
        else:
            if m_count >= self.n1:
                A_prime = ordered[:self.l2]
            else:
                A_prime = ordered[:self.l1]
        if not A_prime:
            returns = state.returns()
            return returns[self.player_id]

        best_vals = []
        for a in A_prime:
            child = state.clone()
            child.apply_action(a)

            if current != self.player_id:
                pos = ordered.index(a)
                if pos >= self.l2: new_m = m_count + 1
                else: new_m = m_count
            else:
                new_m = m_count

            val = self._opps_search(child, depth - 1, new_m)
            best_vals.append(val)


        if current == self.player_id:
            return max(best_vals)
        else:
            return min(best_vals)

    def step(self, state):
        if state.is_terminal():   return None

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            r = np.random.random()
            cum = 0.0
            for action, prob in outcomes:
                cum += prob
                if r < cum:
                    return action
            return outcomes[-1][0]

        legal = state.legal_actions()
        best_action = None
        best_val = -float('inf')
        for a in legal:
            child = state.clone()
            child.apply_action(a)
            val = self._opps_search(child, self.default_depth - 1, 0)
            if best_action is None or val > best_val:
                best_action = a
                best_val = val

        return best_action
