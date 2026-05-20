import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyspiel
from open_spiel.python import rl_agent
from open_spiel.python.algorithms import outcome_sampling_mccfr


class OpponentMLP(nn.Module):
    """
    Simple feedforward network that predicts the opponent's next action
    from their observation / information-state tensor.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class RWYWEAgent(rl_agent.AbstractAgent):
    """
    Risk What You've Won in Expectation agent.

    High-level behavior:
    1. Learn a baseline strategy with Outcome Sampling MCCFR.
    2. For each legal action, estimate its value with a few rollouts.
    3. Use a learned opponent model to bias toward actions that are likely
       to exploit the opponent.
    4. Only deviate from the baseline if the estimated value is "safe"
       relative to the running safety budget k.
    5. Update the opponent model from observed opponent actions.
    """

    def __init__(self, game: pyspiel.Game, player_id: int, name: str = "rwywe_agent"):
        super().__init__(game, player_id, name)

        self.game = game
        self.player_id = player_id
        self.opp_id = 1 - player_id

        # RWYWE safety budget.
        # k tracks how much "surplus" over the baseline value we have accumulated.
        self.k = -0.018
        self.v_star = -0.056

        # Baseline game value under equilibrium / baseline play.

        # Number of rollout continuations used to estimate the value of an action.
        self.num_rollouts = 3

        # Determine observation size from the actual game instead of hard-coding 121*3.
        # This avoids dimension mismatches in games other than the one you originally tested.
        initial_state = game.new_initial_state()
        obs = initial_state.observation_tensor(player_id)
        self.input_dim = len(obs)

        self.num_actions = game.num_distinct_actions()

        # Opponent behavior model: predicts opponent action logits.
        self.model = OpponentMLP(self.input_dim, self.num_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Replay buffer stores (opponent observation tensor, opponent action).
        self.replay = deque(maxlen=10000)
        self.batch_size = 32
        

        # Baseline solver.
        self.solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)

        # Pretrain the baseline strategy.
        for _ in range(1500):
            self.solver.iteration()

        # Temporary storage so inform_action can record the opponent action
        # together with the opponent observation at the moment just before that action.
        self.pending_opp_obs = None

    def restart(self):
        self.pending_opp_obs  = None
        self._last_opp_action = None
        self._last_dist       = None
        self._last_evs        = None

    def _fallback_info_tensor(self, state, player):
        """
        Returns the observation tensor for a given player as a torch tensor.
        This is used as the model input.
        """
        board = state.observation_tensor(player)
        return torch.tensor(board, dtype=torch.float32)

    def _rollout_policy_action(self, state, cp):
        """
        Sample one action from the MCCFR average policy at the given state.
        Falls back to uniform over legal actions if the policy gives zero mass everywhere.
        """
        legal = state.legal_actions(cp)
        if not legal:
            return None

        policy = self.solver.average_policy()
        dist = policy.action_probabilities(state, cp)

        probs = np.array([dist.get(a, 0.0) for a in legal], dtype=np.float64)
        total = probs.sum()

        if total <= 0:
            probs = np.ones(len(legal), dtype=np.float64) / len(legal)
        else:
            probs /= total

        return np.random.choice(legal, p=probs)

    def _simulate_rollout(self, state):
        """
        Roll the game forward from the given state using the baseline policy
        until terminal or a depth cap. Returns this agent's payoff.
        """
        s = state.clone()
        depth = 0

        while not s.is_terminal() and depth < 40:
            cp = s.current_player()

            # Stop if chance / simultaneous-move handling is not implemented here.
            if cp < 0:
                break

            a = self._rollout_policy_action(s, cp)
            if a is None:
                break

            s.apply_action(a)
            depth += 1

        if s.is_terminal():
            return s.returns()[self.player_id]

        return 0.0

    def _estimate_lower_confidence_bound(self, mean_value, iters, confidence=0.95):
        """
        Lower confidence bound used as a conservative action-value estimate.

        This matches the idea in your code: estimate value and subtract an uncertainty term.
        """
        # Hoeffding-style radius.
        eps = np.sqrt(np.log(2.0 / (1.0 - confidence)) / (2.0 * max(iters, 1)))
        return mean_value - eps

    def _estimate_action_value(self, state, action):
        """
        Estimate the value of forcing 'action' now and then following the baseline policy.
        Returns a conservative lower-confidence estimate.
        """
        total = 0.0

        for _ in range(self.num_rollouts):
            s2 = state.clone()
            s2.apply_action(action)
            total += self._simulate_rollout(s2)

        avg = total / self.num_rollouts
        return self._estimate_lower_confidence_bound(avg, self.num_rollouts, confidence=0.95)

    def _predict_opponent_action_probs(self, state):
        """
        Use the learned opponent model to predict opponent action probabilities
        from the opponent's current observation tensor.

        Returns a length-num_actions numpy array.
        """
        opp_obs = self._fallback_info_tensor(state, self.opp_id).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(opp_obs)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return probs

    def step(self, state):
        """
        Choose an action at the current state.

        Logic:
        - Estimate values for legal actions.
        - Predict opponent behavior.
        - Score actions by combining safe value estimate with opponent-model bias.
        - If the best candidate is safe enough, take it.
        - Otherwise sample from the baseline average policy.
        """
        if state.is_terminal():
            return None

        if state.current_player() != self.player_id:
            return None

        legal = state.legal_actions(self.player_id)
        if not legal:
            return None

        # Conservative value estimates for each legal action.
        evs = {}
        sampled_actions = random.sample(legal, min(len(legal), 10))

        for a in sampled_actions:
            evs[a] = self._estimate_action_value(state, a)

        # Ensure every legal action has some value entry if sampled_actions was a subset.
        # Unsampled legal actions get -inf so they cannot beat sampled candidates.
        for a in legal:
            if a not in evs:
                evs[a] = float("-inf")

        # Opponent-model probabilities over all distinct actions.
        opp_action_probs = self._predict_opponent_action_probs(state)

        # Since the opponent model predicts opponent actions rather than our own action values,
        # there is not a clean one-to-one mapping from "predicted opponent action" to
        # "best response action" in arbitrary games. This version uses the model only as a bias:
        # actions with higher IDs matching likely opponent actions get a slight preference.
        scored_actions = []
        for a in legal:
            model_bonus = opp_action_probs[a] if a < len(opp_action_probs) else 0.0
            score = evs[a] + 0.05 * model_bonus
            scored_actions.append((a, score))

        best_a, best_score = max(scored_actions, key=lambda x: x[1])
        best_ev = evs[best_a]

        # Store mixed strategy and EVs for RWYWE k-update in on_terminal
        self._last_dist = {}
        self._last_evs  = dict(evs)
        if best_ev - self.v_star >= self.k:
            action = best_a
            self._last_dist = {a: 1.0 if a == best_a else 0.0 for a in legal}
        else:
            dist = self.solver.average_policy().action_probabilities(state, self.player_id)
            probs = np.array([dist.get(a, 0.0) for a in legal], dtype=np.float64)
            total = probs.sum()
            if total <= 0:
                probs = np.ones(len(legal), dtype=np.float64) / len(legal)
            else:
                probs /= total
            action = np.random.choice(legal, p=probs)
            self._last_dist = {a: float(dist.get(a, 0.0)) for a in legal}

        return action
    

    def getdist(self, state):
        if state.is_terminal(): return None
        if state.current_player() != self.player_id: return None
        legal = state.legal_actions(self.player_id)
        if not legal: return None

        evs = {}
        sampled_actions = random.sample(legal, min(len(legal), 10))

        for a in sampled_actions:
            evs[a] = self._estimate_action_value(state, a)

        for a in legal:
            if a not in evs: evs[a] = float("-inf")
        opp_action_probs = self._predict_opponent_action_probs(state)
        scored_actions = []

        for a in legal:
            model_bonus = opp_action_probs[a] if a < len(opp_action_probs) else 0.0
            score = evs[a] + 0.05 * model_bonus
            scored_actions.append((a, score))

        best_a, best_score = max(scored_actions, key=lambda x: x[1])
        best_ev = evs[best_a]

        if best_ev - self.v_star >= self.k:
            dist = {a : np.float64(0) if a != best_a else np.float64(1) for a in legal}
        else:
            dist = self.solver.average_policy().action_probabilities(state, self.player_id)
        return dist

    def inform_action(self, state, player, action):
        if player != self.opp_id:
            return
        opp_obs = self._fallback_info_tensor(state, self.opp_id)
        if 0 <= action < self.num_actions:
            self.replay.append((opp_obs, action))
        self._last_opp_action = action

    def on_terminal(self, state):
        """
        End-of-episode update.

        RWYWE k-update (Algorithm 6):
            k_{t+1} = k_t + u_i(pi_t_i, a^t_{-i}) - v*

        u_i(pi_t_i, a^t_{-i}) is the expected payoff of our MIXED strategy
        pi_t against the opponent's observed action — not the realized payoff.
        This is computed by getdist() giving the mixed strategy, then taking
        the weighted average of per-action EVs against the observed opp action.
        """
        realized_return = state.player_return(self.player_id)

        # Get the mixed strategy that was played this hand and the opp action
        opp_action = getattr(self, '_last_opp_action', None)

        if opp_action is not None and hasattr(self, '_last_state_before_opp'):
            # Compute u_i(pi_t, a^t_{-i}): expected payoff of our mixed strategy
            # against the opponent's observed action.
            # We use the per-action EVs estimated during step() and weight by pi_t.
            mixed_ev = sum(
                self._last_dist.get(a, 0.0) * self._last_evs.get(a, 0.0)
                for a in self._last_dist
            ) if hasattr(self, '_last_dist') and self._last_dist else realized_return
        else:
            mixed_ev = realized_return

        self.k = max(0, self.k + (mixed_ev - self.v_star))

        # Train the opponent model if enough data is available.
        if len(self.replay) >= self.batch_size:
            batch = random.sample(self.replay, self.batch_size)

            inputs  = torch.stack([b[0] for b in batch])
            targets = torch.tensor([b[1] for b in batch], dtype=torch.long)

            logits = self.model(inputs)
            loss   = F.cross_entropy(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Reset per-hand state
        self._last_opp_action = None
        self._last_dist       = None
        self._last_evs        = None


