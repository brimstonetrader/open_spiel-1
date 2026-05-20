# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python implementation for Monte Carlo Counterfactual Regret Minimization."""

import numpy as np
from open_spiel.python.algorithms import mccfr
import pyspiel
import torch
import torch.nn as nn
import torch.optim as optim

class AdvantageNetwork(nn.Module):
    """Network to predict advantages for each action at infostates"""
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class QBaselineNetwork(nn.Module):
    """Network to predict Q-values as baselines"""
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ReservoirBuffer:
    """Reservoir sampling buffer for experience replay"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.num_seen = 0
    
    def add(self, experience, weight=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append((experience, weight))
        else:
            # Reservoir sampling
            if random.random() < self.capacity / (self.num_seen + 1):
                idx = random.randint(0, self.capacity - 1)
                self.buffer[idx] = (experience, weight)
        self.num_seen += 1
    
    def sample_batch(self, batch_size):
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        experiences, weights = zip(*samples)
        return experiences, weights

class OutcomeSamplingSolver(mccfr.MCCFRSolverBase):
  """An implementation of outcome sampling MCCFR."""

  def __init__(self, game):
    super().__init__(game)
    # This is the epsilon exploration factor. When sampling episodes, the
    # updating player will sampling according to expl * uniform + (1 - expl) *
    # current_policy.
    self._expl = 0.6

    assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
        "MCCFR requires sequential games. If you're trying to run it " +
        "on a simultaneous (or normal-form) game, please first transform it " +
        "using turn_based_simultaneous_game.")

  def iteration(self):
    """Performs one iteration of outcome sampling.

    An iteration consists of one episode for each player as the update
    player.
    """
    for update_player in range(self._num_players):
      state = self._game.new_initial_state()
      self._episode(
          state, update_player, my_reach=1.0, opp_reach=1.0, sample_reach=1.0)

  def _baseline(self, state, info_state, aidx):  # pylint: disable=unused-argument
    # Default to vanilla outcome sampling
    return 0

  def _baseline_corrected_child_value(self, state, info_state, sampled_aidx,
                                      aidx, child_value, sample_prob):
    # Applies Eq. 9 of Schmid et al. '19
    baseline = self._baseline(state, info_state, aidx)
    if aidx == sampled_aidx:
      return baseline + (child_value - baseline) / sample_prob
    else:
      return baseline

  def _episode(self, state, update_player, my_reach, opp_reach, sample_reach):
    """Runs an episode of outcome sampling.

    Args:
      state: the open spiel state to run from (will be modified in-place).
      update_player: the player to update regrets for (the other players
        update average strategies)
      my_reach: reach probability of the update player
      opp_reach: reach probability of all the opponents (including chance)
      sample_reach: reach probability of the sampling (behavior) policy

    Returns:
      util is a real value representing the utility of the update player
    """
    if state.is_terminal():
      return state.player_return(update_player)

    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      aidx = np.random.choice(range(len(outcomes)), p=probs)
      state.apply_action(outcomes[aidx])
      return self._episode(state, update_player, my_reach,
                           probs[aidx] * opp_reach, probs[aidx] * sample_reach)

    cur_player = state.current_player()
    info_state_key = state.information_state_string(cur_player)
    legal_actions = state.legal_actions()
    num_legal_actions = len(legal_actions)
    infostate_info = self._lookup_infostate_info(info_state_key,
                                                 num_legal_actions)
    policy = self._regret_matching(infostate_info[mccfr.REGRET_INDEX],
                                   num_legal_actions)
    if cur_player == update_player:
      uniform_policy = (
          np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions)
      sample_policy = self._expl * uniform_policy + (1.0 - self._expl) * policy
    else:
      sample_policy = policy
    sampled_aidx = np.random.choice(range(num_legal_actions), p=sample_policy)
    state.apply_action(legal_actions[sampled_aidx])
    if cur_player == update_player:
      new_my_reach = my_reach * policy[sampled_aidx]
      new_opp_reach = opp_reach
    else:
      new_my_reach = my_reach
      new_opp_reach = opp_reach * policy[sampled_aidx]
    new_sample_reach = sample_reach * sample_policy[sampled_aidx]
    child_value = self._episode(state, update_player, new_my_reach,
                                new_opp_reach, new_sample_reach)

    # Compute each of the child estimated values.
    child_values = np.zeros(num_legal_actions, dtype=np.float64)
    for aidx in range(num_legal_actions):
      child_values[aidx] = self._baseline_corrected_child_value(
          state, infostate_info, sampled_aidx, aidx, child_value,
          sample_policy[aidx])
    value_estimate = 0
    for aidx in range(num_legal_actions):
      value_estimate += policy[aidx] * child_values[aidx]

    # Update regrets and avg strategies
    if cur_player == update_player:
      # Estimate for the counterfactual value of the policy.
      cf_value = value_estimate * opp_reach / sample_reach

      # Update regrets.
      #
      # Note: different from Chapter 4 of Lanctot '13 thesis, the utilities
      # coming back from the recursion are already multiplied by the players'
      # tail reaches and divided by the sample tail reach. So when adding
      # regrets to the table, we need only multiply by the opponent reach and
      # divide by the sample reach to this point.
      for aidx in range(num_legal_actions):
        # Estimate for the counterfactual value of the policy replaced by always
        # choosing sampled_aidx at this information state.
        cf_action_value = child_values[aidx] * opp_reach / sample_reach
        self._add_regret(info_state_key, aidx, cf_action_value - cf_value)

      # update the average policy
      for aidx in range(num_legal_actions):
        increment = my_reach * policy[aidx] / sample_reach
        self._add_avstrat(info_state_key, aidx, increment)

    return value_estimate


class DREAMSolver(OutcomeSamplingSolver):
  def __init__(self, game, exploration_epsilon=0.6):
      super().__init__(game)
      self._expl = exploration_epsilon
      
      # Neural networks
      self.advantage_net = AdvantageNetwork(input_dim=game.information_state_tensor_shape()[0], 
                                          action_dim=game.num_distinct_actions())
      self.q_net = QBaselineNetwork(input_dim=game.information_state_tensor_shape()[0] * 2,  # Both players' perspectives
                                  action_dim=game.num_distinct_actions())
      
      # Optimizers
      self.advantage_optimizer = optim.Adam(self.advantage_net.parameters(), lr=0.001)
      self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
      
      # Experience buffers
      self.advantage_buffer = ReservoirBuffer(capacity=2000000)  # 2M for Leduc
      self.q_buffer = ReservoirBuffer(capacity=200000)  # 200K for Q-values
      
      # Store models for average policy (like SD-CFR)
      self.saved_models = []
      
  def _encode_infostate(self, state, player):
      """Convert information state to tensor representation"""
      return torch.FloatTensor(state.information_state_tensor(player))
  
  def _encode_joint_infostate(self, state):
      """Encode infostates for both players (for Q-network)"""
      if state.current_player() == 0:
          p1_info = state.information_state_tensor(0)
          p2_info = state.information_state_tensor(1)
      else:
          p1_info = state.information_state_tensor(1)
          p2_info = state.information_state_tensor(0)
      return torch.FloatTensor(np.concatenate([p1_info, p2_info]))
  
  def _dream_value_estimate(self, state, action, sampled_action, child_value, 
                        sample_prob, info_state, iteration):
    """Implement DREAM's baseline-adjusted value estimation"""
    
    # Get Q-baseline prediction
    joint_state = self._encode_joint_infostate(state)
    with torch.no_grad():
        q_baseline = self.q_net(joint_state)[action]
    
    if action == sampled_action:
        # Use baseline adjustment for sampled action
        return q_baseline + (child_value - q_baseline) / sample_prob
    else:
        # Return baseline for non-sampled actions
        return q_baseline
    


  def train_advantage_network(self, iteration):
    """Train advantage network with linear CFR weighting"""
    if len(self.advantage_buffer.buffer) == 0:
        return
    
    batch_size = 2048
    num_batches = 3000  # For Leduc
    
    for _ in range(num_batches):
        experiences, weights = self.advantage_buffer.sample_batch(batch_size)
        
        # Unpack experiences: (infostate_tensor, action, sampled_advantage, iteration_weight)
        states, actions, advantages, iter_weights = zip(*experiences)
        
        states = torch.stack(states)
        predicted_advantages = self.advantage_net(states)
        
        # Linear CFR: weight by iteration number
        total_weights = torch.FloatTensor([w * iter_w for w, iter_w in zip(weights, iter_weights)])
        
        # Compute loss
        loss = 0
        for i, (state, action, advantage, pred_adv) in enumerate(zip(states, actions, advantages, predicted_advantages)):
            loss += total_weights[i] * (pred_adv[action] - advantage) ** 2
        
        loss = loss / len(experiences)
        
        self.advantage_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), 1.0)
        self.advantage_optimizer.step()

  def train_q_network(self):
    """Train Q-network using Expected SARSA"""
    if len(self.q_buffer.buffer) == 0:
        return
    
    batch_size = 512
    num_batches = 1000  # As specified in paper
    
    for _ in range(num_batches):
        experiences, _ = self.q_buffer.sample_batch(batch_size)
        # Q-network training logic here...
        # Implement Expected SARSA update as described in paper


  def train_advantage_network(self, iteration):
    """Train advantage network with linear CFR weighting"""
    if len(self.advantage_buffer.buffer) == 0:
        return
    
    batch_size = 2048
    num_batches = 3000  # For Leduc
    
    for _ in range(num_batches):
        experiences, weights = self.advantage_buffer.sample_batch(batch_size)
        
        # Unpack experiences: (infostate_tensor, action, sampled_advantage, iteration_weight)
        states, actions, advantages, iter_weights = zip(*experiences)
        
        states = torch.stack(states)
        predicted_advantages = self.advantage_net(states)
        
        # Linear CFR: weight by iteration number
        total_weights = torch.FloatTensor([w * iter_w for w, iter_w in zip(weights, iter_weights)])
        
        # Compute loss
        loss = 0
        for i, (state, action, advantage, pred_adv) in enumerate(zip(states, actions, advantages, predicted_advantages)):
            loss += total_weights[i] * (pred_adv[action] - advantage) ** 2
        
        loss = loss / len(experiences)
        
        self.advantage_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), 1.0)
        self.advantage_optimizer.step()

  def train_q_network(self):
      """Train Q-network using Expected SARSA"""
      if len(self.q_buffer.buffer) == 0:
          return
      
      batch_size = 512
      num_batches = 1000  # As specified in paper
      
      for _ in range(num_batches):
          experiences, _ = self.q_buffer.sample_batch(batch_size)
          # Q-network training logic here...
          # Implement Expected SARSA update as described in paper


  def _episode(self, state, update_player, my_reach, opp_reach, sample_reach, iteration):
    """Modified episode method with DREAM components"""
    
    if state.is_terminal():
        return state.player_return(update_player)
    
    # ... existing chance node handling ...
    
    cur_player = state.current_player()
    info_state_key = state.information_state_string(cur_player)
    legal_actions = state.legal_actions()
    
    # Use neural network for policy instead of tabular regrets
    info_state_tensor = self._encode_infostate(state, cur_player)
    with torch.no_grad():
        advantages = self.advantage_net(info_state_tensor).numpy()
    
    # Convert advantages to policy via regret matching
    policy = self._neural_regret_matching(advantages, legal_actions)
    
    # Exploration policy
    if cur_player == update_player:
        uniform_policy = np.ones(len(legal_actions)) / len(legal_actions)
        sample_policy = self._expl * uniform_policy + (1.0 - self._expl) * policy
    else:
        sample_policy = policy
    
    # Sample action and recurse
    sampled_aidx = np.random.choice(range(len(legal_actions)), p=sample_policy)
    
    # Store experience for training
    if cur_player == update_player:
        self._store_advantage_experience(state, legal_actions, advantages, 
                                       sampled_aidx, iteration, my_reach)
    
    # Continue with recursion and value updates using DREAM's baseline adjustment
    # ...

  def _neural_regret_matching(self, advantages, legal_actions):
    """Convert advantages to policy via regret matching"""
    positive_advantages = np.maximum(advantages[legal_actions], 0)
    sum_positive = np.sum(positive_advantages)
    
    if sum_positive > 0:
        return positive_advantages / sum_positive
    else:
        # Choose action with highest advantage when all are negative
        best_action = np.argmax(advantages[legal_actions])
        policy = np.zeros(len(legal_actions))
        policy[best_action] = 1.0
        return policy

  def _store_advantage_experience(self, state, legal_actions, advantages, 
                                sampled_aidx, iteration, reach_prob):
      """Store experience for advantage network training"""
      info_state_tensor = self._encode_infostate(state, state.current_player())
      
      # Compute sampled advantage using DREAM's formulation
      sampled_advantage = self._compute_sampled_advantage(...)  # Implement Eq. 8
      
      # Weight by 1/sampling_probability for correction
      weight = 1.0 / reach_prob if reach_prob > 0 else 1.0
      
      # Linear CFR: weight by iteration number
      iteration_weight = iteration
      
      experience = (info_state_tensor, legal_actions[sampled_aidx], 
                    sampled_advantage, iteration_weight)
      self.advantage_buffer.add(experience, weight)