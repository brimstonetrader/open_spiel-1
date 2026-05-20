# open_spiel/python/algorithms/rwywe_callable.py
import numpy as np
import pyspiel
from open_spiel.python import rl_agent
from open_spiel.python.examples.rwywe3 import RWYWEAgent  # adjust if your rwywe3.py lives elsewhere


class RWYWECallableAgent(rl_agent.AbstractAgent):
  """Stationary RWYWE exploitee for rl_response.py: snapshot RWYWE average_policy -> lookup table."""
  def __init__(self, game_name, player_id, num_actions, seed=0):
    super().__init__(player_id=player_id, name="rwywe_callable")
    self._player_id = player_id
    self._num_actions = num_actions
    self._rng = np.random.RandomState(seed + player_id)

    game = pyspiel.load_game(game_name)
    rwywe = RWYWEAgent(game, player_id=player_id)
    policy = rwywe.solver.average_policy()  # snapshot
    self._policy = rwywe.solver.average_policy()  # snapshot

    self._table = {}
    self._build_table(game, policy)

  def _key(self, info_state_tensor):
    return tuple(float(x) for x in info_state_tensor)

  def _build_table(self, game, policy):
    root = game.new_initial_state()
    stack = [root]
    seen = set()

    while stack:
      s = stack.pop()
      sid = s.history_str()
      if sid in seen:
        continue
      seen.add(sid)

      if s.is_terminal():
        continue
      if s.is_chance_node():
        for a, _ in s.chance_outcomes():
          stack.append(s.child(a))
        continue

      p = s.current_player()
      legal = s.legal_actions(p)
      key = self._key(s.information_state_tensor(p))

      probs = np.zeros(self._num_actions, dtype=np.float64)
      d = policy.action_probabilities(s, p)
      for a in legal:
        probs[a] = float(d.get(a, 0.0))
      z = probs.sum()
      probs[legal] = (1.0 / len(legal)) if z <= 0 else probs[legal] / z

      self._table[key] = probs
      for a in legal:
        stack.append(s.child(a))

  def step(self, time_step, is_evaluation=False):
    if time_step.last():
      return rl_agent.StepOutput(action=0, probs=np.zeros(self._num_actions))

    cur = time_step.observations["current_player"]
    if cur != self._player_id:
      return rl_agent.StepOutput(action=0, probs=np.zeros(self._num_actions))

    legal = time_step.observations["legal_actions"][self._player_id]
    info = time_step.observations["info_state"][self._player_id]
    probs = self._table.get(self._key(info))

    if probs is None:
      probs = np.zeros(self._num_actions, dtype=np.float64)
      probs[legal] = 1.0 / len(legal)

    masked = np.zeros(self._num_actions, dtype=np.float64)
    masked[legal] = probs[legal]
    z = masked.sum()
    masked[legal] = (1.0 / len(legal)) if z <= 0 else masked[legal] / z

    a = int(self._rng.choice(np.arange(self._num_actions), p=masked))
    return rl_agent.StepOutput(action=a, probs=masked)
