import numpy as np
import pyspiel
from open_spiel.python import rl_agent
from open_spiel.python.algorithms import outcome_sampling_mccfr


class MCCFRAgent(rl_agent.AbstractAgent):
  def __init__(self, game_name, player_id, num_actions, iters=5000, seed=0):
    super().__init__(player_id=player_id, name="mccfr")
    self._player_id = player_id
    self._num_actions = num_actions
    self._rng = np.random.RandomState(seed + player_id)

    game = pyspiel.load_game(game_name)
    solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
    for _ in range(iters):
      solver.iteration()
    policy = solver.average_policy()

    # Build lookup: info_state_tensor(tuple) -> probs over actions (numpy array)
    self._table = {}
    self._build_table(game, policy)

  def _key(self, info_state_tensor):
    # Works for list/np.ndarray; tensor entries are 0/1 floats in RL env.
    return tuple(float(x) for x in info_state_tensor)

  def _build_table(self, game, policy):
    root = game.new_initial_state()
    stack = [root]
    seen = set()

    while stack:
      s = stack.pop()
      sid = s.history_str()  # sufficient for small games like kuhn/leduc
      if sid in seen:
        continue
      seen.add(sid)

      if s.is_terminal():
        continue

      if s.is_chance_node():
        for a, _ in s.chance_outcomes():
          ns = s.child(a)
          stack.append(ns)
        continue

      p = s.current_player()
      legal = s.legal_actions(p)
      info = s.information_state_tensor(p)
      key = self._key(info)

      probs = np.zeros(self._num_actions, dtype=np.float64)
      d = policy.action_probabilities(s, p)
      for a in legal:
        probs[a] = float(d.get(a, 0.0))
      z = probs.sum()
      if z <= 0:
        probs[legal] = 1.0 / len(legal)
      else:
        probs /= z

      self._table[key] = probs

      for a in legal:
        stack.append(s.child(a))

  def step(self, time_step, is_evaluation=False):
    # Terminal: no action required
    if time_step.last():
        return rl_agent.StepOutput(action=0, probs=np.zeros(self._num_actions))

    cur = time_step.observations["current_player"]

    # If it's not this agent's turn, return a dummy StepOutput.
    # rl_response.py only uses the output for the acting player in turn-based games.
    if cur != self._player_id:
        return rl_agent.StepOutput(action=0, probs=np.zeros(self._num_actions))

    legal = time_step.observations["legal_actions"][self._player_id]
    if not legal:
        return rl_agent.StepOutput(action=0, probs=np.zeros(self._num_actions))

    info = time_step.observations["info_state"][self._player_id]
    key = self._key(info)

    probs = self._table.get(key)
    if probs is None:
        probs = np.zeros(self._num_actions, dtype=np.float64)
        probs[legal] = 1.0 / len(legal)

    masked = np.zeros(self._num_actions, dtype=np.float64)
    masked[legal] = probs[legal]
    z = masked.sum()
    if z <= 0:
        masked[legal] = 1.0 / len(legal)
    else:
        masked /= z

    action = int(self._rng.choice(np.arange(self._num_actions), p=masked))
    return rl_agent.StepOutput(action=action, probs=masked)
