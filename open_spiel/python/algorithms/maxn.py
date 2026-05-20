import pyspiel
import numpy as np


def std_value_function(state, p, c=100):
    returns = 0 
    for _ in range(c):
        clone = state.clone()
        while not clone.is_terminal():
            la = clone.legal_actions()
            clone.apply_action(la[np.random.randint(len(la))])
        returns += clone.returns()[p]
    return returns / c

def _maxn(state, depth, best_action_holder=None):
    num_players = state.num_players()

    # Terminal node: return returns for all players
    if state.is_terminal():
        return list(state.returns())

    if depth == 0:
        return [std_value_function(state, p) for p in range(num_players)]

    current_player = state.current_player()

    # Chance node: expectation over chance outcomes
    if state.is_chance_node():
        # initialize sum of values
        values = [0.0] * num_players
        for action, prob in state.chance_outcomes():
            child = state.clone()
            child.apply_action(action)
            child_vals = _maxn(child, depth, None)
            for p in range(num_players):
                values[p] += prob * child_vals[p]
        return values

    # Decision node: maximize for current player
    best_value = float("-inf")
    best_vals = [0.0] * num_players
    best_action = None

    for action in state.legal_actions():
        child = state.clone()
        child.apply_action(action)
        child_vals = _maxn(child, depth - 1, None)
        if child_vals[current_player] > best_value:
            best_value = child_vals[current_player]
            best_vals = child_vals
            best_action = action

    # record best action if desired
    if best_action_holder is not None and best_action is not None:
        best_action_holder[0] = best_action

    return best_vals


def maxn_search(game, state=None, depth_limit=100):
    # basic checks
    info = game.get_type()
    if info.chance_mode not in (pyspiel.GameType.ChanceMode.DETERMINISTIC,
                                pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC):
        raise ValueError("Game chance_mode must be deterministic or explicit stochastic.")
    if info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("Game must be sequential.")
    if info.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
        raise ValueError("Game reward_model must be terminal.")

    # initialize root state
    root = state.clone() if state is not None else game.new_initial_state()
    if root.is_chance_node():
        raise ValueError("Root must not be a chance node.")

    # holder for best action
    best_action_holder = [None]
    values = _maxn(root, depth_limit, best_action_holder)
    return values, best_action_holder[0]
