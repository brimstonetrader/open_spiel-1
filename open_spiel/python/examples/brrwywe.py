import numpy as np
import pyspiel

from open_spiel.python.algorithms import exploitability, best_response, outcome_sampling_mccfr


class PolicyBot:
    def __init__(self, player_id, policy):
        self.player_id = player_id
        self.policy = policy

    def step(self, state):
        legal_actions = state.legal_actions(self.player_id)
        action_probs = self.policy.action_probabilities(state, self.player_id)

        probs = np.array([action_probs.get(a, 0.0) for a in legal_actions], dtype=np.float64)
        s = probs.sum()

        if s <= 0:
            probs = np.ones(len(legal_actions), dtype=np.float64) / len(legal_actions)
        else:
            probs /= s

        return np.random.choice(legal_actions, p=probs)


def play_episode(game, p0_bot, p1_bot):
    state = game.new_initial_state()

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = np.random.choice(outcomes, p=probs)
            state.apply_action(action)
            continue

        player = state.current_player()

        if player == 0:
            action = p0_bot.step(state)
        else:
            action = p1_bot.step(state)

        state.apply_action(action)

    return state.returns()


games = [
    ("Kuhn Poker", pyspiel.load_game("kuhn_poker")),
    ("Leduc Poker", pyspiel.load_game("leduc_poker")),
    ("Tic Tac Toe", pyspiel.load_game("tic_tac_toe"))
]

opponent_iters = [50000]

for game_name, game in games:
    print("\n" + "=" * 80)
    print(f"BEST RESPONSE VS OS-MCCFR AVG POLICY: {game_name}")
    print("=" * 80)

    for iters in opponent_iters:
        solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
        for _ in range(iters):
            solver.iteration()

        avg_policy = solver.average_policy()
        br_policy = best_response.BestResponsePolicy(
            game=game,
            player_id=1,
            policy=avg_policy
        )

        avg_bot = PolicyBot(0, avg_policy)
        br_bot = PolicyBot(1, br_policy)

        returns = np.zeros(2, dtype=np.float64)
        num_games = 1000

        for _ in range(num_games):
            ep_ret = play_episode(game, avg_bot, br_bot)
            returns += np.array(ep_ret)

        returns /= num_games

        try:
            nash_conv = exploitability.nash_conv(game, avg_policy)
        except Exception:
            nash_conv = float("nan")

        print(
            f"OS-MCCFR({iters:>4}) | "
            f"NashConv: {nash_conv:>10.6f} | "
            f"BR root value p1: {br_policy.value(game.new_initial_state()):>10.6f} | "
            f"avg_policy p0: {returns[0]:>10.6f} | "
            f"BR p1: {returns[1]:>10.6f}"
        )