import pyspiel

game = pyspiel.load_game(
    'repeated_game(num_repetitions=5,stage_game=matrix_brps())'
)
state = game.new_initial_state()

