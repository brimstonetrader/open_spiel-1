import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python.examples import gto_kuhn_poker

game = pyspiel.load_game("kuhn_poker")
for i in range(31):
    strat = gto_kuhn_poker.GTOKuhnPolicy(game, i/90)
    exp = (exploitability.nash_conv(game, strat))
    print(f'alpha: {i/90:.4f}, exploitability: {exp}')