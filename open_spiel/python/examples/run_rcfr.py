import pyspiel
from open_spiel.python.pytorch import rcfr
from open_spiel.python.algorithms import exploitability
import torch
import torch.nn.functional as F

game = pyspiel.load_game("kuhn_poker")
solver = rcfr.RcfrSolver(game,
                         [rcfr.DeepRcfrModel(game,1000), rcfr.DeepRcfrModel(game,1000)])

def train_fn(model, dataset):
    # convert the dataset (TensorDataset) into a DataLoader
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    # we’ll do exactly one pass (epoch) over the buffer each time RCFR calls us
    for features, targets in loader:
        preds = model(features)                   # forward pass
        loss = F.mse_loss(preds, targets)         # mean‐squared error on regrets
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # no need to return anything—RCFR just updates its internal weights for you.

for i in range(1000):
    solver.evaluate_and_update_policy(train_fn)
    if i % 100 == 0:
        avg_policy = solver.average_policy()
        nash_conv = exploitability.nash_conv(game, avg_policy)  # :contentReference[oaicite:0]{index=0}
        print(nash_conv)

# finally, you can also save or inspect solver.average_policy() as you like