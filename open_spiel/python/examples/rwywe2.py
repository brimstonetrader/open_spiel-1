import numpy as np
import random
from collections import defaultdict
import pyspiel
from open_spiel.python import rl_agent, policy
from open_spiel.python.algorithms import mcts, mcts_agent, minimax, outcome_sampling_mccfr
import torch
import torch.nn as nn
from open_spiel.python.pytorch import deep_cfr
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

class RWYWEAgent(rl_agent.AbstractAgent):
    """
    RWYWE agent for Kuhn Poker with full rollouts and correct infosets.
    """
    def __init__(self, game: pyspiel.Game, player_id: int, name='rwywe_agent'):
        super().__init__(game, player_id, name)
        self.game = game
        self.player_id = player_id
        self.opp_id = 1 - player_id
        self.k = 0.018
        self.opp_model = defaultdict(lambda: defaultdict(int))
        self.gto = outcome_sampling_mccfr.OutcomeSamplingSolver(game) 
        for _ in range(1_000): self.gto.iteration()
        self.v_star = -1.0/18.0 
        self.hand_history = []
        self.num_rollouts = 100
        self.model = TransformerOpponentModel(11)

    def restart(self): self.hand_history.clear()

    def _simulate_rollout(self, state):
        s = state.clone()
        while not s.is_terminal():
            legal = s.legal_actions()                   
            a = legal[random.randint(0, len(legal)-1)]  
            s.apply_action(a)
        return s.returns()[self.player_id]
    
    def _estimate_variance(self, r, iters, confidence):
        epsilon = 2*(np.log(2/confidence) / (2*iters))**0.5
        return (r-epsilon, r+epsilon)

    def step(self, state):
        info = state.information_state_string(self.player_id)
        legal = state.legal_actions()
        evs = {}
        
        
        for a in legal:
            total_ret = 0.0
            for _ in range(self.num_rollouts):
                s2 = state.clone()
                s2.apply_action(a)
                total_ret += self._simulate_rollout(s2)
            r = total_ret / self.num_rollouts
            evs[a] = self._estimate_variance(r, self.num_rollouts, 0.99)[0]

        best_a, best_ev = max(evs.items(), key=lambda x: x[1])
        if best_ev - self.v_star >= self.k:
            action = best_a
        else:
            b, c = True, 0
            action = state.legal_actions()[np.random.randint(len(state.legal_actions()))]
            poli = self.gto.average_policy()
            dist = poli.action_probabilities(state, self.player_id)
            probs = np.array([dist[a] 
                      for a in state.legal_actions()], dtype=float)
            action = np.random.choice(state.legal_actions(), p=probs)
        self.hand_history.append((self.player_id, action))
        return action

    def inform_action(self, state, player, action):
        if player == self.opp_id:
            info = state.information_state_string(self.opp_id)
            self.opp_model[(info, action)] += 1

    def on_terminal(self, state):
        ret = state.player_return(self.player_id)
        self.k = max(0.0, self.k + (ret - self.v_star))
        self.restart()




class HexStateDataset(Dataset):
    """
    Dataset for Hex states and opponent actions.
    Each sample: (board_tensor, action_index)
    board_tensor: shape [N,N], values in {0,1,2} (empty, us, opp)
    action_index: int in [0, N*N)
    """
    def __init__(self, states, actions):
        self.states = states  # list of board tensors
        self.actions = actions  # list of ints

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class SmallHexTransformer(nn.Module):
    """
    Small Transformer for opponent modeling in Hex.
    Input: flattened board of length L=N*N, with token ids {0,1,2}.
    Output: probability distribution over next opponent action tokens.
    """
    def __init__(self, board_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.board_size = board_size
        self.seq_len = board_size * board_size
        self.d_model = d_model

        # token embedding: 3 tokens
        self.token_embed = nn.Embedding(3, d_model)
        # positional encoding
        self.pos_embed = nn.Parameter(torch.randn(self.seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # output head: predict next action over seq_len positions
        self.head = nn.Linear(d_model, self.seq_len)

    def forward(self, board_tensor):
        # board_tensor: [batch_size, seq_len] ints
        x = self.token_embed(board_tensor)  # [B, L, d_model]
        x = x + self.pos_embed.unsqueeze(0)  # add positional
        # transformer expects shape [L, B, d_model]
        x = x.transpose(0, 1)
        x = self.transformer(x)
        # back to [B, L, d_model]
        x = x.transpose(0, 1)
        logits = self.head(x)  # [B, L, seq_len]
        # we want to predict one action: flatten to [B, L*seq_len]? Actually use only final pooling
        # Simplest: pool over positions: take mean
        pooled = x.mean(dim=1)  # [B, d_model]
        action_logits = self.head(pooled)  # [B, seq_len]
        return action_logits

class TransformerOpponentModel:
    def __init__(self, board_size, lr=1e-3, batch_size=32):
        self.model = SmallHexTransformer(board_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.memory = []  # list of (board_tensor, action_idx)

    def observe(self, board, action_idx):
        # board: 2D array, action_idx: int
        tensor = torch.tensor(board.flatten(), dtype=torch.long)
        self.memory.append((tensor, action_idx))
        if len(self.memory) >= self.batch_size:
            self.train_batch()

    def train_batch(self):
        batch = self.memory[-self.batch_size:]
        states, actions = zip(*batch)
        states = torch.stack(states)  # [B, L]
        actions = torch.tensor(actions, dtype=torch.long)
        logits = self.model(states)
        loss = self.criterion(logits, actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict_probs(self, board):
        # board: 2D array
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(board.flatten(), dtype=torch.long).unsqueeze(0)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs  # length seq_len array
