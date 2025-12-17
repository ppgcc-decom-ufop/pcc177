"""
fl_tcn_eto.py

Simulação de Aprendizado Federado para previsão multivariada de ETo.
- Cada cliente carrega seu CSV local (um CSV por cliente).
- Modelo local: TCN (implementação simples).
- Agregador: FedAvg ponderado por número de exemplos.
- Simulação local (sem rede). 

Uso:
    python fl_tcn_eto.py --clients_csvs client1.csv client2.csv ... --target ETo --seq_len 4
    python fl_tcn_eto.py --clients_csvs lat-15.35_lon-55.45_mt.csv lat-19.75_lon-44.45_mg.csv --target ETo --seq_len 4
    python fl_tcn_eto.py --clients_csvs .\path\lat-15.35_lon-55.45_mt.csv .\path\lat-19.75_lon-44.45_mg.csv .\path\lat-2.15_lon-59.85_am.csv .\path\lat-30.75_lon-55.45_rs.csv .\path\lat-4.35_lon-40.05_ce.csv .\path\lat-8.75_lon-35.65_pe.csv --target ETo --seq-len 4

Autor: ChatGPT (esboço)
"""
import argparse
import os
import math
import random
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------------- utilities ----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

# ---------------- dataset + windows ----------------
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype('float32')
        self.y = y.astype('float32')
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

def make_windows_from_df(df: pd.DataFrame, feature_cols: List[str], target_col: str, seq_len: int, scaler: StandardScaler=None, fit_scaler:bool=True):
    df = df.dropna().reset_index(drop=True)
    X_raw = df[feature_cols].values.astype(float)
    y_raw = df[target_col].values.astype(float)
    if scaler is None: scaler = StandardScaler()
    if fit_scaler: scaler.fit(X_raw)
    Xs = scaler.transform(X_raw)
    X_windows, y_windows = [], []
    for i in range(seq_len, len(Xs)):
        X_windows.append(Xs[i-seq_len:i])
        y_windows.append(y_raw[i])
    if len(X_windows)==0:
        return None, None, scaler
    return np.stack(X_windows), np.array(y_windows), scaler

# ---------------- TCN model (simplified) ----------------
class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__(); self.chomp = chomp
    def forward(self, x): return x[:, :, :-self.chomp].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU(); self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU(); self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, channels=(32,32), kernel=2, dropout=0.1):
        super().__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            in_ch = input_size if i==0 else channels[i-1]
            out_ch = channels[i]
            dilation = 2**i
            padding = (kernel-1)*dilation
            layers.append(TemporalBlock(in_ch, out_ch, kernel=kernel, dilation=dilation, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(channels[-1], 32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x):
        # x: B x seq_len x features -> B x features x seq_len
        x = x.transpose(1,2)
        y = self.network(x)
        last = y[:, :, -1]  # B x channels[-1]
        return self.fc(last).squeeze(1)

# ---------------- Federated Client ----------------
class FLClient:
    def __init__(self, client_id: str, df: pd.DataFrame, feature_cols: List[str], target_col: str, seq_len: int,
                 device='cpu', batch_size=32, local_epochs=1, lr=1e-3):
        self.client_id = client_id
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.device = device
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lr = lr

        # build windows and scaler (scaler fitted on local train portion)
        # Simple split: keep last 10% for local test
        n = len(self.df)
        cutoff = int(n*0.8) if n>10 else int(n*0.7)
        self.df_train = self.df.iloc[:cutoff].reset_index(drop=True)
        self.df_test = self.df.iloc[cutoff:].reset_index(drop=True)
        X_tr, y_tr, self.scaler = make_windows_from_df(self.df_train, self.feature_cols, self.target_col, self.seq_len, scaler=None, fit_scaler=True)
        X_te, y_te, _ = make_windows_from_df(self.df_test, self.feature_cols, self.target_col, self.seq_len, scaler=self.scaler, fit_scaler=False)
        if X_tr is None:
            raise ValueError(f"Client {client_id}: insufficient data after windowing (seq_len={seq_len})")
        self.train_loader = DataLoader(WindowDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(WindowDataset(X_te, y_te), batch_size=self.batch_size, shuffle=False) if X_te is not None else None
        self.n_samples = len(y_tr)

        # init local model
        self.model = TCN(input_size=len(self.feature_cols)).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def get_weights(self):
        # return CPU state_dict
        return {k: v.cpu().clone() for k,v in self.model.state_dict().items()}

    def set_weights(self, state_dict):
        # load weights (state_dict can be CPU tensors)
        self.model.load_state_dict(state_dict)

    def local_train(self):
        self.model.train()
        for epoch in range(self.local_epochs):
            total_loss = 0.0
            for xb, yb in self.train_loader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
                total_loss += loss.item()*xb.size(0)
        return total_loss / max(1, self.n_samples)

    def evaluate_local(self):
        if self.test_loader is None:
            return {}
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb = xb.to(self.device)
                preds = self.model(xb).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(yb.numpy())
        return {'rmse': rmse(y_true, y_pred), 'mae': mean_absolute_error(y_true, y_pred), 'r2': r2_score(y_true, y_pred)}

# ---------------- Federated Server ----------------
class FLServer:
    def __init__(self, clients: List[FLClient], global_model: nn.Module=None, device='cpu'):
        self.clients = clients
        self.device = device
        if global_model is None:
            # initialize global model from first client structure
            self.global_model = TCN(input_size=len(clients[0].feature_cols)).to(self.device)
        else:
            self.global_model = global_model.to(self.device)

    @staticmethod
    def weighted_avg_state_dict(state_dicts: List[Dict], weights: List[float]):
        """FedAvg: weighted average of multiple state_dicts. Each state_dict maps name->tensor (cpu tensors)."""
        avg_state = {}
        total_weight = sum(weights)
        for k in state_dicts[0].keys():
            # start with zeros tensor of appropriate shape and dtype
            avg = None
            for sd, w in zip(state_dicts, weights):
                tensor = sd[k].float() * (w / total_weight)
                if avg is None:
                    avg = tensor.clone()
                else:
                    avg += tensor
            avg_state[k] = avg
        return avg_state

    def aggregate(self, client_states: List[Dict], client_samples: List[int]):
        # client_states: list of state_dicts (cpu)
        # client_samples: list of ints (n_samples per client)
        new_state = self.weighted_avg_state_dict(client_states, client_samples)
        # load into global model
        # BUT ensure dtype/shape compatibility
        self.global_model.load_state_dict({k: v for k,v in new_state.items()})
        return

    def distribute_weights(self):
        return {k: v.cpu().clone() for k,v in self.global_model.state_dict().items()}

    def evaluate_global_on_clients(self):
        metrics = {}
        for c in self.clients:
            # set client model to global weights and evaluate on local test set
            c.set_weights({k:v.cpu().clone() for k,v in self.global_model.state_dict().items()})
            metrics[c.client_id] = c.evaluate_local()
        return metrics

# ---------------- main FL loop ----------------
def simulate_federated_learning(client_csvs: List[str], target_col: str='ETo', seq_len:int=4,
                                rounds:int=10, clients_per_round:int=None, local_epochs:int=1,
                                batch_size:int=32, lr:float=1e-3, device='cpu'):

    # load clients
    clients = []
    for csv_path in client_csvs:
        df = pd.read_csv(csv_path, sep=';')  # assume ; by default for your CSVs; change as needed
        # infer columns
        cols = [c for c in df.columns if c != target_col and c.lower()!='datetime']
        client_id = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"Loading client {client_id} with {len(df)} rows and features {cols}")
        client = FLClient(client_id, df, feature_cols=cols, target_col=target_col, seq_len=seq_len,
                          device=device, batch_size=batch_size, local_epochs=local_epochs, lr=lr)
        clients.append(client)

    server = FLServer(clients, device=device)

    clients_per_round = clients_per_round or len(clients)  # default: all clients each round

    history = {'round': [], 'global_eval': []}

    for r in range(1, rounds+1):
        print(f"\n=== Round {r}/{rounds} ===")
        # select subset of clients (simulate partial participation)
        selected = random.sample(clients, min(clients_per_round, len(clients)))
        client_states = []
        client_samples = []

        # distribute global model to selected clients
        global_weights = server.distribute_weights()
        for c in selected:
            c.set_weights(global_weights)

        # each client trains locally
        for c in selected:
            train_loss = c.local_train()
            print(f"Client {c.client_id} local_train loss: {train_loss:.4f} samples: {c.n_samples}")
            client_states.append(c.get_weights())
            client_samples.append(c.n_samples)

        # server aggregates (FedAvg)
        server.aggregate(client_states, client_samples)
        # evaluate global model on clients' test sets
        metrics = server.evaluate_global_on_clients()
        print("Global evaluation per client (after aggregation):")
        for cid, m in metrics.items():
            print(f"  {cid}: {m}")
        history['round'].append(r)
        history['global_eval'].append(metrics)

    return server, history

# ---------------- example usage ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients_csvs', nargs='+', required=True, help='Paths to CSVs, one per client')
    parser.add_argument('--target', default='ETo')
    parser.add_argument('--seq-len', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--clients-per-round', type=int, default=None)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    set_seed(42)
    simulate_federated_learning(args.clients_csvs, target_col=args.target, seq_len=args.seq_len,
                                rounds=args.rounds, clients_per_round=args.clients_per_round,
                                local_epochs=args.local_epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
