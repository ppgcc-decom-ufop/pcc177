"""
tcn_lstm_transformer.py

Implementações básicas em PyTorch para estimativa da Evapotranspiração de Referência (ETo).
Inclui: carregador de CSV, pré-processamento (lags), DataLoader, modelos LSTM, TCN, Transformer,
funções de treino/avaliação e utilitários (métricas).

Uso:
python tcn_lstm_tranformer --csv path/dataset.csv --target ETo --model [lstm | tcn | transformer] --seq-len 4
"""

import argparse
import os
import math
import random
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------- Utilities ---------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


# ----------------------------- Dataset -----------------------------------
class TimeSeriesETODataset(Dataset):
    """Dataset que cria janelas (seq_len) de preditores e o alvo no tempo t.

    CSV deve conter colunas: datetime (opcional), variáveis meteorológicas e a coluna alvo (ETo).
    As colunas a serem usadas como features são deduzidas automaticamente (todas exceto o alvo e opcional datetime),
    ou passadas explicitamente.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 target_col: str = "ETo",
                 feature_cols: Optional[List[str]] = None,
                 seq_len: int = 4,
                 scaler: Optional[StandardScaler] = None,
                 fit_scaler: bool = True):
        # drop rows with na
        self.df = df.copy().dropna().reset_index(drop=True)
        self.target_col = target_col
        if feature_cols is None:
            self.feature_cols = [c for c in self.df.columns if c != target_col and c.lower() != 'datetime']
        else:
            self.feature_cols = feature_cols
        self.seq_len = seq_len

        X = self.df[self.feature_cols].values.astype(float)
        y = self.df[self.target_col].values.astype(float).reshape(-1, 1)

        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler

        if fit_scaler:
            self.scaler.fit(X)

        Xs = self.scaler.transform(X)

        sequences = []
        targets = []
        for i in range(self.seq_len, len(Xs)):
            seq = Xs[i - self.seq_len:i]  # shape seq_len x num_features
            sequences.append(seq)
            targets.append(y[i, 0])

        self.X = np.stack(sequences)  # N x seq_len x num_features
        self.y = np.array(targets)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).float()


# ----------------------------- Models ------------------------------------
# 1) # Definição do modelo LSTM para regressão
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0,
                            bidirectional=bidirectional)
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(1)


# 2) # Definição do modelo TCN para regressão
class Chomp1d(nn.Module): # Remove os elementos adicinais "do futuro", acrescentados com o padding!
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding) # Porque queremos uma convolução estritamente causal!
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                        dilation=dilation_size, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: B x seq_len x features -> for Conv1d we need B x features x seq_len
        x = x.transpose(1, 2)
        y = self.network(x)
        # Global pooling last time-step
        out = y[:, :, -1]
        return out


class TCNRegressor(nn.Module):
    def __init__(self, input_size, num_channels=(64, 64), kernel_size=2, dropout=0.2):
        super().__init__()
        self.tcn = TemporalConvNet(num_inputs=input_size, num_channels=list(num_channels), kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(num_channels[-1], 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out = self.tcn(x)
        return self.fc(out).squeeze(1)


# 3) # Definição do modelo Transformer para regressão
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerRegressor(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 1))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        encoded = self.encoder(x)
        last = encoded[:, -1, :]
        return self.fc(last).squeeze(1)


# ----------------------------- Treinamento e Avaliação ---------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys = []
    yps = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        preds = model(xb)
        ys.append(yb.cpu().numpy())
        yps.append(preds.cpu().numpy())
    y_true = np.concatenate(ys).ravel()
    y_pred = np.concatenate(yps).ravel()
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'y_true': y_true,
        'y_pred': y_pred
    }


def fit(model, train_loader, val_loader, device, epochs=50, lr=1e-3, weight_decay=1e-6, patience=8):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float('inf')
    best_state = None
    wait = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_res = evaluate(model, val_loader, device)
        val_loss = val_res['rmse']
        print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | Val RMSE: {val_loss:.4f} | R2: {val_res['r2']:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ----------------------------- Entrada de dados e Execucação -------------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    if 'datetime' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
        except Exception:
            pass
    return df


def build_data_loaders(df: pd.DataFrame, target: str, seq_len: int, batch_size: int,
                       test_size: float = 0.2, val_size: float = 0.1, shuffle: bool = False,
                       feature_cols: Optional[List[str]] = None, scaler: Optional[StandardScaler] = None):
    n = len(df.dropna())
    test_n = int(n * test_size)
    val_n = int(n * val_size)
    train_n = n - val_n - test_n

    df_train = df.iloc[:train_n].reset_index(drop=True)
    df_val = df.iloc[train_n:train_n + val_n].reset_index(drop=True)
    df_test = df.iloc[train_n + val_n:].reset_index(drop=True)

    train_dataset = TimeSeriesETODataset(df_train, target_col=target, feature_cols=feature_cols, seq_len=seq_len, scaler=scaler, fit_scaler=True)

    val_dataset = TimeSeriesETODataset(df_val, target_col=target, feature_cols=feature_cols, seq_len=seq_len, scaler=train_dataset.scaler, fit_scaler=False)
    test_dataset = TimeSeriesETODataset(df_test, target_col=target, feature_cols=feature_cols, seq_len=seq_len, scaler=train_dataset.scaler, fit_scaler=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.scaler


def select_model(name: str, input_size: int, **kwargs):
    name = name.lower()
    if name == 'lstm':
        return LSTMRegressor(input_size=input_size, hidden_size=kwargs.get('hidden_size', 64),
                             num_layers=kwargs.get('num_layers', 2), dropout=kwargs.get('dropout', 0.2),
                             bidirectional=kwargs.get('bidirectional', False))
    elif name == 'tcn':
        return TCNRegressor(input_size=input_size, num_channels=kwargs.get('num_channels', (64,64)),
                            kernel_size=kwargs.get('kernel_size', 2), dropout=kwargs.get('dropout', 0.2))
    elif name == 'transformer':
        return TransformerRegressor(input_size=input_size, d_model=kwargs.get('d_model', 64),
                                    nhead=kwargs.get('nhead', 4), num_layers=kwargs.get('num_layers', 2),
                                    dim_feedforward=kwargs.get('dim_feedforward', 128), dropout=kwargs.get('dropout', 0.1))
    else:
        raise ValueError(f"Modelo desconhecido: {name}")


def run_training(csv_path: str, target: str = 'ETo', model_name: str = 'lstm', seq_len: int = 4, batch_size: int = 64,
                 epochs: int = 50, device: Optional[str] = None):
    df = load_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")

    feature_cols = [c for c in df.columns if c != target and c.lower() != 'datetime']
    print(f"Using features: {feature_cols}")

    train_loader, val_loader, test_loader, scaler = build_data_loaders(df, target, seq_len, batch_size)
    input_size = len(feature_cols)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = select_model(model_name, input_size)
    model.to(device)
    print(model)

    model = fit(model, train_loader, val_loader, device, epochs=epochs)

    print("Evaluating on test set...")
    res = evaluate(model, test_loader, device)
    print(f"Test RMSE: {res['rmse']:.4f}, MAE: {res['mae']:.4f}, R2: {res['r2']:.4f}")

    return res


# ----------------------------- CLI ---------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treinar modelos deep learning para estimativa de ETo (exemplo)')
    parser.add_argument('--csv', type=str, required=True, help='Caminho para arquivo CSV com dados')
    parser.add_argument('--target', type=str, default='ETo', help='Nome da coluna alvo (default ETo)')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm','tcn','transformer'], help='Modelo a treinar')
    parser.add_argument('--seq-len', type=int, default=4, help='Comprimento da sequência (lags)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    run_training(args.csv, target=args.target, model_name=args.model, seq_len=args.seq_len, batch_size=args.batch_size, epochs=args.epochs)
