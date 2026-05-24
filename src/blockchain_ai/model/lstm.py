import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMWrapper:
    """Sklearn-compatible wrapper around a stacked LSTM regressor."""

    def __init__(
        self,
        sequence_length: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        patience: int = 10,
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.feature_cols: list[str] = []
        self._net: _LSTMNet | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y) -> "LSTMWrapper":
        if isinstance(X, pd.DataFrame):
            self.feature_cols = list(X.columns)
            X = X.values
        X = X.astype(np.float32)
        y_arr = np.array(y.values if hasattr(y, "values") else y, dtype=np.float32)

        n, n_features = X.shape
        seqs, targets = [], []
        for i in range(self.sequence_length, n):
            seqs.append(X[i - self.sequence_length : i])
            targets.append(y_arr[i])
        seqs = np.array(seqs, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        split = max(1, int(len(seqs) * 0.9))
        X_train, X_val = seqs[:split], seqs[split:]
        y_train, y_val = targets[:split], targets[split:]

        self._net = _LSTMNet(n_features, self.hidden_size, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        patience = 0

        for _ in range(self.epochs):
            self._net.train()
            perm = np.random.permutation(len(X_train))
            for start in range(0, len(X_train), self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = torch.from_numpy(X_train[idx])
                yb = torch.from_numpy(y_train[idx])
                optimizer.zero_grad()
                loss_fn(self._net(xb), yb).backward()
                optimizer.step()

            self._net.eval()
            with torch.no_grad():
                val_loss = loss_fn(
                    self._net(torch.from_numpy(X_val)),
                    torch.from_numpy(y_val),
                ).item()

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= self.patience:
                    break

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Call fit() before predict().")
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = X.astype(np.float32)
        n, n_features = X.shape

        sequences = np.zeros((n, self.sequence_length, n_features), dtype=np.float32)
        for i in range(n):
            start = max(0, i - self.sequence_length + 1)
            seq = X[start : i + 1]
            sequences[i, self.sequence_length - len(seq) :] = seq

        self._net.eval()
        with torch.no_grad():
            return self._net(torch.from_numpy(sequences)).numpy()
