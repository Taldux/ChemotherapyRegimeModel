"""
LSTM-based regressor compatible with sklearn's BaseEstimator interface.
Can be used as base_model for S-Learner and T-Learner via sklearn clone().
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin


class _LSTMNet(nn.Module):
    """PyTorch LSTM network for regression on tabular data."""

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # Use final hidden state from top layer
        out = self.fc(h_n[-1])  # (batch, 1)
        return out.squeeze(-1)


class LSTMRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible LSTM regressor for tabular data.

    Accepts 2-D input ``(n_samples, n_features)`` (single-step) **or**
    3-D input ``(n_samples, seq_len, n_features)`` (multi-step sequences).
    When ``fit_panel`` in SLearner / TLearner detects this model it
    reshapes longitudinal data into patient-level sequences so the LSTM
    trains on genuine time-series.

    Parameters
    ----------
    hidden_dim : int
        Hidden state size of the LSTM.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers (ignored when num_layers=1).
    lr : float
        Adam learning rate.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Early-stopping patience (epochs without improvement). 0 = disabled.
    val_fraction : float
        Fraction held out for validation / early stopping.
    random_state : int or None
        Seed for reproducibility.
    device : str
        'auto', 'cpu', or 'cuda'.
    """

    _is_sequential = True

    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.2,
                 lr=1e-3, epochs=200, batch_size=64, patience=15,
                 val_fraction=0.15, random_state=42, device='auto'):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_device(self):
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        dev = self._get_device()
        rng = np.random.RandomState(self.random_state)

        # Accept 2-D (n_samples, features) or 3-D (n_samples, seq_len, features)
        if X.ndim == 3:
            n_samples, seq_len, n_features = X.shape
        else:
            n_samples, n_features = X.shape

        # Normalisation stats computed across all (sample, time-step) pairs
        X_flat = X.reshape(-1, n_features)
        self.x_mean_ = X_flat.mean(axis=0)
        self.x_std_ = X_flat.std(axis=0) + 1e-8
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) + 1e-8

        X_norm = (X - self.x_mean_) / self.x_std_
        y_norm = (y - self.y_mean_) / self.y_std_

        # Reshape to 3-D if needed: (n_samples, seq_len, n_features)
        if X_norm.ndim == 2:
            X_seq = X_norm[:, np.newaxis, :]  # single-step fallback
        else:
            X_seq = X_norm

        # Train / validation split
        if self.patience > 0 and self.val_fraction > 0:
            n_val = max(1, int(n_samples * self.val_fraction))
            idx = rng.permutation(n_samples)
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            X_train, y_train = X_seq[train_idx], y_norm[train_idx]
            X_val = torch.tensor(X_seq[val_idx], device=dev)
            y_val = torch.tensor(y_norm[val_idx], device=dev)
        else:
            X_train, y_train = X_seq, y_norm
            X_val, y_val = None, None

        train_ds = TensorDataset(
            torch.tensor(X_train, device=dev),
            torch.tensor(y_train, device=dev),
        )
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Build network
        torch.manual_seed(self.random_state if self.random_state is not None else 0)
        self.model_ = _LSTMNet(
            n_features, self.hidden_dim, self.num_layers, self.dropout
        ).to(dev)

        optimiser = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val_loss = np.inf
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model_.train()
            for xb, yb in train_dl:
                pred = self.model_(xb)
                loss = criterion(pred, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            # Early stopping on validation set
            if X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_pred = self.model_(X_val)
                    val_loss = criterion(val_pred, y_val).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        # Restore best checkpoint
        if best_state is not None:
            self.model_.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

        self.n_features_in_ = n_features
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_norm = (X - self.x_mean_) / self.x_std_
        if X_norm.ndim == 2:
            X_seq = X_norm[:, np.newaxis, :]  # single-step fallback
        else:
            X_seq = X_norm
        dev = self._get_device()

        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(torch.tensor(X_seq, device=dev))

        # Denormalise
        return preds.cpu().numpy() * self.y_std_ + self.y_mean_

    def predict_with_uncertainty(self, X, n_samples=50):
        """
        @brief Estimate prediction uncertainty via Monte Carlo Dropout.

        Keeps dropout active during inference and runs n_samples stochastic
        forward passes to approximate a predictive distribution.

        @param X          Input array, same shape as accepted by predict().
        @param n_samples  Number of MC samples (default 50).
        @return Tuple (mean, std) each of shape (n_samples_in,).
        """
        X = np.asarray(X, dtype=np.float32)
        X_norm = (X - self.x_mean_) / self.x_std_
        if X_norm.ndim == 2:
            X_seq = X_norm[:, np.newaxis, :]
        else:
            X_seq = X_norm
        dev = self._get_device()
        X_tensor = torch.tensor(X_seq, device=dev)

        self.model_.train()  # keep dropout active
        with torch.no_grad():
            draws = np.stack([
                self.model_(X_tensor).cpu().numpy()
                for _ in range(n_samples)
            ])  # (n_samples, n_obs)
        self.model_.eval()   # restore eval mode

        mean = draws.mean(axis=0) * self.y_std_ + self.y_mean_
        std  = draws.std(axis=0)  * self.y_std_
        return mean, std
