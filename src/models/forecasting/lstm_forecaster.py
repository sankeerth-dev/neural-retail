"""LSTM Demand Forecasting Model using PyTorch Lightning.

Day 8 — NeuralRetail AMX-DS-2026-04
Provides multi-quantile probabilistic demand forecasting with LSTM networks.
Targets MAPE ≤ 10% on 30-day horizon.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class _SlidingWindowDataset(Dataset):
    """Sliding-window dataset for LSTM training.

    Args:
        series: 1-D numpy array of demand values for a single SKU.
        seq_len: Number of historical timesteps fed as input.
        horizon: Number of future timesteps to predict.
    """

    def __init__(self, series: np.ndarray, seq_len: int, horizon: int) -> None:
        self.series = torch.tensor(series, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_windows = len(series) - seq_len - horizon + 1
        if self.n_windows <= 0:
            raise ValueError(
                f"Series length {len(series)} is too short for seq_len={seq_len} "
                f"and horizon={horizon}."
            )

    def __len__(self) -> int:  # noqa: D105
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D105
        x = self.series[idx : idx + self.seq_len].unsqueeze(-1)  # (seq_len, 1)
        y = self.series[idx + self.seq_len : idx + self.seq_len + self.horizon]  # (horizon,)
        return x, y


# ---------------------------------------------------------------------------
# Data Module
# ---------------------------------------------------------------------------

class LSTMDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for SKU-level LSTM demand forecasting.

    Args:
        df: DataFrame with at minimum columns ``[date, sku_id, demand]``.
        sku_id: The SKU identifier to filter on.
        seq_len: Length of the look-back window (default 28 days).
        horizon: Forecast horizon in days (default 30).
        batch_size: Mini-batch size (default 64).
        val_split: Fraction of windows held out for validation (default 0.2).
        num_workers: DataLoader worker count (default 4).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sku_id: str | int | None = None,
        seq_len: int = 28,
        horizon: int = 30,
        batch_size: int = 64,
        val_split: float = 0.2,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.df = df
        self.sku_id = sku_id
        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Build sliding-window datasets ensuring no future leakage.

        The split is performed *chronologically*: the first
        ``(1 - val_split)`` fraction of windows go to training and the
        remainder go to validation. This prevents any future data leaking
        into the training set.

        Args:
            stage: Ignored; required by Lightning interface.
        """
        if self.sku_id is not None:
            series_df = self.df[self.df["sku_id"] == self.sku_id].sort_values("date")
        else:
            series_df = self.df.sort_values("date")

        series = series_df["demand"].to_numpy(dtype=np.float32)

        # Normalise (min-max over training portion to avoid look-ahead)
        n_total_windows = len(series) - self.seq_len - self.horizon + 1
        if n_total_windows <= 0:
            raise ValueError("Not enough data points to create any windows.")

        n_train = max(1, int(n_total_windows * (1 - self.val_split)))
        n_val = n_total_windows - n_train

        # Compute normalisation constants from training region only
        train_end_idx = n_train + self.seq_len + self.horizon - 1
        train_series = series[:train_end_idx]
        self._mean = float(train_series.mean())
        self._std = float(train_series.std()) + 1e-8

        norm_series = (series - self._mean) / self._std
        full_dataset = _SlidingWindowDataset(norm_series, self.seq_len, self.horizon)

        # Chronological split (no shuffle)
        self._train_dataset = torch.utils.data.Subset(full_dataset, list(range(n_train)))
        self._val_dataset = torch.utils.data.Subset(
            full_dataset, list(range(n_train, n_train + n_val))
        )
        logger.info(
            "LSTMDataModule.setup: sku=%s  train_windows=%d  val_windows=%d",
            self.sku_id,
            n_train,
            n_val,
        )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader with pin_memory for GPU efficiency."""
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader with pin_memory for GPU efficiency."""
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    @property
    def normalisation_params(self) -> dict[str, float]:
        """Return ``{mean, std}`` used during setup for de-normalisation."""
        return {"mean": self._mean, "std": self._std}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMModel(pl.LightningModule):
    """Multi-layer LSTM with quantile head for probabilistic demand forecasting.

    Outputs three quantiles (P10, P50, P90) simultaneously via pinball loss,
    enabling prediction-interval estimation for safety-stock calculations.

    Args:
        input_size: Number of input features per timestep (default 1).
        hidden_size: LSTM hidden state dimensionality (default 128).
        num_layers: Number of stacked LSTM layers (default 2).
        dropout: Dropout probability between LSTM layers (default 0.2).
        horizon: Forecast horizon in timesteps (default 30).
        learning_rate: AdamW initial learning rate (default 1e-3).
        quantiles: List of quantile levels to regress (default [0.1, 0.5, 0.9]).
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 30,
        learning_rate: float = 1e-3,
        quantiles: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
        self.horizon = horizon
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        # Output head: (horizon * n_quantiles) units
        self.fc = nn.Linear(hidden_size, horizon * len(self.quantiles))

        # Track epoch-level losses for MLflow logging
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Tensor of shape ``(batch, horizon, n_quantiles)`` representing
            predicted quantiles at each future timestep.
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)  # (batch, horizon * n_quantiles)
        out = out.view(-1, self.horizon, len(self.quantiles))  # (batch, horizon, n_q)
        return out

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _pinball_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute pinball (quantile regression) loss averaged over all quantiles.

        Args:
            preds: Shape ``(batch, horizon, n_quantiles)``.
            targets: Shape ``(batch, horizon)``.

        Returns:
            Scalar pinball loss averaged across quantiles and timesteps.
        """
        targets_expanded = targets.unsqueeze(-1).expand_as(preds)  # (batch, horizon, n_q)
        errors = targets_expanded - preds  # (batch, horizon, n_q)
        losses = []
        for i, q in enumerate(self.quantiles):
            q_tensor = torch.tensor(q, dtype=torch.float32, device=preds.device)
            pinball = torch.where(errors[:, :, i] >= 0, q_tensor * errors[:, :, i], (q_tensor - 1) * errors[:, :, i])
            losses.append(pinball.mean())
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Training / Validation
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute pinball loss on a training batch.

        Args:
            batch: Tuple of (x, y) tensors.
            batch_idx: Batch index (unused).

        Returns:
            Scalar pinball loss.
        """
        x, y = batch
        preds = self(x)
        loss = self._pinball_loss(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Compute val_mape and val_rmse on a validation batch.

        Args:
            batch: Tuple of (x, y) tensors.
            batch_idx: Batch index (unused).
        """
        x, y = batch
        preds = self(x)
        loss = self._pinball_loss(preds, y)

        # P50 (median quantile index 1) for point-forecast metrics
        p50_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else 1
        p50 = preds[:, :, p50_idx]  # (batch, horizon)

        # MAPE — avoid division by zero
        mape = torch.mean(torch.abs((y - p50) / (torch.abs(y) + 1e-8))) * 100.0
        rmse = torch.sqrt(torch.mean((y - p50) ** 2))

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mape", mape, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        """Collect per-epoch val loss for training curve logging."""
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self._val_losses.append(float(val_loss))
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self._train_losses.append(float(train_loss))

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW optimiser with CosineAnnealingWarmRestarts scheduler.

        Returns:
            Dictionary with ``optimizer`` and ``lr_scheduler`` keys compatible
            with PyTorch Lightning's scheduler interface.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------

def train_sku(
    sku_id: str | int,
    df: pd.DataFrame,
    mlflow_experiment_id: str,
    seq_len: int = 28,
    horizon: int = 30,
    max_epochs: int = 50,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    sku_tier: str = "A",
) -> tuple[LSTMModel, dict[str, Any]]:
    """Train an LSTM forecaster for a single SKU and log to MLflow.

    Trains with early stopping (patience=5 on val_mape) and gradient clipping.
    Logs hyperparameters, training curves, per-horizon MAPE metrics, and
    registers the model in MLflow as ``lstm_forecaster_{sku_tier}`` (Staging).

    Args:
        sku_id: SKU identifier used to filter ``df``.
        df: DataFrame containing columns ``[date, sku_id, demand]``.
        mlflow_experiment_id: Target MLflow experiment ID.
        seq_len: Input sequence length in days.
        horizon: Forecast horizon in days.
        max_epochs: Maximum training epochs (early stopping may fire earlier).
        hidden_size: LSTM hidden dimensionality.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability.
        batch_size: Training mini-batch size.
        learning_rate: AdamW initial learning rate.
        sku_tier: Tier label for model registry (e.g. "A", "B", "C").

    Returns:
        Tuple of ``(trained_model, metrics_dict)`` where ``metrics_dict``
        contains ``val_mape_1d``, ``val_mape_7d``, ``val_mape_30d``, and
        ``val_rmse``.
    """
    mlflow.set_experiment(mlflow_experiment_id)

    with mlflow.start_run(run_name=f"lstm_sku_{sku_id}") as run:
        # --- Hyperparams ---
        hparams = {
            "sku_id": str(sku_id),
            "seq_len": seq_len,
            "horizon": horizon,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "sku_tier": sku_tier,
        }
        mlflow.log_params(hparams)

        # --- DataModule ---
        data_module = LSTMDataModule(
            df=df,
            sku_id=sku_id,
            seq_len=seq_len,
            horizon=horizon,
            batch_size=batch_size,
        )
        data_module.setup()

        # --- Model ---
        model = LSTMModel(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            learning_rate=learning_rate,
            quantiles=[0.1, 0.5, 0.9],
        )

        # --- Callbacks ---
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_mape", patience=5, mode="min", verbose=True
        )
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="val_mape", mode="min", save_top_k=1, filename="best-{epoch:02d}-{val_mape:.2f}"
        )

        # --- Trainer ---
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gradient_clip_val=1.0,
            callbacks=[early_stopping, checkpoint],
            enable_progress_bar=True,
            log_every_n_steps=1,
            enable_model_summary=True,
        )
        trainer.fit(model, datamodule=data_module)

        # --- Log training curves ---
        for epoch_idx, (tr_loss, vl_loss) in enumerate(
            zip(model._train_losses, model._val_losses)
        ):
            mlflow.log_metric("train_loss", tr_loss, step=epoch_idx)
            mlflow.log_metric("val_loss", vl_loss, step=epoch_idx)

        # --- Evaluate per-horizon MAPE ---
        val_loader = data_module.val_dataloader()
        model.eval()
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                all_preds.append(out.cpu())
                all_targets.append(y.cpu())

        preds_tensor = torch.cat(all_preds, dim=0)  # (N, horizon, 3)
        targets_tensor = torch.cat(all_targets, dim=0)  # (N, horizon)
        p50 = preds_tensor[:, :, 1]  # median

        def _mape_at(h: int) -> float:
            yh = targets_tensor[:, h - 1]
            ph = p50[:, h - 1]
            return float(torch.mean(torch.abs((yh - ph) / (torch.abs(yh) + 1e-8))) * 100.0)

        val_mape_1d = _mape_at(1)
        val_mape_7d = _mape_at(7)
        val_mape_30d = _mape_at(min(30, horizon))

        metrics = {
            "val_mape_1d": val_mape_1d,
            "val_mape_7d": val_mape_7d,
            "val_mape_30d": val_mape_30d,
        }
        mlflow.log_metrics(metrics)

        logger.info(
            "SKU %s | val_mape_1d=%.2f%%  val_mape_7d=%.2f%%  val_mape_30d=%.2f%%",
            sku_id,
            val_mape_1d,
            val_mape_7d,
            val_mape_30d,
        )

        # --- Register model ---
        model_name = f"lstm_forecaster_{sku_tier.upper()}"
        mlflow.pytorch.log_model(model, artifact_path="lstm_model", registered_model_name=model_name)

        client = mlflow.tracking.MlflowClient()
        # Transition to Staging
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if latest_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=latest_versions[0].version,
                stage="Staging",
            )
            logger.info("Registered %s v%s → Staging", model_name, latest_versions[0].version)

    return model, metrics


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model: LSTMModel, X: np.ndarray) -> dict[str, np.ndarray]:
    """Generate probabilistic forecasts from a trained LSTMModel.

    Produces P10, P50, and P90 quantile forecasts suitable for safety-stock
    interval construction.

    Args:
        model: A trained :class:`LSTMModel` instance.
        X: Input array of shape ``(batch, seq_len, input_size)`` or
           ``(seq_len,)`` / ``(seq_len, input_size)`` for single-sample
           inference (will be batch-expanded automatically).

    Returns:
        Dictionary with keys ``p10``, ``p50``, ``p90`` — each a numpy array
        of shape ``(batch, horizon)`` containing the respective quantile
        forecasts in the original (de-normalised) space. Note: caller is
        responsible for de-normalisation if the model was trained on
        normalised targets.
    """
    model.eval()

    if X.ndim == 1:
        X = X.reshape(1, -1, 1)
    elif X.ndim == 2:
        X = X[np.newaxis, :, :]

    x_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        preds = model(x_tensor)  # (batch, horizon, n_quantiles)

    q_indices = {0.1: 0, 0.5: 1, 0.9: 2}
    p10_idx = 0
    p50_idx = 1
    p90_idx = 2
    if model.quantiles:
        sorted_q = sorted(model.quantiles)
        if len(sorted_q) >= 3:
            p10_idx = model.quantiles.index(sorted_q[0])
            p50_idx = model.quantiles.index(sorted_q[1])
            p90_idx = model.quantiles.index(sorted_q[2])

    return {
        "p10": preds[:, :, p10_idx].numpy(),
        "p50": preds[:, :, p50_idx].numpy(),
        "p90": preds[:, :, p90_idx].numpy(),
    }
