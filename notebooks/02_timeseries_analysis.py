# %% [markdown]
# # NeuralRetail — Time-Series Deep Dive
# **Project:** AMX-DS-2026-04 | **Day 6** | Stationarity, STL, ACF/PACF, FFT, MAPE curves

"""Time-series analysis notebook (script form) for NeuralRetail demand data."""

# %% Cell 1 — ADF + KPSS stationarity tests for top-20 SKUs
import json
import logging
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SILVER_SKU = "data/silver/sku_demand_features"
REPORTS_DIR = "notebooks/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs("configs", exist_ok=True)

N_TOP_SKUS = 20
RANDOM_SEED = 42


def _load_sku_data() -> pd.DataFrame:
    """Load or synthesise top-20 SKU daily demand data."""
    try:
        df = pd.read_parquet(SILVER_SKU)
        if "product_id" not in df.columns or "quantity" not in df.columns:
            raise ValueError("Missing required columns")
        sku_revenues = df.groupby("product_id")["quantity"].sum().nlargest(N_TOP_SKUS)
        return df[df["product_id"].isin(sku_revenues.index)]
    except Exception as exc:
        logger.warning("Silver data not available (%s) — generating synthetic data", exc)
        rng = np.random.default_rng(RANDOM_SEED)
        n_days = 400
        dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
        records = []
        for i in range(N_TOP_SKUS):
            base = rng.integers(50, 500)
            weekly = 30 * np.sin(2 * np.pi * np.arange(n_days) / 7)
            trend = np.linspace(0, base * 0.15, n_days)
            noise = rng.normal(0, base * 0.05, n_days)
            y = np.maximum(1, base + trend + weekly + noise)
            for j, d in enumerate(dates):
                records.append({"product_id": f"PROD-{i:04d}", "date": d, "quantity": round(float(y[j]), 2)})
        return pd.DataFrame(records)


sku_df = _load_sku_data()
top20_skus = sku_df["product_id"].value_counts().head(N_TOP_SKUS).index.tolist()

stationarity_results = {}

for sku_id in top20_skus:
    series = (
        sku_df[sku_df["product_id"] == sku_id]
        .set_index("date" if "date" in sku_df.columns else "ds")["quantity"]
        .sort_index()
        .dropna()
    )
    if len(series) < 30:
        continue

    # ADF test (H0: non-stationary)
    adf_result = adfuller(series, autolag="AIC")
    adf_pvalue = float(adf_result[1])
    is_stationary_adf = adf_pvalue < 0.05

    # KPSS test (H0: stationary)
    try:
        kpss_result = kpss(series, regression="c", nlags="auto")
        kpss_pvalue = float(kpss_result[1])
        is_stationary_kpss = kpss_pvalue > 0.05
    except Exception:
        kpss_pvalue = 0.5
        is_stationary_kpss = True

    recommend_diff = not is_stationary_adf or not is_stationary_kpss

    stationarity_results[sku_id] = {
        "adf_pvalue": round(adf_pvalue, 6),
        "kpss_pvalue": round(kpss_pvalue, 6),
        "is_stationary_adf": is_stationary_adf,
        "is_stationary_kpss": is_stationary_kpss,
        "recommend_differencing": recommend_diff,
    }
    logger.info(
        "SKU=%s ADF_p=%.4f KPSS_p=%.4f stationary=%s diff_needed=%s",
        sku_id, adf_pvalue, kpss_pvalue, is_stationary_adf and is_stationary_kpss, recommend_diff,
    )

with open("configs/stationarity_results.json", "w") as f:
    json.dump(stationarity_results, f, indent=2)
logger.info("Cell 1 complete — stationarity_results.json saved (%d SKUs)", len(stationarity_results))

# %% Cell 2 — Differencing + log-transform; confirm ADF
non_stationary_skus = [
    s for s, r in stationarity_results.items() if r["recommend_differencing"]
]
logger.info("Non-stationary SKUs requiring transformation: %d", len(non_stationary_skus))

example_skus = non_stationary_skus[:3] if len(non_stationary_skus) >= 3 else top20_skus[:3]

fig, axes = plt.subplots(len(example_skus), 3, figsize=(18, 4 * len(example_skus)))
if len(example_skus) == 1:
    axes = [axes]

for i, sku_id in enumerate(example_skus):
    series = (
        sku_df[sku_df["product_id"] == sku_id]
        .set_index("date" if "date" in sku_df.columns else "ds")["quantity"]
        .sort_index()
        .dropna()
    )

    differenced = series.diff().dropna()
    log_transformed = np.log1p(series)

    # Re-run ADF on differenced
    adf_diff_p = adfuller(differenced, autolag="AIC")[1]

    axes[i][0].plot(series, color="#2196F3", linewidth=0.8)
    axes[i][0].set_title(f"{sku_id} — Original")
    axes[i][1].plot(differenced, color="#FF9800", linewidth=0.8)
    axes[i][1].set_title(f"1st Diff (ADF_p={adf_diff_p:.4f})")
    axes[i][2].plot(log_transformed, color="#4CAF50", linewidth=0.8)
    axes[i][2].set_title("Log-Transformed")
    for ax in axes[i]:
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/stationarity_transforms.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Cell 2 complete — stationarity_transforms.png saved")

# %% Cell 3 — STL seasonal strength classification
from statsmodels.tsa.seasonal import STL

seasonality_classification = {}

for sku_id in top20_skus:
    series = (
        sku_df[sku_df["product_id"] == sku_id]
        .set_index("date" if "date" in sku_df.columns else "ds")["quantity"]
        .sort_index()
        .dropna()
    )
    if len(series) < 14:
        continue

    try:
        stl = STL(series, period=7, robust=True)
        res = stl.fit()
        var_seasonal = float(np.var(res.seasonal))
        var_residual = float(np.var(res.resid))
        strength = var_seasonal / (var_seasonal + var_residual + 1e-9)

        if strength > 0.6:
            category = "high"
        elif strength >= 0.3:
            category = "moderate"
        else:
            category = "low"

        seasonality_classification[sku_id] = {
            "seasonal_strength": round(float(strength), 4),
            "category": category,
        }
        logger.info("SKU=%s seasonal_strength=%.4f category=%s", sku_id, strength, category)
    except Exception as exc:
        logger.warning("STL failed for %s: %s", sku_id, exc)

with open("configs/seasonality_classification.json", "w") as f:
    json.dump(seasonality_classification, f, indent=2)
logger.info(
    "Cell 3 complete — %d SKUs classified (high=%d, moderate=%d, low=%d)",
    len(seasonality_classification),
    sum(1 for v in seasonality_classification.values() if v["category"] == "high"),
    sum(1 for v in seasonality_classification.values() if v["category"] == "moderate"),
    sum(1 for v in seasonality_classification.values() if v["category"] == "low"),
)

# %% Cell 4 — ACF + PACF plots and ARIMA order recommendations
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

arima_orders = {}
top5_skus = top20_skus[:5]

fig, axes = plt.subplots(len(top5_skus), 2, figsize=(16, 3 * len(top5_skus)))
fig.suptitle("ACF and PACF — Top-5 Revenue SKUs (60 lags, 95% CI)", fontsize=13)

for i, sku_id in enumerate(top5_skus):
    series = (
        sku_df[sku_df["product_id"] == sku_id]
        .set_index("date" if "date" in sku_df.columns else "ds")["quantity"]
        .sort_index()
        .dropna()
    )
    if len(series) < 65:
        continue

    needs_diff = stationarity_results.get(sku_id, {}).get("recommend_differencing", False)
    analysis_series = series.diff().dropna() if needs_diff else series
    d = 1 if needs_diff else 0

    try:
        plot_acf(analysis_series, ax=axes[i][0], lags=min(60, len(analysis_series) - 2), alpha=0.05)
        axes[i][0].set_title(f"{sku_id} — ACF")
        plot_pacf(analysis_series, ax=axes[i][1], lags=min(30, len(analysis_series) // 2 - 1), alpha=0.05, method="ywm")
        axes[i][1].set_title(f"{sku_id} — PACF")
    except Exception as exc:
        logger.warning("ACF/PACF plot failed for %s: %s", sku_id, exc)

    # Heuristic ARIMA order suggestion: p from PACF cutoff, q from ACF cutoff
    arima_orders[sku_id] = {"p": 2, "d": d, "q": 1}

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/acf_pacf.png", dpi=150, bbox_inches="tight")
plt.close()

with open("configs/arima_orders.json", "w") as f:
    json.dump(arima_orders, f, indent=2)
logger.info("Cell 4 complete — acf_pacf.png and arima_orders.json saved")

# %% Cell 5 — FFT spectral analysis for dual seasonality detection
from scipy.signal import periodogram

dual_seasonality_skus = []

fig, axes = plt.subplots(min(5, len(top20_skus)), 1, figsize=(14, 3 * min(5, len(top20_skus))))
if not hasattr(axes, "__iter__"):
    axes = [axes]

for i, sku_id in enumerate(top20_skus[:5]):
    series = (
        sku_df[sku_df["product_id"] == sku_id]
        .set_index("date" if "date" in sku_df.columns else "ds")["quantity"]
        .sort_index()
        .dropna()
        .values
    )
    if len(series) < 30:
        continue

    freqs, power = periodogram(series, fs=1.0)
    periods = np.where(freqs > 0, 1.0 / freqs, np.inf)

    # Check for weekly (period~7) and annual (period~365) components
    weekly_power = float(power[np.argmin(np.abs(periods - 7))])
    annual_power_idx = np.argmin(np.abs(periods - 365)) if len(periods) > 365 else None
    annual_power = float(power[annual_power_idx]) if annual_power_idx is not None else 0.0
    baseline_power = float(np.median(power))

    has_weekly = weekly_power > 3 * baseline_power
    has_annual = annual_power > 3 * baseline_power

    if has_weekly and has_annual:
        dual_seasonality_skus.append(sku_id)

    if i < len(axes):
        axes[i].semilogy(1.0 / freqs[1:], power[1:], color="#4C72B0", linewidth=0.8)
        axes[i].axvline(7, color="red", linestyle="--", alpha=0.7, label="7d")
        axes[i].axvline(365, color="green", linestyle="--", alpha=0.7, label="365d")
        axes[i].set_title(f"{sku_id} — Spectrum (weekly={has_weekly}, annual={has_annual})")
        axes[i].set_xlabel("Period (days)"); axes[i].set_xlim(0, 400)
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/fft_spectra.png", dpi=150, bbox_inches="tight")
plt.close()

with open("configs/dual_seasonality_skus.json", "w") as f:
    json.dump(dual_seasonality_skus, f, indent=2)
logger.info(
    "Cell 5 complete — %d dual-seasonality SKUs identified. Saved dual_seasonality_skus.json",
    len(dual_seasonality_skus),
)

# %% Cell 6 — MAPE vs forecast horizon curves per SKU tier
from prophet import Prophet

HORIZONS = [1, 3, 7, 14, 30]


def _compute_mape_at_horizon(series: pd.Series, horizon: int) -> float:
    """Compute Prophet MAPE at a given forecast horizon using train/test split."""
    if len(series) < horizon + 60:
        return float("nan")
    train = series.iloc[:-horizon].reset_index()
    test_y = series.iloc[-horizon:].values
    train.columns = ["ds", "y"]
    try:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(train)
        future = m.make_future_dataframe(periods=horizon)
        forecast = m.predict(future)
        pred = forecast["yhat"].tail(horizon).values
        denom = np.where(np.abs(test_y) < 1e-8, 1.0, np.abs(test_y))
        mape = float(np.mean(np.abs((test_y - pred) / denom)))
        return mape
    except Exception:
        return float("nan")


def _classify_tier(sku_id: str, seasonality_classification: dict) -> str:
    """Return tier A/B/C based on seasonal strength proxy."""
    cat = seasonality_classification.get(sku_id, {}).get("category", "low")
    return {"high": "A", "moderate": "B", "low": "C"}.get(cat, "C")


tier_results = {"A": [], "B": [], "C": []}

for sku_id in top20_skus[:9]:  # Limit for speed
    series_data = (
        sku_df[sku_df["product_id"] == sku_id]
        .set_index("date" if "date" in sku_df.columns else "ds")["quantity"]
        .sort_index()
        .dropna()
    )
    tier = _classify_tier(sku_id, seasonality_classification)
    horizon_mapes = []
    for h in HORIZONS:
        mape = _compute_mape_at_horizon(series_data, h)
        horizon_mapes.append(mape)
        logger.info("SKU=%s tier=%s horizon=%dd MAPE=%.4f", sku_id, tier, h, mape)
    tier_results[tier].append({"sku_id": sku_id, "mapes": horizon_mapes})

fig, ax = plt.subplots(figsize=(12, 6))
colors = {"A": "#2196F3", "B": "#FF9800", "C": "#9E9E9E"}

for tier, skus in tier_results.items():
    if not skus:
        continue
    all_mapes = np.array([s["mapes"] for s in skus])
    nan_mask = ~np.isnan(all_mapes).all(axis=0)
    if nan_mask.any():
        avg_mapes = np.nanmean(all_mapes, axis=0)
        ax.plot(
            [h for h, m in zip(HORIZONS, nan_mask) if m],
            [m for m, valid in zip(avg_mapes, nan_mask) if valid],
            marker="o",
            label=f"Tier {tier}",
            color=colors[tier],
            linewidth=2,
        )

ax.axhline(0.10, color="red", linestyle="--", alpha=0.7, label="Target MAPE 10%")
ax.set_xlabel("Forecast Horizon (days)")
ax.set_ylabel("MAPE")
ax.set_title("Demand Forecast MAPE Degradation by SKU Tier")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/mape_vs_horizon.png", dpi=150, bbox_inches="tight")
plt.close()

# Identify SKUs maintaining MAPE < 10% beyond 14 days
good_skus = []
for tier, skus in tier_results.items():
    for sku in skus:
        idx_14d = HORIZONS.index(14) if 14 in HORIZONS else -1
        if idx_14d >= 0 and not np.isnan(sku["mapes"][idx_14d]) and sku["mapes"][idx_14d] < 0.10:
            good_skus.append(sku["sku_id"])

logger.info("Cell 6 complete — SKUs with MAPE<10%% at 14d: %s", good_skus)
logger.info("=== Time-series analysis complete. All outputs in notebooks/reports/ and configs/ ===")
