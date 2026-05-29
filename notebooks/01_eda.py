# %% [markdown]
# # NeuralRetail — Exploratory Data Analysis
# **Project:** AMX-DS-2026-04 | **Author:** Amdox DS Team | **Date:** 2026-05
#
# This script performs comprehensive EDA on the NeuralRetail bronze Delta tables.
# Run as a Jupyter notebook (using Jupyter or VS Code with Pylance) or as a plain script.

"""EDA notebook (script form) for NeuralRetail bronze layer data."""

# %% Cell 1 — Load bronze Delta tables and sample to Pandas
import logging
import os
import warnings

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BRONZE_POS = os.environ.get("NR_BRONZE_POS", "data/bronze/pos_transactions")
BRONZE_ERP = os.environ.get("NR_BRONZE_ERP", "data/bronze/erp_inventory")
SAMPLE_FRACTION = 0.10
RANDOM_SEED = 42

spark = (
    SparkSession.builder.appName("NeuralRetail-EDA")
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
    .master("local[*]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

logger.info("Loading POS bronze table: %s", BRONZE_POS)
try:
    pos_spark = spark.read.format("delta").load(BRONZE_POS)
    pos_sample = pos_spark.sample(fraction=SAMPLE_FRACTION, seed=RANDOM_SEED).toPandas()
    logger.info("POS sample loaded: %d rows, %d columns", *pos_sample.shape)
except Exception as e:
    logger.warning("POS bronze table not found (%s) — using synthetic data", e)
    import numpy as np
    rng = np.random.default_rng(RANDOM_SEED)
    n = 50_000
    pos_sample = pd.DataFrame({
        "transaction_id": [f"TXN-{i:07d}" for i in range(n)],
        "customer_id": [f"CUST-{rng.integers(1, 5000):05d}" for _ in range(n)],
        "product_id": [f"PROD-{rng.integers(1, 500):04d}" for _ in range(n)],
        "store_id": [f"STORE-{rng.integers(1, 20):02d}" for _ in range(n)],
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="1min"),
        "quantity": rng.integers(1, 50, size=n),
        "unit_price": rng.uniform(10.0, 5000.0, size=n).round(2),
        "total_amount": rng.uniform(10.0, 100_000.0, size=n).round(2),
        "return_flag": rng.choice([False, True], size=n, p=[0.95, 0.05]),
    })

logger.info("Cell 1 complete. POS sample shape: %s", pos_sample.shape)

# %% Cell 2 — Distributions and IQR outlier detection
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NUMERIC_COLS = ["unit_price", "quantity", "total_amount"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("NeuralRetail POS — Numeric Feature Distributions", fontsize=14, fontweight="bold")

for ax, col in zip(axes, NUMERIC_COLS):
    ax.hist(pos_sample[col].dropna(), bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title(col, fontsize=11)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
os.makedirs("notebooks/reports", exist_ok=True)
plt.savefig("notebooks/reports/distributions.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Saved distributions plot.")

# IQR outlier detection
outlier_report = {}
for col in NUMERIC_COLS:
    q1 = pos_sample[col].quantile(0.25)
    q3 = pos_sample[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((pos_sample[col] < lower) | (pos_sample[col] > upper)).sum()
    pct = n_outliers / len(pos_sample) * 100
    outlier_report[col] = {"lower": lower, "upper": upper, "count": int(n_outliers), "pct": round(pct, 2)}
    logger.info("Outliers in %-20s: %6d rows (%.2f%%)", col, n_outliers, pct)

logger.info("Cell 2 complete.")

# %% Cell 3 — Correlation heatmap (top-20 numeric features)
import seaborn as sns

numeric_df = pos_sample.select_dtypes(include=[np.number]).dropna(axis=1)
# Take top-20 by variance
top20_cols = numeric_df.var().nlargest(min(20, numeric_df.shape[1])).index.tolist()
corr_matrix = numeric_df[top20_cols].corr(method="pearson")

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    ax=ax,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 8},
)
ax.set_title("Pearson Correlation — Top-20 Numeric Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("notebooks/reports/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Cell 3 complete. Saved correlation_heatmap.png")

# %% Cell 4 — STL decomposition for top-5 revenue SKUs
from statsmodels.tsa.seasonal import STL

pos_sample["date"] = pd.to_datetime(pos_sample["timestamp"]).dt.date

sku_revenue = (
    pos_sample.groupby("product_id")["total_amount"].sum().nlargest(5).index.tolist()
)
logger.info("Top-5 revenue SKUs: %s", sku_revenue)

fig, axes = plt.subplots(len(sku_revenue), 3, figsize=(18, 4 * len(sku_revenue)))
fig.suptitle("STL Decomposition — Top-5 Revenue SKUs (period=7)", fontsize=13, fontweight="bold")

for i, sku in enumerate(sku_revenue):
    sku_df = (
        pos_sample[pos_sample["product_id"] == sku]
        .groupby("date")["total_amount"]
        .sum()
        .reset_index()
        .set_index("date")
        .asfreq("D", fill_value=0)
    )
    if len(sku_df) < 14:
        logger.warning("Not enough data for STL on SKU %s — skipping", sku)
        continue

    stl = STL(sku_df["total_amount"], period=7, robust=True)
    res = stl.fit()

    axes[i, 0].plot(res.trend, color="#2196F3"); axes[i, 0].set_title(f"{sku} — Trend")
    axes[i, 1].plot(res.seasonal, color="#4CAF50"); axes[i, 1].set_title(f"{sku} — Seasonal")
    axes[i, 2].plot(res.resid, color="#FF5722"); axes[i, 2].set_title(f"{sku} — Residual")
    for ax in axes[i]:
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/reports/stl_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Cell 4 complete. Saved stl_decomposition.png")

# %% Cell 5 — Missing value heatmap using missingno
try:
    import missingno as msno

    fig, ax = plt.subplots(figsize=(14, 6))
    msno.matrix(pos_sample.sample(min(1000, len(pos_sample)), random_state=RANDOM_SEED), ax=ax, sparkline=False)
    ax.set_title("Missing Value Matrix — POS Sample", fontsize=13)
    plt.tight_layout()
    plt.savefig("notebooks/reports/missing_values.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved missing values matrix.")
except ImportError:
    logger.warning("missingno not installed — skipping missing value heatmap")

MISSING_THRESHOLD = 0.05
flagged_cols = []
for col in pos_sample.columns:
    missing_pct = pos_sample[col].isna().mean()
    if missing_pct > MISSING_THRESHOLD:
        flagged_cols.append((col, round(missing_pct * 100, 2)))
        logger.warning("Column '%s' has %.2f%% missing values — FLAGGED", col, missing_pct * 100)

logger.info("Cell 5 complete. Flagged columns: %s", flagged_cols)

# %% Cell 6 — Customer purchase frequency by decile; top-20% revenue customers
customer_stats = (
    pos_sample.groupby("customer_id")
    .agg(
        txn_count=("transaction_id", "count"),
        total_revenue=("total_amount", "sum"),
        avg_basket=("total_amount", "mean"),
    )
    .reset_index()
)

customer_stats["revenue_decile"] = pd.qcut(
    customer_stats["total_revenue"], q=10, labels=False, duplicates="drop"
)

decile_summary = (
    customer_stats.groupby("revenue_decile")
    .agg(
        n_customers=("customer_id", "count"),
        total_revenue=("total_revenue", "sum"),
        avg_txn_count=("txn_count", "mean"),
    )
    .reset_index()
)
decile_summary["revenue_share_pct"] = (
    decile_summary["total_revenue"] / decile_summary["total_revenue"].sum() * 100
).round(2)

logger.info("\nCustomer revenue decile distribution:\n%s", decile_summary.to_string())

# Top-20% revenue customers (top-2 deciles)
top20_revenue_threshold = customer_stats["total_revenue"].quantile(0.80)
top_customers = customer_stats[customer_stats["total_revenue"] >= top20_revenue_threshold]
top20_revenue_share = top_customers["total_revenue"].sum() / customer_stats["total_revenue"].sum() * 100
logger.info(
    "Top-20%% customers (%d): %.1f%% of total revenue",
    len(top_customers),
    top20_revenue_share,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(decile_summary["revenue_decile"].astype(str), decile_summary["total_revenue"], color="#4C72B0")
axes[0].set_title("Total Revenue by Customer Decile"); axes[0].set_xlabel("Decile"); axes[0].set_ylabel("Revenue")
axes[1].bar(decile_summary["revenue_decile"].astype(str), decile_summary["revenue_share_pct"], color="#DD8452")
axes[1].set_title("Revenue Share % by Customer Decile"); axes[1].set_xlabel("Decile"); axes[1].set_ylabel("Share %")
for ax in axes: ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("notebooks/reports/customer_revenue_deciles.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Cell 6 complete.")

# %% Cell 7 — ydata_profiling full report
try:
    from ydata_profiling import ProfileReport

    report = ProfileReport(
        pos_sample.sample(min(5000, len(pos_sample)), random_state=RANDOM_SEED),
        title="NeuralRetail POS Bronze — EDA Profile",
        minimal=False,
        explorative=True,
        dark_mode=True,
    )
    os.makedirs("notebooks/reports", exist_ok=True)
    output_path = "notebooks/reports/eda_report.html"
    report.to_file(output_path)
    logger.info("Cell 7 complete. EDA profile saved to: %s", output_path)
except ImportError:
    logger.warning("ydata_profiling not installed — skipping full profile report")

logger.info("=== EDA script complete. All reports saved to notebooks/reports/ ===")
