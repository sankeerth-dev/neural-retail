# NeuralRetail — Sprint Board
**Project:** AMX-DS-2026-04 | **Sprint:** Week 1–2 | **Updated:** 2026-05-22

---

## DONE ✅ (Week 1 — Days 1–7)

### Day 1 — Project Setup & Data Discovery
- [x] `pyproject.toml` — Poetry config with all pinned dependencies
- [x] `ruff.toml` — Linter configuration
- [x] `.pre-commit-config.yaml` — Hooks: ruff, black, commitizen
- [x] `docker-compose.yml` — Full 7-service local dev stack with health checks
- [x] `infrastructure/mlflow/Dockerfile` — MLflow tracking server image
- [x] `infrastructure/streamlit/Dockerfile` — Streamlit dashboard image
- [x] `scripts/validate_setup.py` — 7-service health check script
- [x] `configs/datasets.md` — Dataset catalogue with schemas and download commands

### Day 2 — Data Ingestion Pipeline
- [x] `src/ingestion/config.py` — Path constants and thresholds
- [x] `src/ingestion/spark_ingest.py` — PySpark SparkIngestor with 4 source ingestors
- [x] `src/ingestion/lineage.py` — OpenLineage emit_start / emit_complete
- [x] `dags/dag_bronze_ingestion.py` — Airflow DAG with SLA and retry config
- [x] `configs/ge_suite_bronze.py` — GE expectation suites + DQThresholdError
- [x] `tests/unit/test_ingestion.py` — 4 unit tests with moto S3 mock

### Day 3 — EDA & Feature Engineering
- [x] `notebooks/01_eda.py` — 7-cell EDA: distributions, correlation, STL, missing values, deciles, ydata-profiling
- [x] `src/features/feature_engineering.py` — Polars FeatureEngineer (RFM, demand, date, external join)
- [x] `src/features/external_data.py` — fetch_weather (Open-Meteo + backoff) + fetch_cpi
- [x] `configs/holiday_calendar.json` — 37 retail/national holidays 2024–2026
- [x] `src/features/silver_writer.py` — Delta MERGE SilverWriter + MLflow stats logging

### Day 4 — Feast Feature Store & DQ Automation
- [x] `configs/feature_store.yaml` — Feast config (Redis online, file offline)
- [x] `src/features/feast_definitions.py` — Entities, 4 feature views, churn_risk ODFV
- [x] `src/features/materialize.py` — Materialization, online retrieval, historical PIT joins
- [x] `dags/dag_dq_checkpoint.py` — DQ checkpoint DAG triggered by bronze ingestion
- [x] `src/features/profiling.py` — ydata-profiling HTML reports with summary dict
- [x] `tests/unit/test_feast_features.py` — 4 Feast unit tests

### Day 5 — Baseline Models & MLflow Setup
- [x] `infrastructure/mlflow/setup_experiments.py` — Create 4 MLflow experiments
- [x] `src/models/forecasting/baseline_prophet.py` — BaselineProphetForecaster with cross-validation
- [x] `src/models/churn/baseline_logistic.py` — BaselineChurnClassifier with ROC + CM plots
- [x] `src/models/run_baseline_experiments.py` — CLI runner with summary table and hard gates
- [x] `configs/model_card_template.json` — Model documentation schema
- [x] `tests/unit/test_baseline_models.py` — 4 baseline model unit tests

### Day 6 — Time-Series Deep Dive & Prophet HPO
- [x] `notebooks/02_timeseries_analysis.py` — 6-cell analysis: ADF/KPSS, STL, ACF/PACF, FFT, MAPE curves
- [x] `src/models/forecasting/prophet_hpo.py` — Grid-search HPO with parallel ProcessPoolExecutor
- [x] `src/models/forecasting/prophet_dual_season.py` — DualSeasonalityProphet with weekly+annual Fourier
- [x] `src/models/forecasting/arima_benchmark.py` — ARIMA vs Prophet vs auto_arima comparison
- [x] `tests/unit/test_timeseries.py` — 4 time-series unit tests

### Day 7 — Week 1 Checkpoint & Documentation
- [x] `docs/week1_review.md` — DQ scores, model results, HPO params, feature inventory, blockers
- [x] `docs/adr/ADR-001-feature-store-strategy.md` — Full ADR: Feast 0.40 + Redis decision
- [x] `docs/sprint_board.md` — This board
- [x] `src/models/week1_mlflow_summary.py` — MLflow summary script with tabulate output
- [x] `tests/integration/test_week1_pipeline.py` — 5 integration tests (Docker stack required)

---

## IN PROGRESS 🔄

*(Sprint 2 starts Day 8 — currently idle)*

---

## NEXT 📋 (Week 2 — Days 8–14)

### Day 8 — LSTM Demand Forecasting
- [ ] LSTM encoder-decoder in PyTorch Lightning (seq2seq, look-back 60d, horizon 30d)
- [ ] Dataset class with sliding window and normalisation
- [ ] Training loop with early stopping and LR scheduler
- [ ] MLflow PyTorch Lightning logger integration
- [ ] `tests/unit/test_lstm_model.py`

### Day 9 — Prophet + LSTM Ensemble
- [ ] Weighted ensemble combiner class (Optuna weight optimisation)
- [ ] Out-of-fold ensemble training to prevent leakage
- [ ] Ensemble evaluation: MAPE target ≤ 10% on 30-day horizon
- [ ] MLflow child run for each ensemble weight trial
- [ ] `configs/ensemble_weights.json`

### Day 10 — XGBoost Churn (Optuna HPO)
- [ ] XGBoostChurnClassifier with tabular features (RFM + behavioural)
- [ ] Optuna study with 200 trials, MedianPruner, sampler=TPESampler
- [ ] AUC-ROC target ≥ 0.90
- [ ] SHAP TreeExplainer feature importance per customer segment
- [ ] `tests/unit/test_xgboost_churn.py`

### Day 11 — LightGBM Churn + Stacking
- [ ] LightGBMChurnClassifier with dart booster and leaf-wise growth
- [ ] Stacking meta-learner (LogisticRegression L2 on OOF predictions)
- [ ] Calibration (Platt scaling) for probability output
- [ ] `tests/unit/test_lightgbm_stacking.py`

### Day 12 — SHAP Explainability
- [ ] SHAP TreeExplainer for XGBoost + LightGBM churn models
- [ ] SHAP DeepExplainer for LSTM demand model
- [ ] Captum integrated gradients for LSTM
- [ ] Waterfall plots, beeswarm plots, dependence plots to MLflow
- [ ] `notebooks/03_shap_explainability.py`

### Day 13 — Customer Segmentation
- [ ] K-Means (k=3–10, elbow + silhouette selection)
- [ ] DBSCAN (eps grid search, min_samples=5)
- [ ] Gaussian Mixture Model (BIC model selection)
- [ ] Segment profiling: revenue, churn risk, product affinity per cluster
- [ ] `notebooks/04_customer_segmentation.py`

### Day 14 — Price Elasticity + Week 2 Checkpoint
- [ ] DoWhy causal graph: price → sales with confounders (promotion, season, CPI)
- [ ] EconML DML estimator for heterogeneous price elasticity by segment
- [ ] Elasticity heatmap by product × customer segment
- [ ] Week 2 checkpoint: integration tests, model registry promotion (Staging → Production)
- [ ] Streamlit dashboard v1: KPI cards, forecast chart, churn leaderboard
- [ ] `docs/week2_review.md`
