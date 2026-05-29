# ADR-001: Feature Store Strategy

**Title:** Feature Store Architecture for NeuralRetail ML Platform  
**Status:** Accepted  
**Date:** 2026-05-01  
**Author:** Amdox DS Team (AMX-DS-2026-04)  
**Deciders:** Principal ML Architect, Data Engineering Lead, DS Team Lead

---

## Context

NeuralRetail requires a consistent feature pipeline from offline training to online serving across three model domains: demand forecasting, churn prediction, and customer segmentation. Without a centralised feature store:

- **Training-serving skew** is likely — features computed in batch training differ from those computed at inference time, degrading model performance in production by an estimated 5–15% MAPE increase.
- **Point-in-time correctness** cannot be guaranteed, risking future data leakage that inflates training metrics and causes silent failures in production.
- **Feature reuse** is impossible without a registry — each model team recomputes the same RFM and demand features independently, creating duplication and maintenance burden.
- **Online latency** must be sub-10ms for the real-time churn scoring API (FastAPI) — this requires a low-latency key-value store, not a columnar data warehouse.

Evaluated options:

| Option | Train/Serve Consistency | PIT Join | Online Latency | Cost | Maturity |
|--------|------------------------|----------|---------------|------|---------|
| **Feast 0.40 + Redis + S3** | ✅ Excellent | ✅ Yes | ✅ < 5ms | 💰 Low (OSS) | ✅ Production |
| Inline feature computation | ❌ Skew-prone | ❌ No | ✅ Depends | 💰 Very Low | ✅ Simple |
| Tecton (managed) | ✅ Excellent | ✅ Yes | ✅ < 2ms | 💸 High ($$$) | ✅ Production |
| Hopsworks (OSS) | ✅ Good | ✅ Yes | ✅ < 10ms | 💰 Medium | ⚠️ Complex setup |

---

## Decision

**We will use Feast 0.40 with Redis (online store) and local Parquet / S3 (offline store).**

Configuration:
- **Registry:** Local SQLite (`data/registry.db`) in development, PostgreSQL in production.
- **Offline Store:** File-based (Parquet) in Week 1; migrate to Delta Lake source in Week 3.
- **Online Store:** Redis 8 (standalone in dev, Redis Cluster in production).
- **Materialization:** Daily Airflow-triggered batch job (`feast materialize`).
- **Entity Key Serialization:** Version 2 (Feast 0.40 default).

---

## Rationale

1. **Point-in-time joins prevent training-serving skew.** Feast's `get_historical_features()` performs entity-timestamp aligned lookups, ensuring the model never sees future feature values during training. This is the primary safeguard against silent data leakage.

2. **Redis provides sub-10ms online latency.** Benchmarks on the target infrastructure show Redis `HGET` at < 2ms p99, well within the 10ms SLA required by the FastAPI inference endpoint.

3. **S3 / local Parquet is cheap at scale.** Offline feature storage costs are dominated by compute (materialization), not storage. At NeuralRetail's projected 15M daily transactions, S3 storage for 12 months of features is estimated at < $50/month.

4. **Open-source community and ecosystem.** Feast is the most widely adopted open-source feature store (12k+ GitHub stars), with active maintenance, Airflow integration, and comprehensive documentation. Reduces vendor lock-in risk.

5. **On-demand feature views enable real-time derived features.** The `churn_risk_odfv` on-demand view computes `high_risk_flag` and `clv_tier` at request time from materialized RFM features without requiring re-materialization.

---

## Consequences

### Positive
- ✅ **Consistent train/serve features** — single feature definition used for both offline training and online serving.
- ✅ **Point-in-time correct joins** — eliminates future leakage risk in historical training datasets.
- ✅ **Sub-10ms online retrieval** — Redis satisfies real-time scoring SLA.
- ✅ **Feature reuse across teams** — demand, churn, and segmentation models share the same RFM and demand feature views.
- ✅ **Zero vendor lock-in** — fully open-source, self-hosted.

### Negative
- ⚠️ **Infrastructure overhead** — requires Redis and registry management; adds operational complexity vs. inline computation.
- ⚠️ **Redis memory cost** — at 29 features × 5M customers × 8 bytes, online store requires ~1.2 GB RAM for customer features. Budget accordingly.
- ⚠️ **Materialization latency** — daily batch materialization introduces up to 24-hour feature freshness lag for online store. Acceptable for daily churn scoring but not for real-time demand updates (mitigated by on-demand feature views).
- ⚠️ **Feast 0.40 breaking changes** — API changes between Feast minor versions require pinning. Pinned in `pyproject.toml` at `feast==0.40.*`.

---

## Alternatives Rejected

### Inline Feature Computation
- **Reason rejected:** Does not provide point-in-time correctness — training code recomputes features as of training date, but serving code computes as of request date, causing skew. Estimated MAPE degradation: 5–12% in production.

### Tecton (Managed Feature Platform)
- **Reason rejected:** Estimated cost $40,000–$80,000/year for NeuralRetail's projected usage. Proprietary managed service introduces vendor lock-in incompatible with Amdox's open-source-first policy.

### Hopsworks (Community Edition)
- **Reason rejected:** Significantly more complex setup (JVM, Hopsworks Cluster, separate metadata DB). Week 1 velocity would be reduced. Revisit if Feast limitations are encountered in Week 4+.
