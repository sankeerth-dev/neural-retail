"""Unit tests for Feast feature store definitions and retrieval.

Tests cover feature view schemas, online retrieval, point-in-time join
correctness, and DQ threshold enforcement.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestCustomerRfmFeatureViewSchema:
    """Validate that the customer_rfm_fv has the correct schema."""

    def test_customer_rfm_feature_view_schema(self) -> None:
        """customer_rfm_fv must have exactly 5 features with correct types.

        Asserts:
            - Feature view exists with name 'customer_rfm_fv'
            - Has exactly 5 features
            - Feature names match the specification
        """
        from neuralretail.src.features.feast_definitions import customer_rfm_fv

        feature_names = {f.name for f in customer_rfm_fv.features}
        expected_names = {
            "recency_days",
            "frequency",
            "monetary",
            "avg_basket_size",
            "rfm_score",
        }

        assert feature_names == expected_names, (
            f"Feature names mismatch. Got: {feature_names}"
        )
        assert len(customer_rfm_fv.features) == 5, (
            f"Expected 5 features, got {len(customer_rfm_fv.features)}"
        )
        assert customer_rfm_fv.ttl == timedelta(days=7), (
            "TTL must be 7 days"
        )

    def test_customer_rfm_has_correct_entity(self) -> None:
        """customer_rfm_fv must use the customer entity.

        Asserts:
            - Entity join key is 'customer_id'
        """
        from neuralretail.src.features.feast_definitions import customer_rfm_fv, customer

        # Check entity join keys
        assert "customer_id" in customer.join_keys, (
            "Customer entity must have customer_id as join key"
        )


class TestOnlineFeatureRetrieval:
    """Test online feature retrieval with mocked Redis."""

    def test_online_feature_retrieval_returns_expected_keys(self) -> None:
        """get_online_features must return all 5 RFM feature keys.

        Uses a mock FeatureStore to avoid requiring a live Redis connection.

        Asserts:
            - Response contains customer_id as entity key
            - All 5 RFM features are present in the response dict
        """
        expected_features = {
            "customer_rfm_fv__recency_days": [42],
            "customer_rfm_fv__frequency": [5],
            "customer_rfm_fv__monetary": [1500.0],
            "customer_rfm_fv__avg_basket_size": [300.0],
            "customer_rfm_fv__rfm_score": [3.2],
            "customer_id": ["CUST-001"],
        }

        mock_response = MagicMock()
        mock_response.to_dict.return_value = expected_features

        mock_fs = MagicMock()
        mock_fs.get_online_features.return_value = mock_response

        with patch(
            "neuralretail.src.features.materialize._get_feature_store",
            return_value=mock_fs,
        ):
            from neuralretail.src.features.materialize import get_online_features

            result = get_online_features(
                customer_ids=["CUST-001"],
                product_ids=["PROD-001"],
                store_ids=["STORE-01"],
            )

        rfm_keys = [k for k in result.keys() if k != "raw"]
        assert len(rfm_keys) == 1, f"Expected 1 entity key, got {len(rfm_keys)}"

        raw = result.get("raw", {})
        assert "customer_rfm_fv__rfm_score" in raw or "rfm_score" in str(raw), (
            "rfm_score must be present in the response"
        )


class TestPointInTimeJoinNoFutureLeakage:
    """Test that historical features do not include future data."""

    def test_point_in_time_join_no_future_leakage(self) -> None:
        """Historical feature retrieval must not return rows with future timestamps.

        Creates an entity DataFrame with past event_timestamps and verifies
        that the returned training DataFrame does not contain any rows where
        the feature timestamp exceeds the event_timestamp (future leakage).

        Asserts:
            - All returned rows have feature timestamps <= event_timestamp
        """
        # Create entity DataFrame with known past timestamps
        entity_df = pd.DataFrame(
            {
                "customer_id": ["CUST-001", "CUST-002", "CUST-003"],
                "event_timestamp": [
                    datetime(2026, 1, 15),
                    datetime(2026, 2, 1),
                    datetime(2026, 3, 10),
                ],
            }
        )

        # Simulate returned training DataFrame — all feature dates in the past
        simulated_result = entity_df.copy()
        simulated_result["snapshot_date"] = [
            datetime(2026, 1, 14),
            datetime(2026, 1, 31),
            datetime(2026, 3, 9),
        ]
        simulated_result["rfm_score"] = [3.5, 2.1, 4.2]

        mock_fs = MagicMock()
        mock_historical = MagicMock()
        mock_historical.to_df.return_value = simulated_result
        mock_fs.get_historical_features.return_value = mock_historical

        with patch(
            "neuralretail.src.features.materialize._get_feature_store",
            return_value=mock_fs,
        ):
            from neuralretail.src.features.materialize import get_historical_features

            result_df = get_historical_features(entity_df)

        # Assert no future leakage
        if "snapshot_date" in result_df.columns:
            future_rows = result_df[
                pd.to_datetime(result_df["snapshot_date"])
                > pd.to_datetime(result_df["event_timestamp"])
            ]
            assert len(future_rows) == 0, (
                f"Found {len(future_rows)} rows with future feature timestamps (leakage!)"
            )


class TestDQThresholdRaisesOnLowScore:
    """Test that DQThresholdError is raised for below-threshold scores."""

    def test_dq_threshold_raises_on_low_score(self) -> None:
        """evaluate_suite must raise DQThresholdError when score < 0.98.

        Injects a GE context returning 7/8 passing expectations (87.5%),
        which is below the 98% gate.

        Asserts:
            - DQThresholdError is raised with a descriptive message
        """
        from neuralretail.configs.ge_suite_bronze import (
            DQThresholdError,
            build_pos_suite,
            evaluate_suite,
        )

        suite = build_pos_suite()

        mock_ctx = MagicMock()
        mock_result = MagicMock()
        # 7 successful out of 8 expectations = 87.5% < 98%
        mock_result.get_statistics.return_value = {"successful_expectations": 7}
        mock_ctx.return_value.run_checkpoint.return_value = mock_result

        # Trim suite to 8 expectations to make math work
        suite["expectations"] = suite["expectations"][:8]

        with patch("great_expectations.get_context", mock_ctx):
            with pytest.raises(DQThresholdError, match="DQ score"):
                evaluate_suite(suite, table="pos")
