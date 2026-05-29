"""Dual-seasonality Prophet forecaster for NeuralRetail high-seasonality SKUs.

Extends BaselineProphetForecaster to add weekly + annual custom seasonality,
Indian national holidays, and additional regressors (temp_c, cpi_index).
Applied only to SKUs classified as high-seasonality in Day 6 analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from prophet import Prophet

from neuralretail.src.models.forecasting.baseline_prophet import BaselineProphetForecaster

logger = logging.getLogger(__name__)

SEASONALITY_CLASSIFICATION_PATH = Path("configs/seasonality_classification.json")


def _load_high_seasonality_skus() -> set[str]:
    """Load the set of high-seasonality SKU IDs from the classification file.

    Returns:
        Set of SKU ID strings classified as high-seasonality.
    """
    if not SEASONALITY_CLASSIFICATION_PATH.exists():
        logger.warning(
            "Seasonality classification file not found at %s. "
            "All SKUs will be treated as high-seasonality.",
            SEASONALITY_CLASSIFICATION_PATH,
        )
        return set()
    with open(SEASONALITY_CLASSIFICATION_PATH) as f:
        classification = json.load(f)
    high_skus = {
        sku_id
        for sku_id, info in classification.items()
        if info.get("category") == "high"
    }
    logger.info("Loaded %d high-seasonality SKUs", len(high_skus))
    return high_skus


class DualSeasonalityProphet(BaselineProphetForecaster):
    """Prophet forecaster with explicit dual (weekly + annual) seasonality.

    Extends the baseline forecaster to model both weekly and annual seasonal
    patterns using custom Fourier terms, adds Indian national holidays, and
    incorporates weather (temp_c) and macroeconomic (cpi_index) regressors.

    Applied only to SKUs classified as high-seasonality in the Day 6 STL
    analysis (configs/seasonality_classification.json).

    Example:
        >>> dual = DualSeasonalityProphet(horizon_days=30)
        >>> model = dual.train("PROD-0001", sku_df)
    """

    def __init__(
        self,
        horizon_days: int = 30,
        seasonality_mode: str = "multiplicative",
    ) -> None:
        """Initialise DualSeasonalityProphet.

        Args:
            horizon_days: Forecast horizon in days.
            seasonality_mode: Prophet seasonality mode (forced to "multiplicative").
        """
        super().__init__(
            horizon_days=horizon_days,
            seasonality_mode="multiplicative",  # Always multiplicative for dual-season
        )
        self._high_seasonality_skus: set[str] = _load_high_seasonality_skus()

    def train(self, sku_id: str, df: pd.DataFrame) -> Prophet:
        """Fit a dual-seasonality Prophet model for a high-seasonality SKU.

        If the SKU is not in the high-seasonality set, falls back to the
        baseline Prophet training.

        Custom seasonality configuration:
        - Weekly: period=7, fourier_order=3
        - Annual: period=365.25, fourier_order=10
        - Indian national holidays added via add_country_holidays("IN")
        - Regressors: is_promotional_period, temp_c, cpi_index

        Args:
            sku_id: SKU identifier string.
            df: Prophet-format DataFrame with ds and y columns. Must contain
                is_promotional_period, temp_c, and cpi_index columns for
                regressor addition.

        Returns:
            Fitted Prophet model with dual seasonality configuration.
        """
        if (
            self._high_seasonality_skus
            and sku_id not in self._high_seasonality_skus
        ):
            logger.info(
                "SKU %s is not high-seasonality — falling back to baseline Prophet",
                sku_id,
            )
            return super().train(sku_id, df)

        logger.info("Training dual-seasonality Prophet for high-seasonality SKU=%s", sku_id)

        model = Prophet(
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            interval_width=0.90,
            yearly_seasonality=False,   # We add custom annual seasonality below
            weekly_seasonality=False,   # We add custom weekly seasonality below
            daily_seasonality=False,
        )

        # Custom weekly seasonality with higher Fourier order for sharper peaks
        model.add_seasonality(
            name="weekly",
            period=7,
            fourier_order=3,
            prior_scale=10.0,
            mode="multiplicative",
        )

        # Custom annual seasonality with high Fourier order for festival effects
        model.add_seasonality(
            name="annual",
            period=365.25,
            fourier_order=10,
            prior_scale=10.0,
            mode="multiplicative",
        )

        # Indian national holidays (Diwali, Holi, Republic Day, etc.)
        model.add_country_holidays(country_name="IN")

        # Add regressors if present in the training DataFrame
        available_cols = set(df.columns)
        regressors = ["is_promotional_period", "temp_c", "cpi_index"]
        for regressor in regressors:
            if regressor in available_cols:
                model.add_regressor(
                    regressor,
                    standardize=True,
                    mode="multiplicative",
                )
                logger.debug("Added regressor '%s' for SKU=%s", regressor, sku_id)
            else:
                logger.debug(
                    "Regressor '%s' not found in DataFrame — skipping for SKU=%s",
                    regressor,
                    sku_id,
                )

        model.fit(df)
        logger.info(
            "DualSeasonalityProphet fitted for SKU=%s rows=%d regressors=%s",
            sku_id,
            len(df),
            [r for r in regressors if r in available_cols],
        )
        return model
