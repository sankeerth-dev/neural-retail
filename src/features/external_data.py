"""External data fetchers for NeuralRetail feature engineering.

Provides weather data from Open-Meteo API and CPI index data from a local
CSV or a stub DataFrame. Implements exponential backoff for API resilience.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
CPI_CSV_PATH = Path("configs/cpi_data.csv")
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2.0


def _fetch_with_backoff(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Fetch a URL with exponential backoff retry logic.

    Args:
        url: API endpoint URL.
        params: Query parameters dict.

    Returns:
        Parsed JSON response dict.

    Raises:
        requests.HTTPError: If all retries are exhausted.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, requests.HTTPError) as exc:
            wait = BACKOFF_BASE_SECONDS ** (attempt + 1)
            logger.warning(
                "API request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt + 1,
                MAX_RETRIES,
                exc,
                wait,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
            else:
                logger.error("All %d retries exhausted for URL: %s", MAX_RETRIES, url)
                raise


def fetch_weather(
    store_locations: list[dict[str, Any]],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily weather data for a list of store locations.

    Calls the Open-Meteo API (https://api.open-meteo.com/v1/forecast) for each
    store and returns a combined DataFrame. Adds is_extreme_weather flag when
    temp > 40°C or rain > 50mm.

    Args:
        store_locations: List of dicts with keys:
            store_id (str), latitude (float), longitude (float).
        start_date: ISO format start date string (e.g., "2026-01-01").
        end_date: ISO format end date string (e.g., "2026-01-31").

    Returns:
        DataFrame with columns: date, store_id, temp_c, rain_mm, is_extreme_weather.
        Returns a stub empty DataFrame if the API is unreachable.
    """
    records: list[dict[str, Any]] = []

    for loc in store_locations:
        store_id = loc["store_id"]
        params = {
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "daily": "temperature_2m_mean,precipitation_sum",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "Asia/Kolkata",
        }

        try:
            data = _fetch_with_backoff(OPEN_METEO_URL, params)
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            temps = daily.get("temperature_2m_mean", [None] * len(dates))
            rains = daily.get("precipitation_sum", [None] * len(dates))

            for d, t, r in zip(dates, temps, rains):
                records.append(
                    {
                        "date": pd.to_datetime(d).date(),
                        "store_id": store_id,
                        "temp_c": float(t) if t is not None else None,
                        "rain_mm": float(r) if r is not None else None,
                    }
                )
        except Exception as exc:
            logger.warning(
                "Weather fetch failed for store %s: %s — inserting nulls", store_id, exc
            )
            # Insert null row to maintain date coverage
            for d in pd.date_range(start_date, end_date, freq="D"):
                records.append(
                    {
                        "date": d.date(),
                        "store_id": store_id,
                        "temp_c": None,
                        "rain_mm": None,
                    }
                )

    if not records:
        logger.warning("No weather data fetched — returning empty stub DataFrame")
        return pd.DataFrame(
            columns=["date", "store_id", "temp_c", "rain_mm", "is_extreme_weather"]
        )

    df = pd.DataFrame(records)
    df["is_extreme_weather"] = (df["temp_c"] > 40) | (df["rain_mm"] > 50)
    df["is_extreme_weather"] = df["is_extreme_weather"].fillna(False)

    logger.info(
        "Weather data fetched: %d rows for %d stores from %s to %s",
        len(df),
        len(store_locations),
        start_date,
        end_date,
    )
    return df


def fetch_cpi(
    categories: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch CPI index data for a list of retail categories.

    Loads from configs/cpi_data.csv if available; otherwise returns a stub
    DataFrame with placeholder values.

    Args:
        categories: List of category strings (e.g., ["HOBBIES", "FOODS"]).
        start_date: ISO format start date string.
        end_date: ISO format end date string.

    Returns:
        DataFrame with columns: date, category, cpi_index, cpi_mom_change.
        All category values are present for every month in the date range.
    """
    if CPI_CSV_PATH.exists():
        logger.info("Loading CPI data from %s", CPI_CSV_PATH)
        df = pd.read_csv(CPI_CSV_PATH, parse_dates=["date"])
        df = df[df["category"].isin(categories)]
        df = df[
            (df["date"] >= pd.to_datetime(start_date))
            & (df["date"] <= pd.to_datetime(end_date))
        ]
        logger.info("CPI data loaded: %d rows", len(df))
        return df[["date", "category", "cpi_index", "cpi_mom_change"]]

    # Return stub DataFrame
    logger.warning(
        "CPI data file not found at %s — returning stub DataFrame", CPI_CSV_PATH
    )
    months = pd.date_range(start=start_date, end=end_date, freq="MS")
    records = []
    for month in months:
        for cat in categories:
            records.append(
                {
                    "date": month.date(),
                    "category": cat,
                    "cpi_index": 100.0,  # Baseline CPI
                    "cpi_mom_change": 0.0,
                }
            )

    stub_df = pd.DataFrame(records)
    stub_df["date"] = pd.to_datetime(stub_df["date"])
    return stub_df
