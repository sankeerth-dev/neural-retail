"""Data profiling utilities for NeuralRetail using ydata-profiling.

Generates comprehensive HTML profile reports for any Pandas DataFrame and
returns a summary dict for logging and monitoring purposes.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def generate_profile(
    df: pd.DataFrame,
    title: str,
    output_dir: str,
) -> dict[str, Any]:
    """Generate a ydata-profiling HTML report for a DataFrame.

    Runs a full (non-minimal) explorative profile and saves the HTML output
    to output_dir/{title}_{date}.html. Returns a summary dictionary
    suitable for logging to MLflow or Prometheus.

    Args:
        df: Pandas DataFrame to profile.
        title: Human-readable title for the profile report.
        output_dir: Directory path where the HTML report will be saved.
            Created automatically if it does not exist.

    Returns:
        Summary dict with keys:
            n_rows (int): Number of rows in the DataFrame.
            n_columns (int): Number of columns.
            missing_cells_pct (float): Percentage of missing cells overall.
            duplicate_rows_pct (float): Percentage of duplicate rows.
            correlation_warning_count (int): Number of high-correlation pairs.
            output_path (str): Absolute path to the generated HTML file.

    Example:
        >>> summary = generate_profile(df, "POS Bronze Daily", "reports/")
        >>> print(summary["missing_cells_pct"])
        0.12
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    today_str = date.today().isoformat()
    safe_title = title.replace(" ", "_").replace("/", "-")
    filename = f"{safe_title}_{today_str}.html"
    output_path = str(Path(output_dir) / filename)

    n_rows, n_columns = df.shape
    total_cells = n_rows * n_columns

    # Compute summary stats without profiling library as fallback
    missing_cells = int(df.isnull().sum().sum())
    missing_cells_pct = round(missing_cells / max(total_cells, 1) * 100, 4)
    duplicate_rows = int(df.duplicated().sum())
    duplicate_rows_pct = round(duplicate_rows / max(n_rows, 1) * 100, 4)

    # Count high-correlation pairs (|r| > 0.95)
    correlation_warning_count = 0
    try:
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr().abs()
            high_corr = (corr > 0.95) & (corr < 1.0)
            correlation_warning_count = int(high_corr.sum().sum() // 2)
    except Exception as exc:
        logger.warning("Correlation analysis failed: %s", exc)

    # Generate full profile report
    try:
        from ydata_profiling import ProfileReport

        profile = ProfileReport(
            df,
            title=title,
            minimal=False,
            explorative=True,
            dark_mode=True,
            progress_bar=False,
        )
        profile.to_file(output_path)
        logger.info("Profile report saved: %s", output_path)

        # Attempt to extract richer stats from the profile
        try:
            desc = profile.get_description()
            table_stats = desc.table
            missing_cells_pct = round(
                getattr(table_stats, "p_cells_missing", missing_cells_pct / 100) * 100, 4
            )
            duplicate_rows_pct = round(
                getattr(table_stats, "p_duplicates", duplicate_rows_pct / 100) * 100, 4
            )
        except Exception:
            pass  # Fallback values already set above

    except ImportError:
        logger.warning(
            "ydata_profiling not installed — generating lightweight HTML summary instead"
        )
        _write_lightweight_html(df, title, output_path)
    except Exception as exc:
        logger.error("Profile generation failed: %s", exc)
        _write_lightweight_html(df, title, output_path)

    summary: dict[str, Any] = {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "missing_cells_pct": missing_cells_pct,
        "duplicate_rows_pct": duplicate_rows_pct,
        "correlation_warning_count": correlation_warning_count,
        "output_path": os.path.abspath(output_path),
    }

    logger.info(
        "Profile summary: rows=%d cols=%d missing=%.2f%% duplicates=%.2f%% corr_warnings=%d",
        n_rows,
        n_columns,
        missing_cells_pct,
        duplicate_rows_pct,
        correlation_warning_count,
    )

    return summary


def _write_lightweight_html(df: pd.DataFrame, title: str, output_path: str) -> None:
    """Write a minimal HTML summary when ydata_profiling is unavailable.

    Args:
        df: DataFrame to summarise.
        title: Report title string.
        output_path: Full path to write the HTML file.
    """
    describe_html = df.describe(include="all").to_html(classes="table")
    html = f"""<!DOCTYPE html>
<html>
<head><title>{title}</title>
<style>body{{font-family:sans-serif;padding:20px}}.table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:8px}}</style>
</head>
<body>
<h1>{title}</h1>
<p><strong>Rows:</strong> {len(df):,} | <strong>Columns:</strong> {len(df.columns)}</p>
<h2>Descriptive Statistics</h2>
{describe_html}
</body></html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Lightweight HTML summary written: %s", output_path)
