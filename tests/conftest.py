"""Pytest configuration and shared fixtures for NeuralRetail tests."""

from __future__ import annotations

import pytest


def pytest_configure(config) -> None:
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests requiring the full Docker Compose stack"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests with no external dependencies"
    )
