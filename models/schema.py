"""Data contract for the dashboard.

This file describes what the DASHBOARD needs, not what the team's pipeline
returns. When the pipeline is integrated, map its output into RESULT_COLUMNS
inside services/pipeline_adapter.py.
"""
from __future__ import annotations

import pandas as pd

RISK_TIERS = ["Low", "Medium", "High", "Critical"]

TIER_COLORS = {
    "Low": "#3FB68B",
    "Medium": "#E3B341",
    "High": "#F0883E",
    "Critical": "#E5534B",
}

RESULT_COLUMNS = [
    "Vendor",
    "Criticality",
    "AI Usage Evidence",
    "Source URLs",
    "Sub-processors",
    "Risk Tier",
    "Recommended Action",
]

DEMO_LABEL = "DEMO DATA — NOT REAL FINDINGS"


class SchemaError(ValueError):
    """Raised when results do not match the dashboard's expected format."""


def validate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Check a results table and return a clean copy.

    Raises SchemaError with a readable message if something is wrong.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaError("Assessment output must be a pandas DataFrame.")
    missing = [c for c in RESULT_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError("Assessment output is missing column(s): " + ", ".join(missing))
    bad = sorted(set(df["Risk Tier"].dropna().astype(str)) - set(RISK_TIERS))
    if bad:
        raise SchemaError(
            "Unknown risk tier value(s): " + ", ".join(bad)
            + ". Allowed: " + ", ".join(RISK_TIERS) + "."
        )
    return df.copy().reset_index(drop=True)


def tier_counts(df: pd.DataFrame | None) -> dict[str, int]:
    """Number of vendors in each risk tier (all four tiers always present)."""
    counts = {t: 0 for t in RISK_TIERS}
    if df is not None and "Risk Tier" in df.columns:
        for tier, n in df["Risk Tier"].value_counts().items():
            if tier in counts:
                counts[tier] = int(n)
    return counts
