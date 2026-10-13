"""Synthetic data for UI development ONLY.

Everything here is invented by a random generator. It is not produced by the
team's pipeline and must never be presented as a real finding.
"""
from __future__ import annotations

import random

import pandas as pd

from models.schema import DEMO_LABEL, RESULT_COLUMNS

_ACTIONS = {
    "Low": "[DEMO] Routine annual review.",
    "Medium": "[DEMO] Request AI usage disclosure from vendor.",
    "High": "[DEMO] Schedule security review and request sub-processor list.",
    "Critical": "[DEMO] Escalate to risk committee and review contract terms.",
}


def generate_demo_results(n: int = 24, seed: int = 42) -> pd.DataFrame:
    """Return n clearly labelled synthetic rows (fixed seed = repeatable)."""
    rng = random.Random(seed)
    rows = []
    for i in range(1, n + 1):
        criticality = rng.randint(1, 10)
        uses_ai = rng.random() < 0.6
        score = criticality + (3 if uses_ai else 0) + rng.randint(-1, 1)
        if score >= 12:
            tier = "Critical"
        elif score >= 9:
            tier = "High"
        elif score >= 6:
            tier = "Medium"
        else:
            tier = "Low"

        if uses_ai:
            evidence = "[DEMO] Synthetic placeholder: pretend evidence of AI usage."
            urls = f"https://example.com/demo/vendor-{i:02d}/evidence"
            subs = "; ".join(f"Demo Sub-processor {c}" for c in "ABC"[: rng.randint(1, 3)])
        else:
            evidence = "[DEMO] Synthetic placeholder: no AI usage evidence."
            urls = ""
            subs = ""

        rows.append(
            {
                "Vendor": f"Demo Vendor {i:02d}",
                "Criticality": criticality,
                "AI Usage Evidence": evidence,
                "Source URLs": urls,
                "Sub-processors": subs,
                "Risk Tier": tier,
                "Recommended Action": _ACTIONS[tier],
                "Data Label": DEMO_LABEL,
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS + ["Data Label"])
