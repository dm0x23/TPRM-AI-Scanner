"""The single place the assessment pipeline plugs into the dashboard.

STATUS: CONNECTED to the temporary integration in services/optiv_pipeline/, which is
generated from optiv.ipynb. To switch to the team's official modules later, replace
the import of ``run_pipeline`` below (or point PIPELINE_MODULE at the new package) and
keep ``to_dashboard_results`` returning models.schema.RESULT_COLUMNS. Nothing else in
the app needs to change.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from services.optiv_pipeline import orchestrator
from services.optiv_pipeline.orchestrator import (   # noqa: F401  (re-exported for the app)
    CONTACT_ENV_VAR, PipelineInputError, PipelineRun, resolve_export_columns,
    resolve_input_columns, sources_checked_table,
)

PIPELINE_CONNECTED = True
PIPELINE_NAME = "Optiv notebook pipeline (services/optiv_pipeline, generated from optiv.ipynb)"

# progress_callback(fraction between 0 and 1, message)
ProgressCallback = Callable[[float, str], None]

# The libraries the NLP stage needs. They are imported only when a run starts.
_MODEL_LIBS = ("torch", "transformers")


class PipelineNotConnectedError(RuntimeError):
    """Raised when the pipeline cannot run in this environment."""


def missing_dependencies() -> list[str]:
    return [m for m in _MODEL_LIBS + ("requests", "bs4", "numpy")
            if importlib.util.find_spec(m) is None]


def is_pipeline_connected() -> bool:
    """True when the pipeline code is wired in AND its dependencies are installed.

    This does not mean an assessment has succeeded; the app reports that separately
    from the outcome of the last run.
    """
    return PIPELINE_CONNECTED and not missing_dependencies()


@dataclass
class AssessmentOutput:
    results: pd.DataFrame        # dashboard table (RESULT_COLUMNS + extras)
    run: PipelineRun             # everything the pipeline produced


def _criticality_text(row) -> object:
    return int(row["Criticality"]) if pd.notna(row["Criticality"]) else None


def to_dashboard_results(run: PipelineRun) -> pd.DataFrame:
    """Map pipeline output to the dashboard schema. No values are invented here."""
    f = run.final_df
    out = pd.DataFrame({
        "Vendor ID": f["Vendor_ID"].astype(str),
        "Vendor": f["Vendor Name"],
        "Criticality": f.apply(_criticality_text, axis=1),
        "Criticality Level": f["Criticality Level"],
        "AI Usage Detected (Y/N)": f["AI Usage Detected (Y/N)"],
        "Evidentiary Classification": f.get("Evidentiary Classification"),
        "AI Usage Evidence": f["Evidence Description"],
        "How Vendor May Be Using AI": f["How Vendor May Be Using AI"],
        "Source URLs": f.get("Source_URL"),
        "Sub-processors": f["Named AI Sub-processors / Model Providers"],
        "Risk Tier": f["AI Security Risk Classification"],
        "Risk Rationale": f["Risk Rationale"],
        "Recommended Action": f["Recommended Action"],
        "Assessment Depth Applied": f["Assessment Depth Applied"],
        "Needs Analyst Review": f.get("Needs_Analyst_Review"),
        "Assessment Status": f["Assessment Status"],
        "Assessment Notes": f["Assessment Notes"],
    })
    out.loc[out["Vendor ID"] == orchestrator.V000_ID, "Assessment Notes"] = out.loc[
        out["Vendor ID"] == orchestrator.V000_ID, "Assessment Notes"].map(
        lambda s: "Optiv example record (V-000): assessed for the regression check; its "
                  "pre-filled row is kept unchanged in the completed template" + (f"; {s}" if s else ""))
    tiers = out["Risk Tier"].astype(object)
    out["Risk Tier"] = tiers.where(tiers.notna(), None)   # None = no tier assigned
    out["Source URLs"] = out["Source URLs"].where(out["Source URLs"].notna(), "")
    return out


def run_assessment_full(
    vendors: pd.DataFrame,
    progress_callback: Optional[ProgressCallback] = None,
    *,
    sheet_table: Optional[pd.DataFrame] = None,
    contact_email: Optional[str] = None,
) -> AssessmentOutput:
    """Run the real pipeline on the selected worksheet.

    ``vendors`` is the preview table. ``sheet_table`` is the same sheet with empty
    columns kept; it is used when given so the completed template keeps every column.
    """
    if not PIPELINE_CONNECTED:
        raise PipelineNotConnectedError("The assessment pipeline is not connected.")
    missing = missing_dependencies()
    if missing:
        raise PipelineNotConnectedError(
            "Required packages are not installed: " + ", ".join(missing)
            + ". Run: pip install -r requirements.txt")
    table = sheet_table if sheet_table is not None else vendors
    run = orchestrator.run_pipeline(table.copy(), contact_email=contact_email,
                                    progress=progress_callback)
    return AssessmentOutput(results=to_dashboard_results(run), run=run)


def run_assessment(
    vendors: pd.DataFrame,
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """Original adapter contract: return the dashboard results table only."""
    return run_assessment_full(vendors, progress_callback).results
