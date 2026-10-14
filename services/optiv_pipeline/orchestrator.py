"""Runs the notebook's ``run_full_pipeline()`` stage by stage, for the Streamlit app.

The assessment rules live in criticality.py, scraper.py, nlp_engine.py and
risk.py (generated from optiv.ipynb). This file only reproduces the *orchestration*
from notebook cell 6 (``run_full_pipeline``), with the technical changes needed to run
outside Colab. Every change is listed here and tagged ``[integration]`` in the code.

Technical changes (assessment logic unaffected)
------------------------------------------------
T1  Input is the worksheet the user chose in the app (a DataFrame), not a hard-coded
    file name and sheet ('Meridian_Vendor_Input_2.xlsx' / 'Test Vendor Input').
T2  The contact e-mail is a parameter (from the app sidebar or the
    MERIDIAN_CONTACT_EMAIL environment variable). The notebook hard-coded a personal
    address. With no contact, the scraper's own rule applies: SEC EDGAR is skipped
    and logged in "sources checked" (EDGAR is used for Criticality 3 and 4 only).
T3  No IPython ``display()`` and no file written to disk: the V-000 comparison and the
    completed template are returned to the caller.
T4  A failure for one vendor (for example a blank name or website, which makes
    ``scrape_vendor`` raise) is recorded for that vendor instead of stopping the run.
T5  Models are loaded once per process and reused across runs.
T6  Output columns are matched to the sheet's headers ignoring case, spacing and
    punctuation, the same normalisation ``assess_criticality`` uses. The notebook
    wrote the date to 'Assessedd By / Date' (typo). If the sheet has a column with
    that name, or one named 'Assessed By / Date', the date goes there; otherwise the
    notebook's 'Assessedd By / Date' column is created.
T7  Vendors scrape concurrently (default 4), with main-thread callbacks, ordered
    results, per-host pacing, exact-input NLP caches and per-stage timings.

Methodology deviations (each can be switched off with a flag below)
-------------------------------------------------------------------
D1  MARK_UNREACHABLE_AS_INCOMPLETE: a vendor with no evidence because none of its
    first-party pages could be fetched (unreachable, blocked by robots.txt, or the
    scrape raised an error) is marked "Not assessed" with no risk tier. The notebook
    filled 'No (0.0% Confidence)' and so rated it Low. A vendor whose site WAS read
    but showed no AI signals still gets the notebook's Low rating, unchanged.
D2  MARK_NLP_FAILURE_AS_INCOMPLETE: if the models cannot be loaded or inference fails,
    vendors that had evidence are marked "Not assessed". The notebook would crash.
D3  If no vendor has any evidence, the notebook printed a message and returned None.
    Here the same per-vendor fill rules are applied (Low for a site that was read and
    had no AI signals, D1 for unreachable sites), so the run still produces a report.
D4  DROP_INTERNAL_SCORING_COLUMNS: ``assess_criticality`` adds helper columns
    (``_dependency_score`` and so on) to the input DataFrame in place. Because the
    notebook copied that DataFrame for export, those helper columns ended up in its
    Excel output, although the code comments say they are internal. They are dropped
    from the completed template. 'Criticality' and 'Criticality Level', which the
    notebook export also contained, are kept.
"""
from __future__ import annotations

import functools
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

from . import criticality as crit_mod
from . import nlp_engine as nlp
from . import risk as risk_mod
from . import scraper as scr

MARK_UNREACHABLE_AS_INCOMPLETE = True   # D1
MARK_NLP_FAILURE_AS_INCOMPLETE = True   # D2
DROP_INTERNAL_SCORING_COLUMNS = True    # D4

V000_ID = "V-000"
NOT_ASSESSED_TIER_TEXT = "Not assessed (incomplete)"
CONTACT_ENV_VAR = "MERIDIAN_CONTACT_EMAIL"

FIRST_PARTY_TYPES = {"homepage", "product", "trust_privacy", "careers", "news_press", "docs_dev"}

# (final_df column, template header candidates). The first candidate is the header the
# notebook wrote to and is created if no candidate exists in the sheet.
EXPORT_COLUMN_MAP = [
    ("Assessment Depth Applied", ["Assessment Depth Applied"]),
    ("AI Usage Detected (Y/N)", ["AI Usage Detected (Y/N)"]),
    ("Evidence Description", ["Evidence Description along with OSINT Source(s) / Evidence Link"]),
    ("How Vendor May Be Using AI", ["How Vendor May Be Using AI"]),
    ("Named AI Sub-processors / Model Providers", ["Named AI Sub-processors / Model Providers"]),
    ("AI Security Risk Classification", ["AI Security Risk Classification"]),
    ("Risk Rationale", ["Risk Rationale"]),
    ("Recommended Action", ["Recommended Action"]),
    ("Assessed By / Date", ["Assessedd By / Date", "Assessed By / Date"]),
]

# Fields compared in the notebook's V-000 regression test (cell 6, step 6).
V000_FIELDS = [
    ("AI Usage Detected", "AI Usage Detected (Y/N)", "AI Usage Detected (Y/N)"),
    ("Risk Classification", "AI Security Risk Classification", "AI Security Risk Classification"),
    ("Named Providers", "Named AI Sub-processors / Model Providers", "Named AI Sub-processors / Model Providers"),
]

ProgressCallback = Callable[[float, str], None]


class PipelineInputError(ValueError):
    """The worksheet cannot be assessed (for example, required columns are missing)."""


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------
@dataclass
class VendorStatus:
    vendor_id: str
    vendor_name: str
    status: str = "Complete"          # Complete | Incomplete
    reason: str = ""
    notes: List[str] = field(default_factory=list)
    evidence_rows: int = 0
    first_party_pages_read: int = 0


@dataclass
class PipelineRun:
    crit_df: pd.DataFrame
    scraper_results: List[dict]
    evidence_df: pd.DataFrame
    nlp_df: pd.DataFrame
    nlp_metrics: Dict[str, object]
    final_df: pd.DataFrame
    export_df: pd.DataFrame
    vendor_status: Dict[str, VendorStatus]
    v000_comparison: Optional[pd.DataFrame]
    column_mapping: Dict[str, str]
    warnings: List[str]
    nlp_error: Optional[str] = None
    stage_timings: Dict[str, float] = field(default_factory=dict)
    contact_used: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm(name) -> str:
    return crit_mod.normalize_column_name(name)


def find_column(df: pd.DataFrame, candidates) -> Optional[str]:
    lookup = {}
    for c in df.columns:
        lookup.setdefault(_norm(c), c)
    for cand in candidates:
        if _norm(cand) in lookup:
            return lookup[_norm(cand)]
    return None


REQUIRED_INPUTS = {
    # mirrors assess_criticality's column_map (cell 2)
    "Vendor ID": ["Vendor ID"],
    "Vendor Name": ["Vendor Name"],
    "Public Website": ["Public Website", "Website", "Vendor Website", "URL", "Domain"],
    "Category": ["Category"],
    "Operational Dependency": ["Operational Dependency"],
    "Data Classification Accessed": ["Data Classification Accessed", "Data Classification"],
    "Data Volume (annual)": ["Data Volume (annual)", "Data Volume Annual", "Data Volume"],
}


def resolve_input_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Which sheet column feeds each pipeline input (None = missing)."""
    return {k: find_column(df, v) for k, v in REQUIRED_INPUTS.items()}


def resolve_export_columns(df: pd.DataFrame) -> Dict[str, str]:
    """final_df column -> template header that will receive it."""
    return {src: (find_column(df, cands) or cands[0]) for src, cands in EXPORT_COLUMN_MAP}


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""


_MODEL_LOCK = threading.RLock()


@functools.lru_cache(maxsize=1)
def _load_models():
    # [integration T5] loaded once; same classes and CFG as the notebook
    return nlp.ZeroShotClassifier(nlp.CFG), nlp.CapabilitySummarizer(nlp.CFG)


def default_model_factory():
    with _MODEL_LOCK:
        return _load_models()


def _coverage(result: dict) -> int:
    """Number of first-party pages that were fetched and read."""
    return sum(1 for n in result.get("sources_checked", [])
               if n.get("source_type") in FIRST_PARTY_TYPES and n.get("outcome") in ("hits", "no_hits"))


def _unreachable_detail(result: dict) -> str:
    for n in result.get("sources_checked", []):
        if n.get("source_type") == "homepage":
            return f"homepage {n.get('outcome')} ({n.get('detail')})"
        if n.get("outcome") == "error":
            return f"{n.get('source_type')} error: {n.get('detail')}"
    return "no first-party page could be fetched"


class _Reporter:
    """Adapts nlp_engine progress updates (0-1) to a slice of the app's progress bar."""

    def __init__(self, cb: Optional[ProgressCallback], lo: float, hi: float):
        self.cb, self.lo, self.hi = cb, lo, hi

    def update(self, frac: float, msg: str) -> None:
        if self.cb:
            self.cb(self.lo + (self.hi - self.lo) * min(max(frac, 0.0), 1.0), msg)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_pipeline(
    sheet_df: pd.DataFrame,
    contact_email: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
    model_factory: Optional[Callable[[], tuple]] = None,
    scrape_fn: Optional[Callable[..., dict]] = None,
    fetcher_factory: Optional[Callable[[Optional[str]], object]] = None,
    scrape_workers: Optional[int] = None,
) -> PipelineRun:
    """Run the Meridian assessment on one worksheet.

    ``sheet_df`` is the selected worksheet with its header row applied, including
    columns that are empty (so the completed template keeps the sheet's layout).
    ``model_factory`` / ``scrape_fn`` / ``fetcher_factory`` exist so tests can inject
    replayed scrape results or stand-in models; the app uses the defaults. Injected
    fetchers must support concurrent get() calls, or set scrape_workers=1. The caller
    owns injected fetcher cleanup. scrape_workers defaults to MERIDIAN_SCRAPE_WORKERS
    (4); progress callbacks always run on the calling thread.
    """
    def report(frac, msg):
        if progress:
            progress(frac, msg)

    started = time.perf_counter()
    stage_timings = {}
    try:
        workers = int(scrape_workers if scrape_workers is not None else
                      os.getenv("MERIDIAN_SCRAPE_WORKERS", "4"))
    except (TypeError, ValueError) as exc:
        raise PipelineInputError("Scrape workers must be an integer between 1 and 16") from exc
    if not 1 <= workers <= 16:
        raise PipelineInputError("Scrape workers must be between 1 and 16")
    warnings_: List[str] = []
    contact = (contact_email or os.getenv(CONTACT_ENV_VAR) or "").strip() or None   # [integration T2]
    scrape_fn = scrape_fn or scr.scrape_vendor
    model_factory = model_factory or default_model_factory

    # ---------------- 1. Criticality (notebook step 1) -------------------------
    report(0.0, "1/5 Assessing criticality…")
    mapping = resolve_input_columns(sheet_df)
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise PipelineInputError(
            "The selected sheet is missing column(s) the assessment needs: " + ", ".join(missing)
            + ". Accepted header names: "
            + "; ".join(f"{k} = {' / '.join(REQUIRED_INPUTS[k])}" for k in missing)
            + ". Check that the right sheet and header row are selected.")
    raw_df = sheet_df.copy()           # plays the role of the notebook's raw_df
    try:
        crit_df = crit_mod.assess_criticality(raw_df)   # mutates raw_df, as in the notebook
    except ValueError as exc:
        raise PipelineInputError(str(exc)) from exc
    crit_df["Vendor ID"] = crit_df["Vendor ID"].astype(str)

    dups = crit_df["Vendor ID"][crit_df["Vendor ID"].duplicated()].unique().tolist()
    if dups:
        warnings_.append("Duplicate Vendor IDs in the sheet: " + ", ".join(dups)
                         + ". Results are merged by Vendor ID, so these rows share one result.")

    # ---------------- 2. Scraping (notebook step 2) ----------------------------
    stage_timings["criticality_s"] = time.perf_counter() - started
    scrape_started = time.perf_counter()
    fetcher = (fetcher_factory or (lambda c: scr.Fetcher(contact=c)))(contact)
    statuses: Dict[str, VendorStatus] = {}
    scraper_results: List[dict] = []
    rows = crit_df.to_dict(orient="records")
    def scrape_one(row):
        return scrape_fn(
            vendor_id=str(row["Vendor ID"]), name=str(row["Vendor Name"]),
            website=str(row["Public Website"]), level=int(row["Criticality"]),
            contact=contact, fetcher=fetcher)

    # Only the main thread mutates statuses and invokes Streamlit callbacks.
    ordered_results = {}
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="meridian-scrape") as pool:
            pending = {}
            for i, row in enumerate(rows):
                vid, name = str(row["Vendor ID"]), row["Vendor Name"]
                st_ = statuses.setdefault(vid, VendorStatus(vid, "" if _blank(name) else str(name)))
                if _blank(vid):
                    st_.status, st_.reason = "Incomplete", "Vendor ID is blank"
                    continue
                if _blank(name) or _blank(row["Public Website"]):
                    st_.status, st_.reason = "Incomplete", "Vendor name or public website is blank"
                    continue
                pending[pool.submit(scrape_one, row)] = (i, vid)
            report(0.05, f"2/5 Scraping {len(pending)} vendors with {workers} workers…")
            for done, future in enumerate(as_completed(pending), 1):
                i, vid = pending[future]
                st_ = statuses[vid]
                try:
                    ordered_results[i] = future.result()
                except Exception as exc:
                    st_.status = "Incomplete"
                    st_.reason = f"Scrape could not start: {type(exc).__name__}: {exc}"
                report(0.05 + 0.50 * done / max(len(pending), 1),
                       f"2/5 Scraped {done}/{len(pending)}: {st_.vendor_name or vid}")
    finally:
        if fetcher_factory is None:
            fetcher.close()

    # Restore worksheet order: completion order must not alter tie-breaking.
    for i in sorted(ordered_results):
        result = ordered_results[i]
        st_ = statuses[str(rows[i]["Vendor ID"])]
        scraper_results.append(result)
        st_.evidence_rows += len(result.get("evidence", []))
        st_.first_party_pages_read += _coverage(result)
        for n in result.get("sources_checked", []):
            if n.get("outcome") == "error":
                st_.notes.append(f"{n.get('source_type')} collector error: {n.get('detail')}")
            if n.get("outcome") == "thin_content":
                st_.notes.append(f"thin content at {n.get('target')} (may be JavaScript-rendered)")

    stage_timings["scraping_s"] = time.perf_counter() - scrape_started

    # ---------------- 3. Flatten (notebook step 3) -----------------------------
    report(0.55, "3/5 Formatting evidence for the NLP engine…")
    evidence_df = risk_mod.flatten_scraper_evidence(scraper_results)

    # ---------------- 4. NLP (notebook step 4) ---------------------------------
    nlp_started = time.perf_counter()
    nlp_df, metrics, nlp_error = None, {}, None
    if evidence_df.empty:
        warnings_.append("No AI keyword evidence was found for any vendor; the NLP stage was skipped. "
                         "(The notebook stopped here and produced no output; see deviation D3.)")
    else:
        try:
            report(0.57, f"4/5 Loading NLP models ({nlp.CFG.nli_model}, {nlp.CFG.gen_model})…")
            # Models and EgressGuard modify process-wide state; serialize NLP runs.
            with _MODEL_LOCK:
                load_started = time.perf_counter()
                clf, summ = model_factory()
                stage_timings["model_loading_s"] = time.perf_counter() - load_started
                audit = nlp.analyze(raw=evidence_df, clf=clf, summarizer=summ, cfg=nlp.CFG,
                                    rep=_Reporter(progress, 0.60, 0.95))
            nlp_df, metrics = audit.df, audit.metrics
        except Exception as exc:   # [integration D2]
            nlp_error = f"{type(exc).__name__}: {exc}"
            warnings_.append("The NLP stage failed, so AI usage could not be verified for vendors with "
                             f"evidence: {nlp_error}")
            if not MARK_NLP_FAILURE_AS_INCOMPLETE:
                raise
    if nlp_df is None:
        nlp_df = pd.DataFrame(columns=_NLP_COLUMNS)

    stage_timings["nlp_s"] = time.perf_counter() - nlp_started

    # ---------------- 5. Aggregate + risk (notebook step 5, unchanged) ---------
    report(0.96, "5/5 Aggregating results and applying the risk matrix…")
    aggregated_rows = []

    # Aggregate to 1 "best" evidence row per vendor
    for vid, group in nlp_df.groupby('Vendor_ID'):
        yes_rows = group[group['AI Usage Detected (Y/N)'].str.startswith('Yes')]
        if not yes_rows.empty:
            best_row = yes_rows.loc[yes_rows['P_Embedded_AI'].idxmax()]
        else:
            best_row = group.loc[group['P_Embedded_AI'].idxmax()]
        aggregated_rows.append(best_row)

    # [integration] columns= keeps the merge working when there are no NLP rows (D2/D3)
    final_nlp_df = pd.DataFrame(aggregated_rows, columns=nlp_df.columns) if not aggregated_rows \
        else pd.DataFrame(aggregated_rows)

    # Merge back with Criticality Data
    final_df = final_nlp_df.merge(crit_df[['Vendor ID', 'Criticality', 'Criticality Level']], left_on='Vendor_ID', right_on='Vendor ID', how='right')

    # Fill NaN for vendors that had absolutely no evidence rows generated
    final_df['Vendor_ID'] = final_df['Vendor_ID'].fillna(final_df['Vendor ID'])
    final_df['AI Usage Detected (Y/N)'] = final_df['AI Usage Detected (Y/N)'].fillna('No (0.0% Confidence)')
    final_df['Context_For_NER'] = final_df['Context_For_NER'].fillna('')
    final_df['Evidence Description'] = final_df['Evidence Description'].fillna('No AI signals detected in public footprint.')
    final_df['How Vendor May Be Using AI'] = final_df['How Vendor May Be Using AI'].fillna('No AI capabilities identified.')

    # Run Stage 4: Risk Phase Additions
    final_df["Assessment Depth Applied"] = final_df["Criticality Level"].apply(lambda x: f"Depth configured to {x} Tier")
    final_df["Named AI Sub-processors / Model Providers"] = final_df["Context_For_NER"].apply(risk_mod.extract_subprocessors)
    final_df[["AI Security Risk Classification", "Risk Rationale", "Recommended Action"]] = final_df.apply(risk_mod.calculate_risk, axis=1)
    final_df["Assessed By / Date"] = f"Optiv Case Team / {datetime.now().strftime('%Y-%m-%d')}"

    # ---------------- Completeness (deviations D1/D2) ---------------------------
    names = dict(zip(crit_df["Vendor ID"], crit_df["Vendor Name"]))
    analysed_ids = set(nlp_df["Vendor_ID"].astype(str)) if len(nlp_df) else set()
    for vid, st_ in statuses.items():
        if st_.status == "Incomplete":
            continue
        if st_.evidence_rows == 0 and st_.first_party_pages_read == 0 and MARK_UNREACHABLE_AS_INCOMPLETE:
            res = next((r for r in scraper_results if str(r["meta"]["vendor_id"]) == vid), {})
            st_.status, st_.reason = "Incomplete", "Website could not be read: " + _unreachable_detail(res)
        elif st_.evidence_rows > 0 and vid not in analysed_ids:
            st_.status = "Incomplete"
            st_.reason = ("NLP verification failed: " + nlp_error) if nlp_error else \
                "Evidence was collected but no NLP result matched this Vendor ID"
        elif st_.first_party_pages_read == 0:
            st_.notes.append("first-party website could not be read; result rests on third-party / "
                             "registry sources only")
        elif st_.evidence_rows == 0:
            st_.notes.append(f"{st_.first_party_pages_read} first-party page(s) read; no AI keyword signals found")

    incomplete = {vid for vid, s in statuses.items() if s.status == "Incomplete"}
    mask = final_df["Vendor_ID"].astype(str).isin(incomplete)
    if mask.any():
        final_df.loc[mask, "AI Usage Detected (Y/N)"] = "Not assessed"
        final_df.loc[mask, "Evidence Description"] = final_df.loc[mask, "Vendor_ID"].map(
            lambda v: "Assessment incomplete: " + statuses[str(v)].reason)
        final_df.loc[mask, "How Vendor May Be Using AI"] = "Not assessed."
        final_df.loc[mask, "Named AI Sub-processors / Model Providers"] = "Not assessed"
        final_df.loc[mask, "AI Security Risk Classification"] = None
        final_df.loc[mask, "Risk Rationale"] = "No risk tier assigned: the assessment could not be completed."
        final_df.loc[mask, "Recommended Action"] = ("Resolve the issue (check the website / vendor details) "
                                                    "and re-run, or assess this vendor manually.")
    final_df["Vendor Name"] = final_df["Vendor_ID"].astype(str).map(names)
    final_df["Assessment Status"] = final_df["Vendor_ID"].astype(str).map(
        lambda v: statuses[v].status if v in statuses else "Incomplete")
    final_df["Assessment Notes"] = final_df["Vendor_ID"].astype(str).map(
        lambda v: "; ".join(([statuses[v].reason] if statuses[v].reason else []) + statuses[v].notes)
        if v in statuses else "")

    # ---------------- 6. V-000 comparison (notebook step 6) --------------------
    export_df = raw_df.copy()
    if DROP_INTERNAL_SCORING_COLUMNS:   # [integration D4]
        export_df = export_df[[c for c in export_df.columns if not str(c).startswith("_")]]
    vid_col = find_column(export_df, ["Vendor ID"])
    export_cols = resolve_export_columns(export_df)

    v000_cmp = None
    v000_pipeline = final_df[final_df['Vendor_ID'] == V000_ID]
    v000_original = export_df[export_df[vid_col].astype(str) == V000_ID]
    if not v000_pipeline.empty and not v000_original.empty:
        recs = []
        for label, fcol, tcol in V000_FIELDS:
            tcol_real = find_column(export_df, [tcol])
            recs.append({
                "Field": label,
                "Your Pipeline Generated": v000_pipeline[fcol].values[0],
                "Optiv Pre-Filled Answer": (v000_original[tcol_real].values[0] if tcol_real
                                            else "(column not in sheet)"),
            })
        v000_cmp = pd.DataFrame(recs)

    # ---------------- 7. Map to the Excel template (notebook step 7) -----------
    for src, dst in export_cols.items():
        if dst not in export_df.columns:
            export_df[dst] = ""
        export_df[dst] = export_df[dst].astype(object)
    for index, row in final_df.iterrows():
        vid = row['Vendor_ID']

        # DO NOT OVERWRITE V-000 (Keep Optiv's original example intact)
        if vid == V000_ID:
            continue

        match_idx = export_df.index[export_df[vid_col].astype(str) == str(vid)].tolist()
        if match_idx:
            idx = match_idx[0]
            for src, dst in export_cols.items():
                value = row[src]
                if src == "AI Security Risk Classification" and _blank(value):
                    value = NOT_ASSESSED_TIER_TEXT
                export_df.at[idx, dst] = value

    stage_timings["total_s"] = time.perf_counter() - started
    report(1.0, f"Assessment complete in {stage_timings['total_s']:.1f}s")
    return PipelineRun(
        crit_df=crit_df, scraper_results=scraper_results, evidence_df=evidence_df, nlp_df=nlp_df,
        nlp_metrics=metrics, final_df=final_df, export_df=export_df.reset_index(drop=True),
        vendor_status=statuses, v000_comparison=v000_cmp,
        column_mapping={**{f"input: {k}": (v or "MISSING") for k, v in mapping.items()},
                        **{f"output: {k}": v for k, v in export_cols.items()}},
        warnings=warnings_, nlp_error=nlp_error, stage_timings=stage_timings, contact_used=bool(contact),
    )


# Columns of nlp_engine.analyze() output (cell 5, "assemble handoff dataframe").
_NLP_COLUMNS = [
    "Vendor_ID", "Vendor_Name", "AI Usage Detected (Y/N)", "Evidentiary Classification",
    "Evidence Description", "How Vendor May Be Using AI", "Source_URL", "Context_For_NER",
    "Content_SHA256", "Hash_Provenance", "Confidence_Pct", "P_Embedded_AI",
    "Needs_Analyst_Review", "Rule_Signals", "Label_Probabilities", "Analysis_Timestamp_UTC",
]


def sources_checked_table(run: PipelineRun) -> pd.DataFrame:
    rows = []
    for r in run.scraper_results:
        m = r["meta"]
        for n in r.get("sources_checked", []):
            rows.append({"Vendor ID": m["vendor_id"], "Vendor Name": m["vendor_name"],
                         "Depth": m["level_label"], **n})
    return pd.DataFrame(rows)
