"""Meridian AI Vendor Risk Assessment — Streamlit dashboard.

Run with:  streamlit run app.py
"""
import logging
import os
from datetime import datetime

import pandas as pd

import streamlit as st

from components.styles import inject_css
from components.summary_cards import render_summary_cards
from components.tables import render_preview, render_results
from components.upload import render_upload
from models.schema import DEMO_LABEL, SchemaError, tier_counts, validate_results
from services import pipeline_adapter
from services.demo_data import generate_demo_results
from services.excel_io import build_report

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Meridian AI Vendor Risk Assessment", page_icon="🛡️", layout="wide")


def init_state() -> None:
    for key in ("upload_key", "raw_sheets", "vendors_df", "table_signature", "results_df",
                "sheet_table_df", "pipeline_run", "sheet_name"):
        st.session_state.setdefault(key, None)


def render_sidebar() -> bool:
    with st.sidebar:
        st.header("Controls")
        demo = st.toggle(
            "Demo mode (synthetic data)",
            help="Shows invented data so you can preview the interface. Not real findings.",
        )
        st.divider()
        st.markdown("**Pipeline status**")
        missing = pipeline_adapter.missing_dependencies()
        run = st.session_state.get("pipeline_run")
        if missing:
            st.warning("Pipeline code is wired in, but these packages are missing: "
                       + ", ".join(missing) + ". Run `pip install -r requirements.txt`.")
        elif run is not None:
            n_done = sum(s.status == "Complete" for s in run.vendor_status.values())
            st.success(f"Connected. Last run: {n_done} of {len(run.vendor_status)} vendors "
                       "fully assessed.")
        else:
            st.info("Pipeline ready (dependencies installed). No assessment has run in this session.")
        st.text_input(
            "Contact e-mail for scraper User-Agent",
            key="contact_email",
            value=os.getenv(pipeline_adapter.CONTACT_ENV_VAR, ""),
            help="Sent in the User-Agent header to vendor sites. SEC EDGAR requires one; without it, "
                 "EDGAR lookups (Criticality 3-4) are skipped and logged. You can also set the "
                 f"{pipeline_adapter.CONTACT_ENV_VAR} environment variable.",
        )
        st.caption("Vendor files are processed in memory and never written to disk by this app. "
                   "Website text and model inference stay on this machine; the NLP models are "
                   "downloaded from Hugging Face on first use.")
    return demo


def active_results(demo: bool):
    """Return (results, is_demo). Demo data and real results are never mixed."""
    if demo:
        if "demo_df" not in st.session_state:
            st.session_state.demo_df = generate_demo_results()
        return st.session_state.demo_df, True
    return st.session_state.results_df, False


def run_pipeline(vendors) -> None:
    bar = st.progress(0.0, text="Starting assessment…")

    def on_progress(fraction: float, message: str) -> None:
        bar.progress(min(max(float(fraction), 0.0), 1.0), text=message)

    try:
        with st.spinner("Running assessment pipeline (live scraping + local NLP; this can take a while)…"):
            output = pipeline_adapter.run_assessment_full(
                vendors.copy(), progress_callback=on_progress,
                sheet_table=st.session_state.get("sheet_table_df"),
                contact_email=st.session_state.get("contact_email") or None,
            )
        results = validate_results(output.results)
    except pipeline_adapter.PipelineNotConnectedError as exc:
        bar.empty()
        st.error(str(exc))
    except pipeline_adapter.PipelineInputError as exc:
        bar.empty()
        st.error(f"The selected sheet cannot be assessed. {exc}")
    except SchemaError as exc:
        bar.empty()
        st.error(f"The pipeline returned data in an unexpected format. {exc}")
    except Exception:
        bar.empty()
        logger.exception("Assessment failed")
        st.error("The assessment failed unexpectedly. Check the terminal for details.")
    else:
        bar.empty()
        st.session_state.results_df = results
        st.session_state.pipeline_run = output.run
        incomplete = int(results["Risk Tier"].isna().sum())
        msg = f"Assessment finished for {len(results)} vendors."
        if incomplete:
            st.warning(msg + f" {incomplete} could not be completed and have no risk tier; "
                       "see Assessment Results for the reasons.")
        else:
            st.success(msg)
        for w in output.run.warnings:
            st.warning(w)


def render_workflow(vendors, results, demo: bool) -> None:
    def step(done: bool, text: str) -> str:
        return f'<div class="mv-step">{"✅" if done else "⬜"} {text}</div>'

    html = (
        step(vendors is not None, "Vendor file uploaded and header row confirmed")
        + step(results is not None, "Assessment run" + (" (demo data)" if demo else ""))
    )
    st.markdown("**Workflow**")
    st.markdown(html, unsafe_allow_html=True)
    if vendors is not None and results is None and not demo:
        st.info("A file is loaded, but **no assessment has been run**. Risk tiers will appear "
                "only after the assessment pipeline has processed the vendors.")


def render_run_details(run) -> None:
    """V-000 regression check and per-vendor completeness, from the notebook run."""
    if run.v000_comparison is not None:
        with st.expander("V-000 regression check (pipeline vs Optiv pre-filled example)", expanded=True):
            st.dataframe(run.v000_comparison.astype(str), hide_index=True)
            st.caption("As in the notebook, V-000 is assessed for this comparison only; its original "
                       "pre-filled row is kept unchanged in the Completed Assessment sheet.")
    status = pd.DataFrame([{
        "Vendor ID": s.vendor_id, "Vendor": s.vendor_name, "Status": s.status,
        "First-party pages read": s.first_party_pages_read, "Evidence rows": s.evidence_rows,
        "Reason / notes": "; ".join(([s.reason] if s.reason else []) + s.notes),
    } for s in run.vendor_status.values()])
    with st.expander("Assessment completeness per vendor", expanded=bool((status["Status"] != "Complete").any())):
        st.dataframe(status, hide_index=True)
        m = run.nlp_metrics
        if m:
            st.caption(f"NLP: {m.get('total')} evidence rows scored · {m.get('verified')} verified embedded AI · "
                       f"{m.get('fp_filtered')} filtered as hype/defensive/legacy · {m.get('needs_review')} "
                       f"flagged for analyst review · {m.get('elapsed_s')}s.")
        if not run.contact_used:
            st.caption("No contact e-mail was set, so SEC EDGAR lookups were skipped (logged per vendor).")
    with st.expander("Sources checked (scraper log)"):
        st.dataframe(pipeline_adapter.sources_checked_table(run), hide_index=True)


def main() -> None:
    init_state()
    inject_css()
    demo = render_sidebar()

    st.markdown('<div class="mv-title">Meridian AI Vendor Risk Assessment</div>'
                '<div class="mv-sub">Optiv Case Study 1 · Third-party AI usage and risk tiering</div>',
                unsafe_allow_html=True)
    if demo:
        st.markdown(f'<div class="mv-demo">{DEMO_LABEL}</div>', unsafe_allow_html=True)

    tab_dash, tab_upload, tab_results, tab_report = st.tabs(
        ["Dashboard", "Upload & Preview", "Assessment Results", "Report"]
    )

    # The upload tab runs first so state is current for the other tabs.
    with tab_upload:
        st.subheader("Upload & preview vendor data")
        vendors = render_upload()
        if vendors is not None:
            render_preview(vendors)
            st.divider()
            st.subheader("Run assessment")
            can_run = pipeline_adapter.is_pipeline_connected() and not demo
            if demo:
                st.caption("Demo mode is on: the pipeline is not used and no real assessment runs.")
            elif not can_run:
                st.caption("Disabled: required packages are missing ("
                           + ", ".join(pipeline_adapter.missing_dependencies())
                           + "). Run `pip install -r requirements.txt`.")
            sheet_table = st.session_state.get("sheet_table_df")
            if sheet_table is not None and not demo:
                in_map = pipeline_adapter.resolve_input_columns(sheet_table)
                out_map = pipeline_adapter.resolve_export_columns(sheet_table)
                missing_cols = [k for k, v in in_map.items() if v is None]
                with st.expander("Column mapping used by the pipeline",
                                 expanded=bool(missing_cols)):
                    st.dataframe(pd.DataFrame(
                        [{"Pipeline field": k, "Sheet column": v or "MISSING", "Direction": "input"}
                         for k, v in in_map.items()]
                        + [{"Pipeline field": k, "Sheet column": v + ("" if v in sheet_table.columns
                                                                     else " (will be added)"),
                            "Direction": "output"} for k, v in out_map.items()]),
                        hide_index=True)
                if missing_cols:
                    st.error("Required column(s) not found in this sheet: " + ", ".join(missing_cols)
                             + ". Check the sheet and header row.")
                    can_run = False
            if st.button("Run assessment", type="primary", disabled=not can_run):
                run_pipeline(vendors)

    vendors = st.session_state.vendors_df
    results, is_demo = active_results(demo)

    with tab_dash:
        total = len(results) if is_demo else (len(vendors) if vendors is not None else None)
        # Only vendors that received a risk tier count as assessed.
        assessed = int(results["Risk Tier"].notna().sum()) if results is not None else 0
        render_summary_cards(total, assessed, tier_counts(results),
                             pipeline_adapter.is_pipeline_connected(), is_demo)
        st.divider()
        render_workflow(vendors, results, is_demo)
        if results is not None and not is_demo and results["Risk Tier"].isna().any():
            st.warning(f"{int(results['Risk Tier'].isna().sum())} vendor(s) could not be fully "
                       "assessed and have no risk tier (see Assessment Results).")

    with tab_results:
        st.subheader("Vendor assessment results")
        if results is None:
            st.info("No assessment results yet. Uploading a file does not run an assessment.")
        else:
            render_results(results)
            run = st.session_state.get("pipeline_run")
            if run is not None and not is_demo:
                render_run_details(run)

    with tab_report:
        st.subheader("Excel assessment report")
        if results is None:
            st.info("There is nothing to export yet. A report can be downloaded once "
                    "an assessment has produced results.")
        else:
            if is_demo:
                st.warning(f"This report will be watermarked: {DEMO_LABEL}")
            try:
                stamp = datetime.now().strftime("%Y%m%d_%H%M")
                name = f"{'DEMO_' if is_demo else ''}meridian_vendor_risk_{stamp}.xlsx"
                run = None if is_demo else st.session_state.get("pipeline_run")
                extra = None
                if run is not None:
                    extra = {
                        "Completed Assessment": run.export_df,
                        "Evidence (all rows)": run.nlp_df,
                        "Sources Checked": pipeline_adapter.sources_checked_table(run),
                    }
                    if run.v000_comparison is not None:
                        extra["V-000 Regression Check"] = run.v000_comparison
                    st.caption("The report contains: Summary, Assessment Results (dashboard view), "
                               "Completed Assessment (your sheet with the pipeline's output columns "
                               "filled, as in the notebook), Evidence (every NLP-scored evidence row), "
                               "Sources Checked (every page/source tried and its outcome)"
                               + (", V-000 Regression Check." if run.v000_comparison is not None else "."))
                st.download_button(
                    "Download Excel report", data=build_report(results, is_demo, extra), file_name=name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                )
            except Exception:
                logger.exception("Report generation failed")
                st.error("The report could not be generated. Check the terminal for details.")


main()
