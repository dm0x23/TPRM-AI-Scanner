"""Upload widget: reads the workbook, finds the header row, builds the table."""
import pandas as pd
import streamlit as st

from services.excel_io import (
    ExcelReadError, build_table, detect_header_row, instruction_lines, read_workbook_raw,
)


def _clear_upload_state() -> None:
    for key in ("upload_key", "raw_sheets", "vendors_df", "table_signature", "results_df",
                "sheet_table_df", "pipeline_run", "sheet_name"):
        st.session_state[key] = None


def render_upload() -> pd.DataFrame | None:
    """Show the uploader. Returns the vendor table, or None if not ready.

    Uploading only loads data. It never starts or implies an assessment.
    """
    st.markdown("Upload your vendor workbook (.xlsx, max 10 MB). Files stay in memory "
                "for this session only and are not saved to disk.")
    uploaded = st.file_uploader("Vendor workbook", type=["xlsx"], label_visibility="collapsed")

    if uploaded is None:
        if st.session_state.get("upload_key"):
            _clear_upload_state()  # file removed -> forget its data
        return None

    file_key = f"{uploaded.name}-{uploaded.size}"
    if st.session_state.get("upload_key") != file_key:
        try:
            with st.status("Reading workbook…", expanded=False) as status:
                raw_sheets = read_workbook_raw(uploaded.getvalue(), uploaded.name)
                status.update(label="Workbook read successfully", state="complete")
        except ExcelReadError as exc:
            _clear_upload_state()
            st.error(str(exc))
            return None
        _clear_upload_state()
        st.session_state.upload_key = file_key
        st.session_state.raw_sheets = raw_sheets

    raw_sheets = st.session_state.raw_sheets
    sheet_names = list(raw_sheets)
    sheet = sheet_names[0]
    if len(sheet_names) > 1:
        sheet = st.selectbox("Sheet", sheet_names)
    raw = raw_sheets[sheet]

    detected = detect_header_row(raw)
    if detected is None:
        st.warning("Could not detect the header row automatically. Please choose it below.")
    default_row = (detected or 0) + 1
    header_number = st.number_input(
        "Header row (row number in the sheet as read; 1 = first row)",
        min_value=1, max_value=len(raw), value=min(default_row, len(raw)),
        step=1, key=f"hdr_{file_key}_{sheet}",
        help="Rows above this are treated as instructions and ignored.",
    )
    header_row = int(header_number) - 1

    notes = instruction_lines(raw, header_row)
    if notes:
        with st.expander(f"{len(notes)} instruction row(s) above the header (ignored)"):
            st.text("\n".join(notes))

    try:
        table = build_table(raw, header_row)
        # Same rows, but empty columns kept: the pipeline fills the sheet's own
        # output columns, so the completed template must keep them.
        sheet_table = build_table(raw, header_row, drop_blank_columns=False)
    except ExcelReadError as exc:
        st.error(str(exc))
        st.session_state.vendors_df = None
        st.session_state.sheet_table_df = None
        return None

    # A different file/sheet/header makes earlier results invalid.
    signature = (file_key, sheet, header_row)
    if st.session_state.get("table_signature") != signature:
        st.session_state.table_signature = signature
        st.session_state.results_df = None
        st.session_state.pipeline_run = None
    st.session_state.vendors_df = table
    st.session_state.sheet_table_df = sheet_table
    st.session_state.sheet_name = sheet
    return table
