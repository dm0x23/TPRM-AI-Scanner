"""Preview and results tables."""
import pandas as pd
import streamlit as st

from models.schema import RISK_TIERS, TIER_COLORS

NOT_ASSIGNED = "Not assigned"


def render_preview(df: pd.DataFrame, rows: int = 100) -> None:
    st.caption(f"{len(df)} vendor row(s) and {len(df.columns)} column(s) detected. "
               f"Showing the first {min(rows, len(df))}.")
    st.dataframe(df.head(rows), hide_index=True)


def _tier_style(value):
    if value in TIER_COLORS:
        return f"background-color: {TIER_COLORS[value]}; color: white; font-weight: 600;"
    return ""


def render_results(df: pd.DataFrame, key_prefix: str = "res") -> pd.DataFrame:
    """Filterable results table. Returns the filtered rows."""
    c1, c2 = st.columns([1, 2])
    options = RISK_TIERS + [NOT_ASSIGNED]
    tiers = c1.multiselect("Risk tier", options, default=options, key=f"{key_prefix}_tier",
                           help=f"'{NOT_ASSIGNED}' = the assessment could not be completed, so no tier was given.")
    search = c2.text_input("Search vendor name", key=f"{key_prefix}_search")

    keep = df["Risk Tier"].isin(tiers)
    if NOT_ASSIGNED in tiers:
        keep |= df["Risk Tier"].isna()
    view = df[keep]
    if search.strip():
        view = view[view["Vendor"].astype(str).str.contains(search.strip(), case=False, regex=False)]

    st.caption(f"Showing {len(view)} of {len(df)} vendors.")
    st.dataframe(
        view.style.map(_tier_style, subset=["Risk Tier"]),
        hide_index=True,
        column_config={
            "AI Usage Evidence": st.column_config.TextColumn(width="large"),
            "Source URLs": st.column_config.TextColumn(width="large"),
            "Recommended Action": st.column_config.TextColumn(width="large"),
            "Risk Rationale": st.column_config.TextColumn(width="large"),
            "How Vendor May Be Using AI": st.column_config.TextColumn(width="large"),
            "Assessment Notes": st.column_config.TextColumn(width="large"),
        },
    )
    return view
