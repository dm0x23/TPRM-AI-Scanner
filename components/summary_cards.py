"""Dashboard metric cards."""
import streamlit as st

from models.schema import RISK_TIERS, TIER_COLORS


def _card(label: str, value, note: str = "", color: str = "#5B8DEF") -> str:
    # Only fixed strings and numbers are inserted here, never vendor text.
    return (
        f'<div class="mv-card" style="border-top-color:{color}">'
        f'<div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="note">{note}</div></div>'
    )


def render_summary_cards(total_vendors, assessed, counts, pipeline_connected, demo):
    suffix = " (demo)" if demo else ""
    top = st.columns(3)
    top[0].markdown(
        _card("Vendors loaded" + suffix, total_vendors if total_vendors is not None else "—",
              "" if total_vendors is not None else "No file uploaded"),
        unsafe_allow_html=True,
    )
    top[1].markdown(
        _card("Vendors assessed" + suffix, assessed if assessed else "—",
              "" if assessed else "No assessment has run"),
        unsafe_allow_html=True,
    )
    status = "Demo mode" if demo else ("Connected" if pipeline_connected else "Not connected")
    top[2].markdown(_card("Assessment pipeline", status), unsafe_allow_html=True)

    cols = st.columns(4)
    for col, tier in zip(cols, RISK_TIERS):
        if assessed:
            share = round(100 * counts[tier] / assessed)
            value, note = counts[tier], f"{share}% of assessed vendors"
        else:
            value, note = "—", "Awaiting assessment"
        col.markdown(_card(f"{tier} risk{suffix}", value, note, TIER_COLORS[tier]),
                     unsafe_allow_html=True)
