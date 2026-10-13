import pandas as pd
from src.risk_matrix import extract_subprocessors, calculate_risk


def test_extract_subprocessors_found():
    text = "We use Azure OpenAI and Anthropic to power our chat features."
    result = extract_subprocessors(text)
    assert "Azure OpenAI" in result
    assert "Anthropic" in result


def test_extract_subprocessors_none():
    text = "We built our own custom proprietary system."
    result = extract_subprocessors(text)
    assert result == "Undisclosed / None Named"


def test_calculate_risk_critical_undisclosed():
    row = pd.Series(
        {
            "AI Usage Detected (Y/N)": "Yes (90% Confidence)",
            "Criticality": 3,
            "Named AI Sub-processors / Model Providers": "Undisclosed / None Named",
        }
    )
    result = calculate_risk(row)
    assert result.iloc[0] == "Critical"
    assert "shadow-AI risk" in result.iloc[1]


def test_calculate_risk_low():
    row = pd.Series(
        {
            "AI Usage Detected (Y/N)": "No (0% Confidence)",
            "Criticality": 4,
            "Named AI Sub-processors / Model Providers": "",
        }
    )
    result = calculate_risk(row)
    assert result.iloc[0] == "Low"
