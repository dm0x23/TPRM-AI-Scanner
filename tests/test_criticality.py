import pandas as pd
from src.criticality import assess_criticality


def test_assess_criticality_high_tier():
    # Provide the minimum required columns mapped in criticality.py
    mock_data = pd.DataFrame(
        {
            "Vendor ID": ["V-101"],
            "Vendor Name": ["DataCorp"],
            "Public Website": ["datacorp.example"],
            "Category": ["core platform"],
            "Operational Dependency": ["high"],
            "Data Classification": ["highly confidential / pii"],
            "Data Volume": [150000000],  # 150M triggers escalation
        }
    )

    result_df = assess_criticality(mock_data)

    # Assert the output structure and mathematical logic
    assert not result_df.empty
    assert "Criticality Level" in result_df.columns
    assert result_df["Criticality"].iloc[0] == 4
    assert result_df["Criticality Level"].iloc[0] == "Critical"


def test_assess_criticality_low_tier():
    mock_data = pd.DataFrame(
        {
            "Vendor ID": ["V-102"],
            "Vendor Name": ["DesignTool"],
            "Public Website": ["designtool.example"],
            "Category": ["professional services"],
            "Operational Dependency": ["low"],
            "Data Classification": ["internal"],
            "Data Volume": [500000],
        }
    )

    result_df = assess_criticality(mock_data)
    assert result_df["Criticality"].iloc[0] == 1
    assert result_df["Criticality Level"].iloc[0] == "Low"
