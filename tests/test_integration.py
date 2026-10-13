import pandas as pd
from src.criticality import assess_criticality
from src.risk_matrix import calculate_risk


def test_pipeline_data_handoff():
    # 1. Mock the initial Excel read
    mock_input = pd.DataFrame(
        {
            "Vendor ID": ["V-999"],
            "Vendor Name": ["IntegrationCorp"],
            "Public Website": ["integration.example"],
            "Operational Dependency": ["High"],
            "Data Classification Accessed": ["Highly Confidential / PII"],
            "Data Volume (annual)": ["1,000,000"],
        }
    )

    # 2. Execute Stage 1
    crit_df = assess_criticality(mock_input)
    assert crit_df["Criticality"].iloc[0] == 4

    # 3. Simulate Stage 4 (NLP Engine) appending columns to Stage 1's output
    simulated_nlp_row = crit_df.iloc[0].copy()
    simulated_nlp_row["AI Usage Detected (Y/N)"] = "Yes (99% Confidence)"
    simulated_nlp_row["Named AI Sub-processors / Model Providers"] = "OpenAI"

    # 4. Execute Stage 5
    final_risk = calculate_risk(simulated_nlp_row)

    # 5. Verify the integration succeeded
    assert final_risk.iloc[0] == "Critical"
    assert "Review DPA" in final_risk.iloc[2]
