import re
import pandas as pd


def extract_subprocessors(text: str) -> str:
    """
    Scans the AI context text to identify specific AI sub-processors or model providers.
    Uses Named Entity Recognition (NER) / Regex matching for known platforms.
    """
    if not isinstance(text, str) or not text.strip():
        return "Undisclosed / None Named"

    # List of known foundational models and enterprise AI providers
    # This can easily be moved to config.py later if it grows
    known_providers = [
        "Azure OpenAI",
        "OpenAI",
        "Anthropic",
        "Google Gemini",
        "Google Vertex",
        "AWS Bedrock",
        "Cohere",
        "Meta Llama",
        "Mistral",
        "Hugging Face",
        "IBM Watson",
    ]

    found_providers = []

    for provider in known_providers:
        # Use regex to find whole-word matches, ignoring case
        pattern = r"\b" + re.escape(provider) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            found_providers.append(provider)

    # Clean up overlaps (e.g., if both "Azure OpenAI" and "OpenAI" hit, just keep "Azure OpenAI")
    if "Azure OpenAI" in found_providers and "OpenAI" in found_providers:
        found_providers.remove("OpenAI")

    if found_providers:
        return ", ".join(found_providers)

    return "Undisclosed / None Named"


def calculate_risk(row: pd.Series) -> pd.Series:
    """
    Calculates the final AI Security Risk Classification, Rationale, and Recommended Action.
    Logic is based on:
      1. AI Detection Confidence (From Stage 4 NLP)
      2. Vendor Criticality Tier 1-4 (From Stage 1 Ingestion)
      3. Transparency of AI Sub-processors
    """
    ai_detected = str(row.get("AI Usage Detected (Y/N)", "")).lower()

    # Safely parse criticality, default to 1 (Low) if missing
    try:
        criticality = int(row.get("Criticality", 1))
    except (ValueError, TypeError):
        criticality = 1

    providers = str(row.get("Named AI Sub-processors / Model Providers", ""))

    is_ai_active = "yes" in ai_detected

    # --- RISK MATRIX LOGIC ---

    # 1. No AI Detected
    if not is_ai_active:
        classification = "Low"
        rationale = "No genuine embedded AI capabilities detected in the vendor's public footprint."
        action = "No immediate action required. Monitor annually during standard TPRM renewal."

    # 2. AI Detected
    else:
        # High/Critical Tier Vendors (Tier 3 & 4) - High exposure to sensitive data
        if criticality >= 3:
            if "Undisclosed" in providers:
                classification = "Critical"
                rationale = f"Tier {criticality} vendor is actively utilizing AI without transparent sub-processor disclosure, posing significant shadow-AI risk to sensitive data."
                action = "Immediate intervention. Review DPA, mandate AI sub-processor disclosure, and verify data isolation controls."
            else:
                classification = "High"
                rationale = f"Tier {criticality} vendor utilizing known AI sub-processors ({providers}). Requires architectural review of data flow."
                action = "Review DPA for approved AI models. Ensure enterprise opt-out mechanisms for model training are active."

        # Medium Tier Vendors (Tier 2) - Internal data, moderate dependency
        elif criticality == 2:
            classification = "Medium"
            rationale = "Tier 2 vendor utilizing AI. Moderate operational dependency; requires standard review to ensure non-PII compliance."
            action = (
                "Update vendor inventory with AI usage flag. Validate terms of service."
            )

        # Low Tier Vendors (Tier 1) - Public data, low dependency
        else:
            classification = "Low"
            rationale = "Tier 1 vendor utilizing AI. Minimal risk footprint due to lack of sensitive data exposure."
            action = "Log AI usage in vendor inventory. No further architectural review required."

    # Return as a Pandas Series so it maps perfectly to the DataFrame columns in pipeline.py
    return pd.Series(
        [classification, rationale, action],
        index=[
            "AI Security Risk Classification",
            "Risk Rationale",
            "Recommended Action",
        ],
    )
