import pandas as pd
import os
import warnings
from src.criticality import assess_criticality
from src.scraper import scrape_vendor, Fetcher
from src.nlp_engine import default_model_factory, LABEL_ORDER
from src.risk_matrix import extract_subprocessors, calculate_risk

# Suppress warnings for clean terminal output
warnings.filterwarnings("ignore")


def run_full_pipeline(input_path: str, output_path: str, contact_email: str = ""):
    """
    Executes the end-to-end Optiv AI Risk Assessment pipeline.
    Suitable for both CLI headless execution and Streamlit web UI.
    """
    print(f"[*] Starting Pipeline for input: {input_path}")
    # ---------------------------------------------------------
    # STAGE 1: INGESTION & CRITICALITY SCORING
    # ---------------------------------------------------------
    print("[*] Stage 1: Ingesting Vendor Data & Calculating Criticality...")
    try:
        # Aggressively scan the first 5 rows to find the true header
        raw_df = None
        for i in range(5):
            temp_df = pd.read_excel(input_path, header=i)
            # Convert columns to lowercase strings to ignore Excel spacing/formatting
            clean_cols = [str(c).strip().lower() for c in temp_df.columns]
            if "vendor id" in clean_cols or "vendor_id" in clean_cols:
                raw_df = temp_df
                break

        if raw_df is None:
            raise ValueError("Could not find a header row containing 'Vendor ID'.")

    except Exception as e:
        raise ValueError(f"Failed to read input Excel file: {e}")

    # Apply actuarial scoring
    crit_df = assess_criticality(raw_df)
    pipeline_df = crit_df.copy()

    # ---------------------------------------------------------
    # STAGE 2 & 3: OSINT SCRAPER & EVIDENCE FLATTENING
    # ---------------------------------------------------------
    print("[*] Stage 2 & 3: Sweeping Public OSINT Surfaces...")
    # Initialize the fetcher with a polite delay and contact email
    fetcher = Fetcher(contact=contact_email, delay=1.0)
    scraper_outputs = {}

    for _, row in pipeline_df.iterrows():
        print(f"    Scraping {row['Vendor Name']} (Tier {row['Criticality']})...")
        try:
            res = scrape_vendor(
                vendor_id=row["Vendor ID"],
                name=row["Vendor Name"],
                website=row["Public Website"],
                level=row["Criticality"],
                contact=contact_email,
                fetcher=fetcher,
            )
            scraper_outputs[row["Vendor ID"]] = res
        except Exception as e:
            print(f"    ! Error scraping {row['Vendor ID']}: {e}")
            scraper_outputs[row["Vendor ID"]] = {"evidence": []}

    # ---------------------------------------------------------
    # STAGE 4: NLP EVIDENCE CLASSIFICATION
    # ---------------------------------------------------------
    print("[*] Stage 4: Verifying AI Evidence via NLP (BART-MNLI + Qwen)...")
    # Spin up the PyTorch models
    classifier, summarizer = default_model_factory()

    ai_detected_col = []
    evidence_context_col = []

    for _, row in pipeline_df.iterrows():
        vid = row["Vendor ID"]
        evidence_list = scraper_outputs.get(vid, {}).get("evidence", [])

        if not evidence_list:
            ai_detected_col.append("Unknown - insufficient evidence")
            evidence_context_col.append("No text extracted.")
            continue

        # Extract textual excerpts from the scraper results
        texts = [e["excerpt"] for e in evidence_list]
        probs = classifier.predict_proba(texts)

        ai_detected = False
        top_confidence = 0.0
        best_text = ""

        # Analyze confidence scores against the LABEL_ORDER array
        for i, prob_array in enumerate(probs):
            top_idx = prob_array.argmax()
            confidence = prob_array[top_idx]

            # Index 0 corresponds to "Embedded AI / Production ML"
            if top_idx == 0 and confidence > 0.60:
                ai_detected = True
                if confidence > top_confidence:
                    top_confidence = confidence
                    best_text = texts[i]

        if ai_detected:
            conf_pct = round(top_confidence * 100, 1)
            ai_detected_col.append(f"Yes ({conf_pct}% Confidence)")
            # Generate a 1-sentence capability summary for the audit log
            summary = summarizer.summarize([best_text])[0]
            evidence_context_col.append(f"{summary} Evidence: {best_text}")
        else:
            ai_detected_col.append("No")
            evidence_context_col.append("No AI usage detected in public footprint.")

    pipeline_df["AI Usage Detected (Y/N)"] = ai_detected_col
    pipeline_df["Evidence Context"] = evidence_context_col

    # ---------------------------------------------------------
    # STAGE 5: RISK MATRIX & ESCALATION
    # ---------------------------------------------------------
    print("[*] Stage 5: Applying Optiv Risk Matrix Logic...")

    # Extract known subprocessors (e.g., OpenAI, Hugging Face) using regex logic
    pipeline_df["Named AI Sub-processors / Model Providers"] = pipeline_df[
        "Evidence Context"
    ].apply(lambda x: extract_subprocessors(str(x)))

    # Apply the final classification rules (Low, Medium, High, Critical)
    risk_results = pipeline_df.apply(calculate_risk, axis=1)
    final_df = pd.concat([pipeline_df, risk_results], axis=1)

    # ---------------------------------------------------------
    # STAGE 6: REPORT GENERATION & EXPORT
    # ---------------------------------------------------------
    print(f"[*] Stage 6: Exporting Final Assessment to {output_path}")

    core_cols = [
        "Vendor ID",
        "Vendor Name",
        "Criticality",
        "Criticality Level",
        "AI Usage Detected (Y/N)",
        "Named AI Sub-processors / Model Providers",
        "AI Security Risk Classification",
        "Risk Rationale",
        "Recommended Action",
    ]

    available_cols = [col for col in core_cols if col in final_df.columns]
    export_df = final_df[available_cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="Vendor Inventory", index=False)

    print("[+] Pipeline Execution Completed Successfully!")
    return export_df
