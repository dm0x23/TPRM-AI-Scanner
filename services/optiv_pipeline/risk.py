"""Stage 4 - evidence flattening, sub-processor extraction and risk matrix.

GENERATED FROM optiv.ipynb, cell 6. The code below the import block is the
notebook code unchanged (helpers 1-3 only; run_full_pipeline is re-implemented stage by stage in orchestrator.py). Do not edit scoring rules, thresholds or model settings here
without updating the notebook too: tools/verify_parity.py checks that this module and the
notebook produce identical outputs.
"""
# --- import block: notebook cell 1, unchanged ---
import argparse
import contextlib
import csv
import difflib
import hashlib
import html as htmllib
import json
import os
import re
import socket
import sys
import time
import unicodedata
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib import robotparser
from urllib.parse import urljoin, urlparse, urldefrag, quote_plus

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, NavigableString
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning)

# --- notebook code ---
# ---------------------------------------------------------
# HELPER 1: FLATTEN SCRAPER OUTPUT
# ---------------------------------------------------------
def flatten_scraper_evidence(scraper_results):
    """Converts the scraper's nested dictionary output into a flat DataFrame."""
    import pandas as pd
    flat_records = []
    for vendor_data in scraper_results:
        for evidence in vendor_data.get('evidence', []):
            flat_records.append({
                "vendor_id": evidence.get("vendor_id"),
                "vendor_name": evidence.get("vendor_name"),
                "excerpt": evidence.get("excerpt", ""),
                "context": evidence.get("context", ""),
                "source_url": evidence.get("source_url", ""),
                "content_sha256": evidence.get("content_sha256", "")
            })
    return pd.DataFrame(flat_records)

# ---------------------------------------------------------
# HELPER 2: NER SUB-PROCESSOR EXTRACTION
# ---------------------------------------------------------
def extract_subprocessors(text):
    """Extracts known AI foundation model providers from the evidence text."""
    import re
    PROVIDERS_DICT = {
        "OpenAI": r"\bOpenAI\b|\bChatGPT\b|\bGPT-?\d",
        "Azure OpenAI": r"Azure OpenAI|Azure AI",
        "Anthropic / Claude": r"\bAnthropic\b|\bClaude\b",
        "Google (Gemini/Vertex)": r"\bGemini\b|Vertex AI|Google Cloud AI|\bPaLM\b",
        "AWS Bedrock": r"Bedrock|SageMaker",
        "Microsoft Copilot": r"Microsoft (365 )?Copilot|GitHub Copilot",
        "Cohere": r"\bCohere\b",
        "Meta Llama": r"\bLlama[\s-]?\d?",
        "Databricks": r"Databricks|MosaicML",
        "IBM watsonx": r"watsonx",
        "NVIDIA": r"\bNVIDIA\b"
    }
    found = []
    for provider, pattern in PROVIDERS_DICT.items():
        if re.search(pattern, str(text), re.I):
            found.append(provider)
    return ", ".join(found) if found else "Undisclosed / None Named"

# ---------------------------------------------------------
# HELPER 3: ACTUARIAL RISK SCORING
# ---------------------------------------------------------
def calculate_risk(row):
    """Applies Optiv's determinisitic risk matrix based on AI usage and Data Criticality."""
    import pandas as pd
    ai_detected = str(row.get('AI Usage Detected (Y/N)', 'No'))
    crit = int(row.get('Criticality', 2))
    providers = str(row.get('Named AI Sub-processors / Model Providers', ''))

    if not ai_detected.startswith('Yes'):
        tier = "Low"
        rationale = "No embedded generative AI capabilities verified in the public footprint."
        action = "Standard third-party risk monitoring."
    else:
        if crit >= 3 and "Undisclosed" in providers:
            tier = "Critical"
            rationale = f"Vendor handles highly sensitive data (Criticality Tier {crit}) and uses embedded AI, but the foundation model provider is undisclosed, creating unquantified shadow-AI risk."
            action = "Issue targeted due diligence questionnaire requesting model provider, hosting region, and data retention policies. Escalate to Third-Party Risk Committee if unresolved."
        elif crit == 4:
            tier = "Critical"
            rationale = "Vendor handles core operational/PII data and exposes it to third-party generative AI models."
            action = "Review DPA for explicit AI sub-processor clauses. Ensure opt-outs for model training are actively enforced."
        elif crit == 3:
            tier = "High"
            rationale = "Vendor handles confidential business data via embedded AI features."
            action = "Verify the named AI sub-processors are listed in the vendor's Trust Center and align with internal AI governance policies."
        else:
            tier = "Medium"
            rationale = "Vendor uses embedded AI, but handles lower-classification data limiting enterprise exposure."
            action = "Log AI capability in the vendor inventory. Review terms at next standard renewal."

    return pd.Series([tier, rationale, action])
