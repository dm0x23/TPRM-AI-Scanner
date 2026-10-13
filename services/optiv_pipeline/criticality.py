"""Stage 1 - vendor criticality assessment.

GENERATED FROM optiv.ipynb, cell 2. The code below the import block is the
notebook code unchanged. Do not edit scoring rules, thresholds or model settings here
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
CATEGORY_SCORES = {
    "core platform": 4,
    "payments": 4,
    "data and reporting": 3,
    "document and communications": 2,
    "professional services": 1,
}

DEPENDENCY_SCORES = {
    "low": 1,
    "moderate": 2,
    "high": 3,
    "total": 4,
}

CLASSIFICATION_SCORES = {
    "internal": 1,
    "confidential": 2,
    "highly confidential / pii": 4,
}

WEIGHTS = {
    "dependency": 0.35,
    "category": 0.25,
    "classification": 0.25,
    "volume": 0.15,
}

# Provisional scores are used only when input information is unresolved.
# They prevent missing information from being treated as low risk while
# keeping the agreed weighting model unchanged.
PROVISIONAL_SCORE = 2


CRITICALITY_LEVELS = {
    1: "Low",
    2: "Moderate",
    3: "High",
    4: "Critical",
}

def normalize_column_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return name.strip()

def assess_criticality(df):
  normalized_columns = {}

  for column in df.columns:
      normalized_name = normalize_column_name(column)

      if normalized_name in normalized_columns:
          raise ValueError(
              "Duplicate input columns found after normalization: "
              + normalized_name
          )

      normalized_columns[normalized_name] = column


  def find_column(possible_names):
      for name in possible_names:
          normalized_name = normalize_column_name(name)

          if normalized_name in normalized_columns:
              return normalized_columns[normalized_name]

      return None


  column_map = {
      "vendor_id": find_column(["Vendor ID"]),
      "vendor_name": find_column(["Vendor Name"]),
      "website": find_column(["Public Website", "Website", "Vendor Website", "URL", "Domain"]),
      "category": find_column(["Category"]),
      "dependency": find_column(["Operational Dependency"]),
      "classification": find_column(
          ["Data Classification Accessed", "Data Classification"]
      ),
      "volume": find_column(
          ["Data Volume (annual)", "Data Volume Annual", "Data Volume"]
      ),
  }


  missing_columns = [
      key for key, value in column_map.items()
      if value is None
  ]

  if missing_columns:
      raise ValueError(
          "Missing required columns: " + ", ".join(missing_columns)
      )



  def normalize_value(value):
      if pd.isna(value):
          return None

      text = str(value).strip().lower()

      if not text or text in {"n/a", "na", "none", "null", "nan"}:
          return None

      # Normalize spacing around the slash in:
      # "Highly Confidential / PII"
      text = re.sub(r"\s*/\s*", " / ", text)
      text = re.sub(r"\s+", " ", text)

      return text.strip()


  def score_categorical_value(value, score_map):
      normalized_value = normalize_value(value)

      if normalized_value is None:
          return PROVISIONAL_SCORE, False

      score = score_map.get(normalized_value)

      if score is None:
          return PROVISIONAL_SCORE, False

      return score, True


  def parse_volume(value):
      """
      Parse numeric annual volume values and common suffixes.

      Supported examples include:
      1000000
      1,000,000
      1M
      1.5 million
      2B
      500K

      Missing, N/A, negative, and unparseable values return None.
      """
      if pd.isna(value):
          return None

      if isinstance(value, bool):
          return None

      if isinstance(value, (int, float)):
          numeric_value = float(value)

          if numeric_value < 0:
              return None

          return numeric_value

      text = str(value).strip().lower()

      if not text or text in {"n/a", "na", "none", "null", "nan"}:
          return None

      text = text.replace(",", "")
      text = text.replace("$", "").strip()

      pattern = re.compile(
          r"^([0-9]+(?:\.[0-9]+)?)\s*"
          r"(k|thousand|m|mn|million|b|bn|billion)?$"
      )

      match = pattern.fullmatch(text)

      if not match:
          return None

      number = float(match.group(1))
      unit = match.group(2)

      multipliers = {
          "k": 1_000,
          "thousand": 1_000,
          "m": 1_000_000,
          "mn": 1_000_000,
          "million": 1_000_000,
          "b": 1_000_000_000,
          "bn": 1_000_000_000,
          "billion": 1_000_000_000,
      }

      return number * multipliers.get(unit, 1)


  def volume_score(volume):
      if volume is None:
          return PROVISIONAL_SCORE, False

      if volume >= 100_000_000:
          return 4, True

      if volume >= 10_000_000:
          return 3, True

      if volume >= 1_000_000:
          return 2, True

      return 1, True


  def base_criticality(score):
      if score >= 3.25:
          return 4

      if score >= 2.50:
          return 3

      if score >= 1.75:
          return 2

      return 1


  # --------------------------------------------------------------------
  # Internal assessment fields
  # --------------------------------------------------------------------

  dependency_results = df[column_map["dependency"]].apply(
      lambda value: score_categorical_value(value, DEPENDENCY_SCORES)
  )

  df["_dependency_score"] = dependency_results.apply(lambda result: result[0])
  df["_dependency_known"] = dependency_results.apply(lambda result: result[1])


  category_results = df[column_map["category"]].apply(
      lambda value: score_categorical_value(value, CATEGORY_SCORES)
  )

  df["_category_score"] = category_results.apply(lambda result: result[0])
  df["_category_known"] = category_results.apply(lambda result: result[1])


  classification_results = df[column_map["classification"]].apply(
      lambda value: score_categorical_value(value, CLASSIFICATION_SCORES)
  )

  df["_classification_score"] = classification_results.apply(
      lambda result: result[0]
  )
  df["_classification_known"] = classification_results.apply(
      lambda result: result[1]
  )

  df["_classification_status"] = df["_classification_known"].map(
      {
          True: "known",
          False: "unresolved",
      }
  )


  df["_volume_value"] = df[column_map["volume"]].apply(parse_volume)

  volume_results = df["_volume_value"].apply(volume_score)

  df["_volume_score"] = volume_results.apply(lambda result: result[0])
  df["_volume_known"] = volume_results.apply(lambda result: result[1])


  # This flag is retained internally and is not included in the output.
  df["_information_complete"] = (
      df["_dependency_known"]
      & df["_category_known"]
      & df["_classification_known"]
      & df["_volume_known"]
  )


  # --------------------------------------------------------------------
  # Weighted criticality calculation
  # --------------------------------------------------------------------

  df["_criticality_score"] = (
      df["_dependency_score"] * WEIGHTS["dependency"]
      + df["_category_score"] * WEIGHTS["category"]
      + df["_classification_score"] * WEIGHTS["classification"]
      + df["_volume_score"] * WEIGHTS["volume"]
  )

  df["_criticality"] = df["_criticality_score"].apply(base_criticality)


  # --------------------------------------------------------------------
  # Escalation rules
  # --------------------------------------------------------------------

  def apply_escalation(row):
      dependency = row["_dependency_score"]
      category = row["_category_score"]
      classification = row["_classification_score"]
      volume = row["_volume_value"]
      criticality = row["_criticality"]

      dependency_known = row["_dependency_known"]
      category_known = row["_category_known"]
      classification_known = row["_classification_known"]
      volume_known = row["_volume_known"]

      # Classification-based rules require a valid company-provided
      # classification. Provisional classifications cannot trigger them.
      if (
          dependency_known
          and classification_known
          and dependency == 4
          and classification == 4
      ):
          return 4

      if (
          dependency_known
          and category_known
          and dependency == 4
          and category == 4
      ):
          return 4

      if (
          dependency_known
          and classification_known
          and dependency == 3
          and classification == 4
      ):
          return 4

      # Missing or invalid volume cannot trigger this rule.
      if (
          dependency_known
          and volume_known
          and dependency == 3
          and volume is not None
          and volume >= 100_000_000
      ):
          return 4

      return criticality


  df["Criticality"] = df.apply(apply_escalation, axis=1).astype(int)
  df["Criticality"] = df["Criticality"].clip(lower=1, upper=4)
  df["Criticality Level"] = df["Criticality"].map(CRITICALITY_LEVELS)


  # --------------------------------------------------------------------
  # Final output
  # --------------------------------------------------------------------

  # Construct a new output DataFrame so existing or repeated Criticality
  # columns in the input cannot create duplicate output columns.
  output = pd.DataFrame(
      {
          "Vendor ID": df[column_map["vendor_id"]],
          "Vendor Name": df[column_map["vendor_name"]],
          "Public Website": df[column_map["website"]],
          "Criticality": df["Criticality"],
          "Criticality Level": df["Criticality Level"],
      }
  )
  return output
