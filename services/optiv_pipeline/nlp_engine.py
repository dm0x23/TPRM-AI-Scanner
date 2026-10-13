"""Stage 3 - local NLP AI-usage verification (zero-shot NLI + capability summaries).

GENERATED FROM optiv.ipynb, cell 4 and 5. The code below the import block is the
notebook code unchanged except that the Colab-only display code is removed (see the [integration] note). Do not edit scoring rules, thresholds or model settings here
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
@dataclass(frozen=True)
class Config:
    input_csv: str = "evidence_log.csv"
    output_csv: str = "nlp_verified_evidence.csv"
    force_mock: bool = False                      # True -> ignore input_csv, use mock data

    nli_model: str = "facebook/bart-large-mnli"
    gen_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    nli_batch_size: int = 8                       # passages per NLI call (x5 labels each)
    gen_batch_size: int = 8
    max_new_tokens: int = 80

    max_chars: int = 2500                         # hard cap on text fed to models
    min_text_chars: int = 25                      # below this -> "Insufficient Evidence"
    excerpt_weight: float = 0.40                  # fusion weight: excerpt-only vs excerpt+context
    yes_threshold: float = 0.50                   # P(embedded AI) needed for "Yes"
    review_band_pct: float = 70.0                 # confidence below this -> analyst review flag
    lexical_cap: float = 1.2                      # cap on summed log-multiplier from lexical rules
    preview_rows: int = 25
    seed: int = 42


CFG = Config()

REQUIRED_COLUMNS = ["vendor_id", "vendor_name", "excerpt", "context", "source_url", "content_sha256"]
_SHA_RX = re.compile(r"^[0-9a-f]{64}$", re.I)


# %% ------------------------------------------------- 3. TAXONOMY & SIGNALS
class L:
    EMBEDDED = "Embedded / Production AI"
    HYPE = "Marketing Hype / Thought Leadership"
    DEFENSIVE = "Defensive Security Control"
    LEGACY = "Legacy Rule-Based Automation"
    OTHER = "Not AI-Related"
    INSUFFICIENT = "Insufficient Evidence"


LABEL_ORDER = [L.EMBEDDED, L.HYPE, L.DEFENSIVE, L.LEGACY, L.OTHER]

# Natural-language hypotheses fed to the NLI model ("This passage describes {}.")
LABEL_HYPOTHESES = {
    L.EMBEDDED: "an AI or machine learning feature that the vendor's own product delivers to customers in production",
    L.HYPE: "marketing hype, thought leadership, or future plans about AI with no concrete shipped product feature",
    L.DEFENSIVE: "a security control that protects against AI-driven threats rather than the vendor's own AI feature",
    L.LEGACY: "traditional rule-based automation or scripting that does not use machine learning",
    L.OTHER: "content unrelated to artificial intelligence",
}
HYPOTHESIS_TEMPLATE = "This passage describes {}."

NO_AI_EXPLANATIONS = {
    L.HYPE: "No embedded capability identified; the passage reads as marketing or forward-looking AI messaging.",
    L.DEFENSIVE: "No vendor-embedded AI identified; the passage describes a defensive control against AI-enabled threats.",
    L.LEGACY: "No AI/ML identified; the passage describes conventional rule-based or scripted automation.",
    L.OTHER: "The passage is not related to an AI capability.",
    L.INSUFFICIENT: "Insufficient text available to assess AI usage.",
    L.EMBEDDED: "Embedded-AI likelihood below the verification threshold; flagged for analyst review.",
}

_DEFENSIVE_RX = (
    r"\b(defen[cd]\w*|protect\w*|guard\w*|safeguard\w*|detect\w*|block\w*|mitigat\w*|prevent\w*|"
    r"stop\w*|resist\w*|counter\w*)\b[^.]{0,80}?\b(?:ai|artificial intelligence|machine learning|ml|llm|"
    r"deepfake|automated)[- ]?(?:driven|powered|generated|enabled|based|assisted|augmented)?\s+"
    r"(?:attacks?|threats?|fraud|phishing|malware|bots?|exploits?|impersonation|intrusions?|campaigns?)\b"
)

# (label, rule name, regex, log-odds weight).  Weights are deliberately modest:
# the NLI model leads, rules nudge - and every nudge is auditable.
SIGNAL_RULES: List[Tuple[str, str, str, float]] = [
    (L.EMBEDDED, "first-party AI feature phrasing",
     r"\b(our|the)\s+(\w+\s+){0,2}(platform|product|solution|engine|assistant|service|software|application|models?)\b"
     r"[^.]{0,40}\b(uses?|leverages?|employs?|incorporates?|embeds?|is powered by|runs on|built on)\b"
     r"[^.]{0,60}\b(ai|artificial intelligence|machine learning|ml|llm|language model|neural|deep learning|nlp)\b", 0.5),
    (L.EMBEDDED, "model/ML terminology",
     r"\b(machine learning model|ml model|neural networks?|large language models?|llms?|deep learning|fine-tuned|"
     r"trained on|inference|embeddings?|classifier|anomaly detection|natural language processing|ocr)\b", 0.3),
    (L.EMBEDDED, "shipped / GA language",
     r"\b(generally available|now available|in production|ships with|built[- ]in|available (today|now)|deployed)\b", 0.3),
    (L.HYPE, "forward-looking / roadmap language",
     r"\b(will (soon |eventually )?(launch|ship|release|transform|revolutioni[sz]e|enable)|plans? to|"
     r"on (our|the) roadmap|exploring|we envision|future of|coming soon|next year|aspire|investigating)\b", 0.6),
    (L.HYPE, "thought-leadership / marketing language",
     r"\b(webinar|whitepaper|white paper|thought leadership|keynote|podcast|predictions?|we believe|"
     r"revolutioni[sz]e|game[- ]chang\w*|paradigm|new era|unlock the power|cutting[- ]edge|industry leaders)\b", 0.5),
    (L.DEFENSIVE, "defends AGAINST AI-enabled threats", _DEFENSIVE_RX, 1.0),
    (L.DEFENSIVE, "adversarial-ML / deepfake defence terms",
     r"\b(adversarial (machine learning|attacks?)|prompt injection|model poisoning|deepfakes?)\b", 0.3),
    (L.LEGACY, "rule-based / scripted automation",
     r"\b(rule[- ]based|if[- ]then|scripted|macros?|robotic process automation|rpa|workflow automation|"
     r"business rules|deterministic|batch jobs?|cron|no machine learning|without (ai|machine learning))\b", 0.7),
]
_COMPILED_RULES = [(lab, name, re.compile(rx, re.I), w) for lab, name, rx, w in SIGNAL_RULES]


def lexical_multipliers(text: str, cap: float = CFG.lexical_cap) -> Tuple[np.ndarray, List[str]]:
    """Return per-label probability multipliers and the names of rules that fired."""
    log_w = {lab: 0.0 for lab in LABEL_ORDER}
    fired: List[str] = []
    for lab, name, rx, w in _COMPILED_RULES:
        if rx.search(text):
            log_w[lab] += w
            fired.append(f"{lab.split(' /')[0]}: {name}")
    mult = np.array([np.exp(min(log_w[lab], cap)) for lab in LABEL_ORDER], dtype=np.float64)
    return mult, fired


# %% ------------------------------------------------------ 4. TEXT UTILITIES
_TAG_RX = re.compile(r"<[^>]+>")
_ZW_RX = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_WS_RX = re.compile(r"\s+")
_NULLISH = {"nan", "none", "null", "n/a", "na", "<na>", "nat"}


def clean_text(value) -> str:
    """Null-safe normalisation: HTML entities/tags, unicode, zero-width chars, whitespace."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(value)
    if s.strip().lower() in _NULLISH:
        return ""
    s = htmllib.unescape(s)
    s = _TAG_RX.sub(" ", s)
    s = unicodedata.normalize("NFKC", s)
    s = _ZW_RX.sub("", s)
    return _WS_RX.sub(" ", s).strip()


def truncate_words(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + "..."


def first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n]).strip()


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class EgressGuard:
    """Blocks non-loopback TCP connects while active; counts attempts as an audit metric."""

    def __init__(self):
        self.blocked = 0

    @contextlib.contextmanager
    def active(self):
        real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
        guard = self

        def _is_external(address) -> bool:
            if isinstance(address, tuple) and address:
                host = str(address[0])
                return host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "")
            return False  # AF_UNIX paths etc.

        def guarded_connect(sock, address, *a, **k):
            if _is_external(address):
                guard.blocked += 1
                raise ConnectionError(f"[EgressGuard] outbound connection to {address[0]} blocked")
            return real_connect(sock, address, *a, **k)

        def guarded_connect_ex(sock, address, *a, **k):
            if _is_external(address):
                guard.blocked += 1
                return 111  # ECONNREFUSED
            return real_connect_ex(sock, address, *a, **k)

        socket.socket.connect, socket.socket.connect_ex = guarded_connect, guarded_connect_ex
        try:
            yield self
        finally:
            socket.socket.connect, socket.socket.connect_ex = real_connect, real_connect_ex


# %% ---------------------------------------------------------- 7. MODELS
class ZeroShotClassifier:
    """facebook/bart-large-mnli via the stable zero-shot pipeline. Returns probs in LABEL_ORDER."""

    def __init__(self, cfg: Config = CFG):
        import torch
        from transformers import pipeline

        self.cfg = cfg
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("zero-shot-classification", model=cfg.nli_model, device=self.device)
        if self.device == 0:
            self.pipe.model.half()  # fp16 on T4
        self.hyp_to_label = {LABEL_HYPOTHESES[lab]: lab for lab in LABEL_ORDER}
        self.hypotheses = [LABEL_HYPOTHESES[lab] for lab in LABEL_ORDER]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        out = self.pipe(
            list(texts),
            candidate_labels=self.hypotheses,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False,
            batch_size=max(1, len(texts)) * len(self.hypotheses),
            truncation=True,
        )
        if isinstance(out, dict):
            out = [out]
        probs = np.zeros((len(texts), len(LABEL_ORDER)), dtype=np.float64)
        for i, res in enumerate(out):
            by_label = {self.hyp_to_label[h]: s for h, s in zip(res["labels"], res["scores"])}
            probs[i] = [by_label[lab] for lab in LABEL_ORDER]
        return probs


class CapabilitySummarizer:
    """Qwen2.5-0.5B-Instruct, greedy decoding, plain generate() (no pipeline registration risk)."""

    SYSTEM = ("You are a precise analyst for a third-party risk team. "
              "You only use facts that are present in the provided text.")

    def __init__(self, cfg: Config = CFG):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg, self.torch = cfg, torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(cfg.gen_model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(cfg.gen_model)
        self.model = self.model.to(self.device, dtype=torch.float16) if self.device == "cuda" else self.model
        self.model.eval()

    def _prompt(self, vendor: str, text: str) -> str:
        user = (f"Vendor: {vendor}\nText: {text}\n\n"
                "In 1-2 sentences, state what AI or machine-learning capability this vendor's product appears "
                "to provide and what it is used for. Rely only on the text. Do not guess, and do not speculate "
                "about third-party providers. Begin with the vendor name. Output only the sentences.")
        msgs = [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def summarize(self, items: Sequence[Tuple[str, str]]) -> List[str]:
        torch = self.torch
        prompts = [self._prompt(v, t) for v, t in items]
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.cfg.max_new_tokens, do_sample=False,
                repetition_penalty=1.05, pad_token_id=self.tok.pad_token_id)
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        return [self.tok.decode(t, skip_special_tokens=True).strip() for t in new_tokens]


def finalize_summary(raw: str, vendor: str, source_text: str, fallback_text: str) -> str:
    """Trim to <=2 sentences / 320 chars; fall back to extractive text if empty or ungrounded."""
    s = _WS_RX.sub(" ", raw or "").strip()
    s = re.sub(r"^(summary|answer|output)\s*:\s*", "", s, flags=re.I)
    s = first_sentences(s, 2)
    if len(s) > 320:
        s = first_sentences(s, 1)[:320].rsplit(" ", 1)[0].rstrip(",;: ") + "."
    words = {w for w in re.findall(r"[a-z]{4,}", s.lower())} - set(re.findall(r"[a-z]{4,}", vendor.lower()))
    src = source_text.lower()
    grounded = (sum(w in src for w in words) / len(words)) if words else 0.0
    if len(s) < 25 or grounded < 0.30:
        return f"{vendor}: {first_sentences(fallback_text, 1)[:240]}"
    return s


# %% --------------------------------------------------- 8. PROGRESS REPORTING
class NullReporter:
    def update(self, frac: float, msg: str) -> None:
        print(f"[{frac * 100:5.1f}%] {msg}")



# [integration] NotebookReporter, make_reporter (ipywidgets) and the section-10
# 'EXECUTIVE DASHBOARD' HTML renderers were removed: they are Colab display code,
# not assessment logic, and NotebookReporter referenced an undefined name `html`.

# %% ------------------------------------------------------- 9. CORE ENGINE
@dataclass
class AuditResult:
    df: pd.DataFrame
    metrics: Dict[str, object] = field(default_factory=dict)


def prepare_input(raw: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """Validate schema, null-proof every field, build analysis texts, repair audit hashes."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"evidence_log is missing required columns: {missing}")
    recs = []
    for i, r in enumerate(df.to_dict("records")):
        vid = clean_text(r["vendor_id"]) or f"UNKNOWN-{i + 1:04d}"
        excerpt, context = clean_text(r["excerpt"]), clean_text(r["context"])
        combined = context if (excerpt and excerpt in context) else f"{excerpt} {context}".strip()
        combined = truncate_words(combined, cfg.max_chars)
        sha = clean_text(r["content_sha256"]).lower()
        sha_src = "scraper"
        if not _SHA_RX.match(sha):
            sha, sha_src = sha256_hex(f"{excerpt}|{context}"), "recomputed (scraper hash missing/invalid)"
        recs.append(dict(
            vendor_id=vid, vendor_name=clean_text(r["vendor_name"]) or "Unknown Vendor",
            excerpt=excerpt, context=context, source_url=clean_text(r["source_url"]) or "N/A (missing)",
            sha=sha, sha_src=sha_src, text_ctx=combined, text_exc=truncate_words(excerpt or combined, cfg.max_chars),
            insufficient=len(combined) < cfg.min_text_chars))
    return pd.DataFrame(recs)


def _classify_unique(texts: List[str], clf, cfg: Config, rep, lo: float, hi: float) -> Dict[str, np.ndarray]:
    uniq = list(dict.fromkeys(texts))
    cache: Dict[str, np.ndarray] = {}
    step = cfg.nli_batch_size
    for s in range(0, len(uniq), step):
        chunk = uniq[s:s + step]
        for t, p in zip(chunk, clf.predict_proba(chunk)):
            cache[t] = np.asarray(p, dtype=np.float64)
        done = min(s + step, len(uniq))
        rep.update(lo + (hi - lo) * done / max(len(uniq), 1), f"Semantic classification: {done}/{len(uniq)} unique passages")
    return cache


def decide(probs_final: np.ndarray, cfg: Config) -> Dict[str, object]:
    p_yes = float(probs_final[0])
    is_yes = p_yes >= cfg.yes_threshold
    if is_yes:
        label, conf = L.EMBEDDED, p_yes
    else:
        label = LABEL_ORDER[1 + int(np.argmax(probs_final[1:]))]   # winning NON-embedded label
        conf = 1.0 - p_yes
    return dict(is_yes=is_yes, label=label, conf_pct=round(conf * 100, 1), p_yes=p_yes)


def analyze(raw: pd.DataFrame, clf, summarizer, cfg: Config = CFG, rep=None,
            guard: Optional[EgressGuard] = None, span: Tuple[float, float] = (0.10, 0.92)) -> AuditResult:
    rep = rep or NullReporter()
    guard = guard or EgressGuard()
    t0 = time.time()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lo, hi = span
    prep = prepare_input(raw, cfg)
    rep.update(lo, f"Validated {len(prep)} evidence rows")

    with guard.active():
        # ---- classification (excerpt view + excerpt+context view, fused) -----------
        live = prep[~prep.insufficient]
        texts = list(live.text_exc) + list(live.text_ctx)
        cls_hi = lo + (hi - lo) * 0.65
        cache = _classify_unique(texts, clf, cfg, rep, lo, cls_hi) if texts else {}

        results: List[Dict[str, object]] = []
        for r in prep.itertuples():
            if r.insufficient:
                results.append(dict(is_yes=False, label=L.INSUFFICIENT, conf_pct=None, p_yes=0.0,
                                    probs=None, signals=["text too short or missing"]))
                continue
            p_exc, p_ctx = cache[r.text_exc], cache[r.text_ctx]
            fused = cfg.excerpt_weight * p_exc + (1 - cfg.excerpt_weight) * p_ctx
            mult, fired = lexical_multipliers(r.text_ctx, cfg.lexical_cap)
            adj = fused * mult
            adj = adj / adj.sum()
            d = decide(adj, cfg)
            d.update(probs=adj, signals=fired)
            results.append(d)

        # ---- capability summaries (verified rows only) -----------------------------
        summaries: Dict[int, str] = {}
        verified_idx = [i for i, d in enumerate(results) if d["is_yes"]]
        for s in range(0, len(verified_idx), cfg.gen_batch_size):
            batch = verified_idx[s:s + cfg.gen_batch_size]
            items = [(prep.vendor_name[i], prep.text_ctx[i]) for i in batch]
            raws = summarizer.summarize(items)
            for i, raw_s in zip(batch, raws):
                summaries[i] = finalize_summary(raw_s, prep.vendor_name[i], prep.text_ctx[i],
                                                prep.excerpt[i] or prep.text_ctx[i])
            done = min(s + cfg.gen_batch_size, len(verified_idx))
            rep.update(cls_hi + (hi - cls_hi) * done / max(len(verified_idx), 1),
                       f"Synthesizing capability summaries: {done}/{len(verified_idx)}")

    # ---- assemble handoff dataframe ------------------------------------------------
    out_rows = []
    for i, (r, d) in enumerate(zip(prep.itertuples(), results)):
        if d["label"] == L.INSUFFICIENT:
            flag = "No (Insufficient Evidence)"
        else:
            flag = f"{'Yes' if d['is_yes'] else 'No'} ({d['conf_pct']:.1f}% Confidence)"
        quote = truncate_words(r.excerpt or "[excerpt unavailable - classified on context]", 300)
        evidence = f"\"{quote}\" | Source: {r.source_url} | SHA-256: {r.sha[:16]}... | Analyzed: {ts}"
        how = summaries.get(i) or NO_AI_EXPLANATIONS[d["label"]]
        needs_review = (d["label"] == L.INSUFFICIENT) or (d["conf_pct"] is not None and d["conf_pct"] < cfg.review_band_pct)
        probs = d.get("probs")
        out_rows.append({
            "Vendor_ID": r.vendor_id,
            "Vendor_Name": r.vendor_name,
            "AI Usage Detected (Y/N)": flag,
            "Evidentiary Classification": d["label"],
            "Evidence Description": evidence,
            "How Vendor May Be Using AI": how,
            "Source_URL": r.source_url,
            "Context_For_NER": r.context or r.text_ctx,
            # --- audit extras (kept to the right of the contractual schema) ---
            "Content_SHA256": r.sha,
            "Hash_Provenance": r.sha_src,
            "Confidence_Pct": d["conf_pct"],
            "P_Embedded_AI": round(d["p_yes"], 4),
            "Needs_Analyst_Review": bool(needs_review),
            "Rule_Signals": "; ".join(d["signals"]) if d["signals"] else "none",
            "Label_Probabilities": json.dumps({lab: round(float(p), 4) for lab, p in zip(LABEL_ORDER, probs)}) if probs is not None else "{}",
            "Analysis_Timestamp_UTC": ts,
        })
    out = pd.DataFrame(out_rows)

    n = len(out)
    cls_counts = out["Evidentiary Classification"].value_counts().to_dict()
    metrics = {
        "total": n,
        "verified": int((out["Evidentiary Classification"] == L.EMBEDDED).sum()),
        "fp_filtered": int(sum(cls_counts.get(k, 0) for k in (L.HYPE, L.DEFENSIVE, L.LEGACY))),
        "not_ai": int(cls_counts.get(L.OTHER, 0)),
        "insufficient": int(cls_counts.get(L.INSUFFICIENT, 0)),
        "by_class": {lab: int(cls_counts.get(lab, 0)) for lab in LABEL_ORDER + [L.INSUFFICIENT]},
        "needs_review": int(out["Needs_Analyst_Review"].sum()),
        "avg_conf": float(out["Confidence_Pct"].dropna().mean()) if out["Confidence_Pct"].notna().any() else float("nan"),
        "hash_repaired": int((out["Hash_Provenance"] != "scraper").sum()),
        "egress_blocked": guard.blocked,
        "elapsed_s": round(time.time() - t0, 1),
    }
    return AuditResult(out, metrics)


def export_handoff(df: pd.DataFrame, path: str) -> str:
    df.to_csv(path, index=False, encoding="utf-8")
    return os.path.abspath(path)
