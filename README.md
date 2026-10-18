# Meridian AI Vendor Risk Assessment Dashboard

A Streamlit dashboard for Optiv Case Study 1. It lets you upload a vendor workbook,
preview it, view assessment results, and download an Excel report.

> **Status:** the Optiv notebook pipeline (`optiv.ipynb`) is **connected** through
> `services/optiv_pipeline/`. **Run assessment** runs criticality scoring, live public-website
> scraping, local NLP AI detection (`facebook/bart-large-mnli` + `Qwen/Qwen2.5-0.5B-Instruct`)
> evidence aggregation, sub-processor extraction and the risk matrix on the selected sheet.
> Uploading a file on its own still only loads data. **Demo mode** still shows synthetic data
> labelled `DEMO DATA — NOT REAL FINDINGS` and never runs the pipeline.

## Setup (Windows + VS Code)

Open the project folder in VS Code, then open a terminal (**Terminal > New Terminal**, PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

Use Python 3.11–3.13 (the pinned numpy/pandas have no wheels for 3.14). The first
assessment downloads about 2.5 GB of models from Hugging Face; later runs reuse them. On a
CPU-only laptop the NLP stage takes minutes, and scraping waits 1 s between requests to
the same site, so a Critical-tier vendor alone can take several minutes.

Optionally set a contact e-mail (or type it in the sidebar). It goes in the scraper's
User-Agent, and SEC EDGAR lookups (Criticality 3–4 vendors) only run when one is set:

```powershell
$env:MERIDIAN_CONTACT_EMAIL = "you@example.com"
```

The app opens at http://localhost:8501. Stop it with `Ctrl+C`.

If PowerShell blocks activation, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt and
`.venv\Scripts\activate.bat`. In VS Code, pick the `.venv` interpreter
(**Ctrl+Shift+P > Python: Select Interpreter**).

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Entry point: page layout, tabs, workflow |
| `models/schema.py` | Risk tiers, colours, the results columns the dashboard expects |
| `services/excel_io.py` | Reads workbooks (auto-detects the header row below instructions) and builds the report |
| `services/pipeline_adapter.py` | The single place the pipeline plugs in; maps its output to the dashboard schema |
| `services/optiv_pipeline/` | Temporary integration of `optiv.ipynb`: `criticality.py`, `scraper.py`, `nlp_engine.py`, `risk.py` originate from the notebook, with documented performance changes; `orchestrator.py` runs them like `run_full_pipeline()` and lists every change |
| `tools/verify_parity.py` | Runs the notebook's own code next to the integrated code on the same inputs and reports differences |
| `tests/` | Local fictional test websites, TEST-DOUBLE models (offline use only), end-to-end and app smoke tests |
| `services/demo_data.py` | Synthetic demo data |
| `components/` | UI pieces: styles, summary cards, tables, upload |
| `.streamlit/config.toml` | Dark navy theme and 10 MB upload limit |

## Workbook format

Instruction rows above the table are fine. The app scans the top rows for the header row and
shows its guess; you can change the **Header row** number if it guesses wrong.

## How a run works

1. The selected sheet and header row (Upload tab) are passed to the pipeline. The
   **Column mapping** expander shows which sheet column feeds each input and which column
   each result is written to (case, spacing and punctuation are ignored, as in the notebook).
2. The notebook's assessment rules are retained, with concurrent web collection and exact-input inference reuse. Per vendor, the result shown is the notebook's
   "best" evidence row: the highest P(embedded AI) among the "Yes" rows, otherwise the
   highest among all rows.
3. **Incomplete assessments are not given a tier.** If a vendor's site could not be read
   (unreachable, blocked by robots.txt, missing website) or the NLP models failed, the
   vendor shows *Not assessed* with the reason. The notebook rated these vendors **Low**;
   this is the one deliberate methodology change (see `orchestrator.py`, D1/D2, which can be
   switched off). A site that *was* read but had no AI signals still gets the notebook's Low.
4. V-000 (Optiv's example) is assessed for the regression check, shown in Assessment
   Results, and its pre-filled row is kept unchanged in the completed template, as in the notebook.

The Excel report contains: **Summary**, **Assessment Results** (dashboard view),
**Completed Assessment** (your sheet with the notebook's output columns filled),
**Evidence (all rows)**, **Sources Checked** and **V-000 Regression Check**.

## Faster pipeline

Web collection now runs up to **4 vendors concurrently**. Requests to the same host
still share the 1-second pacing, retry policy, robots checks and response cache.
Blank names/websites are marked incomplete before any requests. If an origin never
responds and two separate fetches exhaust their connection retries, remaining probes
to that origin are skipped for that run with an explicit fetch-log reason. The next
run tries again. An origin that has responded (even with an HTTP error) is never
short-circuited. This changes retry coverage for dead hosts, not the risk matrix;
vendors with no readable first-party pages and no evidence remain incomplete.
Each worker has its own HTTP session; results are restored to worksheet order before
NLP and aggregation so completion order cannot change evidence tie-breaking.

NLP retains the same BART and Qwen models, prompts, thresholds and evidence rows.
Exact repeated classifier inputs are reused across runs in this Python process
(up to 2,048 passages). Identical `(vendor name, context)` summary prompts are generated
once, with up to 1,024 raw summaries reused across runs. Final summaries are still
grounded and finalized separately for each evidence row. Website responses are cached
**only within a run**, so a new assessment still fetches current website content.
Restarting the app clears the inference caches. Shared model inference is serialized
between simultaneous app sessions to protect model/cache state.

The **Assessment completeness per vendor** panel now shows web collection, model
loading, NLP (including loading), and total elapsed time. First-time model downloads,
CPU inference on new text, one slow vendor, and many vendors sharing one host can
still dominate runtime; concurrency does not promise a fixed end-to-end speedup.

The default is automatic. To change the worker count in PowerShell before starting:

```powershell
$env:MERIDIAN_SCRAPE_WORKERS = "4"  # 1 for sequential; allowed range 1–16
python -m streamlit run app.py
```

Run the offline performance checks with:

```powershell
python tests/test_performance.py
```

These compare sequential/concurrent outputs, measure local fixture timing, check
per-host pacing and cache deduplication, exercise a failed vendor, and check NLP
reuse with test doubles. They do not benchmark the downloaded neural models.

## Checking parity with the notebook

```powershell
python tools/verify_parity.py --notebook path\to\optiv.ipynb --workbook path\to\Meridian_Vendor_Input_2.xlsx --limit 3
python tools/verify_parity.py --notebook path\to\optiv.ipynb --workbook path\to\Meridian_Vendor_Input_2.xlsx
```

Performance changes intentionally alter the source of `Fetcher`, `scrape_vendor`,
`ZeroShotClassifier`, `EgressGuard`, and `analyze`; the strict source comparison in
this legacy tool will therefore report differences against the original notebook.
Do not interpret that source mismatch as a scoring comparison, or claim notebook
parity without checking real model outputs. The notebook was not included in this ZIP.

It replays one live scrape through both the notebook's code and the integrated code with the
same loaded models, so live-site changes and model randomness cannot explain any difference.
It prints PASS/FAIL per stage, labels the documented deviations EXPECTED, and writes
`parity_report/`. Offline check: `--fixtures --stub-models` (local fictional sites and
TEST-DOUBLE models; this checks the plumbing only, not the models' judgements).

Other tests: `python tests/test_integration.py`, `python tests/smoke_app.py`.

## Replacing the integration with the team's official modules

Only `services/pipeline_adapter.py` needs to change: call the official modules instead of
`orchestrator.run_pipeline`, and keep `to_dashboard_results` returning `RESULT_COLUMNS`
(`models/schema.py`). The risk tier must be Low, Medium, High, Critical, or empty
(None) when no tier could be assigned.

## Protecting data on GitHub

`.gitignore` blocks Excel/CSV files, `data/`, `reports/`, `.env` files, keys and
`secrets.toml`. Before your first commit run `git status` and check that no vendor file is listed.
If a file was already committed, ignoring it is not enough; ask your team how to remove it from history.

```powershell
git init
git add .
git status
git commit -m "Initial Streamlit dashboard"
```
