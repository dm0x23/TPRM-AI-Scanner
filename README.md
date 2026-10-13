# Meridian AI Vendor Risk Assessment Dashboard

A Streamlit dashboard for Optiv Case Study 1. It lets you upload a vendor workbook,
preview it, view assessment results, and download an Excel report.

> **Status:** the Optiv notebook pipeline (`optiv.ipynb`) is **connected** through
> `services/optiv_pipeline/`. **Run assessment** runs criticality scoring, live public-website
> scraping, local NLP AI detection (`facebook/bart-large-mnli` + `Qwen/Qwen2.5-0.5B-Instruct`),
> evidence aggregation, sub-processor extraction and the risk matrix on the selected sheet.
> Uploading a file on its own still only loads data. **Demo mode** still shows synthetic data
> labelled `DEMO DATA — NOT REAL FINDINGS` and never runs the pipeline.

## Setup (Windows + VS Code)

Open the project folder in VS Code, then open a terminal (**Terminal > New Terminal**, PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
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
| `services/optiv_pipeline/` | Temporary integration of `optiv.ipynb`: `criticality.py`, `scraper.py`, `nlp_engine.py`, `risk.py` are generated from the notebook cells unchanged; `orchestrator.py` runs them like `run_full_pipeline()` and lists every change |
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
2. The notebook's stages run unchanged. Per vendor, the result shown is the notebook's
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

## Checking parity with the notebook

```powershell
python tools/verify_parity.py --notebook path\to\optiv.ipynb --workbook path\to\Meridian_Vendor_Input_2.xlsx --limit 3
python tools/verify_parity.py --notebook path\to\optiv.ipynb --workbook path\to\Meridian_Vendor_Input_2.xlsx
```

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
