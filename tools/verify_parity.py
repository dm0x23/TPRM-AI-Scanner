"""Check that the integrated pipeline reproduces the original optiv.ipynb.

What it does
  1. SOURCE CHECK: every function, class and constant defined in the notebook is
     compared (as a parsed syntax tree) with the copy in services/optiv_pipeline/.
  2. RUN CHECK: the integrated pipeline runs on a workbook through the same table
     loader the Streamlit app uses. Each vendor's scrape result is recorded.
  3. The notebook's own code (exec'd straight from optiv.ipynb, including its
     run_full_pipeline) runs on the same workbook, with scrape_vendor REPLAYING the
     recorded results and the SAME model instances. Live-site changes and model
     nondeterminism are therefore removed, so any remaining difference comes from code.
  4. Stage-by-stage diff: criticality, evidence rows, NLP rows, per-vendor template
     output, V-000 regression values. Differences that come from the documented
     deviations (orchestrator.py D1-D4, T6) are labelled EXPECTED; everything else is
     UNEXPECTED and makes the script exit with status 1.

Usage (on a machine that can reach vendor sites and huggingface.co):
    python tools/verify_parity.py --notebook path/to/optiv.ipynb --workbook Meridian_Vendor_Input_2.xlsx \
        [--sheet "Test Vendor Input"] [--contact you@example.com] [--limit 3]

Sandbox / offline mode (local fictional sites + deterministic TEST-DOUBLE models):
    python tools/verify_parity.py --notebook optiv.ipynb --fixtures --stub-models
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import io
import json
import os
import re
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from services import excel_io  # noqa: E402
from services.optiv_pipeline import criticality, nlp_engine, orchestrator, risk, scraper  # noqa: E402

NOTEBOOK_SHEET = "Test Vendor Input"


# ---------------------------------------------------------------------------
# Notebook loading
# ---------------------------------------------------------------------------
class _FakeHTML:
    def __init__(self, data):
        self.data = data


def _fake_ipython(captured: list):
    ip = types.ModuleType("IPython")
    disp = types.ModuleType("IPython.display")
    disp.HTML = _FakeHTML
    disp.display = lambda *objs, **k: captured.extend(objs)
    ip.display = disp
    ip.get_ipython = lambda: None
    return {"IPython": ip, "IPython.display": disp}


def notebook_code_cells(nb_path: str) -> list[tuple[int, str]]:
    nb = json.load(open(nb_path, encoding="utf-8"))
    cells = []
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith("!") or "load_workbook(" in src:
            continue   # pip install cell and the 'inspect sheet names' cell
        cells.append((i, src))
    return cells


def load_notebook(nb_path: str, captured: list) -> dict:
    # not __main__, so cell 6's execute block is skipped; registered so @dataclass works
    mod = types.ModuleType("optiv_notebook")
    sys.modules["optiv_notebook"] = mod
    ns = mod.__dict__
    saved = {k: sys.modules.get(k) for k in ("IPython", "IPython.display")}
    sys.modules.update(_fake_ipython(captured))
    try:
        for i, src in notebook_code_cells(nb_path):
            exec(compile(src, f"optiv.ipynb[cell {i}]", "exec"), ns)
    finally:
        pass   # fake IPython stays installed until restore_ipython() is called
    ns["__saved_ipython__"] = saved
    return ns


def restore_ipython(ns):
    for k, v in ns.get("__saved_ipython__", {}).items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


# ---------------------------------------------------------------------------
# 1. Source check
# ---------------------------------------------------------------------------
def _top_defs(tree) -> dict:
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            out[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node
    return out


def source_check(nb_path: str) -> dict:
    nb_defs = {}
    for i, src in notebook_code_cells(nb_path):
        for k, v in _top_defs(ast.parse(src)).items():
            nb_defs[k] = (i, v)
    mod_defs = {}
    for mod in (criticality, scraper, nlp_engine, risk):
        for k, v in _top_defs(ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))).items():
            mod_defs.setdefault(k, (mod.__name__.split(".")[-1], v))
    identical, differ, absent = [], [], []
    for name, (cell, node) in nb_defs.items():
        if name not in mod_defs:
            absent.append(f"{name} (cell {cell})")
        elif ast.dump(node) == ast.dump(mod_defs[name][1]):
            identical.append(name)
        else:
            differ.append(f"{name} (cell {cell} vs {mod_defs[name][0]}.py)")
    # Not carried over on purpose: Colab display code, and run_full_pipeline/CONTACT_EMAIL,
    # which orchestrator.py replaces.
    expected_absent = {"NotebookReporter", "make_reporter", "PALETTE", "_FONT", "_kpi", "render_header",
                       "render_audit_panel", "_gpu_available", "_styled_table_html",
                       "render_interactive_table", "run_full_pipeline", "wb"}
    return {
        "identical": sorted(identical),
        "different": sorted(differ),
        "not_copied_expected": sorted(a for a in absent if a.split(" ")[0] in expected_absent),
        "not_copied_UNEXPECTED": sorted(a for a in absent if a.split(" ")[0] not in expected_absent),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_TS = re.compile(r"Analyzed: [^|]*$")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def norm(v) -> str:
    s = excel_io._cell_text(v)
    s = _TS.sub("Analyzed: <ts>", s)
    return s


def app_table(workbook: str, sheet: str | None):
    """Load the sheet exactly as the Streamlit app does."""
    data = Path(workbook).read_bytes()
    sheets = excel_io.read_workbook_raw(data, Path(workbook).name)
    sheet = sheet or (NOTEBOOK_SHEET if NOTEBOOK_SHEET in sheets else next(iter(sheets)))
    raw = sheets[sheet]
    hdr = excel_io.detect_header_row(raw)
    return sheet, hdr, excel_io.build_table(raw, hdr, drop_blank_columns=False)


def notebook_header_idx(workbook, sheet):
    """The notebook's own header detection (cell 6, run_full_pipeline step 1)."""
    temp_df = pd.read_excel(workbook, sheet_name=sheet, header=None)
    for i, row in temp_df.iterrows():
        if 'Vendor ID' in row.astype(str).values or 'Vendor Name' in row.astype(str).values:
            return i
    return 0


@contextlib.contextmanager
def redirect_sheet(sheet):
    """The notebook hard-codes sheet 'Test Vendor Input'; point it at the chosen sheet."""
    if sheet == NOTEBOOK_SHEET:
        yield
        return
    real = pd.read_excel

    def patched(*a, **k):
        if k.get("sheet_name") == NOTEBOOK_SHEET:
            k["sheet_name"] = sheet
        return real(*a, **k)
    pd.read_excel = patched
    try:
        yield
    finally:
        pd.read_excel = real


def compare_frames(a: pd.DataFrame, b: pd.DataFrame, key: str, cols: list[str], drop_cols=()):
    diffs = []
    a = a.copy(); b = b.copy()
    a["_k"] = a[key].map(norm); b["_k"] = b[key].map(norm)
    if len(a) != len(b):
        diffs.append(f"row count {len(a)} (notebook) vs {len(b)} (integrated)")
    for i in range(min(len(a), len(b))):
        ra, rb = a.iloc[i], b.iloc[i]
        for c in cols:
            if c in drop_cols:
                continue
            va = norm(ra[c]) if c in a.columns else "<missing>"
            vb = norm(rb[c]) if c in b.columns else "<missing>"
            if va != vb:
                diffs.append(f"row {i} [{ra['_k']}] {c}: notebook={va[:120]!r} integrated={vb[:120]!r}")
    return diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", required=True)
    ap.add_argument("--workbook")
    ap.add_argument("--sheet")
    ap.add_argument("--contact", default=os.getenv(orchestrator.CONTACT_ENV_VAR))
    ap.add_argument("--limit", type=int, help="only the first N vendors (writes a trimmed copy of the sheet)")
    ap.add_argument("--fixtures", action="store_true", help="use local fictional websites + test workbook")
    ap.add_argument("--stub-models", action="store_true", help="TEST DOUBLE models (offline plumbing check only)")
    ap.add_argument("--fast-fetch", action="store_true", help="Fetcher delay 0 s (fixtures only)")
    ap.add_argument("--out", default="parity_report")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = {"mode": {"fixtures": args.fixtures, "stub_models": args.stub_models,
                       "contact_set": bool(args.contact), "pandas": pd.__version__}}

    # ---- 1. source check ------------------------------------------------------
    sc = source_check(args.notebook)
    report["source_check"] = sc
    print(f"[source] identical: {len(sc['identical'])} | different: {sc['different']} | "
          f"not copied (expected): {len(sc['not_copied_expected'])} | not copied (UNEXPECTED): "
          f"{sc['not_copied_UNEXPECTED']}")

    # ---- workbook -------------------------------------------------------------
    workdir = Path(tempfile.mkdtemp(prefix="parity_"))
    if args.fixtures:
        sys.path.insert(0, str(ROOT))
        from tests.fixtures import build_workbook, start_sites
        urls = start_sites()
        workbook = build_workbook(str(workdir / "fixture_input.xlsx"), urls, n_vendors=args.limit)
    else:
        if not args.workbook:
            ap.error("--workbook is required unless --fixtures is used")
        workbook = args.workbook
        if args.limit:
            sheet, hdr, _ = app_table(workbook, args.sheet)
            raw = pd.read_excel(workbook, sheet_name=sheet, header=None)
            trimmed = raw.iloc[: hdr + 1 + args.limit]
            workbook = str(workdir / "trimmed_input.xlsx")
            with pd.ExcelWriter(workbook) as w:
                trimmed.to_excel(w, sheet_name=sheet, header=False, index=False)
    sheet, app_hdr, table = app_table(workbook, args.sheet)
    nb_hdr = notebook_header_idx(workbook, sheet)
    report["header_row"] = {"app_detected": app_hdr, "notebook_detected": int(nb_hdr)}
    print(f"[input] sheet={sheet!r} header row app={app_hdr} notebook={nb_hdr} vendors={len(table)}")

    # ---- models (one set, shared by both sides) --------------------------------
    if args.stub_models:
        from tests.stub_models import stub_model_factory
        clf, summ = stub_model_factory()
        print("[models] TEST DOUBLES (not the real NLP models)")
    else:
        t = time.time()
        clf, summ = nlp_engine.ZeroShotClassifier(nlp_engine.CFG), nlp_engine.CapabilitySummarizer(nlp_engine.CFG)
        print(f"[models] loaded {nlp_engine.CFG.nli_model} + {nlp_engine.CFG.gen_model} in {time.time() - t:.0f}s")

    fetcher_factory = None
    if args.fast_fetch:
        fetcher_factory = lambda c: scraper.Fetcher(contact=c, delay=0.0, timeout=5)  # noqa: E731

    # ---- 2. integrated run (recording scrape) ---------------------------------
    recorded = {}

    def recording_scrape(**kw):
        r = scraper.scrape_vendor(**kw)
        recorded[str(kw["vendor_id"])] = copy.deepcopy(r)
        return r

    t = time.time()
    run = orchestrator.run_pipeline(table.copy(), contact_email=args.contact,
                                    progress=lambda f, m: print(f"  [integrated {f * 100:5.1f}%] {m}"),
                                    model_factory=lambda: (clf, summ), scrape_fn=recording_scrape,
                                    fetcher_factory=fetcher_factory)
    report["integrated_seconds"] = round(time.time() - t, 1)

    # ---- 3. notebook run (replayed scrape, same models) -----------------------
    captured = []
    ns = load_notebook(args.notebook, captured)
    nb_scrape = ns["scrape_vendor"]
    replay_notes = []

    def replay_scrape(vendor_id, name, website, level=2, contact=None, fetcher=None, **kw):
        if vendor_id in recorded:
            r = recorded[vendor_id]
            if int(r["meta"]["level"]) != int(level):
                replay_notes.append(f"{vendor_id}: notebook level {level} != integrated {r['meta']['level']}")
            if r["meta"]["vendor_name"] != name or r["meta"]["website"] != scraper.norm_url(website):
                replay_notes.append(f"{vendor_id}: notebook passed name/website {name!r}/{website!r}, "
                                    f"integrated {r['meta']['vendor_name']!r}/{r['meta']['website']!r}")
            return copy.deepcopy(r)
        replay_notes.append(f"{vendor_id}: not scraped by integrated run; notebook scraped it live "
                            f"(name={name!r}, website={website!r})")
        f = fetcher
        if args.fast_fetch:
            f = scraper.Fetcher(contact=contact, delay=0.0, timeout=5)
        return nb_scrape(vendor_id=vendor_id, name=name, website=website, level=level,
                         contact=contact, fetcher=f, **kw)

    ns["scrape_vendor"] = replay_scrape
    ns["ZeroShotClassifier"] = lambda *a, **k: clf
    ns["CapabilitySummarizer"] = lambda *a, **k: summ
    nb_export, nb_error = None, None
    cwd = os.getcwd()
    os.chdir(workdir)
    stdout = io.StringIO()
    try:
        with redirect_sheet(sheet), contextlib.redirect_stdout(stdout):
            nb_export = ns["run_full_pipeline"](workbook, args.contact)
    except Exception as exc:
        import traceback
        nb_error = f"{type(exc).__name__}: {exc}"
        report["notebook_traceback"] = traceback.format_exc()
    finally:
        os.chdir(cwd)
    report["notebook_stdout"] = stdout.getvalue()
    report["notebook_error"] = nb_error
    report["replay_notes"] = replay_notes
    print(f"[notebook] run_full_pipeline: {'ERROR ' + nb_error if nb_error else 'completed'}")

    # ---- 4. stage comparisons ---------------------------------------------------
    stages = {}
    with redirect_sheet(sheet):
        nb_raw = pd.read_excel(workbook, sheet_name=sheet, header=nb_hdr)
    nb_raw = nb_raw.dropna(how="all").reset_index(drop=True)
    nb_crit = ns["assess_criticality"](nb_raw.copy())
    stages["criticality"] = compare_frames(nb_crit, run.crit_df, "Vendor ID",
                                           ["Vendor ID", "Vendor Name", "Public Website", "Criticality",
                                            "Criticality Level"])

    nb_evidence = ns["flatten_scraper_evidence"](list(recorded.values()))
    stages["evidence_rows"] = compare_frames(nb_evidence, run.evidence_df, "vendor_id",
                                             list(run.evidence_df.columns)) if len(run.evidence_df) else (
        [] if nb_evidence.empty else ["integrated has no evidence but notebook does"])

    if len(run.evidence_df):
        nb_nlp = ns["analyze"](raw=nb_evidence, clf=clf, summarizer=summ, cfg=ns["CFG"]).df
        stages["nlp_rows"] = compare_frames(nb_nlp, run.nlp_df, "Vendor_ID", list(nb_nlp.columns),
                                            drop_cols=("Analysis_Timestamp_UTC",))
    else:
        stages["nlp_rows"] = []

    # per-vendor template output
    expected, unexpected = [], []
    incomplete = {v for v, s in run.vendor_status.items() if s.status == "Incomplete"}
    if nb_export is not None:
        out_cols = list(orchestrator.resolve_export_columns(run.export_df).values())
        nb_cols = set(map(str, nb_export.columns))
        internal = sorted(c for c in nb_cols if c.startswith("_"))
        if internal:
            expected.append(f"D4: notebook export contains internal scoring columns {internal}; integrated drops them")
        nb_by_id = {norm(r["Vendor ID"]): r for _, r in nb_export.iterrows()}
        for _, row in run.export_df.iterrows():
            vid = norm(row[orchestrator.find_column(run.export_df, ["Vendor ID"])])
            nb_row = nb_by_id.get(vid)
            if nb_row is None:
                unexpected.append(f"{vid}: missing from notebook export")
                continue
            for c in out_cols + ["Criticality", "Criticality Level"]:
                va = norm(nb_row[c]) if c in nb_row.index else "<missing>"
                vb = norm(row[c])
                if c in ("Assessedd By / Date", "Assessed By / Date"):
                    va, vb = _DATE.sub("<date>", va), _DATE.sub("<date>", vb)
                if va == vb:
                    continue
                msg = f"{vid} {c}: notebook={va[:110]!r} integrated={vb[:110]!r}"
                if vid in incomplete:
                    expected.append(f"D1/D2 ({run.vendor_status[vid].reason[:70]}): " + msg)
                else:
                    unexpected.append(msg)
        stages["template_output"] = unexpected
    else:
        stages["template_output"] = [f"notebook run_full_pipeline did not complete: {nb_error}"]

    # V-000 regression values
    nb_html = " ".join(getattr(o, "data", "") for o in captured)
    if run.v000_comparison is not None:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", nb_html, re.S)
        nb_vals = [cells[i + 1].strip() for i in range(0, len(cells), 3)] if cells else []
        int_vals = [str(v) for v in run.v000_comparison["Your Pipeline Generated"]]
        stages["v000_regression"] = [] if nb_vals == int_vals else [
            f"notebook={nb_vals} integrated={int_vals}"]
    else:
        stages["v000_regression"] = [] if "V-000" not in nb_html else ["notebook showed V-000, integrated did not"]

    report["stages"] = {k: {"differences": v, "pass": not v} for k, v in stages.items()}
    report["expected_deviations"] = expected
    report["vendor_status"] = {k: vars(v) for k, v in run.vendor_status.items()}
    (out / "parity_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    run.export_df.to_excel(out / "integrated_completed_assessment.xlsx", index=False)
    if nb_export is not None:
        nb_export.to_excel(out / "notebook_completed_assessment.xlsx", index=False)
    restore_ipython(ns)

    print("\n================ PARITY RESULT ================")
    for k, v in stages.items():
        print(f"{'PASS' if not v else 'FAIL'}  {k}" + ("" if not v else f"  ({len(v)} differences)"))
        for d in v[:15]:
            print("      - " + d)
    print(f"EXPECTED deviations ({len(expected)}):")
    for d in expected:
        print("      - " + d)
    if replay_notes:
        print("Replay notes:")
        for d in replay_notes:
            print("      - " + d)
    print(f"Report: {out / 'parity_report.json'}")
    ok = all(not v for v in stages.values()) and not sc["different"] and not sc["not_copied_UNEXPECTED"]
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
