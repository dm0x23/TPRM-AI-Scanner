"""Executes app.py top to bottom with a minimal fake `streamlit` module.

This is NOT a substitute for `streamlit run app.py`; it only proves app.py's own
Python runs without errors through upload -> run -> results -> report, and captures
the downloaded report bytes. Use it where Streamlit cannot be installed.

    python tests/smoke_app.py
"""
from __future__ import annotations

import io
import runpy
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CALLS = {"error": [], "warning": [], "success": [], "info": [], "download": []}


class Ctx:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(FAKE, name)


class SessionState(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class Uploaded:
    def __init__(self, path):
        self.name, self._b = Path(path).name, Path(path).read_bytes()
        self.size = len(self._b)

    def getvalue(self):
        return self._b


class Progress:
    def progress(self, *a, **k):
        pass

    def empty(self):
        pass


def make_fake(upload_path, click_run):
    st = types.ModuleType("streamlit")
    st.session_state = SessionState()
    noop = lambda *a, **k: None  # noqa: E731
    for n in ("set_page_config", "header", "divider", "markdown", "caption", "subheader", "text",
              "dataframe"):
        setattr(st, n, noop)
    for n in ("error", "warning", "success", "info"):
        setattr(st, n, (lambda key: lambda msg, *a, **k: CALLS[key].append(str(msg)))(n))
    st.sidebar = Ctx()
    st.spinner = st.expander = Ctx
    st.status = lambda *a, **k: _Status()
    st.toggle = lambda *a, **k: False
    st.tabs = lambda labels: [Ctx() for _ in labels]
    st.columns = lambda spec, **k: [Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.file_uploader = lambda *a, **k: Uploaded(upload_path)
    st.selectbox = lambda label, options, **k: "Test Vendor Input" if "Test Vendor Input" in options else options[0]
    st.number_input = lambda *a, value=1, **k: value
    st.text_input = lambda *a, value="", key=None, **k: st.session_state.setdefault(key, value) if key else value
    st.multiselect = lambda label, options, default=None, **k: list(default or [])
    st.button = lambda label, **k: bool(click_run and label == "Run assessment" and not k.get("disabled"))
    st.progress = lambda *a, **k: Progress()

    def download_button(label, data, file_name, **k):
        CALLS["download"].append((file_name, data))
    st.download_button = download_button
    st.column_config = types.SimpleNamespace(TextColumn=lambda *a, **k: None)
    return st


class _Status(Ctx):
    def update(self, **k):
        pass


FAKE = None


def main():
    global FAKE
    from services import pipeline_adapter
    from services.optiv_pipeline import orchestrator
    from tests.fixtures import build_workbook, start_sites
    from tests.stub_models import stub_model_factory
    import openpyxl

    urls = start_sites(base_port=18901)
    wb = build_workbook(str(Path(tempfile.mkdtemp()) / "app_input.xlsx"), urls, n_vendors=3)

    # Sandbox only: torch/transformers cannot be installed, so use TEST-DOUBLE models.
    pipeline_adapter.missing_dependencies = lambda: []
    orchestrator.default_model_factory = stub_model_factory

    FAKE = make_fake(wb, click_run=True)
    sys.modules["streamlit"] = FAKE
    runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    ss = FAKE.session_state
    assert not CALLS["error"], CALLS["error"]
    assert ss.results_df is not None and len(ss.results_df) == 3, "no results stored"
    assert ss.pipeline_run is not None
    assert CALLS["download"], "report download button not rendered"
    name, data = CALLS["download"][-1]
    sheets = openpyxl.load_workbook(io.BytesIO(data)).sheetnames
    print("success messages:", CALLS["success"])
    print("warnings:", CALLS["warning"])
    print("report:", name, sheets)
    print(ss.results_df[["Vendor ID", "Vendor", "Criticality", "Risk Tier", "Assessment Status"]].to_string())
    print("APP SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
