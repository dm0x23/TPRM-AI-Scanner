"""Offline regression checks: python tests/test_performance.py. No model downloads."""
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.optiv_pipeline import nlp_engine as nlp, orchestrator as orch, scraper
from tests.stub_models import StubCapabilitySummarizer, stub_model_factory


def sheet():
    return pd.DataFrame([{'Vendor ID': f'V-{i:03}', 'Vendor Name': f'Vendor {i}',
        'Public Website': f'https://vendor{i}.example', 'Category': 'Payments',
        'Operational Dependency': 'High', 'Data Classification Accessed': 'Confidential',
        'Data Volume (annual)': '12M'} for i in range(1, 9)])


def replay_scrape(vendor_id, name, **kwargs):
    time.sleep(.02 * (10 - int(vendor_id.split('-')[1])))
    return {'meta': {'vendor_id': vendor_id, 'vendor_name': name, 'level_label': 'Deep (High)'},
        'evidence': [], 'sources_checked': [dict(source_type='homepage', outcome='no_hits')]}


def test_pipeline():
    runs, elapsed = [], []
    for workers in [1, 4]:
        progress = []
        start = time.perf_counter()
        run = orch.run_pipeline(sheet(), scrape_fn=replay_scrape, scrape_workers=workers,
            progress=lambda f, m: progress.append((f, threading.get_ident())))
        elapsed.append(time.perf_counter() - start)
        runs.append(run)
        assert all(a[0] <= b[0] for a, b in zip(progress, progress[1:]))
        assert all(t == threading.get_ident() for _, t in progress)
    pd.testing.assert_frame_equal(runs[0].final_df, runs[1].final_df)
    pd.testing.assert_frame_equal(runs[0].export_df, runs[1].export_df)
    assert runs[0].scraper_results == runs[1].scraper_results
    assert elapsed[1] < elapsed[0] * .75
    print(f'8-vendor simulated I/O: sequential {elapsed[0]:.3f}s; 4 workers {elapsed[1]:.3f}s; {elapsed[0]/elapsed[1]:.2f}x')
    def broken(**kw):
        if kw['vendor_id'] == 'V-003': raise RuntimeError('fixture failure')
        return replay_scrape(**kw)
    failed = orch.run_pipeline(sheet(), scrape_fn=broken, scrape_workers=4)
    assert failed.vendor_status['V-003'].status == 'Incomplete'
    assert failed.final_df.loc[failed.final_df.Vendor_ID == 'V-003', 'AI Security Risk Classification'].isna().all()
    assert sum(s.status == 'Complete' for s in failed.vendor_status.values()) == 7


def test_fetcher():
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append((self.server.server_port, self.path, time.monotonic()))
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'User-agent: *\nAllow: /\n' if self.path == '/robots.txt' else b'source text')
        def log_message(self, *args): pass
    servers = [ThreadingHTTPServer(('127.0.0.1', 0), Handler) for _ in range(2)]
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    f = scraper.Fetcher(delay=.04)
    urls = [f'http://127.0.0.1:{s.server_port}' for s in servers]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(f.get, [urls[0]+'/a', urls[0]+'/a', urls[0]+'/b', urls[1]+'/a']))
        assert all(r.ok for r in results)
        for server in servers:
            records = [(path, t) for port, path, t in received if port == server.server_port]
            assert sum(path == '/robots.txt' for path, _ in records) == 1
            assert sum(path == '/a' for path, _ in records) == 1
            assert all(b[1]-a[1] >= .03 for a, b in zip(records, records[1:])), records
        assert len(f._sessions) >= 2
    finally:
        f.close()
        for server in servers:
            server.shutdown()
            server.server_close()
    print('HTTP checks passed: cache deduplication, robots, per-host pacing, worker sessions')


def test_nlp_reuse():
    text = 'Our platform uses machine learning models in production for anomaly detection.'
    raw = pd.DataFrame([dict(vendor_id=f'V-{i}', vendor_name='Vendor', excerpt=text,
        context=text, source_url=f'https://example.com/{i}', content_sha256='') for i in range(12)])
    class CountingSummary(StubCapabilitySummarizer):
        def __init__(self): self.inputs = []
        def summarize(self, items):
            self.inputs.extend(items)
            return super().summarize(items)
    clf, _ = stub_model_factory()
    summ = CountingSummary()
    first = nlp.analyze(raw, clf, summ)
    second = nlp.analyze(raw, clf, summ)
    assert len(summ.inputs) == 1
    assert len(first.df) == 12 and first.metrics['verified'] == 12
    assert second.metrics['summary_cache_hits'] == 1
    expected = StubCapabilitySummarizer().summarize([('Vendor', text)])[0]
    assert all(first.df['How Vendor May Be Using AI'] == nlp.finalize_summary(expected, 'Vendor', text, text))
    cached = object.__new__(nlp.ZeroShotClassifier)
    cached._prob_cache = OrderedDict()
    calls = []
    def forward(texts):
        calls.extend(texts)
        return np.tile([.8,.05,.05,.05,.05], (len(texts),1))
    cached._predict_uncached = forward
    a = cached.predict_proba(['same','same','other'])
    b = cached.predict_proba(['same'])
    assert calls == ['same','other']
    np.testing.assert_array_equal(a[0], b[0])
    print('NLP checks passed: 12 duplicate summaries reduced to 1; rerun reuse; NLI exact-input reuse')



def test_dead_host():
    import requests
    f = scraper.Fetcher(delay=0, respect_robots=False)
    calls = []
    def fail(url, **kw):
        calls.append(url)
        raise requests.ConnectionError('offline fixture')
    f.s.get = fail
    try:
        for path in ('/a', '/b', '/c', '/d'):
            result = f.get('https://dead.example' + path)
        assert len(calls) == 2
        assert result.note == 'host_unreachable_after_repeated_connection_failures'
        # A different origin isn't affected, including HTTP vs HTTPS.
        f.get('http://dead.example/a')
        assert len(calls) == 3
        f._responsive_origins.add('https://dead.example')
        f.get('https://dead.example/reachable-before')
        assert len(calls) == 4
    finally:
        f.close()
    bad = sheet().iloc[:1].copy()
    bad['Public Website'] = None
    def no_requests(**kw): raise AssertionError('blank website must never be scraped')
    run = orch.run_pipeline(bad, scrape_fn=no_requests)
    assert run.vendor_status['V-001'].status == 'Incomplete'
    assert 'blank' in run.vendor_status['V-001'].reason
    print('Dead-host retry bound and blank-website checks passed')


def test_real_scraper():
    from tests.fixtures import start_sites, build_workbook
    from services import excel_io
    import tempfile
    urls = start_sites(base_port=19101)
    path = build_workbook(str(Path(tempfile.mkdtemp()) / 'vendors.xlsx'), urls)
    raw = excel_io.read_workbook_raw(Path(path).read_bytes(), 'vendors.xlsx')['Test Vendor Input']
    table = excel_io.build_table(raw, excel_io.detect_header_row(raw), drop_blank_columns=False)
    # Fictional local servers cannot supply public GitHub/news/EDGAR data.
    levels = {k: dict(v, github=False, edgar=False, news=False) for k, v in scraper.DEFAULT_LEVELS.items()}
    def local_scrape(**kw):
        return scraper.scrape_vendor(**kw, levels=levels)
    runs, elapsed = [], []
    fetchers = []
    def make_fetcher(contact):
        f = scraper.Fetcher(contact=contact, delay=.03, timeout=.1)
        fetchers.append(f)
        return f
    for workers in (1, 4):
        start = time.perf_counter()
        runs.append(orch.run_pipeline(table, model_factory=stub_model_factory,
            scrape_fn=local_scrape, scrape_workers=workers,
            fetcher_factory=make_fetcher))
        fetchers[-1].close()
        elapsed.append(time.perf_counter()-start)
    for attr in ('final_df', 'export_df', 'nlp_df'):
        frames = [getattr(run, attr).copy() for run in runs]
        for frame in frames:
            frame.drop(columns=['Analysis_Timestamp_UTC'], errors='ignore', inplace=True)
            for col in frame.select_dtypes(include='object'):
                frame[col] = frame[col].map(lambda v: __import__('re').sub(
                    r'Analyzed: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC', 'Analyzed: TIMESTAMP', v)
                    if isinstance(v, str) else v)
        pd.testing.assert_frame_equal(*frames)
    assert all(run.nlp_error is None for run in runs)
    print(f'Actual local scraper, 7 vendors, test-double models: {elapsed[0]:.3f}s -> {elapsed[1]:.3f}s ({elapsed[0]/elapsed[1]:.2f}x)')

if __name__ == '__main__':
    test_pipeline()
    test_fetcher()
    test_nlp_reuse()
    test_dead_host()
    test_real_scraper()
    print('All performance regression checks passed.')
