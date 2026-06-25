"""Headless-GeoGebra validator, packaged as a reusable class.

This is the linchpin of the data-generation pipeline: every candidate problem the
LLM produces is rendered in a real (headless) GeoGebra engine, and is accepted
only if every construction step produces an object that actually exists. It reuses
the existing `test_harness.html` (which loads GeoGebra from the CDN and exposes
`window.runSteps`).

Usage:
    with GeoGebraValidator() as v:
        ok, failed = v.validate(construction_steps)      # one random-t render
        ok, info  = v.validate_robust(steps, trials=4)   # must pass every trial
"""
import os, functools, threading, http.server, socketserver

# test_harness.html lives in the parent (geometry-gen) directory.
HARNESS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GeoGebraValidator:
    def __init__(self, harness_dir: str = HARNESS_DIR, ready_timeout_ms: int = 120000):
        self.harness_dir = harness_dir
        self.ready_timeout_ms = ready_timeout_ms
        self._httpd = None
        self._port = None
        self._pw = None
        self._browser = None
        self._page = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        handler = functools.partial(_QuietHandler, directory=self.harness_dir)
        self._httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        self._port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._page = self._browser.new_page()
        self._page.goto(f"http://127.0.0.1:{self._port}/test_harness.html")
        self._page.wait_for_function("window.__ggbReady === true", timeout=self.ready_timeout_ms)

    def stop(self):
        try:
            if self._browser: self._browser.close()
        finally:
            if self._pw: self._pw.stop()
            if self._httpd: self._httpd.shutdown()

    # ── validation ───────────────────────────────────────────────────────────
    def validate(self, construction_steps, random_t: bool = True):
        """Render once. Returns (ok, failed_step_descriptions)."""
        res = self._page.evaluate(
            "([s, r]) => window.runSteps(s, r)", [construction_steps, random_t]
        )
        failed = res.get("failed", [])
        return (len(failed) == 0, failed)

    def probe(self, construction_steps, random_t: bool = True):
        """Build once and return (failed, coords) with FULL-PRECISION point coords,
        for numeric theorem checking. coords maps point name -> [x, y]."""
        res = self._page.evaluate(
            "([s, r]) => window.probeFull(s, r)", [construction_steps, random_t]
        )
        return res.get("failed", []), res.get("coords", {})

    def validate_robust(self, construction_steps, trials: int = 4, random_t: bool = True):
        """Render `trials` times (fresh random t each). Accept only if every trial is
        clean — this rejects constructions that merely got lucky with one t.
        Returns (ok, {clean, trials, sample_failure})."""
        clean = 0
        sample = None
        for _ in range(trials):
            ok, failed = self.validate(construction_steps, random_t=random_t)
            if ok:
                clean += 1
            elif sample is None:
                sample = failed
        return (clean == trials, {"clean": clean, "trials": trials, "sample_failure": sample})


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request stderr logging
        pass


# ── smoke test: validate a known-good construction without an API key ────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    good = [
        {"name": "O", "type": "final", "class": "point", "command": "(0, 0)"},
        {"name": "c", "type": "final", "class": "circle", "command": "Circle(O, 3)"},
        {"name": "A", "type": "final", "class": "point", "command": "Point(c)"},
        {"name": "B", "type": "final", "class": "point", "command": "Point(c)"},
        {"name": "seg", "type": "final", "class": "segment", "command": "Segment(A, B)"},
    ]
    bad = [
        {"name": "A", "type": "final", "class": "point", "command": "(0, 0)"},
        {"name": "x", "type": "final", "class": "point", "command": "Intersect(nope, nada)"},
    ]
    with GeoGebraValidator() as v:
        print("good ->", v.validate_robust(good, trials=4))
        print("bad  ->", v.validate(bad))
