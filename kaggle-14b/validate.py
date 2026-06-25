"""Headless GeoGebra batch validator.

Serves test_harness.html over localhost, boots the real GeoGebra engine once,
then runs every problem in geometry_problems.jsonl through it and reports any
construction step whose object fails to exist after evalCommand.
"""
import json, os, sys, threading, functools, http.server, socketserver
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, "geometry_problems.jsonl")
TRIALS = 8   # random-t trials per problem (matches the renderer's randomizer)

def start_server():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=HERE)
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port

def main():
    problems = []
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    httpd, port = start_server()
    print(f"serving on http://127.0.0.1:{port}  ({len(problems)} problems)")

    from playwright.sync_api import sync_playwright
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{port}/test_harness.html")
        # GeoGebra engine loads from CDN; give it time.
        page.wait_for_function("window.__ggbReady === true", timeout=120000)
        print("GeoGebra engine ready. Running problems...\n")

        for prob in problems:
            pid = prob.get("id")
            total = len(prob["construction_steps"])
            clean = 0
            for _ in range(TRIALS):
                nf = page.evaluate("(d) => window.runProblemRandom(d)", prob)
                if nf == 0:
                    clean += 1
            results.append({"id": pid, "total": total, "clean": clean})
            if clean == TRIALS:
                print(f"  id {pid:>3}: OK  ({total} steps, {clean}/{TRIALS} trials)")
            else:
                print(f"  id {pid:>3}: {clean}/{TRIALS} trials clean  <-- not fully robust")

        browser.close()
    httpd.shutdown()

    bad_problems = [r["id"] for r in results if r["clean"] != TRIALS]
    print("\n" + "=" * 60)
    print(f"problems: {len(results)}   fully-clean: {len(results) - len(bad_problems)}   "
          f"not-robust: {len(bad_problems)}")
    if bad_problems:
        print("problems still showing issues:", bad_problems)

    with open(os.path.join(HERE, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("full report -> validation_report.json")

if __name__ == "__main__":
    main()
