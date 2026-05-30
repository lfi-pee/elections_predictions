"""Vérifie que le panneau-instrument rend la viz quantitative des contributions
(barres divergentes, libellés lisibles) sans exception JS."""

from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8124/index.html"


@contextmanager
def server():
    p = subprocess.Popen(
        ["python", "-m", "http.server", "8124"],
        cwd="report_app",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.2)
    try:
        yield
    finally:
        p.terminate()


def run() -> None:
    errors: list[str] = []
    with server(), sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page()
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.goto(URL, wait_until="networkidle")
        pg.wait_for_timeout(1200)

        pg.evaluate("openPanel({ l: '01002_0001' })")
        pg.wait_for_selector("#panel:not(.hidden) .dv-row", timeout=5000)
        rows = pg.locator(".dv-row").count()
        header = pg.locator(".pv-why-h").inner_text()
        labels = pg.eval_on_selector_all(
            ".dv-lab", "els => els.map(e => e.textContent)"
        )
        vals = pg.eval_on_selector_all(".dv-val", "els => els.map(e => e.textContent)")
        bad = [x for x in labels if any(t in x for t in ("lag", "Pct", "Taux", "_"))]
        b.close()

    assert not errors, f"JS errors: {errors}"
    assert rows == 6, f"expected 6 driver rows, got {rows}"
    assert not bad, f"unreadable labels: {bad}"
    print(f"panel OK · {rows} barres · « {header} »")
    print("labels:", labels)
    print("valeurs:", vals)


if __name__ == "__main__":
    run()
