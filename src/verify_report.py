"""Vérifie le site en navigateur headless : zéro exception JS, panneaux rendus,
cible/curseurs réactifs. Lance un serveur statique le temps du test."""

from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8123/index.html"


def digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


@contextmanager
def server():
    p = subprocess.Popen(
        ["python", "-m", "http.server", "8123"],
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
        pg.on(
            "console",
            lambda m: errors.append(m.text) if m.type == "error" else None,
        )
        pg.goto(URL, wait_until="networkidle")
        pg.wait_for_timeout(1500)

        acc_bars = pg.locator("#accmargin rect.accbar").count()
        cov_rows = pg.locator("#covbadge .cov-row").count()
        prov_rows = pg.locator("#provbars .prov-row").count()
        pollgap_rows = pg.locator("#pollgap .pg-row").count()
        deploy_rows = pg.locator("#deployment .tc-row").count()
        pool_rows = pg.locator("#pools .pool-row").count()
        realite_rows = pg.locator("#realite .ls-row").count()
        seat_rows = pg.locator("#seats .ls-row").count()
        fil_rows = pg.locator("#fil .fil-row").count()
        landscape = pg.locator("#landscape").count()
        flip0 = pg.locator("#flip-count").inner_text()
        seat0 = pg.locator("#seat-flip").inner_text()
        fil_flip0 = pg.locator("#fil .fil-row.flipped").count()

        mode_default = pg.evaluate("APP.state.mode")
        pg.locator("#lead").check()
        pg.wait_for_timeout(200)
        mode_lead = pg.evaluate("APP.state.mode")
        pg.locator("#lead").uncheck()
        pg.wait_for_timeout(200)
        mode_off = pg.evaluate("APP.state.mode")

        pg.locator("#sliders input[data-b=ED]").fill("3")
        pg.locator("#sliders input[data-b=ED]").dispatch_event("input")
        pg.wait_for_timeout(300)
        flip3 = pg.locator("#flip-count").inner_text()
        seat3 = pg.locator("#seat-flip").inner_text()
        fil_flip3 = pg.locator("#fil .fil-row.flipped").count()
        b.close()

    # network 404s for tiles/glyphs are expected offline; keep only JS faults
    js = [e for e in errors if "Failed to load resource" not in e and "ERR_" not in e]
    print(
        "acc_bars",
        acc_bars,
        "| cov_rows",
        cov_rows,
        "| prov_rows",
        prov_rows,
        "| pollgap",
        pollgap_rows,
        "| deployment",
        deploy_rows,
        "| pools",
        pool_rows,
        "| realite",
        realite_rows,
        "| seats",
        seat_rows,
        "| fil",
        fil_rows,
        "| #landscape",
        landscape,
    )
    print(
        "flip@0",
        flip0,
        "| flip@+3ED",
        flip3,
        "| seat@0",
        seat0,
        "| seat@+3ED",
        seat3,
        "| mode default/lead/off",
        mode_default,
        mode_lead,
        mode_off,
        "| fil flips 0/+3ED",
        fil_flip0,
        fil_flip3,
    )
    print("JS errors:", js if js else "none")
    ok = (
        acc_bars == 7
        and cov_rows == 3
        and prov_rows == 4
        and pollgap_rows == 2
        and deploy_rows == 12
        and pool_rows == 1
        and realite_rows == 1
        and seat_rows == 2
        and fil_rows == 12
        and fil_flip0 == 0
        and fil_flip3 > 0
        and landscape == 0
        and digits(flip0) == "0"
        and digits(flip3) == "6923"
        and digits(seat0) == "0"
        and digits(seat3) == "75"
        and mode_default == "mobil"
        and mode_lead == "lead"
        and mode_off == "mobil"
        and not js
    )
    print("VERDICT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
