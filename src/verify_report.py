"""Vérifie le site en navigateur headless : zéro exception JS, panneaux rendus,
calque bloc en tête réactif. Lance un serveur statique le temps du test."""

from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8123/index.html"


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

        prov_rows = pg.locator("#provbars .prov-row").count()
        pollgap_rows = pg.locator("#pollgap .pg-row").count()
        deploy_rows = pg.locator("#deployment .tc-row").count()
        gamma_paths = pg.locator("#gammacurve path").count()
        gamma_legend = pg.locator("#gammacurve line").count()
        pool_rows = pg.locator("#pools .pool-row").count()
        realite_rows = pg.locator("#realite .ls-row").count()
        ic = pg.locator("#ic-explain").count()
        acc_gone = pg.locator("#accmargin").count()
        landscape = pg.locator("#landscape").count()

        mode_default = pg.evaluate("APP.state.mode")
        pg.locator("#lead").check()
        pg.wait_for_timeout(200)
        mode_lead = pg.evaluate("APP.state.mode")
        pg.locator("#lead").uncheck()
        pg.wait_for_timeout(200)
        mode_off = pg.evaluate("APP.state.mode")
        b.close()

    # network 404s for tiles/glyphs are expected offline; keep only JS faults
    js = [e for e in errors if "Failed to load resource" not in e and "ERR_" not in e]
    print(
        "prov_rows",
        prov_rows,
        "| pollgap",
        pollgap_rows,
        "| deployment",
        deploy_rows,
        "| gamma_paths",
        gamma_paths,
        "| gamma_legend",
        gamma_legend,
        "| pools",
        pool_rows,
        "| realite",
        realite_rows,
        "| ic",
        ic,
        "| #accmargin (must be 0)",
        acc_gone,
        "| #landscape",
        landscape,
        "| mode default/lead/off",
        mode_default,
        mode_lead,
        mode_off,
    )
    print("JS errors:", js if js else "none")
    ok = (
        prov_rows == 4
        and pollgap_rows == 2
        and deploy_rows == 12
        and gamma_paths == 3
        and gamma_legend == 3
        and pool_rows == 1
        and realite_rows == 1
        and ic == 1
        and acc_gone == 0
        and landscape == 0
        and mode_default == "mobil"
        and mode_lead == "lead"
        and mode_off == "mobil"
        and not js
    )
    print("VERDICT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
