"""Capture des screenshots du site (page entière + un panneau bureau) pour envoi."""

from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8124/index.html"
OUT = Path("screenshots")


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


PICK_BUREAU = """
async () => {
  // Strongest-mobilization bureau across the top deployment departments, so the panel
  // shows a real GOTV story (per-bureau mob is small, ~tens — pick the max, not a fixed
  // threshold). Requires Left-model drivers so the "pourquoi mobilisable" bars render.
  let best = null, bestMob = -1;
  for (const dept of ['13', '75', '76', '87', '30']) {
    const d = await (await fetch(`data/detail/${dept}.json?v=${APP.DATAV}`)).json();
    for (const [loc, rec] of Object.entries(d)) {
      if (rec.mob > bestMob && rec.i > 900 && rec.gdrivers && rec.gdrivers.length) {
        best = loc; bestMob = rec.mob;
      }
    }
  }
  const com = APP.data.communes.find((c) => String(c.code_commune) === best.slice(0, 5));
  if (com) flyToCommune(com);
  await openPanel({ l: best });
  return best + ' (mob ' + bestMob + ')';
}
"""


def run() -> None:
    OUT.mkdir(exist_ok=True)
    with server(), sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1480, "height": 940}, device_scale_factor=2)
        pg.goto(URL, wait_until="networkidle")
        pg.wait_for_timeout(
            3500
        )  # laisser les tuiles de carte et les données se peindre

        pg.screenshot(path=str(OUT / "01_site_complet.png"), full_page=True)
        pg.screenshot(path=str(OUT / "02_hero_carte.png"), full_page=False)

        pg.locator("#lead").check()
        pg.wait_for_timeout(1200)
        pg.screenshot(path=str(OUT / "03_calque_bloc_en_tete.png"), full_page=False)
        pg.locator("#lead").uncheck()
        pg.wait_for_timeout(800)

        loc = pg.evaluate(PICK_BUREAU)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(OUT / "04_panneau_bureau.png"), full_page=False)
        b.close()
    print("bureau:", loc)
    for f in sorted(OUT.glob("*.png")):
        print(f"{f}  {f.stat().st_size // 1024} Ko")


if __name__ == "__main__":
    run()
