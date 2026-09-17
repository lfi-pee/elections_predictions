"""Rendu réel de la page « Négocier les circonscriptions » (Playwright) : chargement du JSON,
tableau (577 lignes en « Toutes », « en jeu » par défaut), postures miroir du Python et
survolables, curseur de part LFI, tri, colonnes redimensionnables, export CSV, aucune erreur
JS, pas de défilement horizontal à 1440 px. Capture d'écran dans screenshots/.

    python3 -u -m src.test_negotiation_page_2027
"""
from __future__ import annotations

import csv
import io
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401
        pass


def main() -> None:
    served = json.loads((ROOT / "report_app/2027/data/negotiation.json").read_text())
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT / "report_app")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/2027/negotiation.html"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(url)
            page.wait_for_function("document.querySelectorAll('#rows tr').length > 0")
            assert page.locator("#rows tr").count() == served["groups"]["en_jeu"], "vue par défaut = en jeu"
            # Rien au-dessus du tableau hormis l'explication et les filtres.
            assert page.locator(".tile, .chart, #n").count() == 0
            page.select_option("#group", "")
            page.wait_for_function("document.querySelectorAll('#rows tr').length === 577")
            # Postures : miroir Python, survolables (title), et le curseur de part LFI les recalcule.
            assert page.evaluate("NEG.share === NEG.data.split.near")
            served_postures = {r["id"]: r["posture"] for r in served["rows"]}
            assert page.evaluate("Object.fromEntries(NEG.rows.map(r => [r.id, r.posture]))") == served_postures
            assert page.locator("#rows .pos[title]").count() == sum(1 for v in served_postures.values() if v)
            assert page.locator("#rows .grp[title]").count() == 577 - sum(1 for v in served_postures.values() if v)
            page.select_option("#share", served["split"]["shares"][-1])
            assert page.evaluate("NEG.rows.filter(r => r.posture === 'exiger').length") > served["postures"]["exiger"]
            page.select_option("#share", served["split"]["near"])
            page.select_option("#posture", "monnaie")
            assert page.locator("#rows tr").count() == served["postures"]["monnaie"]
            page.select_option("#posture", "")
            # Tri par chance LFI décroissante.
            page.click('th button[data-sort="p_lfi"]')
            vals = page.evaluate("negVisible().map(r => r.p_lfi).filter(v => v != null)")
            assert vals == sorted(vals, reverse=True)
            # Tient sur une largeur ; les colonnes se redimensionnent à la souris.
            assert page.evaluate("(() => { const t = document.querySelector('.table-scroll'); return t.scrollWidth <= t.clientWidth + 1; })()"), "le tableau déborde en largeur"
            th = page.locator('th[data-key="depute"]')
            w0 = th.evaluate("e => e.offsetWidth")
            box = page.locator('th[data-key="depute"] .rs').bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.down(); page.mouse.move(box["x"] + 80, box["y"] + box["height"] / 2, steps=4); page.mouse.up()
            assert th.evaluate("e => e.offsetWidth") > w0 + 40, "la colonne ne s'est pas élargie"
            assert page.locator("#neg-table.resized").count() == 1
            # Export CSV : autant de lignes que le tableau, entête stable, aucune colonne partenaire.
            with page.expect_download() as dl:
                page.click("#export")
            text = Path(dl.value.path()).read_text(encoding="utf-8-sig")
            body = [ln for ln in text.splitlines() if not ln.startswith("#")]
            rows = list(csv.reader(io.StringIO("\n".join(body)), delimiter=";"))
            assert rows[0][:3] == ["rang", "circo", "nom"] and not any("autre" in h or "prix" in h for h in rows[0])
            assert len(rows) - 1 == 577
            (ROOT / "screenshots").mkdir(exist_ok=True)
            page.screenshot(path=str(ROOT / "screenshots/negotiation_2027.png"), full_page=False)
            browser.close()
            assert not errors, errors
    finally:
        server.shutdown()
    print("OK — page négociation : tableau 577, postures survolables miroir du Python, part LFI, tri, colonnes redimensionnables, export CSV, 0 erreur JS.")


if __name__ == "__main__":
    main()
