"""Rendu réel de la page « Négocier les circonscriptions » (Playwright) : chargement du JSON,
tuiles, courbe, tableau (577 lignes en « Tous »), surlignage de la demande, tri, export CSV,
aucune erreur JS. Capture d'écran dans screenshots/ pour relecture visuelle.

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
            n_neg = served["groups"]["libre"] + served["groups"]["a_negocier"]
            assert page.locator("#rows tr").count() == n_neg, "vue par défaut = négociables"
            page.select_option("#group", "")
            page.wait_for_function("document.querySelectorAll('#rows tr').length === 577")
            assert page.locator(".tile").count() == 7
            # Rapport de force : la part LFI par défaut est la plus proche des sondages ; changer la
            # part recalcule les postures (miroir Python) et le tableau.
            assert page.evaluate("NEG.share === NEG.data.split.near")
            assert page.evaluate("NEG.rows.every(r => r.posture === (r.pub && r.group !== 'acquis' ? negPosture(r.group, r.q_lfi, r.q_other) : null))")
            served_postures = {r["id"]: r["posture"] for r in served["rows"]}
            assert page.evaluate("Object.fromEntries(NEG.rows.map(r => [r.id, r.posture]))") == served_postures
            page.select_option("#share", served["split"]["shares"][-1])
            assert page.evaluate("NEG.rows.filter(r => r.posture === 'exiger').length") > sum(1 for v in served_postures.values() if v == "exiger")
            page.select_option("#share", served["split"]["near"])
            page.select_option("#posture", "difficile")
            assert page.locator("#rows tr").count() == served["postures"]["difficile"]
            page.select_option("#posture", "")
            assert page.locator("#chart-lfi svg path.line").count() == 1
            assert page.locator("#chart-cost svg path.line").count() == 1
            # Tuiles remplies depuis le JSON (aucun chiffre figé) : le compte des acquis y figure.
            assert str(served["groups"]["acquis"]) in page.locator(".tile .v").first.text_content()
            # Demande par défaut = carte 2024 − sortant·es ; surlignage = autant de lignes.
            n_default = served["totals"]["slate2024_n"] - served["totals"]["n_acquis"]
            assert page.evaluate("NEG.n") == n_default
            assert page.locator("#rows tr.in-slate").count() == n_default
            # Le curseur bouge la lecture et le surlignage.
            page.evaluate("(() => { const s = document.getElementById('n'); s.value = 50; s.dispatchEvent(new Event('input')); })()")
            assert page.locator("#rows tr.in-slate").count() == 50
            expected = served["totals"]["acquis_expected"] + served["curve"]["cum_lfi"][49]
            txt = page.locator("#slate-read").text_content()
            assert f"{expected:.1f}".replace(".", ",") in txt, txt
            # Tri par prix décroissant.
            page.click('th button[data-sort="price"]')
            prices = page.evaluate("[...document.querySelectorAll('#rows tr td:nth-child(6)')].slice(0,5).map(t=>t.textContent)")
            vals = [float(p.replace(",", ".")) for p in prices if p != "—"]
            assert vals == sorted(vals, reverse=True), vals
            # Export CSV : autant de lignes que le tableau, entête stable.
            with page.expect_download() as dl:
                page.click("#export")
            text = Path(dl.value.path()).read_text(encoding="utf-8-sig")
            body = [ln for ln in text.splitlines() if not ln.startswith("#")]
            rows = list(csv.reader(io.StringIO("\n".join(body)), delimiter=";"))
            assert rows[0][:3] == ["rang", "circo", "nom"]
            assert len(rows) - 1 == 577
            (ROOT / "screenshots").mkdir(exist_ok=True)
            page.screenshot(path=str(ROOT / "screenshots/negotiation_2027.png"), full_page=False)
            browser.close()
            assert not errors, errors
    finally:
        server.shutdown()
    print("OK — page négociation : tuiles, courbes, tableau 577, surlignage, tri, export CSV, 0 erreur JS.")


if __name__ == "__main__":
    main()
