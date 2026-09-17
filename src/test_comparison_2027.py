"""Browser checks for the comparison table. Run: python -m src.test_comparison_2027."""
from __future__ import annotations

import csv
import io
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT / "report_app")))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/2027/comparison.html"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url)
            page.wait_for_function("document.querySelectorAll('#rows tr').length === 577")
            assert page.locator('th[scope="colgroup"]').all_text_contents() == [
                "Prédiction 2027 (modèle)", "Résultats passés et extrapolations simples de 2024"]
            # Tri par défaut : prédiction 2027 décroissante.
            assert page.evaluate("COMPARISON.sort === 'reference' && COMPARISON.ascending === false")
            assert page.evaluate("(() => { const v = comparisonVisibleRows().map(r => r.reference?.G).filter(x => x != null); return v.every((x, i) => !i || x <= v[i-1]); })()")
            assert page.locator('#preset').input_value() == 'turnout2024'
            assert page.evaluate("COMPARISON.variant.AB === APP.data.history.elections.find(e=>e.key==='2024').participation.abstention_pct")
            assert page.evaluate("COMPARISON.rows.some(r=>r.reference && Math.abs(r.reference.G-r.variant.G)>0.1)")
            initial_variant = page.evaluate("COMPARISON.rows.map(r=>r.variant)")
            for year in ('2022', '2024'):
                page.locator('#preset').select_option('turnout' + year)
                assert page.evaluate("COMPARISON.variant.AB === APP.data.history.elections.find(e=>e.key===COMPARISON.preset.slice(-4)).participation.abstention_pct")
                assert page.evaluate("APP.VOTE.every(b=>COMPARISON.variant[b]===COMPARISON.reference[b])")
                assert year in page.locator('[data-sort=variant]').inner_text()
            page.locator('#preset').select_option('reference')
            assert page.evaluate("COMPARISON.rows.every(r=>JSON.stringify(r.reference)===JSON.stringify(r.variant))")
            page.locator('#preset').select_option('turnout2024')
            assert page.evaluate('''() => COMPARISON.rows.every(r =>
                [r.h2017, r.h2022, r.h2024].every(h => h && Object.values(h).every(Number.isFinite)))''')
            # Published table scores must match the existing calculator, with identical parameters.
            assert page.evaluate('''() => {
                let ok = true;
                eachCirco((r, i) => {
                    const row = COMPARISON.rows[i];
                    if (!covIsPublishable(row.id)) {
                        ok = ok && [row.reference,row.variant,row.uniform,row.proportional].every(v => v === null);
                    } else {
                        ok = ok && Math.abs(row.reference.G - r.g) < 1e-10 &&
                            Math.abs(row.reference.CD - r.cd) < 1e-10 && Math.abs(row.reference.ED - r.ed) < 1e-10;
                    }
                }); return ok;
            }''')
            before = page.evaluate('''() => COMPARISON.rows.map(r => [r.reference, r.uniform, r.proportional, r.h2024])''')
            # Moving a national control preserves the initial three-block budget (Autre fixed).
            page.locator('#level').fill(str(page.evaluate('COMPARISON.reference.G')))
            assert page.evaluate("Math.abs(COMPARISON.variant.CD - COMPARISON.reference.CD) < 1e-9")
            page.locator('#level').fill('40')
            assert page.evaluate("Math.abs(APP.VOTE.reduce((s,b)=>s+COMPARISON.variant[b]-COMPARISON.reference[b],0)) < 1e-9")
            page.locator('#ab').fill('35')
            assert page.locator('#preset').input_value() == 'custom'
            assert page.evaluate('''() => COMPARISON.rows.some(r => r.reference && Math.abs(r.reference.G-r.variant.G)>0.1)''')
            after = page.evaluate('''() => COMPARISON.rows.map(r => [r.reference, r.uniform, r.proportional, r.h2024])''')
            assert before == after, "Changing the variant must preserve reference and historical columns"
            page.locator('[data-sort="variant"]').click()
            for ascending in (False, True):
                assert page.evaluate('''() => {
                    const values = comparisonVisibleRows().map(r => r.variant?.G ?? null);
                    const last = values.indexOf(null);
                    if (last >= 0 && values.slice(last).some(v => v !== null)) return false;
                    const nums = values.filter(v => v !== null);
                    return nums.slice(1).every((v,i) => COMPARISON.ascending ? v>=nums[i] : v<=nums[i]);
                }''')
                if not ascending:
                    page.locator('[data-sort="variant"]').click()
            page.locator('#reset-variant').click()
            assert page.locator('#preset').input_value() == 'turnout2024'
            assert page.evaluate('COMPARISON.rows.map(r=>r.variant)') == initial_variant
            for bloc in ('CD', 'ED', 'LFI', 'AG', 'G'):
                page.locator('#bloc').select_option(bloc)
                assert page.evaluate('COMPARISON.block') == bloc
                assert page.locator('#rows tr').count() == 577
            # LFI seule : 2027 = G × part locale ; 2017 séparable (nuance FI), 2022/2024 non.
            assert page.evaluate("COMPARISON.rows.every(r => !r.reference || Math.abs(r.reference.LFI + r.reference.AG - r.reference.G) < 1e-9)")
            assert page.evaluate("COMPARISON.rows.filter(r => r.h2017 && r.h2017.LFI != null).length > 500")
            assert page.evaluate("COMPARISON.rows.every(r => !r.h2024 || r.h2024.LFI == null)")
            # Le curseur « part LFI » ne bouge que la variante LFI/AG, pas la gauche entière.
            g_before = page.evaluate("COMPARISON.rows.map(r => r.variant && r.variant.G)")
            page.locator('#share').fill('60')
            assert page.locator('#preset').input_value() == 'custom'
            assert page.evaluate("COMPARISON.rows.map(r => r.variant && r.variant.G)") == g_before
            assert page.evaluate("COMPARISON.rows.some(r => r.reference && r.variant.LFI > r.reference.LFI + 1)")
            page.locator('#reset-variant').click()
            assert page.evaluate("COMPARISON.share === COMPARISON.refShare")
            page.locator('#filter').fill('01-01')
            assert page.locator('#rows tr').count() == 1
            with page.expect_download() as download:
                page.locator('#export').click()
            exported = Path(download.value.path()).read_text(encoding='utf-8-sig')
            rows = list(csv.DictReader(io.StringIO(exported), delimiter=';'))
            assert len(rows) == 1 and rows[0]['circonscription'] == '01-01'
            assert '2024' in rows[0]['sources_et_methodes']
            assert 'AB' in rows[0]['parametres_variante']
            assert rows[0]['prereglage_variante'] == 'Participation 2024'
            assert 'abstention_pct' in rows[0]['sources_et_methodes']
            assert abs(float(rows[0]['Prédiction 2027 (% exprimés)']) - page.evaluate('comparisonVisibleRows()[0].reference.G')) < 1e-9
            page.locator('#filter').fill('no-such-circo')
            assert page.locator('#rows tr').count() == 0
            page.locator('#filter').fill('')
            page.screenshot(path='/tmp/comparison-2027-desktop.png', full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), "Page overflows on mobile"
            page.screenshot(path='/tmp/comparison-2027-mobile.png', full_page=True)
            assert not errors, errors
            # A failed historical download must show a visible error and keep export disabled.
            failed = browser.new_page()
            failed.route('**/comparison_history.json*', lambda route: route.fulfill(status=503, body='unavailable'))
            failed.goto(url)
            failed.wait_for_function("document.querySelector('#status').textContent.includes('Impossible')")
            assert failed.locator('#export').is_disabled()
            browser.close()
        print('PASS: 577 rows, historical coverage, model parity, independent variant, all blocks, sorting, search, CSV, mobile, loading failure')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
