"""Rendu réel de la page « Négocier les circonscriptions » (Playwright) : chargement du JSON,
tableau (577 lignes en « Toutes », « en jeu » par défaut), postures miroir du Python et
survolables, argumentaire (registre par posture, faits publics seulement, jamais un chiffre
faible sur un siège qu'on demande), curseur de part LFI, tri, colonnes redimensionnables,
export CSV, aucune erreur JS, pas de défilement horizontal à 1440 px. Capture d'écran dans
screenshots/.

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

from src import negotiation_2027 as N

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
            # Vue par défaut : toutes les circonscriptions (« rien à jouer » sort des circos en
            # jeu pour LFI, les quatre postures doivent être visibles d'emblée).
            assert page.locator("#rows tr").count() == 577, "vue par défaut = toutes"
            assert page.locator("#rows .pos[data-tip]", has_text="Monnaie").count() == served["postures"]["monnaie"]
            # Rien au-dessus du tableau hormis l'explication et les filtres.
            assert page.locator(".tile, .chart, #n").count() == 0
            page.select_option("#group", "en_jeu")
            page.wait_for_function(f"document.querySelectorAll('#rows tr').length === {served['groups']['en_jeu']}")
            page.select_option("#group", "")
            page.wait_for_function("document.querySelectorAll('#rows tr').length === 577")
            # Infobulles des postures : la règle de calcul (seuils) y figure.
            tip = page.locator("#rows .pos[data-tip]", has_text="Exiger").first.get_attribute("data-tip")
            assert "≥ 5 %" in tip and "≥ 50 %" in tip, tip
            # Ventilations : gauche 2024 par nuance (NFP + hors NFP) ; 2027 partagé LFI / reste.
            row = page.locator("#rows tr", has_text="93-01").first
            assert "NFP-" in row.locator("td").nth(8).inner_text()
            t27 = row.locator("td").nth(11).inner_text()
            assert all(k in t27 for k in ("LFI", "PS", "Écolo.", "PCF")), t27
            # Mélenchon : part brute, moyenne nationale affichée sous le titre.
            mel_txt = row.locator("td").nth(10).inner_text()
            assert mel_txt.strip() == f"{round(served_row_mel := next(r for r in served['rows'] if r['id'] == '93-01')['mel'] * 100)} %", mel_txt
            assert page.locator("#mel-nat").inner_text().strip() == f"{round(served['presidential']['national']['LFI'] * 100)} %"
            # Pastille : la règle puis les chiffres de la ligne.
            tipm = page.locator("#rows .pos[data-tip]", has_text="Monnaie").first.get_attribute("data-tip")
            assert "gauche unie" in tipm and "Ici :" in tipm, tipm
            # Chaque chiffre cité au survol a sa colonne : le survol n'invente aucune valeur.
            for key in ("p_lfi", "p_left", "q_lfi", "q_oth"):
                assert page.locator(f'th[data-key="{key}"]').count() == 1, key
            # Argumentaire : une carte par ligne, construite sur les seules colonnes de droite.
            assert page.locator('th[data-key="arg"]').count() == 1
            assert page.locator("#rows .arg[data-card]").count() == 577
            # Amener la ligne à l'écran AVANT de survoler : le défilement est asynchrone dans
            # Chrome, et l'événement `scroll` (qui masque l'infobulle) arriverait après le survol.
            # Puis écarter la souris, sinon `hover()` ne la déplace pas et aucun mouseover ne part.
            argp = page.locator("#rows tr", has_text="92-01").first.locator(".arg")
            argp.scroll_into_view_if_needed()
            page.mouse.move(0, 0)
            page.wait_for_timeout(100)
            argp.hover()
            card = page.locator(".tt.card")
            assert card.is_visible() and "Exiger" in card.inner_text(), card.inner_text()
            # text_content, pas inner_text : les intertitres de la carte sont en petites capitales CSS.
            txt = card.text_content()
            assert "À mettre sur la table" in txt and "Ce que le partenaire sortira" in txt, txt
            assert card.locator("li").count() >= 2
            # La règle de négociation, ligne à ligne : sur un siège qu'on demande (exiger, obtenir)
            # comme sur un siège dont on fait payer la cession (monnaie), aucun fait défavorable ne
            # passe du côté de ce qu'on met sur la table ; « rien à jouer » fait exactement
            # l'inverse (le chiffre défavorable, avancé le premier).
            bad = page.evaluate("""() => NEG.rows.filter(r => ['exiger','obtenir','monnaie'].includes(r.posture))
              .filter(r => argFacts(r).some(f => !f.pro && r.card.pos.includes(f.text))).map(r => r.id)""")
            assert bad == [], bad
            conceded = page.evaluate("""() => NEG.rows.filter(r => r.posture === 'rien')
              .filter(r => r.card.pos.length && !argFacts(r).filter(f => !f.pro).map(f => f.text).includes(r.card.pos[0])).map(r => r.id)""")
            assert conceded == [], conceded
            # Aucune probabilité du modèle ne fuit dans un argumentaire : rien de contestable.
            leak = page.evaluate("""() => NEG.rows.flatMap(r => r.card.pos.concat(r.card.opp))
              .filter(t => /chance|probabilit|sans accord|2nd tour/i.test(t))""")
            assert leak == [], leak[:2]
            # « Gauche hors union » : la carte s'interdit en pied d'avancer le score de gauche
            # d'ici, gonflé par un·e sortant·e hors union — donc aucun fait posé ne le cite.
            gonfle = page.evaluate("""() => NEG.rows.filter(r => r.group === 'hors_union')
              .filter(r => r.card.pos.some(t => /part par part|de toute la gauche ici/.test(t))).map(r => r.id)""")
            assert gonfle == [], gonfle
            # Le registre suit la posture, et la pastille compte les faits disponibles.
            heads = page.evaluate("""() => Object.fromEntries(['exiger','obtenir','monnaie','rien']
              .map(k => [k, [...new Set(NEG.rows.filter(r => r.posture === k).map(r => r.card.head))]]))""")
            assert all(len(v) == 1 for v in heads.values()), heads
            assert page.evaluate("NEG.rows.every(r => r.card.n === r.card.pos.length)")
            # La pastille dit la même chose que l'intitulé de la liste qu'elle annonce : sur un
            # siège de sortant·e LFI, les faits ne se posent que si le siège est contesté.
            assert page.evaluate("""() => NEG.rows.filter(r => r.group === 'acquis' && r.card.n)
              .every(r => r.card.badge.includes('si contesté'))""")
            # Le partage 2027 suit le filtre de part LFI et respecte la règle servie.
            served_row = next(r for r in served["rows"] if r["id"] == "93-01")
            def lfi27(share):
                lo, hi = served["params"]["rad_clip"]
                rad = min(hi, max(lo, share + served["params"]["rad_gain"] * (served_row["rdev"] or 0)))
                return round(served_row["ext_plus_G"] * rad, 1)
            def shown_lfi():
                import re as _re
                m = _re.search(r"LFI\s+([\d,]+)", row.locator("td").nth(11).inner_text())
                return float(m.group(1).replace(",", "."))
            assert abs(shown_lfi() - lfi27(float(served["split"]["near"]))) < 0.11
            page.select_option("#share", served["split"]["shares"][-1])
            assert abs(shown_lfi() - lfi27(float(served["split"]["shares"][-1]))) < 0.11
            page.select_option("#share", served["split"]["near"])
            # Postures : miroir Python, survolables (title), et le curseur de part LFI les recalcule.
            assert page.evaluate("NEG.share === NEG.data.split.near")
            served_postures = {r["id"]: r["posture"] for r in served["rows"]}
            assert page.evaluate("Object.fromEntries(NEG.rows.map(r => [r.id, r.posture]))") == served_postures
            assert page.locator("#rows .pos[data-tip]").count() == sum(1 for v in served_postures.values() if v)
            assert page.locator("#rows .grp[data-tip]").count() == 577 - sum(1 for v in served_postures.values() if v)
            # Le « Calcul » d'une pastille ne doit citer QUE ce que la règle lit. Nommer la chance
            # de la gauche unie y rendait chaque pastille falsifiable avec les chiffres de sa
            # propre ligne : à 25 % de part LFI, 30-01 satisfaisait mot pour mot le calcul annoncé
            # pour « monnaie d'échange » tout en affichant « rien à jouer ».
            tips = page.evaluate("Object.fromEntries(['exiger','obtenir','monnaie','rien']"
                                 ".map(k => [k, POSTURE_TIP[k]]))")
            assert not any("gauche unie" in v for v in tips.values()), tips
            assert all("député·e LFI" in v for v in tips.values()), tips
            # L'infobulle apparaît immédiatement au survol d'une pastille et d'un en-tête, puis disparaît.
            page.locator("#rows .pos[data-tip]").first.hover()
            assert page.locator(".tt").is_visible() and len(page.locator(".tt").text_content()) > 20
            page.locator('th[data-key="mel"] button').hover()
            assert "Mélenchon" in page.locator(".tt").text_content()
            page.locator("h1").hover()
            assert not page.locator(".tt").is_visible()
            # Le miroir JS ↔ Python doit tenir à CHAQUE cran du curseur, pas au seul cran servi :
            # c'est le curseur qui recalcule les options extérieures, donc les postures. Un test
            # au cran par défaut ne peut pas voir une règle qui dérape ailleurs.
            nd, zc = served["params"]["draws"], served["params"]["ci_z"]
            lev = served["split"]["leverage_q"]
            def crosses(v, t):
                if v is None:
                    return False
                lo, hi = N.wilson(v, nd, zc)
                return lo < t < hi
            for k in served["split"]["shares"]:
                page.select_option("#share", k)
                b = served["split"]["by_share"][k]
                want = {r["id"]: N.posture(r["group"], b["q_lfi"][i], b["q_oth"][i])
                        for i, r in enumerate(served["rows"])}
                assert page.evaluate("Object.fromEntries(NEG.rows.map(r => [r.id, r.posture]))") == want, k
                assert page.evaluate("NEG.rows.every(r => r.posture === null || r.posture === 'rien'"
                                     " || r.group === 'en_jeu')"), k
                # Le compte « à cheval sur un seuil » ne doit compter QUE les lignes dont le seuil
                # décide la posture. q_oth n'est lu que si q_lfi est SOUS le seuil : au-dessus,
                # la posture vaut « exiger » quoi que fasse q_oth. Sans cette condition la page
                # annonçait 7 postures indécises pour 5 réelles à la part 0,55 — et ce cran-là
                # n'apparaît qu'en balayant tout le curseur, d'où le test dans cette boucle.
                want_q = sum(1 for i, r in enumerate(served["rows"])
                             if r["group"] == "en_jeu"
                             and (crosses(b["q_lfi"][i], lev)
                                  or (b["q_lfi"][i] is not None and b["q_lfi"][i] < lev
                                      and crosses(b["q_oth"][i], lev))))
                assert page.evaluate("straddle().q") == want_q, (k, page.evaluate("straddle()"), want_q)
                decided = page.evaluate(
                    "(lv) => NEG.rows.filter(r => r.group === 'en_jeu')"
                    ".every(r => !(r.q_lfi >= lv) || negPosture(r.group, r.q_lfi, r.q_oth) === 'exiger')", lev)
                assert decided, k
            page.select_option("#share", served["split"]["shares"][-1])
            assert page.evaluate("NEG.rows.filter(r => r.posture === 'exiger').length") > served["postures"]["exiger"]
            page.select_option("#share", served["split"]["near"])
            # Bornes à 95 % sous chaque probabilité : rendues, et miroir exact du Wilson Python.
            row0 = served["rows"][0]
            shown = page.evaluate("""() => [...document.querySelectorAll('#rows tr')]
                .find(tr => tr.querySelector('th').textContent.includes(%r))
                .querySelectorAll('td.num .pb small')""" % row0["id"]
                + """.length""")
            assert shown == 4, shown
            # Le texte des bornes est un miroir exact du Python, sur TOUTE la plage de valeurs :
            # bords compris, où la convention « jamais de certitude » doit aussi s'appliquer.
            vals = sorted({r["p_lfi"] for r in served["rows"] if r["p_lfi"] is not None}
                          | {0.0, 0.004, 0.05, 0.5, 0.996, 1.0})
            got = page.evaluate("(vs) => vs.map(v => ciTxt(v))", vals)
            assert got == [N.ci_txt(v, served["params"]["draws"]) for v in vals], [
                (v, g, N.ci_txt(v, served["params"]["draws"]))
                for v, g in zip(vals, got) if g != N.ci_txt(v, served["params"]["draws"])][:5]
            assert "100" not in page.evaluate("ciTxt(1)"), page.evaluate("ciTxt(1)")
            # `wilsonCI` comparé au Python NUMÉRIQUEMENT, pas seulement à travers l'arrondi
            # d'affichage de `ciTxt` : deux formules divergentes peuvent s'écrire pareil une fois
            # arrondies au point de pourcentage, et c'est la moitié basse de la colonne qui le
            # cacherait le mieux.
            probe = [0.0, 1e-4, 0.004, 0.05, 0.2, 0.5, 0.8, 0.95, 0.999, 1.0]
            js_ci = page.evaluate("(vs) => vs.map(v => wilsonCI(v, NEG.data.params.draws, NEG.data.params.ci_z))", probe)
            for v, (jl, jh) in zip(probe, js_ci):
                pl, ph = N.wilson(v, nd, zc)
                assert abs(jl - pl) < 1e-12 and abs(jh - ph) < 1e-12, (v, (jl, jh), (pl, ph))
            # Le flou de rang est SERVI (il porte sur un écart, que la page ne peut pas redériver
            # de ses bornes marginales) : la page doit lire ce chiffre-là, pas en recalculer un.
            assert page.evaluate("rankBlur()") == served["params"]["rank_blur"]
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
            # Le CSV est l'endroit où l'on soustrait deux colonnes : la réserve sur des bornes
            # marginales doit y figurer, pas seulement dans la page.
            note = " ".join(ln for ln in text.splitlines() if ln.startswith("#"))
            assert "PRIS SEUL" in note and "mêmes" in note and "tirages" in note, note
            body = [ln for ln in text.splitlines() if not ln.startswith("#")]
            rows = list(csv.reader(io.StringIO("\n".join(body)), delimiter=";"))
            assert rows[0][:3] == ["rang", "circo", "nom"] and not any("autre" in h or "prix" in h for h in rows[0])
            assert rows[0][-1] == "argumentaire" and all(len(r[-1]) > 60 for r in rows[1:])
            assert len(rows) - 1 == 577
            # Chaque probabilité exportée porte ses bornes, et toutes les lignes ont la même
            # largeur que l'entête (un décalage d'une colonne rendrait le CSV muet et faux).
            assert [h for h in rows[0] if h.endswith(("_ic95_bas", "_ic95_haut"))] == [
                "chance_depute_lfi_ic95_bas", "chance_depute_lfi_ic95_haut",
                "chance_gauche_unie_ic95_bas", "chance_gauche_unie_ic95_haut",
                "lfi_seule_ic95_bas", "lfi_seule_ic95_haut",
                "reste_gauche_seul_ic95_bas", "reste_gauche_seul_ic95_haut"], rows[0]
            assert {len(r) for r in rows} == {len(rows[0])}, {len(r) for r in rows}
            iv = rows[0].index("chance_depute_lfi")
            assert all(float(r[iv + 1]) <= float(r[iv]) <= float(r[iv + 2])
                       for r in rows[1:] if r[iv]), "bornes CSV n'encadrent pas la chance"
            (ROOT / "screenshots").mkdir(exist_ok=True)
            page.screenshot(path=str(ROOT / "screenshots/negotiation_2027.png"), full_page=False)
            browser.close()
            assert not errors, errors
    finally:
        server.shutdown()
    print("OK — page négociation : tableau 577, postures survolables miroir du Python, argumentaire par posture (faits publics seulement), part LFI, tri, colonnes redimensionnables, export CSV, 0 erreur JS.")


if __name__ == "__main__":
    main()
