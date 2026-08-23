"use strict";

async function boot() {
  const [summary, circoArr, circoGeo, insets, gamma] = await Promise.all([
    loadJSON("data/summary.json"),
    loadJSON("data/circo.json"),
    loadJSON("data/circo_display.geojson"),
    loadJSON("data/circo_insets.json"),
    loadJSON("data/gamma_curve.json"),
  ]);
  APP.data = { summary, circoArr, circoGeo, insets, gamma };
  APP.scenario = summary.default_scenario;
  APP.scnObj = summary.scenarios.find((s) => s.key === APP.scenario);
  APP.nat = { ...APP.scnObj.means };
  // Repère « niveau 2024 » = résultat réel 2024 calculé (backtest_2024.levels, voix brutes 1er
  // tour → 3 blocs). Pilote les pointillés des curseurs et le bouton « Rejouer 2024 ».
  if (summary.backtest_2024 && summary.backtest_2024.levels) {
    APP.REF2024 = { ...APP.REF2024, ...summary.backtest_2024.levels };
  }

  await initMap();
  initSplitter();
  initControls();
  initSearch();
  initPanel();
  applyColor();
  updateLegend();
  recomputeAll();
  renderIntro();
  renderInputs();
}

// Séparateur glissable : l'utilisateur ajuste la largeur des panneaux ↔ carte. La largeur
// (px du panneau de gauche) est bornée puis persistée ; la carte se retaille en direct.
function initSplitter() {
  const bar = $("dragbar"), side = $("side"), KEY = "p2027-side-w";
  if (!bar) return;
  const maxW = () => Math.min(760, window.innerWidth - 420);
  const apply = (w) => { side.style.width = clamp(w, 340, maxW()) + "px"; if (APP.map) APP.map.resize(); };
  const saved = parseInt(localStorage.getItem(KEY), 10);
  if (saved) apply(saved);

  let drag = false;
  const move = (e) => { if (drag) apply((e.touches ? e.touches[0].clientX : e.clientX)); };
  const start = (e) => { drag = true; bar.classList.add("drag"); document.body.style.userSelect = "none"; e.preventDefault(); };
  const stop = () => {
    if (!drag) return;
    drag = false; bar.classList.remove("drag"); document.body.style.userSelect = "";
    localStorage.setItem(KEY, parseInt(side.style.width, 10));
  };
  bar.addEventListener("mousedown", start);
  bar.addEventListener("touchstart", start, { passive: false });
  window.addEventListener("mousemove", move);
  window.addEventListener("touchmove", move, { passive: false });
  window.addEventListener("mouseup", stop);
  window.addEventListener("touchend", stop);
}

// Documente les entrées du modèle POUR CETTE élection (transparence, en bas de page).
function renderInputs() {
  const s = APP.data.summary, cv = s.cv_halfwidth_90 || {};
  const e2e = s.backtest_2024_e2e;
  const li = (t) => `<li>${t}</li>`;
  $("inputs-body").innerHTML = `<ul>
    ${li(`<b>Cible</b> : Législatives 2027, 1<sup>er</sup> tour, <b>${s.n_circo}</b> circonscriptions,
        agrégées depuis <b>${fmt(s.n_bv)}</b> bureaux de vote (${fmt(s.total_inscrits)} inscrits).`)}
    ${li(`<b>Entraînement</b> : les législatives <b>2002→2024</b> (2024 désormais incluse), en
        <i>déviations</i> par bureau à la moyenne nationale de chaque scrutin.`)}
    ${li(`<b>Prédicteurs locaux</b> : surtout l'<b>héritage de vote</b> du bureau — sa déviation
        aux deux derniers législatifs (<b>2024</b> puis <b>2022</b>) — plus <b>52 indicateurs
        INSEE</b> (dernier millésime). Modèle : Ridge (régression linéaire régularisée) + PCA,
        un réglage par bloc.`)}
    ${li(`<b>Ancre nationale</b> : posée par vous (curseurs). Présélections = intentions de vote
        1<sup>er</sup> tour, rafraîchies en 2026 (tendance agrégée PolitPro / Toute l'Europe :
        RN ~35, gauche unie ~24, Ensemble ~14, LR ~12 ; le baromètre législatif des instituts
        n'est plus mis à jour depuis oct. 2025, le suivi s'étant reporté sur la présidentielle),
        renormalisées sur 3 blocs. Abstention = axe séparé ; la baisser réaffecte les revenants
        selon la courbe γ (résultat 2024), ce qui relève les parts <i>effectives</i> de gauche.`)}
    ${li(`<b>Part de la gauche radicale (LFI)</b> : <b>réglable au curseur</b>. Défaut = dernier
        test « gauche divisée » disponible (2025 : LFI ~35 % du bloc de gauche) ; aucun sondage
        législatif 2026 ne la scinde. Sans effet en « gauche unie ».`)}
    ${li(`<b>Fourchette d'incertitude</b> (prédiction conforme, erreur locale) — demi-largeur à
        90 %, par bloc : G ±${cv.G}, C+D ±${cv.CD}, ED ±${cv.ED}, Abst. ±${cv.AB} pts.`)}
    ${li(`<b>Second tour</b> (jouabilité &amp; sièges) : qualification à <b>12,5 % des inscrits</b> ;
        reports <b>réglables au curseur</b> (défauts : barrage centre-droit→gauche 45 % contre le
        RN, report RN→gauche 15 %, <b>réunification imparfaite</b> d'une gauche divisée 72 %) ;
        « union des droites » = report LR→RN.`)}
    ${e2e ? li(`<b>Validation (à l'aveugle)</b> : 2024 <b>entièrement retiré de l'entraînement</b>,
        la chaîne complète (prévision du motif local de 1<sup>er</sup> tour, niveau national posé au réel → modèle de sièges) rejoue 2024 et
        appelle le bon vainqueur dans <b>${e2e.n_correct}/${e2e.n_circo}</b> circonscriptions
        (<b>${e2e.accuracy_seats} %</b> ; ${e2e.accuracy} % pondéré inscrits). C'est la preuve
        « hors échantillon » — le bouton <i>↻ Rejouer 2024</i> la détaille. Réserve connue : le
        modèle, privé de 2024, <b>sous-estime le RN</b> (${e2e.model.ED} sièges projetés vs
        ${e2e.actual.ED} réels).`) : ""}
  </ul>`;
}

function renderIntro() {
  $("intro").innerHTML =
    `<b>577 circonscriptions</b>, législatives 2027. Choisissez un <b>scénario</b> d'union et
     réglez le <b>niveau national</b> : la carte, les <b>sièges</b> et la <b>jouabilité</b>
     pour la gauche se recalculent en direct.
     <span class="muted"><a href="../" target="_blank">Méthode : carte 2024 ↗</a></span>`;
}

boot();
