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

  await initMap();
  initControls();
  initSearch();
  initPanel();
  applyColor();
  updateLegend();
  recomputeAll();
  renderIntro();
}

// Bandeau d'accroche : chiffres de preuve (validation croisée 2024) + rappel de méthode.
function renderIntro() {
  const p = APP.data.summary.proof_2024 || {};
  const cv = APP.data.summary.cv_halfwidth_90 || {};
  $("intro").innerHTML =
    `Prévision des <b>577 circonscriptions</b> pour les législatives 2027, au bureau de vote
     agrégé. Le modèle pose l'<b>écart local</b> de chaque circo à la moyenne nationale ;
     vous posez le <b>niveau national</b> (curseurs) et l'<b>hypothèse d'union</b> de la gauche
     (scénarios). Il en déduit, en direct, les parts par bloc, le <b>vainqueur probable</b> de
     chaque siège et la <b>jouabilité</b> pour la gauche.
     <span class="muted">Preuve : rejoué sur 2024 (désormais dans la validation croisée),
     le modèle désigne le bon bloc en tête dans <b>${p.lead_accuracy ?? "—"} %</b> des bureaux.
     Méthode complète : <a href="../" target="_blank">carte 2024 ↗</a>.</span>`;
}

boot();
