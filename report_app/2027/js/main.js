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

function renderIntro() {
  $("intro").innerHTML =
    `<b>577 circonscriptions</b>, législatives 2027. Choisissez un <b>scénario</b> d'union et
     réglez le <b>niveau national</b> : la carte, les <b>sièges</b> et la <b>jouabilité</b>
     pour la gauche se recalculent en direct.
     <span class="muted"><a href="../" target="_blank">Méthode : carte 2024 ↗</a></span>`;
}

boot();
