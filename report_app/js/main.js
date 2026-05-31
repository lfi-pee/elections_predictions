"use strict";

function injectSummary() {
  const s = APP.data.summary;
  $("hook-acc").textContent = s.lead_accuracy.toLocaleString("fr-FR", { minimumFractionDigits: 1 }) + " %";
  $("hook-n").textContent = fmt(s.n_bv);
}

function wireControls() {
  $("lead").addEventListener("change", (e) => setMode(e.target.checked ? "lead" : "mobil"));
}

async function boot() {
  const [summary, communes, national, provenance, gamma] = await Promise.all([
    loadJSON("data/summary.json"),
    loadJSON("data/communes.json"),
    loadJSON("data/national.json"),
    loadJSON("data/provenance.json"),
    loadJSON("data/gamma_curve.json"),
  ]);
  APP.data = { summary, communes, national, provenance, gamma };
  injectSummary();

  await initMap();
  ensureDept("69");

  renderRealite();
  renderPollGap();
  renderProvenance();
  renderPools();
  renderDeployment();
  renderGamma();
  initSearch();
  initPanel();
  wireControls();
  updateLegend();
}

boot();
