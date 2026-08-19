"use strict";

function wireControls() {
  $("lead").addEventListener("change", (e) => setMode(e.target.checked ? "lead" : "mobil"));
  // On a tablet the intro card sits on top of the map and swallows a quarter of it,
  // so it has to be dismissable — otherwise taps in that area hit the text, not a zone.
  const ov = document.querySelector(".hero-overlay"), tog = $("ov-toggle");
  tog.onclick = () => {
    const folded = ov.classList.toggle("folded");
    tog.textContent = folded ? "Afficher le texte" : "Masquer le texte";
    tog.setAttribute("aria-expanded", String(!folded));
  };
  if (window.matchMedia("(pointer:coarse)").matches) tog.click();
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
