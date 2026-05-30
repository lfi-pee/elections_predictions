"use strict";

function injectSummary() {
  const s = APP.data.summary;
  $("hook-acc").textContent = s.lead_accuracy.toLocaleString("fr-FR", { minimumFractionDigits: 1 }) + " %";
  $("hook-n").textContent = fmt(s.n_bv);
}

function wireControls() {
  const layers = ["lead", "honesty"];
  const exclusive = (on) => layers.forEach((l) => { if (l !== on) $(l).checked = false; });
  for (const l of layers) {
    $(l).addEventListener("change", (e) => {
      if (e.target.checked) exclusive(l);
      setMode(e.target.checked ? l : "mobil");
    });
  }
}

function precomputeBaseLead() {
  const d = APP.data.national, n = d.pg.length, base = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const g = d.pg[i], c = d.pc[i], e = d.pe[i];
    base[i] = g >= c && g >= e ? 0 : c >= e ? 1 : 2;
  }
  APP.data.baseLead = base;
}

function precomputeSeatBase() {
  const c = APP.data.circo, n = c.g.length, base = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const g = c.g[i], cd = c.c[i], e = c.e[i];
    base[i] = g >= cd && g >= e ? 0 : cd >= e ? 1 : 2;
  }
  APP.data.seatBase = base;
}

function resizeBurst() {
  let n = 0;
  const id = setInterval(() => { APP.map.resize(); if (++n > 14) clearInterval(id); }, 30);
}

function initMinimap() {
  const setMini = (want) => {
    if (want === document.body.classList.contains("mini")) return;
    document.body.classList.toggle("mini", want);
    if (!want) document.body.classList.remove("expanded");
    resizeBurst();
  };
  new IntersectionObserver(
    ([e]) => setMini(!e.isIntersecting && e.boundingClientRect.top < 0),
    { rootMargin: "-60px 0px 0px 0px", threshold: 0 },
  ).observe($("hero-end"));
  $("map-expand").addEventListener("click", () => {
    document.body.classList.toggle("expanded");
    resizeBurst();
  });
}

async function boot() {
  const [summary, communes, national, provenance, coverage, circo] = await Promise.all([
    loadJSON("data/summary.json"),
    loadJSON("data/communes.json"),
    loadJSON("data/national.json"),
    loadJSON("data/provenance.json"),
    loadJSON("data/coverage.json"),
    loadJSON("data/circo.json"),
  ]);
  APP.data = { summary, communes, national, provenance, coverage, circo };
  APP.SWING = summary.swing;
  precomputeBaseLead();
  precomputeSeatBase();
  injectSummary();

  await initMap();
  ensureDept("69");
  initMinimap();

  renderRealite();
  renderAccuracy();
  renderCoverage();
  renderPollGap();
  renderProvenance();
  renderPools();
  renderDeployment();
  buildSliders();
  initSearch();
  initPanel();
  wireControls();
  updateLegend();
  updateScenario();
}

boot();
