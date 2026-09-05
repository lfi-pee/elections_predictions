// Harnais de RENDU pour le garde-fou de publication (circos non couvertes par la nomenclature).
//
// Marquer les circos ne sert à rien si le site oublie de le montrer. Ce harnais exécute le VRAI
// JS du site (config + compute + coverage + panel + map + controls) sur les VRAIES données
// servies, rend le panneau de quelques circos et renvoie ce qui y figure — que
// src/test_coverage_2027.py compare à ce qui DOIT y figurer.
//
//   node src/coverage_harness.js <js_dir> <data_dir> <ids.json>

const fs = require("fs");
const vm = require("vm");
const path = require("path");

const [jsDir, dataDir, idsFile] = process.argv.slice(2);
const FILES = ["config.js", "compute.js", "coverage.js", "panel.js", "map.js", "controls.js"];
const src = FILES.map((f) => fs.readFileSync(path.join(jsDir, f), "utf8")).join("\n") +
  "\nglobalThis.__A = { APP, initCoverage, covUnpublishableIds, covIsPublishable, covWarning," +
  " renderCirco, updateLegend, winColorExpr, seatColorExpr };";

// DOM minimal : un élément unique rendu par getElementById, suffisant pour updateLegend et
// pour que panel.js/controls.js se chargent sans navigateur.
const el = {
  innerHTML: "", textContent: "", style: {},
  classList: { add() {}, remove() {}, toggle() {} },
  scrollIntoView() {}, addEventListener() {}, querySelectorAll: () => [],
};
const sandbox = {
  console, Math, JSON, Array, Object, Number, Map, Set, String, Boolean,
  isNaN, parseFloat, parseInt,
  document: { getElementById: () => el, querySelectorAll: () => [], addEventListener() {} },
  window: {}, localStorage: { getItem: () => null, setItem() {} },
  matchMedia: () => ({ matches: false }),
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: "coverage_bundle.js" });

const A = sandbox.__A, APP = A.APP;
const read = (f) => JSON.parse(fs.readFileSync(path.join(dataDir, f), "utf8"));
APP.data = { summary: read("summary.json"), circoArr: read("circo.json"),
             gamma: read("gamma_curve.json"), coverage: read("coverage.json") };
const a = APP.data.circoArr;
APP.idIdx = new Map(a.id.map((id, i) => [id, i]));
APP.scnObj = APP.data.summary.scenarios.find((s) => s.key === APP.data.summary.default_scenario);
APP.nat = { ...APP.scnObj.means };
APP.map = { getLayer: () => null, setFilter() {} };
A.initCoverage();

const props = (id) => {
  const i = APP.idIdx.get(id);
  return { id: a.id[i], nm: a.nm[i], dept: a.dept[i], ins: a.ins[i], nbv: a.nbv[i],
           dG: a.dG[i], dCD: a.dCD[i], dED: a.dED[i], dAB: a.dAB[i], rdev: a.rdev[i] };
};

const panels = {};
for (const id of JSON.parse(fs.readFileSync(idsFile, "utf8"))) {
  const html = A.renderCirco(props(id));
  panels[id] = {
    publishable: A.covIsPublishable(id),
    warning_box: html.includes("pv-nopub"),
    unmeasured_tag: html.includes(APP.COV_LAB),
    seat_line: html.includes("Siège probable"),
    no_score_heading: html.includes("Pourquoi pas de score"),
    warning_text: A.covWarning(id).replace(/\s+/g, " ").trim(),
  };
}

A.updateLegend();
process.stdout.write(JSON.stringify({
  unpublishable: A.covUnpublishableIds(),
  n_low: APP.cov.nLow,
  legend_has_chip: el.innerHTML.includes(APP.COV_LAB),
  win_color_expr: JSON.stringify(A.winColorExpr()),
  seat_color_expr: JSON.stringify(A.seatColorExpr()),
  grey: APP.COV_GREY,
  panels,
}));
