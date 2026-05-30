"use strict";
const APP = {
  COL: { G: "#E4572E", CD: "#4A90D9", ED: "#6A4C93", AB: "#9AA0A6" },
  // pale = each bloc hue mixed ~78% toward paper; the map fades to it as the
  // margin shrinks, so a knife-edge bureau reads pale and a landslide reads solid.
  PALE: { G: "#F5D5C9", CD: "#CFE1F2", ED: "#DBD2E8" },
  MARGIN_FULL: 12,
  NAME: { G: "Gauche", CD: "Centre+Droite", ED: "Extrême Droite", AB: "Abstention" },
  VOTE: ["G", "CD", "ED"],
  LYON: { center: [4.8357, 45.758], zoom: 12 },
  state: { dG: 0, dC: 0, dE: 0, mode: "mobil", marginT: 5 },
  data: {},
  map: null,
  bvByDept: new Map(),
  // Bump on any change to served data files (national/communes/bv/detail/…) so the
  // browser refetches instead of serving a stale cache. Appended to every loadJSON.
  DATAV: "5",
};

const $ = (id) => document.getElementById(id);
const fmt = (n) => Math.round(n).toLocaleString("fr-FR");
const fmtM = (n) => (n / 1e6).toLocaleString("fr-FR", { maximumFractionDigits: 1 }) + " M";

async function loadJSON(path) {
  const sep = path.includes("?") ? "&" : "?";
  const r = await fetch(path + sep + "v=" + APP.DATAV);
  if (!r.ok) throw new Error("load " + path);
  return r.json();
}

// Vote-share-conserving scenario deltas: a national gain for one bloc is funded
// proportionally by the others (applied_j = d_j − Σ_{k≠j} d_k·w_j/(1−w_k)), so the
// three always sum to 0 — no vote conjured from nowhere. Mirrors report_data.conserved.
function conservedDeltas(d) {
  const w = APP.SWING, a = {};
  for (const j of APP.VOTE) {
    a[j] = d[j] - APP.VOTE.reduce((acc, k) => (k === j ? acc : acc + d[k] * w[j] / (1 - w[k])), 0);
  }
  return a;
}

function appliedDeltas() {
  const s = APP.state;
  return conservedDeltas({ G: s.dG, CD: s.dC, ED: s.dE });
}

// Lead-block color expression for the current scenario deltas.
// gK = ["+", ["get", key], appliedK] so the map recolours instantly on slider move.
// Hue = winning bloc; saturation = decisiveness: we interpolate from the pale tone
// (margin 0) to the full tone (margin ≥ MARGIN_FULL). Margin = top − second, both
// computed live so a bureau that a slider just flipped correctly reads pale.
function leadColorExpr(keys) {
  const a = appliedDeltas();
  const g = ["+", ["get", keys.G], a.G];
  const c = ["+", ["get", keys.CD], a.CD];
  const e = ["+", ["get", keys.ED], a.ED];
  const margin = ["-",
    ["max", g, c, e],
    ["max", ["min", g, c], ["min", c, e], ["min", g, e]]];
  const ramp = (b) =>
    ["interpolate", ["linear"], margin, 0, APP.PALE[b], APP.MARGIN_FULL, APP.COL[b]];
  return [
    "case",
    ["all", [">=", g, c], [">=", g, e]], ramp("G"),
    [">=", c, e], ramp("CD"),
    ramp("ED"),
  ];
}

// Sequential ramp for the abstention layer.
function abstColorExpr(key) {
  return [
    "interpolate", ["linear"], ["get", key],
    15, "#f3f4f6", 28, "#cdd2da", 38, "#8b93a3", 50, "#454b59",
  ];
}

// Mobilization layer: density of mobilizable Left abstainers (abstainers × γ),
// in the Gauche hue (pale = few, full = a dense pocket). Thresholds differ for
// per-bureau counts vs the dezoomed commune sums.
function voterColorExpr(key, t1, t2, t3) {
  return [
    "interpolate", ["linear"], ["get", key],
    0, "#eef0f2", t1, APP.PALE.G, t2, APP.COL.G, t3, "#a83214",
  ];
}
