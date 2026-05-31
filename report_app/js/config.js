"use strict";
const APP = {
  COL: { G: "#E4572E", CD: "#4A90D9", ED: "#6A4C93", AB: "#9AA0A6" },
  // Accent de marque (client = LFI) : touche éditoriale sur titres de bande / accroche,
  // sans toucher la sémantique des blocs (la Gauche reste #E4572E sur la carte).
  ACCENT: "#cc2229",
  // pale = each bloc hue mixed ~78% toward paper; the map fades to it as the
  // margin shrinks, so a knife-edge bureau reads pale and a landslide reads solid.
  PALE: { G: "#F5D5C9", CD: "#CFE1F2", ED: "#DBD2E8" },
  MARGIN_FULL: 12,
  NAME: { G: "Gauche", CD: "Centre+Droite", ED: "Extrême Droite", AB: "Abstention" },
  VOTE: ["G", "CD", "ED"],
  LYON: { center: [4.8357, 45.758], zoom: 12 },
  state: { mode: "mobil" },
  data: {},
  map: null,
  bvByDept: new Map(),
  // Bump on any change to served data files (national/communes/bv/detail/…) so the
  // browser refetches instead of serving a stale cache. Appended to every loadJSON.
  DATAV: "9",
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

// Lead-block color expression. Hue = winning bloc; saturation = decisiveness: we
// interpolate from the pale tone (margin 0) to the full tone (margin ≥ MARGIN_FULL).
// Margin = top − second, computed live so a knife-edge bureau reads pale.
function leadColorExpr(keys) {
  const g = ["get", keys.G], c = ["get", keys.CD], e = ["get", keys.ED];
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

// Mobilization layer: density of mobilizable Left abstainers (abstainers × γ),
// in the Gauche hue (pale = few, full = a dense pocket). Thresholds differ for
// per-bureau counts vs the dezoomed commune sums.
function voterColorExpr(key, t1, t2, t3) {
  return [
    "interpolate", ["linear"], ["get", key],
    0, "#eef0f2", t1, APP.PALE.G, t2, APP.COL.G, t3, "#a83214",
  ];
}
