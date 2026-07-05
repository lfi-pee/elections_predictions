"use strict";
const APP = {
  COL: { G: "#E4572E", CD: "#4A90D9", ED: "#6A4C93", AB: "#9AA0A6" },
  // Accent de marque (client = LFI) : touche éditoriale sur titres de bande / accroche,
  // sans toucher la sémantique des blocs (la Gauche reste #E4572E sur la carte).
  ACCENT: "#cc2229",
  // Dark-theme faint end: each bloc hue mixed ~45% over the dark basemap, so the
  // map fades toward the background as the margin shrinks — a knife-edge bureau
  // sinks into the dark base, a landslide reads as the solid bloc colour.
  PALE: { G: "#743627", CD: "#2F5074", ED: "#3D3155" },
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
  DATAV: "10",
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
// as a dark→hot heat ramp in the Gauche hue (few = sinks into the dark base,
// dense = a bright orange pocket). Thresholds differ for per-bureau counts vs
// the dezoomed commune sums.
function voterColorExpr(key, t1, t2, t3) {
  return [
    "interpolate", ["linear"], ["get", key],
    0, "#20222b", t1, APP.PALE.G, t2, APP.COL.G, t3, "#ff7a4d",
  ];
}
