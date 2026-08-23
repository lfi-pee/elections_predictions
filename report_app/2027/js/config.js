"use strict";
// Site 2027 — prévision (curseurs nationaux + scénarios + jouabilité par circonscription).
// Le modèle donne, par bureau, l'ÉCART au national (déviation) ; le niveau national est
// posé par l'utilisateur (curseurs). On calcule en direct pred_b = national_b + dev_b.
const APP = {
  COL: { G: "#E4572E", CD: "#4A90D9", ED: "#6A4C93", AB: "#9AA0A6" },
  ACCENT: "#cc2229",
  PALE: { G: "#743627", CD: "#2F5074", ED: "#3D3155" },
  // Échelle de couleur des scores de jouabilité (1 = victoire facile → 5 = impossible).
  WIN: { 1: "#1a9850", 2: "#91cf60", 3: "#fee08b", 4: "#fc8d59", 5: "#4d4d4d" },
  WIN_LAB: { 1: "victoire facile", 2: "jouable", 3: "disputé", 4: "difficile", 5: "quasi impossible" },
  MARGIN_FULL: 12,
  NAME: { G: "Gauche", CD: "Centre+Droite", ED: "Extrême Droite", AB: "Abstention" },
  VOTE: ["G", "CD", "ED"],
  LYON: { center: [4.8357, 45.758], zoom: 12 },
  // mode carte : "win" (jouabilité circo) par défaut, "mobil" (gisement), "lead" (bloc en tête)
  state: { mode: "win" },
  // État national courant (parts de bloc %, abstention % inscrits) — piloté par les curseurs.
  nat: { G: 32.2, CD: 30.6, ED: 37.3, AB: 48 },
  scenario: "split2",
  data: {},
  map: null,
  bvByDept: new Map(),   // dept -> features brutes (dev), pour recalcul au curseur
  DATAV: "1",
};

const $ = (id) => document.getElementById(id);
const fmt = (n) => Math.round(n).toLocaleString("fr-FR");
const fmt1 = (n) => n.toLocaleString("fr-FR", { maximumFractionDigits: 1 });
const fmtM = (n) => (n / 1e6).toLocaleString("fr-FR", { maximumFractionDigits: 2 }) + " M";
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

async function loadJSON(path) {
  const sep = path.includes("?") ? "&" : "?";
  const r = await fetch(path + sep + "v=" + APP.DATAV);
  if (!r.ok) throw new Error("load " + path);
  return r.json();
}

// γ : part de gauche du votant marginal, lue sur la courbe « législatives » en fonction
// du niveau de gauche prédit du bureau. Interpolation linéaire (courbe = points [niveau,%]).
function gammaAt(pG) {
  const c = (APP.data.gamma && APP.data.gamma.Legislatives_T1) || null;
  if (!c || !c.length) return 40;
  if (pG <= c[0][0]) return c[0][1];
  for (let i = 1; i < c.length; i++) {
    if (pG <= c[i][0]) {
      const [x0, y0] = c[i - 1], [x1, y1] = c[i];
      return y0 + ((y1 - y0) * (pG - x0)) / (x1 - x0);
    }
  }
  return c[c.length - 1][1];
}

// Couleur « bloc en tête » : teinte = bloc gagnant, saturation = netteté de la marge.
function leadColorExpr(keys) {
  const g = ["get", keys.G], c = ["get", keys.CD], e = ["get", keys.ED];
  const margin = ["-", ["max", g, c, e], ["max", ["min", g, c], ["min", c, e], ["min", g, e]]];
  const ramp = (b) => ["interpolate", ["linear"], margin, 0, APP.PALE[b], APP.MARGIN_FULL, APP.COL[b]];
  return ["case", ["all", [">=", g, c], [">=", g, e]], ramp("G"), [">=", c, e], ramp("CD"), ramp("ED")];
}

// Couleur « mobilisation » : densité d'abstentionnistes de gauche mobilisables.
function voterColorExpr(key, t1, t2, t3) {
  return ["interpolate", ["linear"], ["get", key], 0, "#20222b", t1, APP.PALE.G, t2, APP.COL.G, t3, "#ff7a4d"];
}

// Couleur « jouabilité » : score 1→5 discret (attribué côté client, champ `sc`).
function winColorExpr() {
  return ["match", ["get", "sc"], 1, APP.WIN[1], 2, APP.WIN[2], 3, APP.WIN[3], 4, APP.WIN[4], APP.WIN[5]];
}
