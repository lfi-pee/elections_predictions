"use strict";
// Site 2027 — prévision (curseurs nationaux + scénarios + jouabilité par circonscription).
// Le modèle donne, par bureau, l'ÉCART au national (déviation) ; le niveau national est
// posé par l'utilisateur (curseurs). On calcule en direct pred_b = national_b + dev_b.
const APP = {
  COL: { G: "#E4572E", CD: "#4A90D9", ED: "#6A4C93", AB: "#9AA0A6" },
  ACCENT: "#cc2229",
  PALE: { G: "#743627", CD: "#2F5074", ED: "#3D3155" },
  // Échelle de couleur des scores de jouabilité (1 = victoire facile → 5 = impossible, rouge plein).
  WIN: { 1: "#1a9850", 2: "#91cf60", 3: "#fee08b", 4: "#fc8d59", 5: "#d7191c" },
  WIN_LAB: { 1: "victoire facile", 2: "jouable", 3: "disputé", 4: "difficile", 5: "quasi impossible" },
  MARGIN_FULL: 12,
  NAME: { G: "Gauche", CD: "Centre+Droite", ED: "Extrême Droite", AB: "Abstention" },
  VOTE: ["G", "CD", "ED"],
  LYON: { center: [4.8357, 45.758], zoom: 12 },
  // mode carte : "win" (jouabilité circo) par défaut, "seat" (vainqueur du siège) ;
  // seatDetail : barre des sièges détaillée par pôle de gauche (radicale/soc.-dém./éco).
  state: { mode: "win", seatDetail: false },
  // État national courant (parts de bloc %, abstention % inscrits) — piloté par les curseurs.
  nat: { G: 32.2, CD: 30.6, ED: 37.3, AB: 48 },
  scenario: "split2",
  // Abstention de référence à laquelle les curseurs de parts (G/CD/ED) sont calés : en
  // deçà, les revenants aux urnes se répartissent selon la courbe γ (les abstentionnistes
  // mobilisables penchent à gauche — le résultat clé de 2024), ce qui relève la gauche.
  AB_REF: 48,
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

// Pôles de gauche selon la configuration du scénario (pour le détail par parti des sièges).
// Ce sont les seules sous-composantes que le modèle résout par circonscription.
function poleMeta(cfg) {
  if (cfg === "split2") return [
    { lab: "Gauche radicale (LFI)", col: "#a01722" },
    { lab: "Soc.-dém. (PS/PP/EELV/PCF)", col: "#ef7a5a" }];
  if (cfg === "split3") return [
    { lab: "Radicale (LFI)", col: "#a01722" },
    { lab: "PS / Place publique", col: "#ef7a5a" },
    { lab: "Écologistes / PCF", col: "#e39a3a" }];
  return [{ lab: "Gauche unie (NFP)", col: APP.COL.G }];
}

// Couleur « jouabilité » : score 1→5 lu dans l'ÉTAT d'entité (feature-state), mis à jour
// au curseur sans retoucher la géométrie. Défaut (état non posé) = gris neutre.
function winColorExpr() {
  return ["match", ["feature-state", "sc"],
    1, APP.WIN[1], 2, APP.WIN[2], 3, APP.WIN[3], 4, APP.WIN[4], 5, APP.WIN[5], "#c8ccd2"];
}
// Couleur « vainqueur du siège » depuis l'état d'entité.
function seatColorExpr() {
  return ["match", ["feature-state", "win"],
    "G", APP.COL.G, "CD", APP.COL.CD, "ED", APP.COL.ED, "#c8ccd2"];
}
