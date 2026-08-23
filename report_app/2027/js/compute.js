"use strict";
// Cœur de calcul : à partir des déviations servies par circonscription + de l'état national
// (curseurs) et de la configuration de gauche (scénario), on calcule en direct, par circo :
//   pred_b = national_b + dev_b   → parts de bloc,
//   score de jouabilité 1→5 de la gauche,
//   bloc vainqueur du siège → projection en sièges (barre dynamique).
// Le serveur n'envoie que le motif spatial (dev) ; tout le reste réagit aux curseurs.

// Reports NON réglables (secondaires) : ils restent des constantes (miroir winnability_2027.py).
const BARR = { cd2ed: 0.25, ed2cd: 0.45, elimL2cd: 0.55, elimL2ed: 0.10 };
// Les trois coefficients RÉGLABLES au curseur vivent dans APP.coef (défauts = winnability_2027.py) :
//   cd2left = barrage centre-droit→gauche · ed2left = report RN→gauche · reunif = réunification
//   imparfaite d'une gauche divisée au 2nd tour. Lus à chaque évaluation → réagissent au curseur.

function leftCandidates(g, cfg, rad) {
  if (cfg === "union") return [g];
  if (cfg === "split2") return [g * rad, g * (1 - rad)];
  const o = g * (1 - rad);
  return [g * rad, o * 0.6, o * 0.4];
}

// Un bloc se qualifie s'il est dans le top 2 OU s'il atteint 12,5 % des inscrits.
function qual(share, second, thr) { return share >= second - 1e-9 || share >= thr; }

// Score de gauche réuni au 2nd tour + un pôle qualifié ? (pôles qualifiés pleins +
// pôles éliminés × APP.coef.reunif, le taux de réunification réglable au curseur).
function leftT2(left, second, thr) {
  const ql = left.filter((p) => qual(p, second, thr));
  if (!ql.length) return [0, false];
  const elim = left.filter((p) => !qual(p, second, thr)).reduce((a, b) => a + b, 0);
  return [ql.reduce((a, b) => a + b, 0) + APP.coef.reunif * elim, true];
}

// Reports du centre-droit au 2nd tour (union des droites → LR se reporte sur le RN).
// Hors union des droites, le barrage CD→gauche est le curseur APP.coef.cd2left.
function cdTransfer(ru) { return ru ? [0.20, 0.55] : [APP.coef.cd2left, BARR.cd2ed]; }

// Score 1→5 de la GAUCHE (miroir de src/winnability_2027.py).
function scoreCirco(g, cd, ed, ab, cfg, rad, ru) {
  const turnout = Math.max(0.05, 1 - ab / 100), thr = 12.5 / turnout;
  const left = leftCandidates(g, cfg, rad), lbest = Math.max(...left);
  const cands = left.concat([cd, ed]).sort((a, b) => b - a);
  const leader = cands[0], second = cands[1];
  const [lbase, qL] = leftT2(left, second, thr);
  const [cd2l, cd2e] = cdTransfer(ru);
  if (!qL) return { sc: 5, lbest, ql: false, mt2: null, opp: ed >= cd ? "ED" : "CD" };
  let l2, oppT2, opp;
  if (ed >= cd) { l2 = lbase + cd2l * cd; oppT2 = ed + cd2e * cd; opp = "ED"; }
  else { l2 = lbase + APP.coef.ed2left * ed; oppT2 = cd + BARR.ed2cd * ed; opp = "CD"; }
  const mt2 = l2 - oppT2, leadsFirst = lbest >= leader - 1e-9;
  let sc;
  if (leadsFirst && mt2 > 8) sc = 1; else if (mt2 > 0) sc = 2;
  else if (mt2 > -8) sc = 3; else sc = 4;
  return { sc, lbest, ql: true, mt2, opp };
}

// Bloc vainqueur du SIÈGE (G/CD/ED). La division de la gauche l'affaiblit au 2nd tour
// (réunification imparfaite) et peut l'éliminer dès le 1er (aucun pôle qualifié).
function seatWinner(g, cd, ed, ab, cfg, rad, ru) {
  const turnout = Math.max(0.05, 1 - ab / 100), thr = 12.5 / turnout;
  const left = leftCandidates(g, cfg, rad);
  const cands = left.concat([cd, ed]).sort((a, b) => b - a), second = cands[1];
  const [lbase, qL] = leftT2(left, second, thr);
  const qC = qual(cd, second, thr), qE = qual(ed, second, thr);
  const [cd2l, cd2e] = cdTransfer(ru);
  let sL = qL ? lbase : 0, sC = qC ? cd : 0, sE = qE ? ed : 0;
  if (!qL) { if (qC) sC += BARR.elimL2cd * g; if (qE) sE += BARR.elimL2ed * g; }
  if (!qC) { if (qL) sL += cd2l * cd; if (qE) sE += cd2e * cd; }
  if (!qE) { if (qL) sL += APP.coef.ed2left * ed; if (qC) sC += BARR.ed2cd * ed; }
  const arr = [["G", sL, qL], ["CD", sC, qC], ["ED", sE, qE]].filter((x) => x[2]);
  arr.sort((a, b) => b[1] - a[1]);
  const win = arr.length ? arr[0][0] : (g >= cd && g >= ed ? "G" : cd >= ed ? "CD" : "ED");
  // Pôle de gauche qui emporte le siège = le plus fort pôle qualifié.
  let pole = -1;
  if (win === "G") {
    const ql = left.map((p, i) => [p, i]).filter(([p]) => qual(p, second, thr));
    pole = (ql.length ? ql.reduce((a, b) => (b[0] > a[0] ? b : a)) : [0, left.indexOf(Math.max(...left))])[1];
  }
  return { win, pole };
}

// Couplage participation → parts (résultat clé de 2024) : sous l'abstention de référence,
// les électeurs qui reviennent aux urnes se répartissent selon la courbe γ — nettement plus
// à gauche que l'électorat assis (de ~24 % dans les bureaux les plus à droite à ~56 % dans
// les plus à gauche). Baisser l'abstention relève donc la part de gauche, partout.
function turnoutAdjust(g, cd, ed, ab, dAB) {
  const T = clamp(1 - ab / 100, 0.05, 0.98);
  const T0 = clamp(1 - (APP.AB_REF + dAB) / 100, 0.05, 0.98);
  const dT = T - T0;
  if (Math.abs(dT) < 1e-6) return [g, cd, ed];
  const gm = gammaAt(g) / 100, rest = Math.max(0, 1 - gm), den = cd + ed || 1;
  let Gv = Math.max(0, g / 100 * T0 + gm * dT);
  let CDv = Math.max(0, cd / 100 * T0 + rest * dT * cd / den);
  let EDv = Math.max(0, ed / 100 * T0 + rest * dT * ed / den);
  const tot = Gv + CDv + EDv || 1;
  return [100 * Gv / tot, 100 * CDv / tot, 100 * EDv / tot];
}

// pred + score + vainqueur d'une circo depuis les déviations portées par sa feature.
function circoEval(pr) {
  const n = APP.nat, s = APP.scnObj;
  const g0 = clamp(n.G + pr.dG, 0, 100), cd0 = clamp(n.CD + pr.dCD, 0, 100),
    ed0 = clamp(n.ED + pr.dED, 0, 100), ab = clamp(n.AB + pr.dAB, 0, 100);
  const [g, cd, ed] = turnoutAdjust(g0, cd0, ed0, ab, pr.dAB);
  const ru = s.right_union;
  // Part radicale (LFI) : base = curseur (APP.radOverride) sinon valeur du scénario (ancrage
  // sondages), puis modulée localement — le pôle radical pèse davantage là où la gauche est
  // forte (bastions urbains/populaires), moins là où elle est faible ; sinon un partage
  // national uniforme donnerait 0 siège au pôle radical partout.
  const radBase = APP.radOverride != null ? APP.radOverride : s.radical_share;
  const rad = s.left_config === "union" ? 1.0 : clamp(radBase + 0.006 * pr.dG, 0.12, 0.68);
  const r = scoreCirco(g, cd, ed, ab, s.left_config, rad, ru);
  const sw = seatWinner(g, cd, ed, ab, s.left_config, rad, ru);
  return { g, cd, ed, ab, win: sw.win, pole: sw.pole, ...r };
}

// La géométrie (15 Mo) est chargée UNE fois comme données de la source. Au curseur, on ne
// met à jour qu'un état léger par entité (score/vainqueur) via setFeatureState — aucune
// re-sérialisation ni re-tuilage des polygones : c'est ce qui rend le glissement fluide.
function updateCircoStates() {
  const map = APP.map, src = APP.data.circoGeo;
  if (!map || !src || !map.getSource("circo")) return;
  for (const f of src.features) {
    const r = circoEval(f.properties);
    map.setFeatureState({ source: "circo", id: f.properties.id }, { sc: r.sc, win: r.win });
  }
}

// Itère sur les 577 circos (tableaux de circo.json), y compris outre-mer/étranger sans
// contour — pour que les projections en sièges et la répartition somment sur l'Assemblée
// entière, pas seulement sur les circos cartographiées.
function eachCirco(fn) {
  const a = APP.data.circoArr;
  if (!a) return;
  for (let i = 0; i < a.id.length; i++)
    fn(circoEval({ dG: a.dG[i], dCD: a.dCD[i], dED: a.dED[i], dAB: a.dAB[i] }), i);
}

// Projection en sièges (tally des vainqueurs) au scénario/curseur courant (577 circos).
// `poles` = répartition des sièges de gauche entre pôles (index → nb) pour le détail parti.
function seatTally() {
  const t = { G: 0, CD: 0, ED: 0, poles: {} };
  eachCirco((r) => { t[r.win]++; if (r.win === "G") t.poles[r.pole] = (t.poles[r.pole] || 0) + 1; });
  return t;
}

// Répartition des circos par score de jouabilité (1→5) au scénario courant (577 circos).
function scoreTally() {
  const t = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
  eachCirco((r) => t[r.sc]++);
  return t;
}

// Recalcule carte + barres après changement de curseur / scénario. Coalescé sur une frame
// d'animation : dix « input » de curseur dans la même frame → un seul recalcul.
let _raf = 0;
function recomputeAll() {
  if (_raf) return;
  _raf = requestAnimationFrame(() => {
    _raf = 0;
    updateCircoStates();
    updateSeatBar();
    updateNatBar();
    updateWinSummary();
  });
}
