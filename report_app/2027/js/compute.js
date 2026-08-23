"use strict";
// Cœur de calcul : à partir des déviations servies par circonscription + de l'état national
// (curseurs) et de la configuration de gauche (scénario), on calcule en direct, par circo :
//   pred_b = national_b + dev_b   → parts de bloc,
//   score de jouabilité 1→5 de la gauche,
//   bloc vainqueur du siège → projection en sièges (barre dynamique).
// Le serveur n'envoie que le motif spatial (dev) ; tout le reste réagit aux curseurs.

const BARR = { cd2left: 0.45, cd2ed: 0.25, ed2left: 0.15, ed2cd: 0.45,
  elimL2cd: 0.55, elimL2ed: 0.10 };
// Réunification imparfaite au 2nd tour d'une gauche divisée (voir winnability_2027.py).
const REUNIF = 0.72;

function leftCandidates(g, cfg, rad) {
  if (cfg === "union") return [g];
  if (cfg === "split2") return [g * rad, g * (1 - rad)];
  const o = g * (1 - rad);
  return [g * rad, o * 0.6, o * 0.4];
}

// Un bloc se qualifie s'il est dans le top 2 OU s'il atteint 12,5 % des inscrits.
function qual(share, second, thr) { return share >= second - 1e-9 || share >= thr; }

// Score de gauche réuni au 2nd tour + un pôle qualifié ? (pôles qualifiés pleins +
// pôles éliminés × REUNIF).
function leftT2(left, second, thr) {
  const ql = left.filter((p) => qual(p, second, thr));
  if (!ql.length) return [0, false];
  const elim = left.filter((p) => !qual(p, second, thr)).reduce((a, b) => a + b, 0);
  return [ql.reduce((a, b) => a + b, 0) + REUNIF * elim, true];
}

// Score 1→5 de la GAUCHE (miroir de src/winnability_2027.py).
function scoreCirco(g, cd, ed, ab, cfg, rad) {
  const turnout = Math.max(0.05, 1 - ab / 100), thr = 12.5 / turnout;
  const left = leftCandidates(g, cfg, rad), lbest = Math.max(...left);
  const cands = left.concat([cd, ed]).sort((a, b) => b - a);
  const leader = cands[0], second = cands[1];
  const [lbase, qL] = leftT2(left, second, thr);
  if (!qL) return { sc: 5, lbest, ql: false, mt2: null, opp: ed >= cd ? "ED" : "CD" };
  let l2, oppT2, opp;
  if (ed >= cd) { l2 = lbase + BARR.cd2left * cd; oppT2 = ed + BARR.cd2ed * cd; opp = "ED"; }
  else { l2 = lbase + BARR.ed2left * ed; oppT2 = cd + BARR.ed2cd * ed; opp = "CD"; }
  const mt2 = l2 - oppT2, leadsFirst = lbest >= leader - 1e-9;
  let sc;
  if (leadsFirst && mt2 > 8) sc = 1; else if (mt2 > 0) sc = 2;
  else if (mt2 > -8) sc = 3; else sc = 4;
  return { sc, lbest, ql: true, mt2, opp };
}

// Bloc vainqueur du SIÈGE (G/CD/ED). La division de la gauche l'affaiblit au 2nd tour
// (réunification imparfaite) et peut l'éliminer dès le 1er (aucun pôle qualifié).
function seatWinner(g, cd, ed, ab, cfg, rad) {
  const turnout = Math.max(0.05, 1 - ab / 100), thr = 12.5 / turnout;
  const left = leftCandidates(g, cfg, rad);
  const cands = left.concat([cd, ed]).sort((a, b) => b - a), second = cands[1];
  const [lbase, qL] = leftT2(left, second, thr);
  const qC = qual(cd, second, thr), qE = qual(ed, second, thr);
  let sL = qL ? lbase : 0, sC = qC ? cd : 0, sE = qE ? ed : 0;
  if (!qL) { if (qC) sC += BARR.elimL2cd * g; if (qE) sE += BARR.elimL2ed * g; }
  if (!qC) { if (qL) sL += BARR.cd2left * cd; if (qE) sE += BARR.cd2ed * cd; }
  if (!qE) { if (qL) sL += BARR.ed2left * ed; if (qC) sC += BARR.ed2cd * ed; }
  const arr = [["G", sL, qL], ["CD", sC, qC], ["ED", sE, qE]].filter((x) => x[2]);
  arr.sort((a, b) => b[1] - a[1]);
  return arr.length ? arr[0][0] : (g >= cd && g >= ed ? "G" : cd >= ed ? "CD" : "ED");
}

// pred + score + vainqueur d'une circo depuis les déviations portées par sa feature.
function circoEval(pr) {
  const n = APP.nat, s = APP.scnObj;
  const g = clamp(n.G + pr.dG, 0, 100), cd = clamp(n.CD + pr.dCD, 0, 100),
    ed = clamp(n.ED + pr.dED, 0, 100), ab = clamp(n.AB + pr.dAB, 0, 100);
  const r = scoreCirco(g, cd, ed, ab, s.left_config, s.radical_share);
  const win = seatWinner(g, cd, ed, ab, s.left_config, s.radical_share);
  return { g, cd, ed, ab, win, ...r };
}

// Choroplèthe : on ré-étiquette chaque polygone servi (dev bruts) au scénario courant.
function circoFC() {
  const src = APP.data.circoGeo;
  if (!src) return { type: "FeatureCollection", features: [] };
  return {
    type: "FeatureCollection",
    features: src.features.map((f) => {
      const pr = f.properties, r = circoEval(pr);
      return {
        type: "Feature", geometry: f.geometry,
        properties: {
          id: pr.id, nm: pr.nm, dept: pr.dept, ins: pr.ins, nbv: pr.nbv,
          dG: pr.dG, dCD: pr.dCD, dED: pr.dED, dAB: pr.dAB,
          pG: r.g, pCD: r.cd, pED: r.ed, pAB: r.ab,
          sc: r.sc, win: r.win, lbest: +r.lbest.toFixed(1),
          mt2: r.mt2 === null ? -99 : +r.mt2.toFixed(1), opp: r.opp,
        },
      };
    }),
  };
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
function seatTally() {
  const t = { G: 0, CD: 0, ED: 0 };
  eachCirco((r) => t[r.win]++);
  return t;
}

// Répartition des circos par score de jouabilité (1→5) au scénario courant (577 circos).
function scoreTally() {
  const t = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
  eachCirco((r) => t[r.sc]++);
  return t;
}

// Recalcule carte + barres après changement de curseur / scénario.
function recomputeAll() {
  if (APP.map && APP.map.getSource("circo")) APP.map.getSource("circo").setData(circoFC());
  updateSeatBar();
  updateNatBar();
  updateWinSummary();
}
