"use strict";
// Cœur de calcul : à partir des déviations servies par circonscription + de l'état national
// (curseurs) et de la configuration de gauche (scénario), on calcule en direct, par circo :
//   pred_b = national_b + dev_b   → parts de bloc,
//   score de jouabilité 1→5 de la gauche,
//   bloc vainqueur du siège → projection en sièges (barre dynamique).
// Le serveur n'envoie que le motif spatial (dev) ; tout le reste réagit aux curseurs.

// Reports NON réglables (secondaires) : ils restent des constantes (miroir winnability_2027.py).
const BARR = { ed2cd: 0.45, elimL2cd: 0.55, elimL2ed: 0.10 };
// Coefficients RÉGLABLES au curseur dans APP.coef (défauts = winnability_2027.py) : desist
// (désistement front républicain), cdLR (part LR du bloc CD → compose les reports du CD via
// APP.CDT), ed2left (report RN→gauche), reunif (réunification gauche divisée). Lus à chaque
// évaluation → réagissent au curseur.

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

// Reports du centre-droit, composés Ensemble (barrage) + LR (ambivalent, part APP.coef.cdLR).
// Sous droites unies, seul LR bascule vers le RN ; Ensemble fait toujours barrage. Miroir Python.
function cdTransfer(ru) {
  const lr = APP.coef.cdLR, ens = 1 - lr, T = APP.CDT;
  const [lrl, lre] = ru ? [T.lrLru, T.lrEru] : [T.lrL, T.lrE];
  return [ens * T.ensL + lr * lrl, ens * T.ensE + lr * lre];
}

// Score 1→5 de la GAUCHE (miroir de src/winnability_2027.py).
function scoreCirco(g, cd, ed, ab, cfg, rad, ru) {
  const turnout = Math.max(0.05, 1 - ab / 100), thr = 12.5 / turnout;
  const left = leftCandidates(g, cfg, rad), lbest = Math.max(...left);
  const cands = left.concat([cd, ed]).sort((a, b) => b - a);
  const leader = cands[0], second = cands[1];
  const [lbase, qL] = leftT2(left, second, thr);
  const [cd2l, cd2e] = cdTransfer(ru);
  const qC = qual(cd, second, thr), qE = qual(ed, second, thr);
  if (!qL) return { sc: 5, lbest, ql: false, mt2: null, opp: ed >= cd ? "ED" : "CD" };
  let l2, oppT2, opp;
  if (ed >= cd) {
    opp = "ED";
    if (qC && qE && !ru) {
      // Triangulaire face au RN : le centre-droit se DÉSISTE pour la gauche (front républicain).
      l2 = lbase + APP.coef.desist * cd; oppT2 = ed + APP.DESIST_ED * cd;
    } else {
      // Duel (CD éliminé) ou droites unies (pas de désistement) : barrage classique.
      l2 = lbase + cd2l * cd; oppT2 = ed + cd2e * cd;
    }
  } else { l2 = lbase + APP.coef.ed2left * ed; oppT2 = cd + BARR.ed2cd * ed; opp = "CD"; }
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
  // Désistement (front républicain) en triangulaire face au RN : le pôle anti-RN le plus faible
  // se retire au profit du plus fort (mécanisme calibré sur 2024). Droites unies : le CD refuse.
  let qLd = qL, qCd = qC;
  if (qL && qC && qE) {
    if (sL >= sC) {
      if (!ru) { sL += APP.coef.desist * sC; sE += APP.DESIST_ED * sC; qCd = false; }
    } else { sC += APP.coef.desist * sL; sE += APP.DESIST_ED * sL; qLd = false; }
  }
  const arr = [["G", sL, qLd], ["CD", sC, qCd], ["ED", sE, qE]].filter((x) => x[2]);
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

// Parts nationales « effectives » = parts de base (posées à l'abstention de référence) APRÈS
// couplage participation γ, au niveau national. Les curseurs de bloc AFFICHENT l'effectif ;
// APP.nat garde la base (à la référence), que le calcul par circo consomme telle quelle.
function natEffective(g, cd, ed, ab) { return turnoutAdjust(g, cd, ed, ab, 0); }
// Inverse : parts de base telles que natEffective(base, ab) = cible (point fixe, ~identité à la
// référence). Sert quand l'utilisateur bouge un curseur de bloc à abstention ≠ référence.
function natBaseFromEffective(tg, tc, te, ab) {
  let g = tg, cd = tc, ed = te;
  for (let it = 0; it < 6; it++) {
    const [fg, fc, fe] = natEffective(g, cd, ed, ab);
    g = Math.max(0, g + tg - fg); cd = Math.max(0, cd + tc - fc); ed = Math.max(0, ed + te - fe);
    const s = g + cd + ed || 1; g = 100 * g / s; cd = 100 * cd / s; ed = 100 * ed / s;
  }
  return [g, cd, ed];
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

// ── Incertitude : Monte-Carlo des fourchettes conformes à travers le modèle de sièges ──
// Le modèle donne une erreur LOCALE (par bureau) calibrée par validation croisée ; la
// demi-largeur 90 % par bloc (summary.cv_halfwidth_90) donne un σ ≈ hw/1,645. On tire un bruit
// gaussien indépendant par circo et par bloc, on rejoue le 2nd tour, et on lit la distribution
// des sièges. Indépendant par circo car l'erreur est locale (le niveau national est POSÉ, pas
// prédit) : les erreurs se compensent en masse et seules les circos serrées basculent — donc
// la fourchette dit « combien de sièges dépendent vraiment de circos indécises ».
let _gaussSpare = null;
function gauss() {
  if (_gaussSpare != null) { const v = _gaussSpare; _gaussSpare = null; return v; }
  const u = Math.random() || 1e-9, v = Math.random();
  const r = Math.sqrt(-2 * Math.log(u));
  _gaussSpare = r * Math.sin(2 * Math.PI * v);
  return r * Math.cos(2 * Math.PI * v);
}

// Outcome (score + vainqueur) d'une circo avec bruit ajouté aux parts prédites — même pipeline
// que circoEval mais allégé (pas de champs d'affichage), pour le tirage Monte-Carlo.
function circoOutcome(dG, dCD, dED, dAB, eG, eCD, eED) {
  const n = APP.nat, s = APP.scnObj;
  const g0 = clamp(n.G + dG + eG, 0, 100), cd0 = clamp(n.CD + dCD + eCD, 0, 100),
    ed0 = clamp(n.ED + dED + eED, 0, 100), ab = clamp(n.AB + dAB, 0, 100);
  const [g, cd, ed] = turnoutAdjust(g0, cd0, ed0, ab, dAB);
  const ru = s.right_union;
  const radBase = APP.radOverride != null ? APP.radOverride : s.radical_share;
  const rad = s.left_config === "union" ? 1.0 : clamp(radBase + 0.006 * dG, 0.12, 0.68);
  const sc = scoreCirco(g, cd, ed, ab, s.left_config, rad, ru).sc;
  const win = seatWinner(g, cd, ed, ab, s.left_config, rad, ru).win;
  return { sc, win };
}

function _pctl(sorted, q) {
  const i = clamp(Math.round((sorted.length - 1) * q), 0, sorted.length - 1);
  return sorted[i];
}

// Distribution des sièges (et des circos jouables) sur `nDraws` tirages. Renvoie médiane et
// bornes 5 %/95 % par bloc + jouables.
function seatDistribution(nDraws) {
  const a = APP.data.circoArr; if (!a) return null;
  // Demi-largeur CIRCO (pas par bureau) : erreurs fortement corrélées dans une circo — voir
  // report_data_2027.circo_halfwidth. Repli sur la conforme par bureau si absente.
  const cv = APP.data.summary.circo_halfwidth_90 || APP.data.summary.cv_halfwidth_90 || {};
  const Z = 1.645; // demi-largeur 90 % → σ
  const sG = (cv.G || 0) / Z, sC = (cv.CD || 0) / Z, sE = (cv.ED || 0) / Z;
  const n = a.id.length;
  const G = new Array(nDraws), C = new Array(nDraws), E = new Array(nDraws), P = new Array(nDraws);
  for (let d = 0; d < nDraws; d++) {
    let g = 0, c = 0, e = 0, play = 0;
    for (let i = 0; i < n; i++) {
      const r = circoOutcome(a.dG[i], a.dCD[i], a.dED[i], a.dAB[i],
        sG * gauss(), sC * gauss(), sE * gauss());
      if (r.win === "G") g++; else if (r.win === "CD") c++; else e++;
      if (r.sc <= 3) play++;
    }
    G[d] = g; C[d] = c; E[d] = e; P[d] = play;
  }
  const band = (arr) => { const s = arr.slice().sort((x, y) => x - y);
    return { med: _pctl(s, 0.5), lo: _pctl(s, 0.05), hi: _pctl(s, 0.95) }; };
  return { G: band(G), CD: band(C), ED: band(E), play: band(P) };
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
    scheduleUncertainty();
  });
}

// La fourchette Monte-Carlo (577 circos × ~240 tirages) est trop lourde pour chaque frame de
// glissement : on la calcule une fois le curseur STABILISÉ (débounce), sans bloquer le glisser.
const MC_DRAWS = 240;
let _uncT = 0;
function scheduleUncertainty() {
  if (_uncT) clearTimeout(_uncT);
  _uncT = setTimeout(() => { _uncT = 0; if (typeof updateUncertainty === "function") updateUncertainty(); }, 180);
}
