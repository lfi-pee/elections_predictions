"use strict";
// Couverture de la nomenclature de blocs, par circonscription — miroir de src/coverage_2027.py.
//
// Le modèle ne connaît que trois blocs (Gauche, Centre+Droite, Extrême Droite). Les nuances
// régionalistes / autonomistes du ministère n'y entrent pas : là où elles dominent (Guyane,
// Martinique, Nouvelle-Calédonie, Polynésie, Corse), une grande part du vote sort du modèle et
// la prédiction n'est pas publiable. On MESURE ce trou depuis les parts réelles 2024 servies
// (r24*) et on marque les circos concernées : elles s'affichent en gris hachuré sur la carte et
// portent un avertissement dans le panneau, au lieu d'un score qui ne tient pas.
//
// La parité de ce calcul avec Python est vérifiée par src/test_parity_2027.py (mêmes seuil,
// mêmes couvertures, mêmes étiquettes sur les 577 circos servies).

const COV_OK = "mesuree", COV_LOW = "faible", COV_UNKNOWN = "inconnue";

// Seuil = 100 − la plus large demi-largeur à 90 % servie : en deçà, la part de vote non
// rattachée à un bloc dépasse à elle seule l'incertitude que le modèle s'accorde.
function covThreshold(summary) {
  const hw = (summary && summary.circo_halfwidth_90) || {};
  const vals = Object.keys(hw).map((k) => Number(hw[k]));
  if (!vals.length) return 0;
  return Math.round((100 - Math.max.apply(null, vals)) * 1000) / 1000;
}

// La table d'attribution est-elle passée dans la CHAÎNE, ou seulement mesurée ? Estampille
// posée par report_data_2027 à la reconstruction. Tant qu'elle manque, les déviations 2027
// servies descendent de l'ancienne nomenclature : la donnée 2024 est réparée, la PRÉVISION non.
function covApplied(summary) { return !!(summary && summary.attribution_applied); }

// Couverture (%) + origine par circo, lue dans coverage.json — la mesure officielle produite
// par src/attribution_2027.py (résultats du ministère + table d'attribution des voix
// régionalistes). Connue pour les 577 circos ; `null` si une circo en est absente. On prend la
// couverture d'AVANT attribution tant que la chaîne n'a pas été reconstruite : lever le marquage
// sur la seule foi de la donnée 2024 publierait une prévision périmée.
function covCompute(a, coverage, summary) {
  const after = covApplied(summary);
  const cov = (coverage && coverage[after ? "cov_apres" : "cov_avant"]) || {};
  const r3 = (x) => Math.round(x * 1000) / 1000;
  const val = a.id.map((id) => (cov[id] == null ? null : r3(Number(cov[id]))));
  const lab = covApplied(summary) ? "mesure" : "mesure (avant reconstruction)";
  const src = val.map((v) => (v == null ? null : lab));
  return { val, src };
}

function covFlag(cov, thr) {
  if (cov == null) return COV_UNKNOWN;
  return cov < thr ? COV_LOW : COV_OK;
}

// Prépare l'index de fiabilité une fois les données chargées (appelé par boot()).
function initCoverage() {
  const a = APP.data.circoArr, s = APP.data.summary;
  const thr = covThreshold(s), { val, src } = covCompute(a, APP.data.coverage, s);
  const flag = val.map((v) => covFlag(v, thr));
  const byId = new Map();
  for (let i = 0; i < a.id.length; i++) byId.set(a.id[i], { cov: val[i], src: src[i], flag: flag[i] });
  APP.cov = { thr, val, src, flag, byId, nLow: flag.filter((f) => f !== COV_OK).length };
}

// Fiabilité d'une circo par son identifiant (défaut : mesurée, si l'index n'est pas prêt).
function covOf(id) {
  return (APP.cov && APP.cov.byId.get(id)) || { cov: null, src: null, flag: COV_OK };
}
function covIsPublishable(id) { return covOf(id).flag === COV_OK; }

// Identifiants des circos non publiables — filtre de la couche de liseré tireté sur la carte.
function covUnpublishableIds() {
  if (!APP.cov) return [];
  const a = APP.data.circoArr;
  return a.id.filter((_, i) => APP.cov.flag[i] !== COV_OK);
}

// Phrase d'avertissement affichée dans le panneau / l'infobulle.
function covWarning(id) {
  const c = covOf(id);
  if (c.flag === COV_OK) return "";
  if (c.flag === COV_UNKNOWN)
    return `Aucun résultat 2024 exploitable sur ce territoire : la prévision n'y est pas
      vérifiable. <b>Chiffre non publiable.</b>`;
  return `Seuls <b>${fmt1(c.cov)} %</b> du vote exprimé de 2024 de cette circonscription sont
    rattachés à l'un des trois blocs du modèle. Le reste va à des candidatures
    <b>régionalistes, autonomistes ou diverses</b> dont l'alignement n'est pas établi — la
    mouvance corse, par exemple, siège au groupe LIOT, ni à gauche ni au centre-droit. Le modèle
    ne voit donc qu'une partie de l'électorat local. <b>Chiffre non publiable.</b>`;
}
