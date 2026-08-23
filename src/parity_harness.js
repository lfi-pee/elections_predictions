// Harnais de PARITÉ Python ↔ JavaScript pour le modèle de sièges 2027.
//
// Le cœur de calcul existe en double : winnability_2027.py (chiffres servis, backtests) et
// report_app/2027/js/compute.js (recalcul en direct au curseur). S'ils divergent, le site ment.
// Ce harnais exécute le VRAI compute.js (chargé tel quel, avec config.js pour les constantes) et
// renvoie ses sorties pour une grille de cas, que test_parity_2027.py compare à Python.
//
//   node src/parity_harness.js <js_dir> <vectors.json>   # écrit les résultats JSON sur stdout

const fs = require("fs");
const vm = require("vm");
const path = require("path");

const jsDir = process.argv[2];
const vectorsFile = process.argv[3];

// config.js + compute.js sont des scripts navigateur (globales, pas de modules). On les concatène
// et on expose les fonctions du modèle via globalThis pour les récupérer après exécution.
const src =
  fs.readFileSync(path.join(jsDir, "config.js"), "utf8") + "\n" +
  fs.readFileSync(path.join(jsDir, "compute.js"), "utf8") + "\n" +
  "globalThis.__API = { leftCandidates, qual, leftT2, cdTransfer, scoreCirco, seatWinner," +
  " turnoutAdjust, gammaAt, APP };";

const sandbox = {
  console, Math, JSON, Array, Object, Number, isNaN, parseFloat, parseInt,
  document: { getElementById: () => null },
  window: {},
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: "parity_bundle.js" });

const API = sandbox.__API;
const vectors = JSON.parse(fs.readFileSync(vectorsFile, "utf8"));
API.APP.data = API.APP.data || {};
API.APP.data.gamma = vectors.gamma; // pour turnoutAdjust (γ)

const out = vectors.cases.map((c) => {
  const sw = API.seatWinner(c.g, c.cd, c.ed, c.ab, c.cfg, c.rad, c.ru);
  const sc = API.scoreCirco(c.g, c.cd, c.ed, c.ab, c.cfg, c.rad, c.ru);
  const ta = API.turnoutAdjust(c.g, c.cd, c.ed, c.ab, c.dAB || 0);
  return {
    win: sw.win, pole: sw.pole,
    score: sc.sc, qualifies: sc.ql, margin_t2: sc.mt2, opp: sc.opp,
    ta_g: ta[0], ta_cd: ta[1], ta_ed: ta[2],
  };
});

// Constantes lues dans le JS exécuté (pour vérifier l'égalité des défauts côté Python).
const consts = {
  desist: API.APP.coef.desist, cdLR: API.APP.coef.cdLR, ed2left: API.APP.coef.ed2left,
  reunif: API.APP.coef.reunif, DESIST_ED: API.APP.DESIST_ED, RAD_GAIN: API.APP.RAD_GAIN,
  AB_REF: API.APP.AB_REF, CDT: API.APP.CDT,
};

process.stdout.write(JSON.stringify({ out, consts }));
