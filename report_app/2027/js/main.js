"use strict";

async function boot() {
  const [summary, circoArr, circoGeo, insets, gamma] = await Promise.all([
    loadJSON("data/summary.json"),
    loadJSON("data/circo.json"),
    loadJSON("data/circo_display.geojson"),
    loadJSON("data/circo_insets.json"),
    loadJSON("data/gamma_curve.json"),
  ]);
  APP.data = { summary, circoArr, circoGeo, insets, gamma };
  APP.scenario = summary.default_scenario;
  APP.scnObj = summary.scenarios.find((s) => s.key === APP.scenario);
  APP.nat = { ...APP.scnObj.means };

  await initMap();
  initControls();
  initSearch();
  initPanel();
  applyColor();
  updateLegend();
  recomputeAll();
  renderIntro();
  renderInputs();
}

// Documente les entrées du modèle POUR CETTE élection (transparence, en bas de page).
function renderInputs() {
  const s = APP.data.summary, cv = s.cv_halfwidth_90 || {}, p = s.proof_2024 || {};
  const li = (t) => `<li>${t}</li>`;
  $("inputs-body").innerHTML = `<ul>
    ${li(`<b>Cible</b> : Législatives 2027, 1<sup>er</sup> tour, <b>${s.n_circo}</b> circonscriptions,
        agrégées depuis <b>${fmt(s.n_bv)}</b> bureaux de vote (${fmt(s.total_inscrits)} inscrits).`)}
    ${li(`<b>Entraînement</b> : les législatives <b>2002→2024</b> (2024 désormais incluse), en
        <i>déviations</i> par bureau à la moyenne nationale de chaque scrutin.`)}
    ${li(`<b>Prédicteurs locaux</b> : surtout l'<b>héritage de vote</b> du bureau — sa déviation
        aux deux derniers législatifs (<b>2024</b> puis <b>2022</b>) — plus <b>52 indicateurs
        INSEE</b> (dernier millésime). Modèle : Ridge (régression linéaire régularisée) + PCA,
        un réglage par bloc.`)}
    ${li(`<b>Ancre nationale</b> : posée par vous (curseurs). Présélections = intentions de vote
        1<sup>er</sup> tour 2025 (Ifop, OpinionWay, Elabe, Cluster17, Harris), renormalisées sur
        3 blocs. Abstention = axe séparé ; la baisser réaffecte les revenants selon la courbe γ
        (résultat 2024).`)}
    ${li(`<b>Fourchette d'incertitude</b> (prédiction conforme, erreur locale) — demi-largeur à
        90 %, par bloc : G ±${cv.G}, C+D ±${cv.CD}, ED ±${cv.ED}, Abst. ±${cv.AB} pts.`)}
    ${li(`<b>Second tour</b> (jouabilité &amp; sièges) : qualification à <b>12,5 % des inscrits</b> ;
        barrage centre-droit→gauche 45 % (contre le RN) ; <b>réunification imparfaite</b> d'une
        gauche divisée (72 %) ; « union des droites » = report LR→RN.`)}
    ${li(`<b>Contours</b> : circonscriptions reconstituées depuis les législatives 2022 (stables
        2012–2024) ; outre-mer et étranger ramenés en encarts.`)}
    ${li(`<b>Validité</b> : rejoué sur 2024, le modèle désigne le bon bloc en tête dans
        <b>${p.lead_accuracy ?? "—"} %</b> des bureaux. Il <b>ne voit pas</b> les réalignements
        brutaux, les dynamiques de campagne, ni les désistements/candidatures locales précises.`)}
  </ul>`;
}

function renderIntro() {
  $("intro").innerHTML =
    `<b>577 circonscriptions</b>, législatives 2027. Choisissez un <b>scénario</b> d'union et
     réglez le <b>niveau national</b> : la carte, les <b>sièges</b> et la <b>jouabilité</b>
     pour la gauche se recalculent en direct.
     <span class="muted"><a href="../" target="_blank">Méthode : carte 2024 ↗</a></span>`;
}

boot();
