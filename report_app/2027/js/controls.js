"use strict";
// Barre de contrôle : présélections de scénario, curseurs nationaux par parti, bascule de
// mode carte, et les deux barres dynamiques (sièges projetés, jouabilité). Tout changement
// appelle recomputeAll() → recalcul de la carte et des barres.

const BLOCKS = ["G", "CD", "ED", "AB"];
const MAJORITY = 289; // sièges pour la majorité absolue (577)

function currentScenario() {
  return APP.data.summary.scenarios.find((s) => s.key === APP.scenario);
}

// Repère pointillé « niveau 2024 » posé sur la piste d'un curseur (valeur → position en %).
function refTick(lo, hi, val, label) {
  if (val == null) return "";
  const pct = clamp((val - lo) / (hi - lo) * 100, 0, 100);
  return `<span class="sl-ref" style="left:${pct}%" title="${label} : ${fmt1(val)}"></span>`;
}

// Un scénario ne change QUE la configuration (union/division à gauche, union des droites) :
// il **préserve** le niveau national réglé aux curseurs. Le niveau, c'est « valeurs prédites »
// ou vos réglages — jamais le scénario. (« Réinitialiser » ramène aux valeurs prédites.)
function setScenario(key) {
  const s = APP.data.summary.scenarios.find((x) => x.key === key);
  if (!s) return;
  APP.scenario = key;
  APP.scnObj = s;
  // Changer de scénario réancre la part LFI sur l'ancrage sondages de ce scénario.
  APP.radOverride = null;
  document.querySelectorAll(".scn-btn").forEach((b) =>
    b.classList.toggle("on", b.dataset.k === key));
  $("scn-desc").textContent = s.desc;
  renderLfiShare();
  recomputeAll();
}

// Réinitialise les curseurs aux valeurs **prédites** du scénario courant (ancrage sondages).
function resetSliders() {
  exitReplay();
  APP.nat = { ...APP.scnObj.means };
  syncSliders();
  recomputeAll();
}

// Les curseurs G/CD/ED affichent les parts EFFECTIVES (après couplage participation γ) au
// niveau d'abstention courant ; APP.nat garde la base posée à l'abstention de référence.
function effShares() {
  const n = APP.nat;
  const [g, cd, ed] = natEffective(n.G, n.CD, n.ED, n.AB);
  return { G: g, CD: cd, ED: ed, AB: n.AB };
}
function syncSliders() {
  const eff = effShares();
  for (const b of BLOCKS) {
    const sl = $("sl-" + b);
    if (sl) sl.value = eff[b];
    const v = $("slv-" + b);
    if (v) v.textContent = fmt1(eff[b]) + " %";
  }
}

function initControls() {
  const sm = APP.data.summary;
  APP.scnObj = currentScenario();

  // Présélections de scénario.
  $("scenarios").innerHTML = sm.scenarios.map((s) =>
    `<button class="scn-btn${s.key === APP.scenario ? " on" : ""}" data-k="${s.key}">${s.label}</button>`
  ).join("");
  document.querySelectorAll(".scn-btn").forEach((b) =>
    (b.onclick = () => { exitReplay(); setScenario(b.dataset.k); }));
  $("scn-desc").textContent = APP.scnObj.desc;

  // Curseurs nationaux par parti.
  const R = sm.slider_ranges;
  const names = { G: "Gauche", CD: "Centre+Droite", ED: "Extrême Droite", AB: "Abstention" };
  $("sliders").innerHTML = BLOCKS.map((b) => {
    const [lo, hi] = R[b];
    return `<div class="sl-row"><label style="color:${APP.COL[b]}">${names[b]}
      <span class="sl-v" id="slv-${b}">${fmt1(APP.nat[b])} %</span></label>
      <div class="sl-track"><input type="range" id="sl-${b}" min="${lo}" max="${hi}" step="0.5"
        value="${APP.nat[b]}" style="--c:${APP.COL[b]}">${refTick(lo, hi, APP.REF2024[b], "niveau 2024")}</div></div>`;
  }).join("");
  for (const b of BLOCKS) {
    $("sl-" + b).addEventListener("input", (e) => {
      exitReplay();
      setNat(b, parseFloat(e.target.value));
      recomputeAll();
    });
  }
  if ($("reset")) $("reset").onclick = resetSliders;

  renderLfiShare();
  renderTransfers();

  // Bascule de mode carte.
  $("mode-win").onclick = () => { setMode("win"); syncModeBtns(); };
  $("mode-seat").onclick = () => { setMode("seat"); syncModeBtns(); };
  syncModeBtns();

  // Bascule « détail par pôle de gauche » de la barre des sièges.
  if ($("seat-detail")) $("seat-detail").onclick = () => {
    APP.state.seatDetail = !APP.state.seatDetail;
    $("seat-detail").classList.toggle("on", APP.state.seatDetail);
    updateSeatBar();
  };

  if ($("replay-2024")) $("replay-2024").onclick = replay2024;
}

// « Rejouer 2024 » : bascule en mode REJEU — la barre de sièges, la carte et la répartition
// évaluent le modèle de 2nd tour sur les parts de 1er tour RÉELLES 2024 par circo (gauche unie,
// comme le NFP), reproduisant le backtest « modèle de sièges seul » À L'IDENTIQUE (fini le motif
// spatial 2027 approché). On pose aussi les curseurs au niveau national 2024 (indicatif) et on
// affiche la validation. Le moindre curseur/scénario (exitReplay) rebascule vers la prévision 2027.
function replay2024() {
  APP._replaying = true;          // empêche recomputeAll() de masquer la validation qu'on affiche
  APP.replayMode = true;          // barres + carte lisent désormais les parts RÉELLES 2024 (circoArr.r24*)
  // Curseurs au niveau national 2024 : REF2024 est le résultat EFFECTIF (à ~33 % d'abstention) ;
  // on inverse le couplage γ pour la base. Indicatif en rejeu (le calcul par circo lit les parts
  // réelles, pas les curseurs) mais cohérent avec le repère « niveau 2024 » des curseurs.
  const r = APP.REF2024;
  APP.nat.AB = r.AB;
  const [bg, bc, be] = natBaseFromEffective(r.G, r.CD, r.ED, r.AB);
  APP.nat.G = bg; APP.nat.CD = bc; APP.nat.ED = be;
  APP.radOverride = null;
  setScenario("union");           // 2024 : gauche unie ; recompute (lit replayMode) + rendu LFI
  syncSliders();
  APP._replaying = false;
  const bt = APP.data.summary.backtest_2024;
  const box = $("replay-box");
  if (!box) return;
  if (!bt) { box.className = "replay"; box.textContent = "Backtest 2024 indisponible."; return; }
  const seats = (o) => `<b style="color:${APP.COL.G}">${o.G}</b> · ` +
    `<b style="color:${APP.COL.CD}">${o.CD}</b> · <b style="color:${APP.COL.ED}">${o.ED}</b>`;
  const e2e = APP.data.summary.backtest_2024_e2e;
  box.className = "replay";
  box.innerHTML =
    `<div class="rp-h">Validation — rejeu de 2024 <span class="muted">(${bt.n_circo} circonscriptions cartographiables)</span></div>` +
    `<div>résultat réel&nbsp;: ${seats(bt.actual)} <span class="muted">(G · C+D · ED)</span></div>` +
    (e2e
      ? `<div class="rp-sub"><b>À l'aveugle — chaîne complète</b>&nbsp;: ${seats(e2e.model)} ` +
        `<span class="muted">— <b>${e2e.n_correct}/${e2e.n_circo}</b> bons vainqueurs (<b>${e2e.accuracy_seats}&nbsp;%</b> ; ${e2e.accuracy}&nbsp;% pondéré inscrits)</span></div>` +
        `<div class="muted">2024 <b>entièrement retiré de l'entraînement</b> : le modèle prédit le <b>motif local</b> du 1<sup>er</sup> tour à l'aveugle (niveau national posé au réel 2024, comme les curseurs le poseront pour 2027 ; erreur ~${e2e.mae_t1.G}–${e2e.mae_t1.CD} pts/circo) — puis le modèle de sièges tranche. Validation honnête de <b>bout en bout</b>. Il <b>sous-estime le RN</b> (${e2e.model.ED} sièges projetés vs ${e2e.actual.ED} réels) : privé de 2024, il lisse la poussée spatiale du RN.</div>`
      : "") +
    `<div class="rp-sub">Modèle de sièges seul <span class="muted">(1<sup>er</sup> tour réel)</span>&nbsp;: ${seats(bt.model)} ` +
    `<span class="muted">— <b>${bt.accuracy}&nbsp;%</b> (pondéré inscrits) ; isole l'erreur du 2nd tour, ancre de 1<sup>er</sup> tour parfaite.</span></div>` +
    `<div class="muted">La barre de sièges et la carte affichent MAINTENANT <b>exactement</b> cette dernière ligne : le modèle de 2nd tour sur les parts de 1<sup>er</sup> tour <b>réelles</b> de 2024 (gauche unie, ${bt.n_circo} circos). Bougez un curseur pour revenir à la prévision 2027.</div>`;
}

// Sort du mode rejeu (au moindre réglage curseur/scénario) : retour à la prévision 2027 et on
// masque la validation. Rappelé par tous les points d'entrée « réglage » ci-dessous.
function exitReplay() {
  if (!APP.replayMode) return;
  APP.replayMode = false;
  const rb = $("replay-box");
  if (rb) rb.className = "replay hidden";
}

// Déplacer un curseur de bloc (G/CD/ED) redistribue le reste sur les deux autres au prorata,
// pour que G+C+D+ED = 100 % des exprimés. L'abstention est un axe à part (% des inscrits).
function setNat(b, v) {
  if (b === "AB") {
    // Bouger l'abstention ne touche pas la base : les curseurs de bloc se re-synchronisent sur
    // les nouvelles parts EFFECTIVES (c'est là que l'abstention agit visiblement sur les blocs).
    APP.nat.AB = v;
    syncSliders();
    return;
  }
  // v = part EFFECTIVE voulue pour le bloc b (ce que montre le curseur). On redistribue les deux
  // autres au prorata de leur effectif courant, puis on inverse le couplage γ pour stocker la
  // base (à l'abstention de référence) que le calcul par circo consomme.
  const cur = effShares();
  const others = APP.VOTE.filter((x) => x !== b);
  const rem = 100 - v, sum = others.reduce((a, x) => a + cur[x], 0);
  const tgt = { [b]: v };
  for (const x of others) tgt[x] = sum > 0 ? rem * cur[x] / sum : rem / others.length;
  const [bg, bc, be] = natBaseFromEffective(tgt.G, tgt.CD, tgt.ED, APP.nat.AB);
  APP.nat.G = bg; APP.nat.CD = bc; APP.nat.ED = be;
  syncSliders();
}

function syncModeBtns() {
  $("mode-win").classList.toggle("on", APP.state.mode === "win");
  $("mode-seat").classList.toggle("on", APP.state.mode === "seat");
}

// ── Part de la gauche radicale (LFI) dans le bloc de gauche ──
// Ancrage sondages via le scénario ; réglable au curseur (sans effet en gauche unie).
const POLE_RAD = "#a01722";
function currentRad() {
  return APP.radOverride != null ? APP.radOverride : APP.scnObj.radical_share;
}
function renderLfiShare() {
  const el = $("lfi-share");
  if (!el) return;
  if (APP.scnObj.left_config === "union") {
    el.innerHTML = `<div class="lfi-na">Gauche unie : une candidature unique capte tout le
      bloc — la part LFI ne s'applique pas.</div>`;
    return;
  }
  const pct = Math.round(currentRad() * 100);
  el.innerHTML =
    `<div class="ctl-h sub">Part de la gauche radicale (LFI)
       <span class="info">i<span class="tip">Le modèle prédit le bloc de gauche entier ; cette
         part fixe le poids du pôle radical (LFI) face au pôle social-démocrate
         (PS·Place publique·EELV·PCF) quand la gauche concourt divisée. Défaut = ancrage
         sondages « gauche divisée » (2025 ; aucun sondage législatif 2026 ne la scinde).
         Modulée localement selon la force de la gauche dans la circo.</span></span>
       <button id="lfi-reset" class="reset-btn" title="Revenir à la part du scénario">↺</button></div>
     <div class="sl-row"><label style="color:${POLE_RAD}">LFI
       <span class="sl-v" id="lfiv">${pct} %</span></label>
       <input type="range" id="sl-lfi" min="5" max="80" step="1" value="${pct}" style="--c:${POLE_RAD}"></div>
     <div class="natsum muted" id="lfi-rest">reste du bloc → PS·PP·EELV·PCF : ${100 - pct} %</div>`;
  $("sl-lfi").addEventListener("input", (e) => {
    exitReplay();
    APP.radOverride = parseFloat(e.target.value) / 100;
    const p = Math.round(APP.radOverride * 100);
    $("lfiv").textContent = p + " %";
    $("lfi-rest").textContent = `reste du bloc → PS·PP·EELV·PCF : ${100 - p} %`;
    recomputeAll();
  });
  $("lfi-reset").onclick = () => { exitReplay(); APP.radOverride = null; renderLfiShare(); recomputeAll(); };
}

// ── Reports de 2nd tour réglables (miroir des défauts de winnability_2027.py) ──
const COEF_DEFAULT = { desist: 0.60, cdLR: 0.46, ed2left: 0.15, reunif: 0.72 };
const COEF_META = [
  { k: "desist", lab: "Désistement « front républicain » (triangulaire face au RN)" },
  { k: "cdLR", lab: "Part LR dans le bloc Centre+Droite (le reste = Ensemble)" },
  { k: "ed2left", lab: "Report : électeurs RN → gauche (duel gauche vs centre-droit)" },
  { k: "reunif", lab: "Gauche divisée : voix d'un pôle éliminé → pôle de gauche restant" },
];
function renderTransfers() {
  const el = $("transfers");
  if (!el) return;
  el.innerHTML = COEF_META.map((c) => {
    const pct = Math.round(APP.coef[c.k] * 100);
    return `<div class="sl-row"><label>${c.lab}
      <span class="sl-v" id="cfv-${c.k}">${pct} %</span></label>
      <div class="sl-track"><input type="range" id="cf-${c.k}" min="0" max="100" step="1" value="${pct}"
        style="--c:#8a8f98">${refTick(0, 100, COEF_DEFAULT[c.k] * 100, "calibré 2024")}</div></div>`;
  }).join("");
  for (const c of COEF_META) {
    $("cf-" + c.k).addEventListener("input", (e) => {
      exitReplay();
      APP.coef[c.k] = parseFloat(e.target.value) / 100;
      $("cfv-" + c.k).textContent = Math.round(APP.coef[c.k] * 100) + " %";
      recomputeAll();
    });
  }
  if ($("transfers-reset")) $("transfers-reset").onclick = () => {
    exitReplay();
    Object.assign(APP.coef, COEF_DEFAULT);
    renderTransfers();
    recomputeAll();
  };
}

// ── Barre dynamique des sièges (projection) ──
function updateSeatBar() {
  const t = seatTally(), tot = t.G + t.CD + t.ED || 1;
  const stn = $("seat-total-note");
  // En rejeu, la barre somme sur les 501 circos cartographiables (= backtest) ; sinon 577.
  if (stn) stn.textContent = APP.replayMode
    ? "· rejeu 2024 · 501 circos cartographiables" : "· 577 · majorité 289";
  const seg = (w, col, lab, n) =>
    `<div class="seat-seg" style="width:${(n / tot) * 100}%;background:${col}"
      title="${lab} : ${n} sièges">${n >= 20 ? n : ""}</div>`;
  const maj = `<div class="seat-maj" style="left:${(MAJORITY / tot) * 100}%" title="majorité absolue : ${MAJORITY}"></div>`;

  let segs, legend;
  if (APP.state.seatDetail) {
    // Détail : sièges de gauche ventilés par pôle (seule sous-composante résolue par circo).
    const pm = poleMeta(APP.scnObj.left_config);
    const pseg = pm.map((p, i) => seg("G", p.col, p.lab, t.poles[i] || 0)).join("");
    segs = pseg + seg("CD", APP.COL.CD, APP.NAME.CD, t.CD) + seg("ED", APP.COL.ED, APP.NAME.ED, t.ED);
    legend = pm.map((p, i) =>
      `<span class="sl-leg"><i style="background:${p.col}"></i>${p.lab} <b>${t.poles[i] || 0}</b></span>`).join("") +
      ["CD", "ED"].map((b) =>
        `<span class="sl-leg"><i style="background:${APP.COL[b]}"></i>${APP.NAME[b]} <b>${t[b]}</b></span>`).join("");
  } else {
    segs = ["G", "CD", "ED"].map((b) => seg(b, APP.COL[b], APP.NAME[b], t[b])).join("");
    legend = ["G", "CD", "ED"].map((b) =>
      `<span class="sl-leg"><i style="background:${APP.COL[b]}"></i>${APP.NAME[b]} <b>${t[b]}</b></span>`).join("");
  }
  $("seatbar").innerHTML = segs + maj;
  const lead = t.G >= t.CD && t.G >= t.ED ? "G" : t.CD >= t.ED ? "CD" : "ED";
  const hasMaj = Math.max(t.G, t.CD, t.ED) >= MAJORITY;
  $("seat-legend").innerHTML = legend +
    `<span class="seat-note">${hasMaj ? `majorité absolue : <b style="color:${APP.COL[lead]}">${APP.NAME[lead]}</b>`
      : `pas de majorité (${MAJORITY} requis) — <b style="color:${APP.COL[lead]}">${APP.NAME[lead]}</b> en tête</b>`}</span>`;
}

function updateNatBar() {
  const n = APP.nat;
  // Les curseurs G/CD/ED montrent déjà les parts EFFECTIVES (couplage γ à l'abstention courante).
  // Ici on rappelle l'abstention et, si elle s'écarte de la référence, l'ancrage « posé » sous-
  // jacent (à la référence) — la base que vous avez fixée, avant redistribution des revenants.
  const moved = Math.abs(n.AB - APP.AB_REF) > 0.4;
  const note = moved
    ? `<span class="eff">curseurs = parts <b>effectives</b> à ${fmt1(n.AB)} % d'abstention ` +
      `<span class="muted">(les revenants penchent à gauche — γ 2024) · ancrage posé à ` +
      `${fmt1(APP.AB_REF)} % : G ${fmt1(n.G)} · C+D ${fmt1(n.CD)} · ED ${fmt1(n.ED)}</span></span>`
    : `<span class="eff muted">à l'abstention de référence (${fmt1(APP.AB_REF)} %). ` +
      `La baisser fera monter la gauche (courbe γ 2024).</span>`;
  $("natsum").innerHTML = `abstention ${fmt1(n.AB)} %` + note;
}

// ── Répartition de la jouabilité (1→5) ──
function updateWinSummary() {
  const t = scoreTally(), tot = Object.values(t).reduce((a, b) => a + b, 0) || 1;
  const playable = t[1] + t[2] + t[3];
  $("winsum").innerHTML =
    `<div class="ws-head"><b>${playable}</b> circonscriptions jouables pour la gauche ` +
    `<span class="muted">(scores 1–3 sur 577)</span></div>` +
    `<div class="ws-bar">` +
    [1, 2, 3, 4, 5].map((s) =>
      `<div class="ws-seg" style="width:${(t[s] / tot) * 100}%;background:${APP.WIN[s]}"
        title="${APP.WIN_LAB[s]} : ${t[s]}">${t[s] >= 18 ? t[s] : ""}</div>`).join("") +
    `</div>` +
    `<div class="ws-key">` +
    [1, 2, 3, 4, 5].map((s) =>
      `<span><i style="background:${APP.WIN[s]}"></i>${s} · ${APP.WIN_LAB[s]}</span>`).join("") +
    `</div>`;
}

// Fourchette d'incertitude (Monte-Carlo des intervalles conformes) — calcul débounce, hors frame.
function updateUncertainty() {
  // En rejeu 2024, il n'y a pas d'incertitude de prévision : c'est un résultat réel projeté par
  // le modèle de sièges. On remplace la fourchette par la mention du rejeu (ponctuel).
  if (APP.replayMode) {
    const se = $("seat-range");
    if (se) se.innerHTML = `<span class="muted">rejeu 2024 — projection ponctuelle sur les parts réelles (pas de fourchette de prévision)</span>`;
    const we = $("win-range");
    if (we) we.innerHTML = "";
    return;
  }
  const d = seatDistribution(MC_DRAWS);
  if (!d) return;
  const rng = (b) => `${b.lo}–${b.hi}`;
  const se = $("seat-range");
  if (se) se.innerHTML =
    `fourchette 90 % <span class="muted">(incertitude locale ; niveau national posé)</span> : ` +
    `<b style="color:${APP.COL.G}">G ${rng(d.G)}</b> · ` +
    `<b style="color:${APP.COL.CD}">C+D ${rng(d.CD)}</b> · ` +
    `<b style="color:${APP.COL.ED}">ED ${rng(d.ED)}</b>`;
  const we = $("win-range");
  if (we) we.innerHTML =
    `fourchette 90 % : <b>${rng(d.play)}</b> circonscriptions jouables ` +
    `<span class="muted">(médiane ${d.play.med})</span>`;
}

function updateLegend() {
  const el = $("legend");
  if (APP.state.mode === "seat") {
    el.innerHTML = `<span class="legend-lab">vainqueur probable du siège</span>` +
      ["G", "CD", "ED"].map((b) => `<i style="background:${APP.COL[b]}"></i>${APP.NAME[b]}`).join(" ");
  } else {
    el.innerHTML = `<span class="legend-lab">jouabilité pour la gauche</span>` +
      [1, 2, 3, 4, 5].map((s) => `<i style="background:${APP.WIN[s]}"></i>${s}`).join("") +
      `<span class="legend-ends">facile → impossible</span>`;
  }
}
