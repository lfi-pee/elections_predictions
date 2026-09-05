"use strict";
// Panneau d'explication d'une circonscription : parts de bloc prédites au scénario courant,
// vainqueur probable du siège, score de jouabilité de la gauche et son raisonnement
// (configuration de gauche → qualification au 2nd tour → report/barrage), et l'écart de la
// circo au national. Tout se recalcule si l'on rouvre après avoir bougé un curseur.

function openCirco(pr) {
  APP.selected = pr.id;
  if (APP.map.getLayer("circo-sel")) APP.map.setFilter("circo-sel", ["==", "id", pr.id]);
  $("explain-body").innerHTML = renderCirco(pr);
  $("explain").classList.remove("hidden");
  $("explain").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function sharesBar(g, cd, ed) {
  const tot = g + cd + ed || 1;
  const seg = (b, v) => `<div style="width:${(v / tot) * 100}%;background:${APP.COL[b]}"
      title="${APP.NAME[b]} ${fmt1(v)} %">${(v / tot) * 100 >= 12 ? fmt1(v) + " %" : ""}</div>`;
  return `<div class="cs-bar">${seg("G", g)}${seg("CD", cd)}${seg("ED", ed)}</div>`;
}

function leftBreakdown(g, s) {
  if (s.left_config === "union")
    return `une candidature unique de gauche capte tout le bloc (<b>${fmt1(g)} %</b>).`;
  if (s.left_config === "split2") {
    const rad = g * s.radical_share, neo = g * (1 - s.radical_share);
    return `le bloc se scinde en deux : pôle radical <b>${fmt1(rad)} %</b> et pôle ` +
      `social-démocrate/néolibéral <b>${fmt1(neo)} %</b> — aucun ne pèse le total.`;
  }
  const rad = g * s.radical_share, o = g * (1 - s.radical_share);
  return `le bloc éclate en trois (radical ${fmt1(rad)} %, PS/PP ${fmt1(o * 0.6)} %, ` +
    `éco/PCF ${fmt1(o * 0.4)} %) — dispersion maximale.`;
}

function renderCirco(prIn) {
  // Recalcule tout au scénario/curseur courant à partir des déviations de la circo. En rejeu
  // 2024, on lit à la place les parts de 1er tour RÉELLES de la circo (comme la barre & la carte),
  // pour que le panneau soit cohérent avec la projection affichée.
  const s = APP.scnObj;
  let r = null;
  if (APP.replayMode) {
    const i = APP.idIdx.get(prIn.id);
    if (i != null) r = replayEval(i);
  }
  if (!r) r = circoEval(prIn);
  const turnout = Math.max(0.05, 1 - r.ab / 100), thr = 12.5 / turnout;
  const sc = r.sc, win = r.win;

  // Circos où la nomenclature de blocs ne couvre pas l'électorat (forces régionalistes hors
  // des trois blocs) : on N'AFFICHE PAS de score ni de siège probable — ils ne veulent rien
  // dire — et on explique pourquoi à la place. Cf. js/coverage.js.
  const pub = covIsPublishable(prIn.id);
  const dispTag = pub
    ? `<span class="tag" style="background:${APP.WIN[sc]}22;border-color:${APP.WIN[sc]}">${sc} · ${APP.WIN_LAB[sc]}</span>`
    : `<span class="tag" style="background:${APP.COV_GREY}44;border-color:${APP.COV_GREY}">${APP.COV_LAB}</span>`;
  const covBox = pub ? "" :
    `<div class="pv-nopub"><b>Prévision non fiable ici.</b> ${covWarning(prIn.id)}</div>`;
  const winLine = !pub ? "" :
    `<div class="pv-lead" style="background:${APP.COL[win]}22;border-left:3px solid ${APP.COL[win]}">
    Siège probable : <b>${APP.NAME[win]}</b> ${win === "G"
      ? "" : "— la gauche part " + (sc >= 4 ? "avec un net retard" : "au coude-à-coude")}.</div>`;

  // Raisonnement de la qualification / 2nd tour.
  let reason;
  if (!pub) {
    reason = `Le raisonnement habituel (qualification au second tour, reports, marge) n'est pas
      reproduit ici : il s'appuierait sur des parts de bloc qui ne représentent qu'une partie
      du corps électoral. Les chiffres ci-dessous sont donnés pour mémoire, <b>ils ne sont pas
      exploitables</b>.`;
  } else if (!r.ql) {
    reason = `Avec cette configuration, ${leftBreakdown(r.g, s)} La meilleure candidature de
      gauche (<b>${fmt1(r.lbest)} %</b> des exprimés) reste sous le seuil de qualification
      (~<b>${fmt1(thr)} %</b> des exprimés = 12,5 % des inscrits, participation ${fmt1(100 - r.ab)} %)
      et hors du duo de tête : <b>elle n'accède pas au second tour</b> → score 5.`;
  } else {
    const oppN = APP.NAME[r.opp];
    reason = `Avec cette configuration, ${leftBreakdown(r.g, s)} La gauche se qualifie
      (meilleur pôle ${fmt1(r.lbest)} %). Au second tour, réunie face à <b>${oppN}</b>, avec
      report/barrage estimé, l'écart est de <b>${r.mt2 > 0 ? "+" : ""}${fmt1(r.mt2)} pts</b>
      ${r.mt2 > 0 ? "en sa faveur" : "en sa défaveur"} → score ${sc} (${APP.WIN_LAB[sc]}).`;
  }

  const dev = (v) => (v >= 0 ? "+" : "") + fmt1(v);
  const devLine = `Cette circonscription vote, par rapport à la moyenne nationale :
    Gauche <b>${dev(prIn.dG)}</b>, Centre+Droite <b>${dev(prIn.dCD)}</b>,
    Extrême Droite <b>${dev(prIn.dED)}</b>, abstention <b>${dev(prIn.dAB)}</b> pts.
    C'est le motif local que le modèle prédit ; le niveau national vient des curseurs.`;

  return `<div class="pv-head"><h3>${prIn.id} · ${prIn.nm}</h3>
      <div class="sub">${prIn.dept} · ${fmt(prIn.ins)} inscrits · ${prIn.nbv} bureaux ${dispTag}</div></div>
    ${covBox}
    ${winLine}
    <div class="cs-h">Parts de bloc prédites (1er tour, exprimés)${pub ? "" : " — pour mémoire"}</div>
    ${sharesBar(r.g, r.cd, r.ed)}
    <div class="cs-leg">
      <span><i style="background:${APP.COL.G}"></i>Gauche ${fmt1(r.g)}</span>
      <span><i style="background:${APP.COL.CD}"></i>Centre+Droite ${fmt1(r.cd)}</span>
      <span><i style="background:${APP.COL.ED}"></i>Extrême Droite ${fmt1(r.ed)}</span>
      <span><i style="background:${APP.COL.AB}"></i>abstention ${fmt1(r.ab)} %</span></div>
    <div class="pv-why"><span class="pv-why-h">${pub ? "Pourquoi ce score" : "Pourquoi pas de score"}</span>
      <p class="cs-reason">${reason}</p></div>
    <p class="cap">${devLine}</p>
    <p class="cap muted">Bougez les curseurs nationaux ou changez de scénario : ce diagnostic
      se recalcule. Méthode et validation : voir <a href="../" target="_blank">la carte 2024</a>.</p>`;
}

function initPanel() {
  const close = () => {
    $("explain").classList.add("hidden");
    APP.selected = null;
    if (APP.map.getLayer("circo-sel")) APP.map.setFilter("circo-sel", ["==", "id", "___none___"]);
  };
  if ($("explain-close")) $("explain-close").onclick = close;
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
}
