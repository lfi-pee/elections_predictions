"use strict";
const _detailCache = new Map();
const SCALE = 60;

async function fetchDetail(dept) {
  if (_detailCache.has(dept)) return _detailCache.get(dept);
  const d = await loadJSON(`data/detail/${dept}.json`).catch(() => ({}));
  _detailCache.set(dept, d);
  return d;
}

async function openPanel(props) {
  const loc = props.l;
  const dept = loc.slice(0, 2);
  const detail = await fetchDetail(dept);
  const rec = detail[loc];
  if (!rec) return;
  $("panel-body").innerHTML = renderPanel(loc, rec);
  $("panel").classList.remove("hidden");
}

function bar(b, blk) {
  const w = (x) => Math.max(0, Math.min(100, (x / SCALE) * 100));
  const ciL = w(blk.lo), ciW = Math.max(0.5, w(blk.hi) - w(blk.lo));
  const center = (blk.lo + blk.hi) / 2, hw = (blk.hi - blk.lo) / 2;
  const locFrac = Math.sqrt((APP.data.provenance.blocks[b].local_share || 100) / 100);
  const locL = w(center - hw * locFrac);
  const locW = Math.max(0.5, w(center + hw * locFrac) - locL);
  return `<div class="bar"><div class="lab"><span style="color:${APP.COL[b]}">${APP.NAME[b]}</span>
    <span><b>${blk.pred.toLocaleString("fr-FR", { minimumFractionDigits: 1 })}</b> %
    <span style="color:#9a9aa2"> · réel ${blk.act.toLocaleString("fr-FR", { minimumFractionDigits: 1 })}</span></span></div>
    <div class="track">
      <div class="fill" style="width:${w(blk.pred)}%;background:${APP.COL[b]}"></div>
      <div class="ci ci-nat" style="left:${ciL}%;width:${ciW}%"></div>
      <div class="ci ci-loc" style="left:${locL}%;width:${locW}%"></div>
      <div class="act" style="left:${w(blk.act)}%"></div>
    </div></div>`;
}

function driverBars(drivers, col) {
  const max = Math.max(...drivers.map((d) => Math.abs(d[1])), 0.1);
  return drivers.map(([lab, v]) => {
    const w = Math.max(2, (Math.abs(v) / max) * 50);
    const pos = v >= 0;
    const fill = pos
      ? `<div class="dv-bar" style="left:50%;width:${w}%;background:${col}"></div>`
      : `<div class="dv-bar dv-neg" style="right:50%;width:${w}%"></div>`;
    const val = (pos ? "+" : "") + v.toLocaleString("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
    return `<div class="dv-row"><span class="dv-lab" title="${lab}">${lab}</span>
      <span class="dv-track">${fill}</span><span class="dv-val">${val}</span></div>`;
  }).join("");
}

function whyBlock(rec) {
  const drivers = rec.drivers;
  if (!drivers || !drivers.length) return "";
  return `<div class="pv-why"><span class="pv-why-h">Pourquoi ${APP.NAME[rec.lead]} dévie du national</span>
    <div class="dv-cap">contribution de chaque facteur à l'écart au national, en points</div>
    <div class="dv-chart">${driverBars(drivers, APP.COL[rec.lead])}</div></div>`;
}

// The mobilization score explained: mv = abstainers × γ, plus the Left-model SHAP
// drivers that make this bureau lean Left (hence its γ). The "avec le SHAP" answer.
function whyMobil(rec) {
  if (rec.mob === undefined) return "";
  const abs = Math.round((rec.blocks.AB.act / 100) * rec.i);
  const g = abs > 0 ? Math.round((rec.mob / abs) * 100) : 0;
  const phrase = rec.wleft ? `<div class="dv-cap">Ce bureau ${rec.wleft}.</div>` : "";
  const bars = (rec.gdrivers && rec.gdrivers.length)
    ? `<div class="dv-cap" style="margin-top:8px">ce qui règle son niveau de gauche — donc la part qui penche à gauche — contribution de chaque facteur au score Gauche, en points</div>
       <div class="dv-chart">${driverBars(rec.gdrivers, APP.COL.G)}</div>`
    : "";
  return `<div class="pv-why"><span class="pv-why-h" style="color:${APP.COL.G}">Pourquoi ce bureau est mobilisable</span>
    <div class="pv-mob-eq"><b>${fmt(rec.mob)}</b> électeurs mobilisables
      = ${fmt(abs)} abstentionnistes, dont ${g} % pencheraient à gauche</div>
    ${phrase}${bars}</div>`;
}

function renderPanel(loc, rec) {
  const [code, num] = loc.split("_");
  const lead = leadUnderScenario(rec);
  const baseLead = rec.lead;
  const flipped = lead !== baseLead;
  const tip = rec.tip;
  const tipTxt = tip > 0
    ? `Il faudrait <b>+${tip.toLocaleString("fr-FR", { minimumFractionDigits: 1 })} pts</b> d'Extrême Droite au national pour faire basculer ce bureau en sa faveur.`
    : `L'Extrême Droite y devance déjà : il faudrait <b>${tip.toLocaleString("fr-FR", { minimumFractionDigits: 1 })} pts</b> au national pour la faire passer derrière.`;
  const dispTag = rec.m >= 8
    ? `<span class="tag">rang net</span>`
    : rec.m >= 3
      ? `<span class="tag">rang serré</span>`
      : `<span class="tag warn">rang disputé</span>`;
  const order = ["G", "CD", "ED", "AB"];
  const bars = order.map((b) => bar(b, rec.blocks[b])).join("");
  const natShare = Math.round(APP.data.provenance.blocks[baseLead].national_share);
  const provTxt = `Sur ${APP.NAME[baseLead]}, environ <b>${natShare} %</b> de cette marge
    d'erreur tient au national (sondages) ; le reste vient de notre lecture du terrain
    propre à ce bureau.`;
  const flipTxt = flipped
    ? `<div class="pv-tip"><span class="pv-tip-h">Sous le scénario courant</span><br>
       Ce bureau <b>bascule</b> : ${APP.NAME[baseLead]} → ${APP.NAME[lead]}.</div>`
    : "";
  return `<div class="pv-head"><h3>${rec.n}</h3>
      <div class="sub">Bureau ${num} · commune ${code} · ${fmt(rec.i)} inscrits ${dispTag}</div></div>
    <div class="pv-lead" style="background:${APP.COL[baseLead]}22;border-left:3px solid ${APP.COL[baseLead]}">
      Bloc en tête prédit : <b>${APP.NAME[baseLead]}</b>, marge ${rec.m.toLocaleString("fr-FR", { minimumFractionDigits: 1 })} pts sur ${APP.NAME[rec.ru]}.</div>
    ${flipTxt}
    ${bars}
    <p class="cap">Barre pleine = prédit · trait noir = réel · fourchette de prévision
    à 90 % : partie <b>sombre</b> = notre lecture locale, prolongement <b>clair</b> = la
    part qui vient du national (sondages).</p>
    <div class="pv-tip">${provTxt}</div>
    <div class="pv-tip">${tipTxt}</div>
    ${APP.state.mode === "mobil" ? whyMobil(rec) : whyBlock(rec)}`;
}

function leadUnderScenario(rec) {
  const a = appliedDeltas(), bl = rec.blocks;
  const g = bl.G.pred + a.G, c = bl.CD.pred + a.CD, e = bl.ED.pred + a.ED;
  return g >= c && g >= e ? "G" : c >= e ? "CD" : "ED";
}

function initPanel() {
  $("panel-close").onclick = () => $("panel").classList.add("hidden");
}
