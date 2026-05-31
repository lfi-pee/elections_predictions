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
  return drivers.map(([lab, v, fv]) => {
    const w = Math.max(2, (Math.abs(v) / max) * 50);
    const pos = v >= 0;
    const fill = pos
      ? `<div class="dv-bar" style="left:50%;width:${w}%;background:${col}"></div>`
      : `<div class="dv-bar dv-neg" style="right:50%;width:${w}%"></div>`;
    const val = (pos ? "+" : "−") + Math.abs(v).toLocaleString("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
    const sub = fv ? `<span class="dv-fval">${fv}</span>` : "";
    return `<div class="dv-row"><span class="dv-lab"><span class="dv-name">${lab}</span>${sub}</span>
      <span class="dv-track">${fill}</span><span class="dv-val">${val}</span></div>`;
  }).join("");
}

function whyBlock(rec) {
  const drivers = rec.drivers;
  if (!drivers || !drivers.length) return "";
  return `<div class="pv-why"><span class="pv-why-h">Pourquoi ${APP.NAME[rec.lead]} dévie du national</span>
    <div class="dv-cap">Le facteur dominant est presque toujours l'<b>héritage de vote</b> du bureau (n‑1) — il vote largement comme la dernière fois ; la démographie n'ajoute qu'une correction. Barre : contribution de chaque facteur à l'écart au national, en points. Sous chaque facteur, sa valeur dans ce bureau (votes passés : écart au national).</div>
    <div class="dv-chart">${driverBars(drivers, APP.COL[rec.lead])}</div></div>`;
}

// The mobilization score explained: mobilizable = conjunctural abstainers × γ, the
// resulting Left score if they all turn out, plus the Left-model SHAP drivers that make
// this bureau lean Left (hence its γ). Conjunctural = abstainers minus the chronic floor.
function whyMobil(rec) {
  if (rec.mob === undefined) return "";
  const abs = Math.round((rec.blocks.AB.act / 100) * rec.i);
  const conj = rec.conj !== undefined ? rec.conj : abs;
  const g = conj > 0 ? Math.round((rec.mob / conj) * 100) : 0;
  const voters = rec.i * (1 - rec.blocks.AB.act / 100);
  const cur = rec.blocks.G.pred;
  // If the whole conjunctural frange returns (not just its Left share), turnout grows by
  // `conj` and the Left gains `mob` of them — the honest GOTV arithmetic, not a best case
  // that assumes you bring only Left voters.
  const next = voters > 0 ? ((cur / 100) * voters + rec.mob) / (voters + conj) * 100 : cur;
  const phrase = rec.wleft ? `<div class="dv-cap">Ce bureau ${rec.wleft}.</div>` : "";
  const bars = (rec.gdrivers && rec.gdrivers.length)
    ? `<div class="dv-cap" style="margin-top:8px">Ce qui tire le niveau de gauche de ce bureau, facteur par facteur. Barre : contribution au score Gauche, en points. Sous chaque facteur, sa valeur dans ce bureau (votes passés : écart au national).</div>
       <div class="dv-chart">${driverBars(rec.gdrivers, APP.COL.G)}</div>`
    : "";
  return `<div class="pv-why"><span class="pv-why-h" style="color:${APP.COL.G}">Pourquoi ce bureau est mobilisable</span>
    <div class="pv-mob-eq"><b>${fmt(rec.mob)}</b> électeurs à aller chercher
      = ${fmt(conj)} abstentionnistes conjoncturels, dont ${g} % pencheraient à gauche</div>
    <div class="dv-cap">conjoncturels = abstentionnistes hors abstention de fond (chronique) — ${fmt(abs)} abstentionnistes au total</div>
    <div class="pv-mob-score">Si toute cette frange conjoncturelle revient voter : Gauche
      <b>${cur.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} % → ${next.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} %</b></div>
    ${phrase}${bars}</div>`;
}

function renderPanel(loc, rec) {
  const num = loc.split("_")[1];
  const baseLead = rec.lead;
  const dispTag = rec.m >= 8
    ? `<span class="tag">avance nette</span>`
    : rec.m >= 3
      ? `<span class="tag">avance serrée</span>`
      : `<span class="tag warn">issue incertaine</span>`;
  const order = ["G", "CD", "ED", "AB"];
  const bars = order.map((b) => bar(b, rec.blocks[b])).join("");
  return `<div class="pv-head"><h3>${rec.n}</h3>
      <div class="sub">Bureau de vote n°${num} · ${fmt(rec.i)} inscrits ${dispTag}</div></div>
    <div class="pv-lead" style="background:${APP.COL[baseLead]}22;border-left:3px solid ${APP.COL[baseLead]}">
      Bloc en tête prédit : <b>${APP.NAME[baseLead]}</b>, marge ${rec.m.toLocaleString("fr-FR", { minimumFractionDigits: 1 })} pts sur ${APP.NAME[rec.ru]}.</div>
    ${bars}
    <p class="cap">Barre pleine = prédit · trait noir = réel · fourchette de prévision
    à 90 %, partagée selon la part <b>moyenne</b> d'incertitude du bloc : <b>sombre</b> =
    notre lecture locale, <b>clair</b> = ce qui vient du national (sondages).</p>
    ${APP.state.mode === "mobil" ? whyMobil(rec) : whyBlock(rec)}`;
}

function initPanel() {
  $("panel-close").onclick = () => $("panel").classList.add("hidden");
}
