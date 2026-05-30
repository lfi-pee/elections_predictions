"use strict";

// Stacked G/CD/ED bar — shared by the "réalité partagée" counterpoint and the seats
// rapport-de-force. `vals` are raw counts; segments show the count when wide enough.
function stackedBar(label, vals) {
  const tot = vals.reduce((a, b) => a + b, 0) || 1, blocs = ["G", "CD", "ED"];
  const segs = blocs.map((b, k) => {
    const pct = (vals[k] / tot) * 100;
    return `<div class="ls-seg" style="width:${pct}%;background:${APP.COL[b]}">${pct >= 9 ? fmt(vals[k]) : ""}</div>`;
  }).join("");
  return `<div class="ls-row"><div class="lab">${label}</div><div class="ls-bar">${segs}</div></div>`;
}

// The counterpoint to "un sondage = un seul favori partout" : the real lead is split
// across three blocs, bureau by bureau.
function renderRealite() {
  const base = APP.data.baseLead, cnt = [0, 0, 0];
  for (let i = 0; i < base.length; i++) cnt[base[i]]++;
  $("realite").innerHTML = stackedBar("La réalité, bureau par bureau", cnt);
}

function seatLead(i, ad) {
  const c = APP.data.circo;
  const g = c.g[i] + ad.G, cd = c.c[i] + ad.CD, e = c.e[i] + ad.ED;
  return g >= cd && g >= e ? 0 : cd >= e ? 1 : 2;
}

function seatCounts(ad) {
  const cnt = [0, 0, 0];
  for (let i = 0; i < APP.data.circo.g.length; i++) cnt[seatLead(i, ad)]++;
  return cnt;
}

function seatFlips(ad) {
  const base = APP.data.seatBase;
  let f = 0;
  for (let i = 0; i < base.length; i++) if (seatLead(i, ad) !== base[i]) f++;
  return f;
}

function drawSeats() {
  const zero = { G: 0, CD: 0, ED: 0 };
  $("seats").innerHTML =
    stackedBar("Aujourd'hui", seatCounts(zero)) +
    stackedBar("Sous le scénario courant", seatCounts(appliedDeltas()));
}

// The seats made interrogable: the circonscriptions with the thinnest current
// margin under the scenario, named by their largest commune. Reorders live; the
// ones that just flipped vs. baseline light up. This is the decision-maker's unit.
function drawFil(ad) {
  const c = APP.data.circo, base = APP.data.seatBase, blocs = ["G", "CD", "ED"];
  const rows = [];
  for (let i = 0; i < c.g.length; i++) {
    const v = [c.g[i] + ad.G, c.c[i] + ad.CD, c.e[i] + ad.ED];
    const order = [0, 1, 2].sort((x, y) => v[y] - v[x]);
    rows.push({
      i, lead: order[0], ru: order[1],
      margin: v[order[0]] - v[order[1]],
      flipped: order[0] !== base[i],
    });
  }
  rows.sort((a, b) => a.margin - b.margin);
  $("fil").innerHTML = rows.slice(0, 12).map((r) => {
    const [d, n] = c.id[r.i].split("-");
    const lb = blocs[r.lead], rb = blocs[r.ru];
    return `<div class="fil-row${r.flipped ? " flipped" : ""}">` +
      `<span class="fil-name">${c.nm[r.i]}<small>${d} · circ. ${+n}</small></span>` +
      `<span class="fil-lead"><b style="background:${APP.COL[lb]}"></b>${APP.NAME[lb]}` +
      `${r.flipped ? '<span class="fil-tag">vient de basculer</span>' : ""}</span>` +
      `<span class="fil-margin">marge <b>${r.margin.toLocaleString("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}</b> pts sur ${APP.NAME[rb]}</span>` +
      `</div>`;
  }).join("");
}

function renderAccuracy() {
  const pts = APP.data.summary.accuracy_by_margin;
  const W = 360, H = 170, padL = 30, padB = 26, padT = 10;
  const n = pts.length, gap = 6;
  const bw = (W - padL - 6 - gap * (n - 1)) / n;
  const Y = (a) => padT + (1 - (a - 40) / 60) * (H - padT - padB);
  let bars = "";
  pts.forEach((p, k) => {
    const x = padL + k * (bw + gap), y = Y(p.acc), h = Math.max(0, H - padB - y);
    const hot = p.hi <= APP.state.marginT;
    const lab = p.hi >= 100 ? `${p.lo}+` : `${p.lo}–${p.hi}`;
    bars +=
      `<rect class="accbar" data-hi="${p.hi}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" ` +
      `width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="2" fill="${hot ? APP.COL.G : "#1A1A2E"}"></rect>` +
      `<text x="${(x + bw / 2).toFixed(1)}" y="${(y - 3).toFixed(1)}" font-size="8.5" ` +
      `fill="${hot ? APP.COL.G : "#5a5a64"}" text-anchor="middle">${Math.round(p.acc)}</text>` +
      `<text x="${(x + bw / 2).toFixed(1)}" y="${H - 8}" font-size="8" fill="#9a9aa2" text-anchor="middle">${lab}</text>`;
  });
  $("accmargin").innerHTML =
    `<text x="2" y="${Y(100) + 3}" font-size="8" fill="#9a9aa2">100%</text>` +
    `<text x="2" y="${Y(40) + 3}" font-size="8" fill="#9a9aa2">40%</text>` +
    bars +
    `<text x="${padL}" y="${H - 0.5}" font-size="8" fill="#9a9aa2">écart entre le 1ᵉ et le 2ᵉ bloc (points)</text>`;
  const lo = pts[0], hi = pts[pts.length - 1];
  $("acc-cap").innerHTML =
    `Le bon bloc désigné dans <b>${Math.round(hi.acc)} %</b> des bureaux à écart net, ` +
    `<b>${Math.round(lo.acc)} %</b> quand il est sous ${lo.hi} pt. En orange : le terrain serré.`;
}

function renderCoverage() {
  const cov = APP.data.coverage, blocs = ["G", "CD", "ED", "AB"];
  $("covbadge").innerHTML = [80, 90, 95].map((lvl) => {
    const emp = Math.min(...blocs.map((b) => cov[b][lvl]));
    return `<div class="cov-row"><span class="promise">on annonce ${lvl} %</span>` +
      `<div class="cov-track"><div class="cov-fill" style="width:${emp}%"></div></div>` +
      `<span class="v">≥ ${emp.toLocaleString("fr-FR", { minimumFractionDigits: 1 })} %</span></div>`;
  }).join("");
}

function renderPollGap() {
  const poll = APP.data.summary.flat_poll, ours = APP.data.summary.lead_accuracy;
  const row = (lab, val, cls) =>
    `<div class="pg-row ${cls}"><span class="pg-lab">${lab}</span>` +
    `<div class="pg-track"><div class="pg-fill" style="width:${val}%"></div>` +
    `<span class="pg-v">${val.toLocaleString("fr-FR", { minimumFractionDigits: 1 })} %</span></div></div>`;
  $("pollgap").innerHTML =
    row(`Un sondage national<small>même bloc (${APP.NAME[poll.bloc]}) partout</small>`, poll.accuracy, "poll") +
    row("Notre carte<small>bureau par bureau</small>", ours, "ours");
}

// The one defensible reservoir of convincible Left voters (mainland), in millions:
// abstainers who lean Left = abstainers × γ (marginal-voter Left share, MOVABILITY §11).
function renderPools() {
  const lg = APP.data.summary.left_gain;
  $("pools").innerHTML =
    `<div class="pool-row mob"><span class="pool-lab">Mobilisation` +
    `<small>abstentionnistes qui penchent à gauche · ${lg.gamma_mean} % en moyenne</small></span>` +
    `<b class="pool-v">${fmtM(lg.mobilization_voters)}</b><span class="pool-u">électeurs</span></div>`;
}

// Deployment order in INDIVIDUAL VOTERS (not bureaux), mainland only: communes
// ranked by mobilizable Left abstainers.
function renderDeployment() {
  const rows = APP.data.summary.left_gain.deployment;
  const max = Math.max(1, ...rows.map((r) => r.mob));
  $("deployment").innerHTML = rows
    .map(
      (r) =>
        `<div class="tc-row"><span class="tc-name">${r.nom}<small>${r.dept}</small></span>` +
        `<div class="tc-track" title="${fmt(r.mob)} abstentionnistes mobilisables">` +
        `<div class="tc-fill" style="width:${(r.mob / max) * 100}%"></div></div>` +
        `<span class="tc-v"><b>${fmt(r.mob)}</b> électeurs</span></div>`,
    )
    .join("");
}

function renderProvenance() {
  const p = APP.data.provenance.blocks;
  $("provbars").innerHTML = ["G", "CD", "ED", "AB"].map((b) => {
    const loc = p[b].local_share, nat = p[b].national_share;
    const lab = (v) => (v >= 13 ? Math.round(v) + " %" : "");
    return `<div class="prov-row"><span class="pl"><b style="color:${APP.COL[b]}">●</b> ${APP.NAME[b]}</span>` +
      `<div class="prov-bar" title="${APP.NAME[b]} : ${Math.round(loc)} % local · ${Math.round(nat)} % national">` +
      `<div class="prov-loc" style="width:${loc}%;background:${APP.COL[b]}">${lab(loc)}</div>` +
      `<div class="prov-nat" style="width:${nat}%">${lab(nat)}</div></div></div>`;
  }).join("");
}

