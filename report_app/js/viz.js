"use strict";

// Stacked G/CD/ED bar for the "réalité partagée" counterpoint. `vals` are raw counts;
// segments show the count when wide enough.
function stackedBar(label, vals) {
  const tot = vals.reduce((a, b) => a + b, 0) || 1, blocs = ["G", "CD", "ED"];
  const segs = blocs.map((b, k) => {
    const pct = (vals[k] / tot) * 100;
    return `<div class="ls-seg" style="width:${pct}%;background:${APP.COL[b]}">${pct >= 9 ? fmt(vals[k]) : ""}</div>`;
  }).join("");
  const lab = label ? `<div class="lab">${label}</div>` : "";
  return `<div class="ls-row">${lab}<div class="ls-bar">${segs}</div></div>`;
}

// The counterpoint to "un sondage = un seul favori partout" : the real 2024 lead is
// split across three blocs, bureau by bureau. Uses the OBSERVED result (summary.
// observed_lead), not our prediction — that is the ground truth a poll cannot see, and
// what the 81,6 % is measured against.
function renderRealite() {
  const o = APP.data.summary.observed_lead;
  const cnt = ["G", "CD", "ED"].map((b) => o[b]);
  $("realite").innerHTML =
    `<div class="viz-cap">Bureaux où chaque bloc est arrivé en tête — résultat réel 2024` +
    `<span class="viz-cap-sub">sur 69 358</span></div>` +
    stackedBar("", cnt);
}

function renderPollGap() {
  const poll = APP.data.summary.flat_poll, ours = APP.data.summary.lead_accuracy;
  const row = (lab, val, cls) =>
    `<div class="pg-row ${cls}"><span class="pg-lab">${lab}</span>` +
    `<div class="pg-track"><div class="pg-fill" style="width:${val}%"></div>` +
    `<span class="pg-v">${val.toLocaleString("fr-FR", { minimumFractionDigits: 1 })} %</span></div></div>`;
  $("pollgap").innerHTML =
    `<div class="viz-cap">Part des bureaux où le bloc arrivé en tête est correctement désigné</div>` +
    row(`Un sondage national<small>même bloc (${APP.NAME[poll.bloc]}) partout</small>`, poll.accuracy, "poll") +
    row("Notre carte<small>bureau par bureau</small>", ours, "ours");
}

// The one defensible reservoir of convincible Left voters (mainland), in millions:
// abstainers who lean Left = abstainers × γ (marginal-voter Left share, MOVABILITY §11).
function renderPools() {
  const lg = APP.data.summary.left_gain;
  const M = (n) =>
    (n / 1e6).toLocaleString("fr-FR", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }) + " M";
  const pct = lg.gamma_mean.toLocaleString("fr-FR", { minimumFractionDigits: 1 });
  $("pools").innerHTML =
    `<div class="pool-row mob"><span class="pool-lab">Mobilisation` +
    `<small>électeurs de gauche à aller chercher</small></span>` +
    `<b class="pool-v">${M(lg.mobilization_voters)}</b><span class="pool-u">électeurs</span></div>` +
    `<div class="pool-calc">` +
    `<span class="pc-step"><b>${M(lg.total_abstainers)}</b> abstentionnistes prévus dans les bureaux</span>` +
    `<span class="op">on écarte l'abstention <b>de fond</b> — ceux qui ne votent jamais</span>` +
    `<span class="pc-step"><b>${M(lg.conjunctural_abstainers)}</b> <b>conjoncturels</b> : reviennent voter quand la participation monte</span>` +
    `<span class="op">× <b>${pct} %</b> d'entre eux penchent à gauche en moyenne — la part se lit bureau par bureau (voir la courbe), et ces abstentionnistes mobilisables se trouvent dans des bureaux un peu plus à gauche que l'ensemble</span>` +
    `<span class="pc-step total"><b>= ${M(lg.mobilization_voters)}</b> électeurs de gauche à aller chercher</span>` +
    `</div>`;
}

// Deployment in INDIVIDUAL VOTERS (mainland): communes ranked by mobilizable Left
// abstainers. Two columns — volume (mob, tracks city size) AND yield (γ, the model's
// own signal: what share of the conjunctural abstainers a campaign would convert).
function renderDeployment() {
  const rows = APP.data.summary.left_gain.deployment;
  const max = Math.max(1, ...rows.map((r) => r.mob));
  $("deployment").innerHTML = rows
    .map(
      (r) =>
        `<div class="tc-row"><span class="tc-name">${r.nom}<small>${r.dept}</small></span>` +
        `<div class="tc-track" title="${fmt(r.mob)} abstentionnistes mobilisables · rendement ${r.gamma} %">` +
        `<div class="tc-fill" style="width:${(r.mob / max) * 100}%"></div></div>` +
        `<span class="tc-v"><b>${fmt(r.mob)}</b> électeurs</span>` +
        `<span class="tc-g" title="part des abstentionnistes conjoncturels qui penchent à gauche">${r.gamma} %</span></div>`,
    )
    .join("");
}

// The γ graph the client asked for, named in plain words (no Greek on screen): who comes
// back to vote when turnout rises, and what share of them chooses the Left — one line per
// election type (legislative / European / presidential), because the marginal voter
// differs sharply by turnout regime. The reason "γ is not the same formula per election".
function renderGamma() {
  const g = APP.data.gamma;
  if (!g) return;
  const series = [
    { key: "Legislatives_T1", lab: "législatives", col: APP.COL.G, dash: "" },
    { key: "Europeennes_T1", lab: "européennes", col: "#3a9", dash: "5 3" },
    { key: "Presidentielle_T1", lab: "présidentielle", col: "#9a9aa2", dash: "2 3" },
  ].filter((s) => g[s.key]);
  const W = 360, H = 175, padL = 40, padR = 10, padB = 30, padT = 30;
  const all = series.flatMap((s) => g[s.key]);
  const xMin = Math.min(...all.map((p) => p[0])), xMax = Math.max(...all.map((p) => p[0]));
  const yMax = Math.max(60, ...all.map((p) => p[1]));
  const X = (v) => padL + ((v - xMin) / (xMax - xMin)) * (W - padL - padR);
  const Y = (v) => padT + (1 - v / yMax) * (H - padT - padB);
  const path = (pts) => pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join(" ");
  const dots = (pts, col) => pts.map((p) => `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="2.2" fill="${col}"/>`).join("");
  // In-chart legend across the top: a line sample (solid/dashed like the curve) + label,
  // so the curve↔scrutin mapping is read on the graph, not only in the caption.
  let lx = padL;
  const legend = series.map((s) => {
    const seg =
      `<line x1="${lx}" y1="12" x2="${lx + 16}" y2="12" stroke="${s.col}" stroke-width="2"` +
      `${s.dash ? ` stroke-dasharray="${s.dash}"` : ""}/>` +
      `<text x="${lx + 20}" y="15" font-size="8.5" fill="#5a5a64">${s.lab}</text>`;
    lx += 20 + s.lab.length * 4.7 + 18;
    return seg;
  }).join("");
  const midY = (padT + (H - padB)) / 2;
  $("gammacurve").innerHTML =
    legend +
    `<text x="11" y="${midY.toFixed(1)}" font-size="8" fill="#9a9aa2" ` +
    `transform="rotate(-90 11 ${midY.toFixed(1)})" text-anchor="middle">sur 100 revenants, combien votent à gauche</text>` +
    `<text x="22" y="${Y(yMax) + 3}" font-size="8" fill="#9a9aa2">${yMax}</text>` +
    `<text x="22" y="${Y(0)}" font-size="8" fill="#9a9aa2">0</text>` +
    series.map((s) =>
      `<path d="${path(g[s.key])}" fill="none" stroke="${s.col}" stroke-width="${s.dash ? 1.6 : 2}"` +
      `${s.dash ? ` stroke-dasharray="${s.dash}"` : ""}/>` + dots(g[s.key], s.col)).join("") +
    `<text x="${padL}" y="${H - 16}" font-size="8.5" fill="#9a9aa2">bureau plutôt à droite</text>` +
    `<text x="${W - padR}" y="${H - 16}" font-size="8.5" fill="#9a9aa2" text-anchor="end">plutôt à gauche</text>` +
    `<text x="${padL}" y="${H - 4}" font-size="8.5" fill="#9a9aa2">niveau de gauche du bureau →</text>`;
  $("gamma-cap").innerHTML =
    "<b>Lecture :</b> quand la participation monte, combien de voix <b>nettes</b> la gauche " +
    "capte sur 100 ramenées — un gain net (arrivées, départs et reports confondus), pas la " +
    "seule couleur des nouveaux venus. Ça dépend du scrutin — à une législative " +
    "~39 sur 100 <b>en moyenne</b>, et d'autant plus que le bureau penche déjà à gauche " +
    "(de ~24 dans les bureaux les plus à droite à ~56 dans les plus à gauche) ; à une " +
    "européenne ~24, à une présidentielle ~12. Comme les abstentionnistes qu'on peut " +
    "ramener se concentrent dans des bureaux plutôt à gauche, le gisement se cale sur " +
    "<b>~43 sur 100</b> — un peu au-dessus de la moyenne tous bureaux. Le gisement change " +
    "donc avec l'élection visée comme avec le profil du bureau.";
}

function renderProvenance() {
  const p = APP.data.provenance.blocks;
  $("provbars").innerHTML =
    `<div class="viz-cap">D'où vient l'incertitude restante, bloc par bloc` +
    `<span class="viz-cap-sub">part locale (notre apport) vs nationale (sondages)</span></div>` +
    ["G", "CD", "ED", "AB"].map((b) => {
    const loc = p[b].local_share, nat = p[b].national_share;
    const lab = (v) => (v >= 13 ? Math.round(v) + " %" : "");
    return `<div class="prov-row"><span class="pl"><b style="color:${APP.COL[b]}">●</b> ${APP.NAME[b]}</span>` +
      `<div class="prov-bar" title="${APP.NAME[b]} : ${Math.round(loc)} % local · ${Math.round(nat)} % national">` +
      `<div class="prov-loc" style="width:${loc}%;background:${APP.COL[b]}">${lab(loc)}</div>` +
      `<div class="prov-nat" style="width:${nat}%">${lab(nat)}</div></div></div>`;
  }).join("");
  // Bureau-level skill, free of the national poll error: R² with the TRUE national level
  // (oracle) barely beats R² with the poll (realistic) — the skill is local, not borrowed.
  const r2 = (b, k) => p[b][k].toLocaleString("fr-FR", { minimumFractionDigits: 2 });
  const cells = ["G", "CD", "ED", "AB"].map((b) =>
    `<span><b style="color:${APP.COL[b]}">${APP.NAME[b]}</b> ${r2(b, "r2_real")}</span>`).join("");
  $("provbars").innerHTML +=
    `<p class="provr2"><b>Et au bureau près ?</b> Part de la variabilité entre bureaux que le ` +
    `modèle explique (R²) : ${cells}. Donnez-lui le vrai national plutôt que le sondage, le R² ` +
    `bouge à peine (ED ${r2("ED", "r2_real")}→${r2("ED", "r2_oracle")}) — cette finesse est locale, ` +
    `pas empruntée au sondage. <span class="muted">(Pourquoi pas en contradiction avec les ` +
    `${Math.round(p.ED.national_share)} % « national » d'Extrême Droite ? L'erreur de sondage est ` +
    `un décalage <i>uniforme</i> : un gros morceau de l'incertitude d'un même bureau, mais presque ` +
    `rien de ce qui distingue les bureaux entre eux — donc du R².)</span></p>`;
}

