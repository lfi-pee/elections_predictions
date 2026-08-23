"use strict";
// Barre de contrôle : présélections de scénario, curseurs nationaux par parti, bascule de
// mode carte, et les deux barres dynamiques (sièges projetés, jouabilité). Tout changement
// appelle recomputeAll() → recalcul de la carte et des barres.

const BLOCKS = ["G", "CD", "ED", "AB"];
const MAJORITY = 289; // sièges pour la majorité absolue (577)

function currentScenario() {
  return APP.data.summary.scenarios.find((s) => s.key === APP.scenario);
}

function setScenario(key) {
  const s = APP.data.summary.scenarios.find((x) => x.key === key);
  if (!s) return;
  APP.scenario = key;
  APP.scnObj = s;
  APP.nat = { G: s.means.G, CD: s.means.CD, ED: s.means.ED, AB: s.means.AB };
  syncSliders();
  document.querySelectorAll(".scn-btn").forEach((b) =>
    b.classList.toggle("on", b.dataset.k === key));
  $("scn-desc").textContent = s.desc;
  recomputeAll();
}

// Réinitialise les curseurs aux valeurs **prédites** du scénario courant (ancrage sondages).
function resetSliders() {
  APP.nat = { ...APP.scnObj.means };
  syncSliders();
  recomputeAll();
}

function syncSliders() {
  for (const b of BLOCKS) {
    const sl = $("sl-" + b);
    if (sl) sl.value = APP.nat[b];
    const v = $("slv-" + b);
    if (v) v.textContent = fmt1(APP.nat[b]) + " %";
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
    (b.onclick = () => setScenario(b.dataset.k)));
  $("scn-desc").textContent = APP.scnObj.desc;

  // Curseurs nationaux par parti.
  const R = sm.slider_ranges;
  const names = { G: "Gauche", CD: "Centre+Droite", ED: "Extrême Droite", AB: "Abstention" };
  $("sliders").innerHTML = BLOCKS.map((b) => {
    const [lo, hi] = R[b];
    return `<div class="sl-row"><label style="color:${APP.COL[b]}">${names[b]}
      <span class="sl-v" id="slv-${b}">${fmt1(APP.nat[b])} %</span></label>
      <input type="range" id="sl-${b}" min="${lo}" max="${hi}" step="0.5" value="${APP.nat[b]}"
        style="--c:${APP.COL[b]}"></div>`;
  }).join("");
  for (const b of BLOCKS) {
    $("sl-" + b).addEventListener("input", (e) => {
      APP.nat[b] = parseFloat(e.target.value);
      $("slv-" + b).textContent = fmt1(APP.nat[b]) + " %";
      recomputeAll();
    });
  }
  if ($("reset")) $("reset").onclick = resetSliders;

  // Bascule de mode carte.
  $("mode-win").onclick = () => { setMode("win"); syncModeBtns(); };
  $("mode-seat").onclick = () => { setMode("seat"); syncModeBtns(); };
  syncModeBtns();
}

function syncModeBtns() {
  $("mode-win").classList.toggle("on", APP.state.mode === "win");
  $("mode-seat").classList.toggle("on", APP.state.mode === "seat");
}

// ── Barre dynamique des sièges (projection) ──
function updateSeatBar() {
  const t = seatTally(), tot = t.G + t.CD + t.ED || 1;
  const seg = (b) => `<div class="seat-seg" style="width:${(t[b] / tot) * 100}%;background:${APP.COL[b]}"
      title="${APP.NAME[b]} : ${t[b]} sièges">${t[b] >= 20 ? t[b] : ""}</div>`;
  $("seatbar").innerHTML = seg("G") + seg("CD") + seg("ED") +
    `<div class="seat-maj" style="left:${(MAJORITY / tot) * 100}%" title="majorité absolue : ${MAJORITY}"></div>`;
  const lead = t.G >= t.CD && t.G >= t.ED ? "G" : t.CD >= t.ED ? "CD" : "ED";
  const maj = Math.max(t.G, t.CD, t.ED) >= MAJORITY;
  $("seat-legend").innerHTML =
    ["G", "CD", "ED"].map((b) =>
      `<span class="sl-leg"><i style="background:${APP.COL[b]}"></i>${APP.NAME[b]} <b>${t[b]}</b></span>`).join("") +
    `<span class="seat-note">${maj ? `majorité absolue : <b style="color:${APP.COL[lead]}">${APP.NAME[lead]}</b>`
      : `aucune majorité absolue (${MAJORITY} requis) — <b style="color:${APP.COL[lead]}">${APP.NAME[lead]}</b> en tête`}</span>`;
}

function updateNatBar() {
  const n = APP.nat, s = (n.G + n.CD + n.ED);
  $("natsum").innerHTML =
    `bloc : G ${fmt1(n.G)} · C+D ${fmt1(n.CD)} · ED ${fmt1(n.ED)} ` +
    `<span class="muted">(somme ${fmt1(s)} %)</span> · abstention ${fmt1(n.AB)} %`;
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
