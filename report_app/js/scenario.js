"use strict";

function buildSliders() {
  const host = $("sliders");
  for (const b of APP.VOTE) {
    const row = document.createElement("div");
    row.className = "srow";
    row.innerHTML =
      `<label><b style="background:${APP.COL[b]}"></b>${APP.NAME[b]}</label>` +
      `<input type="range" min="-8" max="8" step="0.5" value="0" data-b="${b}">` +
      `<span class="delta" id="d-${b}">0,0 pt</span>`;
    host.appendChild(row);
  }
  host.querySelectorAll("input").forEach((el) =>
    el.addEventListener("input", () => {
      const b = el.dataset.b, v = parseFloat(el.value);
      APP.state["d" + (b === "CD" ? "C" : b === "ED" ? "E" : "G")] = v;
      $("d-" + b).textContent = (v >= 0 ? "+" : "") + v.toLocaleString("fr-FR", { minimumFractionDigits: 1 }) + " pt";
      applyColor();
      updateScenario();
    })
  );
}

function applyColor() {
  const map = APP.map, s = APP.state;
  if (s.mode === "mobil") {
    map.setPaintProperty("com-circ", "circle-color", voterColorExpr("cmv", 1000, 15000, 60000));
    map.setPaintProperty("bv-fill", "fill-color", voterColorExpr("mv", 40, 150, 400));
  } else if (s.mode === "honesty") {
    map.setPaintProperty("com-circ", "circle-color", "#c4c8cf");
    map.setPaintProperty("bv-fill", "fill-color",
      ["interpolate", ["linear"], ["get", "u"],
        18, "#eef0f2", 30, "#f3c9a8", 45, "#e07b39", 65, "#a83214"]);
  } else {
    map.setPaintProperty("com-circ", "circle-color", leadColorExpr({ G: "pG", CD: "pCD", ED: "pED" }));
    map.setPaintProperty("bv-fill", "fill-color", leadColorExpr({ G: "pg", CD: "pc", ED: "pe" }));
  }
  APP.map.setPaintProperty("bv-fill", "fill-opacity", 0.78);
}

function updateScenario() {
  const d = APP.data.national, base = APP.data.baseLead;
  const ad = appliedDeltas();
  let flips = 0, el = 0, knife = 0;
  const n = d.pg.length;
  for (let i = 0; i < n; i++) {
    const g = d.pg[i] + ad.G, c = d.pc[i] + ad.CD, e = d.pe[i] + ad.ED;
    const lead = g >= c && g >= e ? 0 : c >= e ? 1 : 2;
    if (lead !== base[i]) { flips++; el += d.ins[i]; }
    const sorted = [g, c, e].sort((x, y) => y - x);
    if (sorted[0] - sorted[1] < 0.7) knife++;
  }
  $("flip-count").textContent = fmt(flips);
  $("flip-el").textContent = fmt(el);
  $("flip-band").textContent = `≈ ${fmt(knife)} bureaux au seuil (fragiles)`;
  $("seat-flip").textContent = fmt(seatFlips(ad));
  drawSwingNote(ad);
  drawSeats();
  drawFil(ad);
  drawCurve(APP.state.dE);
}

// Make the conservation honest and visible: a national gain for one bloc is funded
// by the others; the three applied deltas always sum to ~0.
function drawSwingNote(ad) {
  const note = $("swing-note");
  if (!note) return;
  const moved = Math.abs(ad.G) + Math.abs(ad.CD) + Math.abs(ad.ED) > 0.05;
  if (!moved) {
    note.innerHTML = "Un seul cadran à somme nulle : un bloc gagne, les autres le " +
      "financent au prorata de leur taille. Tournez un curseur.";
    return;
  }
  const seg = (b) =>
    `<span><b style="background:${APP.COL[b]}"></b>${signed(ad[b])}</span>`;
  note.innerHTML = `<em>Effet net, à somme nulle&nbsp;:</em> ` +
    APP.VOTE.map(seg).join(" ");
}

function signed(v) {
  return (v >= 0 ? "+" : "") + v.toLocaleString("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
}

function circoFlipsAt(shift) {
  const ad = conservedDeltas({ G: 0, CD: 0, ED: shift });
  const base = APP.data.seatBase;
  let f = 0;
  for (let i = 0; i < base.length; i++) if (seatLead(i, ad) !== base[i]) f++;
  return f;
}

// Twin-axis response: how many bureaux (left, ink) and how many circonscriptions
// (right, ED violet) change lead block as the national ED shift sweeps −4…+6 pts.
// One dial, two units — the climax claim made literal.
function drawCurve(markShift) {
  const svg = $("flipcurve"), pts = APP.data.summary.flip_curve;
  const W = 360, H = 170, padL = 34, padR = 34, padB = 26, padT = 12;
  const xs = pts.map((p) => p.shift);
  const cir = pts.map((p) => circoFlipsAt(p.shift));
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const bMax = Math.max(...pts.map((p) => p.flips)) || 1;
  const cMax = Math.max(...cir) || 1;
  const X = (s) => padL + ((s - xmin) / (xmax - xmin)) * (W - padL - padR);
  const Yb = (f) => H - padB - (f / bMax) * (H - padB - padT);
  const Yc = (f) => H - padB - (f / cMax) * (H - padB - padT);
  const pathB = pts.map((p, i) => (i ? "L" : "M") + X(p.shift).toFixed(1) + " " + Yb(p.flips).toFixed(1)).join(" ");
  const pathC = pts.map((p, i) => (i ? "L" : "M") + X(p.shift).toFixed(1) + " " + Yc(cir[i]).toFixed(1)).join(" ");
  const k = pts.reduce((a, p, i) => Math.abs(p.shift - markShift) < Math.abs(pts[a].shift - markShift) ? i : a, 0);
  const np = pts[k], nc = cir[k];
  svg.innerHTML =
    `<line x1="${X(0)}" y1="${padT}" x2="${X(0)}" y2="${H - padB}" stroke="#e4e2da"/>` +
    `<text x="2" y="${padT + 3}" font-size="8" fill="#9a9aa2">${fmt(bMax)}</text>` +
    `<text x="2" y="${H - padB}" font-size="8" fill="#9a9aa2">0</text>` +
    `<text x="${W - padR + 4}" y="${padT + 3}" font-size="8" fill="${APP.COL.ED}">${fmt(cMax)}</text>` +
    `<path d="${pathC}" fill="none" stroke="${APP.COL.ED}" stroke-width="1.6" stroke-dasharray="4 3"/>` +
    `<path d="${pathB}" fill="none" stroke="#1A1A2E" stroke-width="2"/>` +
    `<circle cx="${X(np.shift)}" cy="${Yc(nc)}" r="3.5" fill="${APP.COL.ED}"/>` +
    `<circle cx="${X(np.shift)}" cy="${Yb(np.flips)}" r="4.5" fill="#E4572E"/>` +
    `<text x="${X(np.shift)}" y="${Yb(np.flips) - 8}" font-size="11" fill="#E4572E" text-anchor="middle">${fmt(np.flips)}</text>` +
    `<text x="${X(xmin)}" y="${H - 7}" font-size="9" fill="#9a9aa2">${xmin} pt ED</text>` +
    `<text x="${X(0)}" y="${H - 7}" font-size="9" fill="#9a9aa2" text-anchor="middle">0</text>` +
    `<text x="${X(xmax)}" y="${H - 7}" font-size="9" fill="#9a9aa2" text-anchor="end">+${xmax} pt ED</text>`;
}

function setMode(mode) { APP.state.mode = mode; applyColor(); updateLegend(); }

// The header legend mirrors what the map is currently coloured by — mobilizable
// voters by default, lead block or uncertainty when their layer is on.
function updateLegend() {
  const grad = (from, to, title) =>
    `<span class="legend-grad" title="${title || ""}" style="--gA:${from};--gB:${to}">`;
  const html = {
    mobil:
      `<span class="legend-lab">mobilisation</span>` +
      grad(APP.PALE.G, "#a83214",
        "score = abstentionnistes × γ (part de gauche du votant marginal)") +
      "peu → beaucoup d'électeurs gagnables</span>",
    lead:
      `<i data-b="G"></i>Gauche <i data-b="CD"></i>Centre+Droite <i data-b="ED"></i>Extrême&nbsp;Droite` +
      grad(APP.PALE.ED, APP.COL.ED, "pâle = marge serrée") + "pâle = serré</span>",
    honesty: grad("#f3c9a8", "#a83214", "largeur de l'intervalle conforme 90 %") +
      "intervalle étroit → large</span>",
  };
  $("legend").innerHTML = html[APP.state.mode];
}

const norm = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
function initSearch() {
  const input = $("search"), results = $("results");
  const idx = APP.data.communes.map((c) => ({ c, k: norm(c.nom) }));
  input.addEventListener("input", () => {
    const q = norm(input.value.trim());
    results.innerHTML = "";
    if (q.length < 2) return;
    idx.filter((o) => o.k.startsWith(q)).slice(0, 8).forEach((o) => {
      const li = document.createElement("li");
      li.innerHTML = `${o.c.nom}<small>${o.c.dept} · ${fmt(o.c.inscrits)} inscrits</small>`;
      li.onclick = () => { flyToCommune(o.c); results.innerHTML = ""; input.value = o.c.nom; };
      results.appendChild(li);
    });
  });
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search")) results.innerHTML = "";
  });
}
