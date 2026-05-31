"use strict";

function applyColor() {
  const map = APP.map, s = APP.state;
  if (s.mode === "mobil") {
    map.setPaintProperty("com-circ", "circle-color", voterColorExpr("cmv", 1000, 15000, 60000));
    map.setPaintProperty("bv-fill", "fill-color", voterColorExpr("mv", 40, 150, 400));
  } else {
    map.setPaintProperty("com-circ", "circle-color", leadColorExpr({ G: "pG", CD: "pCD", ED: "pED" }));
    map.setPaintProperty("bv-fill", "fill-color", leadColorExpr({ G: "pg", CD: "pc", ED: "pe" }));
  }
  APP.map.setPaintProperty("bv-fill", "fill-opacity", 0.78);
}

function setMode(mode) { APP.state.mode = mode; applyColor(); updateLegend(); }

// The header legend mirrors what the map is currently coloured by — mobilizable
// voters by default, lead block when that layer is on.
function updateLegend() {
  const grad = (from, to, title) =>
    `<span class="legend-grad" title="${title || ""}" style="--gA:${from};--gB:${to}">`;
  const html = {
    mobil:
      `<span class="legend-lab">mobilisation</span>` +
      grad(APP.PALE.G, "#a83214",
        "nombre d'abstentionnistes qui, en venant voter, choisiraient la gauche") +
      "peu → beaucoup d'électeurs gagnables</span>",
    lead:
      `<i data-b="G"></i>Gauche <i data-b="CD"></i>Centre+Droite <i data-b="ED"></i>Extrême&nbsp;Droite` +
      grad(APP.PALE.ED, APP.COL.ED, "pâle = marge serrée") + "pâle = serré</span>",
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
