"use strict";

function baseStyle() {
  return {
    version: 8,
    glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
    sources: {
      carto: {
        type: "raster",
        tiles: ["https://a.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png",
          "https://b.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png"],
        tileSize: 256, attribution: "© OpenStreetMap · CARTO",
      },
      labels: {
        type: "raster",
        tiles: ["https://a.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png"],
        tileSize: 256,
      },
    },
    layers: [
      { id: "bg", type: "raster", source: "carto" },
    ],
  };
}

function initMap() {
  const map = new maplibregl.Map({
    container: "map", style: baseStyle(),
    center: APP.LYON.center, zoom: APP.LYON.zoom, maxZoom: 17, minZoom: 5,
    attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  APP.map = map;
  return new Promise((res) => map.on("load", () => { addLayers(); res(map); }));
}

function communeFC() {
  return {
    type: "FeatureCollection",
    features: APP.data.communes.map((c) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [c.lon, c.lat] },
      properties: { pG: c.pG, pCD: c.pCD, pED: c.pED, pAB: c.pAB,
        cmv: c.cmv, cab: c.cab, ccj: c.ccj, nb: c.n_bv,
        i: c.inscrits, n: c.nom, code: c.code_commune, dept: c.dept },
    })),
  };
}

function addLayers() {
  const map = APP.map;
  map.addSource("communes", { type: "geojson", data: communeFC() });
  map.addSource("bv", { type: "geojson", data: { type: "FeatureCollection", features: [] } });

  map.addLayer({
    id: "com-circ", type: "circle", source: "communes", maxzoom: 10,
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"],
        5, ["interpolate", ["linear"], ["get", "i"], 200, 1.5, 20000, 9],
        10, ["interpolate", ["linear"], ["get", "i"], 200, 4, 20000, 22]],
      "circle-color": voterColorExpr("cmv", 1000, 15000, 60000),
      "circle-opacity": 0.82, "circle-stroke-width": 0.3, "circle-stroke-color": "#fff",
    },
  });
  map.addLayer({
    id: "bv-fill", type: "fill", source: "bv", minzoom: 9,
    paint: { "fill-color": voterColorExpr("mv", 40, 150, 400), "fill-opacity": 0.78 },
  });
  map.addLayer({
    id: "bv-line", type: "line", source: "bv", minzoom: 11,
    paint: { "line-color": "#ffffff", "line-width": 0.4, "line-opacity": 0.5 },
  });
  map.addLayer({ id: "labels", type: "raster", source: "labels" });

  map.on("click", "bv-fill", (e) => openPanel(e.features[0].properties));
  map.on("click", "com-circ", (e) => zoomToCommune(e.features[0]));
  for (const ly of ["bv-fill", "com-circ"]) {
    map.on("mouseenter", ly, () => (map.getCanvas().style.cursor = "pointer"));
    // Hover only lives over a feature on the map: drop the popup on leave so it never
    // lingers over the page once the cursor moves off the map.
    map.on("mouseleave", ly, () => {
      map.getCanvas().style.cursor = "";
      if (popup) popup.remove();
    });
    map.on("mousemove", ly, (e) => hover(e));
  }
  map.on("moveend", autoLoadDept);
}

let popup = null;
function hover(e) {
  const p = e.features[0].properties;
  if (!popup) popup = new maplibregl.Popup({ closeButton: false, className: "mini" });
  popup.setLngLat(e.lngLat).setHTML(`<b>${hoverTitle(p)}</b>${hoverBody(p)}`).addTo(APP.map);
}

// Always name the geographic unit first: a polygon ("le shape") is a polling station,
// a circle is a whole commune. The earlier popup left this implicit — the client asked
// what the shape was. Bureau number is parsed from the location id (commune_num).
function hoverTitle(p) {
  const name = p.n || "commune";
  if (p.l !== undefined) return `Bureau de vote n°${+p.l.split("_")[1]} · ${name}`;
  return p.nb ? `${name} · ${fmt(p.nb)} bureaux de vote` : name;
}

// New Gauche share if a TARGETED GOTV effort brings out the `mv` left-leaning
// mobilizables (turnout grows by mv, all Left): (Gshare·voters + mv) / (voters + mv).
// This is the action the tool models — canvass YOUR voters — not a broad turnout
// surge where the whole frange returns (which, since γ < the bureau's Left share,
// would dilute it). The label states the targeting assumption so it isn't oversold.
function mobilizedScore(p, bv) {
  const abPct = bv ? p.pa : p.pAB, gShare = bv ? p.pg : p.pG;
  const mv = bv ? p.mv : p.cmv;
  const voters = p.i * (1 - abPct / 100);
  if (voters <= 0 || mv <= 0) return null;
  const cur = gShare, next = (gShare / 100 * voters + mv) / (voters + mv) * 100;
  return { cur, next };
}

// Hover text follows what the map is coloured by. In mobilization mode it explains the
// score itself — mobilizable = conjunctural abstainers × γ — then shows the resulting
// Left score if they all turn out, and the per-bureau SHAP reason (`w`) when available.
function hoverBody(p) {
  const bv = p.l !== undefined;
  if (APP.state.mode === "mobil") {
    const mv = bv ? p.mv : p.cmv, abs = bv ? p.ab : p.cab, conj = bv ? p.cj : p.ccj;
    let s = `<br><b>${fmt(mv)}</b> électeurs à aller chercher` +
      `<br><span class="mini-cap">la couleur = densité d'abstentionnistes qui, en venant voter, choisiraient la gauche</span>`;
    // γ = mobilizable ÷ CONJUNCTURAL abstainers (same as the click panel), never
    // mobilizable ÷ all abstainers — that conflated the conjunctural filter with γ
    // and printed a misleading lean-left share (e.g. "2 %" on a γ≈45 % bureau). The
    // share is only shown when the conjunctural base is large enough not to be pure
    // rounding noise (mv/cj on a handful of voters reads as 100 %/0 %).
    if (abs > 0) {
      s += `<br><span class="mini-sub">${fmt(abs)} abstentionnistes, dont ` +
        `<b>${fmt(conj)}</b> conjoncturels (qui reviennent quand l'enjeu monte)`;
      if (conj >= 10) s += ` — ${Math.round((mv / conj) * 100)} % pencheraient à gauche`;
      s += `</span>`;
    }
    const sc = mobilizedScore(p, bv);
    if (sc) {
      s += `<br><span class="mini-score">si votre campagne ramène ces électeurs de gauche : Gauche ` +
        `${sc.cur.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} % → ` +
        `<b>${sc.next.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} %</b></span>`;
    }
    if (bv && p.w) s += `<br><span class="mini-why">${p.w}</span>`;
    return s;
  }
  const lead = leadOf(p, bv);
  const fb = bv && p.fb ? `<br><span class="mini-fb">⚠︎ prédiction peu fiable (repli communal)</span>` : "";
  return `<br>${APP.NAME[lead]} en tête${p.m !== undefined ? " · marge " + p.m + " pts" : ""}${fb}`;
}

function leadOf(p, bv) {
  const g = bv ? p.pg : p.pG, c = bv ? p.pc : p.pCD, e = bv ? p.pe : p.pED;
  return g >= c && g >= e ? "G" : c >= e ? "CD" : "ED";
}

async function ensureDept(dept) {
  if (!dept || APP.bvByDept.has(dept)) return;
  try {
    const fc = await loadJSON(`data/bv/${dept}.geojson`);
    APP.bvByDept.set(dept, fc.features);
    if (APP.bvByDept.size > 10) APP.bvByDept.delete(APP.bvByDept.keys().next().value);
    const feats = [].concat(...APP.bvByDept.values());
    APP.map.getSource("bv").setData({ type: "FeatureCollection", features: feats });
  } catch (_) { /* dept sans contour */ }
}

function nearestDept(center) {
  let best = null, bd = Infinity;
  for (const c of APP.data.communes) {
    const dx = c.lon - center.lng, dy = c.lat - center.lat, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = c.dept; }
  }
  return best;
}

function autoLoadDept() {
  if (APP.map.getZoom() < 9) return;
  ensureDept(nearestDept(APP.map.getCenter()));
}

function zoomToCommune(f) {
  ensureDept(f.properties.dept);
  APP.map.flyTo({ center: f.geometry.coordinates, zoom: 13, speed: 1.4 });
}

function flyToCommune(c) {
  ensureDept(c.dept);
  APP.map.flyTo({ center: [c.lon, c.lat], zoom: 13.5, speed: 1.4 });
  $("map-hint").classList.add("gone");
}
