"use strict";

function baseStyle() {
  return {
    version: 8,
    glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
    sources: {
      carto: {
        type: "raster",
        tiles: ["https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
          "https://b.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png"],
        tileSize: 256, attribution: "© OpenStreetMap · CARTO",
      },
      labels: {
        type: "raster",
        tiles: ["https://a.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png"],
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
        cmv: c.cmv, cab: c.cab,
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
  map.on("click", "com-circ", (e) => zoomToCommune(e.features[0].properties));
  for (const ly of ["bv-fill", "com-circ"]) {
    map.on("mouseenter", ly, () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", ly, () => (map.getCanvas().style.cursor = ""));
    map.on("mousemove", ly, (e) => hover(e));
  }
  map.on("moveend", autoLoadDept);
}

let popup = null;
function hover(e) {
  const p = e.features[0].properties;
  if (!popup) popup = new maplibregl.Popup({ closeButton: false, className: "mini" });
  const name = p.n || "commune";
  popup.setLngLat(e.lngLat).setHTML(`<b>${name}</b><br>${hoverBody(p)}`).addTo(APP.map);
}

// Hover text follows what the map is coloured by. In mobilization mode it explains
// the score itself — mv = abstainers × γ (marginal-voter Left share) — and appends
// the per-bureau SHAP reason (`w`) when available, so the legend isn't a party score.
function hoverBody(p) {
  const bv = p.l !== undefined;
  if (APP.state.mode === "mobil") {
    const mv = bv ? p.mv : p.cmv, abs = bv ? p.ab : p.cab;
    let s = `<b>${fmt(mv)}</b> électeurs mobilisables`;
    if (abs > 0) {
      s += `<br><span class="mini-sub">${fmt(abs)} abstentionnistes × ` +
        `${Math.round((mv / abs) * 100)} % de gauche (γ)</span>`;
    }
    if (bv && p.w) s += `<br><span class="mini-why">${p.w}</span>`;
    return s;
  }
  if (APP.state.mode === "honesty") {
    return bv ? `intervalle conforme ±${p.u} pts (90 %)` : "zoomez pour l'incertitude au bureau";
  }
  const lead = leadOf(p, bv);
  return `${APP.NAME[lead]} en tête${p.m !== undefined ? " · marge " + p.m + " pts" : ""}`;
}

function leadOf(p, bv) {
  const s = APP.state;
  const g = (bv ? p.pg : p.pG) + s.dG, c = (bv ? p.pc : p.pCD) + s.dC, e = (bv ? p.pe : p.pED) + s.dE;
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

function zoomToCommune(p) {
  ensureDept(p.dept);
  APP.map.flyTo({ center: APP.map.getCenter(), zoom: 13 });
}

function flyToCommune(c) {
  ensureDept(c.dept);
  APP.map.flyTo({ center: [c.lon, c.lat], zoom: 13.5, speed: 1.4 });
  $("map-hint").classList.add("gone");
}
