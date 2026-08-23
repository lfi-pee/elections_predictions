"use strict";

function baseStyle() {
  return {
    version: 8,
    glyphs: "https://fonts.openmaptiles.org/{fontstack}/{range}.pbf",
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
    layers: [{ id: "bg", type: "raster", source: "carto" }],
  };
}

function initMap() {
  const map = new maplibregl.Map({
    container: "map", style: baseStyle(),
    center: [-1.5, 46.6], zoom: 5, maxZoom: 12, minZoom: 3.5,
    attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  APP.map = map;
  return new Promise((res) => map.on("load", () => {
    addLayers();
    // Cadre initial : métropole + encarts outre-mer/étranger (à gauche) tous visibles.
    map.fitBounds([[-15, 41], [10, 51.6]], { padding: 24, animate: false });
    res(map);
  }));
}

function addLayers() {
  const map = APP.map;
  map.addSource("circo", { type: "geojson", data: circoFC() });
  map.addLayer({
    id: "circo-fill", type: "fill", source: "circo",
    paint: { "fill-color": winColorExpr(), "fill-opacity": 0.72 },
  });
  map.addLayer({
    id: "circo-line", type: "line", source: "circo",
    paint: { "line-color": "#0d0f14", "line-width": 0.4, "line-opacity": 0.55 },
  });
  map.addLayer({
    id: "circo-sel", type: "line", source: "circo",
    paint: { "line-color": "#fff", "line-width": 2 },
    filter: ["==", "id", "___none___"],
  });
  addInsets();
  map.addLayer({ id: "labels", type: "raster", source: "labels" });

  map.on("click", "circo-fill", (e) => openCirco(e.features[0].properties));
  if (!COARSE) {
    map.on("mouseenter", "circo-fill", () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", "circo-fill", () => {
      map.getCanvas().style.cursor = "";
      if (popup) popup.remove();
    });
    map.on("mousemove", "circo-fill", hover);
  }
}

const COARSE = typeof matchMedia === "function" && matchMedia("(pointer:coarse)").matches;

// Cadres + libellés des encarts outre-mer / étranger (ramenés à gauche de la métropole).
function addInsets() {
  const map = APP.map, insets = APP.data.insets || [];
  const frames = insets.map((o) => {
    const [x0, y0, x1, y1] = o.box;
    return { type: "Feature", properties: { label: o.label },
      geometry: { type: "LineString",
        coordinates: [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]] } };
  });
  const labels = insets.map((o) => {
    const [x0, y0, x1, y1] = o.box;
    return { type: "Feature", properties: { label: o.label },
      geometry: { type: "Point", coordinates: [x0, y1] } };
  });
  map.addSource("inset-frames", { type: "geojson", data: { type: "FeatureCollection", features: frames } });
  map.addSource("inset-labels", { type: "geojson", data: { type: "FeatureCollection", features: labels } });
  map.addLayer({ id: "inset-frame", type: "line", source: "inset-frames",
    paint: { "line-color": "#8a8f98", "line-width": 0.8, "line-dasharray": [2, 2] } });
  map.addLayer({ id: "inset-label", type: "symbol", source: "inset-labels",
    layout: { "text-field": ["get", "label"], "text-size": 10, "text-anchor": "bottom-left",
      "text-offset": [0.2, -0.2], "text-font": ["Open Sans Regular"] },
    paint: { "text-color": "#33373f", "text-halo-color": "#ffffff", "text-halo-width": 1.4 } });
}

// Couleur des polygones selon le mode : jouabilité (score 1→5) ou vainqueur du siège.
function applyColor() {
  const expr = APP.state.mode === "seat"
    ? ["match", ["get", "win"], "G", APP.COL.G, "CD", APP.COL.CD, "ED", APP.COL.ED, "#888"]
    : winColorExpr();
  APP.map.setPaintProperty("circo-fill", "fill-color", expr);
  APP.map.setPaintProperty("circo-fill", "fill-opacity", APP.state.mode === "seat" ? 0.8 : 0.72);
}

function setMode(mode) { APP.state.mode = mode; applyColor(); updateLegend(); }

let popup = null;
function showPopup(p, lngLat) {
  if (!popup) popup = new maplibregl.Popup({ closeButton: COARSE, closeOnClick: false, className: "mini" });
  const w = APP.NAME[p.win];
  const body = APP.state.mode === "seat"
    ? `<br>Siège probable : <b style="color:${APP.COL[p.win]}">${w}</b>`
    : `<br>Gauche : <b>${APP.WIN_LAB[p.sc]}</b>`;
  popup.setLngLat(lngLat).setHTML(`<b>${p.id} · ${p.nm}</b>${body}`).addTo(APP.map);
}
function hover(e) { showPopup(e.features[0].properties, e.lngLat); }

// Recherche par nom de circo / commune-ancre / département.
const norm = (s) => (s || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
function initSearch() {
  const input = $("search"), results = $("results");
  const idx = APP.data.circoGeo.features.map((f) => ({
    p: f.properties, k: norm(f.properties.nm) + " " + f.properties.id,
    c: centroid(f.geometry),
  }));
  input.addEventListener("input", () => {
    const q = norm(input.value.trim());
    results.innerHTML = "";
    if (q.length < 2) return;
    idx.filter((o) => o.k.includes(q)).slice(0, 8).forEach((o) => {
      const li = document.createElement("li");
      li.innerHTML = `${o.p.id} · ${o.p.nm}<small>${o.p.dept} · ${fmt(o.p.ins)} inscrits</small>`;
      li.onclick = () => {
        results.innerHTML = ""; input.value = o.p.nm;
        if (o.c) APP.map.flyTo({ center: o.c, zoom: 8, speed: 1.4 });
        openCirco(o.p);
      };
      results.appendChild(li);
    });
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search")) results.innerHTML = ""; });
}

function centroid(g) {
  const rings = g.type === "Polygon" ? g.coordinates : [].concat(...(g.coordinates || []));
  let sx = 0, sy = 0, n = 0;
  for (const ring of rings || []) for (const c of ring) { sx += c[0]; sy += c[1]; n++; }
  return n ? [sx / n, sy / n] : null;
}
