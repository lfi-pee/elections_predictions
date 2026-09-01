"use strict";

// Basemap: OpenFreeMap VECTOR tiles (OpenStreetMap data) rendered by MapLibre — the same
// provider as the atlas. Three reasons it replaced the CARTO raster fond:
//   1. it is FREE, with neither key nor quota. CARTO now requires an API key and stamps
//      "API KEY REQUIRED" across anonymous tiles — that is what broke this map;
//   2. the LABELS. A vector style is a JSON we can retouch, so every name is forced onto
//      `name:fr` (see NAME_FR). Raster tiles arrive with their names baked in, and
//      anglicised ("New Aquitania" from zoom 8) — hence the old _nolabels fond plus a
//      separate labels raster stacked on top, a workaround this makes pointless;
//   3. the ZOOM. Tiles stop at level 14 and MapLibre keeps redrawing past it: going down
//      to the street or the bureau costs NO request.
// The flavour follows the page theme (see setBasemapTheme, called from theme.js).
const OFM = (t) => `https://tiles.openfreemap.org/styles/${t === "light" ? "positron" : "dark"}`;

function mapTheme() {
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}

// OFM tiles carry ~a hundred `name:xx` fields and the styles settle for `name:latin` —
// the LOCAL name, i.e. "España" and "Bay of Biscay" along the border.
const NAME_FR = ["coalesce", ["get", "name:fr"], ["get", "name:latin"], ["get", "name"]];

// The colour parser is the BROWSER: `fillStyle` takes everything CSS and MapLibre take
// (#rgb, rgb(), hsl()) and hands it back normalised, where a hand-rolled regex would
// cover one notation in three. The sentinel catches what is NOT a colour — `fillStyle`
// then keeps its previous value, and a MapLibre expression is full of strings that are
// no colour at all ("interpolate", "zoom"…).
const SENTINEL = "#fedcba";
const _c2d = document.createElement("canvas").getContext("2d");
function rgbaOf(color) {
  _c2d.fillStyle = SENTINEL; _c2d.fillStyle = color;
  const s = _c2d.fillStyle;
  if (s === SENTINEL && color !== SENTINEL) return null;
  if (s[0] === "#") {
    return [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16), parseInt(s.slice(5, 7), 16), 1];
  }
  const m = (s.match(/[\d.]+/g) || []).map(Number);
  return m.length >= 3 ? [m[0], m[1], m[2], m[3] != null ? m[3] : 1] : null;
}

// LIFTING THE DARK STYLE. Where positron really steps the light theme apart (land 242,
// water 194), OpenFreeMap's `dark` is crushed onto black: land rgb(12,12,12), water
// rgb(27,27,29). Fifteen levels separate the river from the ground, and on that part of
// the curve the eye reads almost nothing — under a 0.78 fill the basemap simply
// DISAPPEARED (no Rhône, no urban fabric, no coastline to get one's bearings). The
// hierarchy IS there, only packed against zero: a gamma redeploys it. One rule over every
// colour of the style, not a layer-by-layer table that would rot on the next upstream
// edit — the dark style has no authored hue anyway, it is grey throughout.
const LIFT = 0.62; // c' = 255·(c/255)^LIFT : 12 → 39, 27 → 65, 60 → 106
function lift(color) {
  const v = rgbaOf(color);
  if (!v) return color; // "interpolate", "zoom"… : left as is
  const f = (x) => Math.round(255 * Math.pow(x / 255, LIFT));
  return `rgba(${f(v[0])},${f(v[1])},${f(v[2])},${v[3]})`;
}
// A colour property can be an expression: walk into it, colours are the leaves
// (["interpolate",["linear"],["zoom"],5.8,"hsla(0,0%,85%,.53)",6,"#000"]).
const liftVal = (v) => (Array.isArray(v) ? v.map(liftVal) : typeof v === "string" ? lift(v) : v);

// MapLibre cannot filter a style at load time, so the retouch happens after the fact, on
// every `style.load` — hence also on every theme swap, which re-issues setStyle.
function patchBasemap() {
  const map = APP.map, dark = mapTheme() !== "light";
  for (const l of map.getStyle().layers) {
    if (dark) {
      for (const k in l.paint || {}) {
        if (k.includes("color")) map.setPaintProperty(l.id, k, liftVal(l.paint[k]));
      }
    }
    // Only layers whose label already reads a NAME are retouched. Motorway shields show a
    // NUMBER (`["to-string",["get","ref"]]`): handing them NAME_FR left EMPTY cartouches
    // on the map. The test is on the expression, not on the layer id, which changes from
    // one style to the other.
    const tf = l.layout && l.layout["text-field"];
    if (tf && JSON.stringify(tf).includes('"name')) map.setLayoutProperty(l.id, "text-field", NAME_FR);
  }
}

// A theme swap is a whole new vector style. Left to itself MapLibre computes a DIFF
// against the current style and then never emits `style.load` — the retouch above would
// not replay and our own layers would be reordered under a duplicated basemap. A clean
// reload costs a few dozen kB of JSON here, the tiles themselves being already cached;
// `style.load` re-adds sources and layers on the other side.
function setBasemapTheme(theme) {
  if (APP.map) APP.map.setStyle(OFM(theme === "light" ? "light" : "dark"), { diff: false });
}

function initMap() {
  const map = new maplibregl.Map({
    container: "map", style: OFM(mapTheme()),
    center: APP.LYON.center, zoom: APP.LYON.zoom, maxZoom: 17, minZoom: 5,
    attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  APP.map = map;
  // `style.load` and not `load`: it replays on every setStyle, where `load` fires once —
  // and a setStyle wipes our sources and layers along with the basemap's.
  map.on("style.load", () => { patchBasemap(); addLayers(); });
  wireMapEvents(map);
  return new Promise((res) => map.on("load", () => res(map)));
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

// Feature outlines: white reads on the dark basemap but vanishes on the light one, so
// bureaux would blend into an unreadable blob. Pick a dark hairline for the light theme.
const OUTLINE = { dark: "#ffffff", light: "#3a3a44" };

// Runs on every `style.load`, i.e. once at boot and again after each theme swap: sources
// and layers belong to the style that setStyle threw away, so they are rebuilt here from
// the state kept in APP (loaded departments, current colouring mode).
function addLayers() {
  const map = APP.map;
  const outline = OUTLINE[mapTheme()];
  const bvFeats = [].concat(...APP.bvByDept.values());
  map.addSource("communes", { type: "geojson", data: communeFC() });
  map.addSource("bv", { type: "geojson", data: { type: "FeatureCollection", features: bvFeats } });

  // Labels now live in the SAME style as the choropleth, so rather than stacking a raster
  // labels tile over everything, our layers slip UNDER the basemap's first symbol layer:
  // street and commune names keep printing over the fill, which is what lets the fill stay
  // opaque without losing the ground underneath.
  const firstSymbol = (map.getStyle().layers.find((l) => l.type === "symbol") || {}).id;

  map.addLayer({
    id: "com-circ", type: "circle", source: "communes", maxzoom: 10,
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"],
        5, ["interpolate", ["linear"], ["get", "i"], 200, 1.5, 20000, 9],
        10, ["interpolate", ["linear"], ["get", "i"], 200, 4, 20000, 22]],
      "circle-color": voterColorExpr("cmv", 1000, 15000, 60000),
      "circle-opacity": 0.82, "circle-stroke-width": 0.3, "circle-stroke-color": outline,
    },
  }, firstSymbol);
  map.addLayer({
    id: "bv-fill", type: "fill", source: "bv", minzoom: 9,
    paint: { "fill-color": voterColorExpr("mv", 40, 150, 400), "fill-opacity": 0.78 },
  }, firstSymbol);
  map.addLayer({
    id: "bv-line", type: "line", source: "bv", minzoom: 11,
    paint: { "line-color": outline, "line-width": 0.4, "line-opacity": 0.5 },
  }, firstSymbol);
  // The paint above is the mobilisation default; re-assert whatever mode is actually on,
  // since a theme swap rebuilds these layers while the user may sit on the lead layer.
  if (typeof applyColor === "function") applyColor();
}

// Map interactions are wired ONCE, on the map and not on the style: unlike sources and
// layers, handlers survive a setStyle, and re-registering them per style.load would fire
// the popup twice after a theme swap.
function wireMapEvents(map) {
  // One map-level tap handler instead of two layer-scoped ones: a layer-scoped click
  // only fires on an exact pixel hit, which a finger almost never lands (see pickNear),
  // and it gives no way to react to a tap that hits nothing.
  map.on("click", onTap);
  if (!COARSE) {
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
  }
  map.on("moveend", autoLoadDept);
}

// A finger is not a cursor. Touch devices have no hover at all (so the quick-read card
// was unreachable) and land ~10 mm wide, so an exact-pixel hit test made most taps do
// nothing — small bureaux, and commune circles that are ~2 px across when dezoomed.
const COARSE = typeof matchMedia === "function" && matchMedia("(pointer:coarse)").matches;
const TAP_PAD = COARSE ? 18 : 3;

// Rough screen-space anchor of a feature, used only to break ties between candidates
// caught by the padded box — nearest one to the finger wins.
function anchorOf(f) {
  const g = f.geometry;
  if (g.type === "Point") return g.coordinates;
  const rings = g.type === "Polygon" ? g.coordinates : [].concat(...(g.coordinates || []));
  let sx = 0, sy = 0, n = 0;
  for (const ring of rings || []) for (const c of ring) { sx += c[0]; sy += c[1]; n++; }
  return n ? [sx / n, sy / n] : null;
}

function pickNear(pt, layer) {
  const map = APP.map;
  if (!map.getLayer(layer)) return null;
  const exact = map.queryRenderedFeatures(pt, { layers: [layer] })[0];
  if (exact) return exact;
  const box = [[pt.x - TAP_PAD, pt.y - TAP_PAD], [pt.x + TAP_PAD, pt.y + TAP_PAD]];
  const near = map.queryRenderedFeatures(box, { layers: [layer] });
  if (!near.length) return null;
  let best = near[0], bd = Infinity;
  for (const f of near) {
    const a = anchorOf(f);
    if (!a) continue;
    const q = map.project(a), d = (q.x - pt.x) ** 2 + (q.y - pt.y) ** 2;
    if (d < bd) { bd = d; best = f; }
  }
  return best;
}

// Tap/click routing. A bureau opens the instrument panel — the full read, superset of
// the hover card. A commune circle has no panel, so on touch the popup IS its only
// reading; it is shown anchored on the commune and the map flies there. A tap on empty
// map dismisses the popup.
function onTap(e) {
  const bv = pickNear(e.point, "bv-fill");
  if (bv) {
    // On touch the card would sit under the panel with no way to close it; on desktop the
    // hover keeps owning it, so leave it alone there.
    if (COARSE && popup) popup.remove();
    openPanel(bv.properties, e.lngLat);
    return;
  }
  const com = pickNear(e.point, "com-circ");
  if (com) {
    if (COARSE) showPopup(com.properties, com.geometry.coordinates);
    zoomToCommune(com);
    return;
  }
  if (popup) popup.remove();
}

let popup = null;
function showPopup(p, lngLat) {
  if (!popup) {
    // closeOnClick defaults to true, so on touch the synthetic mousemove opened the card
    // and the very same tap's click closed it again — the popup never survived a tap.
    popup = new maplibregl.Popup({ closeButton: COARSE, closeOnClick: false, className: "mini" });
  }
  popup.setLngLat(lngLat).setHTML(`<b>${hoverTitle(p)}</b>${hoverBody(p)}`).addTo(APP.map);
}

function hover(e) {
  showPopup(e.features[0].properties, e.lngLat);
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
