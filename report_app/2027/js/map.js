"use strict";

// Fond de carte : tuiles VECTORIELLES OpenFreeMap (données OpenStreetMap), rendues par
// MapLibre — le même fournisseur que l'atlas. Les tuiles raster CARTO qui servaient ici
// exigent désormais une clé d'API et tamponnent « API KEY REQUIRED » en travers des
// tuiles anonymes ; OpenFreeMap est sans clé ni quota. Un style vectoriel est de surcroît
// un JSON qu'on retouche : les libellés passent sur `name:fr` (voir NAME_FR), là où le
// raster arrivait avec ses noms cuits dedans et anglicisés — d'où le fond _nolabels et la
// couche de libellés séparée que ce changement rend inutiles.
const OFM = "https://tiles.openfreemap.org/styles/positron";

// Les tuiles OFM portent une centaine de champs `name:xx` et le style s'en tient à
// `name:latin` — le nom LOCAL, soit « España » en bord de carte.
const NAME_FR = ["coalesce", ["get", "name:fr"], ["get", "name:latin"], ["get", "name"]];

// MapLibre ne sait pas filtrer un style au chargement : on le retouche après coup.
// On ne touche QUE les couches dont le libellé lit déjà un nom — les écussons d'autoroute
// affichent un NUMÉRO (`["to-string",["get","ref"]]`), leur passer NAME_FR laisserait des
// cartouches VIDES. Le test porte sur l'expression, pas sur l'identifiant de couche.
function patchBasemap() {
  const map = APP.map;
  for (const l of map.getStyle().layers) {
    const tf = l.layout && l.layout["text-field"];
    if (tf && JSON.stringify(tf).includes('"name')) map.setLayoutProperty(l.id, "text-field", NAME_FR);
  }
}

// Cadre initial : métropole + encarts outre-mer/étranger (bloc compact à gauche) visibles.
// Bord est/sud volontairement dégagé pour que la Corse reste dans le cadre même après un
// re-layout (barre du haut / panneau qui grandissent → carte plus courte → recadrage nécessaire,
// sinon MapLibre garde center+zoom et rogne le bord sud, là où est la Corse).
function frameFrance() {
  if (APP.map) APP.map.fitBounds([[-14.8, 40.6], [10.2, 51.5]], { padding: 14, animate: false });
}

function initMap() {
  const map = new maplibregl.Map({
    container: "map", style: OFM,
    center: [-1.5, 46.6], zoom: 5, maxZoom: 12, minZoom: 3.5,
    attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  // Molette moins sensible (défaut 1/450) → réglage plus fin du niveau de zoom.
  map.scrollZoom.setWheelZoomRate(1 / 1100);
  map.scrollZoom.setZoomRate(1 / 300);
  APP.map = map;
  return new Promise((res) => map.on("load", () => {
    patchBasemap();
    addLayers();
    frameFrance();
    res(map);
  }));
}

function addLayers() {
  const map = APP.map;
  // Les libellés vivent désormais dans le MÊME style que la choroplèthe : au lieu d'empiler
  // une tuile raster de noms par-dessus tout, on glisse nos couches SOUS la première couche
  // de symboles du fond — villes et cours d'eau s'impriment ainsi par-dessus les aplats.
  const premierSymbole = (map.getStyle().layers.find((l) => l.type === "symbol") || {}).id;
  // Géométrie chargée UNE fois ; promoteId → chaque entité a un id stable (le code circo)
  // pour setFeatureState. La couleur lit l'état d'entité, mis à jour au curseur.
  map.addSource("circo", { type: "geojson", promoteId: "id", data: APP.data.circoGeo });
  map.addLayer({
    id: "circo-fill", type: "fill", source: "circo",
    paint: { "fill-color": winColorExpr(), "fill-opacity": 0.82 },
  }, premierSymbole);
  map.addLayer({
    id: "circo-line", type: "line", source: "circo",
    paint: { "line-color": "#0d0f14", "line-width": 0.4, "line-opacity": 0.55 },
  }, premierSymbole);
  // Liseré tireté sur les circos non publiables : le gris seul se confondrait avec une
  // circo dont l'état n'est pas encore posé. La liste vient des données (coverage.js).
  map.addLayer({
    id: "circo-nopub", type: "line", source: "circo",
    paint: { "line-color": "#6b7079", "line-width": 1.2, "line-dasharray": [2, 1.5] },
    filter: ["in", ["get", "id"], ["literal", covUnpublishableIds()]],
  }, premierSymbole);
  map.addLayer({
    id: "circo-sel", type: "line", source: "circo",
    paint: { "line-color": "#fff", "line-width": 2 },
    filter: ["==", "id", "___none___"],
  }, premierSymbole);
  addInsets(premierSymbole);

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
function addInsets(premierSymbole) {
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
    paint: { "line-color": "#8a8f98", "line-width": 0.8, "line-dasharray": [2, 2] } }, premierSymbole);
  map.addLayer({ id: "inset-label", type: "symbol", source: "inset-labels",
    layout: { "text-field": ["get", "label"], "text-size": 10, "text-anchor": "bottom-left",
      // La police vient du serveur de glyphes du style, désormais celui d'OpenFreeMap :
      // il sert les fontes Noto, et une pile inconnue (« Open Sans Regular ») ne renvoie
      // aucun glyphe — les libellés d'encarts disparaîtraient.
      "text-offset": [0.2, -0.2], "text-font": ["Noto Sans Regular"] },
    paint: { "text-color": "#33373f", "text-halo-color": "#ffffff", "text-halo-width": 1.4 } });
}

// Couleur des polygones selon le mode : jouabilité (score 1→5) ou vainqueur du siège.
// Les deux lisent l'état d'entité (feature-state) mis à jour au curseur.
function applyColor() {
  APP.map.setPaintProperty("circo-fill", "fill-color",
    APP.state.mode === "seat" ? seatColorExpr() : winColorExpr());
  APP.map.setPaintProperty("circo-fill", "fill-opacity", 0.82);
}

function setMode(mode) { APP.state.mode = mode; applyColor(); updateLegend(); }

let popup = null;
function showPopup(p, lngLat) {
  if (!popup) popup = new maplibregl.Popup({ closeButton: COARSE, closeOnClick: false, className: "mini" });
  // Recalcul léger (une circo) au scénario/curseur courant — les props ne portent que les dev.
  const r = circoEval(p);
  // Circo hors nomenclature (cf. coverage.js) : l'infobulle dit « non mesurée » plutôt que
  // d'annoncer un score que le panneau refuse ensuite d'afficher.
  const body = !covIsPublishable(p.id)
    ? `<br><b style="color:#7a7f88">${APP.COV_LAB}</b> — nomenclature de blocs incomplète`
    : APP.state.mode === "seat"
      ? `<br>Siège probable : <b style="color:${APP.COL[r.win]}">${APP.NAME[r.win]}</b>`
      : `<br>Gauche : <b style="color:${APP.WIN[r.sc]}">${APP.WIN_LAB[r.sc]}</b>`;
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
