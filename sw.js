"use strict";
// Service worker du site chiffré (voir deploy/README.md). Le dépôt public ne contient que des
// fichiers chiffrés aux noms opaques (`_e/<nom>`) : ce worker intercepte chaque requête du site,
// va chercher le fichier chiffré correspondant, le déchiffre (AES-256-GCM, chemin en données
// associées) et le décompresse (gzip). Sans clé — personne ne s'est connecté sur ce navigateur —
// toute page rend l'écran de connexion. Rien de déchiffré n'est écrit sur le disque : le cache du
// navigateur ne garde que le chiffré.
const BASE = new URL("./", self.location).pathname;
const DB = "site-auth", STORE = "k";
const TYPES = { html: "text/html; charset=utf-8", js: "text/javascript; charset=utf-8",
  mjs: "text/javascript; charset=utf-8", css: "text/css; charset=utf-8",
  json: "application/json; charset=utf-8", geojson: "application/json; charset=utf-8",
  svg: "image/svg+xml", png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", webp: "image/webp",
  gif: "image/gif", ico: "image/x-icon", woff2: "font/woff2", woff: "font/woff", pdf: "application/pdf",
  // Markdown et CSV s'affichent comme texte au lieu de se télécharger.
  md: "text/plain; charset=utf-8", csv: "text/plain; charset=utf-8", txt: "text/plain; charset=utf-8" };
const utf8 = new TextEncoder();
let keys = null;

function idb(mode, run) {
  return new Promise((resolve, reject) => {
    const open = indexedDB.open(DB, 1);
    open.onupgradeneeded = () => open.result.createObjectStore(STORE);
    open.onerror = () => reject(open.error);
    open.onsuccess = () => {
      const tx = open.result.transaction(STORE, mode), req = run(tx.objectStore(STORE));
      tx.oncomplete = () => { open.result.close(); resolve(req && req.result); };
      tx.onerror = () => { open.result.close(); reject(tx.error); };
    };
  });
}
async function getKeys() {
  if (!keys) keys = await idb("readonly", (s) => s.get("keys")).catch(() => null) || null;
  return keys;
}
async function dropKeys() {
  keys = null;
  await idb("readwrite", (s) => s.delete("keys")).catch(() => null);
}
const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
const fileName = async (k, rel) => hex(await crypto.subtle.sign("HMAC", k.mac, utf8.encode("path:" + rel))).slice(0, 32);

// Clé d'une version antérieure du site (clé de contenu changée) : on l'oublie et on redemande la
// connexion. Vérifié sur les seules navigations, avec le petit fichier `_k.json`.
async function stale(k) {
  try {
    const r = await fetch(BASE + "_k.json", { cache: "no-cache" });
    return r.ok && (await r.json()).kid !== k.kid;
  } catch (e) { return false; }
}
const shell = () => fetch(BASE + "index.html", { cache: "no-cache" });

async function serve(req, rel, search) {
  const nav = req.mode === "navigate";
  let k = await getKeys();
  if (k && nav && await stale(k)) { await dropKeys(); k = null; }
  if (!k) return nav ? shell() : new Response("Connexion requise", { status: 401 });
  if (rel === "" || rel.endsWith("/")) rel += "index.html";
  let r = await fetch(BASE + "_e/" + await fileName(k, rel) + search);
  // « /2027 » sans barre finale : c'est un dossier, on redirige pour que ses liens relatifs marchent.
  if (r.status === 404 && nav && !/\.[^/]+$/.test(rel)) {
    const probe = await fetch(BASE + "_e/" + await fileName(k, rel + "/index.html"));
    if (probe.ok) { const u = new URL(req.url); u.pathname += "/"; return Response.redirect(u.href, 301); }
  }
  if (!r.ok) return new Response("Introuvable", { status: r.status === 404 ? 404 : 502 });
  const buf = await r.arrayBuffer();
  let plain;
  try {
    plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv: buf.slice(0, 12), additionalData: utf8.encode(rel) },
      k.enc, buf.slice(12));
  } catch (e) {
    if (nav) { await dropKeys(); return shell(); }
    return new Response("Déchiffrement impossible", { status: 500 });
  }
  const body = new Blob([plain]).stream().pipeThrough(new DecompressionStream("gzip"));
  const ext = (rel.match(/\.([^./]+)$/) || [])[1];
  return new Response(body, { status: 200, headers: { "Content-Type": TYPES[(ext || "").toLowerCase()] || "application/octet-stream" } });
}

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("message", (e) => {
  const t = e.data && e.data.type;
  if (t === "login") keys = null;             // relire la clé que la page vient d'enregistrer
  if (t === "logout") e.waitUntil(dropKeys().then(() => e.source && e.source.postMessage({ type: "logged-out" })));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== self.location.origin || !url.pathname.startsWith(BASE)) return;
  const rel = decodeURIComponent(url.pathname.slice(BASE.length));
  // Les seuls fichiers publics en clair : ce worker, l'écran de connexion, les blocs de clé.
  if (rel === "sw.js" || rel === "_k.json" || rel.startsWith("_e/")) return;
  e.respondWith(serve(e.request, rel, url.search));
});
