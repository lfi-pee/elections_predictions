"use strict";
// Service worker du site chiffré (voir deploy/README.md). Le dépôt public ne contient que des
// fichiers chiffrés aux noms opaques (`_e/<nom>`) : ce worker intercepte chaque requête du site,
// va chercher le fichier chiffré correspondant, le déchiffre (AES-256-GCM, chemin en données
// associées) et le décompresse (gzip). Sans clé — personne ne s'est connecté sur ce navigateur —
// toute page rend l'écran de connexion. Rien de déchiffré n'est écrit sur le disque : le cache du
// navigateur ne garde que le chiffré. Le déchiffrement lui-même est dans site_core.js, recopié ici :
// Cœur commun du site chiffré (voir deploy/README.md) : trouver, déchiffrer et décompresser un
// fichier du site. encrypt_site.mjs le recopie dans l'écran de connexion et dans le service
// worker ; l'écran le recopie aussi dans le worker des scénarios quand il n'y a pas de service
// worker. La fonction ne dépend donc de rien d'extérieur : `fetchFn` est le fetch d'origine.
function siteCore(BASE, fetchFn) {
  "use strict";
  const utf8 = new TextEncoder();
  const TYPES = { html: "text/html; charset=utf-8", js: "text/javascript; charset=utf-8",
    mjs: "text/javascript; charset=utf-8", css: "text/css; charset=utf-8",
    json: "application/json; charset=utf-8", geojson: "application/json; charset=utf-8",
    svg: "image/svg+xml", png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", webp: "image/webp",
    gif: "image/gif", ico: "image/x-icon", woff2: "font/woff2", woff: "font/woff", pdf: "application/pdf",
    // Markdown et CSV s'affichent comme texte au lieu de se télécharger.
    md: "text/plain; charset=utf-8", csv: "text/plain; charset=utf-8", txt: "text/plain; charset=utf-8" };
  const type = (rel) => TYPES[((rel.match(/\.([^./]+)$/) || [])[1] || "").toLowerCase()] || "application/octet-stream";
  const hex = (buf) => Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, "0")).join("");

  // Chemin d'un fichier du site (relatif à BASE) pour une adresse, ou null : autre site, ou l'un
  // des seuls fichiers publics en clair (ce worker, les blocs de clé, le chiffré).
  function rel(url) {
    if (url.origin !== location.origin || !url.pathname.startsWith(BASE)) return null;
    let r = decodeURIComponent(url.pathname.slice(BASE.length));
    if (r === "sw.js" || r === "_k.json" || r.startsWith("_e/")) return null;
    if (r === "" || r.endsWith("/")) r += "index.html";
    return r;
  }
  // Clés : objets CryptoKey, ou les 64 octets bruts (rangés ainsi quand le navigateur ne sait pas
  // ranger une CryptoKey, ou pour les passer au worker des scénarios sans service worker).
  async function keys(rec) {
    if (!rec || !rec.raw) return rec;
    const raw = new Uint8Array(rec.raw);
    return { raw, kid: rec.kid,
      enc: await crypto.subtle.importKey("raw", raw.slice(0, 32), "AES-GCM", false, ["decrypt"]),
      mac: await crypto.subtle.importKey("raw", raw.slice(32, 64), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]) };
  }
  const fileName = async (k, r) => hex(await crypto.subtle.sign("HMAC", k.mac, utf8.encode("path:" + r))).slice(0, 32);

  // Un fichier : { status: 200, gz } (gzip déchiffré), { status: 404 | 5xx }, ou { status: "bad" }
  // si la clé ne l'ouvre pas (clé d'une autre version du site).
  async function load(k, r, search, init) {
    const res = await fetchFn(BASE + "_e/" + await fileName(k, r) + (search || ""), init);
    if (!res.ok) return { status: res.status };
    const buf = await res.arrayBuffer();
    try {
      return { status: 200, gz: await crypto.subtle.decrypt({ name: "AES-GCM", iv: buf.slice(0, 12), additionalData: utf8.encode(r) },
        k.enc, buf.slice(12)) };
    } catch (e) { return { status: "bad" }; }
  }

  // gzip : DecompressionStream quand le navigateur l'a, sinon ce décodeur (iOS avant 16.4…).
  const LB = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
  const LX = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
  const DB = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073,
    4097, 6145, 8193, 12289, 16385, 24577];
  const DX = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
  const ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
  // Table de décodage : indexée par les `max` prochains bits, donne (symbole << 4) | longueur.
  function huff(lens) {
    let max = 0;
    for (const l of lens) if (l > max) max = l;
    const count = new Uint16Array(16), next = new Uint16Array(16), t = new Uint16Array(1 << max);
    for (const l of lens) count[l]++;
    count[0] = 0;
    for (let b = 1, code = 0; b <= 15; b++) next[b] = code = (code + count[b - 1]) << 1;
    for (let s = 0; s < lens.length; s++) {
      const l = lens[s];
      if (!l) continue;
      let c = next[l]++, r = 0;
      for (let i = 0; i < l; i++) { r = (r << 1) | (c & 1); c >>= 1; }
      for (let j = r; j < t.length; j += 1 << l) t[j] = (s << 4) | l;
    }
    return { t, mask: (1 << max) - 1 };
  }
  let fixed = null;
  function inflate(src, out) {
    let bp = 0, op = 0, last = 0;
    const peek = () => { const i = bp >>> 3; return (src[i] | src[i + 1] << 8 | src[i + 2] << 16) >>> (bp & 7); };
    const get = (n) => { const v = peek() & ((1 << n) - 1); bp += n; return v; };
    const sym = (h) => { const e = h.t[peek() & h.mask]; bp += e & 15; return e >> 4; };
    while (!last) {
      last = get(1);
      const kind = get(2);
      let L, D;
      if (kind === 0) {
        const i = ((bp + 7) >>> 3) + 4, n = src[i - 4] | src[i - 3] << 8;
        out.set(src.subarray(i, i + n), op); op += n; bp = (i + n) * 8;
        continue;
      } else if (kind === 1) {
        if (!fixed) {
          const l = new Uint8Array(288).fill(8, 0, 144).fill(9, 144, 256).fill(7, 256, 280).fill(8, 280, 288);
          fixed = [huff(l), huff(new Uint8Array(30).fill(5))];
        }
        [L, D] = fixed;
      } else if (kind === 2) {
        const nl = get(5) + 257, nd = get(5) + 1, nc = get(4) + 4, cl = new Uint8Array(19);
        for (let i = 0; i < nc; i++) cl[ORDER[i]] = get(3);
        const C = huff(cl), lens = new Uint8Array(nl + nd);
        for (let i = 0; i < nl + nd;) {
          const s = sym(C);
          if (s < 16) { lens[i++] = s; continue; }
          const v = s === 16 ? lens[i - 1] : 0, n = s === 16 ? 3 + get(2) : s === 17 ? 3 + get(3) : 11 + get(7);
          lens.fill(v, i, i + n); i += n;
        }
        L = huff(lens.subarray(0, nl)); D = huff(lens.subarray(nl));
      } else throw new Error("gzip invalide");
      for (;;) {
        let s = sym(L);
        if (s < 256) { out[op++] = s; continue; }
        if (s === 256) break;
        s -= 257;
        const n = LB[s] + get(LX[s]), ds = sym(D), d = DB[ds] + get(DX[ds]);
        for (let i = 0; i < n; i++, op++) out[op] = out[op - d];
      }
    }
    if (op !== out.length) throw new Error("gzip tronqué");
    return out;
  }
  function gunzip(buf) {
    const b = new Uint8Array(buf), flg = b[3];
    if (b[0] !== 0x1f || b[1] !== 0x8b || b[2] !== 8) throw new Error("gzip invalide");
    let p = 10;
    if (flg & 4) p += 2 + (b[p] | b[p + 1] << 8);
    if (flg & 8) while (b[p++]);
    if (flg & 16) while (b[p++]);
    if (flg & 2) p += 2;
    const n = (b[b.length - 4] | b[b.length - 3] << 8 | b[b.length - 2] << 16 | b[b.length - 1] << 24) >>> 0;
    return inflate(b.subarray(p, b.length - 8), new Uint8Array(n));
  }
  const native = typeof DecompressionStream !== "undefined";
  // Corps d'une réponse : un flux quand le navigateur sait décompresser en flux, sinon les octets.
  const body = (gz) => native ? new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip")) : gunzip(gz);
  const bytes = async (gz) => native ? new Uint8Array(await new Response(body(gz)).arrayBuffer()) : gunzip(gz);
  // Réponse à un fetch du site, pour les pages et le worker des scénarios sans service worker.
  async function respond(k, r, search) {
    const f = await load(k, r, search);
    if (f.status !== 200) return new Response("Introuvable", { status: f.status === 404 ? 404 : 502 });
    return new Response(body(f.gz), { status: 200, headers: { "Content-Type": type(r) } });
  }
  return { rel, keys, load, body, bytes, gunzip, respond, type, hex };
}

const BASE = new URL("./", self.location).pathname;
const CORE = siteCore(BASE, self.fetch.bind(self));
const DB = "site-auth", STORE = "k";
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
  if (!keys) keys = await CORE.keys(await idb("readonly", (s) => s.get("keys")).catch(() => null)) || null;
  return keys;
}
async function dropKeys() {
  keys = null;
  await idb("readwrite", (s) => s.delete("keys")).catch(() => null);
}

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
  const f = await CORE.load(k, rel, search);
  // « /2027 » sans barre finale : c'est un dossier, on redirige pour que ses liens relatifs marchent.
  if (f.status === 404 && nav && !/\.[^/]+$/.test(rel)) {
    const probe = await CORE.load(k, rel + "/index.html", "");
    if (probe.status === 200) { const u = new URL(req.url); u.pathname += "/"; return Response.redirect(u.href, 301); }
  }
  if (f.status === "bad") {
    if (nav) { await dropKeys(); return shell(); }
    return new Response("Déchiffrement impossible", { status: 500 });
  }
  if (f.status !== 200) return new Response("Introuvable", { status: f.status === 404 ? 404 : 502 });
  return new Response(CORE.body(f.gz), { status: 200, headers: { "Content-Type": CORE.type(rel) } });
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
  const rel = e.request.method === "GET" ? CORE.rel(url) : null;
  if (rel !== null) e.respondWith(serve(e.request, rel, url.search));
});
