"use strict";

// Auto-rechargement en développement : interroge la date de modification des
// fichiers source servis sur le port 8000 et recharge à tout changement.
// Inerte sur le livrable diffusé (tout port autre que 8000).
if (location.port === "8000") {
  const files = ["", "style.css", "js/config.js", "js/map.js", "js/scenario.js",
    "js/panel.js", "js/main.js", "js/devreload.js"];
  const stamps = {};
  const poll = async () => {
    for (const f of files) {
      try {
        const r = await fetch(f + "?_=" + Date.now(), { method: "HEAD", cache: "no-store" });
        const m = r.headers.get("last-modified") || r.headers.get("etag") || "";
        if (stamps[f] !== undefined && stamps[f] !== m) { location.reload(); return; }
        stamps[f] = m;
      } catch (_) { /* fichier momentanément indisponible */ }
    }
  };
  setInterval(poll, 1000);
  poll();
}
