"use strict";

// Light / dark theme toggle. The page chrome is driven by the [data-theme] attribute
// on <html> (see style.css); the map basemap and the fade-to-background colours of the
// choropleth are swapped here so the map matches the page instead of staying dark.
// The initial attribute is set by an inline <head> script (before first paint), so this
// file only handles the toggle and the map side-effects.

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}

function applyTheme(theme, opts) {
  const t = theme === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", t);
  try { localStorage.setItem("theme", t); } catch (e) { /* private mode */ }

  // Point the map palette at the matching faint end and re-colour the layers.
  APP.PALE = t === "light" ? APP.PALE_LIGHT : APP.PALE_DARK;
  APP.FAINT = t === "light" ? APP.FAINT_LIGHT : APP.FAINT_DARK;
  if (APP.map && typeof setBasemapTheme === "function") setBasemapTheme(t);
  // applyColor reads APP.PALE/APP.FAINT; only call it once the map layers exist.
  if (APP.map && APP.map.getLayer && APP.map.getLayer("bv-fill") && typeof applyColor === "function") {
    applyColor();
  }

  const btn = document.getElementById("theme-toggle");
  if (btn) {
    // Show the current theme; clicking flips to the other one.
    btn.textContent = t === "light" ? "☀️" : "🌙";
    btn.setAttribute("aria-pressed", String(t === "light"));
  }
}

function initTheme() {
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.addEventListener("click", () => applyTheme(currentTheme() === "light" ? "dark" : "light"));
  // Sync JS-side palette + button glyph with whatever the inline script set.
  applyTheme(currentTheme());
}

initTheme();
