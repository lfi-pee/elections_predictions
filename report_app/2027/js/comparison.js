"use strict";

const COMPARISON = { block: "G", sort: "id", ascending: true, rows: [], reference: null, variant: null, preset: "turnout2024", lastPreset: "turnout2024" };
const COMP_BLOCKS = ["G", "CD", "ED", "AU"];
const COMP_KEYS = ["reference", "variant", "h2017", "h2022", "h2024", "uniform", "proportional"];
const COMP_LABELS = { id: "Circonscription", reference: "Référence 2027", variant: "Variante 2027",
  h2017: "Législatives 2017", h2022: "Législatives 2022", h2024: "Législatives 2024",
  uniform: "Variation uniforme", proportional: "Variation proportionnelle" };
const escapeComparison = (s) => String(s).replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const comparisonSearch = (s) => String(s).normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();

function comparisonTarget(nat) {
  const effective = natEffective(nat.G, nat.CD, nat.ED, nat.AB);
  const values = norm4(...effective, nat.AU || 0);
  return Object.fromEntries(COMP_BLOCKS.map((b, i) => [b, values[i]]));
}

// Descriptive baselines, independent of the learned geographic model.
function comparisonBaseline(local, national, target, method) {
  if (!local) return null;
  const values = COMP_BLOCKS.map((b) => method === "uniform"
    ? Math.max(0, local[b] + target[b] - national[b])
    : (national[b] > 0 ? local[b] * target[b] / national[b] : 0));
  const total = values.reduce((a, b) => a + b, 0);
  if (!total) return null;
  return Object.fromEntries(COMP_BLOCKS.map((b, i) => [b, 100 * values[i] / total]));
}

function comparisonModel(nat) {
  const previous = APP.nat;
  try {
    APP.nat = nat;
    const results = [];
    eachCirco((r, i) => { results[i] = { G: r.g, CD: r.cd, ED: r.ed, AU: r.au }; });
    return results;
  } finally { APP.nat = previous; }
}

function buildComparisonRows() {
  const a = APP.data.circoArr, c = COMPARISON;
  const ref = comparisonModel(c.reference), variant = comparisonModel(c.variant);
  const history = Object.fromEntries(APP.data.history.elections.map((e) => [e.key, e]));
  const target = comparisonTarget(c.reference);
  c.rows = a.id.map((id, i) => {
    const available = covIsPublishable(id), h = history["2024"].rows[id];
    return { id, name: a.nm[i] || "", dept: a.dept[i],
      reference: available ? ref[i] : null, variant: available ? variant[i] : null,
      h2017: history["2017"].rows[id] || null, h2022: history["2022"].rows[id] || null,
      h2024: h || null,
      uniform: available ? comparisonBaseline(h, history["2024"].national, target, "uniform") : null,
      proportional: available ? comparisonBaseline(h, history["2024"].national, target, "proportional") : null };
  });
}

function comparisonVisibleRows() {
  const c = COMPARISON, term = comparisonSearch($("filter").value.trim());
  const rows = c.rows.filter((r) => comparisonSearch(`${r.id} ${r.name} ${r.dept}`).includes(term));
  return rows.sort((a, b) => {
    const fallback = a.id.localeCompare(b.id, "fr", { numeric: true });
    if (c.sort === "id") return (c.ascending ? 1 : -1) * fallback;
    const av = a[c.sort]?.[c.block], bv = b[c.sort]?.[c.block];
    if (av == null && bv == null) return fallback;
    if (av == null) return 1;
    if (bv == null) return -1;
    return (c.ascending ? av - bv : bv - av) || fallback;
  });
}

function renderComparison() {
  const c = COMPARISON, rows = comparisonVisibleRows();
  $("rows").innerHTML = rows.map((r) => `<tr><th scope="row">${escapeComparison(r.id)}<small>${escapeComparison(r.name)}</small></th>` +
    COMP_KEYS.map((k) => `<td>${r[k]?.[c.block] == null ? '<span title="Valeur indisponible ou projection non publiée">—</span>' : fmt1(r[k][c.block]) + " %"}</td>`).join("") + "</tr>").join("");
  $("status").textContent = `${rows.length} / ${c.rows.length} circonscriptions · ${APP.NAME[c.block]} · tri : ${COMP_LABELS[c.sort]} ${c.ascending ? "croissant" : "décroissant"}`;
  $("caption").textContent = `${APP.NAME[c.block]} · Scores de premier tour, en % des suffrages exprimés. Cliquez sur un en-tête pour trier.`;
  document.querySelectorAll("th[data-key]").forEach((th) => {
    const active = th.dataset.key === c.sort;
    th.setAttribute("aria-sort", active ? (c.ascending ? "ascending" : "descending") : "none");
    th.querySelector("button").textContent = COMP_LABELS[th.dataset.key] + (active ? (c.ascending ? " ↑" : " ↓") : "");
  });
}

function comparisonPresetLabel() {
  if (COMPARISON.preset === "custom") return "Réglage personnalisé";
  if (COMPARISON.preset === "reference") return "Identique à la référence";
  return "Participation " + COMPARISON.preset.slice(-4);
}

function applyComparisonPreset(key) {
  const c = COMPARISON;
  if (!["turnout2024", "turnout2022", "reference"].includes(key)) throw new Error("Unknown preset");
  c.variant = { ...c.reference };
  if (key !== "reference") {
    const election = APP.data.history.elections.find((e) => e.key === key.slice(-4));
    const abstention = election?.participation?.abstention_pct;
    if (!Number.isFinite(abstention) || abstention < 0 || abstention > 100) throw new Error("Missing historical turnout");
    c.variant.AB = abstention;
  }
  c.preset = key;
  c.lastPreset = key;
}

function syncComparisonKnobs() {
  const c = COMPARISON;
  $("preset").value = c.preset;
  COMP_LABELS.variant = "Variante · " + comparisonPresetLabel();
  $("preset-description").textContent = c.preset.startsWith("turnout")
    ? `${comparisonPresetLabel()} : ${fmt1(100 - c.variant.AB)} % de participation, soit ${fmt1(c.variant.AB)} % d’abstention. Valeur observée au premier tour ; somme des abstentions divisée par la somme des inscrits. Niveaux de vote de référence conservés avant l’effet de participation. Hypothèse de sensibilité pour 2027, pas une prévision de participation.`
    : c.preset === "reference" ? "Les deux colonnes utilisent exactement les mêmes paramètres."
    : "Paramètres personnalisés. Le bouton réapplique le dernier préréglage sélectionné.";
  $("level").max = APP.VOTE.reduce((sum, b) => sum + c.reference[b], 0);
  $("level").value = c.variant[c.block];
  $("ab").value = c.variant.AB;
  $("level-value").textContent = fmt1(c.variant[c.block]) + " % à l’abstention de référence";
  $("ab-value").textContent = fmt1(c.variant.AB) + " % des inscrits";
  const describe = (n) => `G ${fmt1(n.G)} % · C+D ${fmt1(n.CD)} % · ED ${fmt1(n.ED)} % · abstention ${fmt1(n.AB)} %`;
  $("parameters").textContent = `Référence : ${describe(c.reference)}. Variante : ${describe(c.variant)}. Niveaux G/CD/ED avant couplage de participation et normalisation avec le résidu Autre.`;
}

function comparisonCSV(rows) {
  const quote = (value) => {
    let s = String(value ?? "");
    if (/^[=+@\-\t\r]/.test(s)) s = "'" + s;
    return '"' + s.replace(/"/g, '""') + '"';
  };
  const header = ["circonscription", "nom", "bloc", ...COMP_KEYS.map((k) => COMP_LABELS[k] + " (% exprimés)"),
    "parametres_reference", "parametres_variante", "prereglage_variante", "sources_et_methodes"];
  const provenance = JSON.stringify({ elections: APP.data.history.elections.map(({key, source, mapping, participation}) => ({key, source, mapping, participation})),
    uniform: "local2024 + national_reference - national2024 ; plancher 0 puis normalisation à 100",
    proportional: "local2024 * national_reference / national2024 ; normalisation à 100",
    reference: "compute.js / circoEval ; scénario par défaut ; national_reference effectif après participation et normalisation Autre" });
  return "\uFEFF" + [header, ...rows.map((r) => [r.id, r.name, APP.NAME[COMPARISON.block],
    ...COMP_KEYS.map((k) => r[k]?.[COMPARISON.block] ?? ""),
    JSON.stringify(COMPARISON.reference), JSON.stringify(COMPARISON.variant), comparisonPresetLabel(), provenance])]
    .map((row) => row.map(quote).join(";")).join("\r\n");
}

async function bootComparison() {
  try {
    const [summary, circoArr, gamma, coverage, history] = await Promise.all([
      loadJSON("data/summary.json"), loadJSON("data/circo.json"), loadJSON("data/gamma_curve.json"),
      loadJSON("data/coverage.json"), loadJSON("data/comparison_history.json")]);
    APP.data = { summary, circoArr, gamma, coverage, history };
    APP.scnObj = summary.scenarios.find((s) => s.key === summary.default_scenario);
    APP.scenario = APP.scnObj.key;
    COMPARISON.reference = { ...APP.scnObj.means };
    applyComparisonPreset("turnout2024");
    APP.nat = { ...COMPARISON.reference };
    initCoverage();
    buildComparisonRows(); syncComparisonKnobs(); renderComparison();
    $("sources").innerHTML = history.elections.map((e) => `<li><b>${escapeComparison(e.label)}</b> : ${escapeComparison(e.mapping)} ` +
      (e.source.startsWith("https://") ? `<a href="${escapeComparison(e.source)}" target="_blank" rel="noopener">Fichier officiel</a>` : escapeComparison(e.source)) + "</li>").join("");
    ["export", "level", "ab", "preset", "reset-variant"].forEach((id) => { $(id).disabled = false; });
    $("preset").onchange = () => { applyComparisonPreset($("preset").value); buildComparisonRows(); syncComparisonKnobs(); renderComparison(); };
    $("bloc").onchange = () => { COMPARISON.block = $("bloc").value; syncComparisonKnobs(); renderComparison(); };
    $("filter").oninput = renderComparison;
    document.querySelectorAll("[data-sort]").forEach((button) => { button.onclick = () => {
      const key = button.dataset.sort;
      COMPARISON.ascending = COMPARISON.sort === key ? !COMPARISON.ascending : key === "id";
      COMPARISON.sort = key; renderComparison();
    }; });
    $("level").oninput = () => {
      COMPARISON.preset = "custom";
      const n = COMPARISON.variant, block = COMPARISON.block, v = Number($("level").value);
      const others = APP.VOTE.filter((b) => b !== block), total = others.reduce((s, b) => s + n[b], 0);
      const budget = APP.VOTE.reduce((sum, b) => sum + COMPARISON.reference[b], 0);
      for (const b of others) n[b] = total ? (budget - v) * n[b] / total : (budget - v) / others.length;
      n[block] = v;
      buildComparisonRows(); syncComparisonKnobs(); renderComparison();
    };
    $("ab").oninput = () => { COMPARISON.preset = "custom"; COMPARISON.variant.AB = Number($("ab").value); buildComparisonRows(); syncComparisonKnobs(); renderComparison(); };
    $("reset-variant").onclick = () => { applyComparisonPreset(COMPARISON.lastPreset); buildComparisonRows(); syncComparisonKnobs(); renderComparison(); };
    $("export").onclick = () => {
      const url = URL.createObjectURL(new Blob([comparisonCSV(comparisonVisibleRows())], { type: "text/csv;charset=utf-8" }));
      const a = document.createElement("a"); a.href = url; a.download = `comparaison-2027-${COMPARISON.block}.csv`;
      document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
  } catch (error) {
    $("status").textContent = "Impossible de charger le comparateur. Rechargez la page pour réessayer.";
    console.error(error);
  }
}
bootComparison();
