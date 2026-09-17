"use strict";
// Page « Négocier les circonscriptions » : lit data/negotiation.json (calculé par
// src/negotiation_2027.py — Monte-Carlo sur l'incertitude nationale + locale, effet d'étiquette
// mesuré sur 2024) et le rend : tuiles, courbe « combien en demander », tableau triable, CSV.
// Aucun chiffre n'est saisi ici : tout descend du JSON servi.

const NEG = { data: null, rows: [], sort: "rank", asc: true, n: 0, share: null };
const POSTURE_LAB = { exiger: "Exiger", disputer: "Disputer", obtenir: "Obtenir", difficile: "Difficile",
  monnaie: "Monnaie d'échange", rien: "Rien à jouer" };
const POSTURE_ORDER = { exiger: 0, disputer: 1, obtenir: 2, difficile: 3, monnaie: 4, rien: 5, acquis: 6, hors_union: 7, non_mesure: 8 };
// Une seule colonne d'état : la posture, ou le groupe quand il n'y a pas de posture (acquis,
// gauche hors union, non mesurée).
const stateKey = (r) => r.posture || r.group;
const POSTURE_TIP = {
  exiger: "LFI seule se qualifierait, pas l'autre gauche : terrain LFI, revendication incontestable.",
  disputer: "Les deux se qualifieraient seuls : cœur de la négociation, à trancher au prix et au nombre.",
  obtenir: "Aucun des deux seul : l'union crée le siège. Argument = le prix de l'étiquette est faible.",
  difficile: "Seule l'autre gauche se qualifierait : terrain du partenaire, à ne demander qu'en échange.",
  monnaie: "Siège imprenable, mais LFI y paraît au moins aussi forte : à céder, ça a l'air d'un sacrifice.",
  rien: "Siège imprenable et LFI plus faible : rien à jouer." };
const GROUP_LAB = { acquis: "Acquis", libre: "Libre", a_negocier: "À négocier",
  sans_enjeu: "Sans enjeu", hors_union: "Gauche hors union", non_mesure: "Non mesurée" };
const GROUP_ORDER = { acquis: 0, libre: 1, a_negocier: 2, sans_enjeu: 3, hors_union: 4, non_mesure: 5 };
const GROUP_LAB_DEP = { "LFI-NFP": "LFI", SOC: "PS", ECOS: "Écologistes", GDR: "GDR (PCF & outre-mer)",
  EPR: "Ensemble", DEM: "MoDem", HOR: "Horizons", DR: "LR", UDDPLR: "UDR (Ciotti)", RN: "RN",
  LIOT: "LIOT", NI: "Non inscrit" };
const $n = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fold = (s) => String(s ?? "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
// Une probabilité n'est jamais affichée comme une certitude : >99 % et <1 % aux bords.
const pct = (p) => p == null ? "—" : p >= 0.995 ? ">99 %" : (p > 0 && p < 0.005) ? "<1 %" : Math.round(p * 100) + " %";
const f1 = (x) => x == null ? "—" : x.toLocaleString("fr-FR", { maximumFractionDigits: 1 });
const f2 = (x) => x == null ? "—" : x.toLocaleString("fr-FR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

async function negLoad() {
  const r = await fetch("data/negotiation.json?v=1");
  if (!r.ok) throw new Error("negotiation.json");
  NEG.data = await r.json();
  NEG.rows = NEG.data.rows.map((x) => ({ ...x,
    depute: x.depute && x.depute.nom ? `${x.depute.prenom} ${x.depute.nom}` : "",
    depGroup: x.depute ? x.depute.groupe : "" }));
  // Rapport de force : option extérieure par part nationale LFI (grille précalculée).
  const sp = NEG.data.split;
  NEG.share = sp.near;
  $n("share").innerHTML = sp.shares.map((k) => `<option value="${k}"${k === sp.near ? " selected" : ""}>${Math.round(+k * 100)} %${k === sp.near ? ` (sondages : ${Math.round(sp.default_share * 100)} %)` : ""}</option>`).join("");
  $n("share").addEventListener("change", () => { NEG.share = $n("share").value; negApplyShare(); negTiles(); negTable(); });
  negApplyShare();
  const curve = NEG.data.curve;
  const sl = $n("n");
  sl.max = curve.ids.length;
  // Demande par défaut = taille de la carte 2024 (229 circos FI) moins les sortant·es LFI.
  NEG.n = Math.max(0, Math.min(curve.ids.length, NEG.data.totals.slate2024_n - NEG.data.totals.n_acquis));
  sl.value = NEG.n;
  sl.addEventListener("input", () => { NEG.n = +sl.value; negSlate(); negTable(); });
  const deps = [...new Set(NEG.rows.map((x) => x.depGroup).filter(Boolean))].sort();
  $n("dep").innerHTML = '<option value="">Tous groupes</option>' +
    deps.map((g) => `<option value="${esc(g)}">${esc(GROUP_LAB_DEP[g] || g)}</option>`).join("");
  for (const id of ["group", "posture", "filter", "dep"]) $n(id).addEventListener("input", negTable);
  document.querySelectorAll("th button[data-sort]").forEach((b) => b.addEventListener("click", () => {
    const k = b.dataset.sort;
    if (NEG.sort === k) NEG.asc = !NEG.asc; else { NEG.sort = k; NEG.asc = !["p_lfi", "p_other", "price", "rate", "p_lfi_local", "p_lfi_ru", "q_lfi", "q_other", "h2024_G", "h2022_G", "h2017_G", "h2017_LFI", "rdev", "ext_plus_G", "ext_mult_G"].includes(k); }
    negTable();
  }));
  $n("export").disabled = false;
  $n("export").addEventListener("click", negExport);
  negTiles(); negMethods(); negSlate(); negTable();
}

// Posture = valeur (groupe) × rapport de force (qui, seul, se qualifie). Miroir de
// negotiation_2027.posture (Python) — même règle, mêmes seuils (split.leverage_q).
function negPosture(group, qL, qO) {
  if (group === "acquis" || group === "hors_union" || group === "non_mesure" || qL == null || qO == null) return null;
  const t = NEG.data.split.leverage_q, lfi = qL >= t, oth = qO >= t;
  if (group === "libre" || group === "a_negocier") return lfi && !oth ? "exiger" : lfi && oth ? "disputer" : oth ? "difficile" : "obtenir";
  return qL >= qO ? "monnaie" : "rien";
}

// Applique la part LFI choisie : recopie q_lfi / q_other de la grille et recalcule la posture.
function negApplyShare() {
  const b = NEG.data.split.by_share[NEG.share];
  NEG.rows.forEach((r, i) => {
    r.q_lfi = r.pub ? b.q_lfi[i] : null; r.q_other = r.pub ? b.q_other[i] : null;
    r.posture = negPosture(r.group, r.q_lfi, r.q_other);
  });
}

function negTiles() {
  const d = NEG.data, g = d.groups, t = d.totals, k = d.params.label_effect_k;
  const tiles = [
    { k: "Sortant·es LFI (acquis)", v: g.acquis, s: `${f1(t.acquis_expected)} sièges espérés en 2027 avec la carte actuelle — hors négociation.`, cls: "" },
    { k: "Libres", v: g.libre, s: "L'étiquette LFI n'y coûte rien de mesurable à l'union : à réclamer toutes.", cls: "" },
    { k: "À négocier", v: g.a_negocier, s: "L'union y perd quelque chose à donner la circonscription à LFI. Le prix est affiché ligne par ligne.", cls: "" },
    { k: "Sans enjeu", v: g.sans_enjeu, s: `Aucune étiquette de gauche n'a ${Math.round(d.params.p_min * 100)} % de chance : hors du troc.${g.hors_union ? ` ${g.hors_union} autres sièges sont tenus par une gauche hors union (signalés, hors classement).` : ""}`, cls: "" },
    { k: "Effet d'étiquette 2024", v: `${Math.round(k.fi * 100)} % vs ${Math.round(k.other * 100)} %`,
      s: `Part des voix libérées récupérée au 2nd tour par un·e candidat·e LFI vs PS/écolo/PCF, dans les ${d.params.label_effect_n} duels face au RN (IC 95 % de l'écart : ${f2(d.params.label_effect_ci95[0])} à ${f2(d.params.label_effect_ci95[1])}).`, cls: "warn" },
    { k: "Carte 2024 rejouée", v: f1(t.slate2024_expected), s: `sièges LFI espérés si LFI garde ses ${t.slate2024_n} circonscriptions de 2024 ; ${f1(t.efficient_same_n_expected)} avec le même nombre pris dans l'ordre de ce tableau (prix affiché).`, cls: "" },
  ];
  const cnt = (p) => NEG.rows.filter((r) => r.posture === p).length;
  tiles.push({ k: `Rapport de force à ${Math.round(+NEG.share * 100)} % LFI`, v: `${cnt("exiger") + cnt("disputer")} / ${cnt("difficile")}`,
    s: `circonscriptions précieuses où LFI seule se qualifierait (à exiger ou disputer) / où seule l'autre gauche se qualifierait (difficiles). ${cnt("obtenir")} à obtenir sur l'argument du prix, ${cnt("monnaie")} monnaies d'échange.`, cls: "" });
  $n("tiles").innerHTML = tiles.map((x) => `<div class="tile ${x.cls}"><div class="k">${esc(x.k)}</div><div class="v">${esc(String(x.v))}</div><div class="s">${esc(x.s)}</div></div>`).join("");
}

function negMethods() {
  const d = NEG.data, k = d.params.label_effect_k, le = d.params.label_effect, ci = d.params.label_effect_ci95;
  $n("m-label").innerHTML = `Le parti de chaque candidat·e d'union 2024 est connu par la répartition des circonscriptions du Nouveau Front populaire. Dans les <b>${d.params.label_effect_n} duels</b> candidat·e d'union contre RN (centre-droit éliminé), on mesure la part des voix libérées au 1<sup>er</sup> tour que le·la candidat·e de gauche récupère au 2<sup>nd</sup> : <b>${Math.round(k.fi * 100)} %</b> pour une étiquette LFI, <b>${Math.round(k.other * 100)} %</b> pour PS, écologistes et PCF (moyenne de l'union : ${Math.round(k.union * 100)} %). Écart ${f2(k.fi - k.other)}, intervalle bootstrap à 95 % de ${f2(ci[0])} à ${f2(ci[1])} ; présent dans les trois terciles de force de la gauche (il n'est donc pas compensé dans les bastions) ; ${f1(le.margin_effect_lfi_pts_inscrits)} point d'inscrits sur la marge de 2<sup>nd</sup> tour à marge de 1<sup>er</sup> tour égale. Le modèle de sièges de la carte reçoit cet écart comme décalage du taux de report centre-droit → gauche (${f2(d.params.cd2l_delta.lfi)} pour LFI, +${f2(d.params.cd2l_delta.other)} pour les autres), centré pour que le modèle moyen — calibré sur le nombre réel de sièges RN 2024 — reste inchangé. C'est un taux de report, pas un taux de victoire : le fait que LFI ait reçu des circonscriptions plus dures en 2024 ne le biaise pas.`;
  const hors = NEG.rows.filter((r) => r.group === "hors_union");
  $n("m-hors").innerHTML = `${hors.length} circonscription${hors.length > 1 ? "s" : ""} — ${hors.map((r) => `${esc(r.id)} ${esc(r.nm)} (${esc(r.depute)}, ${esc(GROUP_LAB_DEP[r.depGroup] || r.depGroup)})`).join(" ; ")} — ont été gagnées en 2024 par une candidature codée à gauche mais hors de l'union, dont le ou la titulaire ne siège pas dans un groupe de gauche. Le bloc de gauche prédit y inclut ses voix, qui ne se reporteraient pas sur une candidature d'union : la chance affichée surestime ce qu'obtiendrait LFI ou l'autre gauche. Ces sièges ne sont ni «&nbsp;libres&nbsp;» ni à partager entre LFI et ses partenaires : ils sont sortis du classement et de la courbe, et signalés.`;
  $n("m-groups").innerHTML = `<b>Acquis</b> : député·e sortant·e du groupe LFI (${d.groups.acquis}). <b>Gauche hors union</b> : siège tenu par un·e élu·e de gauche hors de l'union (${d.groups.hors_union || 0}, voir ci-dessous). <b>Sans enjeu</b> : ni l'étiquette LFI ni une autre n'atteint ${Math.round(d.params.p_min * 100)} % de chance (${d.groups.sans_enjeu}). <b>Libre</b> : prix ≤ ${f2(d.params.price_free)} siège espéré (${d.groups.libre}). <b>À négocier</b> : prix supérieur (${d.groups.a_negocier}). <b>Non mesurée</b> : hors nomenclature de blocs (${d.groups.non_mesure}). Le classement (#) ne porte que sur les libres et à négocier, par chance LFI décroissante.`;
  const sp = d.split;
  $n("m-posture").innerHTML = `Une négociation se gagne aussi par ce que chacun ferait <em>sans</em> accord. Pour chaque circonscription, le modèle rejoue une gauche <b>divisée</b> (LFI seule contre le reste de la gauche, part nationale de LFI réglable de ${Math.round(+sp.shares[0] * 100)} à ${Math.round(+sp.shares[sp.shares.length - 1] * 100)} %, sondages : ${Math.round(sp.default_share * 100)} %, motif local du vote Mélenchon à la dernière présidentielle) et mesure la probabilité que <b>LFI seule</b>, et que <b>l'autre gauche seule</b>, se qualifie au 2<sup>nd</sup> tour. Un camp qui se qualifie seul (probabilité ≥ ${Math.round(sp.leverage_q * 100)} %) n'a pas besoin de l'accord dans cette circonscription&nbsp;: sa revendication est incontestable, sa menace d'y aller seul crédible. Croisé avec la valeur de la circonscription, cela donne six postures&nbsp;: <b>exiger</b> (précieuse, LFI seule se qualifie, pas l'autre), <b>disputer</b> (précieuse, les deux se qualifieraient — le cœur de la négociation), <b>obtenir</b> (précieuse, aucun des deux seul — l'union crée le siège, l'argument est le prix), <b>difficile</b> (précieuse, seule l'autre gauche se qualifie — terrain du partenaire, à ne demander qu'en échange), <b>monnaie d'échange</b> (siège imprenable mais LFI y paraît au moins aussi forte — céder a l'air d'un sacrifice) et <b>rien à jouer</b>. Le curseur «&nbsp;part de LFI dans la gauche&nbsp;» montre comment le rapport de force bascule avec le niveau national de LFI&nbsp;: c'est l'autre levier de la négociation.`;
  const hs = d.history || {};
  $n("src-history").innerHTML = "Résultats passés : " + Object.values(hs).map((e) => `<b>${esc(e.label)}</b> (gauche nationale ${f1(e.national_G)} %${e.national_LFI != null ? `, LFI ${f1(e.national_LFI)} %` : ""}) — ${e.source.startsWith("https://") ? `<a href="${esc(e.source)}" target="_blank" rel="noopener">fichier officiel</a>` : esc(e.source)}`).join(" ; ") + ". Les scores sont en % des suffrages exprimés, tous candidats au dénominateur ; « LFI seule » n'est séparable qu'en 2017 (nuance FI), la gauche étant unie au 1<sup>er</sup> tour en 2022 et 2024.";
  const ns = d.params.nat_sigma, ls = d.params.local_sigma, m = d.scenario.means;
  $n("m-unc").innerHTML = `Scénario « ${esc(d.scenario.label)} », ancre sondages G ${f1(m.G)} · C+D ${f1(m.CD)} · ED ${f1(m.ED)} %, abstention ${f1(m.AB)} %. <b>${d.params.draws} tirages</b> Monte-Carlo : le niveau national de chaque bloc est tiré autour de l'ancre avec l'erreur historique des sondages législatifs (écart-type G ${f1(ns.G)}, C+D ${f1(ns.CD)}, ED ${f1(ns.ED)} points — validation croisée 2002→2022), puis chaque circonscription reçoit son erreur locale (G ${f1(ls.G)}, C+D ${f1(ls.CD)}, ED ${f1(ls.ED)} points, la même que la fourchette de la carte). Un classement fait à un seul réglage de curseur ne survivrait pas à une réunion ; celui-ci moyenne sur ce que les sondages peuvent se tromper. La colonne « sondages exacts » garde l'erreur locale seule ; « droites unies » rejoue le scénario où LR refuse le front républicain.`;
}

function negSlate() {
  const d = NEG.data, c = d.curve, n = NEG.n, t = d.totals;
  const lfi = n ? c.cum_lfi[n - 1] : 0, cost = n ? c.cum_price[n - 1] : 0;
  const marg = n ? (c.cum_lfi[n - 1] - (n > 1 ? c.cum_lfi[n - 2] : 0)) : null;
  $n("n-value").textContent = n.toLocaleString("fr-FR");
  $n("n-label").textContent = n.toLocaleString("fr-FR");
  const row = n ? NEG.rows.find((r) => r.id === c.ids[n - 1]) : null;
  $n("slate-read").innerHTML =
    `<div>Avec <b>${n}</b> circonscriptions demandées + <b>${t.n_acquis}</b> sortant·es : <b>${f1(t.acquis_expected + lfi)}</b> sièges LFI espérés (${f1(t.acquis_expected)} acquis + ${f1(lfi)}).</div>` +
    `<div>Coût pour l'union : <b>${f1(cost)}</b> siège${cost >= 1.95 ? "s" : ""} espéré${cost >= 1.95 ? "s" : ""} perdu${cost >= 1.95 ? "s" : ""} par rapport à des candidat·es non-LFI sur ces mêmes circonscriptions.</div>` +
    (row ? `<div>La ${n}<sup>e</sup> circonscription est <b>${esc(row.id)} ${esc(row.nm)}</b> : chance LFI ${pct(row.p_lfi)}, prix ${f2(row.price)} — c'est ce que rapporte la dernière demandée (${f2(marg)} siège).</div>` : "");
  const base = t.acquis_expected;
  negChart("chart-lfi", c.cum_lfi.map((v) => v + base), n, "lfi", { ref: t.slate2024_expected, refLab: `carte 2024 : ${f1(t.slate2024_expected)}` }, base);
  negChart("chart-cost", c.cum_price, n, "cost", null, 0);
}

// Courbe en SVG : une série, grille discrète, curseur = demande courante, tooltip au survol.
function negChart(id, ys, n, cls, ref, y0) {
  const el = $n(id); const W = el.clientWidth || 520, H = el.clientHeight || 220;
  const m = { t: 14, r: 14, b: 26, l: 40 }, w = W - m.l - m.r, h = H - m.t - m.b;
  const N = ys.length, ymax = Math.max(...ys, ref ? ref.ref : 0, 1) * 1.05;
  const X = (i) => m.l + (w * i) / N, Y = (v) => m.t + h - (h * v) / ymax;
  const ticks = 4, ty = Array.from({ length: ticks + 1 }, (_, i) => (ymax * i) / ticks);
  const tx = [0, Math.round(N / 4), Math.round(N / 2), Math.round((3 * N) / 4), N];
  const path = [`M${X(0)},${Y(y0)}`].concat(ys.map((v, i) => `L${X(i + 1)},${Y(v)}`)).join(" ");
  const cur = n ? ys[n - 1] : y0;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">
    <g class="grid">${ty.map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/>`).join("")}</g>
    <g class="axis">${ty.map((v) => `<text x="${m.l - 6}" y="${Y(v) + 3.5}" text-anchor="end">${Math.round(v)}</text>`).join("")}
      ${tx.map((i) => `<text x="${X(i)}" y="${H - 8}" text-anchor="middle">${i}</text>`).join("")}
      <line x1="${m.l}" x2="${W - m.r}" y1="${m.t + h}" y2="${m.t + h}"/></g>
    ${ref ? `<line class="ref" x1="${m.l}" x2="${W - m.r}" y1="${Y(ref.ref)}" y2="${Y(ref.ref)}"/><text class="ref-lab" x="${W - m.r}" y="${Y(ref.ref) - 4}" text-anchor="end">${esc(ref.refLab)}</text>` : ""}
    <path class="line ${cls}" d="${path}"/>
    <line class="cursor" x1="${X(n)}" x2="${X(n)}" y1="${m.t}" y2="${m.t + h}"/>
    <circle class="dot" cx="${X(n)}" cy="${Y(cur)}" r="4.5"/>
  </svg><div class="tip"></div>`;
  const tip = el.querySelector(".tip");
  el.onmousemove = (e) => {
    const r = el.getBoundingClientRect(), i = Math.max(0, Math.min(N, Math.round(((e.clientX - r.left) - m.l) / w * N)));
    const v = i ? ys[i - 1] : y0;
    tip.style.display = "block"; tip.style.left = X(i) + "px"; tip.style.top = Y(v) + "px";
    tip.textContent = `${i} circo${i > 1 ? "s" : ""} → ${f1(v)}`;
  };
  el.onmouseleave = () => { tip.style.display = "none"; };
}

function negVisible() {
  const g = $n("group").value, po = $n("posture").value, dep = $n("dep").value, term = fold($n("filter").value.trim());
  let rows = NEG.rows.filter((r) => (g === "" || (g === "negociables" ? (r.group === "libre" || r.group === "a_negocier") : r.group === g))
    && (!po || r.posture === po)
    && (!dep || r.depGroup === dep)
    && (!term || fold(`${r.id} ${r.nm} ${r.dept} ${r.depute} ${r.lab2024 || ""}`).includes(term)));
  const k = NEG.sort, s = NEG.asc ? 1 : -1;
  const val = (r) => k === "group" ? GROUP_ORDER[r.group] : k === "posture" ? POSTURE_ORDER[stateKey(r)] : k === "pred" ? (r.pred ? r.pred.G : null)
    : k === "lab2024" ? (r.lab2024 || "") : k === "rank" ? (r.rank ?? 1e9) : r[k];
  rows.sort((a, b) => {
    const va = val(a), vb = val(b);
    if (va == null && vb == null) return a.id.localeCompare(b.id, "fr", { numeric: true });
    if (va == null) return 1; if (vb == null) return -1;
    if (typeof va === "string") return s * va.localeCompare(vb, "fr", { numeric: true });
    return s * (va - vb) || a.id.localeCompare(b.id, "fr", { numeric: true });
  });
  return rows;
}

function negTable() {
  const rows = negVisible(), inSlate = new Set(NEG.data.curve.ids.slice(0, NEG.n));
  const pb = (p, cls) => p == null ? "—" : `<span class="pb ${cls}"><span>${pct(p)}</span><i><b style="width:${Math.round(p * 100)}%"></b></i></span>`;
  const state = (r) => r.posture ? `<span class="pos pos-${r.posture}" title="${esc(POSTURE_TIP[r.posture])}">${POSTURE_LAB[r.posture]}</span>`
    : `<span class="grp g-${r.group}"><i></i>${GROUP_LAB[r.group]}</span>`;
  const dep = (r) => r.depute ? `<span class="trunc" title="${esc(r.depute)} (${esc(GROUP_LAB_DEP[r.depGroup] || r.depGroup)})">${esc(r.depute)}</span> <span class="dim">${esc(GROUP_LAB_DEP[r.depGroup] || r.depGroup)}</span>` : "—";
  $n("rows").innerHTML = rows.map((r) => `<tr class="${inSlate.has(r.id) ? "in-slate" : ""}">
    <td class="num dim">${r.rank ?? ""}</td>
    <th scope="row" class="left"><span class="trunc" title="${esc(r.id)} ${esc(r.nm)}">${esc(r.id)} <span class="dim">${esc(r.nm)}</span></span></th>
    <td class="left">${state(r)}</td>
    <td class="num">${pb(r.p_lfi, "")}</td>
    <td class="num">${pb(r.p_other, "oth")}</td>
    <td class="num">${f2(r.price)}</td>
    <td class="num">${r.q_lfi == null ? "—" : `${pct(r.q_lfi)} <span class="dim">·</span> ${pct(r.q_other)}`}</td>
    <td class="num">${r.pred ? f1(r.pred.G) + " %" : "—"}</td>
    <td class="left args first">${dep(r)}</td>
    <td class="left args">${r.lab2024 ? esc(r.lab2024) : "—"} <span class="dim">${r.union_won_2024 == null ? "" : r.union_won_2024 ? "· gagné" : "· perdu"}</span></td>
    <td class="num args">${r.h2024_G == null ? "—" : f1(r.h2024_G) + " %"}</td>
    <td class="num args">${r.h2017_LFI == null ? "—" : f1(r.h2017_LFI) + " %"}</td>
    <td class="num args">${r.rdev == null ? "—" : (r.rdev > 0 ? "+" : "") + f1(r.rdev * 100) + " pt"}</td>
    <td class="num args">${r.ext_plus_G == null ? "—" : f1(r.ext_plus_G) + " %"}</td>
  </tr>`).join("");
  document.querySelectorAll("th[data-key]").forEach((th) => {
    const on = th.dataset.key === NEG.sort;
    th.setAttribute("aria-sort", on ? (NEG.asc ? "ascending" : "descending") : "none");
    const b = th.querySelector("button"); b.textContent = b.textContent.replace(/ [↑↓]$/, "") + (on ? (NEG.asc ? " ↑" : " ↓") : "");
  });
  $n("status").textContent = `${rows.length} circonscription${rows.length > 1 ? "s" : ""} affichée${rows.length > 1 ? "s" : ""} · ${NEG.n} demandées surlignées.`;
}

function negExport() {
  const rows = negVisible(), inSlate = new Set(NEG.data.curve.ids.slice(0, NEG.n)), d = NEG.data;
  const head = ["rang", "circo", "nom", "dept", "groupe", "dans_demande", "chance_lfi", "chance_autre_gauche", "prix_union",
    "taux_echange", "posture", "part_lfi_rapport_de_force", "lfi_seule_qualifiee", "autre_gauche_seule_qualifiee",
    "chance_lfi_locale", "chance_lfi_droites_unies", "pred_G", "pred_CD", "pred_ED", "depute", "groupe_depute", "parti_nfp_2024",
    "siege_union_2024", "gauche_2024", "gauche_2022", "lfi_seule_2017", "gauche_2017", "melenchon_dans_la_gauche_ecart_pts",
    "gauche_2024_plus_evolution_nationale", "gauche_2024_fois_evolution_nationale", "inscrits"];
  const q = (v) => v == null ? "" : /[";\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v);
  const lines = [`# negotiation 2027 — scenario ${d.scenario.key} ; tirages ${d.params.draws} ; demande ${NEG.n} ; tri ${NEG.sort} ${NEG.asc ? "asc" : "desc"} ; filtre groupe "${$n("group").value}" texte "${$n("filter").value}"`,
    head.join(";")].concat(rows.map((r) => [r.rank, r.id, r.nm, r.dept, GROUP_LAB[r.group], inSlate.has(r.id) ? 1 : 0, r.p_lfi, r.p_other, r.price,
      r.rate, r.posture ? POSTURE_LAB[r.posture] : "", NEG.share, r.q_lfi, r.q_other,
      r.p_lfi_local, r.p_lfi_ru, r.pred && r.pred.G, r.pred && r.pred.CD, r.pred && r.pred.ED, r.depute, r.depGroup, r.lab2024,
      r.union_won_2024 == null ? "" : (r.union_won_2024 ? 1 : 0), r.h2024_G, r.h2022_G, r.h2017_LFI, r.h2017_G,
      r.rdev == null ? "" : Math.round(r.rdev * 1000) / 10, r.ext_plus_G, r.ext_mult_G, r.ins].map(q).join(";")));
  const blob = new Blob(["﻿" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "negociation_2027_lfi.csv"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

window.addEventListener("resize", () => { if (NEG.data) negSlate(); });
negLoad().catch((e) => { $n("status").textContent = "Erreur de chargement : " + e.message; });
