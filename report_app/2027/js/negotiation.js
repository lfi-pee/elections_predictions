"use strict";
// Page « Négocier les circonscriptions » (vu de LFI) : lit data/negotiation.json (calculé par
// src/negotiation_2027.py — Monte-Carlo sur l'incertitude nationale + locale, report propre à une
// candidature LFI mesuré sur 2024) et rend UN tableau : filtres, tri, colonnes redimensionnables,
// CSV. Aucun chiffre n'est saisi ici : tout descend du JSON servi.

const NEG = { data: null, rows: [], sort: "rank", asc: true, share: null };
const GROUP_LAB = { acquis: "Acquis", en_jeu: "En jeu", sans_enjeu: "Sans enjeu",
  hors_union: "Gauche hors union", non_mesure: "Non mesurée" };
// Définitions (avec la règle de calcul, en clair) ; les seuils viennent des paramètres servis.
const P = () => NEG.data ? NEG.data.params : { p_min: 0.05 };
const SP = () => NEG.data ? NEG.data.split : { leverage_q: 0.5 };
const pctInt = (x) => Math.round(x * 100) + " %";
const GROUP_TIP = {
  get acquis() { return "Député·e sortant·e du groupe LFI : hors négociation."; },
  get en_jeu() { return `Chance d'élire un·e député·e LFI ≥ ${pctInt(P().p_min)} : une candidature LFI a une vraie chance d'y gagner le siège, à demander.`; },
  get sans_enjeu() { return `Chance d'élire un·e député·e LFI < ${pctInt(P().p_min)} : imprenable pour LFI.`; },
  get hors_union() { return "Siège tenu par un·e élu·e de gauche hors de l'union : la gauche prédite ici inclut ses voix, qui ne suivraient pas une candidature d'union. Hors classement."; },
  get non_mesure() { return "La nomenclature de blocs ne couvre pas l'électorat (Corse, Guyane) : aucune chance publiable."; } };
const POSTURE_LAB = { exiger: "Exiger", obtenir: "Obtenir", monnaie: "Monnaie d'échange", rien: "Rien à jouer" };
const POSTURE_ORDER = { exiger: 0, obtenir: 1, monnaie: 2, rien: 3, acquis: 4, hors_union: 5, non_mesure: 6 };
const POSTURE_TIP = {
  get exiger() { return `Calcul : la gauche unie gagne le siège (≥ ${pctInt(P().p_min)}) ET, sans accord, LFI seule atteindrait le 2nd tour (≥ ${pctInt(SP().leverage_q)}). Sens : LFI tient ce siège sans l'accord — la revendication ne se refuse pas.`; },
  get obtenir() { return `Calcul : la gauche unie gagne le siège (≥ ${pctInt(P().p_min)}) ET, sans accord, AUCUN des deux pôles n'atteindrait le 2nd tour (LFI seule < ${pctInt(SP().leverage_q)}, reste de la gauche seul < ${pctInt(SP().leverage_q)}). Sens : personne ne peut se passer de l'accord ici — le siège se gagne à la table.`; },
  get monnaie() { return `Calcul : la gauche unie gagne le siège (≥ ${pctInt(P().p_min)}), LFI seule n'atteindrait PAS le 2nd tour (< ${pctInt(SP().leverage_q)}) mais le reste de la gauche seul l'atteindrait (≥ ${pctInt(SP().leverage_q)}). Sens : l'option extérieure est du côté du partenaire, pas de LFI. LFI ne peut pas exiger ce siège et devra le céder — et comme la gauche unie le gagne vraiment, le céder a un prix : c'est ce que LFI met dans la balance.`; },
  get rien() { return `Calcul : la gauche unie ne gagne pas le siège (< ${pctInt(P().p_min)}). Sens : rien à demander, rien à céder.`; } };
// Infobulle d'une pastille : la règle, puis les chiffres de la ligne qui la déclenchent.
// Chaque chiffre du survol a SA colonne dans le tableau, sous le même nom : rien d'invisible.
const postureTipRow = (r) => `${POSTURE_TIP[r.posture]} Ici : la gauche unie gagne ${pct(r.p_left)} · sans accord, LFI seule au 2nd tour ${pct(r.q_lfi)} · sans accord, reste de la gauche seul ${pct(r.q_oth)} · et si LFI porte la candidature d'union, elle gagne ${pct(r.p_lfi)}.`;
const GROUP_LAB_DEP = { "LFI-NFP": "LFI", SOC: "PS", ECOS: "Écologistes", GDR: "GDR (PCF & outre-mer)",
  EPR: "Ensemble", DEM: "MoDem", HOR: "Horizons", DR: "LR", UDDPLR: "UDR (Ciotti)", RN: "RN",
  LIOT: "LIOT", NI: "Non inscrit" };
// Nuances de gauche 2024 (hors NFP) et parti porteur de la candidature NFP.
const NUANCE_LAB = { UG: "NFP", DVG: "div. gauche", EXG: "extr. gauche", ECO: "écolo.", SOC: "PS", FI: "LFI", COM: "PCF", VEC: "EELV", RDG: "radicaux", DXG: "extr. gauche", NUP: "NUPES" };
const NFP_PARTY = { FI: "LFI", PS: "PS", PE: "Écologistes", PCF: "PCF" };
// Part locale de LFI dans la gauche : part nationale (filtre) + écart Mélenchon, bornée — la même
// règle que la force réelle. Sert à partager le « calcul simple » 2027.
const radLocal = (r) => { const p = P(); const g = p.rad_gain ?? 1, c = p.rad_clip || [0.05, 0.95]; return Math.min(c[1], Math.max(c[0], +NEG.share + g * (r.rdev || 0))); };
const lfi27 = (r) => r.ext_plus_G == null ? null : Math.round(r.ext_plus_G * radLocal(r) * 10) / 10;
// Reste de la gauche partagé PS / Écologistes / PCF : part nationale (sondages qui les séparent)
// + écart local du·de la candidat·e présidentiel·le de chaque parti, bornée, renormalisée.
const REST = ["PS", "EELV", "PCF"], REST_LAB = { PS: "PS", EELV: "Écolo.", PCF: "PCF" };
const split27 = (r) => {
  if (r.ext_plus_G == null) return null;
  const pa = NEG.data.parties_2027, c = pa.clip, l = lfi27(r), rest = Math.max(0, r.ext_plus_G - l);
  const sh = REST.map((p) => Math.min(c[1], Math.max(c[0], pa.shares_rest[p] + ((r.pres_rest || {})[p] || 0))));
  const s = sh.reduce((a, b) => a + b, 0);
  const out = { LFI: l };
  REST.forEach((p, i) => { out[p] = Math.round(rest * sh[i] / s * 10) / 10; });
  return out;
};
const parts24 = (r) => r.h2024_parts ? Object.entries(r.h2024_parts).sort((a, b) => b[1] - a[1]).map(([nu, v]) =>
  `${nu === "UG" ? "NFP" + (r.lab2024 ? "-" + esc(NFP_PARTY[r.lab2024] || r.lab2024) : "") : esc(NUANCE_LAB[nu] || nu)} ${f1(v)}`).join(" · ") : "";
const NUMERIC_DESC = ["p_lfi", "p_left", "q_lfi", "q_oth", "p_lfi_local", "p_lfi_ru", "h2024_G", "h2017_LFI", "mel", "ext_plus_G"];
const $n = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fold = (s) => String(s ?? "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
// Une probabilité n'est jamais affichée comme une certitude : >99 % et <1 % aux bords.
// Sous 10 %, une décimale : le seuil « en jeu » (5 %) doit rester lisible (4,5 % ≠ 5 %).
const pct = (p) => p == null ? "—" : p >= 0.995 ? ">99 %" : (p > 0 && p < 0.005) ? "<1 %" : p < 0.1 ? (p * 100).toLocaleString("fr-FR", { maximumFractionDigits: 1 }) + " %" : Math.round(p * 100) + " %";
const f1 = (x) => x == null ? "—" : x.toLocaleString("fr-FR", { maximumFractionDigits: 1 });
const f2 = (x) => x == null ? "—" : x.toLocaleString("fr-FR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const stateKey = (r) => r.posture || r.group;

async function negLoad() {
  const r = await fetch("data/negotiation.json?v=2");
  if (!r.ok) throw new Error("negotiation.json");
  NEG.data = await r.json();
  NEG.rows = NEG.data.rows.map((x) => ({ ...x,
    depute: x.depute && x.depute.nom ? `${x.depute.prenom} ${x.depute.nom}` : "",
    depGroup: x.depute ? x.depute.groupe : "" }));
  // Force réelle : option extérieure par part nationale LFI (grille précalculée).
  const sp = NEG.data.split;
  NEG.share = sp.near;
  $n("share").innerHTML = sp.shares.map((k) => `<option value="${k}"${k === sp.near ? " selected" : ""}>${Math.round(+k * 100)} %${k === sp.near ? " (sondages)" : ""}</option>`).join("");
  $n("share").addEventListener("change", () => { NEG.share = $n("share").value; negApplyShare(); negTable(); });
  negApplyShare();
  const deps = [...new Set(NEG.rows.map((x) => x.depGroup).filter(Boolean))].sort();
  $n("dep").innerHTML = '<option value="">Tous groupes</option>' +
    deps.map((g) => `<option value="${esc(g)}">${esc(GROUP_LAB_DEP[g] || g)}</option>`).join("");
  for (const id of ["group", "posture", "filter", "dep"]) $n(id).addEventListener("input", negTable);
  document.querySelectorAll("th button[data-sort]").forEach((b) => b.addEventListener("click", () => {
    const k = b.dataset.sort;
    if (NEG.sort === k) NEG.asc = !NEG.asc; else { NEG.sort = k; NEG.asc = !NUMERIC_DESC.includes(k); }
    negTable();
  }));
  $n("export").disabled = false;
  $n("export").addEventListener("click", negExport);
  const mn = $n("mel-nat"); if (mn) mn.textContent = Math.round(NEG.data.presidential.national.LFI * 100) + " %";
  negResizers();
  negTooltips();
  negMethods(); negTable();
}

// Infobulle immédiate (le « title » natif du navigateur tarde une seconde et ne s'affiche pas
// partout) : un seul élément flottant, alimenté par data-tip. Les title des en-têtes sont
// convertis au chargement ; les pastilles du tableau sont rendues avec data-tip directement.
function negTooltips() {
  const tt = document.createElement("div"); tt.className = "tt"; tt.setAttribute("role", "tooltip"); document.body.appendChild(tt);
  document.querySelectorAll("#neg-table thead button[title]").forEach((el) => { el.dataset.tip = el.getAttribute("title"); el.removeAttribute("title"); });
  const show = (el) => {
    tt.textContent = el.dataset.tip; tt.style.display = "block";
    const r = el.getBoundingClientRect(), w = tt.offsetWidth;
    let x = r.left + r.width / 2 - w / 2; x = Math.max(8, Math.min(window.innerWidth - w - 8, x));
    const below = r.bottom + 8, above = r.top - tt.offsetHeight - 8;
    tt.style.left = x + "px"; tt.style.top = (above > 8 ? above : below) + "px";
  };
  const hide = () => { tt.style.display = "none"; };
  document.addEventListener("mouseover", (e) => { const el = e.target.closest("[data-tip]"); if (el) show(el); else hide(); });
  document.addEventListener("focusin", (e) => { const el = e.target.closest("[data-tip]"); if (el) show(el); });
  document.addEventListener("focusout", hide);
  document.addEventListener("scroll", hide, true);
}

// Posture = valeur du siège (p_left) × qui peut se passer de l'accord (q_lfi vs q_oth, les deux
// options extérieures mesurées à l'identique sur les deux pôles). TROIS probabilités du même
// Monte-Carlo, aucun chiffre de la partie droite du tableau, aucun seuil nouveau.
// Miroir exact de `posture` (src/negotiation_2027.py).
function negPosture(group, qL, qO, pLeft) {
  if (group === "acquis" || group === "hors_union" || group === "non_mesure"
      || qL == null || qO == null || pLeft == null) return null;
  const lev = NEG.data.split.leverage_q;
  if (pLeft < NEG.data.params.p_min) return "rien";
  if (qL >= lev) return "exiger";
  return qO >= lev ? "monnaie" : "obtenir";
}

// Applique la part LFI choisie : recopie les deux options extérieures de la grille (les deux
// pôles bougent ensemble quand la part nationale change) et recalcule la posture.
function negApplyShare() {
  const b = NEG.data.split.by_share[NEG.share];
  NEG.rows.forEach((r, i) => {
    r.q_lfi = r.pub ? b.q_lfi[i] : null;
    r.q_oth = r.pub ? b.q_oth[i] : null;
    r.posture = negPosture(r.group, r.q_lfi, r.q_oth, r.p_left);
  });
}

function negMethods() {
  const d = NEG.data, k = d.params.label_effect_k, le = d.params.label_effect, ci = d.params.label_effect_ci95;
  $n("m-label").innerHTML = `Le parti de chaque candidat·e d'union 2024 est connu par la répartition des circonscriptions du Nouveau Front populaire. Dans les <b>${d.params.label_effect_n} duels</b> candidat·e d'union contre RN de 2024 (centre-droit éliminé), un·e candidat·e LFI a récupéré <b>${Math.round(k.fi * 100)} %</b> des voix libérées au 1<sup>er</sup> tour, contre ${Math.round(k.union * 100)} % pour le·la candidat·e moyen·ne de l'union (${d.params.label_effect_n_fi} duels LFI ; écart ${f2(le.cd2l_delta_lfi)}, intervalle bootstrap à 95 % de ${f2(ci[0])} à ${f2(ci[1])} ; présent dans les trois terciles de force de la gauche, donc pas compensé dans les bastions ; ${f1(le.margin_effect_lfi_pts_inscrits)} point d'inscrits sur la marge de 2<sup>nd</sup> tour à marge de 1<sup>er</sup> tour égale). C'est ce taux de report propre à LFI que le modèle de sièges applique pour calculer la chance d'une candidature LFI. Un taux de report, pas un taux de victoire : le fait que LFI ait reçu des circonscriptions plus dures en 2024 ne le biaise pas.`;
  $n("m-groups").innerHTML = `<b>Acquis</b> : député·e sortant·e du groupe LFI (${d.groups.acquis}). <b>En jeu</b> : une candidature LFI a au moins ${Math.round(d.params.p_min * 100)} % de chance de gagner le siège (${d.groups.en_jeu}) — le classement (#) ne porte que sur elles, par chance décroissante. <b>Sans enjeu</b> : moins de ${Math.round(d.params.p_min * 100)} % (${d.groups.sans_enjeu}). <b>Gauche hors union</b> : siège tenu par un·e élu·e de gauche hors de l'union (${d.groups.hors_union || 0}, voir ci-dessous). <b>Non mesurée</b> : hors nomenclature de blocs (${d.groups.non_mesure}).`;
  const sp = d.split;
  $n("m-posture").innerHTML = `<b>Trois probabilités simulées, une posture</b> : toutes les colonnes « notre lecture » sortent du même Monte-Carlo (${d.params.draws} tirages par circonscription, incertitude nationale des sondages + erreur locale du modèle). <b>Chance de la gauche unie</b> : le siège est-il gagné par une candidature d'union moyenne (report moyen mesuré en 2024, étiquette quelconque) ? C'est la valeur du siège, indépendamment de qui le porte. <b>Sans accord</b> : si la gauche se divise (LFI d'un côté, PS·Place publique·Écologistes·PCF de l'autre, part nationale de LFI réglable de ${Math.round(+sp.shares[0] * 100)} à ${Math.round(+sp.shares[sp.shares.length - 1] * 100)} %, sondages : ${Math.round(sp.default_share * 100)} %, motif local du vote Mélenchon à la dernière présidentielle), lequel des deux pôles atteint seul le second tour ? Les deux sont mesurés <b>à l'identique</b> : c'est l'option extérieure de chacun, celle qui dit qui peut se passer de l'accord. <b>Postures</b>, déduites de ces trois chiffres et d'aucun autre — jamais des colonnes de droite : <b>rien à jouer</b> = la gauche unie gagne < ${Math.round(d.params.p_min * 100)} % ; <b>exiger</b> = elle gagne ≥ ${Math.round(d.params.p_min * 100)} % et LFI seule atteint le 2<sup>nd</sup> tour ≥ ${Math.round(sp.leverage_q * 100)} % (LFI tient le siège sans l'accord) ; <b>monnaie d'échange</b> = elle gagne ≥ ${Math.round(d.params.p_min * 100)} %, LFI seule < ${Math.round(sp.leverage_q * 100)} % mais le reste de la gauche seul ≥ ${Math.round(sp.leverage_q * 100)} % (le partenaire est chez lui : LFI ne peut pas exiger ce siège et devra le céder — et comme c'est un vrai siège, le céder a un prix) ; <b>obtenir</b> = elle gagne ≥ ${Math.round(d.params.p_min * 100)} % et aucun des deux pôles n'atteint seul le 2<sup>nd</sup> tour (personne ne peut se passer de l'accord : le siège se gagne à la table). Le survol d'une pastille redonne les quatre chiffres de la ligne.`;
  const hors = NEG.rows.filter((r) => r.group === "hors_union");
  $n("m-hors").innerHTML = `${hors.length} circonscription${hors.length > 1 ? "s" : ""} — ${hors.map((r) => `${esc(r.id)} ${esc(r.nm)} (${esc(r.depute)}, ${esc(GROUP_LAB_DEP[r.depGroup] || r.depGroup)})`).join(" ; ")} — ont été gagnées en 2024 par une candidature codée à gauche mais hors de l'union, dont le ou la titulaire ne siège pas dans un groupe de gauche. Le bloc de gauche prédit y inclut ses voix, qui ne se reporteraient pas sur une candidature d'union : la chance affichée surestime ce qu'obtiendrait LFI. Ces sièges sont sortis du classement et signalés.`;
  const hs = d.history || {};
  $n("src-history").innerHTML = "Résultats passés : " + Object.values(hs).map((e) => `<b>${esc(e.label)}</b> (gauche nationale ${f1(e.national_G)} %${e.national_LFI != null ? `, LFI ${f1(e.national_LFI)} %` : ""}) — ${e.source.startsWith("https://") ? `<a href="${esc(e.source)}" target="_blank" rel="noopener">fichier officiel</a>` : esc(e.source)}`).join(" ; ") + ". Les scores sont en % des suffrages exprimés, tous candidats au dénominateur ; « LFI seule » n'est séparable qu'en 2017 (nuance FI), la gauche étant unie au 1<sup>er</sup> tour en 2022 et 2024. Sous « Gauche 2024 », la ventilation distingue la candidature NFP (nuance UG, avec le parti qui la portait) des candidatures de gauche hors NFP (nuances DVG, EXG, ECO…). Sous « Gauche 2027, calcul simple », le total est partagé entre les partis : LFI = part nationale de LFI dans la gauche (filtre, sondages par défaut) + écart local du vote Mélenchon 2022, bornée entre ${Math.round(d.params.rad_clip[0] * 100)} et ${Math.round(d.params.rad_clip[1] * 100)} % ; le reste va à PS, Écologistes et PCF selon la seule enquête qui les sépare (${esc(d.parties_2027.polls.join(", "))} : PS ${Math.round(d.parties_2027.shares_rest.PS * 100)} %, Écologistes ${Math.round(d.parties_2027.shares_rest.EELV * 100)} %, PCF ${Math.round(d.parties_2027.shares_rest.PCF * 100)} % du reste), chaque part décalée de l'écart local de son·sa candidat·e à la présidentielle 2022 (Hidalgo, Jadot, Roussel ; France entière : ${Math.round(d.presidential.national.PS * 100)} / ${Math.round(d.presidential.national.EELV * 100)} / ${Math.round(d.presidential.national.PCF * 100)} % du vote des trois), bornée, renormalisée. « Mélenchon dans le vote de gauche » : part brute de Mélenchon parmi les candidat·es de gauche au 1<sup>er</sup> tour de la présidentielle 2022 ; France entière ${Math.round(d.presidential.national.LFI * 100)} %";
  const ns = d.params.nat_sigma, ls = d.params.local_sigma, m = d.scenario.means;
  $n("m-unc").innerHTML = `Scénario « ${esc(d.scenario.label)} », ancre sondages G ${f1(m.G)} · C+D ${f1(m.CD)} · ED ${f1(m.ED)} %, abstention ${f1(m.AB)} %. <b>${d.params.draws} tirages</b> Monte-Carlo : le niveau national de chaque bloc est tiré autour de l'ancre avec l'erreur historique des sondages législatifs (écart-type G ${f1(ns.G)}, C+D ${f1(ns.CD)}, ED ${f1(ns.ED)} points — validation croisée 2002→2022), puis chaque circonscription reçoit son erreur locale (G ${f1(ls.G)}, C+D ${f1(ls.CD)}, ED ${f1(ls.ED)} points, la même que la fourchette de la carte). Un classement fait à un seul réglage de curseur ne survivrait pas à une réunion ; celui-ci moyenne sur ce que les sondages peuvent se tromper. Le CSV donne aussi la chance avec l'erreur locale seule et sous « droites unies » (LR refuse le front républicain).`;
}

function negVisible() {
  const g = $n("group").value, po = $n("posture").value, dep = $n("dep").value, term = fold($n("filter").value.trim());
  const rows = NEG.rows.filter((r) => (!g || r.group === g) && (!po || r.posture === po) && (!dep || r.depGroup === dep)
    && (!term || fold(`${r.id} ${r.nm} ${r.dept} ${r.depute} ${r.lab2024 || ""}`).includes(term)));
  const k = NEG.sort, s = NEG.asc ? 1 : -1;
  const val = (r) => k === "posture" ? POSTURE_ORDER[stateKey(r)]
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
  const rows = negVisible();
  const pb = (p, cls) => p == null ? "—" : `<span class="pb ${cls}"><span>${pct(p)}</span><i><b style="width:${Math.round(p * 100)}%"></b></i></span>`;
  const state = (r) => r.posture ? `<span class="pos pos-${r.posture}" tabindex="0" data-tip="${esc(postureTipRow(r))}">${POSTURE_LAB[r.posture]}</span>`
    : `<span class="grp g-${r.group}" tabindex="0" data-tip="${esc(GROUP_TIP[r.group])}"><i></i>${GROUP_LAB[r.group]}</span>`;
  const dep = (r) => r.depute ? `<span class="trunc" data-tip="${esc(r.depute)} (${esc(GROUP_LAB_DEP[r.depGroup] || r.depGroup)})">${esc(r.depute)}</span> <span class="dim">${esc(GROUP_LAB_DEP[r.depGroup] || r.depGroup)}</span>` : "—";
  $n("rows").innerHTML = rows.map((r) => `<tr>
    <td class="num dim">${r.rank ?? ""}</td>
    <th scope="row" class="left"><span class="trunc" data-tip="${esc(r.id)} ${esc(r.nm)}">${esc(r.id)} <span class="dim">${esc(r.nm)}</span></span></th>
    <td class="left">${state(r)}</td>
    <td class="num">${pb(r.p_lfi, "")}</td>
    <td class="num">${pb(r.p_left, "")}</td>
    <td class="num">${pb(r.q_lfi, "q")}</td>
    <td class="num">${pb(r.q_oth, "q")}</td>
    <td class="left args first">${dep(r)}</td>
    <td class="left args">${r.lab2024 ? esc(r.lab2024) : "—"} <span class="dim">${r.union_won_2024 == null ? "" : r.union_won_2024 ? "· gagné" : "· perdu"}</span></td>
    <td class="num args">${r.h2024_G == null ? "—" : f1(r.h2024_G) + " %"}${r.h2024_parts ? `<small class="parts">${parts24(r)}</small>` : ""}</td>
    <td class="num args">${r.h2017_LFI == null ? "—" : f1(r.h2017_LFI) + " %"}</td>
    <td class="num args">${r.mel == null ? `<span class="dim" data-tip="Non calculable ici : communes à cheval sur plusieurs circonscriptions (Paris, Marseille, Lyon…)">—</span>` : Math.round(r.mel * 100) + " %"}</td>
    <td class="num args">${r.ext_plus_G == null ? "—" : f1(r.ext_plus_G) + " %" + `<small class="parts">${(() => { const o = split27(r); return ["LFI", ...REST].map((p) => `${p === "LFI" ? "LFI" : REST_LAB[p]} ${f1(o[p])}`).join(" · "); })()}</small>`}</td>
  </tr>`).join("");
  document.querySelectorAll("th[data-key]").forEach((th) => {
    const on = th.dataset.key === NEG.sort;
    th.setAttribute("aria-sort", on ? (NEG.asc ? "ascending" : "descending") : "none");
    const b = th.querySelector("button"); b.textContent = b.textContent.replace(/ [↑↓]$/, "") + (on ? (NEG.asc ? " ↑" : " ↓") : "");
  });
  $n("status").textContent = `${rows.length} circonscription${rows.length > 1 ? "s" : ""} affichée${rows.length > 1 ? "s" : ""} · part de LFI dans la gauche : ${Math.round(+NEG.share * 100)} %.`;
}

// Colonnes redimensionnables : une poignée sur le bord droit de chaque en-tête de colonne. Les
// largeurs sont posées sur des <col> (la première ligne d'en-tête fusionne des cellules, donc
// en largeur fixe le navigateur ne lirait pas celles des <th>). Au premier glisser, le tableau
// passe en largeur fixe avec ses largeurs courantes, puis seule la colonne tirée change et le
// tableau s'élargit d'autant (le conteneur défile si besoin). Double-clic : largeurs d'origine.
function negResizers() {
  const table = $n("neg-table"), ths = [...table.querySelectorAll("thead tr:last-child th")];
  const cg = document.createElement("colgroup");
  const cols = ths.map(() => cg.appendChild(document.createElement("col")));
  table.insertBefore(cg, table.firstElementChild);
  const reset = () => { cols.forEach((c) => { c.style.width = ""; }); table.style.width = ""; table.classList.remove("resized"); };
  ths.forEach((th, idx) => {
    const h = document.createElement("span"); h.className = "rs"; h.title = "Tirer pour redimensionner (double-clic : réinitialiser)"; th.appendChild(h);
    h.addEventListener("mousedown", (e) => {
      e.preventDefault();
      if (!table.classList.contains("resized")) {
        ths.forEach((t, k) => { cols[k].style.width = t.offsetWidth + "px"; });
        table.style.width = table.offsetWidth + "px"; table.classList.add("resized");
      }
      const x0 = e.clientX, w0 = th.offsetWidth, tw0 = table.offsetWidth; h.classList.add("on");
      const move = (ev) => {
        const w = Math.max(48, w0 + ev.clientX - x0);
        cols[idx].style.width = w + "px"; table.style.width = (tw0 + w - w0) + "px";
      };
      const up = () => { h.classList.remove("on"); window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
      window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
    });
    h.addEventListener("dblclick", reset);
  });
}

function negExport() {
  const rows = negVisible(), d = NEG.data;
  const head = ["rang", "circo", "nom", "dept", "groupe", "posture", "chance_depute_lfi", "chance_gauche_unie", "part_lfi_force_reelle", "lfi_seule_qualifiee_2nd_tour", "reste_gauche_seul_qualifie_2nd_tour",
    "chance_lfi_incertitude_locale_seule", "chance_lfi_droites_unies", "pred_G", "pred_CD", "pred_ED", "depute", "groupe_depute", "parti_nfp_2024",
    "siege_union_2024", "gauche_2024", "gauche_2022", "lfi_seule_2017", "gauche_2017", "melenchon_part_du_vote_de_gauche_2022",
    "gauche_2024_plus_evolution_nationale", "lfi_2027_calcul_simple", "ps_2027_calcul_simple", "eelv_2027_calcul_simple", "pcf_2027_calcul_simple", "nfp_2024", "gauche_hors_nfp_2024", "gauche_2024_fois_evolution_nationale", "inscrits"];
  const q = (v) => v == null ? "" : /[";\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v);
  const lines = [`# negotiation 2027 (LFI) — scenario ${d.scenario.key} ; tirages ${d.params.draws} ; part LFI ${NEG.share} ; tri ${NEG.sort} ${NEG.asc ? "asc" : "desc"} ; filtre groupe "${$n("group").value}" posture "${$n("posture").value}" texte "${$n("filter").value}"`,
    head.join(";")].concat(rows.map((r) => [r.rank, r.id, r.nm, r.dept, GROUP_LAB[r.group], r.posture ? POSTURE_LAB[r.posture] : "", r.p_lfi, r.p_left, NEG.share, r.q_lfi, r.q_oth,
      r.p_lfi_local, r.p_lfi_ru, r.pred && r.pred.G, r.pred && r.pred.CD, r.pred && r.pred.ED, r.depute, r.depGroup, r.lab2024,
      r.union_won_2024 == null ? "" : (r.union_won_2024 ? 1 : 0), r.h2024_G, r.h2022_G, r.h2017_LFI, r.h2017_G,
      r.mel, r.ext_plus_G, ...(split27(r) ? ["LFI", "PS", "EELV", "PCF"].map((p) => split27(r)[p]) : ["", "", "", ""]),
      r.h2024_parts ? (r.h2024_parts.UG ?? 0) : "", r.h2024_parts ? Math.round(Object.entries(r.h2024_parts).filter(([k]) => k !== "UG").reduce((a, [, v]) => a + v, 0) * 100) / 100 : "",
      r.ext_mult_G, r.ins].map(q).join(";")));
  const blob = new Blob(["﻿" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "negociation_2027_lfi.csv"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

negLoad().catch((e) => { $n("status").textContent = "Erreur de chargement : " + e.message; });
