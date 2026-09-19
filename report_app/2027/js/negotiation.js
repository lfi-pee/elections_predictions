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
  get acquis() { return "Député·e sortant·e du groupe LFI : hors répartition, le siège se défend au lieu de se négocier."; },
  get en_jeu() { return `Chance d'élire un·e député·e LFI ≥ ${pctInt(P().p_min)} : une candidature LFI peut gagner ce siège, à demander.`; },
  get sans_enjeu() { return `Chance d'élire un·e député·e LFI < ${pctInt(P().p_min)} : imprenable pour LFI.`; },
  get hors_union() { return "Siège gagné en 2024 par une candidature de gauche hors de l'union, dont le·la titulaire ne siège pas dans un groupe de gauche : toutes les chances affichées sur cette ligne créditent la gauche des voix de ce·tte sortant·e, qui ne suivraient pas forcément une candidature commune. À lire comme un maximum."; },
  get non_mesure() { return "Aucune chance n'est publiée sur cette ligne, donc aucune posture : trop de voix y vont à des forces (régionalistes, autonomistes) qui n'entrent dans aucun des trois blocs que le modèle prédit — gauche, centre-droit, extrême droite. C'est le cas en Corse et en Guyane, et dans quelques circonscriptions isolées."; } };
const POSTURE_LAB = { exiger: "Exiger", obtenir: "Obtenir", monnaie: "Monnaie d'échange", rien: "Rien à jouer" };
const POSTURE_ORDER = { exiger: 0, obtenir: 1, monnaie: 2, rien: 3, acquis: 4, hors_union: 5, non_mesure: 6 };
// Le texte des postures dit « une chance sur deux » : il suppose le seuil leverage_q = 50 %.
// Aucun « Calcul » ne cite la chance de la gauche unie : elle n'entre pas dans la règle, et l'y
// nommer rendait chaque pastille falsifiable avec les chiffres de sa propre ligne (une circo
// pouvait satisfaire mot pour mot le calcul annoncé pour « monnaie » en affichant « rien »).
const POSTURE_TIP = {
  get exiger() { return `Calcul : chance d'élire un·e député·e LFI ≥ ${pctInt(P().p_min)} ET, sans accord, LFI seule au 2nd tour ≥ ${pctInt(SP().leverage_q)}, quel que soit le chiffre du reste de la gauche. Sens : même sans accord, LFI est au 2nd tour plus d'une fois sur deux — elle n'a pas besoin de céder ce siège.`; },
  get obtenir() { return `Calcul : chance d'élire un·e député·e LFI ≥ ${pctInt(P().p_min)} ET, sans accord, NI LFI seule NI le reste de la gauche seul n'atteint ${pctInt(SP().leverage_q)}. Sens : divisée, aucune des deux parts de la gauche ne peut imposer sa candidature — LFI peut demander ce siège à la table, pas l'exiger.`; },
  get monnaie() { return `Calcul : chance d'élire un·e député·e LFI ≥ ${pctInt(P().p_min)}, sans accord LFI seule < ${pctInt(SP().leverage_q)} MAIS reste de la gauche seul ≥ ${pctInt(SP().leverage_q)}. Sens : sans accord, c'est le reste de la gauche qui reste dans la course, pas LFI. LFI devra céder ce siège : en échange, elle peut en demander un autre.`; },
  get rien() { return `Calcul : chance d'élire un·e député·e LFI < ${pctInt(P().p_min)} — siège imprenable pour LFI. Sens : rien à demander, rien à monnayer.`; } };
// Infobulle d'une pastille : la règle, puis les SEULS chiffres qui la déclenchent — la chance de
// LFI (qui décide « rien à jouer ») et les deux « Sans accord » (qui décident le reste). Chacun
// porte le nom exact de sa colonne, et les deux « Sans accord » sont glosés : ils mesurent une
// présence au 2nd tour, pas un siège gagné. La chance de la gauche unie est donnée à part, parce
// qu'elle n'entre PAS dans la règle et parce que sa juxtaposition avec celle de LFI invite une
// lecture fausse : ce n'est pas la chance du partenaire, c'est celle d'une candidature d'union
// moyenne, toute étiquette confondue. Aucune chance n'est calculée pour un autre parti.
const postureTipRow = (r) => `${POSTURE_TIP[r.posture]} Ici : chance d'élire un·e député·e LFI ${pct(r.p_lfi)} · sans accord, LFI seule au 2nd tour ${pct(r.q_lfi)} · sans accord, reste de la gauche seul ${pct(r.q_oth)}. Pour situer, hors règle : la gauche unie gagne ce siège ${pct(r.p_left)} du temps avec une candidature d'union moyenne — une étiquette quelconque, pas celle du partenaire.`;
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
const NUMERIC_DESC = ["arg", "p_lfi", "p_left", "q_lfi", "q_oth", "p_lfi_local", "p_lfi_ru", "h2024_G", "h2017_LFI", "mel", "ext_plus_G"];
const $n = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fold = (s) => String(s ?? "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
// Une probabilité n'est jamais affichée comme une certitude : >99 % et <1 % aux bords.
// Sous 10 %, une décimale : le seuil « en jeu » (5 %) doit rester lisible (4,5 % ≠ 5 %).
const pct = (p) => p == null ? "—" : p >= 0.995 ? ">99 %" : p < 0.005 ? "<1 %" : p < 0.1 ? (p * 100).toLocaleString("fr-FR", { maximumFractionDigits: 1 }) + " %" : Math.round(p * 100) + " %";
const f1 = (x) => x == null ? "—" : x.toLocaleString("fr-FR", { maximumFractionDigits: 1 });
// Nombre SIGNÉ : le signe porte l'information (un décalage appliqué vers le haut ou vers
// le bas), et il s'écrit avec le moins typographique du reste du paragraphe, pas le tiret.
const f1s = (x) => x == null ? "—" : (x > 0 ? "+" : x < 0 ? "−" : "±") + f1(Math.abs(x));
const f2 = (x) => x == null ? "—" : x.toLocaleString("fr-FR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const stateKey = (r) => r.posture || r.group;
// Erreur de SIMULATION sur une probabilité tirée n fois, et elle seule : de combien le chiffre
// bougerait si on relançait le Monte-Carlo avec une autre graine. L'incertitude de l'ÉLECTION
// (sondages + erreur locale) est DÉJÀ intégrée dans le point estimé — la remettre autour serait
// la compter deux fois. Wilson et non l'approximation normale : les valeurs affichées touchent
// les bords, et à p = 1 « p ± z√(p(1−p)/n) » donnerait ±0. Miroir exact de `wilson` (Python).
const wilsonCI = (p, n, z) => {
  if (!(n > 0)) return [0, 1];
  const d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d;
  const h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  // Serré sur le point estimé : aux bords, le flottant laisse un résidu du mauvais côté, et un
  // intervalle qui n'encadre pas son propre chiffre est le défaut qu'on corrige ailleurs ici.
  return [Math.min(p, Math.max(0, c - h)), Math.max(p, Math.min(1, c + h))];
};
// Bornes affichées sous le chiffre. Arrondies à l'entier comme lui, sauf sous 10 % où la colonne
// garde une décimale (le seuil des 5 % doit rester lisible dans l'intervalle aussi).
const ciTxt = (p) => {
  if (p == null || !NEG.data) return "";
  const [lo, hi] = wilsonCI(p, NEG.data.params.draws, NEG.data.params.ci_z ?? 1.96);
  // Même convention que le chiffre au-dessus : jamais de certitude affichée, une décimale sous
  // 10 %. Deux bornes qui s'écrivent pareil se fondent en un jeton — l'intervalle est alors plus
  // étroit que ce que l'affichage distingue, et « 100–100 » sous « >99 % » serait un démenti.
  // Le seuil bas ne s'écrit PAS « v > 0 » : une borne exactement nulle est le cas le plus
  // fréquent de la colonne (363 des 571 q_lfi servis valent 0), et la laisser passer dans la
  // branche décimale affichait « 0,0–<1 » — deux conventions dans un même jeton, dont l'une est
  // la certitude que la légende promet de ne jamais afficher. minimumFractionDigits fixe la
  // décimale : sans lui « 1,0 » s'écrirait « 1 » ici et « 1,0 » côté Python, et le miroir se
  // briserait sur la moitié basse de la colonne.
  const f = (v) => v >= 0.995 ? ">99" : v < 0.005 ? "<1"
    : p < 0.1 ? (v * 100).toLocaleString("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 })
              : String(Math.round(v * 100));
  const a = f(lo), b = f(hi);
  return a === b ? a : `${a}–${b}`;
};

// ——— Argumentaire : ce qu'on dit à la table, avec les seuls faits publics de la ligne ————————
// La posture (« notre lecture ») choisit le REGISTRE ; les phrases, elles, ne citent que la partie
// droite du tableau — résultats du ministère, investiture NFP 2024, vote Mélenchon 2022, calcul
// simple — que le partenaire a sous les yeux lui aussi. Aucune probabilité simulée n'y entre :
// rien qu'il puisse contester, et notre lecture reste à nous.
// La règle qui commande le registre : ceci est une négociation de REVENDICATION, pas de prix. Une
// circonscription qu'on ne demande pas est une circonscription qu'on n'a pas (une répartition, une
// date, pas de second tour), donc feindre l'indifférence sur un siège qu'on veut le perd. Sur un
// siège qu'on demande, jamais un chiffre qui nous affaiblit ; sur un siège qu'on ne demande pas,
// avancer soi-même le chiffre défavorable — il ne coûte aucun siège, et c'est lui qui fait croire
// les autres. Le seul « bluff » utile est le miroir : rendre cher ce qu'on cédera de toute façon.
const NATH = (y, k) => (NEG.data.history[y] || {})["national_" + k];
const UNION_DEP = { SOC: 1, ECOS: 1, GDR: 1 };

// Un fait = un chiffre public de la ligne et son point de comparaison national. `kind` dit ce
// qu'il porte : « lfi », la force de LFI ici ; « seat », la valeur du siège pour la gauche ;
// « price », le fait que la gauche détient déjà ce siège — qui fait le prix de ce qu'on cède, mais
// ne plaide jamais pour le réclamer (là, c'est le partenaire qui le sortira). `pro` : favorable ou
// non à ce que porte le fait.
function argFacts(r) {
  const out = [], add = (kind, pro, text, src) => out.push({ kind, pro, text, src });
  const melN = NEG.data.presidential.national.LFI, g24N = NATH("2024", "G"), l17N = NATH("2017", "LFI"), g27N = NEG.data.scenario.means.G;
  // « L'union a investi PS, PS a gagné » et « le siège est tenu par un·e PS, un·e sortant·e de
  // l'union ne se déloge pas » sont le même fait ; la seconde le dit en portant l'argument. Quand
  // les deux s'appliquent, une seule phrase — sinon la carte sert deux fois la même objection.
  const unionDep = !!(r.depGroup && UNION_DEP[r.depGroup]);
  if (r.lab2024 && r.union_won_2024 != null) {
    const p = NFP_PARTY[r.lab2024] || r.lab2024, won = r.union_won_2024;
    if (r.lab2024 === "FI") add("lfi", won, `Législatives 2024 : l'union a investi LFI ici, et le siège a été ${won ? "gagné" : "perdu"}.`, "nfp24");
    else if (!won) add("lfi", true, `Législatives 2024 : l'union a investi ${p} ici, et le siège a été perdu.`, "nfp24");
    else if (!unionDep) add("lfi", false, `Législatives 2024 : l'union a investi ${p} ici, et ${p} a gagné le siège.`, "nfp24");
    if (won) add("price", true, `Ce siège, l'union l'a gagné en 2024 avec une candidature ${p} : ce n'est pas un siège perdu d'avance que nous cédons.`, "nfp24");
  }
  if (r.mel != null && Math.abs(r.mel - melN) >= 0.02) {
    const up = r.mel > melN;
    // Valeur communale (Paris, Marseille, Lyon…) : le fait posé sur la table porte sur la ville
    // entière, pas sur la circonscription — on le dit dans la phrase, sinon elle serait fausse.
    const ou = r.mel_com ? `dans tout ${r.mel_com}` : "ici", la = r.mel_com ? `de ${r.mel_com}` : "d'ici";
    add("lfi", up, up ? `Présidentielle 2022 : Mélenchon prend ${Math.round(r.mel * 100)} % du vote de gauche ${ou}, contre ${Math.round(melN * 100)} % en France — la gauche ${la} est plus insoumise que la moyenne.`
      : `Présidentielle 2022 : Mélenchon ne prend que ${Math.round(r.mel * 100)} % du vote de gauche ${ou}, contre ${Math.round(melN * 100)} % en France.`);
  }
  if (r.h2017_LFI != null && l17N != null && Math.abs(r.h2017_LFI - l17N) >= 2) {
    const up = r.h2017_LFI > l17N;
    // Un score de 0 en 2017 n'est pas un échec : c'est une circonscription sans candidature LFI.
    add("lfi", up, r.h2017_LFI === 0 ? `Législatives 2017 : LFI n'y présentait pas de candidature, là où elle faisait ${f1(l17N)} % en France.`
      : `Législatives 2017, dernier scrutin sous sa propre étiquette : LFI ${f1(r.h2017_LFI)} % ici, contre ${f1(l17N)} % en France.`);
  }
  if (r.depute && r.depGroup) {
    const nom = r.depute, lab = GROUP_LAB_DEP[r.depGroup] || r.depGroup;
    if (r.depGroup === NEG.data.params.lfi_group) add("lfi", true, `Le siège est tenu par ${nom} (LFI) : c'est notre sortant·e.`);
    else if (UNION_DEP[r.depGroup]) add("lfi", false, `Le siège est tenu par ${nom} (${lab}) : un·e sortant·e d'un parti de l'union ne se déloge pas.`);
    else if (r.group === "hors_union") add("lfi", true, `Le siège est tenu par ${nom}, élu·e en 2024 sur une candidature de gauche hors de l'union et aujourd'hui au groupe ${lab} : aucun parti de l'union n'y a de sortant·e à protéger.`);
    else add("lfi", true, `Le siège est tenu par ${nom} (${lab}) : personne à gauche n'y a de sortant·e à protéger.`);
  }
  // Le « calcul simple » est le score 2024 décalé du même nombre de points partout : quand il place la
  // circonscription du même côté de la moyenne nationale que 2024, le poser en deuxième fait ferait
  // compter deux fois le même argument. Une seule phrase porte alors les deux chiffres.
  const has24 = r.h2024_G != null && g24N != null, has27 = r.ext_plus_G != null;
  const pro24 = has24 && r.h2024_G >= g24N, pro27 = has27 && r.ext_plus_G >= g27N;
  if (has24 && has27 && pro24 === pro27)
    add("seat", pro24, `Législatives 2024 : la gauche fait ${f1(r.h2024_G)} % au 1ᵉʳ tour ici, contre ${f1(g24N)} % en France ; en « calcul simple 2027 », ${f1(r.ext_plus_G)} % contre ${f1(g27N)} %.`);
  else {
    if (has24) add("seat", pro24, `Législatives 2024 : la gauche fait ${f1(r.h2024_G)} % au 1ᵉʳ tour ici, contre ${f1(g24N)} % en France.`);
    if (has27) add("seat", pro27, `Calcul simple 2027 : la gauche à ${f1(r.ext_plus_G)} % ici, contre ${f1(g27N)} % en France.`);
  }
  if (has27) {
    const o = split27(r), best = REST.map((p) => [REST_LAB[p], o[p]]).sort((a, b) => b[1] - a[1])[0];
    if (best && Math.abs(o.LFI - best[1]) >= 0.2) add("lfi", o.LFI > best[1], `Calcul simple 2027, part par part : sur les ${f1(r.ext_plus_G)} % de toute la gauche ici, LFI en prend ${f1(o.LFI)} points, ${o.LFI > best[1] ? "devant" : "derrière"} ${best[0]} (${f1(best[1])}) — à ${Math.round(+NEG.share * 100)} % de LFI dans la gauche au national (réglable en haut de page).`, "split27");
  }
  return out;
}

// Registre par posture. Ce qui change d'une posture à l'autre n'est pas le ton : c'est QUELS faits
// on pose, et lesquels on laisse au partenaire. Aucune case du tableau n'a intérêt à se montrer
// faible sur le siège lui-même : même en cédant, le prix qu'on obtient est ce que le partenaire
// croit que ça nous coûte.
function argCard(r) {
  const f = argFacts(r), take = (kind, pro) => f.filter((x) => x.kind === kind && x.pro === pro);
  const lp = take("lfi", true), lc = take("lfi", false), sp = take("seat", true), sc = take("seat", false), pr = take("price", true);
  const c = { head: "", posLabel: "À mettre sur la table", pos: [], oppLabel: "Ce que le partenaire sortira — à préparer, pas à dire", opp: [], foot: "" };
  if (r.posture === "rien") {
    c.head = "Rien à jouer — concéder le premier, et avec le chiffre";
    c.posLabel = "À dire tel quel, avant qu'on nous le sorte";
    c.pos = sc.concat(lc).slice(0, 3); c.oppLabel = ""; c.opp = [];
    c.foot = (c.pos.length ? "" : "Aucun chiffre public ne dit la faiblesse de ce siège : le concéder quand même, sans chiffre à l'appui. ")
      + "Ce siège ne vaut rien pour personne : essayer de le lui vendre ne marchera pas. Le lâcher explicitement ne coûte aucun siège — et c'est ce qui fait croire nos chiffres là où nous demandons un siège.";
  } else if (r.posture === "monnaie") {
    c.head = "Monnaie d'échange — céder, mais faire payer";
    c.posLabel = "Le prix, à mettre sur la table : ce que vaut ce qu'on cède";
    c.pos = pr.concat(sp).slice(0, 3);
    c.oppLabel = "Ce que le partenaire dira pour ne pas payer — à préparer, pas à dire";
    // Le prix dit que le siège se gagne ; l'objection, que le partenaire y est sortant : deux faits
    // distincts, qui tiennent ensemble dans une même carte. Le filtre ne couvre que le cas rare où
    // c'est littéralement la même phrase (union gagnante en 2024, sortant·e passé·e hors union).
    c.opp = lc.filter((x) => !c.pos.some((y) => y.src && y.src === x.src)).slice(0, 2);
    c.foot = (c.pos.length ? "" : "Aucun fait public ne fait ici le prix de ce siège. ")
      + "Ne pas disputer ce siège, et ne jamais dire qu'il ne vaut rien : ce qu'on en tire est ce que le partenaire croit qu'il nous coûte. Se faire payer en circonscriptions « exiger » ou « obtenir ».";
  } else if (r.posture === "exiger" || r.posture === "obtenir") {
    const ex = r.posture === "exiger";
    c.head = ex ? "Exiger — dire pourquoi nous ne céderons pas" : "Obtenir — revendiquer, chiffres à l'appui";
    c.pos = lp.concat(sp).slice(0, 3); c.opp = lc.concat(sc).slice(0, 2);
    c.foot = (c.pos.length ? "" : "Aucun fait public ne porte la demande ici : la revendiquer quand même, mais dans un ensemble de circonscriptions, pas chiffre en main. ")
      + (ex ? `Sans accord, LFI est au 2nd tour ici plus d'une fois sur deux : c'est le seul cas où nous pouvons nous passer de l'accord sur un siège — mais c'est notre lecture, pas un fait public${c.pos.length ? " : ce sont les faits ci-dessus qui portent la demande" : ""}. Porter la demande sans menacer l'accord entier, dont nous avons besoin ailleurs.`
            : "Divisée, ni LFI ni le reste de la gauche n'atteint seul le second tour ici : le siège se gagne à la table, en le demandant."
              + (c.pos.length ? " La répartition ne se fait qu'une fois, à une date : une circonscription qu'on n'a pas revendiquée est perdue." : ""));
  } else if (r.group === "acquis") {
    c.head = "Sortant·e LFI — hors répartition";
    c.posLabel = "Si le siège est remis en cause : à mettre sur la table";
    c.pos = lp.concat(sp).slice(0, 3); c.opp = lc.slice(0, 2);
    c.foot = "Ce siège ne se répartit pas : il se défend.";
  } else if (r.group === "hors_union") {
    c.head = "Gauche hors union — chiffres à manier avec précaution";
    // Le partage par parti du « calcul simple » cite le total de gauche d'ici, gonflé par les voix
    // du·de la sortant·e hors union : c'est exactement ce que le pied de carte interdit d'avancer.
    c.pos = lp.filter((x) => x.src !== "split27").slice(0, 3); c.opp = lc.concat(sc).slice(0, 2);
    c.foot = "Le score de gauche affiché sur cette ligne comprend les voix d'un·e élu·e hors union, qui ne se reporteraient pas sur une candidature commune : l'avancer comme notre force se retourne dès que le partenaire le relève.";
  } else {
    c.head = "Non mesurée — aucune probabilité, donc aucune posture";
    c.pos = lp.concat(sp).slice(0, 3); c.opp = lc.concat(sc).slice(0, 2);
    c.foot = "Trop de voix vont ici à des forces (régionalistes, autonomistes) qui n'entrent dans aucun des trois blocs que le modèle prédit — gauche, centre-droit, extrême droite : le modèle n'a rien à en dire. Les faits publics, eux, restent vrais : ils se posent comme ailleurs.";
  }
  c.pos = c.pos.map((x) => x.text); c.opp = c.opp.map((x) => x.text);
  c.n = c.pos.length;
  // Sur un siège acquis, les faits ne servent que si le siège est contesté : la pastille le dit.
  c.badge = !c.n ? "aucun \u00e0 poser" : c.n + (r.group === "acquis" ? "\u00a0si contest\u00e9" : "\u00a0\u00e0 poser");
  return c;
}
const argHTML = (c) => `<b>${esc(c.head)}</b>`
  + (c.pos.length ? `<u>${esc(c.posLabel)}</u><ul>${c.pos.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>` : "")
  + (c.opp.length ? `<u>${esc(c.oppLabel)}</u><ul>${c.opp.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>` : "")
  + `<i>${esc(c.foot)}</i>`;
const argText = (c) => [c.head, c.pos.length ? c.posLabel + " : " + c.pos.join(" ") : "",
  c.opp.length ? c.oppLabel + " : " + c.opp.join(" ") : "", c.foot].filter(Boolean).join(" — ");

async function negLoad() {
  const r = await fetch("data/negotiation.json?v=3");
  if (!r.ok) throw new Error("negotiation.json");
  NEG.data = await r.json();
  NEG.rows = NEG.data.rows.map((x) => ({ ...x,
    depute: x.depute && x.depute.nom ? `${x.depute.prenom} ${x.depute.nom}` : "",
    depGroup: x.depute ? x.depute.groupe : "" }));
  // Force réelle : option extérieure par part nationale LFI (grille précalculée).
  const sp = NEG.data.split;
  NEG.share = sp.near;
  $n("share").innerHTML = sp.shares.map((k) => `<option value="${k}"${k === sp.near ? " selected" : ""}>${Math.round(+k * 100)} %${k === sp.near ? " (sondages)" : ""}</option>`).join("");
  $n("share").addEventListener("change", () => { NEG.share = $n("share").value; negApplyShare(); negTable(); negMethods(); });
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
    const card = el.dataset.card ? NEG.cards[el.dataset.card] : null;
    tt.classList.toggle("card", !!card);
    if (card) tt.innerHTML = card; else tt.textContent = el.dataset.tip;
    tt.style.display = "block";
    const r = el.getBoundingClientRect(), w = tt.offsetWidth;
    let x = r.left + r.width / 2 - w / 2; x = Math.max(8, Math.min(window.innerWidth - w - 8, x));
    const below = r.bottom + 8, above = r.top - tt.offsetHeight - 8;
    tt.style.left = x + "px"; tt.style.top = (above > 8 ? above : below) + "px";
  };
  const hide = () => { tt.style.display = "none"; };
  document.addEventListener("mouseover", (e) => { const el = e.target.closest("[data-tip],[data-card]"); if (el) show(el); else hide(); });
  document.addEventListener("focusin", (e) => { const el = e.target.closest("[data-tip],[data-card]"); if (el) show(el); });
  document.addEventListener("focusout", hide);
  document.addEventListener("scroll", hide, true);
}

// Posture = la chance de LFI (lue via le groupe) × qui peut se passer de l'accord (q_lfi vs q_oth, les deux
// options extérieures mesurées à l'identique sur les deux pôles). TROIS probabilités du même
// Monte-Carlo, aucun chiffre de la partie droite du tableau, aucun seuil nouveau — plus un
// garde-fou : sur un siège imprenable pour LFI (groupe « sans enjeu »), il n'y a rien à jouer.
// Miroir exact de `posture` (src/negotiation_2027.py).
function negPosture(group, qL, qO) {
  if (group === "acquis" || group === "hors_union" || group === "non_mesure"
      || qL == null || qO == null) return null;
  const lev = NEG.data.split.leverage_q;
  // Aucune posture de DEMANDE sur un siège que LFI ne gagne pas : le groupe « sans enjeu » porte
  // déjà ce verdict (chance LFI < p_min). p_lfi ne sépare jamais exiger/obtenir/monnaie.
  // `p_left` n'est plus un paramètre : la garder revenait à la lire dans ce garde-fou tout en
  // écrivant ici qu'on ne la lit pas — et cette lecture doublait « qO == null ».
  if (group === "sans_enjeu") return "rien";
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
    r.posture = negPosture(r.group, r.q_lfi, r.q_oth);
  });
  // L'argumentaire dépend de la posture ET du calcul simple 2027 : il se refait avec la part.
  NEG.cards = {};
  NEG.rows.forEach((r) => { r.card = argCard(r); NEG.cards[r.id] = argHTML(r.card); });
}

// Combien de circonscriptions sont interchangeables avec une circonscription donnée, au bruit
// de simulation près. SERVI par `negotiation_2027.rank_blur`, pas recalculé ici : le critère
// porte sur un ÉCART (|p̂_i − p̂_j| ≤ z·SE de l'écart), et la SE de l'écart n'est pas déductible
// des deux intervalles marginaux — les circos partagent les tirages nationaux, il faut leur
// covariance par tirage, qui n'existe que côté Python. La version qui comptait les p̂_j tombant
// dans l'intervalle de Wilson de p̂_i répondait à une autre question, et sous-estimait le flou
// (la corrélation du bruit entre circos est nettement sous le 0,5 qui rendrait les deux
// critères équivalents, donc SE de l'écart > demi-largeur marginale).
function rankBlur() {
  return NEG.data.params.rank_blur || null;
}

// Lignes dont l'intervalle traverse un seuil de décision : leur chiffre est indécis, mais
// surtout leur CATÉGORIE l'est (groupe, rang, posture, registre de l'argumentaire). C'est
// précisément ce que les bornes servent à rendre visible, donc la page le compte et le dit.
function straddle() {
  const z = NEG.data.params.ci_z ?? 1.96, nd = NEG.data.params.draws;
  const cross = (v, t) => { if (v == null) return false; const [lo, hi] = wilsonCI(v, nd, z); return lo < t && t < hi; };
  const pm = NEG.data.params.p_min, lv = NEG.data.split.leverage_q;
  const graded = (r) => r.group === "en_jeu" || r.group === "sans_enjeu";
  // La règle lit q_oth SEULEMENT quand q_lfi est sous le seuil : au-dessus, la posture vaut
  // « exiger » quoi que fasse q_oth, et un q_oth à cheval ne décide alors rien. Sans cette
  // seconde condition la page annonçait 7 postures indécises pour 5 réelles à la part 0,55 —
  // le défaut même que le filtre de groupe était censé supprimer.
  const qUndecided = (r) => cross(r.q_lfi, lv) || (r.q_lfi != null && r.q_lfi < lv && cross(r.q_oth, lv));
  return { p: NEG.rows.filter((r) => graded(r) && cross(r.p_lfi, pm)).length,
           q: NEG.rows.filter((r) => r.group === "en_jeu" && qUndecided(r)).length };
}

function negMethods() {
  const d = NEG.data, k = d.params.label_effect_k, le = d.params.label_effect, ci = d.params.label_effect_ci95;
  $n("m-label").innerHTML = `Le parti de chaque candidat·e d'union 2024 est connu par la répartition des circonscriptions du Nouveau Front populaire. Dans les <b>${d.params.label_effect_n} duels</b> candidat·e d'union contre RN de 2024 (centre-droit éliminé), un·e candidat·e LFI a récupéré <b>${Math.round(k.fi * 100)} %</b> des voix libérées au 1<sup>er</sup> tour, contre ${Math.round(k.union * 100)} % pour le·la candidat·e moyen·ne de l'union (${d.params.label_effect_n_fi} duels LFI ; écart ${f2(le.cd2l_delta_lfi)}, intervalle bootstrap à 95 % de ${f2(ci[0])} à ${f2(ci[1])} ; présent dans les trois terciles de force de la gauche, donc pas compensé dans les bastions ; ${f1(le.margin_effect_lfi_pts_inscrits)} point d'inscrits sur la marge de 2<sup>nd</sup> tour à marge de 1<sup>er</sup> tour égale). C'est ce taux de report propre à LFI que le modèle de sièges applique pour calculer la chance d'une candidature LFI. Un taux de report, pas un taux de victoire : le fait que LFI ait reçu des circonscriptions plus dures en 2024 ne le biaise pas.`;
  $n("m-groups").innerHTML = `<b>Acquis</b> : député·e sortant·e du groupe LFI (${d.groups.acquis}). <b>En jeu</b> : une candidature LFI a au moins ${Math.round(d.params.p_min * 100)} % de chance de gagner le siège (${d.groups.en_jeu}) — le classement (#) ne porte que sur elles, par chance décroissante. <b>Sans enjeu</b> : moins de ${Math.round(d.params.p_min * 100)} % (${d.groups.sans_enjeu}) — le même seuil commande la posture, qui y vaut « rien à jouer » : on ne revendique pas un siège qu'on ne gagne pas. Le groupe « sans enjeu » et la posture « rien à jouer » désignent donc exactement les mêmes circonscriptions. <b>Gauche hors union</b> : siège gagné en 2024 par une candidature de gauche hors de l'union (${d.groups.hors_union || 0}, voir ci-dessous). <b>Non mesurée</b> : hors nomenclature de blocs (${d.groups.non_mesure}).`;
  const sp = d.split;
  $n("m-posture").innerHTML = `<b>Quatre probabilités simulées, trois qui font la posture</b> : toutes les colonnes « notre lecture » sortent du même Monte-Carlo (${d.params.draws} tirages par circonscription, incertitude nationale des sondages + erreur locale du modèle). La règle lit <b>la chance d'élire un·e député·e LFI</b> et les deux <b>Sans accord</b>. <b>Chance de la gauche unie</b> : le siège est-il gagné par une candidature d'union moyenne (report moyen mesuré en 2024, étiquette quelconque) ? C'est la valeur du siège, indépendamment de qui le porte — affichée pour situer, mais elle n'entre pas dans la règle, et ce n'est pas la chance du partenaire : aucune chance n'est calculée pour un autre parti. <b>Sans accord</b> : si la gauche se divise (LFI d'un côté, PS·Place publique·Écologistes·PCF de l'autre, part nationale de LFI réglable de ${Math.round(+sp.shares[0] * 100)} à ${Math.round(+sp.shares[sp.shares.length - 1] * 100)} %, sondages : ${Math.round(sp.default_share * 100)} %, motif local du vote Mélenchon à la dernière présidentielle), lequel des deux pôles atteint seul le second tour ? Les deux sont mesurés <b>à l'identique</b> : c'est l'option extérieure de chacun, celle qui dit qui peut se passer de l'accord. <b>Postures</b>, déduites des seules colonnes « notre lecture » — jamais des colonnes de droite : <b>rien à jouer</b> = LFI gagne le siège < ${Math.round(d.params.p_min * 100)} % du temps (groupe « sans enjeu ») — on ne revendique pas un siège qu'on ne gagne pas ; <b>exiger</b> = LFI le gagne ≥ ${Math.round(d.params.p_min * 100)} % et LFI seule atteint le 2<sup>nd</sup> tour ≥ ${Math.round(sp.leverage_q * 100)} % (LFI tient le siège sans l'accord) ; <b>monnaie d'échange</b> = LFI le gagne ≥ ${Math.round(d.params.p_min * 100)} %, LFI seule < ${Math.round(sp.leverage_q * 100)} % mais le reste de la gauche seul ≥ ${Math.round(sp.leverage_q * 100)} % (le partenaire est chez lui : LFI ne peut pas exiger ce siège et devra le céder — et comme c'est un vrai siège, le céder a un prix) ; <b>obtenir</b> = LFI le gagne ≥ ${Math.round(d.params.p_min * 100)} % et aucun des deux pôles n'atteint seul le 2<sup>nd</sup> tour (personne ne peut se passer de l'accord : le siège se gagne à la table). Le survol d'une posture redonne sa règle et les chiffres de la ligne. Un seul seuil à ${Math.round(d.params.p_min * 100)} %, et il porte sur la chance de LFI : c'est le même que celui du groupe « sans enjeu », de sorte que le classement et la posture ne peuvent plus se contredire.`;
  const hors = NEG.rows.filter((r) => r.group === "hors_union");
  $n("m-hors").innerHTML = `${hors.length} circonscription${hors.length > 1 ? "s" : ""} — ${hors.map((r) => `${esc(r.id)} ${esc(r.nm)} (${esc(r.depute)}, ${esc(GROUP_LAB_DEP[r.depGroup] || r.depGroup)})`).join(" ; ")} — ont été gagnées en 2024 par une candidature codée à gauche mais hors de l'union, dont le ou la titulaire ne siège pas dans un groupe de gauche. Le bloc de gauche prédit y inclut ses voix, qui ne se reporteraient pas sur une candidature d'union : la chance affichée surestime ce qu'obtiendrait LFI. Ces sièges sont sortis du classement et signalés.`;
  const hs = d.history || {};
  $n("src-history").innerHTML = "Résultats passés : " + Object.values(hs).map((e) => `<b>${esc(e.label)}</b> (gauche nationale ${f1(e.national_G)} %${e.national_LFI != null ? `, LFI ${f1(e.national_LFI)} %` : ""}) — ${e.source.startsWith("https://") ? `<a href="${esc(e.source)}" target="_blank" rel="noopener">fichier officiel</a>` : esc(e.source)}`).join(" ; ") + ". Les scores sont en % des suffrages exprimés, tous candidats au dénominateur ; « LFI seule » n'est séparable qu'en 2017 (nuance FI), la gauche étant unie au 1<sup>er</sup> tour en 2022 et 2024. Sous « Gauche 2024 », la ventilation distingue la candidature NFP (nuance UG, avec le parti qui la portait) des candidatures de gauche hors NFP (nuances DVG, EXG, ECO…). Sous « Gauche 2027, calcul simple », le total est partagé entre les partis : LFI = part nationale de LFI dans la gauche (filtre, sondages par défaut) + écart local du vote Mélenchon 2022, bornée entre ${Math.round(d.params.rad_clip[0] * 100)} et ${Math.round(d.params.rad_clip[1] * 100)} % ; le reste va à PS, Écologistes et PCF selon la seule enquête qui les sépare (${esc(d.parties_2027.polls.join(", "))} : PS ${Math.round(d.parties_2027.shares_rest.PS * 100)} %, Écologistes ${Math.round(d.parties_2027.shares_rest.EELV * 100)} %, PCF ${Math.round(d.parties_2027.shares_rest.PCF * 100)} % du reste), chaque part décalée de l'écart local de son·sa candidat·e à la présidentielle 2022 (Hidalgo, Jadot, Roussel ; France entière : ${Math.round(d.presidential.national.PS * 100)} / ${Math.round(d.presidential.national.EELV * 100)} / ${Math.round(d.presidential.national.PCF * 100)} % du vote des trois), bornée, renormalisée. « Mélenchon dans le vote de gauche » : part brute de Mélenchon parmi les candidat·es de gauche au 1<sup>er</sup> tour de la présidentielle 2022 ; France entière ${Math.round(d.presidential.national.LFI * 100)} %. Les résultats présidentiels s'arrêtent à la commune : une circonscription entièrement située à l'intérieur d'une grande commune (les 18 de Paris, les 7 de Marseille, Lyon, Nice, Toulouse…) porte la valeur de CETTE commune, marquée « commune » et identique pour toutes les circonscriptions de la ville";
  const ns = d.params.nat_sigma, nsd = d.params.nat_sigma_delivered || d.params.nat_sigma;
  const ls = d.params.local_sigma, m = d.scenario.means, rb = rankBlur(), st = straddle();
  const halfPt = (wilsonCI(0.5, d.params.draws, d.params.ci_z ?? 1.96)[1] - 0.5) * 100;
  const nsh = d.params.nat_shift, nbr = d.params.nat_bias_raw, nc = d.params.nat_corr, pgp = d.params.paired_gap;
  $n("m-unc").innerHTML = `Scénario « ${esc(d.scenario.label)} », ancre sondages G ${f1(m.G)} · C+D ${f1(m.CD)} · ED ${f1(m.ED)} %, abstention ${f1(m.AB)} %. <b>${d.params.draws} tirages</b> Monte-Carlo. Le niveau national n'est pas tiré autour de l'ancre BRUTE : aux ${d.params.nat_n} législatives mesurées (2002→2024), les sondages ont sur-estimé l'extrême droite de ${f1(nbr.ED)} points en moyenne, et sous-estimé le centre-droit de ${f1(-nbr.CD)}. Une erreur de centre qu'aucune largeur d'intervalle ne rattrape : on la corrige donc, mais <b>rétractée vers zéro</b> (facteur ${f2(d.params.nat_shrink)} — ${d.params.nat_n} scrutins ne suffisent pas à parier sur le chiffre brut, et 2024 est parti dans l'autre sens), soit ED ${f1s(nsh.ED)} · C+D ${f1s(nsh.CD)} · G ${f1s(nsh.G)} points appliqués à l'ancre. Autour de ce centre corrigé, l'erreur est tirée dans sa <b>loi mesurée</b> : écart-type G ${f1(ns.G)}, C+D ${f1(ns.CD)}, ED ${f1(ns.ED)} points, et surtout les corrélations réelles entre blocs (G/C+D ${f2(nc.GCD)}, G/ED ${f2(nc.GED)}, C+D/ED ${f2(nc.CDED)}) — une part surestimée est prise à une autre, et c'est ce partage qui décide des qualifications au 2<sup>nd</sup> tour. Les trois blocs se partageant un total fixé, l'erreur somme à zéro et le tirage respecte ce total sans renormalisation : ce qui sort vaut exactement ce qui est visé (mesuré : G ${f1(nsd.G)}, C+D ${f1(nsd.CD)}, ED ${f1(nsd.ED)}). Puis chaque circonscription reçoit son erreur locale (G ${f1(ls.G)}, C+D ${f1(ls.CD)}, ED ${f1(ls.ED)} points, la même que la fourchette de la carte). Un classement fait à un seul réglage de curseur ne survivrait pas à une réunion ; celui-ci moyenne sur ce que les sondages peuvent se tromper. Le CSV donne aussi la chance avec l'erreur locale seule et sous « droites unies » (LR refuse le front républicain). <b>Sous chaque chiffre, ses bornes à 95 %</b> : c'est l'erreur de simulation, et elle seule — de combien le chiffre bougerait si on relançait les ${d.params.draws} tirages avec une autre graine (intervalle de Wilson, ±${f1(halfPt)} point au plus large, moins près des bords). Ce n'est <b>pas</b> l'incertitude de l'élection : celle-là — erreur des sondages, erreur locale du modèle — est déjà intégrée dans le chiffre lui-même, et la remettre autour la compterait deux fois. Une réserve sur ce « déjà intégrée » : l'<b>abstention est tenue fixe</b> à la référence du scénario, comme le petit bloc « Autre », et la correction de biais est elle-même incertaine — c'est pourquoi elle est rétractée. Rien de cela n'est dans les bornes. Enfin, ces bornes valent pour chaque chiffre <b>pris seul</b> : les quatre sortent des mêmes tirages, donc un écart entre deux d'entre eux est bien mieux connu que la somme de leurs bornes — sur le plus grand écart d'étiquette du jeu, ±${f2(pgp.paired_pt)} point apparié contre ±${f2(pgp.sum_pt)} en sommant les deux demi-largeurs, soit ${f1(pgp.sum_pt / pgp.paired_pt)}× plus serré. ${rb ? `Conséquence sur le <b>rang</b>, qui est un ordre et pas une mesure fine : une circonscription est interchangeable avec ${rb.med} voisines en médiane, ${rb.max} au plus.` : ""} Et là où un intervalle traverse un seuil de décision, ce n'est pas seulement le chiffre qui est indécis, c'est la <b>catégorie</b> : ${st.p} circonscription${st.p > 1 ? "s" : ""} de part et d'autre du seuil des ${Math.round(d.params.p_min * 100)} % (donc du groupe, du rang et de la posture) et ${st.q} de part et d'autre de celui des ${Math.round(d.split.leverage_q * 100)} % (donc de la posture), au réglage courant. C'est sans effet sur ce à quoi le classement sert : le nombre de sièges espérés d'un paquet de circonscriptions ne dépend pas de leur ordre à l'intérieur du paquet.`;
}

function negVisible() {
  const g = $n("group").value, po = $n("posture").value, dep = $n("dep").value, term = fold($n("filter").value.trim());
  const rows = NEG.rows.filter((r) => (!g || r.group === g) && (!po || r.posture === po) && (!dep || r.depGroup === dep)
    && (!term || fold(`${r.id} ${r.nm} ${r.dept} ${r.depute} ${r.lab2024 || ""}`).includes(term)));
  const k = NEG.sort, s = NEG.asc ? 1 : -1;
  const val = (r) => k === "posture" ? POSTURE_ORDER[stateKey(r)]
    : k === "arg" ? (r.card ? r.card.n : -1) : k === "lab2024" ? (r.lab2024 || "") : k === "rank" ? (r.rank ?? 1e9) : r[k];
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
  const z = NEG.data.params.ci_z ?? 1.96, nd = NEG.data.params.draws;
  const pb = (p, cls) => {
    if (p == null) return "—";
    const [lo, hi] = wilsonCI(p, nd, z);
    return `<span class="pb ${cls}"><span>${pct(p)}</span><i><u style="left:${lo * 100}%;width:${(hi - lo) * 100}%"></u><b style="width:${Math.round(p * 100)}%"></b></i><small data-tip="Intervalle à 95 % sur le BRUIT DE SIMULATION : de combien ce chiffre bougerait si on relançait les ${nd} tirages avec une autre graine. Ce n'est pas l'incertitude de l'élection — celle-là (erreur des sondages, erreur locale du modèle) est déjà intégrée dans le chiffre lui-même. Bornes de ce chiffre PRIS SEUL : les quatre colonnes sortent des mêmes tirages, donc l'écart entre deux d'entre elles est bien mieux connu que la somme de leurs bornes.">${ciTxt(p)}</small></span>`;
  };
  const state = (r) => r.posture ? `<span class="pos pos-${r.posture}" tabindex="0" data-tip="${esc(postureTipRow(r))}">${POSTURE_LAB[r.posture]}</span>`
    : `<span class="grp g-${r.group}" tabindex="0" data-tip="${esc(GROUP_TIP[r.group])}"><i></i>${GROUP_LAB[r.group]}</span>`;
  const arg = (r) => `<span class="arg${r.card.n ? "" : " none"}" tabindex="0" data-card="${esc(r.id)}">${esc(r.card.badge)}</span>`;
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
    <td class="num args">${r.mel == null ? `<span class="dim" data-tip="Non calculable ici : aucun résultat présidentiel rattachable à cette circonscription.">—</span>`
      : Math.round(r.mel * 100) + " %" + (r.mel_com ? `<small class="parts com" data-tip="Les résultats présidentiels s'arrêtent à la commune, et cette circonscription est entièrement à l'intérieur de ${esc(r.mel_com)} : le chiffre est celui de ${esc(r.mel_com)} en entier. Toutes les circonscriptions de ${esc(r.mel_com)} portent donc le même ; il ne dit rien de leurs écarts internes.">valeur de ${esc(r.mel_com)}</small>` : "")}</td>
    <td class="num args">${r.ext_plus_G == null ? "—" : f1(r.ext_plus_G) + " %" + `<small class="parts">${(() => { const o = split27(r); return ["LFI", ...REST].map((p) => `${p === "LFI" ? "LFI" : REST_LAB[p]} ${f1(o[p])}`).join(" · "); })()}</small>`}</td>
    <td class="left args">${arg(r)}</td>
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
  // Chaque probabilité exportée part avec ses bornes à 95 % (bruit de simulation) : un CSV qui
  // ne porterait que le point laisserait croire à une précision que les tirages ne donnent pas.
  const ci4 = (p) => { if (p == null) return ["", "", ""];
    const [lo, hi] = wilsonCI(p, d.params.draws, d.params.ci_z ?? 1.96);
    return [p, Math.round(lo * 1e4) / 1e4, Math.round(hi * 1e4) / 1e4]; };
  const head = ["rang", "circo", "nom", "dept", "groupe", "posture", "chance_depute_lfi", "chance_depute_lfi_ic95_bas", "chance_depute_lfi_ic95_haut",
    "chance_gauche_unie", "chance_gauche_unie_ic95_bas", "chance_gauche_unie_ic95_haut", "part_lfi_force_reelle",
    "lfi_seule_qualifiee_2nd_tour", "lfi_seule_ic95_bas", "lfi_seule_ic95_haut",
    "reste_gauche_seul_qualifie_2nd_tour", "reste_gauche_seul_ic95_bas", "reste_gauche_seul_ic95_haut",
    "chance_lfi_incertitude_locale_seule", "chance_lfi_droites_unies", "pred_G", "pred_CD", "pred_ED", "depute", "groupe_depute", "parti_nfp_2024",
    "siege_union_2024", "gauche_2024", "gauche_2022", "lfi_seule_2017", "gauche_2017", "melenchon_part_du_vote_de_gauche_2022", "melenchon_valeur_de_la_commune",
    "gauche_2024_plus_evolution_nationale", "lfi_2027_calcul_simple", "ps_2027_calcul_simple", "eelv_2027_calcul_simple", "pcf_2027_calcul_simple", "nfp_2024", "gauche_hors_nfp_2024", "gauche_2024_fois_evolution_nationale", "inscrits", "argumentaire"];
  const q = (v) => v == null ? "" : /[";\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v);
  const lines = [`# negotiation 2027 (LFI) — scenario ${d.scenario.key} ; tirages ${d.params.draws} ; part LFI ${NEG.share} ; tri ${NEG.sort} ${NEG.asc ? "asc" : "desc"} ; filtre groupe "${$n("group").value}" posture "${$n("posture").value}" texte "${$n("filter").value}"`,
    `# les colonnes _ic95_ sont des bornes à 95 % sur le BRUIT DE SIMULATION (pas l'incertitude de l'élection, déjà intégrée dans le chiffre), et elles valent pour chaque chiffre PRIS SEUL : les quatre probabilités sortent des mêmes ${d.params.draws} tirages, donc un écart entre deux d'entre elles est bien mieux connu que la somme de leurs bornes — sur le plus grand écart d'étiquette du jeu, ±${f2(d.params.paired_gap.paired_pt)} point apparié contre ±${f2(d.params.paired_gap.sum_pt)} en sommant les deux demi-largeurs.`,
    head.join(";")].concat(rows.map((r) => [r.rank, r.id, r.nm, r.dept, GROUP_LAB[r.group], r.posture ? POSTURE_LAB[r.posture] : "", ...ci4(r.p_lfi), ...ci4(r.p_left), NEG.share, ...ci4(r.q_lfi), ...ci4(r.q_oth),
      r.p_lfi_local, r.p_lfi_ru, r.pred && r.pred.G, r.pred && r.pred.CD, r.pred && r.pred.ED, r.depute, r.depGroup, r.lab2024,
      r.union_won_2024 == null ? "" : (r.union_won_2024 ? 1 : 0), r.h2024_G, r.h2022_G, r.h2017_LFI, r.h2017_G,
      r.mel, r.mel_com || "", r.ext_plus_G, ...(split27(r) ? ["LFI", "PS", "EELV", "PCF"].map((p) => split27(r)[p]) : ["", "", "", ""]),
      r.h2024_parts ? (r.h2024_parts.UG ?? 0) : "", r.h2024_parts ? Math.round(Object.entries(r.h2024_parts).filter(([k]) => k !== "UG").reduce((a, [, v]) => a + v, 0) * 100) / 100 : "",
      r.ext_mult_G, r.ins, argText(r.card)].map(q).join(";")));
  const blob = new Blob(["﻿" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "negociation_2027_lfi.csv"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

negLoad().catch((e) => { $n("status").textContent = "Erreur de chargement : " + e.message; });
