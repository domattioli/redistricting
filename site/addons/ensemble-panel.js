/* Neutral baseline panel for the Draw and Fair Lines toys (add-on).

   Shows where the maps on the page sit among thousands of neutral plans sampled with ReCom
   (DeFord, Duchin & Solomon 2019). The ensembles are precomputed by `python -m redistricting_addons.batch`
   into data/ensembles/<ST>.json (plain ReCom) and <ST>.county20.json (county-weighted).

   It only adds. Draw: reads the page's global `S` and wraps its global refreshAll() to redraw afterwards.
   Fair Lines keeps its state inside a closure, so the page publishes it with one added line
   (`window.__RD_PAGE = { S, isFairNow }`) and this panel redraws whenever the page redraws its seat ladder.
   Loaded by one <script> line at the end of each page; delete that line to remove it. */
(function () {
  'use strict';
  const PAGE = window.__RD_PAGE || null;
  const SS = PAGE ? PAGE.S : (typeof S !== 'undefined' ? S : null);
  if (!SS) return;
  const BASE = typeof __RD_DATA_BASE === 'string' ? __RD_DATA_BASE : '../data/';
  const VARIANTS = [
    { id: 'county20', file: '.county20', name: 'Counties kept whole', note: 'county-internal links weighted 20×, so samples split about as few counties as human-drawn maps' },
    { id: 'plain', file: '', name: 'Plain ReCom', note: 'no county preference' },
  ];
  let variant = 'county20';
  try { const v = localStorage.getItem('ens-variant'); if (VARIANTS.some(x => x.id === v)) variant = v; } catch (e) { /* storage blocked */ }
  // which units count as neighbours: the site's own graph (default) or shared borders from the Census boundary
  // file (data/graphs/<ST>.json, redistricting_addons.graphs); ensembles <ST>[.county20].census.json
  const GRAPHS = [
    { id: 'site', file: '', name: 'Site neighbours', note: 'the toys’ own neighbour graph (a triangulation of block-group centres)' },
    { id: 'census', file: '.census', name: 'Census borders', note: 'neighbours = block groups sharing a border on land (Census 2020 boundary file); islands joined by recorded water links' },
  ];
  let graphSel = 'site';
  try { const g = localStorage.getItem('ens-graph'); if (GRAPHS.some(x => x.id === g)) graphSel = g; } catch (e) { /* storage blocked */ }
  const censusEdges = {};  // abbr -> Int32Array of i, j pairs (shared borders + water links)

  /* ---------- page adapters ---------- */
  const isDraw = !!document.getElementById('dlist'), isFair = !!document.getElementById('ladder');
  if (!isDraw && !isFair) return;

  function complete(labels, k) {
    if (!labels) return false;
    for (let i = 0; i < labels.length; i++) if (labels[i] < 0 || labels[i] >= k) return false;
    return true;
  }
  const ADAPTER = isDraw ? {
    hook: 'refreshAll',
    mount() {
      const score = document.getElementById('score').closest('section');
      const sec = document.createElement('section');
      sec.className = 'card ens'; sec.style.gridColumn = '1 / -1'; sec.style.marginTop = '12px';
      score.after(sec);
      return sec;
    },
    busy: () => !!SS.running,
    context() {
      if (!SS.data) return null;
      const plans = [{ name: 'Your map', labels: SS.labels, role: 'main', missing: 'Draw every district to place your map.' }];
      if (SS.sol && SS.sol.labels) plans.push({ name: (typeof SOLNAME === 'object' && SOLNAME[SS.sol.mode]) || 'Solver', labels: SS.sol.labels, role: 'alt' });
      plans.push({ name: 'Official', labels: SS.officialLabels, role: 'official' });
      return { abbr: SS.data.abbr, data: SS.data, k: SS.k, el: SS.el, swing: 0, plans };
    },
  } : {
    observe: 'ladder',
    mount() {
      const score = document.getElementById('scoreTitle').closest('.panel');
      const div = document.createElement('div');
      div.className = 'panel ens';
      score.after(div);
      return div;
    },
    busy: () => !!SS.runner,
    context() {
      if (!SS.data) return null;
      const fair = PAGE && typeof PAGE.isFairNow === 'function' && SS.k === SS.data.seats && PAGE.isFairNow();
      const plans = [{ name: fair ? 'Fair map' : 'Your map', labels: SS.labels, role: 'main' }];
      if (!fair && SS.fair) plans.push({ name: 'Fair map', labels: SS.fair, role: 'alt' });
      if (SS.data.current) plans.push({ name: 'Official', labels: SS.data.current, role: 'official' });
      return { abbr: SS.data.abbr, data: SS.data, k: SS.k, el: SS.election, swing: SS.swing || 0, plans };
    },
  };

  /* ---------- data ---------- */
  const files = {};
  function load(path) {
    if (!(path in files)) files[path] = fetch(BASE + path).then(r => (r.ok ? r.json() : null)).catch(() => null);
    return files[path];
  }
  const edgeCache = new WeakMap();
  function edgesOf(data) {
    if (graphSel === 'census' && censusEdges[data.abbr]) return censusEdges[data.abbr];
    let e = edgeCache.get(data);
    if (e) return e;
    const seen = new Set(), out = [];
    const add = (a, b) => { if (a === b) return; if (a > b) [a, b] = [b, a]; const key = a * data.n + b; if (!seen.has(key)) { seen.add(key); out.push(a, b); } };
    for (const list of [data.edges || [], data.bridges || []]) for (let i = 0; i + 1 < list.length; i += 2) add(list[i], list[i + 1]);
    e = Int32Array.from(out); edgeCache.set(data, e);
    return e;
  }
  /* log of the number of spanning trees, summed over districts (Kirchhoff's matrix-tree theorem), as in
     redistricting_addons.ensemble.log_spanning_trees. Each district (or each piece of a district in pieces) is
     ordered by reverse Cuthill-McKee and its reduced Laplacian factored by envelope Cholesky:
     log det = 2 * sum(log L_ii). Higher = more compact. Cached per plan. */
  const nbrCache = new WeakMap();  // data -> { site: {start, adj}, census: {start, adj} }
  function neighbours(data) {
    const slot = nbrCache.get(data) || {};
    nbrCache.set(data, slot);
    let A = slot[graphSel];
    if (A) return A;
    const e = edgesOf(data), deg = new Int32Array(data.n);
    for (let i = 0; i < e.length; i++) deg[e[i]]++;
    const start = new Int32Array(data.n + 1);
    for (let i = 0; i < data.n; i++) start[i + 1] = start[i] + deg[i];
    const fill = start.slice(0, data.n), adj = new Int32Array(e.length);
    for (let i = 0; i < e.length; i += 2) { adj[fill[e[i]]++] = e[i + 1]; adj[fill[e[i + 1]]++] = e[i]; }
    A = { start, adj }; slot[graphSel] = A;
    return A;
  }
  const treeCache = new Map();
  function planKey(labels) { let h = 2166136261 >>> 0; for (let i = 0; i < labels.length; i++) h = Math.imul(h ^ (labels[i] + 1), 16777619) >>> 0; return graphSel + ':' + labels.length + ':' + h; }
  function logTrees(data, labels) {
    const key = planKey(labels);
    if (treeCache.has(key)) return treeCache.get(key);
    const { start, adj } = neighbours(data), n = data.n;
    const same = (u, v) => labels[u] === labels[v];
    const seen = new Uint8Array(n), pos = new Int32Array(n).fill(-1);
    let total = 0;
    const bfs = (src, inComp, sortByDeg, deg) => {
      const order = [src], mark = new Set([src]);
      for (let i = 0; i < order.length; i++) {
        const u = order[i], next = [];
        for (let p = start[u]; p < start[u + 1]; p++) { const v = adj[p]; if (inComp(v) && !mark.has(v)) { mark.add(v); next.push(v); } }
        if (sortByDeg) next.sort((a, b) => deg.get(a) - deg.get(b));
        for (const v of next) order.push(v);
      }
      return order;
    };
    for (let s = 0; s < n; s++) {
      if (seen[s]) continue;
      // one connected piece of one district
      const comp = bfs(s, v => same(v, s), false);
      comp.forEach(v => { seen[v] = 1; });
      if (comp.length <= 1) continue;
      const inComp = new Set(comp), deg = new Map();
      for (const u of comp) { let d = 0; for (let p = start[u]; p < start[u + 1]; p++) if (inComp.has(adj[p])) d++; deg.set(u, d); }
      const far = bfs(comp[comp.length - 1], v => inComp.has(v), false);           // a peripheral start node
      const order = bfs(far[far.length - 1], v => inComp.has(v), true, deg).reverse(); // reverse Cuthill-McKee
      const m = order.length - 1;                                                      // drop the last node (reduced Laplacian)
      order.forEach((v, i) => { pos[v] = i; });
      const first = new Int32Array(m), rows = new Array(m);
      for (let i = 0; i < m; i++) {
        const u = order[i]; let f = i;
        for (let p = start[u]; p < start[u + 1]; p++) { const j = pos[adj[p]]; if (j >= 0 && j < m && inComp.has(adj[p]) && j < f) f = j; }
        first[i] = f; rows[i] = new Float64Array(i - f + 1);
        for (let p = start[u]; p < start[u + 1]; p++) { const j = pos[adj[p]]; if (j >= 0 && j < i && inComp.has(adj[p])) rows[i][j - f] = -1; }
        rows[i][i - f] = deg.get(u);
      }
      let logdet = 0;
      for (let i = 0; i < m; i++) {
        const Ri = rows[i], fi = first[i];
        for (let j = fi; j <= i; j++) {
          const Rj = rows[j], fj = first[j], k0 = Math.max(fi, fj);
          let s2 = Ri[j - fi];
          for (let q = k0; q < j; q++) s2 -= Ri[q - fi] * Rj[q - fj];
          if (j < i) Ri[j - fi] = s2 / Rj[j - fj];
          else { const d = Math.sqrt(s2); Ri[i - fi] = d; logdet += Math.log(d); }
        }
      }
      total += 2 * logdet;
      order.forEach(v => { pos[v] = -1; });
    }
    if (treeCache.size > 30) treeCache.clear();
    treeCache.set(key, total);
    return total;
  }

  function stats(data, labels, k, el) {
    const E = data.elections[el], D = new Float64Array(k), R = new Float64Array(k);
    for (let i = 0; i < data.n; i++) { D[labels[i]] += E.d[i]; R[labels[i]] += E.r[i]; }
    let seats = 0; const share = [];
    for (let j = 0; j < k; j++) { const t = D[j] + R[j]; share.push(t > 0 ? D[j] / t : 0.5); if (D[j] > R[j]) seats++; }
    share.sort((a, b) => a - b);
    const e = edgesOf(data); let cut = 0;
    for (let i = 0; i < e.length; i += 2) if (labels[e[i]] !== labels[e[i + 1]]) cut++;
    const first = new Map(), split = new Set();
    for (let i = 0; i < data.n; i++) {
      if (!(data.pop[i] > 0)) continue;
      const c = data.cty[i], f = first.get(c);
      if (f === undefined) first.set(c, labels[i]); else if (f !== labels[i]) split.add(c);
    }
    return { seats, share, cut, splits: split.size, trees: () => logTrees(data, labels) };
  }

  /* Social ties kept (Meta's Social Connectedness Index), as redistricting_addons.sci.ties_kept: friendship weight
     w_ij = SCI_ij * P_i * P_j between different counties; a pair is kept in proportion to how much of both counties'
     people share a district (split counties count fractionally). Percent of all cross-county weight. */
  const sciFiles = {}, sciW = {};
  function sciWeights(data, sci) {
    if (sciW[data.abbr]) return sciW[data.abbr];
    const C = sci.counties.length, P = new Float64Array(C);
    for (let i = 0; i < data.n; i++) P[data.cty[i]] += data.pop[i];
    const out = {};
    for (const name of ['adjusted', 'raw']) {
      const w = new Float64Array(sci[name].length); let t = 0, n = 0;
      for (let i = 0; i < C; i++) for (let j = i + 1; j < C; j++, n++) { w[n] = sci[name][n] * P[i] * P[j]; t += 2 * w[n]; }
      for (let x = 0; x < w.length; x++) w[x] = t > 0 ? 2 * w[x] / t : 0;
      out[name] = w;
    }
    return (sciW[data.abbr] = { C, w: out });
  }
  function sciKept(data, sci, labels, k) {
    const { C, w } = sciWeights(data, sci), fr = new Float64Array(C * k), tot = new Float64Array(C);
    for (let i = 0; i < data.n; i++) { fr[data.cty[i] * k + labels[i]] += data.pop[i]; tot[data.cty[i]] += data.pop[i]; }
    for (let c = 0; c < C; c++) if (tot[c] > 0) for (let d = 0; d < k; d++) fr[c * k + d] /= tot[c];
    const res = {};
    for (const name in w) {
      let s = 0, n = 0;
      for (let i = 0; i < C; i++) for (let j = i + 1; j < C; j++, n++) {
        if (!w[name][n]) continue;
        let dot = 0; for (let d = 0; d < k; d++) dot += fr[i * k + d] * fr[j * k + d];
        s += w[name][n] * dot;
      }
      res[name] = 100 * s;
    }
    return res;
  }
  function tiesBand(q, v) {  // more ties kept = higher
    const P = [5, 25, 50, 75, 95];
    if (v > q[4]) return 'keeps more ties than 95% of plans';
    if (v < q[0]) return 'keeps fewer ties than 95% of plans';
    for (let i = 0; i < 4; i++) if (v <= q[i + 1]) return `between the ${P[i]}th and ${P[i + 1]}th percentile`;
    return '';
  }

  /* ---------- drawing ---------- */
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const fmt = n => Math.round(n).toLocaleString('en-US');
  const pct = x => (x >= 99.95 ? '100' : x > 0 && x < 0.05 ? '<0.1' : x.toFixed(1)) + '%';
  const MARK = { main: 'var(--accent)', alt: 'var(--ink-2)', official: 'var(--official, var(--ink))' };

  function seatChart(E, k, placed) {
    const hist = E.seats_d_histogram, total = Object.values(hist).reduce((a, b) => a + b, 0);
    const keys = Object.keys(hist).map(Number);
    const prop = Math.round(E.statewide_d_share * k);
    const lo = Math.min(...keys, ...placed.map(p => p.st.seats), prop), hi = Math.max(...keys, ...placed.map(p => p.st.seats), prop);
    const n = hi - lo + 1, W = 320, H = 150, left = 8, right = 8, top = 34, bot = 22;
    const bw = (W - left - right) / n, maxF = Math.max(...keys.map(s => hist[s] / total));
    const x = s => left + (s - lo) * bw;
    let g = '';
    for (let s = lo; s <= hi; s++) {
      const f = (hist[s] || 0) / total, h = (H - top - bot) * f / maxF;
      g += `<rect x="${(x(s) + bw * 0.12).toFixed(1)}" y="${(H - bot - h).toFixed(1)}" width="${(bw * 0.76).toFixed(1)}" height="${Math.max(h, f > 0 ? 1 : 0).toFixed(1)}" fill="var(--ink-2)" opacity=".45"><title>${s} D seats: ${pct(100 * f)} of plans</title></rect>`;
      g += `<text x="${(x(s) + bw / 2).toFixed(1)}" y="${H - 6}" text-anchor="middle">${s}</text>`;
    }
    const px = x(prop) + bw / 2;
    g += `<line x1="${px}" x2="${px}" y1="${top - 6}" y2="${H - bot}" stroke="var(--ink)" stroke-dasharray="3 3" opacity=".7"/><text x="${px + 3}" y="${top - 8}">proportional (${prop})</text>`;
    const stack = {};
    for (const p of placed) {
      const s = p.st.seats, i = (stack[s] = (stack[s] || 0) + 1) - 1, cx = x(s) + bw / 2, cy = 12 + i * 9;
      g += `<path d="M${(cx - 5).toFixed(1)} ${cy - 4}h10l-5 7z" fill="${MARK[p.role]}"><title>${esc(p.name)}: ${s} D seats</title></path>`;
    }
    return `<svg class="ens-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Democratic seats across neutral plans">${g}</svg>`;
  }

  function boxPlot(E, k, placed) {
    const Q = E.sorted_d_share_quantiles;
    let lo = 1, hi = 0;
    for (const q of Q) { lo = Math.min(lo, q[0]); hi = Math.max(hi, q[4]); }
    for (const p of placed) for (const s of p.st.share) { lo = Math.min(lo, s); hi = Math.max(hi, s); }
    lo = Math.max(0, Math.floor((lo - 0.02) * 20) / 20); hi = Math.min(1, Math.ceil((hi + 0.02) * 20) / 20);
    const W = Math.max(320, 30 + k * 11), H = 170, left = 30, right = 6, top = 8, bot = 18, cw = (W - left - right) / k;
    const y = v => top + (1 - (v - lo) / (hi - lo)) * (H - top - bot), xc = j => left + (j + 0.5) * cw;
    let g = '';
    for (let v = Math.ceil(lo * 10) / 10; v <= hi + 1e-9; v += 0.1) g += `<line x1="${left}" x2="${W - right}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}" stroke="var(--rule)"/><text x="${left - 3}" y="${(y(v) + 3).toFixed(1)}" text-anchor="end">${Math.round(v * 100)}%</text>`;
    if (lo < 0.5 && hi > 0.5) g += `<line x1="${left}" x2="${W - right}" y1="${y(0.5).toFixed(1)}" y2="${y(0.5).toFixed(1)}" stroke="var(--ink)" opacity=".6"/>`;
    const bw = Math.min(10, cw * 0.6);
    Q.forEach((q, j) => {
      const c = xc(j);
      g += `<g><title>District rank ${j + 1}: neutral plans 5–95% = ${(q[0] * 100).toFixed(1)}–${(q[4] * 100).toFixed(1)}% D</title>` +
        `<line x1="${c.toFixed(1)}" x2="${c.toFixed(1)}" y1="${y(q[0]).toFixed(1)}" y2="${y(q[4]).toFixed(1)}" stroke="var(--ink-2)"/>` +
        `<rect x="${(c - bw / 2).toFixed(1)}" y="${y(q[3]).toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(1, y(q[1]) - y(q[3])).toFixed(1)}" fill="var(--panel-2)" stroke="var(--ink-2)"/>` +
        `<line x1="${(c - bw / 2).toFixed(1)}" x2="${(c + bw / 2).toFixed(1)}" y1="${y(q[2]).toFixed(1)}" y2="${y(q[2]).toFixed(1)}" stroke="var(--ink)"/></g>`;
    });
    for (const p of placed) {
      if (p.role === 'alt') continue;
      p.st.share.forEach((s, j) => {
        const c = xc(j), cy = y(s);
        g += p.role === 'official'
          ? `<circle cx="${c.toFixed(1)}" cy="${cy.toFixed(1)}" r="3.2" fill="none" stroke="${MARK.official}" stroke-width="1.6"><title>${esc(p.name)}, rank ${j + 1}: ${(s * 100).toFixed(1)}% D</title></circle>`
          : `<circle cx="${c.toFixed(1)}" cy="${cy.toFixed(1)}" r="2.8" fill="${MARK.main}"><title>${esc(p.name)}, rank ${j + 1}: ${(s * 100).toFixed(1)}% D</title></circle>`;
      });
    }
    g += `<text x="${left}" y="${H - 4}">least Democratic</text><text x="${W - right}" y="${H - 4}" text-anchor="end">most Democratic</text>`;
    return `<svg class="ens-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="District vote shares by rank against neutral plans">${g}</svg>`;
  }

  function cutBand(q, v) {  // only quantiles are stored for cut edges
    const P = [5, 25, 50, 75, 95];
    if (v < q[0]) return 'more compact than 95% of plans';
    if (v > q[4]) return 'less compact than 95% of plans';
    for (let i = 0; i < 4; i++) if (v <= q[i + 1]) return `between the ${P[i]}th and ${P[i + 1]}th percentile`;
    return '';
  }
  function treeBand(q, v) {  // more spanning trees = more compact
    const P = [5, 25, 50, 75, 95];
    if (v > q[4]) return 'more compact than 95% of plans';
    if (v < q[0]) return 'less compact than 95% of plans';
    for (let i = 0; i < 4; i++) if (v <= q[i + 1]) return `between the ${P[i]}th and ${P[i + 1]}th percentile`;
    return '';
  }
  function splitsPhrase(hist, v) {
    let fewer = 0, more = 0, t = 0;
    for (const [s, c] of Object.entries(hist)) { t += c; if (+s < v) fewer += c; else if (+s > v) more += c; }
    if (fewer === t) return 'more than every sampled plan';
    if (more === t) return 'fewer than every sampled plan';
    return fewer >= more ? `${pct(100 * fewer / t)} of plans split fewer` : `${pct(100 * more / t)} of plans split more`;
  }

  /* ---------- panel ---------- */
  const root = ADAPTER.mount();
  const style = document.createElement('style');
  style.textContent = `
.ens h2, .ens h3 { margin: 0 0 6px; }
.ens .ens-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; justify-content: space-between; }
.ens .ens-chips { display: inline-flex; gap: 4px; flex-wrap: wrap; }
.ens .ens-chips button { font: inherit; font-size: 12px; padding: 3px 8px; border-radius: 999px; border: 1px solid var(--rule); background: transparent; color: var(--ink); cursor: pointer; }
.ens .ens-chips button[aria-pressed="true"] { background: var(--panel-2); border-color: var(--ink-2); font-weight: 700; }
.ens .ens-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 8px; }
.ens .ens-svg { width: 100%; height: auto; display: block; font: 10px var(--font-mono, monospace); fill: var(--ink-2); overflow: visible; }
.ens .ens-cap { font-family: var(--font-mono, monospace); font-size: 11px; letter-spacing: .05em; text-transform: uppercase; color: var(--ink-2); margin: 0 0 4px; }
.ens ul.ens-lines { margin: 6px 0 0; padding: 0; list-style: none; font-size: 13px; }
.ens ul.ens-lines li { margin: 3px 0; }
.ens .ens-key { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; vertical-align: 0; }
.ens table.ens-tab { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.ens table.ens-tab th, .ens table.ens-tab td { text-align: left; padding: 3px 4px; border-bottom: 1px solid var(--rule); vertical-align: top; }
.ens table.ens-tab td.n { font-family: var(--font-mono, monospace); white-space: nowrap; }
.ens .ens-note { font-size: 12px; color: var(--ink-2); margin: 8px 0 0; }
.ens .ens-warn { color: var(--bad-ink, var(--accent)); }
.ens details summary { cursor: pointer; font-size: 12px; color: var(--ink-2); }`;
  document.head.appendChild(style);

  let pending = false, token = 0;
  function schedule() {
    if (pending) return;
    pending = true;
    // a short timer, not requestAnimationFrame: rAF is paused in background tabs and some embedded viewers
    setTimeout(() => { pending = false; update(); }, 40);
  }
  async function update() {
    const my = ++token;
    const ctx = ADAPTER.context();
    if (!ctx) { root.hidden = true; return; }
    root.hidden = false;
    const v = VARIANTS.find(x => x.id === variant), gr = GRAPHS.find(x => x.id === graphSel);
    const [ens, index, graph, sci] = await Promise.all([load(`ensembles/${ctx.abbr}${v.file}${gr.file}.json`), load('ensembles/index.json'),
      gr.id === 'census' ? load(`graphs/${ctx.abbr}.json`) : Promise.resolve(null), load(`sci/${ctx.abbr}.json`)]);
    if (my !== token) return;
    if (sci && sci.counties.length === ctx.data.counties.length) sciFiles[ctx.abbr] = sci;
    if (gr.id === 'census' && graph && !censusEdges[ctx.abbr]) censusEdges[ctx.abbr] = Int32Array.from([...graph.edges, ...graph.water_links]);
    if (gr.id === 'census' && !graph) { render(ctx, null, index, v, gr); return; }
    render(ctx, ens, index, v, gr);
  }

  function head(v, gr) {
    return `<div class="ens-head"><${isDraw ? 'h2' : 'h3'}>Neutral baseline</${isDraw ? 'h2' : 'h3'}>` +
      `<span class="ens-chips" role="group" aria-label="Baseline variant">${VARIANTS.map(x =>
        `<button data-ens-v="${x.id}" aria-pressed="${x.id === v.id}" title="${esc(x.note)}">${esc(x.name)}</button>`).join('')}</span>` +
      `<span class="ens-chips" role="group" aria-label="Neighbour graph">${GRAPHS.map(x =>
        `<button data-ens-g="${x.id}" aria-pressed="${x.id === gr.id}" title="${esc(x.note)}">${esc(x.name)}</button>`).join('')}</span></div>`;
  }
  function wire() {
    root.querySelectorAll('[data-ens-v]').forEach(b => b.onclick = () => {
      variant = b.dataset.ensV;
      try { localStorage.setItem('ens-variant', variant); } catch (e) { /* storage blocked */ }
      update();
    });
    root.querySelectorAll('[data-ens-g]').forEach(b => b.onclick = () => {
      graphSel = b.dataset.ensG;
      try { localStorage.setItem('ens-graph', graphSel); } catch (e) { /* storage blocked */ }
      update();
    });
  }

  function render(ctx, ens, index, v, gr) {
    gr = gr || GRAPHS[0];
    let html = head(v, gr);
    if (!ens) {
      html += `<p class="ens-note">No neutral baseline for ${esc(ctx.abbr)} yet.</p>`;
    } else if (ens.districts === 1) {
      html += `<p class="ens-note">One at-large district: every plan is the same, so there is nothing to compare.</p>`;
    } else if (ctx.k !== ens.districts) {
      html += `<p class="ens-note">The baseline is for ${ens.districts} districts; this map has ${ctx.k}. Set the seat count back to ${ens.districts} to compare.</p>`;
    } else if (!ens.elections[ctx.el]) {
      html += `<p class="ens-note">No baseline for this election.</p>`;
    } else if (ADAPTER.busy()) {
      html += `<p class="ens-note">Updates when the solver stops.</p>`;
    } else {
      const E = ens.elections[ctx.el], k = ctx.k, total = ens.samples;
      const placed = [], missing = [];
      for (const p of ctx.plans) {
        if (complete(p.labels, k)) placed.push({ ...p, st: stats(ctx.data, p.labels, k, ctx.el) });
        else if (p.missing) missing.push(p.missing);
      }
      const hist = E.seats_d_histogram;
      const lines = placed.map(p => {
        let fewer = 0, same = 0, more = 0;
        for (const [s, c] of Object.entries(hist)) { if (+s < p.st.seats) fewer += c; else if (+s === p.st.seats) same += c; else more += c; }
        return `<li><span class="ens-key" style="background:${MARK[p.role]}"></span><b>${esc(p.name)}</b>: ${p.st.seats} D seats. ` +
          `Neutral plans give fewer in ${pct(100 * fewer / total)}, the same in ${pct(100 * same / total)}, more in ${pct(100 * more / total)}.</li>`;
      });
      // every election in the baseline, one row each
      const allRows = Object.entries(ens.elections).map(([key, e]) => {
        const h = e.seats_d_histogram, t = Object.values(h).reduce((a, b) => a + b, 0), sorted = [];
        Object.keys(h).map(Number).sort((a, b) => a - b).forEach(s => { for (let i = 0; i < h[s]; i++) sorted.push(s); });
        const q = p => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
        const cells = placed.map(p => {
          const s = stats(ctx.data, p.labels, k, key).seats;
          let more = 0, fewer = 0; for (const [v, c] of Object.entries(h)) { if (+v > s) more += c; else if (+v < s) fewer += c; }
          const tag = fewer / t >= 0.95 ? ` <span class="ens-warn">above ${pct(100 * fewer / t)}</span>` : more / t >= 0.95 ? ` <span class="ens-warn">below ${pct(100 * more / t)}</span>` : '';
          return `<td class="n">${s}${tag}</td>`;
        }).join('');
        return `<tr><td>${esc(e.label)}${key === ctx.el ? ' ◂' : ''}</td><td class="n">${(100 * e.statewide_d_share).toFixed(1)}%</td><td class="n">${q(0.5)} (${q(0.05)}–${q(0.95)})</td>${cells}</tr>`;
      }).join('');
      const allTable = `<details style="margin-top:8px"><summary>Every election</summary><table class="ens-tab" style="margin-top:4px"><thead><tr><th>Votes</th><th>D share</th><th>Neutral seats, median (5–95%)</th>` +
        placed.map(p => `<th>${esc(p.name)}</th>`).join('') + `</tr></thead><tbody>${allRows}</tbody></table>` +
        `<p class="ens-note">Democratic seats. A map is flagged when it gives more (or fewer) seats than 95% of neutral plans.</p></details>`;
      const ce = ens.cut_edges.quantiles, cs = ens.county_splits;
      const lt = ens.log_spanning_trees && ens.log_spanning_trees.quantiles;
      const shape = placed.map(p => `<li><span class="ens-key" style="background:${MARK[p.role]}"></span><b>${esc(p.name)}</b>: ` +
        `${fmt(p.st.cut)} cut edges, ${cutBand(ce, p.st.cut)}` +
        (lt ? `; spanning-tree score ${fmt(p.st.trees())}, ${treeBand(lt, p.st.trees())}` : '') +
        (cs ? `; ${p.st.splits} counties split, ${splitsPhrase(cs.histogram, p.st.splits)}` : '') + '.</li>').join('');
      const pass = index && index[ctx.abbr] && index[ctx.abbr][v.id + (gr.id === 'census' ? '-census' : '')];
      const warn = pass && pass.passed === false
        ? `<p class="ens-note ens-warn">The two sampling runs for this state did not agree closely (seat histograms ${Math.round(100 * pass.seats_tv)}% apart), so treat this baseline as rough.</p>` : '';
      html += `<p class="ens-note" style="margin-top:2px">Where the maps sit among ${fmt(total)} neutral plans: contiguous, within 1% of equal population, drawn without party data. ${esc(E.label)} votes, two-party.</p>` +
        `<div class="ens-grid">` +
        `<div><p class="ens-cap">Democratic seats across neutral plans</p>${seatChart(E, k, placed)}<ul class="ens-lines">${lines.join('')}</ul>${placed.length ? allTable : ''}</div>` +
        `<div><p class="ens-cap">District vote shares, least to most Democratic</p>${boxPlot(E, k, placed)}` +
        `<p class="ens-note">Box = middle half of neutral plans, whiskers = 5–95%.` +
        placed.filter(p => p.role !== 'alt').map(p => p.role === 'official'
          ? ` <span class="ens-key" style="border:1.6px solid ${MARK.official};width:7px;height:7px;background:none"></span>${esc(p.name)}`
          : ` <span class="ens-key" style="background:${MARK.main}"></span>${esc(p.name)}`).join('') + `</p></div>` +
        `</div>` +
        (placed.length ? `<p class="ens-cap" style="margin-top:12px">Compactness and counties</p>` +
          `<p class="ens-note" style="margin-top:0">Neutral plans, median (5–95%): ${fmt(ce[2])} (${fmt(ce[0])}–${fmt(ce[4])}) cut edges` +
          (lt ? `, spanning-tree score ${fmt(lt[2])} (${fmt(lt[0])}–${fmt(lt[4])})` : '') +
          (cs ? `, ${cs.quantiles[2]} (${cs.quantiles[0]}–${cs.quantiles[4]}) counties split` : '') +
          `. Fewer cut edges = more compact.` +
          (lt ? ` The spanning-tree score is the log of the number of ways to connect each district’s block groups with a tree, summed over districts: compact districts have far more, so higher = more compact.` : '') + `</p>` +
          `<ul class="ens-lines">${shape}</ul>` : '') +
        sciSection(ctx, ens, placed, k) +
        (missing.length ? `<p class="ens-note">${missing.map(esc).join(' ')}</p>` : '') +
        (ctx.swing ? `<p class="ens-note">The baseline uses actual votes; the swing slider is not applied here.</p>` : '') +
        warn +
        `<details style="margin-top:6px"><summary>About this baseline</summary><p class="ens-note">${esc(ens.method)}. ${esc(v.note)}. Neighbours: ${esc(gr.note)}. ` +
        `Two chains from different starting maps, ${fmt(ens.params.moves_per_chain)} moves each, every ${ens.params.keep_every}th plan kept. ` +
        `Cut edges = neighbouring block-group pairs split by a district line (on this site's block-group graph); fewer means more compact. ` +
        `A neutral process does not promise proportional seats: where voters live decides that.</p></details>`;
    }
    const open = [...root.querySelectorAll('details')].map(d => d.open);  // keep sections the reader opened
    root.innerHTML = html;
    root.querySelectorAll('details').forEach((d, i) => { if (open[i]) d.open = true; });
    wire();
    extras(ctx, ens);
  }

  function sciSection(ctx, ens, placed, k) {
    const sci = sciFiles[ctx.abbr], S = ens.sci_ties_kept;
    if (!sci || !S || !S.adjusted || !placed.length) return '';
    const qa = S.adjusted.quantiles, qr = S.raw.quantiles, f1 = x => x.toFixed(1) + '%';
    const rows = placed.map(p => {
      const v = sciKept(ctx.data, sci, p.labels, k);
      return `<li><span class="ens-key" style="background:${MARK[p.role]}"></span><b>${esc(p.name)}</b>: ` +
        `${f1(v.adjusted)} of cross-county ties kept, ${tiesBand(qa, v.adjusted)} (raw ${f1(v.raw)}, ${tiesBand(qr, v.raw)}).</li>`;
    }).join('');
    const miss = sci.counties_missing_from_sci && sci.counties_missing_from_sci.length
      ? ` Not in the index (no ties counted): ${esc(sci.counties_missing_from_sci.join(', '))}.` : '';
    return `<p class="ens-cap" style="margin-top:12px">Social ties kept together</p>` +
      `<p class="ens-note" style="margin-top:0">Share of friendships between people in different counties (Meta’s Social Connectedness Index × population) ` +
      `where both end up in the same district. Distance-adjusted counts only ties stronger than distance alone predicts; raw is shown in brackets. ` +
      `Neutral plans, median (5–95%): ${f1(qa[2])} (${f1(qa[0])}–${f1(qa[4])}), raw ${f1(qr[2])} (${f1(qr[0])}–${f1(qr[4])}).</p>` +
      `<ul class="ens-lines">${rows}</ul>` +
      `<p class="ens-note">A lens, not a neutral rule: friendship follows migration, party and race, so a map that keeps more ties is not fairer by that fact.${miss}</p>`;
  }

  /* ---------- small additions next to the page's own numbers ---------- */
  const EG_NOTE = 'Caveat: the wasted-vote gap is zero only when the winner’s seat share sits twice as far from 50% as its vote share, ' +
    'so a proportional result (60% of votes, 60% of seats) shows a 10-point gap (Duchin 2018).';
  function extras(ctx, ens) {
    if (isDraw) {  // the scorecard note is rewritten on every refresh, so add the caveat back each time
      const note = document.getElementById('scorenote'), score = document.getElementById('score');
      if (note && score && !score.hidden && !note.querySelector('.ens-eg')) note.insertAdjacentHTML('beforeend', ` <span class="ens-eg">${EG_NOTE}</span>`);
      return;
    }
    // Fair Lines: least-change slider readout
    const panel = document.getElementById('lcPanel'), after = document.getElementById('lcSentence');
    if (!panel || !after) return;
    let p = document.getElementById('ens-lc');
    if (!p) { p = document.createElement('p'); p.id = 'ens-lc'; p.className = 'hint'; p.style.margin = '6px 0 0'; after.after(p); }
    const E = ens && ens.districts === ctx.k && ens.elections[ctx.el];
    if (!E || panel.hidden || !complete(ctx.plans[0].labels, ctx.k)) { p.textContent = ''; return; }
    const s = stats(ctx.data, ctx.plans[0].labels, ctx.k, ctx.el).seats, h = E.seats_d_histogram;
    let f = 0, same = 0, m = 0, t = 0;
    for (const [v, c] of Object.entries(h)) { t += c; if (+v < s) f += c; else if (+v === s) same += c; else m += c; }
    p.innerHTML = `<b>Neutral baseline</b>: the map at this point gives Democrats ${s} seats; neutral plans give fewer in ${pct(100 * f / t)}, ` +
      `the same in ${pct(100 * same / t)}, more in ${pct(100 * m / t)} (${esc(E.label)} votes).`;
  }
  function onceFairLines() {
    // efficiency-gap caveat under the score panel's existing explanation
    const eg = document.getElementById('kvEG'), hint = eg && eg.closest('.panel') && eg.closest('.panel').querySelector('p.hint');
    if (hint && !hint.querySelector('.ens-eg')) hint.insertAdjacentHTML('beforeend', ` <span class="ens-eg">${EG_NOTE}</span>`);
    // methods table: a ReCom ensemble row per state, beside the existing rows
    const table = document.querySelector('table.bench');
    if (!table) return;
    const groups = [...table.querySelectorAll('tbody tr th[colspan]')];
    const jobs = groups.map(th => {
      const abbr = (th.textContent.match(/^([A-Z]{2})\b/) || [])[1];
      if (!abbr) return null;
      return load(`ensembles/${abbr}.json`).then(ens => {
        if (!ens || !ens.max_dev_pct) return;
        const q = ens.max_dev_pct.quantiles, tr = document.createElement('tr');
        tr.className = 'ens-row';
        tr.innerHTML = `<td>ReCom ensemble (neutral sample), median and 5–95% ‡</td><td>${q[2].toFixed(3)} (${q[0].toFixed(2)}–${q[4].toFixed(2)})</td><td>–</td><td>${Math.round(ens.seconds)}</td>`;
        let row = th.closest('tr').nextElementSibling;
        while (row && !row.querySelector('th[colspan]') && row.nextElementSibling && !row.nextElementSibling.querySelector('th[colspan]')) row = row.nextElementSibling;
        (row || th.closest('tr')).after(tr);
      });
    }).filter(Boolean);
    Promise.all(jobs).then(() => {
      if (document.getElementById('ens-bench-note')) return;
      const note = document.createElement('p');
      note.id = 'ens-bench-note'; note.className = 'hint';
      note.innerHTML = '‡ <b>ReCom ensemble</b>: this site’s neutral sample (plain ReCom, 10,000 plans from two chains, see the Neutral baseline panel). ' +
        'A sampler aims for typical plans within a 1% tolerance, not the best plan, so its population gap reflects the tolerance it was given. ' +
        'Its compactness is reported as cut edges in the Neutral baseline panel, not as Polsby-Popper. The GerryChain row above is likewise a sampler: the most compact of 5,000 sampled maps.';
      table.after(note);
    });
  }

  /* ---------- hook into the page ---------- */
  if (ADAPTER.hook && typeof window[ADAPTER.hook] === 'function') {
    const original = window[ADAPTER.hook];
    window[ADAPTER.hook] = function () { const r = original.apply(this, arguments); schedule(); return r; };
  }
  if (ADAPTER.observe) {
    const el = document.getElementById(ADAPTER.observe);
    if (el) new MutationObserver(schedule).observe(el, { childList: true });
  }
  if (isFair) onceFairLines();
  if (isDraw && !document.querySelector('script[data-rd-sci]')) {  // added: the friendship-ties slider (sci-slider.js)
    const me = document.currentScript || [...document.scripts].find(s => /ensemble-panel\.js/.test(s.src));
    const s = document.createElement('script');
    s.src = (me && me.src ? me.src.replace(/ensemble-panel\.js.*$/, '') : '../addons/') + 'sci-slider.js';
    s.dataset.rdSci = '1';
    document.body.appendChild(s);
  }
  window.__RD_ENSEMBLE = { logTrees, sciKept: (data, labels, k) => sciFiles[data.abbr] && sciKept(data, sciFiles[data.abbr], labels, k) };  // for checking against redistricting_addons.ensemble.log_spanning_trees
  schedule();
})();
