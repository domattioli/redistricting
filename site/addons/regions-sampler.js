/* Neutral groupings for Re-draw the states (add-on).

   Samples many ways to group the page's 435 borderless districts into the same new states, each state keeping
   its seat count, with ReCom on the district graph: merge two neighbouring states, draw a random spanning tree
   over their districts and cut it where one side has exactly the first state's seat count. This is the
   "nesting" question of Caldera, DeFord, Duchin et al. (2020): how much does the grouping alone change results?
   Fixed districts mean House seats never change; Senate seats and Electoral College votes do.

   Votes per district come from data/ensembles/regions_votes.json (python -m redistricting_addons.regions_votes).
   It only adds: it reads the page's globals (G, SIM, dnew, ST) and uses the page's own adopt(), render() and
   table() to show a sampled grouping. Loaded by one <script> line; delete it to remove. */
(function () {
  'use strict';
  if (typeof G === 'undefined' || typeof SIM === 'undefined' || typeof adopt !== 'function') return;
  const BASE = typeof __RD_DATA_BASE === 'string' ? __RD_DATA_BASE : '../data/';
  const ELS = [['pres2016', '2016'], ['pres2020', '2020'], ['pres2024', '2024']];
  const BURN = 2000, EVERY = 20, SAMPLES = 1000;
  let votes = null, el = 'pres2020', result = null, running = false, sampled = [];

  /* ---------- ReCom on the district graph ---------- */
  function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const EDGES = []; G.nb.forEach((list, a) => list.forEach(b => { if (a < b) EDGES.push([a, b]); }));

  function step(asg, quota, rand) {
    const cut = EDGES.filter(([a, b]) => asg[a] !== asg[b]);
    for (let attempt = 0; attempt < 20; attempt++) {
      const [a0, b0] = cut[Math.floor(rand() * cut.length)], A = asg[a0], B = asg[b0];
      const nodes = [], local = new Map();
      for (let d = 0; d < asg.length; d++) if (asg[d] === A || asg[d] === B) { local.set(d, nodes.length); nodes.push(d); }
      const es = [];
      for (const d of nodes) for (const e of G.nb[d]) if (d < e && local.has(e)) es.push([local.get(d), local.get(e), rand()]);
      for (let t = 0; t < 10; t++) {
        // random spanning tree: Kruskal on random weights
        for (const e of es) e[2] = rand();
        es.sort((x, y) => x[2] - y[2]);
        const n = nodes.length, par = Int32Array.from({ length: n }, (_, i) => i), adj = Array.from({ length: n }, () => []);
        const find = x => { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; };
        let used = 0;
        for (const [u, v] of es) { const ru = find(u), rv = find(v); if (ru !== rv) { par[ru] = rv; adj[u].push(v); adj[v].push(u); used++; } }
        if (used !== n - 1) return false; // merged region not connected
        // subtree sizes from a BFS order
        const order = [0], parent = new Int32Array(n).fill(-1); parent[0] = 0;
        for (let i = 0; i < order.length; i++) for (const v of adj[order[i]]) if (parent[v] < 0) { parent[v] = order[i]; order.push(v); }
        parent[0] = -1;
        const size = new Int32Array(n).fill(1);
        for (let i = order.length - 1; i > 0; i--) size[parent[order[i]]] += size[order[i]];
        const qa = quota.get(A), qb = quota.get(B), cand = [];
        for (let i = 1; i < n; i++) { const v = order[i]; if (size[v] === qa) cand.push([v, A, B]); else if (size[v] === qb) cand.push([v, B, A]); }
        if (!cand.length) continue;
        const [v, inside, outside] = cand[Math.floor(rand() * cand.length)];
        const inSub = new Uint8Array(n); inSub[v] = 1;
        for (let i = 1; i < order.length; i++) { const u = order[i]; if (u !== v && inSub[parent[u]]) inSub[u] = 1; }
        nodes.forEach((d, i) => { asg[d] = inSub[i] ? inside : outside; });
        return true;
      }
    }
    return false;
  }

  function senate(asg) {
    const E = votes.elections[el], D = new Map(), Rv = new Map(), seats = new Map();
    for (let d = 0; d < asg.length; d++) { const s = asg[d]; D.set(s, (D.get(s) || 0) + E.d[d]); Rv.set(s, (Rv.get(s) || 0) + E.r[d]); seats.set(s, (seats.get(s) || 0) + 1); }
    let sen = 0, ec = 0, ecTotal = 0;
    for (const [s, d] of D) { const won = d > Rv.get(s); ecTotal += seats.get(s) + 2; if (won) { sen += 2; ec += seats.get(s) + 2; } }
    return { sen, ec, ecTotal };
  }

  /* ---------- panel ---------- */
  const card = document.createElement('aside');
  card.className = 'card'; card.style.marginTop = '12px';
  (document.querySelector('aside.card') || document.body).after(card);
  const style = document.createElement('style');
  style.textContent = `.rs h2 { margin: 0 0 6px; } .rs .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 6px 0; }
.rs svg { width: 100%; height: auto; display: block; font: 10px var(--font-mono, monospace); fill: var(--ink-2); }
.rs ul { margin: 6px 0 0; padding-left: 18px; font-size: 13px; } .rs li { margin: 3px 0; }`;
  document.head.appendChild(style);
  card.classList.add('rs');

  function quotas() {
    const q = new Map(); for (const x of SIM.seeds) q.set(x.s, x.seats);
    return q;
  }
  function meetsQuota(asg, q) {
    const c = new Map(); for (const s of asg) c.set(s, (c.get(s) || 0) + 1);
    if (c.size !== q.size) return false;
    for (const [s, n] of c) if (q.get(s) !== n) return false;
    return true;
  }

  function histogram(values, mark, label, total) {
    const counts = new Map(); values.forEach(v => counts.set(v, (counts.get(v) || 0) + 1));
    const keys = [...counts.keys(), mark].filter(v => v != null);
    const lo = Math.min(...keys), hi = Math.max(...keys), stepV = label === 'Senate' ? 2 : 1;
    const bins = []; for (let v = lo; v <= hi; v += stepV) bins.push(v);
    const W = 320, H = 120, left = 6, top = 22, bot = 18, bw = (W - 2 * left) / bins.length, maxC = Math.max(...counts.values());
    let g = '';
    bins.forEach((v, i) => {
      const c = counts.get(v) || 0, h = (H - top - bot) * c / maxC, x = left + i * bw;
      g += `<rect x="${(x + bw * 0.1).toFixed(1)}" y="${(H - bot - h).toFixed(1)}" width="${(bw * 0.8).toFixed(1)}" height="${h.toFixed(1)}" fill="var(--ink-2)" opacity=".5"><title>${v}: ${(100 * c / values.length).toFixed(1)}% of groupings</title></rect>`;
      if (bins.length <= 24 || i % Math.ceil(bins.length / 12) === 0) g += `<text x="${(x + bw / 2).toFixed(1)}" y="${H - 5}" text-anchor="middle">${v}</text>`;
    });
    if (mark != null) { const i = bins.indexOf(mark), cx = left + (i + 0.5) * bw; g += `<path d="M${(cx - 5).toFixed(1)} 6h10l-5 8z" fill="var(--accent)"><title>This grouping: ${mark}</title></path>`; }
    if (total) { const half = total / 2, i = (half - lo) / stepV, x = left + (i + 0.5) * bw; if (x > left && x < W - left) g += `<line x1="${x}" x2="${x}" y1="${top - 4}" y2="${H - bot}" stroke="var(--ink)" stroke-dasharray="3 3" opacity=".6"/><text x="${x + 3}" y="${top - 6}">half</text>`; }
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${label} across neutral groupings">${g}</svg>`;
  }
  function place(values, v) {
    let f = 0, s = 0, m = 0; for (const x of values) { if (x < v) f++; else if (x === v) s++; else m++; }
    const p = n => (100 * n / values.length).toFixed(1) + '%';
    return `fewer in ${p(f)}, the same in ${p(s)}, more in ${p(m)}`;
  }

  function draw() {
    let html = `<h2>Neutral groupings</h2><p class="note">How much does the grouping alone matter? Samples ways to group the same 435 districts into the same states, each keeping its seat count (ReCom on the district graph; Caldera, DeFord, Duchin et al. 2020). House seats cannot change; Senate seats and Electoral College votes can.</p>`;
    html += `<div class="row"><span class="note">Votes</span><span class="seg" role="group" aria-label="Election">${ELS.map(([k, t]) => `<button data-rs-el="${k}" aria-pressed="${k === el}">${t}</button>`).join('')}</span>` +
      `<button id="rs-run" ${running ? 'disabled' : ''}>${result ? 'Sample again' : 'Sample ' + SAMPLES.toLocaleString('en-US') + ' groupings'}</button>` +
      (sampled.length ? `<button id="rs-show">Show one on the map</button>` : '') + `</div><div id="rs-out"></div>`;
    card.innerHTML = html;
    card.querySelectorAll('[data-rs-el]').forEach(b => b.onclick = () => { el = b.dataset.rsEl; if (result) summarize(); draw(); });
    card.querySelector('#rs-run').onclick = sample;
    const show = card.querySelector('#rs-show');
    if (show) show.onclick = () => {
      const asg = sampled[Math.floor(Math.random() * sampled.length)];
      if (typeof $ === 'function' && $('seeds')) $('seeds').checked = false;
      adopt(asg); render(); table();
      if (typeof say === 'function') say('Showing one neutral grouping: names are only labels, since neutral groupings ignore today’s lines. Press Reset to go back.');
    };
    out();
  }
  function out() {
    const o = card.querySelector('#rs-out'); if (!o) return;
    if (!votes) { o.innerHTML = '<p class="note">No district votes found (data/ensembles/regions_votes.json).</p>'; return; }
    const q = quotas();
    if (!meetsQuota(dnew, q)) { o.innerHTML = '<p class="note">Every state must meet its seat count before sampling. Press Reset, or run the simulator until it does.</p>'; return; }
    if (running) { o.innerHTML = `<p class="note">Sampling… ${running}</p>`; return; }
    if (!result) { o.innerHTML = '<p class="note">Press Sample to compare this grouping with neutral ones.</p>'; return; }
    const here = senate(dnew), r = result;
    o.innerHTML = `<p class="note" style="margin:6px 0 2px">Democratic senators (of ${2 * q.size})</p>${histogram(r.sen, here.sen, 'Senate', 2 * q.size)}` +
      `<p class="note" style="margin:8px 0 2px">Democratic electoral votes (of ${here.ecTotal}; no DC)</p>${histogram(r.ec, here.ec, 'Electoral votes', here.ecTotal)}` +
      `<ul><li><b>This grouping</b>: ${here.sen} D senators; neutral groupings give ${place(r.sen, here.sen)}.</li>` +
      `<li><b>This grouping</b>: ${here.ec} D electoral votes; neutral groupings give ${place(r.ec, here.ec)}.</li></ul>` +
      `<p class="note">${r.sen.length.toLocaleString('en-US')} groupings, one every ${EVERY} moves after ${BURN.toLocaleString('en-US')} warm-up moves. District votes are block-group estimates placed on this page's districts, so states' totals are approximate. ${ELS.find(x => x[0] === el)[1]} presidential votes, two-party.</p>`;
  }
  function summarize() {
    result = { sen: sampled.map(a => senate(a).sen), ec: sampled.map(a => senate(a).ec) };
  }
  function sample() {
    const q = quotas();
    if (!votes || !meetsQuota(dnew, q)) { draw(); return; }
    const asg = Int16Array.from(dnew), rand = rng(20261112 + sampled.length);
    sampled = []; result = null; let moves = 0, tries = 0;
    const total = BURN + SAMPLES * EVERY;
    running = '0%'; draw();
    const chunk = () => {
      const end = Math.min(total, moves + 500);
      while (moves < end) {
        if (++tries > 20 * total) { running = false; card.querySelector('#rs-out').innerHTML = '<p class="note">The sampler got stuck on this grouping.</p>'; return; }
        if (step(asg, q, rand)) { moves++; if (moves > BURN && (moves - BURN) % EVERY === 0) sampled.push(Int16Array.from(asg)); }
      }
      running = Math.round(100 * moves / total) + '%'; out();
      if (moves < total) setTimeout(chunk, 0); else { running = false; summarize(); draw(); }
    };
    setTimeout(chunk, 0);
  }

  fetch(BASE + 'ensembles/regions_votes.json').then(r => (r.ok ? r.json() : null)).catch(() => null).then(v => { votes = v; draw(); });
  draw();
})();
