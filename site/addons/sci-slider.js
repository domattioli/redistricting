/* sci-slider.js: a "friendship ties" constraint layer for the Draw toy (redistricting fork add-on).

   In short (for people): adds a slider beside the state-line, river and commuting-zone sliders. Turned up, the
   solver prefers to draw district lines between neighbourhoods whose people are rarely Facebook friends (Meta's
   Social Connectedness Index between ZIP-code areas) and to keep strongly connected neighbourhoods together.
   At 0 nothing changes.

   For AI agents (detailed): loaded by ensemble-panel.js on the Draw page only. Reads
   ../data/sci/zcta/<ST>.json (redistricting_addons/sci_zcta.py): edge_raw / edge_adj give, per site graph edge
   (data.edges order), the rank 0-255 of log SCI between the endpoints' ZCTAs. Wraps DrawEngine.penalize (the
   page's own function is called first): each CSR edge's length is further multiplied by
   1 + lambda_sci * (1 - rank / 64) when rank < 64 (the weakest quarter of the state's links; others unchanged:
   penalising every link made borders jagged in testing), and the same amount is added to prep.edgePen, so weakly tied links act like
   a line the districts prefer to stop at (the same mechanism and scale as the page's lambda sliders,
   LAM_STEPS). Sub-problems built inside the engine (subPrep, used by "Finish my map") keep only the page's own
   penalties. Only adds: no page code is changed; the slider row is inserted into the "Constraint layers" card. */
(function () {
  'use strict';
  if (typeof DX === 'undefined' || !DX || typeof DX.penalize !== 'function' || !document.getElementById('layers')) return;
  if (DX.__sciWrapped) return;
  DX.__sciWrapped = true;
  const STEPS = [0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20];  // gentler than the page's LAM_STEPS: ties are scattered, not lines
  const WEAK = 64;  // only the weakest quarter of links (rank below 64 of 255) is penalised, so they act like a line
  const BASE = (typeof __RD_DATA_BASE !== 'undefined' ? __RD_DATA_BASE : '../data/') + 'sci/zcta/';
  const st = { lam: 0, kind: 'adj', data: null, abbr: null };
  window.__RD_SCI = st;
  const orig = DX.penalize;
  let csrCache = { prep: null, map: null };

  function edgeRank(prep) {
    // per CSR entry of prep (the unpenalized base), the 0-255 rank of its site edge, or -1
    const base = prep.base || prep;
    if (csrCache.prep === base && csrCache.kind === st.kind) return csrCache.map;
    const D = (typeof S !== 'undefined' && S.data) || null;
    const rank = st.data && st.data['edge_' + st.kind];
    const map = new Int16Array(base.indices.length).fill(-1);
    if (D && rank && D.edges && rank.length * 2 === D.edges.length) {
      const ip = base.indptr, ix = base.indices;
      for (let t = 0; t < rank.length; t++) {
        const a = D.edges[2 * t], b = D.edges[2 * t + 1];
        for (let e = ip[a]; e < ip[a + 1]; e++) if (ix[e] === b) { map[e] = rank[t]; break; }
        for (let e = ip[b]; e < ip[b + 1]; e++) if (ix[e] === a) { map[e] = rank[t]; break; }
      }
    }
    csrCache = { prep: base, kind: st.kind, map };
    return map;
  }

  DX.penalize = function (prep, lam) {
    const p0 = orig.call(this, prep, lam);
    if (!(st.lam > 0) || !st.data) return p0;
    const base = prep.base || prep, map = edgeRank(base), m = base.indices.length;
    const elen = new Float64Array(m), pen = new Float64Array(m);
    const e0 = p0.elen || base.elen, pn0 = p0.edgePen;
    for (let e = 0; e < m; e++) {
      const f = map[e] >= 0 && map[e] < WEAK ? st.lam * (1 - map[e] / WEAK) : 0;
      elen[e] = e0[e] * (1 + f);
      pen[e] = (pn0 ? pn0[e] : 0) + f;
    }
    const p = {};
    for (const key in p0) p[key] = p0[key];
    p.elen = elen; p.edgePen = pen; p.base = base;
    p.lam = Object.assign({}, p0.lam || { state: 0, geo: 0, cz: 0 }, { sci: st.lam });
    p.cache = { rcb: base.cache.rcb, ws: null, statsWs: null };
    return p;
  };

  function apply() {
    if (typeof S === 'undefined' || !S.prep) return;
    S.pprep = DX.penalize(S.prep, S.lam);
    if (S.prop && typeof recompute === 'function') { recompute(); if (typeof refreshAll === 'function') refreshAll(); }
  }

  function addRow() {
    if (document.getElementById('ly-sci')) return;
    const row = document.createElement('div');
    row.className = 'layer'; row.id = 'ly-sci';
    row.innerHTML = '<label class="chk" title="Meta Social Connectedness Index between ZIP-code areas">' +
      '<span class="sw2" style="border-color:#c08bd6"></span>friendship ties ' +
      '<select id="sci-kind" style="font:inherit;font-size:12px;margin-left:4px"><option value="adj">beyond distance</option><option value="raw">raw</option></select></label>' +
      '<span class="num" id="cx-sci"></span>' +
      '<div class="lam">&lambda;<sub>ties</sub> <input type="range" id="lam-sci" min="0" max="8" step="1" value="0" aria-label="Friendship-ties penalty">' +
      '<output id="lam-sci-v" class="num">0</output></div>';
    const note = document.getElementById('layernote');
    note ? note.parentNode.insertBefore(row, note) : document.getElementById('layers').appendChild(row);
    const hint = document.createElement('small');
    hint.className = 'hint'; hint.style.display = 'block'; hint.id = 'sci-note';
    hint.textContent = 'Friendship ties: the quarter of links between ZIP-code areas whose people are least often Facebook friends get longer by up to 1 + λ, so district lines prefer to fall there; higher values cost compactness (North Carolina, 03bb = 0.25: about a quarter more border). "Beyond distance" ranks ties after removing what distance alone predicts. A lens: ties follow migration, party and race.';
    row.appendChild(hint);
    document.getElementById('lam-sci').oninput = e => { st.lam = STEPS[+e.target.value]; document.getElementById('lam-sci-v').textContent = st.lam; apply(); };
    document.getElementById('sci-kind').onchange = e => { st.kind = e.target.value; csrCache = { prep: null }; apply(); };
  }

  async function load() {
    const D = (typeof S !== 'undefined' && S.data) || null;
    if (!D || !D.abbr || /\+$/.test(D.abbr)) return;  // regions (multi-state) have no ZIP file
    if (st.abbr === D.abbr && st.data) return;
    st.abbr = D.abbr; st.data = null; csrCache = { prep: null };
    addRow();
    const cx = document.getElementById('cx-sci');
    try {
      const r = await fetch(BASE + D.abbr + '.json');
      if (!r.ok) throw new Error(r.status);
      st.data = await r.json();
      cx.textContent = st.data.zcta.length + ' ZIP areas';
      document.getElementById('lam-sci').disabled = false;
      if (st.lam > 0) apply();
    } catch (e) {
      cx.textContent = 'no ZIP data';
      document.getElementById('lam-sci').disabled = true;
    }
  }

  // the page loads its state asynchronously; poll briefly, then follow state changes via refreshAll
  let tries = 0;
  const timer = setInterval(() => { tries++; if (typeof S !== 'undefined' && S.data) { load(); } if (tries > 120) clearInterval(timer); }, 500);
  if (typeof refreshAll === 'function' && !window.__RD_SCI_RA) {
    window.__RD_SCI_RA = true;
    const ra = refreshAll;
    // eslint-disable-next-line no-global-assign
    refreshAll = function () { const out = ra.apply(this, arguments); load(); return out; };
  }
})();
