// Exact equilibrium of the maze game on tiny boards (3×3), as ground truth for the RL work.
//
//   node research/exact_solve.mjs [size=3] [walls=3] [iterations=400]
//
// The round is modelled as a two-player zero-sum game on WALLS HIT (= turns lost):
//   - the designer picks a maze (walls + E + S), possibly at random (a mixed strategy);
//   - the explorer, who is told E and S, picks probes adaptively.
// E and S are public, so the game splits into one subgame per (E, S) pair: the designer picks the
// pair whose subgame is worth the most, then mixes walls inside it. In each subgame, once the maze is
// fixed only the explorer acts, so its best response to a designer mixture is an exact Bayes-optimal
// search (dynamic programming over the set of mazes still consistent with what it saw). The designer
// learns its mixture with Hedge (multiplicative weights) against those best responses; the two
// bounds (best response to the designer's average, best maze against the explorers' average)
// certify the equilibrium value.

import { allEdges, edgeCells } from '../src/game/grid.js';
import { simulateSolve } from '../src/bot/brain.js';
import { seededRng } from '../src/game/random.js';

const n = Number(process.argv[2]) || 3;
const B = Number(process.argv[3]) || 3;
const N = n * n;
const keys = allEdges(n);
const ends = keys.map(edgeCells);
const M = keys.length;

// ------------------------------------------------------------------ enumerate valid mazes

function components(wallMask) {
  const parent = [...Array(N).keys()];
  const find = (x) => (parent[x] === x ? x : (parent[x] = find(parent[x])));
  ends.forEach(([a, b], i) => {
    if (!((wallMask >> i) & 1)) parent[find(a)] = find(b);
  });
  return [...Array(N).keys()].map(find);
}

const mazes = []; // { w: wall bitmask, e, s }
(function enumerate(start, left, mask) {
  if (left === 0) {
    const comp = components(mask);
    for (let e = 0; e < N; e++) for (let s = 0; s < N; s++) if (e !== s && comp[e] === comp[s]) mazes.push({ w: mask, e, s });
    return;
  }
  for (let i = start; i <= M - left; i++) enumerate(i + 1, left - 1, mask | (1 << i));
})(0, B, 0);
const byES = new Map();
mazes.forEach((m, i) => {
  const k = m.e * N + m.s;
  if (!byES.has(k)) byES.set(k, []);
  byES.get(k).push(i);
});

function visitedOf(e, open) {
  let vis = 1 << e;
  for (let changed = true; changed; ) {
    changed = false;
    ends.forEach(([a, b], i) => {
      if (!((open >> i) & 1)) return;
      const va = (vis >> a) & 1;
      const vb = (vis >> b) & 1;
      if (va !== vb) {
        vis |= (1 << a) | (1 << b);
        changed = true;
      }
    });
  }
  return vis;
}

// ------------------------------------------------------------------ explorer best response (exact DP)

/**
 * Bayes-optimal explorer inside one (E, S) group, for prior `weights` (indexed by maze).
 * Returns { hits: Float64Array aligned with `group`, value: expected hits under the prior }.
 *
 * State = the set of mazes still consistent with what was observed. Two histories with the same
 * set have the same future: sides open in every remaining maze can be walked for free, so the
 * explorer is effectively standing on everything reachable through them.
 */
function bestResponse(group, weights) {
  const memo = new Map();
  const { e, s } = mazes[group[0]];
  const hashOf = (cand) => {
    let h1 = 0x811c9dc5;
    let h2 = 0x9e3779b9;
    for (const m of cand) {
      h1 = Math.imul(h1 ^ m, 0x01000193);
      h2 = Math.imul(h2 ^ m, 0x85ebca6b) ^ (h2 >>> 13);
    }
    return `${cand.length}:${h1 >>> 0}:${h2 >>> 0}`;
  };
  function solve(cand) {
    const key = hashOf(cand);
    const hit = memo.get(key);
    if (hit) return hit;
    let allOpen = (1 << M) - 1;
    let allWall = (1 << M) - 1;
    for (const m of cand) {
      allOpen &= ~mazes[m].w;
      allWall &= mazes[m].w;
    }
    const vis = visitedOf(e, allOpen);
    let res = { v: 0, a: -1 };
    if (!((vis >> s) & 1)) {
      let W = 0;
      for (const m of cand) W += weights[m];
      res = { v: Infinity, a: -1 };
      for (let i = 0; i < M; i++) {
        const bit = 1 << i;
        if ((allOpen | allWall) & bit) continue;
        if (!(((vis >> ends[i][0]) & 1) || ((vis >> ends[i][1]) & 1))) continue;
        const openSet = [];
        const wallSet = [];
        let wo = 0;
        for (const m of cand) {
          if (mazes[m].w & bit) wallSet.push(m);
          else {
            openSet.push(m);
            wo += weights[m];
          }
        }
        const pO = W > 0 ? wo / W : openSet.length / cand.length;
        let v = 0;
        if (pO > 0) v += pO * solve(openSet).v;
        if (pO < 1) v += (1 - pO) * (1 + solve(wallSet).v);
        if (v < res.v - 1e-12) res = { v, a: i };
      }
    }
    memo.set(key, res);
    return res;
  }
  const hits = new Float64Array(group.length);
  let value = 0;
  let W = 0;
  group.forEach((mi, k) => {
    let cand = group;
    let h = 0;
    for (;;) {
      const { a } = solve(cand);
      if (a < 0) break;
      const bit = 1 << a;
      const isWall = (mazes[mi].w & bit) !== 0;
      if (isWall) h++;
      cand = cand.filter((m) => ((mazes[m].w & bit) !== 0) === isWall);
    }
    hits[k] = h;
    value += weights[mi] * h;
    W += weights[mi];
  });
  return { hits, value: W > 0 ? value / W : 0 };
}

// ------------------------------------------------------------------ symmetry (8 rotations/reflections)

const symCell = (t, c) => {
  let r = Math.floor(c / n);
  let q = c % n;
  if (t & 1) [r, q] = [q, r];
  if (t & 2) r = n - 1 - r;
  if (t & 4) q = n - 1 - q;
  return r * n + q;
};
const canonicalPair = (e, s) => Math.min(...[0, 1, 2, 3, 4, 5, 6, 7].map((t) => symCell(t, e) * N + symCell(t, s)));

// ------------------------------------------------------------------ the current heuristic bot

const toGame = (m) => ({ walls: keys.filter((_, i) => (m.w >> i) & 1), entrance: m.e, exit: m.s });
const rng = seededRng(7);
const heuristicHits = (mi) => {
  let h = 0;
  for (let k = 0; k < 8; k++) h += simulateSolve(toGame(mazes[mi]), n, 'hard', rng).hits;
  return h / 8;
};

// ------------------------------------------------------------------ equilibrium of each (E, S) subgame

const T = Number(process.argv[4]) || 400;
const ETA = 0.25;
const t0 = Date.now();
const classes = new Map(); // canonical pair → one representative group
for (const [es, group] of byES) {
  const c = canonicalPair(Math.floor(es / N), es % N);
  if (!classes.has(c)) classes.set(c, group);
}

const solved = [];
for (const [, group] of classes) {
  const k = group.length;
  const weights = new Float64Array(mazes.length);
  const cum = new Float64Array(k); // designer's cumulative payoff per maze
  const avgX = new Float64Array(k);
  const avgHits = new Float64Array(k); // explorers' average hits per maze (uniform mix of best responses)
  for (let t = 1; t <= T; t++) {
    const top = Math.max(...cum);
    let z = 0;
    const x = Array.from(cum, (c) => {
      const v = Math.exp(ETA * (c - top));
      z += v;
      return v;
    });
    group.forEach((mi, j) => (weights[mi] = x[j] / z));
    const br = bestResponse(group, weights);
    for (let j = 0; j < k; j++) {
      cum[j] += br.hits[j];
      avgX[j] += x[j] / z / T;
      avgHits[j] += br.hits[j] / T;
    }
  }
  group.forEach((mi, j) => (weights[mi] = avgX[j]));
  const lower = bestResponse(group, weights).value; // designer's average guarantees at least this
  const upper = Math.max(...avgHits); // explorers' average concedes at most this
  const entropy = -avgX.reduce((acc, p) => acc + (p > 0 ? p * Math.log(p) : 0), 0);
  solved.push({ group, avgX: Array.from(avgX), avgHits, lower, upper, support: Math.exp(entropy) });
}
solved.sort((a, b) => b.lower - a.lower);
const best = solved[0];
const { e: bestE, s: bestS } = mazes[best.group[0]];
const label = (c) => String.fromCharCode(65 + (c % n)) + (Math.floor(c / n) + 1);

// The equilibrium designer plays the best (E, S) subgame with its learned wall mixture.
const vsDesigner = (hitsOf) => best.group.reduce((acc, mi, j) => acc + best.avgX[j] * hitsOf(mi, j), 0);
const worstIn = (hitsOf) => Math.max(...best.group.map((mi, j) => hitsOf(mi, j)));
const uniform = new Float64Array(mazes.length).fill(1);
const uniformBR = bestResponse(best.group, uniform);

console.log(`${n}×${n}, ${B} walls: ${mazes.length} valid mazes, ${classes.size} (E, S) classes up to symmetry, ${T} iterations each (${((Date.now() - t0) / 1000).toFixed(1)}s)\n`);
console.log('Best (E, S) subgames for the designer (expected walls hit at equilibrium):');
for (const r of solved.slice(0, 5)) {
  const { e, s } = mazes[r.group[0]];
  console.log(`  E=${label(e)} S=${label(s)}: ${r.lower.toFixed(2)} … ${r.upper.toFixed(2)}   (${r.group.length} mazes, designer mixes ~${r.support.toFixed(0)} of them)`);
}
const worst = solved[solved.length - 1];
console.log(`  … worst: E=${label(mazes[worst.group[0]].e)} S=${label(mazes[worst.group[0]].s)}: ${worst.lower.toFixed(2)}\n`);
console.log(`Game value: ${best.lower.toFixed(2)} … ${best.upper.toFixed(2)} walls hit (max possible ${B}), with E=${label(bestE)}, S=${label(bestS)}\n`);
console.log('Explorer, in that subgame          vs equilibrium designer   vs its worst maze');
const row = (name, f) => console.log(`${name.padEnd(34)} ${vsDesigner(f).toFixed(2).padStart(22)} ${worstIn(f).toFixed(2).padStart(19)}`);
row('current bot ("hard" heuristic)', (mi) => heuristicHits(mi));
row('Bayes-optimal vs random walls', (mi, j) => uniformBR.hits[j]);
row('equilibrium explorer (mixed)', (mi, j) => best.avgHits[j]);
