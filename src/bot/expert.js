// Expert explorer (docs/RL_DESIGN.md, phase P2): a Bayesian search over the opponent's hidden maze.
//
// Belief: "particles", complete mazes consistent with everything seen so far (exactly W walls, S
// reachable from E, known passages open, known walls closed), drawn uniformly among those. A draw is a
// random completion kept only if valid; when that keeps failing, a completion is repaired (a path
// E→S is opened) and then mixed by MCMC moves that swap a wall with an unknown open side and keep the
// maze valid. Because every particle has exactly W walls, the belief "counts": with 11 of 12 walls
// found, the remaining sides are almost all open.
//
// Move choice: every legal probe is tried on every particle, the exploration is finished with a fast
// greedy planner (a rollout), and the walls hit are counted. The probe with the fewest expected hits
// wins (one step of policy improvement over the planner). Work stops at a time budget, so the number
// of particles adapts to the board size and to the machine.

import { allEdges, DIRS, edgeKey, neighbor } from '../game/grid.js';
import { randInt } from '../game/random.js';

const UNKNOWN = 0;
const OPEN = 1;
const WALL = 2;

export const EXPERT_DEFAULTS = {
  budgetMs: 150, // per move, sampling + rollouts (the controller adds its own "thinking" delay)
  maxParticles: 400,
  minParticles: 16, // evaluated even when over budget, so slow machines still play sensibly
  sampleShare: 0.3, // share of the budget sampling may use
  rollout: 'belief', // greedy planner used in rollouts: 'belief' (unknown side costs P(wall)) or 'optimistic' (costs 1)
};

const now = () => globalThis.performance?.now() ?? Date.now();

// ------------------------------------------------------------------ geometry as typed arrays

const geometries = new Map();

/** Edge indices and neighbours per (cell, direction), cached per board size. */
function geometry(size) {
  let g = geometries.get(size);
  if (g) return g;
  const keys = allEdges(size);
  const index = new Map(keys.map((k, i) => [k, i]));
  const N = size * size;
  const nbr = new Int16Array(N * 4).fill(-1);
  const edge = new Int16Array(N * 4).fill(-1);
  for (let c = 0; c < N; c++) {
    DIRS.forEach((d, j) => {
      const n = neighbor(c, d, size);
      if (n < 0) return;
      nbr[c * 4 + j] = n;
      edge[c * 4 + j] = index.get(edgeKey(c, n));
    });
  }
  g = {
    size,
    N,
    M: keys.length,
    keys,
    index,
    nbr,
    edge,
    // scratch buffers (single-threaded, reused by every call)
    seen: new Uint8Array(N),
    queue: new Int16Array(N),
    prevEdge: new Int16Array(N),
    dist: new Float64Array(N),
    done: new Uint8Array(N),
  };
  geometries.set(size, g);
  return g;
}

function knowledgeOf(g, progress) {
  const know = new Int8Array(g.M);
  for (const k of progress.open) know[g.index.get(k)] = OPEN;
  for (const k of progress.walls) know[g.index.get(k)] = WALL;
  const visited = new Uint8Array(g.N);
  for (const c of progress.visited) visited[c] = 1;
  return { know, visited };
}

/** Is `to` reachable from `from` when `wall[e]` = 1 marks walls? */
function connected(g, wall, from, to) {
  const { seen, queue, nbr, edge } = g;
  seen.fill(0);
  seen[from] = 1;
  queue[0] = from;
  for (let head = 0, tail = 1; head < tail; head++) {
    const c = queue[head];
    for (let j = 0; j < 4; j++) {
      const n = nbr[c * 4 + j];
      if (n < 0 || seen[n] || wall[edge[c * 4 + j]]) continue;
      if (n === to) return true;
      seen[n] = 1;
      queue[tail++] = n;
    }
  }
  return from === to;
}

// ------------------------------------------------------------------ belief: particle sampler

const REJECTION_TRIES = 8;

/**
 * Up to `count` mazes (Uint8Array, 1 = wall, indexed like allEdges) drawn uniformly from those
 * consistent with `know`. Returns [] when no valid completion exists.
 */
function sampleParticles(g, know, W, E, S, count, rng, deadline = Infinity) {
  const unknown = [];
  const base = new Uint8Array(g.M);
  let knownWalls = 0;
  for (let e = 0; e < g.M; e++) {
    if (know[e] === UNKNOWN) unknown.push(e);
    else if (know[e] === WALL) {
      base[e] = 1;
      knownWalls++;
    }
  }
  const k = W - knownWalls;
  if (k < 0 || k > unknown.length) return [];
  const pool = Int16Array.from(unknown);
  const out = [];
  for (let i = 0; i < count; i++) {
    if (i > 0 && now() > deadline) break;
    const p = drawParticle(g, know, base, pool, k, E, S, rng);
    if (!p) break;
    out.push(p);
  }
  return out;
}

function drawParticle(g, know, base, pool, k, E, S, rng) {
  let w = null;
  for (let t = 0; t < REJECTION_TRIES; t++) {
    w = randomCompletion(base, pool, k, rng);
    if (connected(g, w, E, S)) return w; // exact uniform draw
  }
  // Rejection keeps failing (tight constraints): repair the last draw, then mix it.
  if (!repair(g, know, w, pool, E, S, rng)) return null;
  mix(g, w, pool, E, S, 4 * k + 20, rng);
  return w;
}

/** `base` plus k walls on uniformly chosen unknown sides (partial Fisher–Yates over `pool`). */
function randomCompletion(base, pool, k, rng) {
  const w = base.slice();
  for (let j = 0; j < k; j++) {
    const r = j + randInt(rng, pool.length - j);
    const x = pool[r];
    pool[r] = pool[j];
    pool[j] = x;
    w[x] = 1;
  }
  return w;
}

/**
 * Open a path E→S that crosses as few unknown sides as possible, moving the walls it crosses to
 * unknown sides off that path. False when S can't be reached at all.
 */
function repair(g, know, w, pool, E, S, rng) {
  const { N, nbr, edge, dist, prevEdge } = g;
  // 0-1 BFS: known passages cost 0, unknown sides 1, known walls are impassable.
  dist.fill(Infinity);
  dist[E] = 0;
  const deque = new Int16Array(2 * N * 4 + 2);
  let head = N * 4;
  let tail = head;
  deque[tail++] = E;
  while (head < tail) {
    const c = deque[head++];
    for (let j = 0; j < 4; j++) {
      const n = nbr[c * 4 + j];
      if (n < 0) continue;
      const e = edge[c * 4 + j];
      if (know[e] === WALL) continue;
      const step = know[e] === OPEN ? 0 : 1;
      if (dist[c] + step < dist[n]) {
        dist[n] = dist[c] + step;
        prevEdge[n] = e;
        if (step === 0) deque[--head] = n;
        else deque[tail++] = n;
      }
    }
  }
  if (dist[S] === Infinity) return false;
  const onPath = new Uint8Array(g.M);
  const toMove = [];
  for (let c = S, guard = 0; c !== E && guard < N; guard++) {
    const e = prevEdge[c];
    onPath[e] = 1;
    if (w[e]) toMove.push(e);
    const [a, b] = edgeEnds(g, e);
    c = a === c ? b : a;
  }
  const spots = [];
  for (const e of pool) if (!w[e] && !onPath[e]) spots.push(e);
  if (spots.length < toMove.length) return false;
  for (const e of toMove) {
    const r = randInt(rng, spots.length);
    w[e] = 0;
    w[spots[r]] = 1;
    spots[r] = spots[spots.length - 1];
    spots.pop();
  }
  return connected(g, w, E, S);
}

function edgeEnds(g, e) {
  const key = g.keys[e];
  const dash = key.indexOf('-');
  return [Number(key.slice(0, dash)), Number(key.slice(dash + 1))];
}

/** Metropolis moves with a uniform target over valid mazes: swap an unknown wall with an unknown open side. */
function mix(g, w, pool, E, S, steps, rng) {
  const walls = [];
  const opens = [];
  for (const e of pool) (w[e] ? walls : opens).push(e);
  if (!walls.length || !opens.length) return;
  for (let s = 0; s < steps; s++) {
    const i = randInt(rng, walls.length);
    const j = randInt(rng, opens.length);
    const a = walls[i];
    const b = opens[j];
    w[a] = 0;
    w[b] = 1;
    if (connected(g, w, E, S)) {
      walls[i] = b;
      opens[j] = a;
    } else {
      w[a] = 1;
      w[b] = 0;
    }
  }
}

/**
 * Public sampler, for tests and research: mazes `{ walls, entrance, exit }` drawn uniformly among those
 * with `walls` walls, S reachable from E, and consistent with `progress` (from match.js progressOf).
 */
export function sampleMazes({ size, walls, entrance, exit, progress, count, rng }) {
  const g = geometry(size);
  const { know } = knowledgeOf(g, progress);
  return sampleParticles(g, know, walls, entrance, exit, count, rng).map((p) => ({
    walls: g.keys.filter((_, e) => p[e]),
    entrance,
    exit,
  }));
}

// ------------------------------------------------------------------ rollout planner

/** Greedy-planner costs to S: known passages 0, unknown sides `cost[e]`, known walls impassable. */
function distancesTo(g, know, cost, S) {
  const { N, nbr, edge, dist, done } = g;
  dist.fill(Infinity);
  done.fill(0);
  dist[S] = 0;
  for (let it = 0; it < N; it++) {
    let u = -1;
    let best = Infinity;
    for (let c = 0; c < N; c++) {
      if (!done[c] && dist[c] < best) {
        best = dist[c];
        u = c;
      }
    }
    if (u < 0) break;
    done[u] = 1;
    for (let j = 0; j < 4; j++) {
      const n = nbr[u * 4 + j];
      if (n < 0 || done[n]) continue;
      const e = edge[u * 4 + j];
      const kn = know[e];
      if (kn === WALL) continue;
      const d = best + (kn === OPEN ? 0 : cost[e]);
      if (d < dist[n]) dist[n] = d;
    }
  }
  return dist;
}

/**
 * The planner's next probe as a slot index (cell * 4 + direction), or −1: the unknown side next to a
 * reached cell that minimizes its cost plus the cost from there to S. Probing between two reached
 * cells only closes a loop, so it gets a small penalty (as in the heuristic levels).
 */
function greedyProbe(g, know, visited, cost, S) {
  const dist = distancesTo(g, know, cost, S);
  const { N, nbr, edge } = g;
  let bestSlot = -1;
  let bestScore = Infinity;
  for (let c = 0; c < N; c++) {
    if (!visited[c]) continue;
    for (let j = 0; j < 4; j++) {
      const slot = c * 4 + j;
      const e = edge[slot];
      if (e < 0 || know[e] !== UNKNOWN) continue;
      const n = nbr[slot];
      const score = cost[e] + dist[n] + (visited[n] ? 0.5 : 0);
      if (score < bestScore) {
        bestScore = score;
        bestSlot = slot;
      }
    }
  }
  return bestSlot;
}

/** Walls hit by the planner finishing the exploration of maze `wall` (mutates know/visited). */
function rollout(g, wall, know, visited, cost, S) {
  let hits = 0;
  for (let guard = 0; guard <= g.M && !visited[S]; guard++) {
    const slot = greedyProbe(g, know, visited, cost, S);
    if (slot < 0) return hits + g.M; // can't happen on a valid particle
    const e = g.edge[slot];
    if (wall[e]) {
      know[e] = WALL;
      hits++;
    } else {
      know[e] = OPEN;
      visited[g.nbr[slot]] = 1;
    }
  }
  return hits;
}

// ------------------------------------------------------------------ move choice

/**
 * The expert's probe `{ from, dir }`, or null when it has nothing to say (no legal probe, or no maze
 * fits the observations, e.g. a dishonest opponent): the caller then falls back to the heuristic.
 *
 * @param o.walls     the opponent's wall budget (settings.walls)
 * @param o.entrance  E of the opponent's maze
 * @param o.exit      S of the opponent's maze
 * @param o.progress  what we know (match.js progressOf)
 * @param o.budgetMs  time budget; Infinity = always use `maxParticles` (reproducible with a seeded rng)
 * @param o.stats     optional object filled with { particles, rollouts, ms }
 */
export function expertProbe({ size, walls, entrance, exit, progress, rng, stats, ...opts }) {
  const o = { ...EXPERT_DEFAULTS, ...opts };
  const t0 = now();
  const g = geometry(size);
  const { know, visited } = knowledgeOf(g, progress);

  const actions = []; // slots of legal probes
  for (let c = 0; c < g.N; c++) {
    if (!visited[c]) continue;
    for (let j = 0; j < 4; j++) {
      const e = g.edge[c * 4 + j];
      if (e >= 0 && know[e] === UNKNOWN) actions.push(c * 4 + j);
    }
  }
  const asProbe = (slot) => ({ from: slot >> 2, dir: DIRS[slot & 3] });
  if (actions.length <= 1) return actions.length ? asProbe(actions[0]) : null;

  const particles = sampleParticles(g, know, walls, entrance, exit, o.maxParticles, rng, t0 + o.budgetMs * o.sampleShare);
  if (!particles.length) return null;

  // Unknown side costs for the rollout planner: P(wall) under the belief, or 1 (optimistic).
  const cost = new Float64Array(g.M).fill(1);
  if (o.rollout === 'belief') {
    cost.fill(0);
    for (const p of particles) for (let e = 0; e < g.M; e++) if (p[e]) cost[e]++;
    for (let e = 0; e < g.M; e++) cost[e] = Math.max(0.01, cost[e] / particles.length);
  }

  // hits[a * P + i]: walls hit when probing action a first, then following the planner, on particle i.
  const A = actions.length;
  const P = particles.length;
  const hits = new Float32Array(A * P);
  const alive = actions.map((_, a) => a);
  const k2 = new Int8Array(g.M);
  const v2 = new Uint8Array(g.N);
  let n = 0;
  let rollouts = 0;
  const deadline = t0 + o.budgetMs;
  for (; n < P; n++) {
    if (n >= o.minParticles && now() > deadline) break;
    const wall = particles[n];
    for (const a of alive) {
      const slot = actions[a];
      const e = g.edge[slot];
      k2.set(know);
      v2.set(visited);
      let h = 0;
      if (wall[e]) {
        k2[e] = WALL;
        h = 1;
      } else {
        k2[e] = OPEN;
        v2[g.nbr[slot]] = 1;
      }
      hits[a * P + n] = h + rollout(g, wall, k2, v2, cost, exit);
      rollouts++;
    }
    if ((n + 1) % 16 === 0 && alive.length > 1) prune(alive, hits, P, n + 1);
  }

  // Lowest mean wins; ties go to the planner's own favourite (so free probes still head for S).
  const mean = (a) => {
    let s = 0;
    for (let i = 0; i < n; i++) s += hits[a * P + i];
    return s / n;
  };
  const dist = distancesTo(g, know, cost, exit);
  const prior = (a) => {
    const slot = actions[a];
    const to = g.nbr[slot];
    return cost[g.edge[slot]] + dist[to] + (visited[to] ? 0.5 : 0);
  };
  let best = alive[0];
  let bestMean = mean(best);
  for (const a of alive.slice(1)) {
    const m = mean(a);
    if (m < bestMean - 1e-9 || (m < bestMean + 1e-9 && prior(a) < prior(best))) {
      best = a;
      bestMean = m;
    }
  }
  if (stats) Object.assign(stats, { particles: n, rollouts, ms: now() - t0, expectedHits: bestMean });
  return asProbe(actions[best]);
}

/** Drop actions that are clearly worse than the current best (paired comparison over the same particles). */
function prune(alive, hits, P, n) {
  const mean = (a) => {
    let s = 0;
    for (let i = 0; i < n; i++) s += hits[a * P + i];
    return s / n;
  };
  let best = alive[0];
  for (const a of alive) if (mean(a) < mean(best)) best = a;
  for (let k = alive.length - 1; k >= 0; k--) {
    const a = alive[k];
    if (a === best) continue;
    let s = 0;
    let s2 = 0;
    for (let i = 0; i < n; i++) {
      const d = hits[a * P + i] - hits[best * P + i];
      s += d;
      s2 += d * d;
    }
    const m = s / n;
    const se = Math.sqrt(Math.max(0, s2 / n - m * m) / n);
    if (m > 3 * se + 1e-9) alive.splice(k, 1);
  }
}
