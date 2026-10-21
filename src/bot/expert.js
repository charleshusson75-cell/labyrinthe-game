// Expert explorer (docs/RL_DESIGN.md, P2 without learning): a Bayesian search over the opponent's
// hidden maze. Measured results: docs/DECISIONS.md D21.
//
// Belief: "particles", complete mazes consistent with everything seen so far (exactly W walls, S
// reachable from E, known passages open, known walls closed). Because every particle has exactly W
// walls, the belief "counts": with 11 of 12 walls found, the remaining sides are almost all open.
// Particles are drawn from a prior over designs, restricted to what was seen: uniform draws come from
// rejection; tilted ones from Metropolis chains that swap a wall with an unknown open side and keep
// the maze valid. The prior matters more than anything else: people (and the built-in designers) draw
// winding paths, so the default prior weighs mazes by their E→S path length, and infers during the
// round how strongly this designer does it.
//
// Move choice: every legal probe is tried on every particle, the exploration is finished with a fast
// greedy planner (a rollout) whose unknown sides cost their probability of being a wall, and the walls
// hit are counted. The probe with the fewest expected hits wins (one step of policy improvement over
// the planner). Work stops at a time budget, so the number of particles adapts to board and machine.

import { allEdges, DIRS, edgeKey, neighbor } from '../game/grid.js';
import { randInt, shuffle } from '../game/random.js';

const UNKNOWN = 0;
const OPEN = 1;
const WALL = 2;

export const EXPERT_DEFAULTS = {
  budgetMs: 150, // per move, sampling + rollouts (the controller adds its own "thinking" delay)
  maxParticles: 400,
  minParticles: 16, // evaluated even when over budget, so slow machines still play sensibly
  sampleShare: 0.3, // share of the budget sampling may use
  // Designer style (see makeSampler and believe): either uniform walls (β = 0) or a designed maze
  // with a winding path (β = 1.5), 1 : 9 a priori, inferred from what the explorer sees.
  prior: { length: [0, 1.5], weights: [1, 9], reach: 0.5 },
  mixSteps: 0, // MCMC burn-in under a tilted prior (0 = automatic)
  chains: 16, // MCMC chains per prior (particles from one chain are correlated)
  rollout: 'belief', // greedy planner used in rollouts: 'belief' (unknown side costs P(wall)) or 'optimistic' (costs 1)
  search: 'rollout', // or 'greedy': the planner alone, without rollouts (faster, about one wall weaker on 5×5)
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
    hops: new Int16Array(N),
    dist: new Float64Array(N),
    heapCell: new Int16Array(4 * N + 4),
    heapKey: new Float64Array(4 * N + 4),
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
const CHAINS = 16; // MCMC chains per prior when not given (particles from one chain are correlated)
const UNIFORM = { length: 0, reach: 0 };

/**
 * A function drawing mazes (Uint8Array, 1 = wall, indexed like allEdges) consistent with `know`, from
 * the prior restricted to them; null when no valid completion exists.
 *
 * prior = { length: β, reach: γ }: weight ∝ exp(β · (E→S path length) + γ · (cells reachable from E)).
 * β = γ = 0 is uniform over valid mazes (exact draws by rejection). Otherwise draws come from a few
 * Metropolis chains: burnt in from a valid completion, then thinned.
 */
function makeSampler(g, know, W, E, S, rng, prior = UNIFORM, mixSteps = 0, chainCount = CHAINS) {
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
  if (k < 0 || k > unknown.length) return null;
  const pool = Int16Array.from(unknown);
  const first = () => drawParticle(g, know, base, pool, k, E, S, rng);
  if (prior.length === 0 && prior.reach === 0) {
    return () => {
      const p = first();
      if (p?.repaired) mix(g, p.w, pool, E, S, 4 * k + 20, rng, UNIFORM);
      return p?.w ?? null;
    };
  }
  const burnIn = mixSteps || 30 * k + 40;
  const thin = Math.max(1, Math.round(burnIn / 12));
  const chains = [];
  let turn = 0;
  return () => {
    const c = turn++ % chainCount;
    if (!chains[c]) {
      const p = first();
      if (!p) return null;
      mix(g, p.w, pool, E, S, burnIn, rng, prior);
      chains[c] = p.w;
    } else mix(g, chains[c], pool, E, S, thin, rng, prior);
    return chains[c].slice();
  };
}

/** Up to `count` particles from `prior` (see makeSampler); [] when nothing fits. */
function sampleParticles(g, know, W, E, S, count, rng, { prior = UNIFORM, mixSteps = 0, chains, deadline = Infinity } = {}) {
  const draw = makeSampler(g, know, W, E, S, rng, prior, mixSteps, chains);
  const out = [];
  for (let i = 0; draw && i < count; i++) {
    if (i > 0 && now() > deadline) break;
    const p = draw();
    if (!p) break;
    out.push(p);
  }
  return out;
}

/**
 * Particles when the designer's style is itself unknown: prior.length is a list of β values with
 * prior weights. Each β gets its own particles, and its posterior weight follows from what was seen.
 * The evidence for β is P(observations | β) = Z_β(obs) / Z_β, and d/dβ log Z = E[path length], so
 *   log P(obs | β) − log P(obs | β₀) = ∫ (E_{b,obs}[L] − E_b[L]) db   from β₀ to β
 * (thermodynamic integration): mazes seen to wind more than the prior expects favour larger β.
 * The particles are then resampled in proportion to the posterior.
 */
function believe(g, know, W, E, S, count, rng, { prior, mixSteps, chains, deadline }) {
  const betas = [].concat(prior.length);
  if (betas.length === 1) {
    const particles = sampleParticles(g, know, W, E, S, count, rng, { prior: { length: betas[0], reach: prior.reach }, mixSteps, chains, deadline });
    return { particles, posterior: [1] };
  }
  const hyper = prior.weights ?? betas.map(() => 1);
  const draws = betas.map((b) => makeSampler(g, know, W, E, S, rng, { length: b, reach: prior.reach }, mixSteps, chains));
  if (!draws[0]) return { particles: [], posterior: [] };
  const groups = betas.map(() => []);
  const per = Math.ceil(count / betas.length);
  fill: for (let i = 0; i < per; i++) {
    for (let j = 0; j < betas.length; j++) {
      if (i > 0 && now() > deadline) break fill;
      const p = draws[j]();
      if (!p) return { particles: [], posterior: [] };
      groups[j].push(p);
    }
  }
  const lengthOf = (w) => score(g, w, E, S, { length: 1, reach: 0 });
  const meanLength = (ps) => ps.reduce((acc, w) => acc + lengthOf(w), 0) / ps.length;
  const seen = groups.map(meanLength);
  const free = betas.map((b) => unconditionedLength(g, W, E, S, { length: b, reach: prior.reach }, mixSteps, rng));
  const logw = [Math.log(hyper[0])];
  for (let j = 1, acc = 0; j < betas.length; j++) {
    acc += ((betas[j] - betas[j - 1]) * (seen[j] - free[j] + seen[j - 1] - free[j - 1])) / 2;
    logw.push(Math.log(hyper[j]) + acc);
  }
  const top = Math.max(...logw);
  const raw = logw.map((x) => Math.exp(x - top));
  const z = raw.reduce((x, y) => x + y, 0);
  const posterior = raw.map((x) => x / z);
  const total = groups.reduce((acc, ps) => acc + ps.length, 0);
  const particles = [];
  groups.forEach((ps, j) => {
    const take = Math.round(posterior[j] * total);
    for (let i = 0; i < take; i++) particles.push(ps[i % ps.length]);
  });
  // Shuffled, so that a move cut short by the time budget still sees every group.
  return { particles: shuffle(rng, particles.length ? particles : groups.flat()), posterior };
}

/** E_β[path length] before any observation, per board, E/S and prior (cached: it only depends on those). */
const freeLengths = new Map();
const FREE_SAMPLES = 120;
function unconditionedLength(g, W, E, S, prior, mixSteps, rng) {
  const key = [g.size, W, E, S, prior.length, prior.reach, mixSteps].join();
  if (!freeLengths.has(key)) {
    const know = new Int8Array(g.M);
    const ps = sampleParticles(g, know, W, E, S, FREE_SAMPLES, rng, { prior, mixSteps });
    const mean = ps.reduce((acc, w) => acc + score(g, w, E, S, { length: 1, reach: 0 }), 0) / Math.max(1, ps.length);
    if (freeLengths.size > 64) freeLengths.clear();
    freeLengths.set(key, mean);
  }
  return freeLengths.get(key);
}

/** A valid completion: uniform when rejection succeeds, otherwise repaired (and to be mixed). */
function drawParticle(g, know, base, pool, k, E, S, rng) {
  let w = null;
  for (let t = 0; t < REJECTION_TRIES; t++) {
    w = randomCompletion(base, pool, k, rng);
    if (connected(g, w, E, S)) return { w, repaired: false };
  }
  return repair(g, know, w, pool, E, S, rng) ? { w, repaired: true } : null;
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

/**
 * Metropolis moves: swap an unknown wall with an unknown open side, keep the maze valid, and accept
 * with the prior's weight ratio (always, for the uniform prior). The proposal is symmetric.
 */
function mix(g, w, pool, E, S, steps, rng, prior) {
  const walls = [];
  const opens = [];
  for (const e of pool) (w[e] ? walls : opens).push(e);
  if (!walls.length || !opens.length) return;
  let energy = score(g, w, E, S, prior);
  for (let s = 0; s < steps; s++) {
    const i = randInt(rng, walls.length);
    const j = randInt(rng, opens.length);
    const a = walls[i];
    const b = opens[j];
    w[a] = 0;
    w[b] = 1;
    const next = score(g, w, E, S, prior);
    if (next > -Infinity && (next >= energy || rng() < Math.exp(next - energy))) {
      walls[i] = b;
      opens[j] = a;
      energy = next;
    } else {
      w[a] = 1;
      w[b] = 0;
    }
  }
}

/** Log prior weight of a maze up to a constant; −Infinity when S is unreachable. */
function score(g, w, E, S, prior) {
  if (prior.length === 0 && prior.reach === 0) return connected(g, w, E, S) ? 0 : -Infinity;
  const { seen, queue, nbr, edge, hops } = g;
  seen.fill(0);
  seen[E] = 1;
  hops[E] = 0;
  queue[0] = E;
  let tail = 1;
  for (let head = 0; head < tail; head++) {
    const c = queue[head];
    for (let j = 0; j < 4; j++) {
      const n = nbr[c * 4 + j];
      if (n < 0 || seen[n] || w[edge[c * 4 + j]]) continue;
      seen[n] = 1;
      hops[n] = hops[c] + 1;
      queue[tail++] = n;
    }
  }
  if (!seen[S]) return -Infinity;
  return prior.length * hops[S] + prior.reach * tail;
}

/**
 * The belief as mazes `{ walls, entrance, exit }`, for tests and research: `walls` walls, S reachable
 * from E, consistent with `progress` (match.js progressOf), drawn from `prior` (default uniform; a list
 * of β values infers the designer's style, see believe).
 */
export function sampleMazes({ size, walls, entrance, exit, progress, count, rng, prior = UNIFORM, mixSteps = 0 }) {
  const g = geometry(size);
  const { know } = knowledgeOf(g, progress);
  const { particles } = believe(g, know, walls, entrance, exit, count, rng, { prior, mixSteps, deadline: Infinity });
  return particles.map((p) => ({
    walls: g.keys.filter((_, e) => p[e]),
    entrance,
    exit,
  }));
}

// ------------------------------------------------------------------ rollout planner

/** Planner costs to S (Dijkstra, binary heap): known passages 0, unknown sides `cost[e]`, known walls impassable. */
function distancesTo(g, know, cost, S) {
  const { nbr, edge, dist, heapCell, heapKey } = g;
  dist.fill(Infinity);
  dist[S] = 0;
  heapCell[0] = S;
  heapKey[0] = 0;
  let size = 1;
  while (size) {
    const c = heapCell[0];
    const d = heapKey[0];
    // pop: move the last entry to the root and sift it down
    size--;
    const lastCell = heapCell[size];
    const lastKey = heapKey[size];
    for (let i = 0; ; ) {
      let m = 2 * i + 1;
      if (m >= size) {
        heapCell[i] = lastCell;
        heapKey[i] = lastKey;
        break;
      }
      if (m + 1 < size && heapKey[m + 1] < heapKey[m]) m++;
      if (heapKey[m] >= lastKey) {
        heapCell[i] = lastCell;
        heapKey[i] = lastKey;
        break;
      }
      heapCell[i] = heapCell[m];
      heapKey[i] = heapKey[m];
      i = m;
    }
    if (d > dist[c]) continue; // stale entry
    for (let j = 0; j < 4; j++) {
      const n = nbr[c * 4 + j];
      if (n < 0) continue;
      const e = edge[c * 4 + j];
      const kn = know[e];
      if (kn === WALL) continue;
      const nd = d + (kn === OPEN ? 0 : cost[e]);
      if (nd >= dist[n]) continue;
      dist[n] = nd;
      // push: sift up
      let i = size++;
      while (i > 0) {
        const parent = (i - 1) >> 1;
        if (heapKey[parent] <= nd) break;
        heapCell[i] = heapCell[parent];
        heapKey[i] = heapKey[parent];
        i = parent;
      }
      heapCell[i] = n;
      heapKey[i] = nd;
    }
  }
  return dist;
}

/**
 * The planner's next probe as a slot index (cell * 4 + direction), or −1: the unknown side next to a
 * reached cell that minimizes its cost plus `dist` from there to S. Probing between two reached cells
 * only closes a loop, so it gets a small penalty (as in the heuristic levels).
 */
function bestProbe(g, know, visited, cost, dist) {
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

/**
 * Walls hit by the planner finishing the exploration of maze `wall` (mutates know/visited).
 * Distances are only recomputed after a wall: finding a passage on the planned path lowers that
 * path's cost and no other's, so the plan stays optimal and the old distances still pick its next step.
 */
function rollout(g, wall, know, visited, cost, S) {
  let hits = 0;
  let dist = distancesTo(g, know, cost, S);
  for (let guard = 0; guard <= g.M && !visited[S]; guard++) {
    const slot = bestProbe(g, know, visited, cost, dist);
    if (slot < 0) return hits + g.M; // can't happen on a valid particle
    const e = g.edge[slot];
    if (wall[e]) {
      know[e] = WALL;
      hits++;
      dist = distancesTo(g, know, cost, S);
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
 * @param o.stats     optional object filled with { particles, rollouts, ms, expectedHits, posterior }
 * Other options: see EXPERT_DEFAULTS.
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

  const { particles, posterior } = believe(g, know, walls, entrance, exit, o.maxParticles, rng, {
    prior: o.prior,
    mixSteps: o.mixSteps,
    chains: o.chains,
    deadline: t0 + o.budgetMs * o.sampleShare,
  });
  if (!particles.length) return null;

  // Unknown side costs for the rollout planner: P(wall) under the belief, or 1 (optimistic).
  const cost = new Float64Array(g.M).fill(1);
  if (o.rollout === 'belief') {
    cost.fill(0);
    for (const p of particles) for (let e = 0; e < g.M; e++) if (p[e]) cost[e]++;
    for (let e = 0; e < g.M; e++) cost[e] = Math.max(0.01, cost[e] / particles.length);
  }
  if (o.search === 'greedy') return asProbe(bestProbe(g, know, visited, cost, distancesTo(g, know, cost, exit)));

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
  const planScore = (a) => {
    const slot = actions[a];
    const to = g.nbr[slot];
    return cost[g.edge[slot]] + dist[to] + (visited[to] ? 0.5 : 0);
  };
  let best = alive[0];
  let bestMean = mean(best);
  for (const a of alive.slice(1)) {
    const m = mean(a);
    if (m < bestMean - 1e-9 || (m < bestMean + 1e-9 && planScore(a) < planScore(best))) {
      best = a;
      bestMean = m;
    }
  }
  if (stats) Object.assign(stats, { particles: n, rollouts, ms: now() - t0, expectedHits: bestMean, posterior });
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
