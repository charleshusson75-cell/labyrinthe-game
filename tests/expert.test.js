import { test } from 'node:test';
import assert from 'node:assert/strict';

import { allEdges, edgeKey, neighbor } from '../src/game/grid.js';
import { distancesFrom, isOpen, mazeIssues, randomMaze } from '../src/game/maze.js';
import { seededRng } from '../src/game/random.js';
import { probeError, progressOf } from '../src/game/match.js';
import { chooseProbe, designMaze, simulateSolve } from '../src/bot/brain.js';
import { expertProbe, sampleMazes } from '../src/bot/expert.js';

const UNIFORM = { length: 0, reach: 0 };
const TILTED = { length: 1.5, reach: 0.5 };
const INFERRED = { length: [0, 0.75, 1.5, 2.25], weights: [1, 1, 2, 2], reach: 0.5 };
const FAST = { budgetMs: Infinity, maxParticles: 60 };

/** Probes of a 'hard' bot on `maze`, cut after `stop` of them. */
function partialGame(maze, size, stop, rng) {
  const walls = new Set(maze.walls);
  const probes = [];
  while (probes.length < stop) {
    const progress = progressOf(probes, maze.entrance, size);
    if (progress.visited.has(maze.exit)) break;
    const p = chooseProbe({ size, progress, exit: maze.exit, level: 'hard', rng });
    probes.push({ from: p.from, dir: p.dir, open: isOpen(walls, p.from, p.dir, size) });
  }
  return probes;
}

/** Σ p(1 − p) over every side: 0 when the belief is a single maze. */
function uncertainty(samples, size) {
  let total = 0;
  for (const key of allEdges(size)) {
    const p = samples.filter((m) => m.walls.includes(key)).length / samples.length;
    total += p * (1 - p);
  }
  return total;
}

test('belief samples respect every constraint, under every prior', () => {
  const rng = seededRng(1);
  for (const [size, walls] of [
    [4, 7],
    [5, 12],
    [6, 19],
  ]) {
    const settings = { size, walls };
    for (let g = 0; g < 4; g++) {
      const maze = randomMaze(settings, rng);
      const probes = partialGame(maze, size, 3 + 6 * g, rng);
      const progress = progressOf(probes, maze.entrance, size);
      for (const prior of [UNIFORM, TILTED, INFERRED]) {
        const samples = sampleMazes({ size, walls, entrance: maze.entrance, exit: maze.exit, progress, count: 24, rng, prior });
        assert.ok(samples.length > 0, 'the true maze fits, so the belief is never empty');
        for (const m of samples) {
          assert.deepEqual(mazeIssues(m, settings), [], 'exact wall count, S reachable from E');
          const set = new Set(m.walls);
          for (const k of progress.open) assert.ok(!set.has(k), `known passage ${k} stays open`);
          for (const k of progress.walls) assert.ok(set.has(k), `known wall ${k} stays a wall`);
        }
      }
    }
  }
});

test('beliefs shrink as observations arrive, down to one maze once every wall is found', () => {
  const rng = seededRng(2);
  const size = 5;
  const walls = 12;
  for (let g = 0; g < 3; g++) {
    const maze = designMaze({ size, walls }, 'normal', rng);
    const all = partialGame(maze, size, Infinity, rng);
    const at = (k) => {
      const progress = progressOf(all.slice(0, k), maze.entrance, size);
      return uncertainty(sampleMazes({ size, walls, entrance: maze.entrance, exit: maze.exit, progress, count: 150, rng, prior: TILTED }), size);
    };
    const u = [0, 0.33, 0.66, 1].map((f) => at(Math.round(f * all.length)));
    for (let i = 1; i < u.length; i++) assert.ok(u[i] < u[i - 1], `uncertainty falls: ${u.map((x) => x.toFixed(2)).join(' → ')}`);
  }

  // Counting: once all walls are known, every other side must be open.
  const maze = randomMaze({ size: 4, walls: 7 }, rng);
  const progress = { visited: new Set([maze.entrance]), open: new Set(), walls: new Set(maze.walls) };
  for (const prior of [UNIFORM, TILTED]) {
    const samples = sampleMazes({ size: 4, walls: 7, entrance: maze.entrance, exit: maze.exit, progress, count: 20, rng, prior });
    assert.ok(samples.every((m) => [...m.walls].sort().join() === [...maze.walls].sort().join()));
  }
});

/** Every valid 3×3 maze with `walls` walls and the given E/S, as sorted wall lists. */
function enumerate3x3(walls, E, S) {
  const keys = allEdges(3);
  const out = [];
  (function rec(start, chosen) {
    if (chosen.length === walls) {
      if (distancesFrom(3, new Set(chosen), E)[S] >= 0) out.push([...chosen]);
      return;
    }
    for (let i = start; i < keys.length; i++) rec(i + 1, [...chosen, keys[i]]);
  })(0, []);
  return out;
}

test('the sampler draws from the right distribution (3×3, checked against enumeration)', () => {
  const rng = seededRng(3);
  const E = 4; // centre
  const S = 2; // top-right corner
  const keys = allEdges(3);
  const wall = edgeKey(4, 5); // observed: E's right side is a wall …
  const open = edgeKey(1, 4); // … and its top side is open
  const progress = { visited: new Set([E, 1]), open: new Set([open]), walls: new Set([wall]) };
  const fits = enumerate3x3(4, E, S).filter((w) => w.includes(wall) && !w.includes(open));
  for (const prior of [UNIFORM, TILTED]) {
    // exact marginals P(wall) under weight ∝ exp(β · path length + γ · reachable cells)
    const weight = (w) => {
      const d = distancesFrom(3, new Set(w), E);
      return Math.exp(prior.length * d[S] + prior.reach * d.filter((x) => x >= 0).length);
    };
    const z = fits.reduce((acc, w) => acc + weight(w), 0);
    const exact = keys.map((k) => fits.reduce((acc, w) => acc + (w.includes(k) ? weight(w) : 0), 0) / z);
    const samples = sampleMazes({ size: 3, walls: 4, entrance: E, exit: S, progress, count: 4000, rng, prior });
    const seen = keys.map((k) => samples.filter((m) => m.walls.includes(k)).length / samples.length);
    keys.forEach((k, i) => assert.ok(Math.abs(seen[i] - exact[i]) < 0.04, `${k}: sampled ${seen[i].toFixed(3)} vs exact ${exact[i].toFixed(3)}`));
  }
});

test('expert probes are legal and its games finish', () => {
  const rng = seededRng(4);
  for (const [size, walls] of [
    [3, 3],
    [4, 7],
    [5, 12],
  ]) {
    for (let g = 0; g < 3; g++) {
      const maze = size === 3 ? { walls: ['0-1', '4-5', '6-7'], entrance: 0, exit: 8 } : randomMaze({ size, walls }, rng);
      const probes = partialGame(maze, size, 4 * g, rng);
      const progress = progressOf(probes, maze.entrance, size);
      if (progress.visited.has(maze.exit)) continue;
      const p = expertProbe({ size, walls, entrance: maze.entrance, exit: maze.exit, progress, rng, ...FAST });
      assert.equal(probeError(progress, p.from, p.dir, size), null);
      assert.ok(neighbor(p.from, p.dir, size) >= 0);
      const { hits } = simulateSolve(maze, size, 'expert', rng, FAST);
      assert.ok(Number.isFinite(hits) && hits <= walls);
    }
  }
});

test('expert falls back to the heuristic when it lacks the wall budget', () => {
  const rng = seededRng(5);
  const maze = randomMaze({ size: 5, walls: 12 }, rng);
  const progress = progressOf([], maze.entrance, 5);
  const p = chooseProbe({ size: 5, progress, exit: maze.exit, level: 'expert', rng });
  assert.equal(probeError(progress, p.from, p.dir, 5), null);
});
