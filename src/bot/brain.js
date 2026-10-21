// Computer opponent: how it designs mazes and how it explores yours.

import { DIRS, edgeKey, neighbor } from '../game/grid.js';
import { probeError, progressOf } from '../game/match.js';
import { isOpen, randomMaze } from '../game/maze.js';
import { pick } from '../game/random.js';
import { expertProbe } from './expert.js';

export const LEVELS = ['easy', 'normal', 'hard', 'expert'];

/**
 * Optimistic distance from every cell to `exit`: known passages cost 0, unknown sides cost 1
 * (assumed open), known walls are impassable. 0-1 BFS.
 */
export function optimisticDistances(size, progress, exit) {
  const dist = new Array(size * size).fill(Infinity);
  dist[exit] = 0;
  const deque = [exit];
  while (deque.length) {
    const c = deque.shift();
    for (const d of DIRS) {
      const n = neighbor(c, d, size);
      if (n < 0) continue;
      const key = edgeKey(c, n);
      if (progress.walls.has(key)) continue;
      const w = progress.open.has(key) ? 0 : 1;
      if (dist[c] + w < dist[n]) {
        dist[n] = dist[c] + w;
        if (w === 0) deque.unshift(n);
        else deque.push(n);
      }
    }
  }
  return dist;
}

/** Every legal probe with a score (lower = more promising). */
export function rankProbes(size, progress, exit) {
  const dist = optimisticDistances(size, progress, exit);
  const out = [];
  for (const from of progress.visited) {
    for (const dir of DIRS) {
      if (probeError(progress, from, dir, size)) continue;
      const to = neighbor(from, dir, size);
      // Prefer probes that reach new cells; probing between two known cells only closes a loop.
      const score = 1 + dist[to] + (progress.visited.has(to) ? 0.5 : 0);
      out.push({ from, dir, score });
    }
  }
  return out.sort((a, b) => a.score - b.score);
}

/**
 * The bot's next probe. 'expert' searches a belief over the hidden maze (expert.js) and needs the
 * opponent's entrance and wall budget; without them, or if no maze fits what it saw, it plays 'hard'.
 */
export function chooseProbe({ size, progress, exit, level = 'normal', rng, entrance, wallCount, expert }) {
  if (level === 'expert' && entrance != null && wallCount != null) {
    const p = expertProbe({ size, walls: wallCount, entrance, exit, progress, rng, ...expert });
    if (p) return p;
  }
  const ranked = rankProbes(size, progress, exit);
  if (!ranked.length) return null;
  const best = ranked.filter((p) => p.score === ranked[0].score);
  const roll = rng();
  if (level === 'easy' && roll < 0.8) return pick(rng, ranked);
  if (level === 'normal' && roll < 0.25) return pick(rng, ranked.slice(0, 4));
  return pick(rng, best);
}

/**
 * Play a whole exploration of `maze` alone; returns how many walls were hit (= turns lost).
 * `expert` = options for the expert level (expert.js), e.g. a fixed particle count for benchmarks.
 */
export function simulateSolve(maze, size, level, rng, expert) {
  const walls = new Set(maze.walls);
  const probes = [];
  for (let guard = 0; guard < 4 * size * size; guard++) {
    const progress = progressOf(probes, maze.entrance, size);
    if (progress.visited.has(maze.exit)) return { hits: progress.hits, moves: progress.moves };
    const p = chooseProbe({ size, progress, exit: maze.exit, level, rng, entrance: maze.entrance, wallCount: walls.size, expert });
    if (!p) break;
    probes.push({ from: p.from, dir: p.dir, open: isOpen(walls, p.from, p.dir, size) });
  }
  return { hits: Infinity, moves: probes.length };
}

/** A maze for the bot. Harder levels generate candidates and keep the one its own solver hates most. */
export function designMaze(settings, level, rng) {
  // Fewer candidates on big boards, where each simulation costs more: keeps design under ~200 ms.
  const scale = Math.min(1, 25 / settings.size ** 2);
  const candidates = Math.max(1, Math.round(({ easy: 1, normal: 6, hard: 30 }[level] ?? 6) * scale));
  let best = null;
  let bestScore = -1;
  for (let i = 0; i < candidates; i++) {
    const maze = randomMaze(settings, rng);
    let total = 0;
    const runs = level === 'hard' ? 4 : 2;
    for (let k = 0; k < runs; k++) total += simulateSolve(maze, settings.size, 'hard', rng).hits;
    if (total > bestScore) {
      best = maze;
      bestScore = total;
    }
  }
  return best;
}
