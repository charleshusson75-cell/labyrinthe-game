// Maze model: { walls: string[] (edge keys), entrance: cell|null, exit: cell|null }.
// Settings: { size, walls } where `walls` is the exact number of walls each player must place.

import { allEdges, DIRS, edgeKey, isCell, isEdge, neighbor } from './grid.js';
import { pick, randInt, shuffle } from './random.js';

export const MIN_SIZE = 4;
export const MAX_SIZE = 8;
export const DEFAULT_SIZE = 5;
export const MIN_WALLS = 1;

/** ≈ ¾ of (size−1)², the wall count of a "perfect" maze (5×5 → 12). */
export const defaultWallBudget = (size) => Math.round(0.75 * (size - 1) ** 2);
export const maxWallBudget = (size) => (size - 1) ** 2 + size;

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

export function normalizeSettings(s = {}) {
  const size = clamp(Math.round(Number(s.size) || DEFAULT_SIZE), MIN_SIZE, MAX_SIZE);
  const walls = clamp(Math.round(Number(s.walls) || defaultWallBudget(size)), MIN_WALLS, maxWallBudget(size));
  return { size, walls };
}

export function emptyMaze() {
  return { walls: [], entrance: null, exit: null };
}

/** Sorted, de-duplicated copy — the form that gets hashed and compared. */
export function canonicalMaze(maze) {
  return { walls: [...new Set(maze.walls)].sort(), entrance: maze.entrance, exit: maze.exit };
}

/** true = passage, false = wall, null = outer border. */
export function isOpen(wallSet, cell, dir, size) {
  const n = neighbor(cell, dir, size);
  if (n < 0) return null;
  return !wallSet.has(edgeKey(cell, n));
}

/** BFS distances from `start` (−1 = unreachable). */
export function distancesFrom(size, wallSet, start) {
  const dist = new Array(size * size).fill(-1);
  dist[start] = 0;
  const queue = [start];
  for (let i = 0; i < queue.length; i++) {
    const c = queue[i];
    for (const d of DIRS) {
      const n = neighbor(c, d, size);
      if (n < 0 || dist[n] >= 0 || wallSet.has(edgeKey(c, n))) continue;
      dist[n] = dist[c] + 1;
      queue.push(n);
    }
  }
  return dist;
}

/** Cells of a shortest path from → to (inclusive), or null if unreachable. */
export function shortestPath(size, wallSet, from, to) {
  const prev = new Array(size * size).fill(-1);
  const seen = new Array(size * size).fill(false);
  seen[from] = true;
  const queue = [from];
  for (let i = 0; i < queue.length; i++) {
    const c = queue[i];
    if (c === to) break;
    for (const d of DIRS) {
      const n = neighbor(c, d, size);
      if (n < 0 || seen[n] || wallSet.has(edgeKey(c, n))) continue;
      seen[n] = true;
      prev[n] = c;
      queue.push(n);
    }
  }
  if (!seen[to]) return null;
  const path = [to];
  while (path[0] !== from) path.unshift(prev[path[0]]);
  return path;
}

/**
 * Rule violations of a maze for the given settings. Empty array = ready to play.
 * Issue codes: bad-walls, walls-left, too-many-walls, no-entrance, no-exit, same-cell, no-path.
 */
export function mazeIssues(maze, { size, walls: budget }) {
  const issues = [];
  const walls = Array.isArray(maze?.walls) ? maze.walls : [];
  if (walls.some((k) => !isEdge(k, size)) || new Set(walls).size !== walls.length) {
    issues.push({ code: 'bad-walls' });
  }
  if (walls.length < budget) issues.push({ code: 'walls-left', count: budget - walls.length });
  if (walls.length > budget) issues.push({ code: 'too-many-walls', count: walls.length - budget });
  const hasE = isCell(maze?.entrance, size);
  const hasS = isCell(maze?.exit, size);
  if (!hasE) issues.push({ code: 'no-entrance' });
  if (!hasS) issues.push({ code: 'no-exit' });
  if (hasE && hasS) {
    if (maze.entrance === maze.exit) issues.push({ code: 'same-cell' });
    else if (!shortestPath(size, new Set(walls), maze.entrance, maze.exit)) issues.push({ code: 'no-path' });
  }
  return issues;
}

/** Random spanning tree (recursive backtracker): returns the set of open edges. */
function spanningTree(size, rng) {
  const open = new Set();
  const seen = new Array(size * size).fill(false);
  const start = randInt(rng, size * size);
  const stack = [start];
  seen[start] = true;
  while (stack.length) {
    const c = stack[stack.length - 1];
    const next = DIRS.map((d) => neighbor(c, d, size)).filter((n) => n >= 0 && !seen[n]);
    if (!next.length) {
      stack.pop();
      continue;
    }
    const n = pick(rng, next);
    seen[n] = true;
    open.add(edgeKey(c, n));
    stack.push(n);
  }
  return open;
}

function tryRandomMaze(size, budget, rng, farRatio) {
  const open = spanningTree(size, rng);
  const walls = new Set(allEdges(size).filter((e) => !open.has(e)));
  const entrance = randInt(rng, size * size);
  const dist = distancesFrom(size, walls, entrance);
  const max = Math.max(...dist);
  const far = [];
  dist.forEach((d, cell) => {
    if (d > 0 && d >= Math.ceil(max * farRatio)) far.push(cell);
  });
  const exit = pick(rng, far);

  if (walls.size > budget) {
    // Fewer walls than a perfect maze: knock random walls down (creates loops), preferring ones
    // that don't open a shortcut between E and S. Shortcuts are only used if nothing else is left.
    const length = () => shortestPath(size, walls, entrance, exit).length;
    const target = length();
    const shortcuts = [];
    for (const e of shuffle(rng, [...walls])) {
      if (walls.size <= budget) break;
      walls.delete(e);
      if (length() < target) {
        walls.add(e);
        shortcuts.push(e);
      }
    }
    for (const e of shortcuts) {
      if (walls.size <= budget) break;
      walls.delete(e);
    }
  } else if (walls.size < budget) {
    // More walls than a perfect maze: close passages off the E→S path (seals dead ends).
    const path = shortestPath(size, walls, entrance, exit);
    const onPath = new Set(path.slice(1).map((c, i) => edgeKey(path[i], c)));
    for (const e of shuffle(rng, [...open].filter((e) => !onPath.has(e)))) {
      if (walls.size >= budget) break;
      walls.add(e);
    }
    if (walls.size < budget) return null;
  }
  return canonicalMaze({ walls: [...walls], entrance, exit });
}

/** A valid random maze with exactly `settings.walls` walls and a far-apart E/S. */
export function randomMaze(settings, rng) {
  const { size, walls } = normalizeSettings(settings);
  for (let attempt = 0; attempt < 60; attempt++) {
    const maze = tryRandomMaze(size, walls, rng, attempt < 30 ? 0.7 : 0.3);
    if (maze) return maze;
  }
  throw new Error(`Could not generate a ${size}×${size} maze with ${walls} walls`);
}
