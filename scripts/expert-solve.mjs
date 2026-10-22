// Walls hit by the expert explorer (src/bot/expert.js, D21) on given mazes: the bridge that lets the
// Python evaluator (rl/labyrinthe_rl/expert.py, bench bot "expert") play it.
//
//   node scripts/expert-solve.mjs < request.json > response.json
//
// request:  { "size": 5, "mazes": [{ "walls": ["0-1", ...], "entrance": 3, "exit": 21, "seed": 7 }, ...],
//             "expert": { "maxParticles": 200 } }      (optional expert.js options, see EXPERT_DEFAULTS)
// response: { "hits": [5, 4, ...] }                    (null where the exploration failed)
//
// Deterministic: a fixed particle count instead of the time budget (as in scripts/benchmark.mjs), and
// each maze is explored with seededRng(seed) by its own fresh copy of expert.js. expert.js caches a
// prior statistic drawn with the caller's rng, so in a shared copy a maze's result would depend on the
// mazes explored before it; a fresh copy makes it depend on the maze and its seed only (≈ 5% slower).
// The explorer budget is the maze's own wall count, as in brain.js simulateSolve.

import { chooseProbe } from '../src/bot/brain.js';
import { progressOf } from '../src/game/match.js';
import { isOpen } from '../src/game/maze.js';
import { seededRng } from '../src/game/random.js';

const EXPERT = { maxParticles: 200 };

/** brain.js simulateSolve at level 'expert', with `expertProbe` from a given copy of expert.js. */
function solve(expertProbe, maze, size, rng, options) {
  const walls = new Set(maze.walls);
  const probes = [];
  for (let guard = 0; guard < 4 * size * size; guard++) {
    const progress = progressOf(probes, maze.entrance, size);
    if (progress.visited.has(maze.exit)) return progress.hits;
    // As chooseProbe at level 'expert': the expert, or the 'hard' heuristic when it has nothing to say.
    const p =
      expertProbe({ size, walls: walls.size, entrance: maze.entrance, exit: maze.exit, progress, rng, ...options }) ??
      chooseProbe({ size, progress, exit: maze.exit, level: 'hard', rng });
    if (!p) break;
    probes.push({ from: p.from, dir: p.dir, open: isOpen(walls, p.from, p.dir, size) });
  }
  return null;
}

let input = '';
for await (const chunk of process.stdin) input += chunk;
const { size, mazes, expert = {} } = JSON.parse(input);
const options = { ...EXPERT, ...expert, budgetMs: Infinity };
const hits = [];
for (const [i, maze] of mazes.entries()) {
  const { expertProbe } = await import(`../src/bot/expert.js?maze=${i}`);
  hits.push(solve(expertProbe, maze, size, seededRng(maze.seed), options));
}
process.stdout.write(JSON.stringify({ hits }));
