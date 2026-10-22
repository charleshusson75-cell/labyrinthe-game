import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { designMaze, simulateSolve } from '../src/bot/brain.js';
import { seededRng } from '../src/game/random.js';

const SCRIPT = fileURLToPath(new URL('../scripts/expert-solve.mjs', import.meta.url));

test('the Python bridge plays exactly like simulateSolve at level expert', () => {
  const size = 4;
  const rng = seededRng(5);
  // Distinct entrance/exit pairs: expert.js caches a statistic per pair, which the bridge resets per maze.
  const mazes = [];
  while (mazes.length < 4) {
    const m = designMaze({ size, walls: 7 }, 'hard', rng);
    if (!mazes.some((o) => o.entrance === m.entrance && o.exit === m.exit)) mazes.push({ ...m, seed: 100 + mazes.length });
  }
  const expert = { maxParticles: 60 };
  const out = execFileSync(process.execPath, [SCRIPT], { input: JSON.stringify({ size, mazes, expert }) });
  const { hits } = JSON.parse(out);
  const direct = mazes.map((m) => simulateSolve(m, size, 'expert', seededRng(m.seed), { ...expert, budgetMs: Infinity }).hits);
  assert.deepEqual(hits, direct);
});
