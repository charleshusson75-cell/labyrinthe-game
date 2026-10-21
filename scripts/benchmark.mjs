// Bot-vs-bot benchmark: how strong is each level, and how much does luck (or moving first) matter?
//
//   node scripts/benchmark.mjs [size] [walls] [games]     e.g. node scripts/benchmark.mjs 7 27 150
//
// For each pairing, both bots design a maze, each explores the other's, and the round is decided by
// the turn rule: you finish on turn (walls hit + 1), and the starter wins ties.
// The expert explorer runs with a fixed particle count instead of its time budget, so results don't
// depend on the machine.

import { designMaze, LEVELS, simulateSolve } from '../src/bot/brain.js';
import { defaultWallBudget, normalizeSettings } from '../src/game/maze.js';
import { seededRng } from '../src/game/random.js';

const [size = 5, walls, games = 200] = process.argv.slice(2).map(Number);
const settings = normalizeSettings({ size, walls: walls || defaultWallBudget(size) });
const rng = seededRng(2026);
const EXPERT = { budgetMs: Infinity, maxParticles: 200 };
const solve = (maze, level) => simulateSolve(maze, settings.size, level, rng, EXPERT).hits;

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const sd = (a) => Math.sqrt(mean(a.map((x) => (x - mean(a)) ** 2)));
const pct = (x) => `${Math.round(100 * x)}%`.padStart(4);

function match(a, b) {
  let winsA = 0;
  let ties = 0;
  let starterWins = 0;
  const hitsA = [];
  const hitsB = [];
  for (let i = 0; i < games; i++) {
    const mazeA = designMaze(settings, a, rng);
    const mazeB = designMaze(settings, b, rng);
    const hA = solve(mazeB, a); // a explores b's maze
    const hB = solve(mazeA, b);
    hitsA.push(hA);
    hitsB.push(hB);
    const aStarts = rng() < 0.5;
    const aWins = aStarts ? hA <= hB : hA < hB;
    if (aWins) winsA++;
    if (hA === hB) ties++;
    if (aWins === aStarts) starterWins++;
  }
  return { winA: winsA / games, ties: ties / games, starterWins: starterWins / games, hitsA, hitsB };
}

const t0 = Date.now();
console.log(`${settings.size}×${settings.size}, ${settings.walls} walls, ${games} rounds per cell or pairing\n`);

// Each explorer against each designer ('expert' designs like 'hard', so it has no column of its own).
const designers = LEVELS.filter((l) => l !== 'expert');
console.log(`walls hit (of ${settings.walls})  ${designers.map((d) => `${d} maze`.padStart(13)).join('')}`);
for (const explorer of LEVELS) {
  const cells = designers.map((designer) => {
    const hits = Array.from({ length: games }, () => solve(designMaze(settings, designer, rng), explorer));
    return `${mean(hits).toFixed(1)} ± ${sd(hits).toFixed(1)}`.padStart(13);
  });
  console.log(`${explorer.padEnd(6)} explores      ${cells.join('')}`);
}

console.log('\npairing          A wins  equal turns  starter wins  walls hit by A  walls hit by B');
const pairings = [
  ['hard', 'easy'],
  ['hard', 'normal'],
  ['normal', 'easy'],
  ['hard', 'hard'],
  ['expert', 'normal'],
  ['expert', 'hard'],
  ['expert', 'expert'],
];
for (const [a, b] of pairings) {
  const r = match(a, b);
  const hits = (h) => `${mean(h).toFixed(1)} ± ${sd(h).toFixed(1)}`.padStart(14);
  console.log(
    `${a.padEnd(6)} vs ${b.padEnd(6)}   ${pct(r.winA)}  ${pct(r.ties).padStart(11)}  ${pct(r.starterWins).padStart(12)}  ${hits(r.hitsA)}  ${hits(r.hitsB)}`,
  );
}
console.log(`\n(${((Date.now() - t0) / 1000).toFixed(0)} s)`);
