// SVG rendering of a board, plus pointer → cell/edge hit-testing.
// Walls get a tiny deterministic wobble so they look hand-drawn, like the paper original.

import { colOf, edgeCells, edgeKey, neighbor, rowOf } from '../game/grid.js';

export const CELL = 60;
export const PAD = 28; // room for coordinates on the top/left
const MARGIN = 10;

export const viewSize = (n) => PAD + n * CELL + MARGIN;
const cx = (cell, n) => PAD + (colOf(cell, n) + 0.5) * CELL;
const cy = (cell, n) => PAD + (rowOf(cell, n) + 0.5) * CELL;
const r1 = (x) => Math.round(x * 10) / 10;

function hash(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** Endpoints of the segment for edge `key`. */
function segment(key, n) {
  const [a, b] = edgeCells(key);
  const r = rowOf(a, n);
  const c = colOf(a, n);
  if (b === a + 1) {
    const x = PAD + (c + 1) * CELL;
    return [x, PAD + r * CELL, x, PAD + (r + 1) * CELL];
  }
  const y = PAD + (r + 1) * CELL;
  return [PAD + c * CELL, y, PAD + (c + 1) * CELL, y];
}

/** Slightly wobbly pen stroke along an edge. */
function wallPath(key, n, inset = 3) {
  const [x1, y1, x2, y2] = segment(key, n);
  const h = hash(key);
  const j = (k, amp) => ((((h >>> (k * 5)) & 31) / 31) - 0.5) * 2 * amp;
  const horizontal = y1 === y2;
  const ax = horizontal ? x1 + inset : x1 + j(0, 1.4);
  const ay = horizontal ? y1 + j(0, 1.4) : y1 + inset;
  const bx = horizontal ? x2 - inset : x2 + j(1, 1.4);
  const by = horizontal ? y2 + j(1, 1.4) : y2 - inset;
  const mx = (x1 + x2) / 2 + (horizontal ? j(2, 3) : j(2, 2.2));
  const my = (y1 + y2) / 2 + (horizontal ? j(3, 2.2) : j(3, 3));
  return `M${r1(ax)} ${r1(ay)}Q${r1(mx)} ${r1(my)} ${r1(bx)} ${r1(by)}`;
}

function passagePath(key, n) {
  const [a, b] = edgeCells(key);
  return `M${cx(a, n)} ${cy(a, n)}L${cx(b, n)} ${cy(b, n)}`;
}

function hitMark(key, n) {
  const [x1, y1, x2, y2] = segment(key, n);
  const x = (x1 + x2) / 2;
  const y = (y1 + y2) / 2;
  const s = 6;
  return `M${x - s} ${y - s}L${x + s} ${y + s}M${x + s} ${y - s}L${x - s} ${y + s}`;
}

function gridPath(n) {
  let d = '';
  for (let i = 1; i < n; i++) {
    const p = PAD + i * CELL;
    d += `M${p} ${PAD}V${PAD + n * CELL}M${PAD} ${p}H${PAD + n * CELL}`;
  }
  return d;
}

function cellRect(cell, n, cls, inset = 0) {
  const x = PAD + colOf(cell, n) * CELL + inset;
  const y = PAD + rowOf(cell, n) * CELL + inset;
  return `<rect class="${cls}" x="${x}" y="${y}" width="${CELL - 2 * inset}" height="${CELL - 2 * inset}" rx="${inset ? 6 : 0}"/>`;
}

function mark(cell, n, letter, cls) {
  if (cell == null) return '';
  return `<text class="b-mark ${cls}" x="${cx(cell, n)}" y="${cy(cell, n) + 10}" text-anchor="middle">${letter}</text>`;
}

/** Little arrow on the side of `cell` facing `dir`: "you can try this way". */
function chevron(cell, dir, n) {
  const x = cx(cell, n);
  const y = cy(cell, n);
  const o = CELL * 0.33;
  const s = 6;
  const pts = {
    U: [[x - s, y - o + s], [x, y - o], [x + s, y - o + s]],
    D: [[x - s, y + o - s], [x, y + o], [x + s, y + o - s]],
    L: [[x - o + s, y - s], [x - o, y], [x - o + s, y + s]],
    R: [[x + o - s, y - s], [x + o, y], [x + o - s, y + s]],
  }[dir];
  return `<polyline class="b-chevron" points="${pts.map((p) => p.join(',')).join(' ')}"/>`;
}

/**
 * @param o.variant  'design' | 'explore' | 'defend' | 'reveal'
 * @param o.who      whose exploration is overlaid: 'me' | 'them'
 */
export function renderBoard(o) {
  const n = o.size;
  const V = viewSize(n);
  const who = o.who ?? 'me';
  const parts = [];

  parts.push(`<rect class="b-paper" x="${PAD}" y="${PAD}" width="${n * CELL}" height="${n * CELL}"/>`);
  if (o.visited) for (const c of o.visited) parts.push(cellRect(c, n, `b-visited is-${who}`));
  for (const t of o.targets ?? []) parts.push(cellRect(neighbor(t.cell, t.dir, n), n, 'b-target', 4));
  parts.push(`<path class="b-grid" d="${gridPath(n)}"/>`);

  if (o.solution?.length > 1) {
    const pts = o.solution.map((c) => `${cx(c, n)},${cy(c, n)}`).join(' ');
    parts.push(`<polyline class="b-solution" points="${pts}"/>`);
  }
  const passages = (o.passages ?? []).filter((k) => k !== o.fresh?.key);
  if (passages.length) parts.push(`<path class="b-passage is-${who}" d="${passages.map((k) => passagePath(k, n)).join('')}"/>`);
  if (o.fresh?.open) parts.push(`<path class="b-passage is-${who} is-fresh" d="${passagePath(o.fresh.key, n)}"/>`);

  if (o.ghostWalls?.length) {
    parts.push(`<path class="b-wall is-ghost" d="${o.ghostWalls.map((k) => wallPath(k, n)).join('')}"/>`);
  }
  const walls = (o.walls ?? []).filter((k) => k !== o.fresh?.key || o.fresh.open);
  if (walls.length) parts.push(`<path class="b-wall" d="${walls.map((k) => wallPath(k, n)).join('')}"/>`);
  if (o.fresh && !o.fresh.open) parts.push(`<path class="b-wall is-fresh" d="${wallPath(o.fresh.key, n)}"/>`);
  if (o.hits?.length) parts.push(`<path class="b-hit is-${who}" d="${o.hits.map((k) => hitMark(k, n)).join('')}"/>`);

  parts.push(`<rect class="b-border" x="${PAD}" y="${PAD}" width="${n * CELL}" height="${n * CELL}" rx="2"/>`);
  for (let i = 0; i < n; i++) {
    parts.push(`<text class="b-label" x="${PAD + (i + 0.5) * CELL}" y="${PAD - 9}" text-anchor="middle">${String.fromCharCode(65 + i)}</text>`);
    parts.push(`<text class="b-label" x="${PAD - 10}" y="${PAD + (i + 0.5) * CELL + 6}" text-anchor="middle">${i + 1}</text>`);
  }

  parts.push(mark(o.entrance, n, 'E', 'is-entrance'));
  parts.push(mark(o.exit, n, 'S', 'is-exit'));
  for (const t of o.targets ?? []) parts.push(chevron(t.cell, t.dir, n));
  if (o.token != null) {
    parts.push(`<circle class="b-token is-${who}" cx="${cx(o.token, n)}" cy="${cy(o.token, n)}" r="${CELL * 0.34}"/>`);
  }
  if (o.variant === 'design') parts.push('<path class="b-preview" d=""/>');

  return `<svg class="board board--${o.variant}" viewBox="0 0 ${V} ${V}" role="img" aria-label="${o.label ?? ''}">${parts.join('')}</svg>`;
}

/** Board coordinates (in viewBox units) of a pointer event. */
function toBoard(svg, e) {
  const rect = svg.getBoundingClientRect();
  const V = svg.viewBox.baseVal.width;
  return [((e.clientX - rect.left) / rect.width) * V, ((e.clientY - rect.top) / rect.height) * V];
}

export function cellAtPoint(svg, e, n) {
  const [x, y] = toBoard(svg, e);
  const c = Math.floor((x - PAD) / CELL);
  const r = Math.floor((y - PAD) / CELL);
  if (r < 0 || c < 0 || r >= n || c >= n) return null;
  return r * n + c;
}

/** Nearest internal edge to the pointer (forgiving on touch screens), or null. */
export function edgeAtPoint(svg, e, n, tolerance = 0.32) {
  const [x, y] = toBoard(svg, e);
  const px = (x - PAD) / CELL;
  const py = (y - PAD) / CELL;
  let best = null;
  let bestDist = tolerance;
  const vx = Math.round(px);
  const vr = Math.floor(py);
  if (vx >= 1 && vx <= n - 1 && vr >= 0 && vr < n && Math.abs(px - vx) < bestDist) {
    bestDist = Math.abs(px - vx);
    best = edgeKey(vr * n + vx - 1, vr * n + vx);
  }
  const hy = Math.round(py);
  const hc = Math.floor(px);
  if (hy >= 1 && hy <= n - 1 && hc >= 0 && hc < n && Math.abs(py - hy) < bestDist) {
    best = edgeKey((hy - 1) * n + hc, hy * n + hc);
  }
  return best;
}

export function previewPath(key, n) {
  return key ? wallPath(key, n) : '';
}

