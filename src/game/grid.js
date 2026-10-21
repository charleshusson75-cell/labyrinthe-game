// Grid geometry. A cell is an integer index: row * size + col (row 0 = top).
// An edge (one side shared by two adjacent cells) is identified by the string "a-b" with a < b.

export const DIRS = ['U', 'R', 'D', 'L'];
export const DELTA = { U: [-1, 0], R: [0, 1], D: [1, 0], L: [0, -1] };
export const ARROW = { U: '↑', R: '→', D: '↓', L: '←' };

export const rowOf = (cell, size) => Math.floor(cell / size);
export const colOf = (cell, size) => cell % size;

export function isCell(cell, size) {
  return Number.isInteger(cell) && cell >= 0 && cell < size * size;
}

export function isDir(dir) {
  return DIRS.includes(dir);
}

/** Cell reached by stepping from `cell` in `dir`, or -1 when that leaves the grid. */
export function neighbor(cell, dir, size) {
  const [dr, dc] = DELTA[dir];
  const r = rowOf(cell, size) + dr;
  const c = colOf(cell, size) + dc;
  if (r < 0 || c < 0 || r >= size || c >= size) return -1;
  return r * size + c;
}

export function edgeKey(a, b) {
  return a < b ? `${a}-${b}` : `${b}-${a}`;
}

export function edgeCells(key) {
  const [a, b] = key.split('-').map(Number);
  return [a, b];
}

/** Edge on the `dir` side of `cell`, or null on the outer border. */
export function edgeOf(cell, dir, size) {
  const n = neighbor(cell, dir, size);
  return n < 0 ? null : edgeKey(cell, n);
}

/** Direction from cell a to adjacent cell b (null if not adjacent). */
export function dirBetween(a, b, size) {
  return DIRS.find((d) => neighbor(a, d, size) === b) ?? null;
}

export function isEdge(key, size) {
  if (typeof key !== 'string' || !/^\d+-\d+$/.test(key)) return false;
  const [a, b] = edgeCells(key);
  if (!isCell(a, size) || !isCell(b, size) || a >= b) return false;
  return dirBetween(a, b, size) !== null;
}

/** Every internal edge of a size×size grid: 2·size·(size−1) of them. */
export function allEdges(size) {
  const out = [];
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const cell = r * size + c;
      if (c < size - 1) out.push(edgeKey(cell, cell + 1));
      if (r < size - 1) out.push(edgeKey(cell, cell + size));
    }
  }
  return out;
}

/** Human coordinates like a chessboard: column letter + row number, "A1" = top-left. */
export function cellLabel(cell, size) {
  return String.fromCharCode(65 + colOf(cell, size)) + (rowOf(cell, size) + 1);
}
