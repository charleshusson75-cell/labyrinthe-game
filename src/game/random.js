// Small randomness helpers. Every function takes an `rng` returning floats in [0, 1),
// so tests can pass a seeded generator and get reproducible results.

/** Deterministic PRNG for tests and simulations. */
export function seededRng(seed) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

export const randInt = (rng, n) => Math.floor(rng() * n);

export const pick = (rng, list) => list[randInt(rng, list.length)];

export function shuffle(rng, list) {
  const a = [...list];
  for (let i = a.length - 1; i > 0; i--) {
    const j = randInt(rng, i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const CODE_ALPHABET = '23456789abcdefghjkmnpqrstuvwxyz'; // no 0/o, 1/l/i

export function randomCode(rng, length = 6) {
  let s = '';
  for (let i = 0; i < length; i++) s += CODE_ALPHABET[randInt(rng, CODE_ALPHABET.length)];
  return s;
}

export function randomHex(rng, bytes = 16) {
  let s = '';
  for (let i = 0; i < bytes; i++) s += randInt(rng, 256).toString(16).padStart(2, '0');
  return s;
}

/** Cryptographically strong source when available (salts must not be guessable). */
export function secureRng() {
  const c = globalThis.crypto;
  if (!c?.getRandomValues) return Math.random;
  const buf = new Uint32Array(1);
  return () => {
    c.getRandomValues(buf);
    return buf[0] / 4294967296;
  };
}
