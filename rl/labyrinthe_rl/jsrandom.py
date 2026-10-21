"""Bit-exact port of src/game/random.js, so seeded runs match the JS bots draw for draw.

Every helper takes `rng`, a callable returning floats in [0, 1): a `SeededRng`, or e.g. the
`.random` method of a `numpy.random.Generator` when JS parity doesn't matter.
"""

import math

_M32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return (a * b) & _M32


class SeededRng:
    """seededRng(seed) from random.js (a mulberry32 variant), computed on uint32 bit patterns."""

    def __init__(self, seed: int):
        self.t = seed & _M32

    def __call__(self) -> float:
        t = self.t = (self.t + 0x6D2B79F5) & _M32
        x = _imul(t ^ (t >> 15), 1 | t)
        x = ((x + _imul(x ^ (x >> 7), 61 | x)) & _M32) ^ x
        return ((x ^ (x >> 14)) & _M32) / 4294967296


def rand_int(rng, n: int) -> int:
    return math.floor(rng() * n)


def pick(rng, items):
    return items[rand_int(rng, len(items))]


def shuffle(rng, items) -> list:
    a = list(items)
    for i in range(len(a) - 1, 0, -1):
        j = rand_int(rng, i + 1)
        a[i], a[j] = a[j], a[i]
    return a
