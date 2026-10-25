"""The explorer's network (DECISIONS.md D17, D23): a fully convolutional residual tower over the
(2N+1)×(2N+1) lattice, the same weights for every board size.

Plain JAX, parameters in a dict: no framework, so a checkpoint is just named arrays, and the forward
pass is easy to rewrite for the browser (P4).

    trunk   3×3 stem, then residual blocks: conv → ReLU → + global-pooling bias → conv, dilated
            (1, 2, 4, ... cycling) so a 21×21 lattice (10×10 board) is covered. The global bias
            (mean and max over the board → dense → added per channel, as in KataGo) lets every
            pixel see board-wide counts such as walls left.
    heads   three 1×1 convolutions read at the side pixels:
            policy   logits of probing each side, masked to legal probes;
            belief   P(wall) of each side (trained on the true maze, privileged in simulation);
            hit      P(this side will be hit before the exit is reached). The value, the expected
                     walls still to hit, is their sum over the unknown sides: bounded, and it
                     scales with the board without any size-specific layer.
    plan    optional (NetConfig.plan, experimental): the policy logits also get −w · plan_scores,
            the expected walls on the cheapest way to the exit through each probe when unknown
            sides cost their belief P(wall): the expert's rollout planner (D21) as a prior, with
            the weight w learned and no gradient into the belief through it.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .engine import neighbor_grid, side_grid
from .geometry import geometry
from .lattice import EXIT, N_PLANES, OPEN, REACHED, UNKNOWN, WALL, action_pixels, cell_pixels

NEG = -1e9  # logit of an illegal probe
FAR = 1e6  # path cost through a known wall
# Convolutions run in this dtype, accumulating in float32; everything else stays float32. Set it
# once at startup, before anything is compiled (calibrate/league --precision): jnp.float16 uses the
# tensor cores of a T4.
COMPUTE_DTYPE = jnp.float32


class NetConfig(NamedTuple):
    channels: int = 48
    blocks: int = 6
    dilations: tuple = (1, 2, 4)  # cycled over the blocks
    plan: bool = False  # add the belief planner's scores to the policy logits (see plan_scores)


class Output(NamedTuple):
    logits: jax.Array  # float32[..., E] policy logits, NEG on illegal probes
    belief: jax.Array  # float32[..., E] logit of P(wall) per side
    value: jax.Array  # float32[...]    expected walls still to hit


def _conv_init(key, k: int, cin: int, cout: int, scale: float = 1.0) -> dict:
    std = scale * np.sqrt(2.0 / (k * k * cin))  # He
    return {"w": std * jax.random.normal(key, (k, k, cin, cout), jnp.float32), "b": jnp.zeros(cout, jnp.float32)}


def init(key, cfg: NetConfig = NetConfig(), in_planes: int = N_PLANES, head_out: int = 3) -> dict:
    """The explorer's parameters; the designer (designer.py) reuses the tower with its own input
    planes and a one-channel head."""
    keys = iter(jax.random.split(key, 3 + 3 * cfg.blocks))
    c = cfg.channels
    blocks = []
    for _ in range(cfg.blocks):
        blocks.append(
            {
                "c1": _conv_init(next(keys), 3, c, c),
                # small last conv: each block starts close to the identity (no normalization layers)
                "c2": _conv_init(next(keys), 3, c, c, scale=0.1),
                "g": {"w": jax.random.normal(next(keys), (2 * c, c), jnp.float32) * np.sqrt(1.0 / (2 * c)), "b": jnp.zeros(c, jnp.float32)},
            }
        )
    head = _conv_init(next(keys), 1, c, head_out, scale=0.1)
    params = {"stem": _conv_init(next(keys), 3, in_planes, c), "blocks": blocks, "head": head}
    if cfg.plan:
        params["plan"] = jnp.float32(2.95)  # softplus → a weight of 3 per expected wall
    return params


def _conv(x, p, dilation: int = 1):
    y = jax.lax.conv_general_dilated(
        x.astype(COMPUTE_DTYPE), p["w"].astype(COMPUTE_DTYPE), window_strides=(1, 1), padding="SAME",
        rhs_dilation=(dilation, dilation), dimension_numbers=("NHWC", "HWIO", "NHWC"),
        preferred_element_type=jnp.float32,
    )
    return y + p["b"]


def trunk(params: dict, obs, dilations=NetConfig().dilations) -> jax.Array:
    """obs float32[B, L, L, N_PLANES] → features float32[B, L, L, C]."""
    h = jax.nn.relu(_conv(obs, params["stem"]))
    for i, blk in enumerate(params["blocks"]):
        d = dilations[i % len(dilations)]
        y = jax.nn.relu(_conv(h, blk["c1"], d))
        pooled = jnp.concatenate([jnp.mean(y, axis=(1, 2)), jnp.max(y, axis=(1, 2))], axis=-1)
        y = y + (pooled @ blk["g"]["w"] + blk["g"]["b"])[:, None, None, :]
        h = jax.nn.relu(h + _conv(y, blk["c2"], d))
    return h


def _costs_to_go(cost, exit_cell, size: int):
    """One game: fewest expected walls from every cell to the exit, crossing side e at cost[e]."""
    side_cost = side_grid(cost, FAR)  # [4, N, N]
    start = jnp.where(exit_cell, 0.0, FAR).reshape(size, size)

    def relax(c):
        d, _ = c
        new = jnp.minimum(d, jnp.min(side_cost + neighbor_grid(d, FAR), axis=0))
        return new, jnp.any(new < d - 1e-6)

    d, _ = jax.lax.while_loop(lambda c: c[1], relax, (start, jnp.bool_(True)))
    return d.reshape(-1)


def plan_scores(obs, p_wall) -> jax.Array:
    """float32[B, E]: for each probe, the expected walls on the cheapest way to the exit that starts
    with it, when an unknown side is a wall with probability p_wall (the belief) and the rest of the
    way is planned on the same costs. The greedy planner of the expert's rollouts (D21) probes the
    minimum; known walls cost FAR, known passages 0."""
    size = (obs.shape[1] - 1) // 2
    g = geometry(size)
    px, cx = action_pixels(size), cell_pixels(size)
    sides = obs[:, px[:, 0], px[:, 1], :]
    cells = obs[:, cx[:, 0], cx[:, 1], :]
    cost = jnp.where(sides[..., OPEN] > 0, 0.0, jnp.where(sides[..., WALL] > 0, FAR, p_wall))
    d = jax.vmap(_costs_to_go, in_axes=(0, 0, None))(cost, cells[..., EXIT] > 0, size)  # [B, N²]
    a, b = g.edge_cells[:, 0], g.edge_cells[:, 1]
    reached = cells[..., REACHED] > 0
    beyond = jnp.where(reached[:, a], d[:, b], d[:, a])  # the far side of the probe
    return cost + beyond


def apply(params: dict, obs, legal, dilations=NetConfig().dilations) -> Output:
    """Forward pass on a batch: obs float32[B, L, L, N_PLANES], legal bool[B, E]."""
    size = (obs.shape[1] - 1) // 2
    px = action_pixels(size)
    maps = _conv(trunk(params, obs, dilations), params["head"])  # [B, L, L, 3]
    sides = maps[:, px[:, 0], px[:, 1], :]  # [B, E, 3]
    unknown = obs[:, px[:, 0], px[:, 1], UNKNOWN]  # [B, E]
    value = jnp.sum(unknown * jax.nn.sigmoid(sides[..., 2]), axis=-1)
    logits = sides[..., 0]
    if "plan" in params:
        q = plan_scores(obs, jax.lax.stop_gradient(jax.nn.sigmoid(sides[..., 1])))
        logits = logits - jax.nn.softplus(params["plan"]) * jnp.minimum(q, 100.0)
    return Output(logits=jnp.where(legal, logits, NEG), belief=sides[..., 1], value=value)


def n_params(params: dict) -> int:
    return sum(int(np.prod(x.shape)) for x in jax.tree.leaves(params))
