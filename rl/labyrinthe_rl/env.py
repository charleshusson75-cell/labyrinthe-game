"""A Gymnasium wrapper around one exploration, for compatibility with standard RL tooling (D18).

Training uses the batched JAX functions directly (engine.py, lattice.py); this wrapper runs one game
at a time through the same compiled functions and returns numpy arrays.

    env = gymnasium.make("Labyrinthe-Explorer-v0", size=5)
    obs, info = env.reset(seed=0)          # obs: float32[2N+1, 2N+1, N_PLANES]
    obs, reward, terminated, truncated, info = env.step(action)   # action: an edge

Reward −1 per wall hit, 0 otherwise; the episode ends when the exit is reached. `info["action_mask"]`
holds the legal probes. An illegal action changes nothing (`info["result"]` is ILLEGAL), and the
episode is truncated after 4·E steps.
"""

import gymnasium as gym
import jax
import numpy as np

from . import engine, lattice
from .geometry import default_wall_budget, geometry
from .mazes import random_maze, uniform_maze

_step = jax.jit(engine.step)
_observe = jax.jit(lattice.observe)
_legal = jax.jit(engine.legal_mask)


class ExplorerEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size: int = 5, walls: int | None = None, sampler: str = "uniform", render_mode=None):
        self.size = size
        self.walls = walls or default_wall_budget(size)
        if sampler not in ("uniform", "random"):
            raise ValueError(f"unknown sampler {sampler!r} (uniform, random)")
        self.sampler = sampler
        self.render_mode = render_mode
        g = geometry(size)
        n = lattice.lattice_size(size)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (n, n, lattice.N_PLANES), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(g.n_edges)
        self.max_steps = 4 * g.n_edges
        self.state = None

    def _info(self, result=None):
        info = {"action_mask": np.asarray(_legal(self.state)).astype(np.int8), "hits": int(self.state.hits)}
        if result is not None:
            info["result"] = result
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.sampler == "uniform":
            maze = uniform_maze(self.size, self.walls, self.np_random)
        else:
            maze = random_maze(self.size, self.walls, self.np_random.random)
        self.state = engine.init(maze.walls, maze.entrance, maze.exit)
        self.steps = 0
        return np.asarray(_observe(self.state)), self._info()

    def step(self, action):
        self.state, result = _step(self.state, int(action))
        self.steps += 1
        result = int(result)
        reward = -1.0 if result == engine.WALL else 0.0
        terminated = bool(self.state.done)
        truncated = not terminated and self.steps >= self.max_steps
        return np.asarray(_observe(self.state)), reward, terminated, truncated, self._info(result)

    def render(self):
        return render_ascii(self.state)


def render_ascii(state: engine.State, show_hidden: bool = False) -> str:
    """The lattice as text: '#' known wall, '?' unknown side, '.' open, '*' reached, E and S.
    With `show_hidden`, unknown walls are drawn as 'x'."""
    size = engine.board_size(state)
    n = lattice.lattice_size(size)
    grid = [[" "] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i % 2 == 0 and j % 2 == 0:
                grid[i][j] = "+"
    border = lattice.border_mask(size)
    for i, j in zip(*np.nonzero(border)):
        grid[i][j] = "#"
    known_open, known_wall, walls = (np.asarray(x) for x in (state.known_open, state.known_wall, state.walls))
    for e, (i, j) in enumerate(lattice.action_pixels(size)):
        grid[i][j] = "." if known_open[e] else "#" if known_wall[e] else ("x" if show_hidden and walls[e] else "?")
    reached = np.asarray(state.reached)
    for c, (i, j) in enumerate(lattice.cell_pixels(size)):
        grid[i][j] = "E" if c == int(state.entrance) else "S" if c == int(state.exit) else "*" if reached[c] else " "
    return "\n".join("".join(row) for row in grid)


gym.register(id="Labyrinthe-Explorer-v0", entry_point=ExplorerEnv)
