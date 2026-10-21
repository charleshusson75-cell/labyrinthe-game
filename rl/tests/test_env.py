import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from labyrinthe_rl import engine, env as env_module  # noqa: F401 (registers the environment)


def test_gymnasium_api():
    env = gym.make("Labyrinthe-Explorer-v0", size=4)
    check_env(env.unwrapped, skip_render_check=True)


def test_episode():
    env = env_module.ExplorerEnv(size=5, sampler="random")
    obs, info = env.reset(seed=3)
    assert obs.shape == (11, 11, 10)
    total, done, steps = 0.0, False, 0
    rng = np.random.default_rng(0)
    while not done:
        action = int(rng.choice(np.flatnonzero(info["action_mask"])))
        obs, reward, done, truncated, info = env.step(action)
        total += reward
        steps += 1
        assert not truncated
    assert -total == info["hits"] and steps <= 40
    _, _, _, _, info = env.step(0)
    assert info["result"] == engine.ILLEGAL
    text = env.render()
    assert text.count("E") == 1 and text.count("S") == 1 and len(text.splitlines()) == 11
