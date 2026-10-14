"""Throughput of cppanatron.VecEnv: learner decisions per second.

    uv run python tools/bench_vec_env.py [--envs 512] [--steps 300] [--threads 1 8]

Actions are uniformly random legal ones, sampled in NumPy from the mask block
(timed separately). "step" counts only VecEnv.step(), which applies the
actions, plays forced moves and bot seats, and encodes the acting rows.
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

import cppanatron as cpp

RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
NUMBERS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
DIRECTIONS = [(1, -1, 0), (0, -1, 1), (-1, 0, 1), (-1, 1, 0), (0, 1, -1), (1, 0, -1)]


def hexagon_spec(radius: int, seed: int) -> dict:
    """A random hexagonal island of the given radius with a water ring."""
    rng = random.Random(seed)
    coordinates = [
        (x, y, -x - y)
        for x in range(-radius - 1, radius + 2)
        for y in range(-radius - 1, radius + 2)
        if abs(-x - y) <= radius + 1
    ]
    land = [c for c in coordinates if max(map(abs, c)) <= radius]
    desert = rng.randrange(len(land))
    tiles = [
        {"coordinate": list(c), "type": "land"}
        if i == desert
        else {"coordinate": list(c), "type": "land", "resource": rng.choice(RESOURCES),
              "number": rng.choice(NUMBERS)}
        for i, c in enumerate(land)
    ]
    tiles += [{"coordinate": list(c), "type": "water"} for c in coordinates if c not in land]
    return {"tiles": tiles}


def scenarios(name: str) -> list:
    if name == "2p BASE self-play":
        return [cpp.Scenario(num_players=2)]
    if name == "2p BASE vs F":
        return [cpp.Scenario(num_players=2, seats=["learner", "F"], shuffle_seats=True)]
    maps = [cpp.CatanMap.from_spec(hexagon_spec(radius, seed))
            for radius in (2, 3) for seed in range(4)]
    if name == "4p mixed maps self-play":
        return [cpp.Scenario(num_players=4, map_type=cpp.MapType.BASE),
                cpp.Scenario(num_players=4, maps=maps)]
    if name == "4p mixed maps vs R":
        seats = ["learner", "R", "R", "R"]
        return [cpp.Scenario(num_players=4, map_type=cpp.MapType.BASE, seats=seats,
                             shuffle_seats=True),
                cpp.Scenario(num_players=4, maps=maps, seats=seats, shuffle_seats=True)]
    raise ValueError(name)


def bench(name: str, num_envs: int, steps: int, threads: int) -> None:
    env = cpp.VecEnv(scenarios(name), num_envs=num_envs, seed=0, num_threads=threads)
    buffers = env.allocate()
    env.reset()
    offset, (size,) = env.encoder.layout()["action_mask"]
    rng = np.random.default_rng(0)
    step_time = sample_time = 0.0
    for _ in range(steps):
        start = time.perf_counter()
        mask = buffers["observations"][:, offset: offset + size]
        buffers["actions"][:] = np.argmax(np.where(mask > 0, rng.random(mask.shape), -1.0), 1)
        middle = time.perf_counter()
        env.step()
        sample_time += middle - start
        step_time += time.perf_counter() - middle
    decisions = num_envs * steps
    stats = env.pop_stats()
    print(
        f"{name:26s} threads={threads:2d} envs={num_envs} rows={env.num_rows} "
        f"obs={env.observation_size}: {decisions / step_time:9.0f} decisions/s in step "
        f"({decisions / (step_time + sample_time):8.0f} with sampling), "
        f"{stats['episodes']} games, {stats['length']:.0f} turns"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument(
        "--only", nargs="*",
        default=["2p BASE self-play", "2p BASE vs F", "4p mixed maps self-play",
                 "4p mixed maps vs R"],
    )
    args = parser.parse_args()
    for name in args.only:
        for threads in args.threads:
            bench(name, args.envs, args.steps, threads)


if __name__ == "__main__":
    main()
