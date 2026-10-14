"""Explicit maps for tests: shapes no template produces.

Each builder returns a MapSpec dict (the JSON shape). The hand-made ones draw
terrain and numbers from ``seed`` and put a water ring around the land with
ports on some water tiles, each facing a land neighbor. The fixtures in
``maps/`` were generated once by catanrl's ``catanrl.maps.generate(generator,
seed)`` (both recorded in each file) and ignore ``seed``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
NUMBERS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
#: Catanatron's UNIT_VECTORS, in cppanatron's Direction order.
DIRECTIONS = {
    "EAST": (1, -1, 0),
    "SOUTHEAST": (0, -1, 1),
    "SOUTHWEST": (-1, 0, 1),
    "WEST": (-1, 1, 0),
    "NORTHWEST": (0, 1, -1),
    "NORTHEAST": (1, 0, -1),
}


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def hexagon(radius: int, center=(0, 0, 0)) -> list[tuple[int, int, int]]:
    """Every coordinate within ``radius`` of ``center``, ring by ring."""
    result = [center]
    for ring in range(1, radius + 1):
        coordinate = add(center, tuple(ring * v for v in DIRECTIONS["NORTHEAST"]))
        for direction in ("SOUTHEAST", "SOUTHWEST", "WEST", "NORTHWEST", "NORTHEAST", "EAST"):
            for _ in range(ring):
                result.append(coordinate)
                coordinate = add(coordinate, DIRECTIONS[direction])
    return result


def build_spec(
    land: list,
    seed: int,
    *,
    deserts: int = 1,
    water: list | None = None,
    port_every: int = 3,
) -> dict:
    """Land in the given order, then ``water`` (default: the ring around the land)."""
    rng = random.Random(seed)
    land_set = set(land)
    if water is None:
        water = []
        for coordinate in land:
            for vector in DIRECTIONS.values():
                neighbor = add(coordinate, vector)
                if neighbor not in land_set and neighbor not in water:
                    water.append(neighbor)
    desert_indices = set(rng.sample(range(len(land)), deserts))
    tiles = []
    for index, coordinate in enumerate(land):
        if index in desert_indices:
            tiles.append({"coordinate": list(coordinate), "type": "land"})
        else:
            tiles.append(
                {
                    "coordinate": list(coordinate),
                    "type": "land",
                    "resource": rng.choice(RESOURCES),
                    "number": rng.choice(NUMBERS),
                }
            )
    for index, coordinate in enumerate(water):
        facing = [d for d, v in DIRECTIONS.items() if add(coordinate, v) in land_set]
        if facing and port_every and index % port_every == 0:
            tiles.append(
                {
                    "coordinate": list(coordinate),
                    "type": "port",
                    "direction": rng.choice(facing),
                    "port_resource": rng.choice(RESOURCES + [None, None]),
                }
            )
        else:
            tiles.append({"coordinate": list(coordinate), "type": "water"})
    return {"tiles": tiles}


def big_hexagon(seed: int) -> dict:
    """A 3-ring hexagon: 37 land tiles, two deserts."""
    return build_spec(hexagon(3), seed, deserts=2)


def strip(seed: int, length: int = 8) -> dict:
    """Two long rows of land, ports all along the coast."""
    land = [
        tuple(c * e + r * s for e, s in zip(DIRECTIONS["EAST"], DIRECTIONS["SOUTHEAST"]))
        for r in range(2)
        for c in range(length)
    ]
    return build_spec(land, seed, port_every=2)


def lake(seed: int) -> dict:
    """A 2-ring hexagon with a two-tile lake: land nodes facing across water."""
    lake_tiles = [(0, 0, 0), (1, -1, 0)]
    land = [c for c in hexagon(2) if c not in lake_tiles]
    coast = [c for c in hexagon(3) if c not in hexagon(2)]
    return build_spec(land, seed, water=coast + lake_tiles)


def islands(seed: int) -> dict:
    """Two islands of seven tiles, water listed first, land interleaved."""
    first, second = hexagon(1), hexagon(1, center=(4, -2, -2))
    land = [tile for pair in zip(first, second) for tile in pair]
    spec = build_spec(land, seed)
    water = [t for t in spec["tiles"] if t["type"] != "land"]
    return {"tiles": water + [t for t in spec["tiles"] if t["type"] == "land"]}


def bare(seed: int) -> dict:
    """Seven land tiles and no water at all."""
    return build_spec(hexagon(1), seed, water=[])


def generated(name: str) -> dict:
    """A fixture map from ``maps/<name>.json``."""
    return json.loads((Path(__file__).parent / "maps" / f"{name}.json").read_text())["spec"]


#: name -> (builder, most players the board seats comfortably)
SHAPES = {
    "big_hexagon": (big_hexagon, 6),
    "strip": (strip, 6),
    "lake": (lake, 6),
    "islands": (islands, 4),
    "bare": (bare, 4),
    # catanrl.maps: a radius-3 hexagon, the 30-tile 5-6 player board, an
    # offset rectangle and a random island.
    **{
        name: (lambda seed, name=name: generated(name), 6)
        for name in ("hexagon3", "extension", "rectangle", "blob")
    },
}
