"""Native boards follow Catanatron's topology, templates and number placement."""

from __future__ import annotations

import random
from collections import Counter

import pytest

pytest.importorskip("catanatron")

from catanatron.models.coordinate_system import Direction  # noqa: E402
from catanatron.models.map import (  # noqa: E402
    BASE_NUMBERS_IN_SPIRAL_ORDER,
    MINI_MAP_TEMPLATE,
    build_map,
)
from catanatron.models.spiral import spiral_land_coordinates  # noqa: E402
from catanatron.models.tiles import LandTile, Port, Water  # noqa: E402

import cppanatron as cpp  # noqa: E402
from cppanatron.catanatron import to_catanatron_resource  # noqa: E402


def tile_keys(game: cpp.Game) -> dict:
    return {
        coordinate: (tile.kind, tile.id, tile.resource, tile.number, tile.port_direction, tile.nodes)
        for coordinate, tile in game.board.map.tiles.items()
    }


def base_game(**options) -> cpp.Game:
    return cpp.Game(2, map_type=cpp.MapType.BASE, **options)


def test_maps_and_games_draw_from_independent_seeds():
    first = base_game(seed=11, map_seed=101)
    assert tile_keys(first) == tile_keys(base_game(seed=22, map_seed=101))
    assert tile_keys(first) != tile_keys(base_game(seed=11, map_seed=202))
    assert tile_keys(base_game(seed=5)) == tile_keys(base_game(seed=6, map_seed=5))


def test_number_placement_changes_only_the_numbers():
    official = base_game(seed=29, map_seed=17, number_placement=cpp.NumberPlacement.OFFICIAL_SPIRAL)
    randomized = base_game(seed=29, map_seed=17, number_placement=cpp.NumberPlacement.RANDOM)
    official_tiles = official.board.map.land_tiles
    random_tiles = randomized.board.map.land_tiles
    assert {c: t.resource for c, t in official_tiles.items()} == {
        c: t.resource for c, t in random_tiles.items()
    }
    assert {c: t.number for c, t in official_tiles.items()} != {
        c: t.number for c, t in random_tiles.items()
    }


@pytest.mark.parametrize("map_type", ["MINI", "BASE"])
@pytest.mark.parametrize("seed", [0, 1, 42, 1234])
def test_official_spiral_numbers_follow_catanatrons_spiral(map_type, seed):
    """Each engine shuffles terrain with its own RNG, but official-spiral numbers
    must follow Catanatron's spiral walk over whatever terrain was drawn."""
    game = cpp.Game(2, map_type=cpp.MapType[map_type], seed=seed)
    tiles = game.board.map.tiles
    start = (2, -2, 0) if map_type == "BASE" else (1, -1, 0)
    expected = iter(BASE_NUMBERS_IN_SPIRAL_ORDER)
    for coordinate in spiral_land_coordinates(build_map(map_type).tiles, start):
        tile = tiles[coordinate]
        assert tile.number == (None if tile.resource is None else next(expected))


@pytest.mark.parametrize("map_type", ["MINI", "BASE"])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_random_boards_match_catanatrons_template(map_type, seed):
    random.seed(seed)
    reference = build_map(map_type, number_placement="random")
    game = cpp.Game(
        2, map_type=cpp.MapType[map_type], seed=seed, number_placement=cpp.NumberPlacement.RANDOM
    )
    tiles = game.board.map.tiles
    assert set(tiles) == set(reference.tiles)

    for coordinate, expected in reference.tiles.items():
        tile = tiles[coordinate]
        assert tile.nodes == tuple(expected.nodes.values())
        if isinstance(expected, LandTile):
            assert (tile.kind, tile.id, tile.port_direction) == (cpp.TileKind.LAND, expected.id, None)
        elif isinstance(expected, Water):
            assert (tile.kind, tile.id, tile.port_direction) == (cpp.TileKind.WATER, -1, None)
        elif isinstance(expected, Port):
            assert tile.kind == cpp.TileKind.PORT and tile.id == expected.id
            assert tile.port_direction.name == Direction(expected.direction).name
        else:  # pragma: no cover - guards an upstream tile-type addition
            raise AssertionError(f"unknown Catanatron tile type {type(expected)}")

    def counts(kind, attribute, source):
        return Counter(
            getattr(tile, attribute) for tile in source.values() if isinstance(tile, kind)
        )

    land = {c: t for c, t in tiles.items() if t.kind == cpp.TileKind.LAND}
    ports = {c: t for c, t in tiles.items() if t.kind == cpp.TileKind.PORT}
    assert Counter(to_catanatron_resource(t.resource) for t in land.values()) == counts(
        LandTile, "resource", reference.tiles
    )
    assert Counter(to_catanatron_resource(t.resource) for t in ports.values()) == counts(
        Port, "resource", reference.tiles
    )
    numbers = Counter(t.number for t in land.values())
    if map_type == "MINI":
        # MINI draws its numbers from the template without using them all.
        assert numbers[None] == 1
        assert set(numbers) - {None} <= set(MINI_MAP_TEMPLATE.numbers)
    else:
        assert numbers == counts(LandTile, "number", reference.tiles)
