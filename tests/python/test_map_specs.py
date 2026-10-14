"""Explicit maps: MapSpec round trips, shared maps, and identical ids in both engines."""

from __future__ import annotations

import gc
import json

import pytest

import cppanatron as cpp
from map_shapes import SHAPES

TEMPLATES = ["BASE", "MINI", "TOURNAMENT"]


def tile_keys(catan_map: cpp.CatanMap) -> dict:
    return {
        coordinate: (
            tile.kind, tile.id, tile.resource, tile.number, tile.port_direction, tile.nodes,
            tile.edges,
        )
        for coordinate, tile in catan_map.tiles.items()
    }


@pytest.mark.parametrize("map_type", TEMPLATES)
@pytest.mark.parametrize("seed", [0, 7])
def test_template_maps_round_trip_through_their_spec(map_type, seed):
    built = cpp.CatanMap.build(cpp.MapType[map_type], seed)
    spec = built.spec()
    assert spec == cpp.MapSpec.build(cpp.MapType[map_type], seed)
    as_json = json.loads(json.dumps(spec.to_dict()))
    assert cpp.MapSpec.from_dict(as_json) == spec
    rebuilt = cpp.CatanMap.from_spec(as_json)
    assert tile_keys(rebuilt) == tile_keys(built)
    assert rebuilt.land_edges == built.land_edges
    # The same board as the MapType constructor, which keeps its old draws.
    game = cpp.Game(2, map_type=cpp.MapType[map_type], seed=seed)
    assert tile_keys(game.board.map) == tile_keys(built)


def test_spec_dicts_have_exactly_the_documented_shape():
    tiles = cpp.MapSpec.build(cpp.MapType.BASE, 3).to_dict()["tiles"]
    keys = {"coordinate", "type", "resource", "number", "direction", "port_resource"}
    assert all(set(tile) == keys for tile in tiles)
    assert {tile["type"] for tile in tiles} == {"land", "water", "port"}
    land = [tile for tile in tiles if tile["type"] == "land"]
    assert sum(tile["resource"] is None for tile in land) == 1
    ports = [tile for tile in tiles if tile["type"] == "port"]
    assert sum(tile["port_resource"] is None for tile in ports) == 4
    assert all(tile["direction"] in cpp.Direction.__members__ for tile in ports)
    assert isinstance(tiles[0]["coordinate"], list)


@pytest.mark.parametrize(
    ("tile", "message"),
    [
        ({"coordinate": [0, 0, 1], "type": "land"}, "sum to zero"),
        # 2**31 - 1 twice plus 2 wraps to zero in 32-bit arithmetic.
        ({"coordinate": [2**31 - 1, 2**31 - 1, 2], "type": "water"}, "within"),
        ({"coordinate": [2**20 + 1, -(2**20) - 1, 0], "type": "water"}, "within"),
        ({"coordinate": [0, 0, 0], "type": "land", "resource": "WOOD"}, "number"),
        ({"coordinate": [0, 0, 0], "type": "land", "resource": "WOOD", "number": 7}, "number"),
        ({"coordinate": [0, 0, 0], "type": "land", "number": 8}, "desert"),
        ({"coordinate": [0, 0, 0], "type": "port"}, "direction"),
        ({"coordinate": [0, 0, 0], "type": "water", "resource": "ORE"}, "water"),
        ({"coordinate": [0, 0, 0], "type": "lava"}, "land, water or port"),
        ({"coordinate": [0, 0, 0], "type": "land", "resource": "GOLD", "number": 8}, "GOLD"),
        ({"coordinate": [0, 0, 0], "type": "land", "colour": "red"}, "colour"),
    ],
)
def test_invalid_specs_are_refused(tile, message):
    desert = {"coordinate": [1, -1, 0], "type": "land"}
    with pytest.raises(ValueError, match=message):
        cpp.CatanMap.from_spec({"tiles": [tile, desert]})


def test_maps_need_land_and_a_desert_and_unique_coordinates():
    with pytest.raises(ValueError, match="desert"):
        cpp.CatanMap.from_spec(
            {"tiles": [{"coordinate": [0, 0, 0], "type": "land", "resource": "ORE", "number": 8}]}
        )
    with pytest.raises(ValueError, match="land"):
        cpp.CatanMap.from_spec({"tiles": [{"coordinate": [0, 0, 0], "type": "water"}]})
    with pytest.raises(ValueError, match="twice"):
        cpp.CatanMap.from_spec(
            {"tiles": [{"coordinate": [0, 0, 0], "type": "land"}] * 2}
        )


def test_games_share_one_immutable_map():
    shared = cpp.CatanMap.from_spec(SHAPES["strip"][0](3))
    first = cpp.Game(3, map=shared, seed=1)
    second = cpp.Game(4, map=shared, seed=2, rules=cpp.GameRules(vps_to_win=8))
    assert first.board.map is shared and second.board.map is shared
    assert first.copy().board.map is shared
    del shared
    gc.collect()
    # The games keep the map alive on the C++ side.
    assert first.board.map.num_land_tiles == 16
    with pytest.raises(ValueError, match="either map or map_type"):
        cpp.Game(2, map=first.board.map, map_type=cpp.MapType.BASE)


def test_the_robber_starts_on_the_first_desert_in_tile_order():
    spec = SHAPES["big_hexagon"][0](5)
    deserts = [
        tuple(tile["coordinate"])
        for tile in spec["tiles"]
        if tile["type"] == "land" and tile.get("resource") is None
    ]
    assert len(deserts) == 2
    game = cpp.Game(2, map=cpp.CatanMap.from_spec(spec))
    assert game.board.robber_coordinate == deserts[0]


def test_slots_index_land_nodes_edges_and_tiles():
    catan_map = cpp.CatanMap.from_spec(SHAPES["islands"][0](2))
    assert catan_map.node_ids == sorted(catan_map.land_nodes)
    assert len(catan_map.land_tile_coordinates) == catan_map.num_land_tiles == 14
    assert list(catan_map.land_tile_coordinates) == list(catan_map.land_tiles)
    for node in catan_map.node_ids:
        for neighbor in catan_map.neighbors(node):
            assert tuple(sorted((node, neighbor))) in catan_map.land_edges


# ---------------------------------------------------------------- Catanatron


catanatron = pytest.importorskip("catanatron")

from catanatron.models.map import build_map  # noqa: E402
from catanatron.models.tiles import LandTile, Port, Water  # noqa: E402

from cppanatron.catanatron import (  # noqa: E402
    catanatron_map_from_spec,
    map_spec_from_catanatron,
)


def land_graph(reference) -> dict[int, set[int]]:
    """Node adjacency along the edges of a Catanatron map's land tiles."""
    graph: dict[int, set[int]] = {}
    for tile in reference.land_tiles.values():
        for a, b in tile.edges.values():
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
    return graph


def assert_same_map(native: cpp.CatanMap, reference) -> None:
    assert list(native.tiles) == list(reference.tiles)
    for coordinate, expected in reference.tiles.items():
        tile = native.tiles[coordinate]
        assert tile.nodes == tuple(expected.nodes.values())
        assert tile.edges == tuple(tuple(sorted(e)) for e in expected.edges.values())
        if isinstance(expected, LandTile):
            assert tile.kind == cpp.TileKind.LAND and tile.id == expected.id
            assert (None if tile.resource is None else tile.resource.name) == expected.resource
            assert tile.number == expected.number
        elif isinstance(expected, Port):
            assert tile.kind == cpp.TileKind.PORT and tile.id == expected.id
            assert (None if tile.resource is None else tile.resource.name) == expected.resource
            assert tile.port_direction.name == expected.direction.name
        else:
            assert isinstance(expected, Water) and tile.kind == cpp.TileKind.WATER
    assert native.land_nodes == reference.land_nodes
    # Catanatron's own node graph, and the land-tile edges it is built from.
    assert sorted(native.land_edges) == sorted(tuple(sorted(e)) for e in reference.graph.edges)
    graph = land_graph(reference)
    assert set(native.land_edges) == {
        (a, b) for a, neighbors in graph.items() for b in neighbors if a < b
    }
    for node in native.node_ids:
        assert native.neighbors(node) == sorted(graph[node])


@pytest.mark.parametrize("map_type", TEMPLATES)
def test_catanatron_template_maps_convert_both_ways(map_type):
    reference = build_map(map_type)
    spec = map_spec_from_catanatron(reference)
    assert_same_map(cpp.CatanMap.from_spec(spec), reference)
    assert_same_map(cpp.CatanMap.from_spec(spec), catanatron_map_from_spec(spec))


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("seed", [0, 1])
def test_explicit_shapes_build_identical_maps_in_both_engines(shape, seed):
    spec = SHAPES[shape][0](seed)
    native = cpp.CatanMap.from_spec(spec)
    reference = catanatron_map_from_spec(spec)
    assert_same_map(native, reference)
    assert map_spec_from_catanatron(reference) == native.spec()
    assert {
        None if resource is None else resource.name: nodes
        for resource, nodes in native.port_nodes.items()
    } == {resource: frozenset(nodes) for resource, nodes in reference.port_nodes.items()}


def test_lake_edges_are_not_land_edges():
    """Two land nodes facing each other across a lake are not neighbors."""
    native = cpp.CatanMap.from_spec(SHAPES["lake"][0](0))
    reference = catanatron_map_from_spec(SHAPES["lake"][0](0))
    lake = {(0, 0, 0), (1, -1, 0)}
    shared = set(reference.tiles[(0, 0, 0)].edges.values()) & set(
        reference.tiles[(1, -1, 0)].edges.values()
    )
    (edge,) = shared
    a, b = edge
    assert a in native.land_nodes and b in native.land_nodes
    assert b not in native.neighbors(a)
    assert lake <= set(reference.tiles)
