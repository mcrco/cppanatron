"""Conversions between cppanatron and Catanatron. Requires ``catanatron``.

Enum members share Catanatron's names, so colors, prompts and action types
convert by name. Resources and development cards are plain strings in
Catanatron (``"WOOD"``, ``"KNIGHT"``) and ``Resource``/``DevelopmentCard``
members here.

Maps convert through ``MapSpec``: ``catanatron_map_from_spec`` walks the tiles
in spec order like Catanatron's ``initialize_tiles``, so both engines number
nodes and edges identically. Stock Catanatron's ``Board`` takes node
adjacency from a module-level graph of the BASE map (``STATIC_GRAPH``), which
is right for every template map but not for other shapes; run Catanatron code
on such maps inside ``map_graph(catan_map)``, which swaps in the map's own
graph of land edges (the adjacency cppanatron uses).
"""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Iterator, Sequence
from functools import cache
from typing import Any

import networkx as nx

from catanatron.game import Game as CatanatronGame
from catanatron.gym.board_tensor_features import (
    HEIGHT,
    WIDTH,
    get_node_and_edge_maps,
    get_tile_coordinate_map,
)
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import Action as CatanatronAction
from catanatron.models.enums import ActionPrompt as CatanatronActionPrompt
from catanatron.models.enums import ActionType as CatanatronActionType
from catanatron.models.decks import Supply
from catanatron.models import board as catanatron_board
from catanatron.models.coordinate_system import Direction as CatanatronDirection
from catanatron.models.map import CatanMap as CatanatronMap
from catanatron.models.map import get_nodes_and_edges
from catanatron.models.player import Color as CatanatronColor
from catanatron.models.player import SimplePlayer
from catanatron.models.tiles import LandTile, Port, Water

from . import _core


def to_catanatron_color(color: _core.Color) -> CatanatronColor:
    return CatanatronColor[color.name]


def from_catanatron_color(color: CatanatronColor) -> _core.Color:
    return _core.Color[color.name]


def to_catanatron_prompt(prompt: _core.ActionPrompt) -> CatanatronActionPrompt:
    return CatanatronActionPrompt[prompt.name]


def to_catanatron_resource(resource: _core.Resource | None) -> str | None:
    return None if resource is None else resource.name


def from_catanatron_resource(resource: str | None) -> _core.Resource | None:
    return None if resource is None else _core.Resource[resource]


def to_catanatron_card(card: _core.DevelopmentCard) -> str:
    return card.name


def from_catanatron_card(card: str) -> _core.DevelopmentCard:
    return _core.DevelopmentCard[card]


def _map_value(action_type: str, value: Any, *, resource, color) -> Any:
    """Translate an action value between engines with the given converters."""
    if action_type in ("DISCARD_RESOURCE", "PLAY_MONOPOLY"):
        return resource(value)
    if action_type == "PLAY_YEAR_OF_PLENTY":
        return tuple(resource(item) for item in value)
    if action_type == "MOVE_ROBBER":
        coordinate, victim = value
        return (tuple(coordinate), None if victim is None else color(victim))
    if action_type == "MARITIME_TRADE":
        return tuple(resource(item) for item in value)
    if action_type == "CONFIRM_TRADE":
        return (*value[:10], color(value[10]))
    if action_type == "BUILD_ROAD":
        return tuple(sorted(value))
    if isinstance(value, list):
        return tuple(value)
    return value


def to_catanatron_action(action: _core.Action) -> CatanatronAction:
    return CatanatronAction(
        to_catanatron_color(action.color),
        CatanatronActionType[action.action_type.name],
        _map_value(
            action.action_type.name,
            action.value,
            resource=to_catanatron_resource,
            color=to_catanatron_color,
        ),
    )


def from_catanatron_action(action: CatanatronAction) -> _core.Action:
    return _core.Action(
        from_catanatron_color(action.color),
        _core.ActionType[action.action_type.name],
        _map_value(
            action.action_type.name,
            action.value,
            resource=from_catanatron_resource,
            color=from_catanatron_color,
        ),
    )


def to_catanatron_supply(rules: _core.GameRules) -> Supply:
    """Catanatron's bank, deck and piece counts for native rules.

    Scoring, discards and the robber are separate Catanatron Game arguments.
    """
    return Supply(
        resource_bank=tuple(rules.resource_bank),
        development_deck=tuple(rules.development_deck),
        roads=rules.roads_per_player,
        settlements=rules.settlements_per_player,
        cities=rules.cities_per_player,
    )


def _as_spec(spec: _core.MapSpec | dict) -> _core.MapSpec:
    return _core.MapSpec.from_dict(spec) if isinstance(spec, dict) else spec


def catanatron_map_from_spec(spec: _core.MapSpec | dict) -> CatanatronMap:
    """Catanatron's board for an explicit map, built through ``from_tiles``.

    Tiles are placed in spec order with Catanatron's own ``get_nodes_and_edges``,
    the walk ``initialize_tiles`` does, so node, edge, land tile and port ids
    match ``cppanatron.CatanMap.from_spec(spec)``.
    """
    tiles: dict = {}
    node_autoinc = land_id = port_id = 0
    for tile in _as_spec(spec).tiles:
        coordinate = tuple(tile.coordinate)
        nodes, edges, node_autoinc = get_nodes_and_edges(tiles, coordinate, node_autoinc)
        if tile.kind == _core.TileKind.LAND:
            resource = to_catanatron_resource(tile.resource)
            tiles[coordinate] = LandTile(land_id, resource, tile.number, nodes, edges)
            land_id += 1
        elif tile.kind == _core.TileKind.PORT:
            tiles[coordinate] = Port(
                port_id,
                to_catanatron_resource(tile.port_resource),
                CatanatronDirection[tile.direction.name],
                nodes,
                edges,
            )
            port_id += 1
        else:
            tiles[coordinate] = Water(nodes, edges)
    return CatanatronMap.from_tiles(tiles)


def map_spec_from_catanatron(catan_map: CatanatronMap) -> _core.MapSpec:
    """The explicit tiles of a Catanatron map, in its tile order.

    Raises ValueError when Catanatron's ids do not follow that order (a map
    whose tiles dict was reordered after building), since the spec could not
    reproduce them.
    """
    specs = []
    for coordinate, tile in catan_map.tiles.items():
        if isinstance(tile, LandTile):
            specs.append(
                _core.TileSpec(
                    coordinate,
                    _core.TileKind.LAND,
                    resource=from_catanatron_resource(tile.resource),
                    number=tile.number,
                )
            )
        elif isinstance(tile, Port):
            specs.append(
                _core.TileSpec(
                    coordinate,
                    _core.TileKind.PORT,
                    direction=_core.Direction[CatanatronDirection(tile.direction).name],
                    port_resource=from_catanatron_resource(tile.resource),
                )
            )
        elif isinstance(tile, Water):
            specs.append(_core.TileSpec(coordinate, _core.TileKind.WATER))
        else:  # pragma: no cover - guards an upstream tile-type addition
            raise TypeError(f"unknown Catanatron tile type {type(tile)}")
    spec = _core.MapSpec(specs)
    native = _core.CatanMap.from_spec(spec)
    for coordinate, tile in catan_map.tiles.items():
        rebuilt = native.tile_at(coordinate)
        if rebuilt.nodes != tuple(tile.nodes.values()) or (
            not isinstance(tile, Water) and rebuilt.id != tile.id
        ):
            raise ValueError("Catanatron map ids do not follow its tile order")
    return spec


def to_catanatron_map(
    native_map: _core.CatanMap, map_type: _core.MapType | None = None
) -> CatanatronMap:
    """The Catanatron board with the same tiles, numbers, ports and ids.

    ``map_type`` is accepted for compatibility and ignored: every native map
    carries its explicit spec.
    """
    del map_type
    return catanatron_map_from_spec(native_map.spec())


_GRAPHS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def catanatron_map_graph(catan_map: CatanatronMap) -> nx.Graph:
    """The node graph of a Catanatron map's land tiles (cppanatron's adjacency)."""
    graph = _GRAPHS.get(catan_map)
    if graph is None:
        graph = nx.Graph()
        for tile in catan_map.land_tiles.values():
            graph.add_nodes_from(tile.nodes.values())
            graph.add_edges_from(tile.edges.values())
        _GRAPHS[catan_map] = graph
    return graph


@contextlib.contextmanager
def map_graph(catan_map: CatanatronMap) -> Iterator[None]:
    """Run stock Catanatron rules on ``catan_map``'s own node graph.

    Catanatron's ``Board`` reads adjacency from ``catanatron.models.board.
    STATIC_GRAPH``, built once from the BASE map. Within this block that name is
    the map's land-edge graph, which agrees with the BASE graph on template
    maps and is the right graph on any other shape. Boards must be created and
    played inside the block (a board caches its buildable subgraph when built).
    Not thread-safe: it swaps a module global.
    """
    previous = catanatron_board.STATIC_GRAPH
    catanatron_board.STATIC_GRAPH = catanatron_map_graph(catan_map)
    try:
        yield
    finally:
        catanatron_board.STATIC_GRAPH = previous


def reference_game(
    game: _core.Game,
    map_type: _core.MapType | None = None,
    *,
    seed: int | None = None,
    players: Sequence[Any] | None = None,
) -> CatanatronGame:
    """A fresh Catanatron game with ``game``'s board, seating and rules.

    Create it, and play it on maps that are not template maps, inside
    ``map_graph(reference.state.board.map)``. ``map_type`` is ignored.
    """
    colors = [to_catanatron_color(color) for color in game.colors]
    players = list(players) if players is not None else [SimplePlayer(color) for color in colors]
    if [player.color for player in players] != colors:
        raise ValueError("players must follow the native game's seating order")
    rules = game.rules
    catan_map = to_catanatron_map(game.board.map, map_type)
    with map_graph(catan_map):
        reference = CatanatronGame(
            players,
            seed=seed,
            discard_limit=rules.discard_limit,
            friendly_robber=rules.friendly_robber,
            vps_to_win=rules.vps_to_win,
            catan_map=catan_map,
            supply=to_catanatron_supply(rules),
        )
    state = reference.state
    state.players = players
    state.colors = tuple(colors)
    state.color_to_index = {color: index for index, color in enumerate(colors)}
    state.current_player_index = 0
    state.current_turn_index = 0
    with map_graph(catan_map):
        reference.playable_actions = generate_playable_actions(state)
    return reference


@cache
def observation_layout() -> _core.ObservationLayout:
    """Board-tensor geometry of Catanatron's ``create_board_tensor``."""
    nodes, edges = get_node_and_edge_maps()
    tiles = {
        tuple(int(axis) for axis in coordinate): (column, row)
        for coordinate, (row, column) in get_tile_coordinate_map().items()
    }
    return _core.ObservationLayout(WIDTH, HEIGHT, nodes, edges, tiles)


__all__ = [
    "catanatron_map_from_spec",
    "catanatron_map_graph",
    "from_catanatron_action",
    "from_catanatron_card",
    "from_catanatron_color",
    "from_catanatron_resource",
    "map_graph",
    "map_spec_from_catanatron",
    "observation_layout",
    "reference_game",
    "to_catanatron_action",
    "to_catanatron_card",
    "to_catanatron_color",
    "to_catanatron_map",
    "to_catanatron_prompt",
    "to_catanatron_resource",
    "to_catanatron_supply",
]
