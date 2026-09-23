"""Conversions between cppanatron and Catanatron. Requires ``catanatron``.

Enum members share Catanatron's names, so colors, prompts and action types
convert by name. Resources and development cards are plain strings in
Catanatron (``"WOOD"``, ``"KNIGHT"``) and ``Resource``/``DevelopmentCard``
members here.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Any

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
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    CatanMap as CatanatronMap,
    build_map,
    initialize_tiles,
)
from catanatron.models.player import Color as CatanatronColor
from catanatron.models.player import SimplePlayer
from catanatron.models.tiles import LandTile

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


def to_catanatron_map(native_map: _core.CatanMap, map_type: _core.MapType) -> CatanatronMap:
    """The Catanatron board with the same tiles, numbers and ports.

    Catanatron numbers nodes and edges while walking its template, so placing
    the same terrain in the same template order reproduces every id.
    """
    if map_type == _core.MapType.TOURNAMENT:
        return build_map("TOURNAMENT")  # Fully fixed in both engines.
    template = MINI_MAP_TEMPLATE if map_type == _core.MapType.MINI else BASE_MAP_TEMPLATE
    tiles = native_map.tiles
    land_resources, numbers, port_resources = [], [], []
    for coordinate, tile_type in template.topology.items():
        tile = tiles[coordinate]
        if tile_type is LandTile:
            land_resources.append(to_catanatron_resource(tile.resource))
            if tile.resource is not None:
                numbers.append(tile.number)
        elif isinstance(tile_type, tuple):
            port_resources.append(to_catanatron_resource(tile.resource))
    # initialize_tiles pops from the end of each list.
    catanatron_tiles = initialize_tiles(
        template,
        shuffled_port_resources_param=port_resources[::-1],
        shuffled_tile_resources_param=land_resources[::-1],
        shuffled_numbers_param=numbers[::-1],
        number_placement="random",
    )
    return CatanatronMap.from_tiles(catanatron_tiles)


def reference_game(
    game: _core.Game,
    map_type: _core.MapType,
    *,
    seed: int | None = None,
    players: Sequence[Any] | None = None,
) -> CatanatronGame:
    """A fresh Catanatron game with ``game``'s board, seating and rules."""
    colors = [to_catanatron_color(color) for color in game.colors]
    players = list(players) if players is not None else [SimplePlayer(color) for color in colors]
    if [player.color for player in players] != colors:
        raise ValueError("players must follow the native game's seating order")
    reference = CatanatronGame(
        players,
        seed=seed,
        discard_limit=game.discard_limit,
        friendly_robber=game.friendly_robber,
        vps_to_win=game.vps_to_win,
        catan_map=to_catanatron_map(game.board.map, map_type),
    )
    state = reference.state
    state.players = players
    state.colors = tuple(colors)
    state.color_to_index = {color: index for index, color in enumerate(colors)}
    state.current_player_index = 0
    state.current_turn_index = 0
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
    "from_catanatron_action",
    "from_catanatron_card",
    "from_catanatron_color",
    "from_catanatron_resource",
    "observation_layout",
    "reference_game",
    "to_catanatron_action",
    "to_catanatron_card",
    "to_catanatron_color",
    "to_catanatron_map",
    "to_catanatron_prompt",
    "to_catanatron_resource",
]
