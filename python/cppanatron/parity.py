"""Run the same Catan scenario on Catanatron and cppanatron.

``new_game(engine, ...)`` returns a :class:`ParityGame` speaking Catanatron's own
types (``Action``, ``Color``, ``ActionPrompt``, resource strings), so a rules test
written once runs unchanged on both engines. Chance outcomes can be forced
through ``execute(action, outcome=...)``, and :meth:`ParityGame.snapshot`
captures every piece of state both engines track, playable actions included,
so differential tests can compare them after each transition.

Both engines share a board: it is generated natively (or given as an explicit
``MapSpec``) and rebuilt tile for tile in Catanatron. Requires ``catanatron``
(our fork, whose maps carry their own node graphs, for shapes other than the
template layouts).
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Union

from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    Action,
    ActionPrompt,
    ActionRecord,
    ActionType,
)
from catanatron.models.actions import generate_playable_actions
from catanatron.models.player import Color
from catanatron.state_functions import player_key

from . import _core
from .catanatron import (
    from_catanatron_action,
    from_catanatron_card,
    from_catanatron_color,
    from_catanatron_resource,
    reference_game,
    to_catanatron_action,
    to_catanatron_color,
    to_catanatron_prompt,
)

Engine = Literal["catanatron", "cppanatron"]
ENGINES: tuple[Engine, ...] = ("catanatron", "cppanatron")

Dice = tuple[int, int]
#: Dice for ROLL, a card name for BUY_DEVELOPMENT_CARD, a resource name for a steal.
Outcome = Union[Dice, str, None]

#: The actions whose result depends on chance.
CHANCE_ACTIONS = frozenset(
    {ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD, ActionType.MOVE_ROBBER}
)


@dataclass(frozen=True)
class PlayerSnapshot:
    victory_points: int
    actual_victory_points: int
    roads_available: int
    settlements_available: int
    cities_available: int
    has_road: bool
    has_army: bool
    has_rolled: bool
    has_played_development_card_in_turn: bool
    longest_road_length: int
    resources: tuple[int, ...]
    development_cards: tuple[int, ...]
    played_development_cards: tuple[int, ...]
    development_card_owned_at_start: tuple[bool, ...]


@dataclass(frozen=True)
class Snapshot:
    """Everything both engines agree to track, in Catanatron's vocabulary."""

    colors: tuple[Color, ...]
    current_color: Color
    current_prompt: ActionPrompt
    is_initial_build_phase: bool
    is_discarding: bool
    is_road_building: bool
    current_turn_index: int
    num_turns: int
    resource_bank: tuple[int, ...]
    development_cards_remaining: int
    robber_coordinate: tuple[int, int, int]
    buildings: frozenset[tuple[int, Color, str]]
    roads: frozenset[tuple[tuple[int, int], Color]]
    players: tuple[PlayerSnapshot, ...]
    winning_color: Color | None
    playable_actions: frozenset[Action]


class ParityGame(abc.ABC):
    """One engine's game behind a shared, Catanatron-shaped interface."""

    engine: Engine

    @property
    @abc.abstractmethod
    def colors(self) -> tuple[Color, ...]: ...

    @property
    @abc.abstractmethod
    def playable_actions(self) -> list[Action]: ...

    @abc.abstractmethod
    def execute(self, action: Action, outcome: Outcome = None) -> Outcome:
        """Apply ``action``, forcing ``outcome`` when given.

        Returns the chance outcome the engine used when it reports one.
        """

    @abc.abstractmethod
    def snapshot(self) -> Snapshot: ...

    @abc.abstractmethod
    def set_resources(self, color: Color, resources: Sequence[int]) -> None:
        """Overwrite a hand (WOOD, BRICK, SHEEP, WHEAT, ORE) without the bank."""

    @abc.abstractmethod
    def set_development_cards(self, color: Color, cards: Sequence[int]) -> None:
        """Overwrite unplayed development cards, playable this turn.

        Counts follow Catanatron's DEVELOPMENT_CARDS order; the victory point
        count must stay unchanged, since those cards score when bought.
        """

    @property
    def current_color(self) -> Color:
        return self.snapshot().current_color

    @property
    def current_prompt(self) -> ActionPrompt:
        return self.snapshot().current_prompt

    @property
    def winning_color(self) -> Color | None:
        return self.snapshot().winning_color

    def player(self, color: Color) -> PlayerSnapshot:
        snapshot = self.snapshot()
        return snapshot.players[snapshot.colors.index(color)]

    def find_action(self, action_type: ActionType, value=..., color: Color | None = None) -> Action:
        """The unique playable action of a type, optionally with a given value."""
        matches = [
            action
            for action in self.playable_actions
            if action.action_type == action_type
            and (value is ... or action.value == value)
            and (color is None or action.color == color)
        ]
        if len(matches) != 1:
            raise LookupError(
                f"expected one playable {action_type.name} with value {value!r}, "
                f"found {len(matches)}"
            )
        return matches[0]

    def play(self, action_type: ActionType, value=..., outcome: Outcome = None) -> Outcome:
        """Execute the unique playable action matching ``action_type``/``value``."""
        return self.execute(self.find_action(action_type, value), outcome)


def _sorted(actions) -> list[Action]:
    return sorted(actions, key=repr)


class CatanatronParityGame(ParityGame):
    engine: Engine = "catanatron"

    def __init__(self, native: _core.Game, seed: int):
        self.game = reference_game(native, seed=seed)

    @property
    def colors(self) -> tuple[Color, ...]:
        return tuple(self.game.state.colors)

    @property
    def playable_actions(self) -> list[Action]:
        return _sorted(self.game.playable_actions)

    def execute(self, action: Action, outcome: Outcome = None) -> Outcome:
        record = None
        if outcome is not None:
            record = ActionRecord(action=action, result=outcome)
        result = self.game.execute(action, action_record=record).result
        # Catanatron also records results of deterministic actions, like discards.
        return result if action.action_type in CHANCE_ACTIONS else None

    def _refresh(self) -> None:
        self.game.playable_actions = generate_playable_actions(self.game.state)

    def set_resources(self, color: Color, resources: Sequence[int]) -> None:
        key = player_key(self.game.state, color)
        for resource, count in zip(RESOURCES, resources, strict=True):
            self.game.state.player_state[f"{key}_{resource}_IN_HAND"] = count
        self._refresh()

    def set_development_cards(self, color: Color, cards: Sequence[int]) -> None:
        state = self.game.state.player_state
        key = player_key(self.game.state, color)
        if cards[-1] != state[f"{key}_VICTORY_POINT_IN_HAND"]:
            raise ValueError("victory point cards must be bought so victory points stay consistent")
        for card, count in zip(DEVELOPMENT_CARDS, cards, strict=True):
            state[f"{key}_{card}_IN_HAND"] = count
            if card != "VICTORY_POINT":
                state[f"{key}_{card}_OWNED_AT_START"] = count > 0
        self._refresh()

    def _player(self, color: Color) -> PlayerSnapshot:
        state = self.game.state.player_state
        key = player_key(self.game.state, color)
        return PlayerSnapshot(
            victory_points=state[f"{key}_VICTORY_POINTS"],
            actual_victory_points=state[f"{key}_ACTUAL_VICTORY_POINTS"],
            roads_available=state[f"{key}_ROADS_AVAILABLE"],
            settlements_available=state[f"{key}_SETTLEMENTS_AVAILABLE"],
            cities_available=state[f"{key}_CITIES_AVAILABLE"],
            has_road=bool(state[f"{key}_HAS_ROAD"]),
            has_army=bool(state[f"{key}_HAS_ARMY"]),
            has_rolled=bool(state[f"{key}_HAS_ROLLED"]),
            has_played_development_card_in_turn=bool(
                state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]
            ),
            longest_road_length=state[f"{key}_LONGEST_ROAD_LENGTH"],
            resources=tuple(state[f"{key}_{resource}_IN_HAND"] for resource in RESOURCES),
            development_cards=tuple(state[f"{key}_{card}_IN_HAND"] for card in DEVELOPMENT_CARDS),
            played_development_cards=tuple(
                state[f"{key}_PLAYED_{card}"] for card in DEVELOPMENT_CARDS
            ),
            development_card_owned_at_start=tuple(
                bool(state[f"{key}_{card}_OWNED_AT_START"]) for card in DEVELOPMENT_CARDS[:-1]
            ),
        )

    def snapshot(self) -> Snapshot:
        state = self.game.state
        board = state.board
        return Snapshot(
            colors=tuple(state.colors),
            current_color=state.current_color(),
            current_prompt=state.current_prompt,
            is_initial_build_phase=state.is_initial_build_phase,
            is_discarding=state.is_discarding,
            is_road_building=state.is_road_building,
            current_turn_index=state.current_turn_index,
            num_turns=state.num_turns,
            resource_bank=tuple(state.resource_freqdeck),
            development_cards_remaining=len(state.development_listdeck),
            robber_coordinate=tuple(board.robber_coordinate),
            buildings=frozenset(
                (node, color, building) for node, (color, building) in board.buildings.items()
            ),
            roads=frozenset((tuple(sorted(edge)), color) for edge, color in board.roads.items()),
            players=tuple(self._player(color) for color in state.colors),
            winning_color=self.game.winning_color(),
            playable_actions=frozenset(self.game.playable_actions),
        )


class CppanatronParityGame(ParityGame):
    engine: Engine = "cppanatron"

    def __init__(self, game: _core.Game):
        self.game = game

    @property
    def colors(self) -> tuple[Color, ...]:
        return tuple(to_catanatron_color(color) for color in self.game.colors)

    @property
    def playable_actions(self) -> list[Action]:
        return _sorted(to_catanatron_action(action) for action in self.game.playable_actions)

    def execute(self, action: Action, outcome: Outcome = None) -> Outcome:
        native = from_catanatron_action(action)
        forced = {}
        if outcome is not None:
            if action.action_type == ActionType.ROLL:
                forced["dice"] = outcome
            elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                forced["development_card"] = from_catanatron_card(outcome)
            elif action.action_type == ActionType.MOVE_ROBBER:
                forced["stolen_resource"] = from_catanatron_resource(outcome)
            else:
                raise ValueError(f"{action.action_type.name} has no chance outcome")
        self.game.execute(native, **forced)
        return outcome

    def set_resources(self, color: Color, resources: Sequence[int]) -> None:
        self.game.set_resources(from_catanatron_color(color), tuple(resources))

    def set_development_cards(self, color: Color, cards: Sequence[int]) -> None:
        self.game.set_development_cards(from_catanatron_color(color), tuple(cards))

    def snapshot(self) -> Snapshot:
        game = self.game
        board = game.board
        winner = game.winning_color
        return Snapshot(
            colors=self.colors,
            current_color=to_catanatron_color(game.current_color),
            current_prompt=to_catanatron_prompt(game.current_prompt),
            is_initial_build_phase=game.is_initial_build_phase,
            is_discarding=game.is_discarding,
            is_road_building=game.is_road_building,
            current_turn_index=game.current_turn_index,
            num_turns=game.num_turns,
            resource_bank=game.resource_bank,
            development_cards_remaining=game.development_cards_remaining,
            robber_coordinate=board.robber_coordinate,
            buildings=frozenset(
                (node, to_catanatron_color(color), building.name)
                for node, (color, building) in board.buildings.items()
            ),
            roads=frozenset(
                (edge, to_catanatron_color(color)) for edge, color in board.roads.items()
            ),
            players=tuple(
                PlayerSnapshot(
                    victory_points=state.victory_points,
                    actual_victory_points=state.actual_victory_points,
                    roads_available=state.roads_available,
                    settlements_available=state.settlements_available,
                    cities_available=state.cities_available,
                    has_road=state.has_road,
                    has_army=state.has_army,
                    has_rolled=state.has_rolled,
                    has_played_development_card_in_turn=state.has_played_development_card_in_turn,
                    longest_road_length=state.longest_road_length,
                    resources=state.resources,
                    development_cards=state.development_cards,
                    played_development_cards=state.played_development_cards,
                    development_card_owned_at_start=state.development_card_owned_at_start,
                )
                for state in game.players
            ),
            winning_color=None if winner is None else to_catanatron_color(winner),
            playable_actions=frozenset(
                to_catanatron_action(action) for action in game.playable_actions
            ),
        )


def new_game(
    engine: Engine,
    players: int | Sequence[Color] = 2,
    *,
    map_type: str = "BASE",
    map_spec: _core.MapSpec | dict | None = None,
    seed: int = 0,
    map_seed: int | None = None,
    number_placement: str = "official_spiral",
    vps_to_win: int = 10,
    discard_limit: int = 7,
    friendly_robber: bool = False,
    resource_bank: Sequence[int] | None = None,
    development_deck: Sequence[int] | None = None,
) -> ParityGame:
    """A game on ``engine``; equal arguments give both engines the same board.

    ``map_spec`` plays on an explicit map instead of a ``map_type`` board.
    ``seed`` feeds each engine's own RNG, so unforced dice, cards and steals
    differ across engines. Force them wherever results are compared.
    ``resource_bank`` and ``development_deck`` default to the base game's
    (Resource and DevelopmentCard order).
    """
    defaults = _core.GameRules()
    rules = _core.GameRules(
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
        friendly_robber=friendly_robber,
        resource_bank=tuple(resource_bank or defaults.resource_bank),
        development_deck=tuple(development_deck or defaults.development_deck),
    )
    native_players = (
        players
        if isinstance(players, int)
        else [from_catanatron_color(color) for color in players]
    )
    if map_spec is not None:
        native = _core.Game(
            native_players, map=_core.CatanMap.from_spec(map_spec), seed=seed, rules=rules
        )
    else:
        native = _core.Game(
            native_players,
            map_type=_core.MapType[map_type],
            seed=seed,
            map_seed=map_seed,
            number_placement=_core.NumberPlacement[number_placement.upper()],
            rules=rules,
        )
    if engine == "cppanatron":
        return CppanatronParityGame(native)
    if engine == "catanatron":
        return CatanatronParityGame(native, seed)
    raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")


__all__ = [
    "CHANCE_ACTIONS",
    "ENGINES",
    "CatanatronParityGame",
    "CppanatronParityGame",
    "Engine",
    "Outcome",
    "ParityGame",
    "PlayerSnapshot",
    "Snapshot",
    "new_game",
]
