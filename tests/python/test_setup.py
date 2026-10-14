"""Game.apply_setup: declaring exact positions."""

from __future__ import annotations

import random

import pytest

import cppanatron as cpp

#: Fields a declared position must reproduce exactly. The turns-since counters
#: are not part of a declaration: they start at "never".
PLAYER_FIELDS = (
    "roads_available",
    "settlements_available",
    "cities_available",
    "has_army",
    "has_rolled",
    "has_played_development_card_in_turn",
    "resources",
    "development_cards",
    "played_development_cards",
)
#: Fields that follow road lengths. In play, the engine (like Catanatron)
#: counts a road's last edge into an opponent's building only if the road was
#: there before the building; a declared position always counts it. Its
#: per-player length also reads 0 until the first build after the initial
#: phase, so declared lengths are compared with the board's.
ROAD_FIELDS = ("victory_points", "actual_victory_points", "has_road")


def has_road_into_opponent(game: cpp.Game) -> bool:
    """Whether some road ends at another player's building."""
    buildings = game.board.buildings
    return any(
        node in buildings and buildings[node][0] != color
        for edge, color in game.board.roads.items()
        for node in edge
    )


def playable_cards(player: cpp.PlayerState) -> list[bool]:
    """Which cards could be played this turn. A card played earlier in the turn
    leaves its owned-at-start flag set with none left, which changes nothing."""
    return [
        owned and count > 0
        for owned, count in zip(player.development_card_owned_at_start, player.development_cards)
    ]


def declare(game: cpp.Game) -> tuple[list[cpp.PlayerSetup], dict]:
    """The PlayerSetups and apply_setup arguments for ``game``'s position."""
    players = [
        cpp.PlayerSetup(
            settlements=player.settlements,
            cities=player.cities,
            roads=player.roads,
            resources=player.resources,
            development_cards=player.development_cards,
            played_development_cards=player.played_development_cards,
        )
        for player in game.players
    ]

    def holder(flag: str) -> int:
        return next((seat for seat, p in enumerate(game.players) if getattr(p, flag)), -1)

    current = game.current_player_index
    options = dict(
        robber=game.board.robber_coordinate,
        current_player=current,
        has_rolled=game.player(current).has_rolled,
        has_played_development_card=game.player(current).has_played_development_card_in_turn,
        completed_turns=game.completed_turns,
        longest_road=holder("has_road"),
        largest_army=holder("has_army"),
    )
    return players, options


def assert_same_position(expected: cpp.Game, actual: cpp.Game, *, roads: bool = True) -> None:
    """Every declared field matches; with ``roads=False``, all but road lengths
    and what follows from them."""
    assert actual.board.buildings == expected.board.buildings
    assert actual.board.roads == expected.board.roads
    assert actual.board.robber_coordinate == expected.board.robber_coordinate
    if roads:
        assert actual.board.longest_road_color == expected.board.longest_road_color
    for seat, (want, got) in enumerate(zip(expected.players, actual.players, strict=True)):
        for field in PLAYER_FIELDS + (ROAD_FIELDS if roads else ()):
            assert getattr(got, field) == getattr(want, field), f"seat {seat}: {field}"
        if roads:
            board_length = expected.board.longest_road(expected.colors[seat])
            assert got.longest_road_length == board_length, f"seat {seat}: longest road"
        assert playable_cards(got) == playable_cards(want), f"seat {seat}: playable cards"
        assert sorted(got.settlements) == sorted(want.settlements)
        assert sorted(got.cities) == sorted(want.cities)
        assert sorted(got.roads) == sorted(want.roads)
    for field in (
        "resource_bank",
        "development_cards_remaining",
        "current_player_index",
        "current_turn_index",
        "current_prompt",
        "is_initial_build_phase",
        "num_turns",
        "completed_turns",
    ) + (("winning_color",) if roads else ()):
        assert getattr(actual, field) == getattr(expected, field), field
    assert set(actual.playable_actions) == set(expected.playable_actions)


def fresh_like(game: cpp.Game, map_type: cpp.MapType, map_seed: int) -> cpp.Game:
    return cpp.Game(
        list(game.colors), map_type=map_type, seed=99, map_seed=map_seed, rules=game.rules
    )


@pytest.mark.parametrize(
    ("num_players", "map_type"),
    [(2, "TOURNAMENT"), (3, "BASE"), (4, "BASE"), (6, "BASE")],
)
def test_every_turn_start_of_random_games_can_be_declared(num_players, map_type):
    native_map = cpp.MapType[map_type]
    rng = random.Random(num_players)
    exact = 0
    for game_index in range(3):
        map_seed = 100 * num_players + game_index
        game = cpp.Game(num_players, map_type=native_map, seed=game_index, map_seed=map_seed)
        checked = 0
        for _ in range(4_000):
            if game.winning_color is not None:
                break
            at_turn_start = (
                not game.is_initial_build_phase
                and game.current_prompt == cpp.ActionPrompt.PLAY_TURN
                and not game.is_road_building
                and not game.player(game.current_player_index).has_rolled
            )
            if at_turn_start and rng.random() < 0.2:
                players, options = declare(game)
                roads = not has_road_into_opponent(game)
                if not roads:
                    # Longer roads may move Longest Road; let the leader take it.
                    options["longest_road"] = None
                declared = fresh_like(game, native_map, map_seed)
                try:
                    declared.apply_setup(players, **options)
                except ValueError as exc:
                    if roads or "Longest Road is tied" not in str(exc):
                        raise
                    continue
                assert_same_position(game, declared, roads=roads)
                checked += 1
                exact += roads
            game.execute(rng.choice(game.playable_actions))
        assert checked > 0
    assert exact > 0


def test_a_declared_setup_matches_one_played_through():
    played = cpp.Game(2, map_type=cpp.MapType.TOURNAMENT, seed=1)
    for node, road in [(1, (1, 6)), (3, (3, 4)), (13, (13, 14)), (20, (20, 22))]:
        played.execute(cpp.Action(played.current_color, cpp.ActionType.BUILD_SETTLEMENT, node))
        played.execute(cpp.Action(played.current_color, cpp.ActionType.BUILD_ROAD, road))
    declared = cpp.Game(2, map_type=cpp.MapType.TOURNAMENT, seed=1)
    declared.apply_setup(
        [
            cpp.PlayerSetup(
                settlements=[1, 20], roads=[(1, 6), (20, 22)], resources=played.player(0).resources
            ),
            cpp.PlayerSetup(
                settlements=[3, 13], roads=[(3, 4), (13, 14)], resources=played.player(1).resources
            ),
        ]
    )
    assert_same_position(played, declared)


def tournament(players: int = 2, **rules) -> cpp.Game:
    return cpp.Game(players, map_type=cpp.MapType.TOURNAMENT, seed=3, rules=cpp.GameRules(**rules))


def test_victory_points_awards_bank_and_deck_are_derived():
    game = tournament(vps_to_win=15)
    red_road = [(1, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
    game.apply_setup(
        [
            cpp.PlayerSetup(
                settlements=[1],
                cities=[20],
                roads=red_road,
                resources=(2, 0, 1, 0, 3),
                development_cards=(1, 0, 0, 0, 2),
                played_development_cards=(3, 0, 0, 0, 0),
            ),
            cpp.PlayerSetup(settlements=[3, 13], roads=[(3, 4)]),
        ],
        current_player=1,
        has_rolled=True,
    )
    red, blue = game.players
    assert red.has_road and red.has_army and red.longest_road_length == 5
    assert red.victory_points == 1 + 2 + 2 + 2 and red.actual_victory_points == 9
    assert blue.victory_points == 2 and not blue.has_road
    assert (red.settlements_available, red.cities_available, red.roads_available) == (4, 3, 10)
    assert game.resource_bank == (17, 19, 18, 19, 16)
    assert game.development_cards_remaining == 25 - 1 - 2 - 3
    assert game.current_player_index == 1 and blue.has_rolled
    assert game.current_prompt == cpp.ActionPrompt.PLAY_TURN
    assert game.board.longest_road_color == cpp.Color.RED


def test_ties_for_an_award_must_name_the_holder():
    knights = cpp.PlayerSetup(settlements=[1], played_development_cards=(3, 0, 0, 0, 0))
    rival = cpp.PlayerSetup(settlements=[13], played_development_cards=(3, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="tied"):
        tournament().apply_setup([knights, rival])
    game = tournament()
    game.apply_setup([knights, rival], largest_army=1)
    assert game.player(1).has_army and not game.player(0).has_army


@pytest.mark.parametrize(
    ("players", "options", "message"),
    [
        ([cpp.PlayerSetup(settlements=[1, 6]), cpp.PlayerSetup()], {}, "distance rule"),
        ([cpp.PlayerSetup(roads=[(1, 6)]), cpp.PlayerSetup()], {}, "does not connect"),
        ([cpp.PlayerSetup(settlements=[1]), cpp.PlayerSetup(cities=[1])], {}, "more than one"),
        ([cpp.PlayerSetup(resources=(20, 0, 0, 0, 0)), cpp.PlayerSetup()], {}, "WOOD"),
        (
            [cpp.PlayerSetup(development_cards=(0, 0, 0, 0, 6)), cpp.PlayerSetup()],
            {},
            "VICTORY_POINT",
        ),
        ([cpp.PlayerSetup(settlements=[1, 3, 13, 20, 42, 49])], {}, "one player entry"),
        ([cpp.PlayerSetup(), cpp.PlayerSetup()], {"current_player": 2}, "not a seat"),
        (
            [cpp.PlayerSetup(settlements=[1]), cpp.PlayerSetup(settlements=[13])],
            {"largest_army": 0},
            "must lead",
        ),
    ],
)
def test_impossible_positions_are_rejected(players, options, message):
    with pytest.raises(ValueError, match=message):
        tournament().apply_setup(players, **options)


def test_more_pieces_than_the_rules_allow_are_rejected():
    with pytest.raises(ValueError, match="more pieces"):
        tournament(settlements_per_player=2).apply_setup(
            [cpp.PlayerSetup(settlements=[1, 20, 42]), cpp.PlayerSetup()]
        )


def test_only_fresh_games_take_a_setup():
    game = tournament()
    game.execute(game.playable_actions[0])
    with pytest.raises(RuntimeError, match="freshly created"):
        game.apply_setup([cpp.PlayerSetup(), cpp.PlayerSetup()])


def test_a_road_cut_by_a_later_settlement_can_be_declared():
    """RED's road runs through BLUE's settlement, as after a cut."""
    game = tournament()
    game.apply_setup(
        [
            cpp.PlayerSetup(settlements=[1], roads=[(1, 6), (6, 7), (7, 8)]),
            cpp.PlayerSetup(settlements=[7]),
        ]
    )
    assert game.player(0).longest_road_length == 2
