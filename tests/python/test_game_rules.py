"""Game rules beyond the base game, and games of up to six players."""

from __future__ import annotations

import random

import numpy as np
import pytest

import cppanatron as cpp
from cppanatron.catanatron import observation_layout

LAYOUT = observation_layout()

EXTENSION = dict(resource_bank=(24, 24, 24, 24, 24), development_deck=(20, 3, 3, 3, 5))


def test_default_rules_are_the_base_game():
    rules = cpp.GameRules()
    assert rules.vps_to_win == 10 and rules.discard_limit == 7 and not rules.friendly_robber
    assert rules.resource_bank == (19, 19, 19, 19, 19)
    assert rules.development_deck == (14, 2, 2, 2, 5)
    assert (rules.roads_per_player, rules.settlements_per_player, rules.cities_per_player) == (
        15,
        5,
        4,
    )
    assert cpp.Game(2).rules == rules
    assert "GameRules(" in repr(rules)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(vps_to_win=0),
        dict(discard_limit=-1),
        dict(resource_bank=(19, 19, -1, 19, 19)),
        dict(development_deck=(14, 2, 2, 2, -5)),
        dict(settlements_per_player=1),
        dict(roads_per_player=1),
    ],
)
def test_unplayable_rules_are_rejected(kwargs):
    with pytest.raises(ValueError):
        cpp.GameRules(**kwargs)


def test_a_game_starts_from_its_rules():
    rules = cpp.GameRules(
        vps_to_win=12,
        discard_limit=9,
        roads_per_player=13,
        settlements_per_player=4,
        cities_per_player=3,
        **EXTENSION,
    )
    game = cpp.Game(6, seed=1, rules=rules)
    assert game.rules == rules
    assert (game.vps_to_win, game.discard_limit) == (12, 9)
    assert game.resource_bank == (24, 24, 24, 24, 24)
    assert game.development_cards_remaining == 34
    for player in game.players:
        assert (player.roads_available, player.settlements_available, player.cities_available) == (
            13,
            4,
            3,
        )


def test_rules_and_loose_rule_arguments_do_not_mix():
    with pytest.raises(ValueError, match="inside rules"):
        cpp.Game(2, rules=cpp.GameRules(), vps_to_win=8)
    with pytest.raises(ValueError, match="inside rules"):
        cpp.GameBatch(1, 2, cpp.MapType.BASE, LAYOUT, rules=cpp.GameRules(), discard_limit=9)


def test_loose_rule_arguments_keep_their_old_meaning():
    game = cpp.Game(2, seed=4, vps_to_win=8, discard_limit=9, friendly_robber=True)
    assert game.rules == cpp.GameRules(vps_to_win=8, discard_limit=9, friendly_robber=True)
    assert game.search_equivalent(cpp.Game(2, seed=4, rules=game.rules))


def test_batches_size_six_player_games():
    batch = cpp.GameBatch(2, 6, cpp.MapType.BASE, LAYOUT, rules=cpp.GameRules(**EXTENSION))
    assert batch.action_space_size == len(cpp.FlatActionSpace(6, cpp.MapType.BASE)) == 389
    assert batch.observation_size == cpp.full_observation_size(6, LAYOUT)
    assert cpp.full_numeric_observation_size(6) == 32 * 6 + 10
    with pytest.raises(ValueError):
        cpp.full_numeric_observation_size(7)


def _play_in_lockstep(games, spaces, steps, seed):
    rng = random.Random(seed)
    for _ in range(steps):
        if games[0].winning_color is not None:
            return
        masks = [space.mask(game) for game, space in zip(games, spaces)]
        for mask in masks[1:]:
            np.testing.assert_array_equal(masks[0], mask)
        index = rng.choice(np.flatnonzero(masks[0]).tolist())
        for game, space in zip(games, spaces):
            game.execute(space.decode(index, game))
        yield


@pytest.mark.parametrize("num_players", [3, 6])
def test_observations_depend_on_seats_not_color_names(num_players):
    """Relabelling every color must not change any seat's observation."""
    colors = list(cpp.Color)
    seated = colors[:num_players]
    relabelled = colors[::-1][:num_players]
    games = [cpp.Game(order, seed=21, map_seed=8) for order in (seated, relabelled)]
    spaces = [cpp.FlatActionSpace(num_players, cpp.MapType.BASE)] * 2
    for _ in _play_in_lockstep(games, spaces, steps=400, seed=num_players):
        for seat in range(num_players):
            np.testing.assert_array_equal(
                cpp.full_observation(games[0], seat, LAYOUT),
                cpp.full_observation(games[1], seat, LAYOUT),
            )


@pytest.mark.parametrize("enemies", [cpp.ValueEnemies.NEXT, cpp.ValueEnemies.ALL])
def test_the_value_expert_plays_six_player_games(enemies):
    game = cpp.Game(6, seed=9, rules=cpp.GameRules(**EXTENSION))
    for _ in range(300):
        if game.winning_color is not None:
            break
        game.execute(cpp.value_action(game, enemies=enemies))
    assert not game.is_initial_build_phase


def test_all_enemies_is_the_next_enemy_with_two_players():
    game = cpp.Game(2, seed=9)
    while game.is_initial_build_phase:
        game.execute(game.playable_actions[0])
    for seat in range(2):
        assert cpp.value_score(game, seat, enemies=cpp.ValueEnemies.ALL) == cpp.value_score(
            game, seat
        )


# --- OpenHex's friendly robber and short bank, on three tiles ---------------------

DESERT, ORE_8, WHEAT_6 = (0, 0, 0), (1, -1, 0), (0, -1, 1)
THREE_TILES = cpp.CatanMap.from_spec({"tiles": [
    {"coordinate": list(DESERT), "type": "land"},
    {"coordinate": list(ORE_8), "type": "land", "resource": "ORE", "number": 8},
    {"coordinate": list(WHEAT_6), "type": "land", "resource": "WHEAT", "number": 6},
]})


def _three_tile_nodes() -> tuple[int, int, int]:
    """The node where the three tiles meet, and a node on the ore tile and one on the
    desert alone, neither next to the corner nor to each other."""
    nodes = {c: set(tile.nodes) for c, tile in THREE_TILES.land_tiles.items()}
    (corner,) = nodes[DESERT] & nodes[ORE_8] & nodes[WHEAT_6]
    far = lambda node, other: node != other and node not in THREE_TILES.neighbors(other)
    ore_only = max(n for n in nodes[ORE_8] - nodes[DESERT] - nodes[WHEAT_6] if far(n, corner))
    desert_only = max(n for n in nodes[DESERT] - nodes[ORE_8] - nodes[WHEAT_6]
                      if far(n, corner) and far(n, ore_only))
    return corner, ore_only, desert_only


CORNER, ORE_ONLY, DESERT_ONLY = _three_tile_nodes()
FRIENDLY_1V1 = dict(vps_to_win=15, discard_limit=9, friendly_robber=True)


def _three_tile_game(red: dict, blue: dict, rules: dict = FRIENDLY_1V1, **options) -> cpp.Game:
    game = cpp.Game(2, map=THREE_TILES, seed=5, rules=cpp.GameRules(**rules))
    game.apply_setup([cpp.PlayerSetup(**red), cpp.PlayerSetup(**blue)], **options)
    return game


def _play(game: cpp.Game, action_type: cpp.ActionType, **outcome) -> None:
    game.execute(next(a for a in game.playable_actions if a.action_type == action_type), **outcome)


def _types(game: cpp.Game) -> set[cpp.ActionType]:
    return {a.action_type for a in game.playable_actions}


def _robber_tiles(game: cpp.Game) -> set[tuple]:
    return {a.value[0] for a in game.playable_actions if a.action_type == cpp.ActionType.MOVE_ROBBER}


@pytest.mark.parametrize("blue_ore", [3, 11])
def test_friendly_robber_a_seven_moves_no_robber_when_every_tile_is_shielded(blue_ore):
    """Blue's corner settlement (1 point) shields both tiles the robber could go to:
    after the 7, and the discards if any, the roller plays on."""
    game = _three_tile_game(dict(settlements=[DESERT_ONLY], resources=(4, 3, 2, 1, 0) if blue_ore == 11 else (1, 1, 0, 0, 0)),
                            dict(settlements=[CORNER], resources=(0, 0, 0, 0, blue_ore)))
    tracker = cpp.HandTracker(game, 1)
    before = game.copy()
    roll = next(a for a in game.playable_actions if a.action_type == cpp.ActionType.ROLL)
    game.execute(roll, dice=(3, 4))
    tracker.observe(before, roll, game)
    discards = 0
    while game.current_prompt == cpp.ActionPrompt.DISCARD:
        before, action = game.copy(), game.playable_actions[0]
        game.execute(action)
        tracker.observe(before, action, game)
        discards += 1
    assert discards == (10 if blue_ore == 11 else 0), "both seats over 9 discard half"
    assert game.current_prompt == cpp.ActionPrompt.PLAY_TURN and game.current_player_index == 0
    assert game.board.robber_coordinate == DESERT and game.player(0).has_rolled
    assert cpp.ActionType.END_TURN in _types(game) and cpp.ActionType.MOVE_ROBBER not in _types(game)
    assert tracker.consistent_with(game)
    _play(game, cpp.ActionType.END_TURN)
    assert game.current_player_index == 1 and _types(game) == {cpp.ActionType.ROLL}


def test_friendly_robber_shields_the_movers_own_tiles():
    """Red's own corner settlement shields both tiles at 1 point; as a city with a
    second settlement (3 points) it shields nothing."""
    blue = dict(settlements=[DESERT_ONLY])
    low = _three_tile_game(dict(settlements=[CORNER]), blue)
    _play(low, cpp.ActionType.ROLL, dice=(1, 6))
    assert low.player(0).victory_points == 1 and low.current_prompt == cpp.ActionPrompt.PLAY_TURN
    high = _three_tile_game(dict(cities=[CORNER], settlements=[ORE_ONLY]), blue)
    _play(high, cpp.ActionType.ROLL, dice=(1, 6))
    assert high.player(0).victory_points == 3 and high.current_prompt == cpp.ActionPrompt.MOVE_ROBBER
    assert _robber_tiles(high) == {ORE_8, WHEAT_6}
    # One tile open: the only move, and a robber prompt with none is no position.
    partial = _three_tile_game(dict(settlements=[DESERT_ONLY]), dict(settlements=[ORE_ONLY]))
    _play(partial, cpp.ActionType.ROLL, dice=(2, 5))
    assert partial.current_prompt == cpp.ActionPrompt.MOVE_ROBBER and _robber_tiles(partial) == {WHEAT_6}
    with pytest.raises(ValueError, match="shields them all"):
        _three_tile_game(dict(settlements=[DESERT_ONLY]), dict(settlements=[CORNER]),
                         has_rolled=True, prompt=cpp.ActionPrompt.MOVE_ROBBER)


@pytest.mark.parametrize("played", [0, 2])
def test_friendly_robber_a_knight_is_played_without_a_robber_move(played):
    game = _three_tile_game(
        dict(settlements=[DESERT_ONLY], development_cards=(2, 0, 0, 0, 0),
             played_development_cards=(played, 0, 0, 0, 0)),
        dict(settlements=[CORNER], resources=(0, 0, 0, 3, 0)))
    _play(game, cpp.ActionType.PLAY_KNIGHT_CARD)
    red = game.player(0)
    assert game.current_prompt == cpp.ActionPrompt.PLAY_TURN and _types(game) == {cpp.ActionType.ROLL}
    assert red.development_cards[0] == 1 and red.played_development_cards[0] == played + 1
    assert red.has_army == (played == 2) and red.victory_points == (3 if played == 2 else 1)
    assert game.board.robber_coordinate == DESERT and game.player(1).resources == (0, 0, 0, 3, 0)
    assert game.largest_army_taken_from is None


def test_friendly_robber_judges_a_knight_after_its_award():
    """Red's corner settlement shields both tiles until its third knight brings
    Largest Army (3 points); Blue's does once Red's fourth knight takes the award."""
    lifted = _three_tile_game(
        dict(settlements=[CORNER], development_cards=(1, 0, 0, 0, 0), played_development_cards=(2, 0, 0, 0, 0)),
        dict(settlements=[DESERT_ONLY]), has_rolled=True)
    _play(lifted, cpp.ActionType.PLAY_KNIGHT_CARD)
    assert lifted.player(0).victory_points == 3 and lifted.current_prompt == cpp.ActionPrompt.MOVE_ROBBER
    assert _robber_tiles(lifted) == {ORE_8, WHEAT_6}
    taken = _three_tile_game(
        dict(settlements=[DESERT_ONLY], development_cards=(1, 0, 0, 0, 0),
             played_development_cards=(3, 0, 0, 0, 0)),
        dict(settlements=[CORNER], played_development_cards=(3, 0, 0, 0, 0)),
        has_rolled=True, largest_army=1)
    assert taken.player(1).victory_points == 3
    _play(taken, cpp.ActionType.PLAY_KNIGHT_CARD)
    assert taken.player(0).has_army and taken.player(1).victory_points == 1
    assert taken.current_prompt == cpp.ActionPrompt.PLAY_TURN and taken.largest_army_taken_from is None
    assert taken.board.robber_coordinate == DESERT


@pytest.mark.parametrize(("bank", "red", "blue", "expected"), [
    (3, dict(cities=[CORNER], settlements=[ORE_ONLY]), dict(settlements=[DESERT_ONLY]), (3, 0, 0)),
    (2, dict(cities=[CORNER], settlements=[ORE_ONLY]), dict(settlements=[DESERT_ONLY]), (2, 0, 0)),
    (1, dict(cities=[CORNER]), dict(settlements=[DESERT_ONLY]), (1, 0, 0)),
    (0, dict(settlements=[ORE_ONLY]), dict(settlements=[DESERT_ONLY]), (0, 0, 0)),
    (2, dict(settlements=[ORE_ONLY]), dict(settlements=[CORNER]), (1, 1, 0)),
    (1, dict(settlements=[ORE_ONLY]), dict(settlements=[CORNER]), (0, 0, 1)),
    (2, dict(cities=[CORNER]), dict(settlements=[ORE_ONLY]), (0, 0, 2)),
])
def test_a_short_bank_pays_a_sole_claimant_what_is_left(bank, red, blue, expected):
    """OpenHex's rule, per resource: one seat claiming it takes what the bank has
    (however many buildings claim), several seats get nothing. Ore: Red, Blue, bank."""
    game = _three_tile_game(red, blue, rules=dict(resource_bank=(19, 19, 19, 19, bank)))
    _play(game, cpp.ActionType.ROLL, dice=(4, 4))
    assert (game.player(0).resources[4], game.player(1).resources[4], game.resource_bank[4]) == expected
