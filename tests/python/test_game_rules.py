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
