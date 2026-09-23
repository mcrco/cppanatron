"""The Python API of the compiled extension."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import cppanatron as cpp
from cppanatron.catanatron import (
    from_catanatron_action,
    observation_layout,
    to_catanatron_action,
)

LAYOUT = observation_layout()


def finish_setup(game: cpp.Game, space: cpp.FlatActionSpace) -> None:
    while game.is_initial_build_phase:
        game.execute(space.decode(int(np.flatnonzero(space.mask(game))[0]), game))


def test_setup_through_the_flat_action_space():
    game = cpp.Game(2, map_type=cpp.MapType.TOURNAMENT, seed=42)
    space = cpp.FlatActionSpace(2, cpp.MapType.TOURNAMENT)
    assert len(space) == 313
    mask = space.mask(game)
    assert mask.dtype == np.bool_ and mask.shape == (313,) and mask.sum() == 54

    played = []
    while game.is_initial_build_phase:
        index = int(np.flatnonzero(space.mask(game))[0])
        played.append(index)
        game.execute(space.decode(index, game))

    assert played == [126, 54, 128, 59, 130, 61, 132, 64]
    assert game.current_prompt == cpp.ActionPrompt.PLAY_TURN
    assert game.player(0).resources == (0, 1, 0, 1, 1)
    assert game.player(cpp.Color.BLUE).resources == (1, 0, 1, 0, 1)
    assert all(space.index(action, game) < len(space) for action in game.playable_actions)


def test_games_seat_explicit_colors():
    game = cpp.Game([cpp.Color.ORANGE, cpp.Color.WHITE])
    assert game.colors == [cpp.Color.ORANGE, cpp.Color.WHITE]
    assert game.current_color == cpp.Color.ORANGE
    with pytest.raises(ValueError, match="unique"):
        cpp.Game([cpp.Color.RED, cpp.Color.RED])
    with pytest.raises(ValueError):
        cpp.Game(5)


def test_unplayable_actions_and_bad_indices_raise():
    game = cpp.Game(2, seed=3)
    with pytest.raises(ValueError, match="not currently playable"):
        game.execute(cpp.Action(cpp.Color.BLUE, cpp.ActionType.END_TURN))
    with pytest.raises(IndexError):
        game.player(2)
    with pytest.raises(IndexError):
        cpp.FlatActionSpace(2, cpp.MapType.BASE)[10_000]
    with pytest.raises(ValueError):
        cpp.Action(cpp.Color.RED, cpp.ActionType.BUILD_ROAD, (1, 2, 3))


def test_copies_are_independent_and_replay_identically():
    game = cpp.Game(3, seed=11)
    space = cpp.FlatActionSpace(3, cpp.MapType.BASE)
    finish_setup(game, space)
    twin = copy.deepcopy(game)
    assert twin.search_equivalent(game)

    roll = game.playable_actions[0]
    game.execute(roll)
    twin.execute(roll)  # Same RNG state, so the same dice.
    assert twin.search_equivalent(game)
    twin.execute(twin.playable_actions[-1])
    assert not twin.search_equivalent(game)


def test_held_actions_survive_later_transitions():
    game = cpp.Game(2, seed=1)
    first = game.playable_actions[0]
    expected = tuple(first)
    game.execute(first)
    game.execute(game.playable_actions[0])
    assert tuple(first) == expected


def test_actions_compare_hash_and_unpack_like_namedtuples():
    edge = cpp.Action(cpp.Color.RED, cpp.ActionType.BUILD_ROAD, (7, 3))
    assert edge.value == (3, 7)  # Edges are normalized.
    assert edge == cpp.Action(cpp.Color.RED, cpp.ActionType.BUILD_ROAD, (3, 7))
    assert len({edge, cpp.Action(cpp.Color.RED, cpp.ActionType.BUILD_ROAD, (3, 7))}) == 1
    color, action_type, value = edge
    assert (color, action_type, value) == (cpp.Color.RED, cpp.ActionType.BUILD_ROAD, (3, 7))


@pytest.mark.parametrize(
    ("action_type", "value"),
    [
        ("ROLL", None),
        ("BUILD_SETTLEMENT", 12),
        ("BUILD_ROAD", (4, 15)),
        ("DISCARD_RESOURCE", "ORE"),
        ("PLAY_MONOPOLY", "SHEEP"),
        ("PLAY_YEAR_OF_PLENTY", ("WOOD", "WOOD")),
        ("PLAY_YEAR_OF_PLENTY", ("BRICK",)),
        ("MOVE_ROBBER", ((0, -1, 1), "BLUE")),
        ("MOVE_ROBBER", ((2, -2, 0), None)),
        ("MARITIME_TRADE", ("WOOD", "WOOD", None, None, "ORE")),
        ("OFFER_TRADE", (0, 0, 0, 1, 0, 0, 1, 0, 0, 0)),
        ("ACCEPT_TRADE", (0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1)),
        ("CONFIRM_TRADE", (0, 0, 0, 1, 0, 0, 1, 0, 0, 0, "WHITE")),
    ],
)
def test_catanatron_actions_round_trip(action_type, value):
    from catanatron.models.enums import Action, ActionType
    from catanatron.models.player import Color

    if action_type == "MOVE_ROBBER" and value[1] is not None:
        value = (value[0], Color[value[1]])
    if action_type == "CONFIRM_TRADE":
        value = (*value[:10], Color[value[10]])
    action = Action(Color.RED, ActionType[action_type], value)
    assert to_catanatron_action(from_catanatron_action(action)) == action


def test_scenario_setters_refresh_playable_actions():
    game = cpp.Game(2, map_type=cpp.MapType.TOURNAMENT, seed=42)
    space = cpp.FlatActionSpace(2, cpp.MapType.TOURNAMENT)
    finish_setup(game, space)
    game.execute(game.playable_actions[0], dice=(1, 1))
    game.set_resources(0, (0, 0, 1, 1, 1))
    assert cpp.ActionType.BUY_DEVELOPMENT_CARD in {a.action_type for a in game.playable_actions}
    game.set_development_cards(0, (1, 0, 0, 0, 0))
    assert cpp.ActionType.PLAY_KNIGHT_CARD in {a.action_type for a in game.playable_actions}
    with pytest.raises(ValueError, match="victory point"):
        game.set_development_cards(0, (0, 0, 0, 0, 1))


def test_board_views_follow_the_live_game():
    game = cpp.Game(2, map_type=cpp.MapType.TOURNAMENT)
    board = game.board
    assert board.buildings == {}
    game.execute(cpp.Action(cpp.Color.RED, cpp.ActionType.BUILD_SETTLEMENT, 1))
    assert board.buildings == {1: (cpp.Color.RED, cpp.Building.SETTLEMENT)}
    assert len(board.map.land_tiles) == 19
    assert board.map.tile_at((1, -1, 0)).resource == cpp.Resource.ORE


def test_full_observations_fill_caller_buffers():
    game = cpp.Game(2, seed=5)
    size = cpp.full_observation_size(2, LAYOUT)
    out = np.full(size, np.nan, dtype=np.float32)
    cpp.write_full_observation(game, 0, LAYOUT, out)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out, cpp.full_observation(game, 0, LAYOUT))


def test_output_buffers_must_already_have_the_written_dtype():
    """A converted temporary would swallow the writes, so conversion is refused."""
    game = cpp.Game(2, seed=5)
    size = cpp.full_observation_size(2, LAYOUT)
    with pytest.raises(TypeError):
        cpp.write_full_observation(game, 0, LAYOUT, np.zeros(size, dtype=np.float64))
    search = cpp.MCTSSearch(game, cpp.FlatActionSpace(2, cpp.MapType.BASE), LAYOUT)
    with pytest.raises(TypeError):
        search.root_observation(np.zeros(size, dtype=np.float64))
    with pytest.raises(TypeError):
        search.root_observation(np.zeros((2, size), dtype=np.float32)[:, 0])  # Strided view.


def test_batches_validate_and_fill_bound_buffers():
    batch = cpp.GameBatch(2, 2, cpp.MapType.MINI, LAYOUT, vps_to_win=6)
    agents = 4
    observation_offset = batch.action_space_size + (-batch.action_space_size % 4)  # Align floats.
    row = observation_offset + 4 * batch.observation_size
    buffers = dict(
        observations=np.zeros((agents, row), dtype=np.uint8),
        action_mask_offset=0,
        observation_offset=observation_offset,
        actions=np.zeros(agents, dtype=np.int32),
        rewards=np.zeros(agents, dtype=np.float32),
        terminals=np.zeros(agents, dtype=np.bool_),
        truncations=np.zeros(agents, dtype=np.bool_),
        masks=np.zeros(agents, dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="one row per agent"):
        batch.bind(**{**buffers, "rewards": np.zeros(3, dtype=np.float32)})
    with pytest.raises(TypeError):
        batch.bind(**{**buffers, "actions": np.zeros(agents, dtype=np.int64)})
    batch.bind(**buffers)
    batch.reset_all(np.array([1, 2], dtype=np.uint64), np.array([3, 4], dtype=np.uint64))

    features = buffers["observations"][:, observation_offset:].view(np.float32)
    game = cpp.Game(2, map_type=cpp.MapType.MINI, seed=3, map_seed=1, vps_to_win=6)
    np.testing.assert_array_equal(
        features[0, : batch.observation_size], cpp.full_observation(game, 0, LAYOUT)
    )
    first_legal = buffers["observations"][:, : batch.action_space_size].argmax(axis=1)
    buffers["actions"][:] = first_legal
    batch.step()


def test_search_writes_leaves_into_the_callers_buffer():
    game = cpp.Game(2, map_type=cpp.MapType.MINI, seed=91, map_seed=73)
    space = cpp.FlatActionSpace(2, cpp.MapType.MINI)
    search = cpp.MCTSSearch(game, space, LAYOUT, seed=7)
    logits = np.zeros(len(space), dtype=np.float32)
    search.initialize_root(logits)
    leaf = np.zeros(search.observation_size, dtype=np.float32)
    for _ in range(16):
        player = search.select_leaf(leaf)
        if player is not None:
            assert search.has_pending_leaf
            search.evaluate_leaf(logits, 0.0)
    visits = search.root_visits()
    assert visits.dtype == np.uint32 and visits.sum() == 16
    assert np.isnan(search.root_action_values()[visits == 0]).all()
    assert search.metrics().simulations == 16
    with pytest.raises(ValueError, match="wrong size"):
        search.select_leaf(np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="flat action space"):
        search.initialize_root(np.zeros(3, dtype=np.float32))
