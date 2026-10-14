"""GraphActionSpace: every playable action has one index that decodes back to it."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

import cppanatron as cpp
from map_shapes import SHAPES

TRADES = {
    cpp.ActionType.OFFER_TRADE,
    cpp.ActionType.ACCEPT_TRADE,
    cpp.ActionType.REJECT_TRADE,
    cpp.ActionType.CONFIRM_TRADE,
    cpp.ActionType.CANCEL_TRADE,
}

MAPS = {
    "BASE": lambda seed: cpp.CatanMap.build(cpp.MapType.BASE, seed),
    "MINI": lambda seed: cpp.CatanMap.build(cpp.MapType.MINI, seed),
    "TOURNAMENT": lambda seed: cpp.CatanMap.build(cpp.MapType.TOURNAMENT),
    **{name: (lambda seed, b=builder: cpp.CatanMap.from_spec(b(seed))) for name, (builder, _) in SHAPES.items()},
}
SEATS = {"MINI": 4, **{name: seats for name, (_, seats) in SHAPES.items()}}
#: Big enough for every test map and six seats.
DIMS = cpp.EncoderDims(max_players=6, max_tiles=40, max_nodes=100, max_edges=140)


def test_layout_and_size():
    dims = cpp.EncoderDims(max_players=4, max_tiles=19, max_nodes=54, max_edges=72)
    space = cpp.GraphActionSpace(dims)
    assert len(space) == 95 + 2 * 54 + 72 + 19 * 4 + 1
    assert space.noop == len(space) - 1
    layout = space.layout()
    assert layout["globals"] == (0, (95,))
    assert layout["settlement"] == (95, (54,))
    assert layout["city"] == (95 + 54, (54,))
    assert layout["road"] == (95 + 108, (72,))
    assert layout["robber"] == (95 + 108 + 72, (19, 4))
    assert layout["noop"] == (len(space) - 1, (1,))
    names = space.names()
    assert names[:5] == ["ROLL", "END_TURN", "BUY_DEVELOPMENT_CARD", "PLAY_KNIGHT_CARD",
                         "PLAY_ROAD_BUILDING"]
    assert names[15] == "YEAR_OF_PLENTY_WOOD_WOOD" and names[34] == "YEAR_OF_PLENTY_ORE"
    assert names[35] == "MARITIME_WOOD_BRICK_4" and names[94] == "MARITIME_ORE_WHEAT_2"
    assert names[-1] == "NOOP" and len(set(names)) == len(names)


@pytest.mark.parametrize("num_players", [2, 3, 6])
def test_every_index_decodes_to_an_action_that_indexes_back(num_players):
    """Pure codec round trip, legal or not, over the whole space of one map."""
    space = cpp.GraphActionSpace(DIMS)
    game = cpp.Game(num_players, map=MAPS["big_hexagon"](3))
    catan_map = game.board.map
    decoded = 0
    for index in range(len(space)):
        name = space.name(index)
        if name == "NOOP":
            with pytest.raises(ValueError):
                space.decode(game, index)
            continue
        if name.startswith("ROBBER@"):
            tile, victim = (int(x) for x in name[len("ROBBER@"):].split("/"))
            if tile >= catan_map.num_land_tiles or victim >= num_players:
                with pytest.raises(IndexError):
                    space.decode(game, index)
                continue
        for prefix, count in (("SETTLEMENT@", catan_map.num_nodes),
                              ("CITY@", catan_map.num_nodes), ("ROAD@", catan_map.num_edges)):
            if name.startswith(prefix) and int(name[len(prefix):]) >= count:
                with pytest.raises(IndexError):
                    space.decode(game, index)
                break
        else:
            action = space.decode(game, index)
            assert action.color == game.current_color
            assert space.index(game, action) == index, name
            decoded += 1
    assert decoded == 95 + 2 * 96 + 132 + 37 * num_players


def check_position(space: cpp.GraphActionSpace, game: cpp.Game, seen: Counter) -> None:
    mask = space.mask(game)
    assert mask.dtype == np.bool_ and mask.shape == (len(space),)
    expected = set()
    for action in game.playable_actions:
        if action.action_type in TRADES:
            continue
        index = space.index(game, action)
        assert index not in expected, f"two playable actions share index {index}"
        expected.add(index)
        assert space.decode(game, index) == action
        seen[space.name(index).split("@")[0].rstrip("_0123456789")] += 1
    assert set(np.flatnonzero(mask)) == expected
    assert not mask[space.noop]


@pytest.mark.parametrize(
    ("map_name", "num_players"),
    [(name, n) for name in MAPS for n in range(2, SEATS.get(name, 6) + 1)],
)
def test_random_games_index_every_playable_action_uniquely(map_name, num_players):
    space = cpp.GraphActionSpace(DIMS)
    seen: Counter = Counter()
    for game_index in range(2):
        seed = 31 * num_players + game_index + len(map_name)
        rng = random.Random(seed)
        game = cpp.Game(num_players, map=MAPS[map_name](seed), seed=seed,
                        rules=cpp.GameRules(vps_to_win=8))
        for _ in range(1_500):
            if game.winning_color is not None:
                break
            check_position(space, game, seen)
            game.execute(rng.choice(game.playable_actions))
    assert seen["SETTLEMENT"] and seen["ROAD"] and seen["ROLL"] and seen["END_TURN"]


def test_single_card_year_of_plenty_when_the_bank_lacks_a_pair():
    rules = cpp.GameRules(resource_bank=(3, 1, 0, 0, 0))
    game = cpp.Game(3, map_type=cpp.MapType.TOURNAMENT, rules=rules)
    game.apply_setup(
        [
            cpp.PlayerSetup(settlements=[1], roads=[(1, 6)], development_cards=(0, 1, 0, 0, 0)),
            cpp.PlayerSetup(settlements=[3], roads=[(3, 4)]),
            cpp.PlayerSetup(settlements=[42]),
        ]
    )
    space = cpp.GraphActionSpace(DIMS)
    seen: Counter = Counter()
    check_position(space, game, seen)
    names = {space.name(i) for i in np.flatnonzero(space.mask(game))}
    assert {"YEAR_OF_PLENTY_WOOD", "YEAR_OF_PLENTY_BRICK", "YEAR_OF_PLENTY_WOOD_WOOD",
            "YEAR_OF_PLENTY_WOOD_BRICK"} <= names


def test_robber_victims_are_relative_seats():
    """WHITE (seat 2) plays a knight: seats 3, 0 and 1 are victim slots 1, 2 and 3."""
    space = cpp.GraphActionSpace(DIMS)
    game = cpp.Game(4, map_type=cpp.MapType.TOURNAMENT, seed=3)
    game.apply_setup(
        [
            cpp.PlayerSetup(settlements=[1], roads=[(1, 6)], resources=(1, 0, 0, 0, 0)),
            cpp.PlayerSetup(settlements=[3], roads=[(3, 4)], resources=(1, 0, 0, 0, 0)),
            cpp.PlayerSetup(settlements=[42], development_cards=(1, 0, 0, 0, 0)),
            cpp.PlayerSetup(settlements=[13], resources=(0, 0, 1, 0, 0)),
        ],
        current_player=2,
    )
    game.execute(next(a for a in game.playable_actions
                      if a.action_type == cpp.ActionType.PLAY_KNIGHT_CARD))
    check_position(space, game, Counter())
    colors = game.colors
    slots = set()
    for action in game.playable_actions:
        _, victim = action.value
        slot = 0 if victim is None else (colors.index(victim) - 2) % 4
        assert space.name(space.index(game, action)).endswith(f"/{slot}")
        slots.add(slot)
    assert slots == {0, 1, 2, 3}


def test_dims_that_do_not_fit_are_refused():
    space = cpp.GraphActionSpace(cpp.EncoderDims(max_players=2, max_tiles=19, max_nodes=54,
                                                 max_edges=72))
    with pytest.raises(ValueError, match="players"):
        space.mask(cpp.Game(3))
    with pytest.raises(ValueError, match="land tiles"):
        space.mask(cpp.Game(2, map=MAPS["big_hexagon"](0)))
    maps = [MAPS["strip"](0), MAPS["lake"](0), MAPS["BASE"](0)]
    dims = cpp.EncoderDims.covering(maps, 4)
    assert (dims.max_tiles, dims.max_nodes, dims.max_edges) == (19, 54, 72)
    assert all(dims.fits(m, 4) for m in maps)
