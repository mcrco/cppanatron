"""GraphEncoder: layout, seat-relative features, topology, mask, and no hidden-info leaks."""

from __future__ import annotations

import random

import numpy as np
import pytest

import cppanatron as cpp
from map_shapes import SHAPES

DIMS = cpp.EncoderDims(max_players=6, max_tiles=40, max_nodes=100, max_edges=140)
LEVELS = [cpp.InfoLevel.PRIVATE, cpp.InfoLevel.PUBLIC, cpp.InfoLevel.FULL]
#: Player features only some levels may read for other seats.
HAND = {f"resource_{r}" for r in ("wood", "brick", "sheep", "wheat", "ore")}
CARDS = (
    {"actual_vps"}
    | {f"dev_{d}" for d in ("knight", "year_of_plenty", "monopoly", "road_building",
                            "victory_point")}
    | {f"dev_playable_{d}" for d in ("knight", "year_of_plenty", "monopoly", "road_building")}
)


def blocks(encoder: cpp.GraphEncoder, row: np.ndarray) -> dict:
    return {
        name: row[offset: offset + int(np.prod(shape))].reshape(shape)
        for name, (offset, shape) in encoder.layout().items()
    }


def random_game(num_players: int, seed: int, steps: int, catan_map=None) -> cpp.Game:
    rng = random.Random(seed)
    game = (
        cpp.Game(num_players, map=catan_map, seed=seed)
        if catan_map is not None
        else cpp.Game(num_players, seed=seed)
    )
    for _ in range(steps):
        if game.winning_color is not None:
            break
        game.execute(rng.choice(game.playable_actions))
    return game


def test_layout_is_contiguous_and_named():
    encoder = cpp.GraphEncoder(DIMS)
    layout = encoder.layout()
    assert list(layout) == ["global", "players", "tiles", "nodes", "edges", "edge_nodes",
                            "tile_nodes", "action_mask"]
    offset = 0
    for name, (start, shape) in layout.items():
        assert start == offset, name
        offset += int(np.prod(shape))
    assert offset == encoder.size == len(encoder)
    names = encoder.feature_names()
    assert layout["global"][1] == (len(names["global"]),)
    assert layout["players"][1] == (6, len(names["players"]))
    assert layout["tiles"][1] == (40, len(names["tiles"]))
    assert layout["nodes"][1] == (100, len(names["nodes"]))
    assert layout["edges"][1] == (140, len(names["edges"]))
    assert layout["edge_nodes"][1] == (140, 2) and layout["tile_nodes"][1] == (40, 6)
    assert layout["action_mask"][1] == (len(encoder.action_space),)
    p = DIMS.max_players
    assert (len(names["global"]), len(names["players"]), len(names["tiles"]),
            len(names["nodes"]), len(names["edges"])) == (23 + p, 46, 19 + p, 16 + 3 * p, 2 + p)
    for block in names.values():
        assert len(set(block)) == len(block)


@pytest.mark.parametrize("shape", ["lake", "islands", "strip"])
@pytest.mark.parametrize("num_players", [2, 4])
def test_topology_validity_and_padding(shape, num_players):
    catan_map = cpp.CatanMap.from_spec(SHAPES[shape][0](1))
    game = random_game(num_players, 3, 200, catan_map)
    encoder = cpp.GraphEncoder(DIMS)
    parts = blocks(encoder, encoder.encode(game, 0))
    slot = {node: index for index, node in enumerate(catan_map.node_ids)}
    edges = catan_map.land_edges
    assert [tuple(row) for row in parts["edge_nodes"][: len(edges)].astype(int)] == [
        (slot[a], slot[b]) for a, b in edges
    ]
    assert (parts["edge_nodes"][len(edges):] == -1).all()
    tiles = list(catan_map.land_tiles.values())
    assert [tuple(row) for row in parts["tile_nodes"][: len(tiles)].astype(int)] == [
        tuple(slot[n] for n in tile.nodes) for tile in tiles
    ]
    assert (parts["tile_nodes"][len(tiles):] == -1).all()
    for name, count in (("players", num_players), ("tiles", len(tiles)),
                        ("nodes", catan_map.num_nodes), ("edges", len(edges))):
        assert (parts[name][:count, 0] == 1).all(), name
        assert (parts[name][count:] == 0).all(), name


def test_features_are_seat_relative():
    game = random_game(4, 11, 400)
    encoder = cpp.GraphEncoder(DIMS, cpp.InfoLevel.FULL)
    names = encoder.feature_names("players")
    vps, wood, acting = (names.index(n) for n in ("public_vps", "resource_wood", "acting"))
    node_names = encoder.feature_names("nodes")
    settlement_0 = node_names.index("settlement_0")
    for observer in range(4):
        parts = blocks(encoder, encoder.encode(game, observer))
        for slot in range(4):
            seat = (observer + slot) % 4
            player = game.player(seat)
            assert parts["players"][slot, vps] == pytest.approx(player.victory_points / 10)
            assert parts["players"][slot, wood] == pytest.approx(player.resources[0] / 19)
            assert parts["players"][slot, acting] == (seat == game.current_player_index)
        mine = set(game.player(observer).settlements)
        ids = game.board.map.node_ids
        assert {ids[i] for i in np.flatnonzero(parts["nodes"][: len(ids), settlement_0])} == mine
    # Every bounded feature of a mid-game row stays in [0, 1].
    row = encoder.encode(game, 0)
    parts = blocks(encoder, row)
    for name in ("global", "players", "tiles", "nodes", "edges", "action_mask"):
        assert parts[name].min() >= 0 and parts[name].max() <= 1.5, name


@pytest.mark.parametrize("num_players", [2, 3, 5])
def test_the_mask_block_is_the_action_spaces_mask_for_the_seat_to_act(num_players):
    encoder = cpp.GraphEncoder(DIMS)
    space = encoder.action_space
    offset, (size,) = encoder.layout()["action_mask"]
    rng = random.Random(num_players)
    game = cpp.Game(num_players, map=cpp.CatanMap.from_spec(SHAPES["big_hexagon"][0](2)))
    for _ in range(300):
        for observer in range(num_players):
            mask = encoder.encode(game, observer)[offset: offset + size]
            if observer == game.current_player_index:
                assert np.array_equal(mask, space.mask(game).astype(np.float32))
            else:
                assert np.flatnonzero(mask).tolist() == [space.noop]
        game.execute(rng.choice(game.playable_actions))


@pytest.mark.parametrize("num_players", [2, 4])
def test_buildable_edges_follow_the_initial_road_rule(num_players):
    """While a seat places an initial road, its buildable edges are its legal roads."""
    encoder = cpp.GraphEncoder(DIMS)
    space = encoder.action_space
    road_offset, _ = space.layout()["road"]
    edges_offset, (_, width) = encoder.layout()["edges"]
    buildable = encoder.feature_names("edges").index("buildable")
    rng = random.Random(num_players)
    game = cpp.Game(num_players, seed=num_players)
    checked = 0
    while game.is_initial_build_phase:
        if game.current_prompt == cpp.ActionPrompt.BUILD_INITIAL_ROAD:
            seat = game.current_player_index
            row = encoder.encode(game, seat)
            edges = row[edges_offset: edges_offset + DIMS.max_edges * width].reshape(-1, width)
            roads = {space.index(game, a) - road_offset for a in game.playable_actions}
            assert set(np.flatnonzero(edges[:, buildable])) == roads
            checked += 1
        game.execute(rng.choice(game.playable_actions))
    assert checked == 2 * num_players


def test_encode_into_writes_the_same_row():
    encoder = cpp.GraphEncoder(DIMS)
    game = random_game(3, 5, 150)
    out = np.full(encoder.size, 7.0, dtype=np.float32)
    encoder.encode_into(game, 1, out)
    assert np.array_equal(out, encoder.encode(game, 1))
    with pytest.raises(TypeError):
        encoder.encode_into(game, 1, np.zeros(encoder.size, dtype=np.float64))
    with pytest.raises(IndexError):
        encoder.encode(game, 3)


# ------------------------------------------------------------ information levels


def hidden_indices(encoder: cpp.GraphEncoder, level) -> set[int]:
    """Row positions of other seats' features ``level`` must not read."""
    offset, (seats, width) = encoder.layout()["players"]
    hidden = CARDS | (HAND if level == cpp.InfoLevel.PRIVATE else set())
    if level == cpp.InfoLevel.FULL:
        hidden = set()
    names = encoder.feature_names("players")
    return {
        offset + slot * width + names.index(name)
        for slot in range(1, seats)
        for name in hidden
    }


def development_pool(game: cpp.Game, seat: int) -> list[int]:
    """Cards ``seat`` could be holding: its hand plus the deck."""
    pool = list(game.rules.development_deck)
    for player in game.players:
        for card in range(5):
            pool[card] -= player.development_cards[card] + player.played_development_cards[card]
    hand = game.player(seat).development_cards
    return [pool[card] + hand[card] for card in range(5)]


def other_hand(hand, pool) -> tuple[int, ...] | None:
    """A different hand of the same size drawable from ``pool``."""
    for give in range(5):
        for take in range(5):
            if give != take and hand[give] > 0 and hand[take] < pool[take]:
                changed = list(hand)
                changed[give] -= 1
                changed[take] += 1
                return tuple(changed)
    return None


@pytest.mark.parametrize("num_players", [2, 4, 6])
def test_hidden_information_never_reaches_rows_that_may_not_see_it(num_players):
    """Swapping what a level cannot see leaves its rows unchanged, and the
    levels differ only in the features documented as hidden."""
    encoders = {level: cpp.GraphEncoder(DIMS, level) for level in LEVELS}
    hidden = {level: hidden_indices(encoders[level], level) for level in LEVELS}
    rng = random.Random(num_players)
    checked_hands = checked_cards = 0
    for game_index in range(3):
        game = cpp.Game(num_players, seed=game_index, rules=cpp.GameRules(
            resource_bank=(24,) * 5, development_deck=(20, 3, 3, 3, 5), vps_to_win=12))
        for step in range(1_500):
            if game.winning_color is not None:
                break
            if step % 25 == 0:
                for observer in range(num_players):
                    rows = {level: encoders[level].encode(game, observer) for level in LEVELS}
                    private, full = rows[cpp.InfoLevel.PRIVATE], rows[cpp.InfoLevel.FULL]
                    assert set(np.flatnonzero(private != full)) <= hidden[cpp.InfoLevel.PRIVATE]
                    assert not private[list(hidden[cpp.InfoLevel.PRIVATE])].any()

                    for other in range(num_players):
                        if other == observer:
                            continue
                        hand = game.player(other).resources
                        rotated = hand[1:] + hand[:1]
                        if rotated != hand:
                            swapped = game.copy()
                            swapped.set_resources(other, rotated)
                            for level in LEVELS:
                                same = np.array_equal(
                                    encoders[level].encode(swapped, observer), rows[level]
                                )
                                assert same == (level == cpp.InfoLevel.PRIVATE), level
                            checked_hands += 1
                        cards = other_hand(game.player(other).development_cards,
                                           development_pool(game, other))
                        if cards is not None:
                            dealt = game.copy()
                            dealt.determinize_development_cards(other, cards, seed=step)
                            if dealt.winning_color is not None:
                                continue
                            for level in LEVELS:
                                same = np.array_equal(
                                    encoders[level].encode(dealt, observer), rows[level]
                                )
                                assert same == (level != cpp.InfoLevel.FULL), level
                            checked_cards += 1
            game.execute(rng.choice(game.playable_actions))
    assert checked_hands > 20 and checked_cards > 5
