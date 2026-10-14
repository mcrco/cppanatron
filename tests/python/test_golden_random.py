"""Golden seeded results: every draw goes through cppanatron's portable
samplers (include/cppanatron/random.hpp), so a seed gives these exact maps,
games, schedules and worlds with any compiler or standard library. The same
game is pinned in tests/test_random.cpp, which CMake builds with the system
compiler while this extension may use another.

A change here means seeded results changed: only accept it on purpose.
"""

from __future__ import annotations

import hashlib
import json

import cppanatron as cpp

ABBREVIATIONS = {None: "--", **{r: r.name[:2] for r in cpp.Resource.__members__.values()}}


def test_a_seeded_game_plays_out_identically():
    game = cpp.Game(4, seed=2024, number_placement=cpp.NumberPlacement.RANDOM)
    land = list(game.board.map.land_tiles.values())
    assert " ".join(ABBREVIATIONS[t.resource] for t in land) == (
        "WH WH BR BR WO WO WH BR WO OR SH OR -- SH OR SH WH SH WO")
    assert [t.number for t in land] == [
        10, 6, 11, 5, 4, 9, 3, 5, 10, 8, 11, 2, None, 3, 12, 9, 6, 4, 8]
    steps = 0
    while game.winning_color is None:
        actions = game.playable_actions
        game.execute(actions[(steps * 7919) % len(actions)])
        steps += 1
    assert (steps, game.num_turns, game.winning_color) == (710, 245, cpp.Color.ORANGE)
    assert game.resource_bank == (14, 18, 11, 13, 17)
    assert game.development_cards_remaining == 13
    assert [p.actual_victory_points for p in game.players] == [2, 2, 2, 10]
    assert [p.resources for p in game.players] == [
        (1, 0, 1, 4, 0), (0, 1, 3, 1, 2), (3, 0, 4, 1, 0), (1, 0, 0, 0, 0)]
    assert game.board.robber_coordinate == (-2, 1, 1)


def test_a_seeded_map_is_pinned():
    spec = cpp.MapSpec.build(cpp.MapType.BASE, 5, cpp.NumberPlacement.RANDOM).to_dict()
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    assert digest == "cd3bdaaae517b73e64003c9d97785bb38a308c5f3b1c4b260905325781fa34f3"


def test_vec_env_draws_are_pinned():
    env = cpp.VecEnv(
        [cpp.Scenario(num_players=2),
         cpp.Scenario(num_players=4, seats=["learner", "R", "F", "R"], shuffle_seats=True,
                      map_type=cpp.MapType.MINI)],
        num_envs=4, seed=9)
    env.allocate()
    env.reset()
    assert [(env.scenario_index(e), env.game_seed(e), env.bot_seed(e), env.lineup_slots(e))
            for e in range(4)] == [
        (1, 14758323655161656236, 12958943884061812320, [3, 2, 1, 0]),
        (1, 2158665653953809831, 2998591474292478537, [3, 2, 0, 1]),
        (0, 16182201733286232505, 6291641560611147241, [0, 1]),
        (0, 45708763756162269, 7302301321936236894, [0, 1]),
    ]


def test_a_sampled_world_is_pinned():
    game = cpp.Game(3, seed=77)
    for step in range(300):
        actions = game.playable_actions
        game.execute(actions[(step * 31) % len(actions)])
    world = cpp.sample_world(game, cpp.HandTracker(game, 0), seed=3)
    assert [p.resources for p in world.players] == [
        (0, 0, 1, 1, 1), (1, 0, 1, 0, 2), (0, 1, 1, 2, 1)]
    assert [p.development_cards for p in world.players] == [
        (0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1)]
    assert world.development_cards_remaining == 17
