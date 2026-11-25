"""OpenHex's forms of two rules: ``GameRules.win_on_turn`` and ``openhex_roads``."""

from __future__ import annotations

import random

import pytest

import cppanatron as cpp

RED, BLUE = cpp.Color.RED, cpp.Color.BLUE


def _play(game: cpp.Game, action_type: cpp.ActionType, value=..., **outcome) -> None:
    game.execute(next(a for a in game.playable_actions
                      if a.action_type == action_type and (value is ... or a.value == value)), **outcome)


@pytest.mark.parametrize("seating", [(BLUE, RED), (RED, BLUE)])
@pytest.mark.parametrize("recount", [False, True])
def test_a_cut_into_a_tie_leaves_the_award_with_its_holder(seating, recount):
    """RED holds Longest Road with 7; BLUE's settlement cuts it to 5, tying BLUE's 5.
    OpenHex: the holder keeps it. Catanatron: whoever built a road first takes it."""
    layout = pytest.importorskip("test_longest_road_ties")
    red, blue, (s_red, x_red), (s_blue, x_blue) = layout.find_layout()
    road = layout.road
    first = {RED: (red[0], road(red[0], red[1])), BLUE: (blue[0], road(blue[0], blue[1]))}
    second = {RED: (s_red, road(s_red, x_red)), BLUE: (s_blue, road(s_blue, x_blue))}
    game = cpp.Game(list(seating), map_type=cpp.MapType.TOURNAMENT,
                    rules=cpp.GameRules(openhex_roads=recount))
    for color in (*seating, *seating[::-1]):
        node, edge = (first if not game.player(color).settlements else second)[color]
        _play(game, cpp.ActionType.BUILD_SETTLEMENT, node)
        _play(game, cpp.ActionType.BUILD_ROAD, edge)

    def turn(color, moves):
        assert game.current_color == color
        _play(game, cpp.ActionType.ROLL, dice=(1, 1))
        for action_type, value in moves:
            _play(game, action_type, value)
        _play(game, cpp.ActionType.END_TURN)

    if seating[0] == BLUE:
        turn(BLUE, [])
    game.set_resources(RED, (6, 6, 0, 0, 0))
    turn(RED, [(cpp.ActionType.BUILD_ROAD, road(a, b)) for a, b in zip(red[1:], red[2:])])
    assert game.player(RED).has_road
    game.set_resources(BLUE, (5, 5, 1, 1, 0))
    turn(BLUE, [(cpp.ActionType.BUILD_ROAD, road(a, b)) for a, b in zip(blue[1:], blue[2:])]
         + [(cpp.ActionType.BUILD_SETTLEMENT, red[5])])
    assert game.player(RED).longest_road_length == game.player(BLUE).longest_road_length == 5
    holder = RED if recount else seating[0]
    assert game.player(holder).has_road and not game.player(BLUE if holder == RED else RED).has_road


def test_only_the_seat_on_turn_wins_under_win_on_turn():
    rules = cpp.GameRules(vps_to_win=5, win_on_turn=True, openhex_roads=True)
    assert rules.win_on_turn and "win_on_turn=True" in repr(rules) and not cpp.GameRules().win_on_turn
    waited = 0
    for seed in range(2000):
        rng = random.Random(seed)
        game = cpp.Game(4, seed=seed, rules=rules)
        for step in range(3000):
            turn = game.current_turn_index
            reached = [s for s in range(4) if game.player(s).actual_victory_points >= 5 and not game.has_left(s)]
            if game.is_initial_build_phase or turn not in reached:
                assert game.winning_color is None
                waited += bool(reached)
                for s in range(4):
                    public = game.player(s).victory_points
                    assert game.points_short_of_win(s) == (public if s == turn and not game.has_left(s) else 0)
            else:
                assert game.winning_color == game.colors[turn]
                break
            if rng.random() < 0.01 and len(game.colors_in_play) > 2:
                game.leave(rng.choice([game.colors.index(c) for c in game.colors_in_play]))
                continue
            game.execute(rng.choice(game.playable_actions))
    assert waited, waited  # a seat had the points on another's turn, and the game went on


def _openhex_length(game: cpp.Game, color) -> int:
    """OpenHex's _compute_longest_road over the engine's board."""
    roads = [tuple(e) for e, c in dict(game.board.roads).items() if c == color]
    blocked = {n for n, (c, _b) in dict(game.board.buildings).items() if c != color}
    adj: dict[int, list[int]] = {}
    for a, b in roads:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    best = 0

    def walk(node, used, length):
        nonlocal best
        best = max(best, length)
        for other in adj[node]:
            edge = (min(node, other), max(node, other))
            if edge in used:
                continue
            if other in blocked:
                best = max(best, length + 1)
            else:
                walk(other, used | {edge}, length + 1)

    for start in adj:
        walk(start, frozenset(), 0)
    return best


def test_roads_and_the_award_follow_openhex_in_random_games():
    """Every length is OpenHex's count; the award moves only as its rule says."""
    longer = changes = 0
    for seed in range(150):
        rng = random.Random(seed)
        game = cpp.Game(4, seed=seed, rules=cpp.GameRules(vps_to_win=14, openhex_roads=True))
        reference = cpp.Game(4, seed=seed, rules=cpp.GameRules(vps_to_win=14))
        holder = None
        while game.winning_color is None and game.num_turns < 400:
            action = rng.choice(game.playable_actions)
            game.execute(action)
            if reference is not None:
                try:
                    reference.execute(action)
                except ValueError:
                    reference = None
            if action.action_type not in (cpp.ActionType.BUILD_ROAD, cpp.ActionType.BUILD_SETTLEMENT):
                continue
            lengths = {c: _openhex_length(game, c) for c in game.colors}
            assert {c: game.board.longest_road(c) for c in game.colors} == lengths
            if reference is not None:
                longer += any(reference.player(c).longest_road_length < lengths[c] for c in game.colors)
            top = max(lengths.values())
            tied = [c for c in game.colors if lengths[c] == top]
            expected = None if top < 5 else tied[0] if len(tied) == 1 else holder if holder in tied else None
            now = next((c for c in game.colors if game.player(c).has_road), None)
            assert now == expected
            changes += now != holder
            holder = now
    assert longer and changes
