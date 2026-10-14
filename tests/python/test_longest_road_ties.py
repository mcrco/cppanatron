"""Longest Road when a cut leaves a tie: both engines follow Catanatron.

Catanatron recomputes the holder after a settlement cuts a road as the first
longest entry of its road_lengths dict, i.e. in the order the players built
their first road (seating order in a game played from the start), not in
color order. With a seating that is not color order, the tie goes to the
seat that built first.
"""

from __future__ import annotations

import pytest

pytest.importorskip("catanatron")

from catanatron.models.enums import ActionType  # noqa: E402
from catanatron.models.player import Color  # noqa: E402

import cppanatron as cpp  # noqa: E402
from cppanatron.parity import ENGINES, new_game  # noqa: E402

MAP = cpp.CatanMap.build(cpp.MapType.TOURNAMENT)
NEIGHBORS = {node: set(MAP.neighbors(node)) for node in MAP.node_ids}


def paths(start: int, edges: int, avoid: frozenset = frozenset()):
    """Simple paths of ``edges`` edges from ``start`` avoiding ``avoid``."""
    stack = [[start]]
    while stack:
        path = stack.pop()
        if len(path) == edges + 1:
            yield path
            continue
        for neighbor in sorted(NEIGHBORS[path[-1]]):
            if neighbor not in path and neighbor not in avoid:
                stack.append(path + [neighbor])


def far(node: int, others) -> bool:
    return all(node != o and node not in NEIGHBORS[o] for o in others)


def find_layout():
    """RED's 7-road path r0..r7 and BLUE's 5-road path b0..b5 = r5 into it,
    plus a far second settlement (and road) for each, all legal. BLUE's
    settlement at r5 leaves RED a longest road of 5 (r5 back to r0: a walk
    may start at an enemy building, not enter one), tying BLUE's 5."""
    for r0 in MAP.node_ids:
        for red in paths(r0, 7):
            cut = red[5]
            for reverse in paths(cut, 5, frozenset(red)):
                blue = reverse[::-1]
                b0 = blue[0]
                if not far(r0, [b0]) or not far(cut, [r0, b0]):
                    continue
                used = set(red) | set(blue)
                spare = [n for n in MAP.node_ids
                         if n not in used and all(m not in used for m in NEIGHBORS[n])]
                for s_red in spare:
                    if not far(s_red, [r0, b0, cut]):
                        continue
                    for s_blue in spare:
                        if not far(s_blue, [r0, b0, cut, s_red]):
                            continue
                        x_red = min(NEIGHBORS[s_red] - used)
                        x_blue = min(NEIGHBORS[s_blue] - used - {x_red})
                        return red, blue, (s_red, x_red), (s_blue, x_blue)
    raise AssertionError("no layout on this map")


def road(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


@pytest.mark.parametrize(
    ("seating", "holder"),
    [((Color.BLUE, Color.RED), Color.BLUE), ((Color.RED, Color.BLUE), Color.RED)],
)
def test_a_cut_into_a_tie_awards_the_first_road_builder(seating, holder):
    red, blue, (s_red, x_red), (s_blue, x_blue) = find_layout()
    first = {Color.RED: (red[0], road(red[0], red[1])), Color.BLUE: (blue[0], road(blue[0], blue[1]))}
    second = {Color.RED: (s_red, road(s_red, x_red)), Color.BLUE: (s_blue, road(s_blue, x_blue))}
    snapshots = []
    for engine in ENGINES:
        game = new_game(engine, list(seating), map_type="TOURNAMENT")
        for color in (seating[0], seating[1]):
            game.play(ActionType.BUILD_SETTLEMENT, first[color][0])
            game.play(ActionType.BUILD_ROAD, first[color][1])
        for color in (seating[1], seating[0]):
            game.play(ActionType.BUILD_SETTLEMENT, second[color][0])
            game.play(ActionType.BUILD_ROAD, second[color][1])

        def turn(color, moves):
            assert game.current_color == color
            game.play(ActionType.ROLL, outcome=(1, 1))
            for action_type, value in moves:
                game.play(action_type, value)
            game.play(ActionType.END_TURN)

        def red_roads():
            game.set_resources(Color.RED, (6, 6, 0, 0, 0))
            return [(ActionType.BUILD_ROAD, road(a, b)) for a, b in zip(red[1:], red[2:])]

        def blue_cut():
            game.set_resources(Color.BLUE, (5, 5, 1, 1, 0))
            return [(ActionType.BUILD_ROAD, road(a, b)) for a, b in zip(blue[1:], blue[2:])] + [
                (ActionType.BUILD_SETTLEMENT, red[5])]

        if seating[0] == Color.BLUE:
            turn(Color.BLUE, [])
        turn(Color.RED, red_roads())
        assert game.player(Color.RED).has_road and game.player(Color.RED).longest_road_length == 7
        turn(Color.BLUE, blue_cut())
        assert game.player(Color.RED).longest_road_length == 5
        assert game.player(Color.BLUE).longest_road_length == 5
        assert game.player(holder).has_road, engine
        snapshots.append(game.snapshot())
    assert snapshots[0] == snapshots[1]
