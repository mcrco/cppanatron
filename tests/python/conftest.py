from __future__ import annotations

import pytest

pytest.importorskip("catanatron")

from catanatron.models.enums import ActionPrompt, ActionType  # noqa: E402
from catanatron.models.player import Color  # noqa: E402

from cppanatron.parity import ENGINES, ParityGame, new_game  # noqa: E402

RED, BLUE, WHITE = Color.RED, Color.BLUE, Color.WHITE

#: TOURNAMENT is fully fixed, so every node below means the same place on both
#: engines. Snake order: RED, BLUE, BLUE, RED.
#:   RED:  node 1 (ORE 6, WHEAT 5) and node 20 (ORE 4, WHEAT 5, SHEEP 3)
#:   BLUE: node 3 (WOOD 3, ORE 11) and node 13 (ORE 11, SHEEP 4, BRICK 8)
TWO_PLAYER_SETUP = [(1, (1, 6)), (3, (3, 4)), (13, (13, 14)), (20, (20, 22))]
#: WHITE joins on the sheep coast: node 42 (WOOD 5) and node 49 (SHEEP 3, SHEEP 8).
THREE_PLAYER_SETUP = [
    (1, (1, 6)), (3, (3, 4)), (42, None), (49, None), (13, (13, 14)), (20, (20, 22)),
]


@pytest.fixture(params=ENGINES)
def engine(request) -> str:
    return request.param


def setup_game(
    engine: str,
    placements=TWO_PLAYER_SETUP,
    players=(RED, BLUE),
    **options,
) -> ParityGame:
    """A TOURNAMENT game past the initial build phase, RED to roll."""
    options.setdefault("map_type", "TOURNAMENT")
    game = new_game(engine, list(players), **options)
    for node, road in placements:
        game.play(ActionType.BUILD_SETTLEMENT, node)
        if road is None:  # Any road will do.
            game.execute(game.playable_actions[0])
        else:
            game.play(ActionType.BUILD_ROAD, tuple(sorted(road)))
    assert game.current_prompt == ActionPrompt.PLAY_TURN
    return game


def next_turn(game: ParityGame, dice=(1, 1)) -> None:
    """End the current turn and roll for the next player (2 pays only WHEAT 2)."""
    game.play(ActionType.END_TURN)
    game.play(ActionType.ROLL, outcome=dice)
