"""Random games replayed in lockstep: Catanatron draws, cppanatron follows.

Before every transition both engines must agree on the complete snapshot,
including the exact set of playable actions. Raise CPPANATRON_PARITY_GAMES to
fuzz more seeds per configuration.
"""

from __future__ import annotations

import os
import random

import pytest

pytest.importorskip("catanatron")

from catanatron.players.value import ValueFunctionPlayer, base_fn  # noqa: E402

import cppanatron as cpp  # noqa: E402
from cppanatron.catanatron import from_catanatron_color, to_catanatron_action  # noqa: E402
from cppanatron.parity import CHANCE_ACTIONS, Snapshot, new_game  # noqa: E402

GAMES_PER_CONFIGURATION = int(os.environ.get("CPPANATRON_PARITY_GAMES", "2"))
MAX_STEPS = 3_000


def assert_same_snapshot(expected: Snapshot, actual: Snapshot, context: str) -> None:
    for field in expected.__dataclass_fields__:
        want, got = getattr(expected, field), getattr(actual, field)
        if field == "playable_actions" and want != got:
            pytest.fail(
                f"{context}: playable actions differ; only Catanatron: "
                f"{sorted(want - got, key=repr)}; only cppanatron: {sorted(got - want, key=repr)}"
            )
        assert want == got, f"{context}: {field} differs"


@pytest.mark.parametrize("number_placement", ["official_spiral", "random"])
@pytest.mark.parametrize("map_type", ["MINI", "BASE", "TOURNAMENT"])
@pytest.mark.parametrize("num_players", [2, 3, 4])
@pytest.mark.parametrize("game_index", range(GAMES_PER_CONFIGURATION))
def test_random_games_stay_in_lockstep(num_players, map_type, number_placement, game_index):
    seed = 1_000 * num_players + 97 * game_index + len(map_type) + len(number_placement)
    options = dict(
        map_type=map_type,
        seed=seed,
        number_placement=number_placement,
        vps_to_win=8,
    )
    reference = new_game("catanatron", num_players, **options)
    native = new_game("cppanatron", num_players, **options)
    rng = random.Random(seed)

    for step in range(MAX_STEPS):
        snapshot = reference.snapshot()
        assert_same_snapshot(snapshot, native.snapshot(), f"seed {seed}, step {step}")
        if snapshot.winning_color is not None:
            break
        action = rng.choice(sorted(snapshot.playable_actions, key=repr))
        outcome = reference.execute(action)
        native.execute(action, outcome)


@pytest.mark.parametrize("map_type", ["MINI", "BASE", "TOURNAMENT"])
@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_the_value_expert_matches_catanatrons_value_function_player(num_players, map_type):
    """The ported heuristic scores every position like Catanatron's base_fn, and
    the native expert picks an action Catanatron values as highly."""
    seed = 7_000 + num_players + len(map_type)
    options = dict(map_type=map_type, seed=seed, vps_to_win=6)
    reference = new_game("catanatron", num_players, **options)
    native = new_game("cppanatron", num_players, **options)
    rng = random.Random(seed)
    value = base_fn()

    for step in range(400):
        snapshot = reference.snapshot()
        if snapshot.winning_color is not None:
            break
        for color in snapshot.colors:
            assert cpp.value_score(native.game, from_catanatron_color(color)) == pytest.approx(
                value(reference.game, color), rel=1e-14, abs=1e-6
            ), f"seed {seed}, step {step}: value of {color} differs"
        if not any(a.action_type in CHANCE_ACTIONS for a in snapshot.playable_actions):
            expert = ValueFunctionPlayer(snapshot.current_color)
            expected = expert.decide(reference.game, reference.game.playable_actions)
            chosen = to_catanatron_action(cpp.value_action(native.game))
            if chosen != expected:  # Equal-valued actions may break ties differently.
                after_expected, after_chosen = reference.game.copy(), reference.game.copy()
                after_expected.execute(expected)
                after_chosen.execute(chosen)
                assert value(after_chosen, chosen.color) == pytest.approx(
                    value(after_expected, expected.color), rel=1e-14, abs=1e-6
                ), f"seed {seed}, step {step}: expert chose {chosen}, reference {expected}"
        action = rng.choice(sorted(snapshot.playable_actions, key=repr))
        native.execute(action, reference.execute(action))
