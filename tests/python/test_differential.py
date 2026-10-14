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
from map_shapes import SHAPES  # noqa: E402

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


#: The parity suite seats at most four players on MINI's 24 nodes.
PLAYER_COUNTS_AND_MAPS = [
    (num_players, map_type)
    for num_players in (2, 3, 4, 5, 6)
    for map_type in ("MINI", "BASE", "TOURNAMENT")
    if num_players <= 4 or map_type != "MINI"
]

#: The 5-6 player extension's bank and development deck.
EXTENSION_SUPPLY = dict(resource_bank=(24, 24, 24, 24, 24), development_deck=(20, 3, 3, 3, 5))


def play_in_lockstep(num_players: int, seed: int, **options) -> None:
    reference = new_game("catanatron", num_players, seed=seed, **options)
    native = new_game("cppanatron", num_players, seed=seed, **options)
    rng = random.Random(seed)

    for step in range(MAX_STEPS):
        snapshot = reference.snapshot()
        assert_same_snapshot(snapshot, native.snapshot(), f"seed {seed}, step {step}")
        if snapshot.winning_color is not None:
            break
        action = rng.choice(sorted(snapshot.playable_actions, key=repr))
        outcome = reference.execute(action)
        native.execute(action, outcome)


@pytest.mark.parametrize("number_placement", ["official_spiral", "random"])
@pytest.mark.parametrize(("num_players", "map_type"), PLAYER_COUNTS_AND_MAPS)
@pytest.mark.parametrize("game_index", range(GAMES_PER_CONFIGURATION))
def test_random_games_stay_in_lockstep(num_players, map_type, number_placement, game_index):
    seed = 1_000 * num_players + 97 * game_index + len(map_type) + len(number_placement)
    play_in_lockstep(
        num_players, seed, map_type=map_type, number_placement=number_placement, vps_to_win=8
    )


@pytest.mark.parametrize("num_players", [5, 6])
@pytest.mark.parametrize("game_index", range(GAMES_PER_CONFIGURATION))
def test_extension_supply_games_stay_in_lockstep(num_players, game_index):
    """A bigger bank and deck change scarcity, draws and when the deck runs out."""
    seed = 3_000 * num_players + 97 * game_index
    play_in_lockstep(num_players, seed, map_type="BASE", vps_to_win=8, **EXTENSION_SUPPLY)


#: Explicit maps: the templates as specs, and shapes no template produces.
EXPLICIT_MAPS = {
    "BASE": lambda seed: cpp.MapSpec.build(cpp.MapType.BASE, seed).to_dict(),
    "MINI": lambda seed: cpp.MapSpec.build(cpp.MapType.MINI, seed).to_dict(),
    "TOURNAMENT": lambda seed: cpp.MapSpec.build(cpp.MapType.TOURNAMENT).to_dict(),
    **{name: builder for name, (builder, _) in SHAPES.items()},
}
EXPLICIT_MAP_SEATS = {"MINI": 4, **{name: seats for name, (_, seats) in SHAPES.items()}}


@pytest.mark.parametrize(
    ("num_players", "map_name"),
    [
        (num_players, name)
        for name in EXPLICIT_MAPS
        for num_players in range(2, EXPLICIT_MAP_SEATS.get(name, 6) + 1)
    ],
)
@pytest.mark.parametrize("game_index", range(GAMES_PER_CONFIGURATION))
def test_explicit_map_games_stay_in_lockstep(num_players, map_name, game_index):
    seed = 5_000 + 1_000 * num_players + 97 * game_index + len(map_name)
    spec = EXPLICIT_MAPS[map_name](seed)
    play_in_lockstep(
        num_players,
        seed,
        map_spec=spec,
        vps_to_win=8,
        resource_bank=(24,) * 5 if num_players > 4 else None,
    )


@pytest.mark.parametrize("enemies", ["next", "all"])
@pytest.mark.parametrize(("num_players", "map_type"), PLAYER_COUNTS_AND_MAPS)
def test_the_value_expert_matches_catanatrons_value_function_player(
    num_players, map_type, enemies
):
    """The ported heuristic scores every position like Catanatron's base_fn, and
    the native expert picks an action Catanatron values as highly."""
    seed = 7_000 + num_players + len(map_type)
    options = dict(map_type=map_type, seed=seed, vps_to_win=6)
    reference = new_game("catanatron", num_players, **options)
    native = new_game("cppanatron", num_players, **options)
    rng = random.Random(seed)
    value = base_fn(enemies=enemies)
    native_enemies = cpp.ValueEnemies[enemies.upper()]

    for step in range(400):
        snapshot = reference.snapshot()
        if snapshot.winning_color is not None:
            break
        for color in snapshot.colors:
            native_value = cpp.value_score(
                native.game, from_catanatron_color(color), enemies=native_enemies
            )
            assert native_value == pytest.approx(
                value(reference.game, color), rel=1e-14, abs=1e-6
            ), f"seed {seed}, step {step}: value of {color} differs"
        if not any(a.action_type in CHANCE_ACTIONS for a in snapshot.playable_actions):
            expert = ValueFunctionPlayer(
                snapshot.current_color, ValueFunctionPlayer.Params(enemies=enemies)
            )
            expected = expert.decide(reference.game, reference.game.playable_actions)
            chosen = to_catanatron_action(cpp.value_action(native.game, enemies=native_enemies))
            if chosen != expected:  # Equal-valued actions may break ties differently.
                after_expected, after_chosen = reference.game.copy(), reference.game.copy()
                after_expected.execute(expected)
                after_chosen.execute(chosen)
                assert value(after_chosen, chosen.color) == pytest.approx(
                    value(after_expected, expected.color), rel=1e-14, abs=1e-6
                ), f"seed {seed}, step {step}: expert chose {chosen}, reference {expected}"
        action = rng.choice(sorted(snapshot.playable_actions, key=repr))
        native.execute(action, reference.execute(action))
