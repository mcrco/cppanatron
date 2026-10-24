"""Balanced dice (OpenHex's 1v1 dice) through the Python bindings: the rules
switch, the state, replayed rolls, declared positions and searches."""

from __future__ import annotations

import random
from fractions import Fraction

import numpy as np
import pytest

import cppanatron as cpp
from cppanatron.catanatron import observation_layout

TWO_D6 = tuple(Fraction(6 - abs(7 - k), 36) for k in range(2, 13))
BALANCED = cpp.GameRules(dice_mode=cpp.DiceMode.BALANCED)


def roll_action(game: cpp.Game) -> cpp.Action:
    return next(a for a in game.playable_actions if a.action_type == cpp.ActionType.ROLL)


def to_next_roll(game: cpp.Game) -> cpp.Game:
    """End turns when possible, else play the first action, until someone must roll."""
    while not (game.current_prompt == cpp.ActionPrompt.PLAY_TURN
               and any(a.action_type == cpp.ActionType.ROLL for a in game.playable_actions)):
        actions = game.playable_actions
        ends = [a for a in actions if a.action_type == cpp.ActionType.END_TURN]
        game.execute(ends[0] if ends else actions[0])
    return game


def test_the_rules_switch():
    assert cpp.GameRules().dice_mode == cpp.DiceMode.RANDOM
    assert BALANCED.dice_mode == cpp.DiceMode.BALANCED
    assert BALANCED != cpp.GameRules()
    assert "dice_mode" not in repr(cpp.GameRules()) and "dice_mode=BALANCED" in repr(BALANCED)
    assert cpp.Game(2).balanced_dice is None
    game = cpp.Game(2, rules=BALANCED)
    assert game.dice_mode == cpp.DiceMode.BALANCED and game.rules == BALANCED
    assert game.balanced_dice == cpp.BalancedDice.fresh(2)
    for rules in (cpp.GameRules(), BALANCED):
        probabilities = cpp.Game(3, rules=rules).roll_probabilities(1)
        assert probabilities == pytest.approx([float(p) for p in TWO_D6], abs=1e-15)


def test_the_state_by_hand():
    dice = cpp.BalancedDice.fresh(2)
    assert dice.remaining == 36 and len(dice.deck) == 36 and dice.counts == (1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1)
    after = dice.apply((3, 4), 0)
    assert dice.remaining == 36, "apply returns a new state"
    assert after.remaining == 35 and (3, 4) not in after.deck
    assert after.recent == (7,) and after.sevens == (1, 0)
    assert (after.streak_player, after.streak_count) == (0, 1)
    assert after.seven_multiplier(1) == (7, 5)
    assert after.sum_probabilities(1)[5] == pytest.approx(231 / 1731, abs=1e-15)
    assert after.sum_probabilities(0)[5] == pytest.approx(99 / 1599, abs=1e-15)
    assert after.card_for_total(7) == (1, 6) and after.can_draw((4, 3)) and not after.can_draw((3, 4))
    with pytest.raises(ValueError, match="not in the balanced deck"):
        after.apply((3, 4), 1)
    uneven = cpp.BalancedDice(sevens=(3, 1))
    assert uneven.seven_multiplier(0) == (10, 20) and uneven.sum_probabilities(1)[5] == pytest.approx(9 / 39)
    declared = cpp.BalancedDice(sevens=(1, 0), deck=after.deck, recent=(7,), streak_player=0, streak_count=1)
    assert declared == after
    copy = after.copy()
    assert copy == after and copy is not after


@pytest.mark.parametrize("kwargs", [
    {"sevens": ()},
    {"sevens": (0,) * 7},
    {"recent": (7, 7, 7, 7, 7, 7)},
    {"recent": (13,)},
    {"streak_player": 0},
    {"streak_count": 1},
    {"streak_player": 2, "streak_count": 1},
    {"deck": [(1, 1), (1, 1)]},
    {"deck": [(0, 1)]},
    {"sevens": (-1, 0)},
])
def test_impossible_states_are_refused(kwargs):
    with pytest.raises(ValueError):
        cpp.BalancedDice(**kwargs)


def test_games_draw_from_the_state_and_replays_update_it():
    game = to_next_roll(cpp.Game(2, rules=BALANCED, seed=12))
    assert game.next_roller == game.current_player_index == 0
    game.execute(roll_action(game), dice=(3, 4))
    assert game.balanced_dice.remaining == 35 and game.balanced_dice.sevens == (1, 0)
    to_next_roll(game)
    assert game.next_roller == 1
    assert game.roll_probabilities() == game.balanced_dice.sum_probabilities(1)
    assert game.roll_probabilities(cpp.Color.BLUE) == game.roll_probabilities(1)
    before = game.copy()
    with pytest.raises(ValueError, match="not in the balanced deck"):
        game.execute(roll_action(game), dice=(3, 4))
    assert game.search_equivalent(before)
    game.execute(roll_action(game))
    assert game.balanced_dice.remaining == 34
    # Random dice still take any faces.
    plain = to_next_roll(cpp.Game(2, seed=12))
    plain.execute(roll_action(plain), dice=(3, 4))
    to_next_roll(plain).execute(roll_action(plain), dice=(3, 4))


def test_equivalence_sees_which_card_was_drawn():
    first = to_next_roll(cpp.Game(2, rules=BALANCED, seed=3))
    second = first.copy()
    first.execute(roll_action(first), dice=(1, 6))
    second.execute(roll_action(second), dice=(6, 1))
    assert first.balanced_dice != second.balanced_dice
    assert not first.search_equivalent(second)
    assert first.copy().search_equivalent(first)


def test_positions_declare_the_dice():
    played = to_next_roll(cpp.Game(2, rules=BALANCED, seed=8))
    for card in [(1, 6), (5, 5), (6, 1)]:
        played.execute(roll_action(played), dice=card)
        to_next_roll(played)
    players = [cpp.PlayerSetup(settlements=p.settlements, cities=p.cities, roads=p.roads, resources=p.resources)
               for p in played.players]
    setup = dict(robber=played.board.robber_coordinate, current_player=played.current_player_index,
                 completed_turns=played.completed_turns)
    declared = cpp.Game(2, rules=BALANCED, seed=8)
    declared.apply_setup(players, dice=played.balanced_dice, **setup)
    assert declared.balanced_dice == played.balanced_dice
    assert declared.roll_probabilities() == played.roll_probabilities()
    fresh = cpp.Game(2, rules=BALANCED, seed=8)
    fresh.apply_setup(players, **setup)
    assert fresh.balanced_dice == cpp.BalancedDice.fresh(2)
    with pytest.raises(ValueError, match="balanced dice"):
        cpp.Game(2, seed=8).apply_setup(players, dice=played.balanced_dice, **setup)
    with pytest.raises(ValueError, match="per seat"):
        cpp.Game(2, rules=BALANCED, seed=8).apply_setup(players, dice=cpp.BalancedDice.fresh(3), **setup)


def test_redealing_hidden_cards_keeps_the_dice():
    game = to_next_roll(cpp.Game(2, rules=BALANCED, seed=5))
    game.execute(roll_action(game), dice=(2, 5))
    hands = [cpp.HiddenHand(resources=p.resources, development_cards=p.development_cards,
                            development_card_owned_at_start=p.development_card_owned_at_start)
             for p in game.players]
    redealt = game.copy()
    redealt.redeal_hidden_cards(hands, seed=9)
    assert redealt.balanced_dice == game.balanced_dice


def test_long_run_totals_are_close_to_two_dice():
    totals = np.zeros(11)
    for seed in range(12):
        game = cpp.Game(2, rules=BALANCED, seed=seed)
        rng = random.Random(seed)
        for _ in range(3000):
            if game.winning_color is not None:
                break
            action = rng.choice(game.playable_actions)
            game.execute(action)
            if action.action_type == cpp.ActionType.ROLL:
                totals[game.balanced_dice.recent[-1] - 2] += 1
    frequencies = totals / totals.sum()
    assert totals.sum() > 2000
    np.testing.assert_allclose(frequencies, [float(p) for p in TWO_D6], atol=0.03)


def test_searches_run_on_balanced_games():
    game = cpp.Game(2, rules=BALANCED, seed=21)
    trackers = [cpp.HandTracker(game, seat) for seat in range(2)]
    rng = random.Random(21)
    for _ in range(150):
        action = rng.choice(game.playable_actions)
        before = game.copy()
        game.execute(action)
        for tracker in trackers:
            tracker.observe(before, action, game)
    assert game.winning_color is None and game.balanced_dice.remaining < 36
    codec = cpp.FlatActionCodec(2, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(2, observation_layout())
    configs = [
        cpp.SearchConfig(simulations=64, max_players=2, transpositions=True, max_pending=4),
        cpp.SearchConfig(simulations=64, max_players=2, root=cpp.RootSelection.GUMBEL, determinizations=2),
    ]
    for config in configs:
        pool = cpp.SearchPool(codec, encoder, config)
        tracker = trackers[game.current_player_index] if config.determinizations else None
        search = pool.add(game, 3, tracker)
        observations = np.zeros((8, pool.observation_size), dtype=np.float32)
        ids = np.zeros(8, dtype=np.int64)
        while n := pool.collect(observations, ids):
            for row in range(n):
                assert pool.pending_game(row).dice_mode == cpp.DiceMode.BALANCED
            pool.apply(np.zeros((n, pool.action_size), dtype=np.float32),
                       np.full((n, 2), 0.5, dtype=np.float32))
        assert pool.visits(search).sum() == 64
        assert pool.stats(search).chance_nodes > 0
