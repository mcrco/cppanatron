"""A seat that leaves the game (OpenHex's leaveGame): ``Game.leave``."""

from __future__ import annotations

import random

import pytest

import cppanatron as cpp

BALANCED = cpp.GameRules(dice_mode=cpp.DiceMode.BALANCED)


def _random_play(game: cpp.Game, rng: random.Random, steps: int) -> None:
    for _ in range(steps):
        if game.winning_color is not None:
            return
        game.execute(rng.choice(game.playable_actions))


def _cards(game: cpp.Game) -> list[int]:
    return [game.resource_bank[k] + sum(p.resources[k] for p in game.players) for k in range(5)]


@pytest.mark.parametrize("seed", range(12))
def test_a_game_goes_on_without_the_seats_that_left(seed):
    rng = random.Random(seed)
    game = cpp.Game(4, seed=seed, rules=BALANCED)
    gone: list[int] = []
    for _ in range(4000):
        if game.winning_color is not None:
            break
        if len(gone) < 2 and rng.random() < (0.05 if game.is_initial_build_phase or game.is_discarding else 0.004):
            seat = rng.choice([s for s in range(4) if s not in gone])
            game.leave(seat)
            gone.append(seat)
            assert sum(game.player(seat).resources) == 0 and game.has_left(seat)
            assert not (game.player(seat).has_road or game.player(seat).has_army)
            assert game.balanced_dice.left == len(gone) and game.balanced_dice.sevens[seat] == 0
        assert game.current_player_index not in gone
        # (The turn stays a leaver's only while the others discard for its 7.)
        assert game.current_turn_index not in gone or game.is_discarding
        assert game.playable_actions
        assert _cards(game) == [19] * 5
        game.execute(rng.choice(game.playable_actions))
    assert [game.colors.index(c) for c in game.colors_in_play] == [s for s in range(4) if s not in gone]
    if game.winning_color is not None:
        assert game.colors.index(game.winning_color) not in gone
    for seat in gone:
        with pytest.raises(ValueError):
            game.leave(seat)


def test_the_initial_placement_skips_a_seat_that_left():
    rng = random.Random(0)
    game = cpp.Game(4, seed=3)
    game.leave(2)  # before its first settlement
    order = []
    while game.is_initial_build_phase:
        if game.current_prompt == cpp.ActionPrompt.BUILD_INITIAL_SETTLEMENT:
            order.append(game.current_player_index)
        if order == [0, 1, 3, 3] and game.current_prompt == cpp.ActionPrompt.BUILD_INITIAL_ROAD:
            game.leave(3)  # its second settlement stands without a road
            continue
        game.execute(rng.choice(game.playable_actions))
    assert order == [0, 1, 3, 3, 1, 0]
    assert game.current_player_index == 0 and game.current_prompt == cpp.ActionPrompt.PLAY_TURN
    with pytest.raises(ValueError, match="two seats"):
        game.leave(0)


def test_the_first_seat_in_play_rolls_first():
    rng = random.Random(1)
    game = cpp.Game(3, seed=5)
    while len(game.player(0).roads) < 1 or game.current_player_index != 0:
        game.execute(rng.choice(game.playable_actions))
    game.leave(0)  # at its second placement, the last of the snake
    assert not game.is_initial_build_phase and game.current_player_index == 1
    assert [a.action_type for a in game.playable_actions] == [cpp.ActionType.ROLL]


def test_a_turn_ends_as_its_seat_leaves_and_its_buildings_produce_nothing():
    rng = random.Random(2)
    game = cpp.Game(3, seed=7)
    while game.is_initial_build_phase:
        game.execute(rng.choice(game.playable_actions))
    assert game.current_player_index == 0
    held = game.player(0).resources
    bank = game.resource_bank
    game.leave(0)
    assert game.resource_bank == tuple(b + h for b, h in zip(bank, held))
    assert game.current_player_index == 1 and game.current_turn_index == 1
    for _ in range(300):
        if game.winning_color is not None:
            break
        game.execute(rng.choice(game.playable_actions))
        assert sum(game.player(0).resources) == 0 and game.current_player_index != 0


def test_a_seven_of_a_seat_that_leaves_ends_after_the_others_discards():
    for seed in range(200):
        rng = random.Random(seed)
        game = cpp.Game(3, seed=seed, rules=cpp.GameRules(discard_limit=2))
        while not game.is_discarding and game.winning_color is None:
            game.execute(rng.choice(game.playable_actions))
        roller = game.current_turn_index
        owing = [s for s in range(3) if game.remaining_discards(s) > 0]
        if game.winning_color is not None or owing == [roller]:
            continue
        game.leave(roller)
        assert game.is_discarding and game.remaining_discards(roller) == 0
        while game.is_discarding:
            assert game.current_player_index != roller
            game.execute(rng.choice(game.playable_actions))
        # No robber move: the next seat in play is to roll.
        assert game.current_turn_index == (roller + 1) % 3
        assert [a.action_type for a in game.playable_actions][0] == cpp.ActionType.ROLL
        return
    pytest.fail("no seven with other seats discarding")


def test_balanced_dice_share_the_sevens_among_the_seats_in_play():
    dice = cpp.BalancedDice(sevens=[3, 1, 2, 0], streak_player=0, streak_count=2)
    game = cpp.Game(4, seed=0, rules=BALANCED)
    assert game.balanced_dice.left == 0
    game.leave(1)
    assert game.balanced_dice.left == 1
    assert dice.left == 0 and dice.seats == 4


def test_a_robber_moved_before_leaving_stays_without_the_steal():
    for seed in range(200):
        rng = random.Random(seed)
        game = cpp.Game(3, seed=seed)
        while game.current_prompt != cpp.ActionPrompt.MOVE_ROBBER and game.winning_color is None:
            game.execute(rng.choice(game.playable_actions))
        robbing = [a for a in game.playable_actions if a.value[1] is not None]
        if game.winning_color is not None or not robbing:
            continue
        mover = game.current_player_index
        tile, victim = robbing[0].value[0], robbing[0].value[1]
        held = game.player(victim).resources
        with pytest.raises(ValueError, match="not to move the robber"):
            game.copy().leave((mover + 1) % 3, robber=tile)
        game.leave(mover, robber=tile)
        assert tuple(game.board.robber_coordinate) == tuple(tile)
        assert game.player(victim).resources == held
        assert game.current_player_index == (mover + 1) % 3
        assert [a.action_type for a in game.playable_actions] == [cpp.ActionType.ROLL]
        return
    pytest.fail("no robber move with a victim")
