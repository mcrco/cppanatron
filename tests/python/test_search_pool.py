"""HandTracker, sample_world and SearchPool through the Python bindings."""

from __future__ import annotations

import random

import numpy as np
import pytest

import cppanatron as cpp
from cppanatron.catanatron import observation_layout


def random_game(players: int, seed: int, steps: int):
    """A game ``steps`` random moves in, with a tracker per seat."""
    game = cpp.Game(players, seed=seed)
    trackers = [cpp.HandTracker(game, seat) for seat in range(players)]
    rng = random.Random(seed)
    for _ in range(steps):
        action = rng.choice(game.playable_actions)
        before = game.copy()
        game.execute(action)
        for tracker in trackers:
            tracker.observe(before, action, game)
            assert tracker.consistent_with(game)
        assert game.winning_color is None
    while len(game.playable_actions) < 2:
        action = game.playable_actions[0]
        before = game.copy()
        game.execute(action)
        for tracker in trackers:
            tracker.observe(before, action, game)
    return game, trackers


@pytest.mark.parametrize("players", [2, 3, 4])
def test_tracker_is_sound_and_samples_ignore_hidden_cards(players):
    game, trackers = random_game(players, 11 + players, 250)
    for tracker in trackers:
        alternative = cpp.sample_world(game, tracker, seed=1)
        assert tracker.consistent_with(alternative)
        first = cpp.sample_world(game, tracker, seed=5)
        second = cpp.sample_world(alternative, tracker, seed=5)
        assert first.search_equivalent(second)
        # The observer's own hand is kept; sizes and totals are public.
        seat = tracker.observer
        assert first.player(seat).resources == game.player(seat).resources
        for other in range(players):
            assert sum(first.player(other).resources) == tracker.hand_size(other)
        # A sample is a legal game.
        rng = random.Random(seat)
        for _ in range(50):
            if first.winning_color is not None:
                break
            first.execute(rng.choice(first.playable_actions))


def test_tracker_event_api():
    game = cpp.Game(3, seed=2)
    tracker = cpp.HandTracker(game, 0)
    tracker.gain(1, (2, 1, 0, 0, 0))
    tracker.gain(2, (0, 0, 1, 0, 0))
    tracker.steal(2, 1)  # unseen by seat 0
    assert tracker.known_resources(1) == (1, 0, 0, 0, 0)
    assert tracker.unknown_count(2) == 1
    assert tracker.unknown_pool == (1, 1, 0, 0, 0)
    assert tracker.expected_resources(2) == pytest.approx((0.5, 0.5, 1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        tracker.steal(0, 1)  # the observer always sees its own steals
    with pytest.raises(ValueError):
        tracker.buy_development_card(1, cpp.DevelopmentCard.KNIGHT)  # hidden from seat 0
    copy = tracker.copy()
    with pytest.raises(RuntimeError):
        tracker.lose(2, (0, 0, 0, 3, 0))
    assert tracker == copy

    event = cpp.TrackerEvent.monopoly(1, cpp.Resource.SHEEP, [0, 0, 1])
    assert event.type == cpp.TrackerEventType.MONOPOLY
    tracker.apply(event)
    assert tracker.known_resources(1)[2] == tracker.resource_totals[2]
    assert tracker.unknown_pool[2] == 0


def test_events_hide_what_the_observer_cannot_see():
    game, trackers = random_game(4, 5, 0)
    rng = random.Random(5)
    hidden = seen = 0
    for _ in range(1500):
        if game.winning_color is not None:
            break
        action = rng.choice(game.playable_actions)
        before = game.copy()
        game.execute(action)
        for seat in range(4):
            events = cpp.HandTracker.events(before, action, game, seat)
            for event in events:
                if event.type == cpp.TrackerEventType.STEAL:
                    involved = seat in (event.player, event.other)
                    assert (event.resource is not None) == involved
                    hidden += not involved
                    seen += involved
                if event.type == cpp.TrackerEventType.BUY_DEVELOPMENT_CARD:
                    assert (event.development_card is not None) == (seat == event.player)
            for event in events:
                trackers[seat].apply(event)
            assert trackers[seat].consistent_with(game)
    assert hidden > 0 and seen > 0


def evaluate(pool: cpp.SearchPool, rows: int, players: int):
    """Uniform logits; values 1/players for everyone."""
    logits = np.zeros((rows, pool.action_size), dtype=np.float32)
    values = np.full((rows, pool.max_players), 1.0 / players, dtype=np.float32)
    return logits, values


def run(pool: cpp.SearchPool, players: int, batch: int = 32, check_rows=None) -> int:
    observations = np.zeros((batch, pool.observation_size), dtype=np.float32)
    ids = np.zeros(batch, dtype=np.int64)
    total = 0
    while n := pool.collect(observations, ids):
        if check_rows is not None:
            check_rows(observations[:n], ids[:n])
        pool.apply(*evaluate(pool, n, players))
        total += n
    return total


def test_search_pool_two_players():
    game, _ = random_game(2, 3, 120)
    codec = cpp.FlatActionCodec(2, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(2, observation_layout())
    config = cpp.SearchConfig(simulations=64, max_players=2, max_pending=4, threads=2)
    pool = cpp.SearchPool(codec, encoder, config)
    ids = [pool.add(game, seed) for seed in range(5)]
    assert len(pool) == 5 and ids[0] in pool

    def check_rows(observations, row_ids):
        for row in range(len(row_ids)):
            leaf = pool.pending_game(row)
            expected = encoder.encode(leaf, leaf.current_player_index)
            np.testing.assert_array_equal(observations[row], expected)
            assert int(row_ids[row]) in ids

    rows = run(pool, 2, check_rows=check_rows)
    assert rows <= 5 * (64 + 1)
    legal = codec.mask(game).astype(bool)
    for search in ids:
        assert pool.done(search)
        visits = pool.visits(search)
        assert visits.dtype == np.uint32 and visits.shape == (codec.size,)
        assert visits.sum() == 64 and not visits[~legal].any()
        policy = pool.policy_target(search)
        assert policy.dtype == np.float32 and policy.sum() == pytest.approx(1.0)
        q = pool.q(search)
        assert np.isnan(q[visits == 0]).all() and not np.isnan(q[visits > 0]).any()
        assert legal[pool.select(search)]
        assert legal[pool.select(search, temperature=1.0)]
        value = pool.value(search)
        assert value.shape == (2,) and value.sum() == pytest.approx(1.0, abs=1e-5)
        assert pool.stats(search).simulations == 64
        pool.remove(search)
    assert len(pool) == 0


def test_search_pool_gumbel_and_determinized_worlds():
    players = 4
    game, trackers = random_game(players, 21, 200)
    tracker = trackers[game.current_player_index]
    alternative = cpp.sample_world(game, tracker, seed=3)
    codec = cpp.FlatActionCodec(players, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(players, observation_layout())
    config = cpp.SearchConfig(
        simulations=48,
        max_players=6,
        root=cpp.RootSelection.GUMBEL,
        gumbel_considered=8,
        determinizations=3,
        max_pending=2,
    )
    pool = cpp.SearchPool(codec, encoder, config)
    truth_id = pool.add(game, 9, tracker)
    alternative_id = pool.add(alternative, 9, tracker)
    with pytest.raises(ValueError):
        pool.add(game, 9)  # determinized search needs the tracker

    # Values depend on the (determinized) position, hidden cards included.
    observations = np.zeros((16, pool.observation_size), dtype=np.float32)
    ids = np.zeros(16, dtype=np.int64)
    while n := pool.collect(observations, ids):
        logits = np.tanh(observations[:n, : codec.size] * 0.01).astype(np.float32)
        values = np.full((n, 6), 0.2, dtype=np.float32)
        values[:, 0] = 0.4 + 0.2 * np.tanh(observations[:n].sum(axis=1) * 1e-3)
        pool.apply(logits, values)
    np.testing.assert_array_equal(pool.visits(truth_id), pool.visits(alternative_id))
    np.testing.assert_array_equal(pool.policy_target(truth_id), pool.policy_target(alternative_id))
    assert pool.select(truth_id) == pool.select(alternative_id)
    assert pool.visits(truth_id).sum() == 48
    assert pool.policy_target(truth_id).sum() == pytest.approx(1.0)


def test_search_pool_buffers_and_reuse():
    game, _ = random_game(2, 8, 60)
    codec = cpp.FlatActionCodec(2, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(2, observation_layout())
    pool = cpp.SearchPool(codec, encoder, cpp.SearchConfig(simulations=40, max_players=2))
    search = pool.add(game, 1)
    with pytest.raises(TypeError):
        pool.collect(np.zeros((4, pool.observation_size)), np.zeros(4, dtype=np.int64))  # float64
    with pytest.raises(ValueError):
        pool.collect(np.zeros((4, 3), dtype=np.float32), np.zeros(4, dtype=np.int64))
    observations = np.zeros((4, pool.observation_size), dtype=np.float32)
    ids = np.zeros(4, dtype=np.int64)
    assert pool.collect(observations, ids) == 1 and ids[0] == search
    with pytest.raises(ValueError):
        pool.apply(*evaluate(pool, 2, 2))  # one row was collected
    pool.apply(*evaluate(pool, 1, 2))
    run(pool, 2)

    # Tree reuse along a deterministic, visited root action.
    visits = pool.visits(search)
    advanced = False
    for index in np.argsort(-visits.astype(np.int64), kind="stable"):
        action = codec.decode(game, int(index))
        if visits[index] == 0:
            break
        if action.action_type in (cpp.ActionType.ROLL, cpp.ActionType.BUY_DEVELOPMENT_CARD):
            assert not pool.advance(search, int(index))
            continue
        if action.action_type == cpp.ActionType.MOVE_ROBBER:
            continue
        retained = int(visits[index])
        assert retained >= 1
        advanced = pool.advance(search, int(index))
        assert advanced and not pool.done(search)
        run(pool, 2)
        assert pool.visits(search).sum() >= 40
        assert pool.stats(search).simulations == 40
        break
    assert advanced


def test_search_pool_rejected_apply_can_be_retried_or_cancelled():
    game, _ = random_game(3, 4, 120)
    codec = cpp.FlatActionCodec(3, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(3, observation_layout())
    pool = cpp.SearchPool(codec, encoder, cpp.SearchConfig(simulations=30, max_players=3, max_pending=4))
    ids = [pool.add(game, seed) for seed in range(3)]
    observations = np.zeros((8, pool.observation_size), dtype=np.float32)
    rows = np.zeros(8, dtype=np.int64)
    for _ in range(3):
        n = pool.collect(observations, rows)
        logits, values = evaluate(pool, n, 3)
        values[0, 0] = np.nan
        with pytest.raises(ValueError):
            pool.apply(logits, values)
        assert pool.pending == n
        values[0, 0] = 0.3
        pool.apply(logits, values)
    assert pool.collect(observations, rows) > 0
    pool.cancel()
    assert pool.pending == 0
    run(pool, 3, batch=8)
    for search in ids:
        assert pool.done(search) and pool.visits(search).sum() == 30


def test_exact_tracker_bindings():
    """The exact mode through the bindings: hypotheses, the deal override and
    the VecEnv's tracker mode (the posterior itself is tested in C++)."""
    game = cpp.Game(3, seed=2)
    exact = cpp.HandTracker(game, 0, cpp.TrackerMode.EXACT)
    pool = cpp.HandTracker(game, 0)
    assert pool.mode == cpp.TrackerMode.POOL and not pool.exact and pool.hypotheses == []
    assert exact.mode == cpp.TrackerMode.EXACT and exact.max_hypotheses == 4096
    assert exact.exact and exact.num_hypotheses == 1
    for tracker in (exact, pool):
        tracker.gain(1, (0, 2, 0, 0, 1))  # seat 1: brick, brick, ore
        tracker.steal(2, 1)  # unseen by seat 0
    assert exact.num_hypotheses == 2 and exact.belief_overflows == 0
    assert sorted(exact.hypotheses) == [
        (pytest.approx(1 / 3), [[0, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 1]]),
        (pytest.approx(2 / 3), [[0, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 0]]),
    ]
    assert exact.expected_resources(2) == pytest.approx((0, 2 / 3, 0, 0, 1 / 3))
    assert pool.expected_resources(2) == pytest.approx((0, 0.5, 0, 0, 0.5))
    assert exact != pool and exact == exact.copy()
    # Sampling needs a game with the same public state: seat 1 holds two cards, seat 2 one.
    truth = cpp.Game(3, seed=2)
    truth.apply_setup([cpp.PlayerSetup(), cpp.PlayerSetup(resources=(0, 1, 0, 0, 1)),
                       cpp.PlayerSetup(resources=(0, 1, 0, 0, 0))])
    assert exact.consistent_with(truth) and pool.consistent_with(truth)
    bricks = sum(cpp.sample_world(truth, exact, seed=s).player(2).resources[1] for s in range(600))
    assert 340 < bricks < 460  # 2/3 of 600
    for seed in range(10):
        forced = cpp.sample_world(truth, exact, seed=seed, deal=cpp.TrackerMode.POOL)
        assert forced.search_equivalent(cpp.sample_world(truth, pool, seed=seed))
    with pytest.raises(ValueError):
        cpp.sample_world(truth, pool, seed=1, deal=cpp.TrackerMode.EXACT)
    capped = cpp.HandTracker(game, 0, cpp.TrackerMode.EXACT, max_hypotheses=1)
    capped.gain(1, (0, 2, 0, 0, 1))
    capped.steal(2, 1)
    assert not capped.exact and capped.belief_overflows == 1 and capped.consistent_with(truth)

    env = cpp.VecEnv([cpp.Scenario(num_players=3)], num_envs=2, seed=1, trackers=True,
                     tracker_mode=cpp.TrackerMode.EXACT)
    env.allocate()
    env.reset()
    assert env.tracker(0, 0).mode == cpp.TrackerMode.EXACT and env.tracker(1, 2).exact
    plain = cpp.VecEnv([cpp.Scenario(num_players=3)], num_envs=1, seed=1, trackers=True)
    plain.allocate()
    plain.reset()
    assert plain.tracker(0, 0).mode == cpp.TrackerMode.POOL


def test_search_pool_deals_exact_worlds():
    players = 3
    game, _ = random_game(players, 31, 150)
    exact = cpp.HandTracker(game, game.current_player_index, cpp.TrackerMode.EXACT)
    codec = cpp.FlatActionCodec(players, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(players, observation_layout())
    pool = cpp.SearchPool(codec, encoder, cpp.SearchConfig(simulations=8, max_players=3, determinizations=2))
    for deal in (None, cpp.TrackerMode.POOL, cpp.TrackerMode.EXACT):
        search = pool.add(game, 3, exact, deal=deal)
        run(pool, players)
        assert pool.stats(search).worlds >= 1
        pool.remove(search)
    with pytest.raises(ValueError):
        pool.add(game, 3, cpp.HandTracker(game, game.current_player_index), deal=cpp.TrackerMode.EXACT)
    assert len(pool) == 0
