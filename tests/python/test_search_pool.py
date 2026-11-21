"""HandTracker, sample_world, likely_worlds and SearchPool through the Python bindings."""

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


@pytest.mark.parametrize("transpositions", [False, True])
def test_search_pool_buffers_and_reuse(transpositions):
    game, _ = random_game(2, 8, 60)
    codec = cpp.FlatActionCodec(2, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(2, observation_layout())
    pool = cpp.SearchPool(
        codec, encoder, cpp.SearchConfig(simulations=40, max_players=2, transpositions=transpositions))
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


def placements(players: int, seed: int) -> list[cpp.PlayerSetup]:
    """The pieces of a random initial build phase, as setups to edit."""
    game = cpp.Game(players, seed=seed)
    rng = random.Random(seed)
    while game.is_initial_build_phase:
        game.execute(rng.choice(game.playable_actions))
    return [cpp.PlayerSetup(settlements=list(game.player(seat).settlements), roads=list(game.player(seat).roads))
            for seat in range(players)]


def test_search_pool_transpositions():
    """Red discards four of two wood, two brick, two sheep and two wheat: every
    order of one multiset of kinds reaches one hand, which the table searches
    once. Off is the default and links nothing; threads change nothing."""
    assert cpp.SearchConfig().transpositions is False
    config = cpp.SearchConfig(simulations=128, max_players=4, transpositions=True)
    assert config.transpositions and "transpositions=True" in repr(config)
    players = placements(4, 12)
    players[0] = cpp.PlayerSetup(settlements=players[0].settlements, roads=players[0].roads,
                                 resources=(2, 2, 2, 2, 0), discards=4)
    game = cpp.Game(4, seed=12)
    game.apply_setup(players, has_rolled=True, prompt=cpp.ActionPrompt.DISCARD)
    assert game.current_prompt == cpp.ActionPrompt.DISCARD and len(game.playable_actions) == 4
    codec = cpp.FlatActionCodec(4, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(4, observation_layout())
    results = {}
    for transpositions in (False, True):
        for threads in (1, 4):
            config = cpp.SearchConfig(simulations=128, max_players=4, transpositions=transpositions, threads=threads)
            pool = cpp.SearchPool(codec, encoder, config)
            search = pool.add(game, 5)
            run(pool, 4, batch=8)
            stats = pool.stats(search)
            assert pool.done(search) and pool.visits(search).sum() == 128 == stats.simulations
            assert (stats.transpositions > 0) == transpositions
            assert codec.mask(game)[pool.select(search)]
            results[transpositions, threads] = pool.visits(search).tolist()
    assert results[True, 1] == results[True, 4] and results[False, 1] == results[False, 4]


@pytest.mark.parametrize("transpositions", [False, True])
def test_advance_keeps_a_one_kind_steal_refusable_after_re_rooting(transpositions):
    """Red, holding a knight and no resources, plays it and robs Blue, who holds two
    brick: in a world dealt from Red's tracker the steal has one outcome, so its node
    is a decision child with a drawn outcome, which a determinized advance refuses
    (the world's cards drew it, not the real game's: here Blue's real hand could be a
    brick and an ore). Re-rooting at the knight must keep that outcome on the child
    (it travels with the edge that discovered it), so the second advance, along the
    robber move, is refused as well; before, the re-parenting dropped it, the search
    advanced to a position the real game never reached and selected from its options."""
    players = placements(4, 21)
    players[0] = cpp.PlayerSetup(settlements=players[0].settlements, roads=players[0].roads,
                                 development_cards=(1, 0, 0, 0, 0))
    players[1] = cpp.PlayerSetup(settlements=players[1].settlements, roads=players[1].roads,
                                 resources=(0, 2, 0, 0, 0))
    game = cpp.Game(4, seed=21)
    game.apply_setup(players, has_rolled=True)
    knight = cpp.Action(game.current_color, cpp.ActionType.PLAY_KNIGHT_CARD, None)
    assert knight in game.playable_actions and len(game.playable_actions) == 2
    codec = cpp.FlatActionCodec(4, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(4, observation_layout())
    config = cpp.SearchConfig(simulations=64, max_players=4, determinizations=1, transpositions=transpositions)
    pool = cpp.SearchPool(codec, encoder, config)
    search = pool.add(game, 9, cpp.HandTracker(game, 0))
    run(pool, 4)
    index = codec.index(game, knight)
    assert pool.visits(search)[index] > 0 and pool.advance(search, index)
    run(pool, 4)
    after = game.copy()
    after.execute(knight)
    victim = game.colors[1]
    robberies = [a for a in after.playable_actions
                 if a.action_type == cpp.ActionType.MOVE_ROBBER and a.value[1] == victim]
    visits = pool.visits(search)
    visited = [codec.index(after, a) for a in robberies if visits[codec.index(after, a)] > 0]
    assert visited
    for index in visited:
        assert not pool.advance(search, index)


def re_rooted_root_visits(game: cpp.Game, actions, config: cpp.SearchConfig, seed: int, tracker: bool):
    """Search ``game``, then advance along each of ``actions`` (an action, or None for
    the selected one) and search again. Per advance: the action's visits before it, the
    new root's child visits right after it and after the new budget."""
    codec = cpp.FlatActionCodec(4, cpp.MapType.BASE)
    pool = cpp.SearchPool(codec, cpp.FlatObservationEncoder(4, observation_layout()), config)
    search = pool.add(game, seed, cpp.HandTracker(game, 0)) if tracker else pool.add(game, seed)
    run(pool, 4)
    game = game.copy()
    counts = []
    for action in actions:
        index = pool.select(search) if action is None else codec.index(game, action)
        retained = int(pool.visits(search)[index])
        assert retained > 0 and pool.advance(search, index)
        kept = int(pool.visits(search).sum())
        run(pool, 4)
        assert pool.stats(search).simulations == config.simulations
        game.execute(codec.decode(game, index))
        assert codec.mask(game)[pool.select(search)]
        counts.append((retained, kept, int(pool.visits(search).sum())))
    return counts


@pytest.mark.parametrize("transpositions", [False, True])
def test_advance_keeps_the_root_children_private(transpositions):
    """A root counts each edge's child as that edge's visits, so place() never links the
    root's children; after advance the old depth two (where links are made) is the new
    root's children. Red holds a monopoly and four wood, nobody else holds a card: every
    monopoly reaches one position, so after a bank trade the five monopolies shared one
    child (the new root counted it five times, and once more per simulation through it),
    and a child that the monopoly-then-trade orders reached as well counted their visits
    too. Now the new root counts only what passed through it: fewer than the action's
    visits right after the advance, one per simulation after. A discard chain (one
    discarder's kinds in any order reach one hand), followed advance after advance, shows
    the second case, and that a child whose statistics restarted passes that on to its
    children, which its own visits no longer count."""
    players = placements(4, 3)
    players[0] = cpp.PlayerSetup(settlements=players[0].settlements, roads=players[0].roads,
                                 resources=(4, 0, 0, 0, 0), development_cards=(0, 0, 1, 0, 0))
    game = cpp.Game(4, seed=3)
    game.apply_setup(players, has_rolled=True)
    trade = next(a for a in game.playable_actions if a.action_type == cpp.ActionType.MARITIME_TRADE)
    assert sum(a.action_type == cpp.ActionType.PLAY_MONOPOLY for a in game.playable_actions) == 5
    for root in (cpp.RootSelection.PUCT, cpp.RootSelection.GUMBEL):
        config = cpp.SearchConfig(simulations=256, max_players=4, determinizations=1, root=root,
                                  transpositions=transpositions)
        [(retained, kept, total)] = re_rooted_root_visits(game, [trade], config, 3, tracker=True)
        assert kept < retained and total == kept + 256

    config = cpp.SearchConfig(simulations=64, max_players=4, transpositions=transpositions)
    for hand in ((2, 2, 2, 2, 0), (1, 1, 1, 1, 4)):
        players = placements(4, 12)
        players[0] = cpp.PlayerSetup(settlements=players[0].settlements, roads=players[0].roads,
                                     resources=hand, discards=4)
        game = cpp.Game(4, seed=12)
        game.apply_setup(players, has_rolled=True, prompt=cpp.ActionPrompt.DISCARD)
        for first in game.playable_actions:
            counts = re_rooted_root_visits(game, [first, None, None], config, 5, tracker=False)
            for retained, kept, total in counts:
                assert kept < retained and total == kept + 64


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


def hashed(observations: np.ndarray, actions: int, players: int):
    """Logits and values that depend on the row (the position), not on the batch."""
    total = observations.sum(axis=1, dtype=np.float32)
    logits = np.sin(total[:, None] * np.linspace(0.01, 0.3, actions, dtype=np.float32)).astype(np.float32)
    values = np.full((len(observations), players), 0.0, dtype=np.float32)
    values[:, 0] = 0.5 + 0.3 * np.sin(total * 0.37)
    values[:, 1:] = (1.0 - values[:, :1]) / (players - 1)
    return logits, values


def run_hashed(pool: cpp.SearchPool, players: int, batch: int = 32, check_rows=None) -> int:
    observations = np.zeros((batch, pool.observation_size), dtype=np.float32)
    ids = np.zeros(batch, dtype=np.int64)
    total = 0
    while n := pool.collect(observations, ids):
        if check_rows is not None:
            check_rows(ids[:n])
        pool.apply(*hashed(observations[:n], pool.action_size, players))
        total += n
    return total


def test_search_options_and_reroot():
    """A search's own budget, leaves per round and noise; reroot's visit cap; the
    clean root priors."""
    game, _ = random_game(2, 8, 60)
    codec = cpp.FlatActionCodec(2, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(2, observation_layout())
    noise = dict(dirichlet_alpha=0.3, dirichlet_fraction=0.25)

    def alone(simulations, max_pending, alpha):
        pool = cpp.SearchPool(codec, encoder, cpp.SearchConfig(
            simulations=simulations, max_pending=max_pending, max_players=2,
            dirichlet_alpha=alpha, dirichlet_fraction=0.25))
        search = pool.add(game, 4)
        run_hashed(pool, 2)
        return pool.visits(search)

    pool = cpp.SearchPool(codec, encoder, cpp.SearchConfig(simulations=120, max_pending=6, max_players=2, **noise))
    full = pool.add(game, 4)
    fast = pool.add(game, 4, simulations=30, max_pending=2, dirichlet_alpha=0.0)

    def check_rows(ids):
        assert (ids == full).sum() <= 6 and (ids == fast).sum() <= 2

    run_hashed(pool, 2, check_rows=check_rows)
    np.testing.assert_array_equal(pool.visits(full), alone(120, 6, 0.3))
    np.testing.assert_array_equal(pool.visits(fast), alone(30, 2, 0.0))
    assert pool.stats(full).simulations == 120 and pool.stats(fast).simulations == 30
    for bad in (dict(simulations=0), dict(max_pending=0), dict(dirichlet_alpha=float("nan"))):
        with pytest.raises(ValueError):
            pool.add(game, 4, **bad)
    assert len(pool) == 2

    # The priors of the tree are the network's, with or without noise in the search.
    legal = codec.mask(game).astype(bool)
    logits, _ = hashed(encoder.encode(game, game.current_player_index)[None], codec.size, 2)
    clean = np.where(legal, np.exp(logits[0].astype(np.float64)), 0.0)
    clean /= clean.sum()
    for search in (full, fast):
        priors = pool.root_priors(search)
        assert priors.dtype == np.float32 and priors.shape == (1, codec.size)
        np.testing.assert_allclose(priors[0], clean, atol=1e-6)

    # Re-root both along the full search's most visited building action.
    visits = pool.visits(full)
    chosen = None
    for index in np.argsort(-visits.astype(np.int64), kind="stable"):
        kind = codec.decode(game, int(index)).action_type
        if visits[index] and pool.visits(fast)[index] and kind not in (
                cpp.ActionType.ROLL, cpp.ActionType.BUY_DEVELOPMENT_CARD, cpp.ActionType.MOVE_ROBBER,
                cpp.ActionType.END_TURN):
            chosen = int(index)
            break
    if chosen is None:
        pytest.skip("no building action was visited by both searches")
    assert not pool.reroot(full, int(np.flatnonzero(~legal)[0]), simulations=50)
    with pytest.raises(ValueError):
        pool.reroot(full, chosen, simulations=0)
    np.testing.assert_array_equal(pool.visits(full), visits)
    assert pool.reroot(full, chosen, simulations=1)
    carried = int(pool.visits(full).sum())
    assert carried >= 1 and pool.done(full) and pool.stats(full).simulations == 0
    assert pool.reroot(fast, chosen, simulations=carried + 40, max_pending=3, dirichlet_alpha=0.3)
    before = int(pool.visits(fast).sum())
    assert not pool.done(fast)

    def only_fast(ids):
        assert (ids == fast).all() and len(ids) <= 3

    run_hashed(pool, 2, check_rows=only_fast)
    assert pool.done(fast) and pool.visits(fast).sum() == carried + 40
    assert pool.stats(fast).simulations == carried + 40 - before
    after = game.copy()
    after.execute(codec.decode(game, chosen))
    legal = codec.mask(after).astype(bool)
    logits, _ = hashed(encoder.encode(after, after.current_player_index)[None], codec.size, 2)
    clean = np.where(legal, np.exp(logits[0].astype(np.float64)), 0.0)
    clean /= clean.sum()
    for search in (full, fast):
        np.testing.assert_allclose(pool.root_priors(search)[0], clean, atol=1e-6)
        assert pool.policy_target(search).sum() == pytest.approx(1.0)
        assert legal[pool.select(search)]

    gumbel = cpp.SearchPool(codec, encoder, cpp.SearchConfig(
        simulations=32, max_players=2, root=cpp.RootSelection.GUMBEL))
    search = gumbel.add(game, 4)
    run_hashed(gumbel, 2)
    with pytest.raises(ValueError):
        gumbel.reroot(search, chosen)


def tempered(priors: np.ndarray, temperature: float) -> np.ndarray:
    """Priors at a root temperature: each to the power 1 / temperature, renormalized."""
    powered = np.where(priors > 0, priors.astype(np.float64) ** (1.0 / temperature), 0.0)
    return powered / powered.sum()


def peaked(observations: np.ndarray, actions: int, players: int):
    """:func:`hashed` with four times the logits: priors under which PUCT leaves root actions unvisited."""
    logits, values = hashed(observations, actions, players)
    return 4.0 * logits, values


def short_of_forced(priors: np.ndarray, visits: np.ndarray, k: float) -> np.ndarray:
    """What each visited root action lacks of sqrt(k * prior * N) visits, N the root's visits
    (float64 as the engine computes it; -inf for an action without a visit, which nothing forces)."""
    visits = visits.astype(np.int64)
    return np.where(visits > 0, np.sqrt(k * priors.astype(np.float64) * visits.sum()) - visits, -np.inf)


def forced_replay(pool: cpp.SearchPool, k: float, evaluate=hashed, players: int = 2, batch: int = 64) -> dict:
    """Run ``pool`` (searches of one root world) and check its root selections against the
    forced playouts' rule at ``k``. Root visits change only in ``collect`` and count a leaf from
    the moment it is issued, so the visits before a round are the state its first selection
    saw, and each selection the rule decides (a visited action is below its count: the one
    furthest below is selected) gives the state the next one saw. A round is replayed until
    its first selection the rule leaves to PUCT, whose choice the visits do not tell; with one
    leaf per round that is every selection. Per search id: [selections the rule decided and the
    search made, rounds in which the rule decided one the search did not make]."""
    observations = np.zeros((batch, pool.observation_size), dtype=np.float32)
    ids = np.zeros(batch, dtype=np.int64)
    states = {search: pool.visits(search).astype(np.int64) for search in pool.ids}
    tally = {search: [0, 0] for search in states}
    while n := pool.collect(observations, ids):
        for search, state in states.items():
            after = pool.visits(search).astype(np.int64)
            priors = pool.root_search_priors(search)[0]
            left = after - state
            assert (left >= 0).all()
            while left.sum():
                missing = short_of_forced(priors, state, k)
                if not missing.max() > 0.0:
                    break  # PUCT's selection
                pick = int(np.argmax(missing))
                if not left[pick]:
                    tally[search][1] += 1
                    break
                state[pick] += 1
                left[pick] -= 1
                tally[search][0] += 1
            states[search] = after
        pool.apply(*evaluate(observations[:n], pool.action_size, players))
    return tally


def test_root_temperature():
    """The root selects by its priors at the temperature; the tree keeps the
    network's, so a re-rooted search meets them unchanged."""
    game, _ = random_game(2, 8, 60)
    codec = cpp.FlatActionCodec(2, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(2, observation_layout())
    assert cpp.SearchConfig().root_policy_temperature == 1.0
    assert "root_policy_temperature=1.0, forced_playouts=0.0" in repr(cpp.SearchConfig())

    plain = cpp.SearchPool(codec, encoder, cpp.SearchConfig(simulations=120, max_players=2))
    reference = plain.add(game, 4)
    run_hashed(plain, 2)
    np.testing.assert_array_equal(plain.root_search_priors(reference), plain.root_priors(reference))

    pool = cpp.SearchPool(codec, encoder, cpp.SearchConfig(
        simulations=120, max_players=2, root_policy_temperature=1.5))
    assert pool.config.root_policy_temperature == 1.5
    search = pool.add(game, 4)
    assert not pool.root_search_priors(search).any()  # the root is not evaluated yet
    run_hashed(pool, 2)
    used = pool.root_search_priors(search)
    assert used.dtype == np.float32 and used.shape == (1, codec.size)
    np.testing.assert_array_equal(pool.root_priors(search), plain.root_priors(reference))
    np.testing.assert_allclose(used[0], tempered(pool.root_priors(search)[0], 1.5), atol=1e-6)
    assert used.max() < pool.root_priors(search).max() - 1e-3
    assert (pool.visits(search) != plain.visits(reference)).any()

    # The noise is mixed into the priors at the temperature: one seed, one noise.
    noisy = cpp.SearchPool(codec, encoder, cpp.SearchConfig(simulations=120, max_players=2, dirichlet_alpha=0.3))
    both = noisy.add(game, 4, root_policy_temperature=1.5)
    alone = noisy.add(game, 4)
    run_hashed(noisy, 2)
    network = noisy.root_priors(alone)[0]
    np.testing.assert_array_equal(noisy.root_priors(both)[0], network)
    noise = noisy.root_search_priors(alone)[0] - 0.75 * network
    assert noise.max() > 1e-2
    np.testing.assert_allclose(noisy.root_search_priors(both)[0] - 0.75 * tempered(network, 1.5), noise, atol=1e-6)

    # Re-rooted, the temperature meets the network's priors again (it never compounds).
    visits = pool.visits(search)
    chosen = None
    for index in np.argsort(-visits.astype(np.int64), kind="stable"):
        kind = codec.decode(game, int(index)).action_type
        if visits[index] and kind not in (
                cpp.ActionType.ROLL, cpp.ActionType.BUY_DEVELOPMENT_CARD, cpp.ActionType.MOVE_ROBBER,
                cpp.ActionType.END_TURN):
            chosen = int(index)
            break
    if chosen is None:
        pytest.skip("no building action was visited")
    assert pool.reroot(search, chosen, simulations=200)
    after = game.copy()
    after.execute(codec.decode(game, chosen))
    legal = codec.mask(after).astype(bool)
    logits, _ = hashed(encoder.encode(after, after.current_player_index)[None], codec.size, 2)
    clean = np.where(legal, np.exp(logits[0].astype(np.float64)), 0.0)
    clean /= clean.sum()
    for searched in (False, True):
        if searched:
            run_hashed(pool, 2)
        np.testing.assert_allclose(pool.root_priors(search)[0], clean, atol=1e-6)
        np.testing.assert_allclose(pool.root_search_priors(search)[0], tempered(clean, 1.5), atol=1e-6)
    assert pool.visits(search).sum() == 200


@pytest.mark.parametrize("max_pending", [1, 4, 16])
@pytest.mark.parametrize("shaped", [False, True])
def test_forced_playouts(max_pending, shaped):
    """With k = 2 the root selects, whenever a visited action is below sqrt(2 * prior * N)
    visits, the visited action furthest below (:func:`forced_replay`: every selection seen,
    all of them with one leaf per round); an action without a visit is never forced, so
    actions stay unvisited as under PUCT alone, which selects otherwise."""
    game, _ = random_game(2, 8, 60)
    codec = cpp.FlatActionCodec(2, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(2, observation_layout())
    assert cpp.SearchConfig().forced_playouts == 0.0
    settings = dict(simulations=300, max_pending=max_pending, max_players=2)
    if shaped:
        settings.update(dirichlet_alpha=0.3, root_policy_temperature=1.25)
    pool = cpp.SearchPool(codec, encoder, cpp.SearchConfig(**settings))
    plain = pool.add(game, 4)
    forced = pool.add(game, 4, forced_playouts=2.0)
    tally = forced_replay(pool, 2.0, peaked)
    np.testing.assert_array_equal(pool.root_search_priors(forced), pool.root_search_priors(plain))
    priors = pool.root_search_priors(forced)[0]
    assert pool.visits(forced).sum() == 300 and pool.stats(forced).simulations == 300
    assert tally[forced][0] > 0 and tally[forced][1] == 0  # the rule's selections, each one made
    assert tally[plain][1] > 0  # PUCT alone makes others
    # KataGo's rule: an action is forced once it has a visit, so some are never opened.
    assert ((priors > 0) & (pool.visits(forced) == 0)).any()
    # Here the visited actions end at their counts, which PUCT alone leaves them short of.
    lacking = [np.ceil(short_of_forced(priors, pool.visits(search), 2.0)).clip(min=0.0).sum()
               for search in (forced, plain)]
    assert lacking[0] < lacking[1]

    # Short searches, where the counts rise faster than one visit pays: the rule decides the same way.
    made = 0
    for budget in range(1, 61):
        brief = cpp.SearchPool(codec, encoder, cpp.SearchConfig(**settings))
        search = brief.add(game, budget, simulations=budget, forced_playouts=2.0)
        decided, missed = forced_replay(brief, 2.0, peaked)[search]
        assert brief.visits(search).sum() == budget and missed == 0
        made += decided
    assert made > 20


def test_root_options_per_search_and_refusals():
    """add's and reroot's root temperature and forced playouts beat the pool's
    config; values no root can run with, and either at a Gumbel root, raise."""
    game, trackers = random_game(2, 8, 60)
    codec = cpp.FlatActionCodec(2, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(2, observation_layout())
    base = dict(simulations=120, max_pending=4, max_players=2, dirichlet_alpha=0.3)

    def result(pool, search):
        return np.concatenate([pool.visits(search), pool.q(search), pool.root_search_priors(search)[0]])

    def alone(**settings):
        pool = cpp.SearchPool(codec, encoder, cpp.SearchConfig(**base, **settings))
        search = pool.add(game, 4)
        run_hashed(pool, 2)
        return result(pool, search)

    plain, shaped = alone(), alone(root_policy_temperature=1.5, forced_playouts=2.0)
    for other in (alone(root_policy_temperature=1.5), alone(forced_playouts=2.0)):
        assert not np.array_equal(other, plain, equal_nan=True)
        assert not np.array_equal(other, shaped, equal_nan=True)
    on = dict(root_policy_temperature=1.5, forced_playouts=2.0)
    off = dict(root_policy_temperature=1.0, forced_playouts=0.0)
    for config, options, expected in ((on, off, plain), ({}, on, shaped)):
        pool = cpp.SearchPool(codec, encoder, cpp.SearchConfig(**base, **config))
        own = pool.add(game, 4, **options)
        pools = pool.add(game, 4)
        run_hashed(pool, 2)
        np.testing.assert_array_equal(result(pool, own), expected)
        np.testing.assert_array_equal(result(pool, pools), shaped if config else plain)

    pool = cpp.SearchPool(codec, encoder, cpp.SearchConfig(**base))
    search = pool.add(game, 4)
    run_hashed(pool, 2)
    before = result(pool, search)
    chosen = int(pool.visits(search).argmax())
    nan, inf = float("nan"), float("inf")
    for bad in (dict(root_policy_temperature=0.0), dict(root_policy_temperature=-1.0),
                dict(root_policy_temperature=nan), dict(root_policy_temperature=inf),
                dict(forced_playouts=-0.5), dict(forced_playouts=nan), dict(forced_playouts=inf)):
        with pytest.raises(ValueError):
            cpp.SearchPool(codec, encoder, cpp.SearchConfig(**base, **bad))
        with pytest.raises(ValueError):
            pool.add(game, 4, **bad)
        with pytest.raises(ValueError):
            pool.reroot(search, chosen, **bad)
        np.testing.assert_array_equal(result(pool, search), before)
    assert len(pool) == 1

    gumbel = dict(simulations=32, max_players=2, root=cpp.RootSelection.GUMBEL)
    halving = cpp.SearchPool(codec, encoder, cpp.SearchConfig(**gumbel))
    for bad in (dict(root_policy_temperature=1.25), dict(forced_playouts=2.0)):
        with pytest.raises(ValueError):
            cpp.SearchPool(codec, encoder, cpp.SearchConfig(**gumbel, **bad))
        with pytest.raises(ValueError):
            halving.add(game, 4, **bad)
    search = halving.add(game, 4, **off)
    run_hashed(halving, 2)
    np.testing.assert_array_equal(halving.root_search_priors(search), halving.root_priors(search))

    # Every root world takes both: the worlds' visits still fill the budget.
    worlds = cpp.SearchPool(codec, encoder, cpp.SearchConfig(
        simulations=90, max_pending=3, max_players=2, determinizations=3, dirichlet_alpha=0.3, **on))
    search = worlds.add(game, 4, trackers[game.current_player_index])
    run_hashed(worlds, 2)
    count = worlds.stats(search).worlds
    assert worlds.done(search) and worlds.visits(search).sum() == 90
    used, network = worlds.root_search_priors(search), worlds.root_priors(search)
    assert used.shape == network.shape == (count, codec.size)
    noise = used[0] - 0.75 * tempered(network[0], 1.5)
    for world in range(count):
        np.testing.assert_allclose(used[world] - 0.75 * tempered(network[world], 1.5), noise, atol=1e-6)


def test_free_backlog():
    """Who frees the trees that ``remove`` and ``reroot`` discard changes no result: the pool
    counts what waits for its threads, and the next ``collect`` frees it."""
    game, _ = random_game(2, 8, 60)
    codec = cpp.FlatActionCodec(2, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(2, observation_layout())
    default = cpp.SearchConfig().free_backlog
    assert default == 65536 and f"forced_playouts=0.0, free_backlog={default})" in repr(cpp.SearchConfig())
    with pytest.raises(ValueError, match="free_backlog"):
        cpp.SearchPool(codec, encoder, cpp.SearchConfig(max_players=2, free_backlog=-1))

    def played(free_backlog: int, threads: int):
        config = cpp.SearchConfig(simulations=80, max_players=2, threads=threads, free_backlog=free_backlog)
        pool = cpp.SearchPool(codec, encoder, config)
        assert pool.config.free_backlog == free_backlog and pool.unfreed_nodes == 0
        waits = free_backlog > 0 and threads > 1
        ids = [pool.add(game, seed) for seed in range(6)]
        results, most, seed = [], 0, 6
        for _ in range(3):
            run_hashed(pool, 2)
            assert pool.unfreed_nodes == 0  # the collect freed what waited
            for i, search in enumerate(ids):
                visits = pool.visits(search)
                before, waiting = pool.stats(search).nodes, pool.unfreed_nodes
                results.append((search, visits.tobytes(), pool.q(search).tobytes(), pool.value(search).tobytes(),
                                pool.policy_target(search).tobytes(), pool.select(search), before))
                order = [int(a) for a in np.argsort(-visits.astype(np.int64), kind="stable") if visits[a] > 0]
                if any(pool.reroot(search, action, simulations=80) for action in order):
                    cut = before - pool.stats(search).nodes  # what the new root left behind
                else:  # only chance actions were visited
                    pool.remove(search)
                    cut = before
                    ids[i] = pool.add(game, seed)
                    seed += 1
                assert cut > 0 and pool.unfreed_nodes == waiting + (cut if waits else 0)
            most = max(most, pool.unfreed_nodes)
        return results, most

    inside, none = played(0, 2)
    threaded, most = played(default, 2)
    single, alone = played(default, 1)
    assert none == 0 and alone == 0 and 0 < most <= default
    assert threaded == inside and single == inside


def hidden_cards_game(hidden: int, seed: int = 1):
    """A two-player game where the seat to move has a choice and the other holds at least ``hidden``
    unplayed development cards, with a tracker per seat."""
    for start in range(seed, seed + 40):
        game = cpp.Game(2, seed=start)
        trackers = [cpp.HandTracker(game, seat) for seat in range(2)]
        rng = random.Random(start)
        for step in range(1200):
            action = rng.choice(game.playable_actions)
            before = game.copy()
            game.execute(action)
            for tracker in trackers:
                tracker.observe(before, action, game)
            if game.winning_color is not None:
                break
            mover = game.current_player_index
            held = trackers[mover].development_card_count(1 - mover)
            if step % 20 == 0 and held >= hidden and len(game.playable_actions) >= 2:
                return game, trackers
    raise AssertionError("no random game reached the position")


def test_likely_worlds_bindings():
    """``likely_worlds`` lists the deals ``sample_world`` draws, most likely first, with their
    probabilities; too many of them, or a finished game, is a ``ValueError``."""
    game, trackers = hidden_cards_game(3)
    mover = game.current_player_index
    tracker = trackers[mover]
    deals = cpp.likely_worlds(game, tracker)
    probabilities = [probability for _, probability in deals]
    assert len(deals) > 8 and sum(probabilities) == pytest.approx(1.0, abs=1e-12)
    assert probabilities == sorted(probabilities, reverse=True)
    again = cpp.likely_worlds(game, tracker, deal=cpp.TrackerMode.POOL)
    assert [probability for _, probability in again] == probabilities
    listed = set()
    for hands, _ in deals:
        assert len(hands) == 2 and isinstance(hands[0], cpp.HiddenHand)
        assert hands[mover].resources == game.player(mover).resources
        assert hands[mover].development_cards == game.player(mover).development_cards
        assert hands[1 - mover].resources == game.player(1 - mover).resources  # one seat: nothing to deal
        assert sum(hands[1 - mover].development_cards) == tracker.development_card_count(1 - mover)
        world = game.copy()
        world.redeal_hidden_cards(hands, 7)
        assert tracker.consistent_with(world)
        listed.add((hands[1 - mover].development_cards, hands[1 - mover].development_card_owned_at_start))
    assert len(listed) == len(deals)
    for seed in range(200):
        other = cpp.sample_world(game, tracker, seed=seed).player(1 - mover)
        assert (other.development_cards, other.development_card_owned_at_start) in listed

    with pytest.raises(ValueError, match="likely worlds need a position whose hidden cards have at most 8"):
        cpp.likely_worlds(game, tracker, limit=8)
    assert len(cpp.likely_worlds(game, tracker, limit=len(deals))) == len(deals)
    with pytest.raises(ValueError, match="exact"):
        cpp.likely_worlds(game, tracker, deal=cpp.TrackerMode.EXACT)


@pytest.mark.parametrize("root", [cpp.RootSelection.PUCT, cpp.RootSelection.GUMBEL])
def test_search_pool_likely_worlds(root):
    """``world_choice=LIKELY`` searches the most likely deals by their probability: the weights,
    the worlds' shares of the simulations and the pooled visits, at both roots."""
    assert cpp.SearchConfig().world_choice == cpp.WorldChoice.SAMPLE
    assert "determinizations=0, world_choice=WorldChoice.SAMPLE" in repr(cpp.SearchConfig())
    game, trackers = hidden_cards_game(3)
    tracker = trackers[game.current_player_index]
    deals = cpp.likely_worlds(game, tracker)
    codec = cpp.FlatActionCodec(2, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(2, observation_layout())
    config = cpp.SearchConfig(
        simulations=300, max_players=2, root=root, determinizations=8, world_choice=cpp.WorldChoice.LIKELY,
        max_pending=4)
    assert config.world_choice == cpp.WorldChoice.LIKELY and "world_choice=WorldChoice.LIKELY" in repr(config)
    with pytest.raises(ValueError, match="likely worlds need determinizations"):
        cpp.SearchPool(codec, encoder, cpp.SearchConfig(max_players=2, world_choice=cpp.WorldChoice.LIKELY))
    pool = cpp.SearchPool(codec, encoder, config)
    assert pool.config.world_choice == cpp.WorldChoice.LIKELY
    with pytest.raises(ValueError):
        pool.add(game, 1)  # likely worlds are the tracker's
    first, second = pool.add(game, 1, tracker), pool.add(game, 2, tracker)
    weights = pool.world_weights(first)
    kept = sum(probability for _, probability in deals[:8])
    assert weights.dtype == np.float64 and weights.shape == (8,)
    np.testing.assert_allclose(weights, [probability / kept for _, probability in deals[:8]], rtol=1e-12)
    np.testing.assert_array_equal(weights, pool.world_weights(second))  # whatever the seed
    run_hashed(pool, 2)
    legal = codec.mask(game).astype(bool)
    for search in (first, second):
        assert pool.done(search) and pool.stats(search).worlds == 8
        visits, by_world = pool.visits(search), pool.world_visits(search)
        assert by_world.dtype == np.uint32 and by_world.shape == (8, codec.size)
        np.testing.assert_array_equal(by_world.sum(axis=0), visits)
        assert visits.sum() == 300 and not visits[~legal].any()
        if root == cpp.RootSelection.PUCT:  # every world within one simulation of its share
            assert np.abs(by_world.sum(axis=1) - weights * 300).max() <= 1 + 1e-9
        else:  # and at a Gumbel root every action is shared so
            assert np.abs(by_world - weights[:, None] * visits[None, :]).max() <= 1 + 1e-9
        assert legal[pool.select(search)] and pool.policy_target(search).sum() == pytest.approx(1.0)
        assert pool.root_priors(search).shape == (8, codec.size)

    # Sampled worlds report their share of the draws.
    sampled = cpp.SearchPool(codec, encoder, cpp.SearchConfig(simulations=64, max_players=2, determinizations=8))
    search = sampled.add(game, 1, tracker)
    shares = sampled.world_weights(search)
    assert shares.shape == (sampled.stats(search).worlds,) and shares.sum() == pytest.approx(1.0)
    run_hashed(sampled, 2)
    np.testing.assert_array_equal(sampled.world_visits(search).sum(axis=1), shares * 64)
