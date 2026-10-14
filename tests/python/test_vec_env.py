"""VecEnv: rows, forced moves and bots, rewards and resets, determinism."""

from __future__ import annotations

import numpy as np
import pytest

import cppanatron as cpp
from map_shapes import SHAPES


def sample_actions(env: cpp.VecEnv, buffers: dict, rng: np.random.Generator) -> None:
    """A uniformly random legal action for every row (the no-op for waiting rows)."""
    offset, (size,) = env.encoder.layout()["action_mask"]
    mask = buffers["observations"][:, offset: offset + size]
    buffers["actions"][:] = np.argmax(np.where(mask > 0, rng.random(mask.shape), -1.0), axis=1)


def check_rows(env: cpp.VecEnv, buffers: dict) -> None:
    encoder = env.encoder
    offset, _ = encoder.layout()["action_mask"]
    noop = encoder.action_space.noop
    players = env.max_players
    for e in range(env.num_envs):
        game = env.game(e)
        kinds = env.seat_kinds(e)
        n = len(game.colors)
        assert buffers["num_players"][e] == n
        rows = slice(e * players, (e + 1) * players)
        assert (buffers["env_index"][rows] == e).all()
        assert buffers["seat"][rows].tolist() == list(range(n)) + [-1] * (players - n)
        learner = [k == cpp.SeatKind.LEARNER for k in kinds] + [False] * (players - n)
        assert buffers["learner"][rows].astype(bool).tolist() == learner
        acting = [s == game.current_player_index and learner[s] for s in range(players)]
        assert buffers["acting"][rows].astype(bool).tolist() == acting
        assert sum(acting) == 1, "a learner seat must be deciding in every env"
        assert len(game.playable_actions) > 1, "forced moves are played inside the env"
        for seat in range(players):
            row = buffers["observations"][e * players + seat]
            if acting[seat]:
                assert np.array_equal(row, encoder.encode(game, seat))
            else:
                assert np.flatnonzero(row).tolist() == [offset + noop]


def test_buffers_must_match_shapes_and_dtypes():
    env = cpp.VecEnv([cpp.Scenario()], num_envs=2)
    buffers = env.allocate()
    assert buffers["observations"].shape == (4, env.observation_size)
    assert buffers["observations"].dtype == np.float32 and buffers["actions"].dtype == np.int32
    assert buffers["terminals"].dtype == np.uint8 and buffers["num_players"].shape == (2,)
    wrong = dict(buffers, rewards=np.zeros(4, dtype=np.float64))
    with pytest.raises(ValueError, match="rewards"):
        env.bind(**wrong)
    wrong = dict(buffers, actions=np.zeros(3, dtype=np.int32))
    with pytest.raises(ValueError, match="actions"):
        env.bind(**wrong)
    with pytest.raises(RuntimeError, match="reset"):
        env.step()


def test_flag_buffers_may_be_bool_like_pufferlibs():
    env = cpp.VecEnv([cpp.Scenario(num_players=2, turn_limit=10)], num_envs=3, seed=1)
    buffers = env.allocate()
    for name in ("terminals", "truncations", "acting", "learner"):
        buffers[name] = np.zeros(env.num_rows, dtype=np.bool_)
    env.bind(**buffers)
    env.reset()
    assert buffers["acting"].sum() == 3 and buffers["learner"].all()
    rng = np.random.default_rng(0)
    for _ in range(200):
        sample_actions(env, buffers, rng)
        env.step()
        if buffers["truncations"].any():
            break
    assert buffers["terminals"].dtype == np.bool_
    assert set(buffers["terminals"].view(np.uint8).tolist()) <= {0, 1}
    assert buffers["truncations"].any() and buffers["terminals"][buffers["truncations"]].all()


def test_mixed_scenarios_pad_rows_and_never_leave_forced_moves():
    maps = [cpp.CatanMap.from_spec(SHAPES[name][0](4)) for name in ("strip", "lake")]
    env = cpp.VecEnv(
        [
            cpp.Scenario(num_players=2, seats=["learner", "value"], shuffle_seats=True),
            cpp.Scenario(num_players=4, maps=maps, seats=["learner", "random", "learner", "F"],
                         weight=2.0),
            cpp.Scenario(num_players=3, map_type=cpp.MapType.MINI, seats=["R", "learner", "R"]),
        ],
        num_envs=6,
        seed=3,
        num_threads=3,
    )
    assert env.max_players == 4 and env.dims.max_nodes == 54
    buffers = env.allocate()
    env.reset()
    rng = np.random.default_rng(0)
    scenarios = set()
    for _ in range(400):
        check_rows(env, buffers)
        scenarios.update(env.scenario_index(e) for e in range(env.num_envs))
        sample_actions(env, buffers, rng)
        env.step()
    assert scenarios == {0, 1, 2}
    stats = env.pop_stats()
    assert stats["episodes"] > 0 and len(stats["scenarios"]) == 3
    assert sum(s["episodes"] for s in stats["scenarios"]) == stats["episodes"]
    assert env.pop_stats()["episodes"] == 0


def test_self_play_steps_match_a_replayed_game():
    """With every seat learning, a step is the action plus forced moves, and the
    winner's row gets the reward; copies replay the env's chance outcomes."""
    env = cpp.VecEnv([cpp.Scenario(num_players=3, rules=cpp.GameRules(vps_to_win=5),
                                   turn_limit=400)], num_envs=3, seed=9)
    buffers = env.allocate()
    env.reset()
    space = env.encoder.action_space
    rng = np.random.default_rng(1)
    finished = 0
    for _ in range(1_500):
        sample_actions(env, buffers, rng)
        expected = []
        for e in range(env.num_envs):
            game = env.game(e)
            game.execute(space.decode(game, int(buffers["actions"][e * 3 + game.current_player_index])))
            while (game.winning_color is None and game.num_turns < 400
                   and len(game.playable_actions) == 1):
                game.execute(game.playable_actions[0])
            expected.append(game)
        env.step()
        for e, game in enumerate(expected):
            rows = slice(e * 3, (e + 1) * 3)
            if game.winning_color is not None:
                assert buffers["terminals"][rows].all() and not buffers["truncations"][rows].any()
                rewards = np.zeros(3, dtype=np.float32)
                rewards[game.winning_player_index] = 1.0
                assert np.array_equal(buffers["rewards"][rows], rewards)
                finished += 1
            elif game.num_turns >= 400:
                # PufferLib reads terminals only: a truncation ends the trajectory too.
                assert buffers["truncations"][rows].all() and buffers["terminals"][rows].all()
                assert not buffers["rewards"][rows].any()
            else:
                assert not buffers["terminals"][rows].any()
                assert not buffers["rewards"][rows].any()
                assert env.game(e).search_equivalent(game)
    assert finished > 3


def test_truncation_at_the_turn_limit():
    env = cpp.VecEnv([cpp.Scenario(num_players=2, turn_limit=12)], num_envs=2, seed=4)
    buffers = env.allocate()
    env.reset()
    rng = np.random.default_rng(2)
    truncations = 0
    for _ in range(300):
        sample_actions(env, buffers, rng)
        env.step()
        truncations += int(buffers["truncations"].sum())
        assert np.array_equal(buffers["terminals"], buffers["truncations"])
        assert not buffers["rewards"].any()
        for e in range(env.num_envs):
            assert env.game(e).num_turns < 12
    assert truncations > 0
    stats = env.pop_stats()
    assert stats["truncated"] == 1.0 and stats["length"] == 12


@pytest.mark.parametrize("threads", [1, 4])
def test_runs_are_determined_by_the_seed(threads):
    def run(seed: int, num_threads: int) -> list[np.ndarray]:
        env = cpp.VecEnv(
            [
                cpp.Scenario(num_players=2, seats=["learner", "value"]),
                cpp.Scenario(num_players=4, map_type=cpp.MapType.BASE,
                             seats=["learner", "random", "learner", "random"],
                             shuffle_seats=True),
            ],
            num_envs=8, seed=seed, num_threads=num_threads,
        )
        buffers = env.allocate()
        env.reset()
        rng = np.random.default_rng(seed)
        trace = []
        for _ in range(150):
            sample_actions(env, buffers, rng)
            env.step()
            trace.append(np.concatenate([buffers["observations"].ravel(),
                                         buffers["rewards"], buffers["terminals"]]))
        return trace

    first = run(5, 1)
    second = run(5, threads)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    third = run(6, threads)
    assert not all(np.array_equal(a, b) for a, b in zip(first, third))


def test_resets_without_a_seed_start_new_games():
    def first_rows(env, buffers, seed=None):
        env.reset(seed)
        return buffers["observations"].copy(), [env.game(e).board.map.spec() for e in range(4)]

    env = cpp.VecEnv([cpp.Scenario(num_players=2)], num_envs=4, seed=7)
    buffers = env.allocate()
    first, first_maps = first_rows(env, buffers)
    second, second_maps = first_rows(env, buffers)
    assert first_maps != second_maps and not np.array_equal(first, second)
    replay, replay_maps = first_rows(env, buffers, seed=7)
    assert replay_maps == first_maps and np.array_equal(replay, first)
    again, again_maps = first_rows(env, buffers)
    assert again_maps == second_maps and np.array_equal(again, second)


def test_vp_shaping_rewards_learner_points():
    env = cpp.VecEnv([cpp.Scenario(num_players=2)], num_envs=4, seed=2, vp_reward=1.0)
    buffers = env.allocate()
    env.reset()
    rng = np.random.default_rng(3)
    shaped = 0.0
    for _ in range(400):
        sample_actions(env, buffers, rng)
        env.step()
        live = ~buffers["terminals"].astype(bool)
        shaped += float(buffers["rewards"][live].sum())
    assert shaped > 0


def test_unplayable_actions_are_refused_before_any_env_moves():
    env = cpp.VecEnv([cpp.Scenario(num_players=2)], num_envs=8, seed=1, num_threads=3)
    buffers = env.allocate()
    env.reset()
    rng = np.random.default_rng(5)
    for _ in range(30):
        sample_actions(env, buffers, rng)
        env.step()
    before = [env.game(e) for e in range(8)]
    sample_actions(env, buffers, rng)
    bad = int(np.flatnonzero(buffers["acting"])[5])
    good = buffers["actions"][bad]
    for action, message in ((env.encoder.action_space.noop, "not a game action"),
                            (len(env.encoder.action_space) + 3, "not a game action")):
        buffers["actions"][bad] = action
        with pytest.raises(ValueError, match=message):
            env.step()
        assert all(env.game(e).search_equivalent(before[e]) for e in range(8))
    # A legal-looking index that is not playable now.
    offset, (size,) = env.encoder.layout()["action_mask"]
    illegal = np.flatnonzero(buffers["observations"][bad, offset: offset + size] == 0)[0]
    buffers["actions"][bad] = illegal
    with pytest.raises(ValueError, match="not playable"):
        env.step()
    assert all(env.game(e).search_equivalent(before[e]) for e in range(8))
    # Fixing the action lets the same step go through.
    buffers["actions"][bad] = good
    env.step()
    assert sum(not env.game(e).search_equivalent(before[e]) for e in range(8)) == 8


def test_waiting_rows_can_be_encoded():
    env = cpp.VecEnv([cpp.Scenario(num_players=3)], num_envs=2, seed=1, encode_waiting=True)
    buffers = env.allocate()
    env.reset()
    for e in range(2):
        game = env.game(e)
        for seat in range(3):
            assert np.array_equal(buffers["observations"][e * 3 + seat],
                                  env.encoder.encode(game, seat))
