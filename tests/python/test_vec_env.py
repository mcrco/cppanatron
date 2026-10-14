"""VecEnv: rows, forced moves and bots, rewards and resets, determinism."""

from __future__ import annotations

import hashlib

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


def test_waiting_rows_stay_right_across_rebinds():
    """Waiting rows are only rewritten when they change, or when new buffers arrive."""
    env = cpp.VecEnv(
        [cpp.Scenario(num_players=2), cpp.Scenario(num_players=4, seats=["learner", "R"] * 2)],
        num_envs=4, seed=6, num_threads=2,
    )
    buffers = env.allocate()
    env.reset()
    rng = np.random.default_rng(7)
    for _ in range(40):
        sample_actions(env, buffers, rng)
        env.step()
        check_rows(env, buffers)
    sample_actions(env, buffers, rng)
    actions = buffers["actions"].copy()
    fresh = {name: np.full_like(array, 7) for name, array in buffers.items()}
    fresh["actions"][:] = actions
    env.bind(**fresh)
    env.step()
    check_rows(env, fresh)


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


def test_turn_limits_must_outlast_the_initial_placement():
    for players, limit in ((2, 2), (4, 6), (6, 10)):
        with pytest.raises(ValueError, match="initial placement"):
            cpp.VecEnv([cpp.Scenario(num_players=players, turn_limit=limit)])
        cpp.VecEnv([cpp.Scenario(num_players=players, turn_limit=limit + 1)])
    # The shortest allowed limit still gives the learner (last to place) decisions.
    env = cpp.VecEnv([cpp.Scenario(num_players=4, seats=["R", "R", "R", "learner"],
                                   turn_limit=7)], num_envs=2, seed=3)
    buffers = env.allocate()
    env.reset()
    rng = np.random.default_rng(1)
    for _ in range(50):
        sample_actions(env, buffers, rng)
        env.step()
    assert env.pop_stats()["truncated"] == 1.0


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


@pytest.mark.parametrize("threads", [1, 4, 7])
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


def test_crowded_maps_end_stuck_games_as_truncations():
    """Six seats fill the seven-tile board: random placements can leave a seat
    no open node, which ends the game as a stuck truncation."""
    crowded = cpp.CatanMap.from_spec(SHAPES["bare"][0](0))
    env = cpp.VecEnv([cpp.Scenario(num_players=6, maps=[crowded])], num_envs=8, seed=2)
    buffers = env.allocate()
    env.reset()
    rng = np.random.default_rng(4)
    stuck_steps = 0
    for _ in range(300):
        sample_actions(env, buffers, rng)
        env.step()
        stuck_steps += int(buffers["truncations"].any())
    stats = env.pop_stats()
    assert stats["stuck"] > 0 and stats["truncated"] >= stats["stuck"]
    assert stuck_steps > 0
    with pytest.raises(ValueError, match="cannot seat 3 players"):
        tiny = cpp.CatanMap.from_spec({"tiles": [{"coordinate": [0, 0, 0], "type": "land"}]})
        cpp.VecEnv([cpp.Scenario(num_players=3, maps=[tiny])])


def test_every_game_has_its_own_seeds():
    """Games take a game seed and a bot seed from the env's stream; the
    game replays from its game seed."""
    env = cpp.VecEnv([cpp.Scenario(num_players=3, seats=["learner", "R", "R"])],
                     num_envs=3, seed=11)
    buffers = env.allocate()
    env.reset()
    seeds = {(env.game_seed(e), env.bot_seed(e)) for e in range(3)}
    assert len(seeds) == 3 and all(g != b for g, b in seeds)
    for e in range(3):
        fresh = cpp.Game(3, map=env.game(e).board.map, seed=env.game_seed(e))
        assert fresh.board.robber_coordinate == env.game(e).board.robber_coordinate
    rng = np.random.default_rng(2)
    seen = set(seeds)
    for _ in range(400):
        sample_actions(env, buffers, rng)
        env.step()
        seen |= {(env.game_seed(e), env.bot_seed(e)) for e in range(3)}
    assert len(seen) > len(seeds)


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


def test_shaped_wins_stay_within_one():
    env = cpp.VecEnv([cpp.Scenario(num_players=2, rules=cpp.GameRules(vps_to_win=5))],
                     num_envs=8, seed=3, vp_reward=2.0)
    buffers = env.allocate()
    env.reset()
    rng = np.random.default_rng(6)
    wins = 0
    for _ in range(1_500):
        sample_actions(env, buffers, rng)
        env.step()
        won = buffers["terminals"].astype(bool) & ~buffers["truncations"].astype(bool)
        if won.any():
            assert buffers["rewards"][won].max() == 1.0
            wins += 1
    assert wins > 0


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


# ---------------------------------------------------------------- schedule mode


def row_policy(buffers: dict, offset: int, size: int) -> None:
    """A legal action for each acting row that depends on that row alone."""
    for row in np.flatnonzero(buffers["acting"]):
        observation = buffers["observations"][row]
        legal = np.flatnonzero(observation[offset: offset + size])
        digest = hashlib.blake2b(observation.tobytes(), digest_size=8).digest()
        buffers["actions"][row] = legal[int.from_bytes(digest, "little") % len(legal)]


def schedule(count: int) -> list[dict]:
    """Mixed games: maps, player counts, learners and bots (some bots only)."""
    maps = [cpp.MapSpec.build(cpp.MapType.BASE, seed) for seed in range(3)] + [
        cpp.MapSpec.from_dict(SHAPES["lake"][0](1))]
    kinds = [["learner", "R"], ["F", "learner"], ["R", "F"], ["learner", "learner"]]
    games = []
    for index in range(count):
        scenario = index % 2
        seats = kinds[index % len(kinds)] if scenario == 0 else ["R", "learner", "R", "learner"]
        games.append(dict(map=maps[index % len(maps)], game_seed=1_000 + index,
                          bot_seed=7 * index, seats=seats, scenario=scenario,
                          lineup_slots=list(range(len(seats)))[::-1]))
    return games


def play_schedule(num_envs: int, threads: int, games: list[dict]) -> dict:
    env = cpp.VecEnv(
        [cpp.Scenario(num_players=2, turn_limit=300), cpp.Scenario(num_players=4, turn_limit=300)],
        num_envs=num_envs, num_threads=threads, auto_reset=False,
        dims=cpp.EncoderDims(max_players=4, max_tiles=19, max_nodes=54, max_edges=72),
    )
    buffers = env.allocate()
    env.reset()
    assert env.idle_envs() == list(range(num_envs))
    offset, (size,) = env.encoder.layout()["action_mask"]
    pending = list(games)
    results = {}
    while pending or any(env.active(e) for e in range(num_envs)):
        idle = env.idle_envs()
        batch = [dict(game, env=e) for e, game in zip(idle, pending)]
        pending = pending[len(batch):]
        if batch:
            env.start_games(batch)
        for game in env.pop_finished():
            results[game["game_seed"]] = game
        row_policy(buffers, offset, size)
        env.step()
        for game in env.pop_finished():
            results[game["game_seed"]] = game
    return {seed: {k: v for k, v in game.items() if k != "env"} for seed, game in results.items()}


def test_scheduled_games_do_not_depend_on_envs_threads_or_history():
    games = schedule(24)
    reference = play_schedule(3, 1, games)
    assert sorted(reference) == sorted(g["game_seed"] for g in games)
    assert play_schedule(7, 4, games) == reference
    assert play_schedule(24, 3, games) == reference
    bots_only = [r for r in reference.values()
                 if cpp.SeatKind.LEARNER not in r["seats"]]
    assert bots_only and all(sum(r["decisions"]) == 0 for r in bots_only)
    for result in reference.values():
        assert result["winner_slot"] == (
            -1 if result["winner"] < 0 else result["lineup_slots"][result["winner"]])


def test_idle_envs_wait_and_scheduled_rows_carry_lineup_slots():
    env = cpp.VecEnv([cpp.Scenario(num_players=3)], num_envs=2, auto_reset=False)
    buffers = env.allocate()
    env.reset()
    noop = env.encoder.action_space.noop
    offset, _ = env.encoder.layout()["action_mask"]
    for name, value in (("acting", 0), ("learner", 0), ("seat", -1), ("lineup_slot", -1)):
        assert (buffers[name] == value).all(), name
    assert (buffers["num_players"] == 0).all()
    assert all(np.flatnonzero(row).tolist() == [offset + noop] for row in buffers["observations"])
    env.step()  # nothing to do
    with pytest.raises(RuntimeError, match="holds no game"):
        env.game(0)

    env.start_game(1, map=cpp.CatanMap.build(cpp.MapType.BASE, 4), game_seed=3,
                   seats=["R", "learner", cpp.SeatKind.VALUE], lineup_slots=[2, 0, 1])
    assert env.idle_envs() == [0] and env.lineup_slots(1) == [2, 0, 1]
    rows = slice(env.max_players, 2 * env.max_players)
    assert buffers["lineup_slot"][rows].tolist() == [2, 0, 1]
    assert buffers["learner"][rows].tolist() == [0, 1, 0]
    assert buffers["acting"][rows].tolist() == [0, 1, 0] and buffers["num_players"][1] == 3

    with pytest.raises(ValueError, match="busy"):
        env.start_game(1, map=cpp.CatanMap.build(cpp.MapType.BASE), game_seed=1,
                       seats=["learner"] * 3)
    with pytest.raises(ValueError, match="each of the scenario's 3 players"):
        env.start_game(0, map=cpp.CatanMap.build(cpp.MapType.BASE), game_seed=1,
                       seats=["learner"] * 2)
    with pytest.raises(ValueError, match="busy"):
        env.start_games([dict(env=0, map=cpp.MapSpec.build(cpp.MapType.BASE), game_seed=1,
                              seats=["learner"] * 3)] * 2)
    assert env.idle_envs() == [0]
    with pytest.raises(RuntimeError, match="schedule mode"):
        auto = cpp.VecEnv([cpp.Scenario()])
        auto.allocate()
        auto.reset()
        auto.start_game(0, map=cpp.CatanMap.build(cpp.MapType.BASE), game_seed=1,
                        seats=["learner", "R"])


def test_auto_reset_lineup_slots_name_the_scenario_seats():
    kinds = ["learner", "R", "F"]
    env = cpp.VecEnv([cpp.Scenario(num_players=3, seats=kinds, shuffle_seats=True)],
                     num_envs=6, seed=5)
    buffers = env.allocate()
    env.reset()
    names = {cpp.SeatKind.LEARNER: "learner", cpp.SeatKind.RANDOM: "R", cpp.SeatKind.VALUE: "F"}
    orders = set()
    for e in range(6):
        slots = env.lineup_slots(e)
        assert sorted(slots) == [0, 1, 2]
        assert [names[k] for k in env.seat_kinds(e)] == [kinds[s] for s in slots]
        assert buffers["lineup_slot"][e * 3: e * 3 + 3].tolist() == slots
        orders.add(tuple(slots))
    assert len(orders) > 1


# ---------------------------------------------------------------- asymmetric critic


def test_critic_rows_carry_the_players_block_at_the_critic_level():
    env = cpp.VecEnv([cpp.Scenario(num_players=3, seats=["learner", "learner", "R"])],
                     num_envs=3, seed=4, level=cpp.InfoLevel.PRIVATE,
                     critic_level=cpp.InfoLevel.FULL)
    plain = cpp.VecEnv([cpp.Scenario(num_players=3)], num_envs=1)
    assert plain.critic_level is None and "critic_players" not in plain.layout()
    critic = cpp.GraphEncoder(env.dims, cpp.InfoLevel.FULL)
    width = len(critic.feature_names("players"))
    offset, shape = env.layout()["critic_players"]
    assert (offset, shape) == (env.encoder.size, (env.max_players, width))
    assert env.observation_size == env.encoder.size + env.max_players * width
    assert env.critic_level == cpp.InfoLevel.FULL and env.level == cpp.InfoLevel.PRIVATE

    buffers = env.allocate()
    env.reset()
    rng = np.random.default_rng(3)
    players_offset, _ = critic.layout()["players"]
    mask_offset, _ = env.encoder.layout()["action_mask"]
    noop = env.encoder.action_space.noop
    differ = 0
    for _ in range(150):
        for row, observation in enumerate(buffers["observations"]):
            e, seat = divmod(row, env.max_players)
            if buffers["acting"][row]:
                game = env.game(e)
                actor = env.encoder.encode(game, seat)
                full = critic.encode(game, seat)
                assert np.array_equal(observation[:offset], actor)
                assert np.array_equal(observation[offset:],
                                      full[players_offset: players_offset + shape[0] * width])
                assert np.array_equal(observation[offset:], critic.encode_players(game, seat))
                differ += not np.array_equal(observation[offset:],
                                             actor[players_offset: players_offset + shape[0] * width])
            else:
                assert np.flatnonzero(observation).tolist() == [mask_offset + noop]
        sample_actions(env, buffers, rng)
        env.step()
    assert differ > 0  # the full-information critic sees what the private actor does not
