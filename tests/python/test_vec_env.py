"""VecEnv: rows, forced moves and bots, rewards and resets, determinism."""

from __future__ import annotations

import hashlib
import itertools

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


def test_expert_actions_label_acting_rows_with_fs_choice():
    env = cpp.VecEnv(
        [cpp.Scenario(num_players=2, seats=["learner", "R"]), cpp.Scenario(num_players=3)],
        num_envs=6, seed=8, num_threads=3)
    buffers = env.allocate()
    env.reset()
    space = env.encoder.action_space
    rng = np.random.default_rng(1)
    for step in range(120):
        enemies = cpp.ValueEnemies.ALL if step % 2 else cpp.ValueEnemies.NEXT
        labels = env.expert_actions(enemies=enemies)
        assert labels.dtype == np.int32 and labels.shape == (env.num_rows,)
        assert ((labels >= 0) == buffers["acting"].astype(bool)).all()
        for row in np.flatnonzero(buffers["acting"]):
            game = env.game(row // env.max_players)
            assert labels[row] == space.index(game, cpp.value_action(game, enemies=enemies))
        out = np.full(env.num_rows, 99, dtype=np.int32)
        assert env.expert_actions(out) is out and np.array_equal(
            out, env.expert_actions(enemies=cpp.ValueEnemies.NEXT))
        # Follow F's labels half the time, so play goes on.
        sample_actions(env, buffers, rng)
        if step % 2:
            buffers["actions"][labels >= 0] = labels[labels >= 0]
        env.step()


# ---------------------------------------------------------------- hand trackers


def test_trackers_follow_every_transition_like_a_replay():
    """With every seat learning, a step is the action plus forced moves; the
    env's trackers equal trackers fed that replay, options included."""
    rules = cpp.GameRules(vps_to_win=6, friendly_robber=True)
    env = cpp.VecEnv([cpp.Scenario(num_players=3, rules=rules, turn_limit=300)],
                     num_envs=3, seed=12, trackers=True)
    buffers = env.allocate()
    env.reset()
    space = env.encoder.action_space
    rng = np.random.default_rng(4)
    compared = 0
    for _ in range(600):
        sample_actions(env, buffers, rng)
        expected = []
        for e in range(env.num_envs):
            game = env.game(e)
            trackers = [env.tracker(e, seat) for seat in range(3)]

            def play(action):
                before = game.copy()
                game.execute(action)
                for tracker in trackers:
                    tracker.observe(before, action, game)
                if game.winning_color is None:
                    trackers[game.current_player_index].observe_options(
                        game, game.playable_actions)

            play(space.decode(game, int(buffers["actions"][e * 3 + game.current_player_index])))
            while (game.winning_color is None and game.num_turns < 300
                   and len(game.playable_actions) == 1):
                play(game.playable_actions[0])
            expected.append((game, trackers))
        env.step()
        for e, (game, trackers) in enumerate(expected):
            if buffers["terminals"][e * 3]:
                continue
            assert env.game(e).search_equivalent(game)
            for seat in range(3):
                assert env.tracker(e, seat) == trackers[seat]
                assert env.tracker(e, seat).consistent_with(game)
            compared += 1
    assert compared > 1_000


def test_trackers_see_bot_moves_and_feed_search():
    env = cpp.VecEnv(
        [cpp.Scenario(num_players=4, seats=["learner", "R", "F", "learner"], shuffle_seats=True,
                      rules=cpp.GameRules(friendly_robber=True))],
        num_envs=4, seed=21, num_threads=2, trackers=True)
    buffers = env.allocate()
    env.reset()
    rng = np.random.default_rng(9)
    codec = env.encoder.action_space
    pool = cpp.SearchPool(codec, env.encoder,
                          cpp.SearchConfig(simulations=8, max_players=4, determinizations=2))
    searched = 0
    for step in range(250):
        for e in range(env.num_envs):
            game = env.game(e)
            for seat, kind in enumerate(env.seat_kinds(e)):
                if kind != cpp.SeatKind.LEARNER:
                    with pytest.raises(IndexError, match="learner seats"):
                        env.tracker(e, seat)
                    continue
                tracker = env.tracker(e, seat)
                assert tracker.observer == seat and tracker.consistent_with(game)
                tracker.check_public(game)
            if step % 50 == 0:
                seat = game.current_player_index
                world = cpp.sample_world(game, env.tracker(e, seat), seed=step)
                assert world.current_player_index == seat
                pool.add(game, step, env.tracker(e, seat))
                searched += 1
        sample_actions(env, buffers, rng)
        env.step()
    observations = np.zeros((32, pool.observation_size), dtype=np.float32)
    ids = np.zeros(32, dtype=np.int64)
    while n := pool.collect(observations, ids):
        pool.apply(np.zeros((n, len(codec)), dtype=np.float32),
                   np.full((n, 4), 0.25, dtype=np.float32))
    assert searched == 20

    plain = cpp.VecEnv([cpp.Scenario()], num_envs=1)
    plain.allocate()
    plain.reset()
    with pytest.raises(RuntimeError, match="no trackers"):
        plain.tracker(0, 0)


def test_actor_critic_encoder_writes_vec_env_rows_and_search_leaves():
    env = cpp.VecEnv([cpp.Scenario(num_players=3)], num_envs=2, seed=2,
                     level=cpp.InfoLevel.PUBLIC, critic_level=cpp.InfoLevel.FULL)
    buffers = env.allocate()
    env.reset()
    encoder = cpp.ActorCriticEncoder(env.dims, cpp.InfoLevel.PUBLIC, cpp.InfoLevel.FULL)
    assert isinstance(encoder, cpp.ObservationEncoder)
    assert encoder.layout() == env.layout() and encoder.size == env.observation_size
    assert isinstance(env.row_encoder, cpp.ActorCriticEncoder)
    assert isinstance(cpp.VecEnv([cpp.Scenario()]).row_encoder, cpp.GraphEncoder)
    names = encoder.feature_names()
    assert names["critic_players"] == names["players"]
    rng = np.random.default_rng(5)
    for _ in range(60):
        for row in np.flatnonzero(buffers["acting"]):
            game = env.game(row // env.max_players)
            seat = row % env.max_players
            assert np.array_equal(buffers["observations"][row], encoder.encode(game, seat))
        sample_actions(env, buffers, rng)
        env.step()

    # As SearchPool's encoder, the root's row is the training row.
    game = env.game(0)
    pool = cpp.SearchPool(encoder.action_space, encoder,
                          cpp.SearchConfig(simulations=4, max_players=3))
    assert pool.observation_size == encoder.size
    pool.add(game, 1)
    observations = np.zeros((4, pool.observation_size), dtype=np.float32)
    ids = np.zeros(4, dtype=np.int64)
    assert pool.collect(observations, ids) >= 1
    assert np.array_equal(observations[0], encoder.encode(game, game.current_player_index))


@pytest.mark.parametrize("critic", [False, True])
def test_dice_state_rows_and_search_leaves_carry_the_dice_state(critic):
    """VecEnv(dice_state=True) writes the dice-state encoder's rows (with a critic block too), on
    balanced dice, and SearchPool leaves from that encoder are the same rows."""
    balanced = cpp.GameRules(dice_mode=cpp.DiceMode.BALANCED)
    options = {"critic_level": cpp.InfoLevel.FULL} if critic else {}
    env = cpp.VecEnv([cpp.Scenario(num_players=2, rules=balanced)], num_envs=3, seed=6,
                     level=cpp.InfoLevel.PUBLIC, dice_state=True, **options)
    plain = cpp.VecEnv([cpp.Scenario(num_players=2, rules=balanced)], num_envs=1,
                       level=cpp.InfoLevel.PUBLIC, **options)
    assert env.dice_state and not plain.dice_state and env.encoder.dice_state
    encoder = (cpp.ActorCriticEncoder(env.dims, cpp.InfoLevel.PUBLIC, cpp.InfoLevel.FULL, dice_state=True)
               if critic else cpp.GraphEncoder(env.dims, cpp.InfoLevel.PUBLIC, dice_state=True))
    assert type(env.row_encoder) is type(encoder) and env.row_encoder.dice_state
    assert encoder.layout() == env.layout()
    assert env.observation_size == plain.observation_size + (14 * 2 + 23) + 19 * (2 + 2)
    buffers = env.allocate()
    env.reset()
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(120):
        for row in np.flatnonzero(buffers["acting"]):
            game = env.game(row // env.max_players)
            assert np.array_equal(buffers["observations"][row], encoder.encode(game, row % env.max_players))
            checked += 1
        sample_actions(env, buffers, rng)
        env.step()
    assert checked > 100
    game = env.game(0)
    pool = cpp.SearchPool(encoder.action_space, encoder, cpp.SearchConfig(simulations=4, max_players=2))
    assert pool.observation_size == encoder.size
    pool.add(game, 1)
    observations = np.zeros((4, pool.observation_size), dtype=np.float32)
    ids = np.zeros(4, dtype=np.int64)
    assert pool.collect(observations, ids) >= 1
    assert np.array_equal(observations[0], encoder.encode(game, game.current_player_index))


# ---------------------------------------------------------------- set_action (domestic trades)


PLAY_TURN = cpp.ActionPrompt.PLAY_TURN
DECIDE_TRADE = cpp.ActionPrompt.DECIDE_TRADE
DECIDE_ACCEPTEES = cpp.ActionPrompt.DECIDE_ACCEPTEES


def scheduled(seats: list[str], *, seed: int, num_envs: int = 1, games: int | None = None,
              turn_limit: int = 60, **options) -> tuple[cpp.VecEnv, dict]:
    """A schedule-mode VecEnv with one base-map game of `seats` in each of the
    first `games` envs (all of them by default)."""
    env = cpp.VecEnv([cpp.Scenario(num_players=len(seats), turn_limit=turn_limit)],
                     num_envs=num_envs, auto_reset=False, **options)
    buffers = env.allocate()
    env.reset()
    env.start_games([dict(env=e, map=cpp.CatanMap.build(cpp.MapType.BASE, seed + e),
                          game_seed=seed + e, seats=seats)
                     for e in range(num_envs if games is None else games)])
    return env, buffers


class Driver:
    """Steps a VecEnv with a fixed policy (roll; else buy a development card,
    so a knight in hand makes a pre-roll row a decision; else end the turn,
    so hands grow and trades come up; else the row's hash choice) and counts
    each env's decisions per seat the way pop_finished does: one per step a
    row acted."""

    def __init__(self, env: cpp.VecEnv, buffers: dict):
        self.env, self.buffers = env, buffers
        self.offset, (self.size,) = env.encoder.layout()["action_mask"]
        self.decisions: dict[int, list[int]] = {}

    def policy(self) -> None:
        """An index for every acting row that has one; a trade prompt (no
        legal index) is left to set_action."""
        space = self.env.encoder.action_space
        for row in np.flatnonzero(self.buffers["acting"]):
            observation = self.buffers["observations"][row]
            legal = np.flatnonzero(observation[self.offset: self.offset + self.size])
            if len(legal) == 0:
                continue
            game = self.env.game(int(row) // self.env.max_players)
            digest = hashlib.blake2b(observation.tobytes(), digest_size=8).digest()
            self.buffers["actions"][row] = legal[int.from_bytes(digest, "little") % len(legal)]
            for wanted in (cpp.ActionType.ROLL, cpp.ActionType.BUY_DEVELOPMENT_CARD,
                           cpp.ActionType.END_TURN):
                action = next((a for a in game.playable_actions if a.action_type == wanted), None)
                if action is not None:
                    self.buffers["actions"][row] = space.index(game, action)
                    break

    def step(self) -> None:
        for row in np.flatnonzero(self.buffers["acting"]):
            e, seat = divmod(int(row), self.env.max_players)
            counts = self.decisions.setdefault(e, [0] * len(self.env.game(e).colors))
            counts[seat] += 1
        self.env.step()

    def until(self, ready, limit: int = 1_000) -> None:
        for _ in range(limit):
            if ready():
                return
            self.policy()
            self.step()
        raise AssertionError("the wanted position never came up")

    def finish(self) -> None:
        self.until(lambda: not any(self.env.active(e) for e in range(self.env.num_envs)),
                   limit=20_000)


def offers_at(game: cpp.Game) -> list[tuple[int, int, list[int]]]:
    """Every one-for-one offer the seat to move can pay after its roll, with
    the other seats holding the asked kind (who may accept); [] elsewhere."""
    seat = game.current_player_index
    players = game.players
    if game.current_prompt != PLAY_TURN or not players[seat].has_rolled:
        return []
    return [(give, get, [s for s in range(len(players))
                         if s != seat and players[s].resources[get] > 0])
            for give in range(5) for get in range(5)
            if give != get and players[seat].resources[give] > 0]


def bundle(give: int, get: int) -> tuple[int, ...]:
    counts = [0] * 10
    counts[give] = 1
    counts[5 + get] = 1
    return tuple(counts)


def one_for_one(game: cpp.Game, give: int, get: int) -> cpp.Action:
    return cpp.Action(game.current_color, cpp.ActionType.OFFER_TRADE, bundle(give, get))


def answer(game: cpp.Game, accept: bool) -> cpp.Action:
    wanted = cpp.ActionType.ACCEPT_TRADE if accept else cpp.ActionType.REJECT_TRADE
    return next(a for a in game.playable_actions if a.action_type == wanted)


def forced_moves(game: cpp.Game, turn_limit: int) -> list[cpp.Action]:
    """Play the game's lone legal actions, as the env does inside a step."""
    played = []
    while (game.winning_color is None and game.num_turns < turn_limit
           and len(game.playable_actions) == 1):
        played.append(game.playable_actions[0])
        game.execute(played[-1])
    return played


def test_set_action_plays_a_negotiation_through_the_learner_rows():
    """An injected offer prompts the learners who can pay (the others' lone
    REJECT is a forced move inside the env), their answers and the confirm go
    through the same rows, and only the rows that acted count as decisions."""
    env, buffers = scheduled(["learner"] * 3, seed=31, turn_limit=40)
    driver = Driver(env, buffers)
    encoder = env.encoder
    mask_offset, (mask_size,) = encoder.layout()["action_mask"]
    global_offset, _ = encoder.layout()["global"]
    names = encoder.feature_names("global")
    assert names[5] == "prompt_decide_trade" and names[6] == "prompt_decide_acceptees"

    def split(offer):
        give, get, payers = offer
        return bool(payers) and len(payers) < 2  # one other seat can pay, one cannot

    driver.until(lambda: any(split(o) for o in offers_at(env.game(0))))
    game = env.game(0)
    give, get, payers = next(o for o in offers_at(game) if split(o))
    proposer = game.current_player_index
    colors = game.colors
    hands_before = [list(p.resources) for p in game.players]
    bank_before = game.resource_bank
    counted_before = list(driver.decisions[0])
    forced: list[cpp.Action] = []
    acted = []

    def inject(action: cpp.Action) -> cpp.Game:
        """Plays `action` in the env and on the copy; returns the copy right
        after it, before the forced moves the env plays on in the same step."""
        buffers["actions"][:] = -1  # never read for an overridden row
        acted.append(game.current_player_index)
        env.set_action(0, action)
        driver.step()
        game.execute(action)
        post = game.copy()
        forced.extend(forced_moves(game, 40))
        assert env.game(0).search_equivalent(game)
        return post

    def acting_row(prompt_flags: list[int]) -> None:
        rows = np.flatnonzero(buffers["acting"]).tolist()
        assert rows == [game.current_player_index]
        row = buffers["observations"][rows[0]]
        assert np.array_equal(row, encoder.encode(game, rows[0]))
        assert not row[mask_offset: mask_offset + mask_size].any()  # no legal index, no no-op
        assert row[global_offset: global_offset + 7].tolist() == prompt_flags
        assert len(game.playable_actions) == 2

    inject(one_for_one(game, give, get))
    acceptees = []
    while game.current_prompt == DECIDE_TRADE:
        seat = game.current_player_index
        acting_row([0, 0, 0, 0, 0, 1, 0])
        assert seat in payers or seat == proposer  # a seat prompted on its own offer may pay
        if seat != proposer:
            acceptees.append(seat)
        inject(answer(game, accept=seat != proposer))
    assert acceptees == payers
    assert forced and all(a.action_type == cpp.ActionType.REJECT_TRADE for a in forced)
    assert game.current_prompt == DECIDE_ACCEPTEES and game.current_player_index == proposer
    acting_row([0, 0, 0, 0, 0, 0, 1])

    partner = acceptees[0]
    # Built in Python, so it names no offering seat; set_action matches the listed one.
    after = inject(cpp.Action(colors[proposer], cpp.ActionType.CONFIRM_TRADE,
                              (*bundle(give, get), colors[partner])))
    assert after.current_prompt == PLAY_TURN and after.current_player_index == proposer
    assert after.players[proposer].has_rolled
    expected = [list(hand) for hand in hands_before]
    expected[proposer][give] -= 1
    expected[proposer][get] += 1
    expected[partner][give] += 1
    expected[partner][get] -= 1
    assert [list(p.resources) for p in after.players] == expected
    assert after.resource_bank == bank_before

    negotiation = [0, 0, 0]
    for seat in acted:
        negotiation[seat] += 1
    assert [a - b for a, b in zip(driver.decisions[0], counted_before)] == negotiation
    assert negotiation[proposer] >= 2 and sum(negotiation) == 2 + len(acceptees) + (
        proposer in acted[1:-1])
    driver.finish()
    (finished,) = env.pop_finished()
    assert finished["decisions"] == driver.decisions[0]


def test_set_action_refusals_leave_every_env_as_it_was():
    env, buffers = scheduled(["learner"] * 3, seed=5, num_envs=3, games=2, turn_limit=200,
                             num_threads=2)
    driver = Driver(env, buffers)
    space = env.encoder.action_space

    def pre_roll() -> bool:
        game = env.game(0)
        player = game.players[game.current_player_index]
        return game.current_prompt == PLAY_TURN and not player.has_rolled and any(player.resources)

    def post_roll() -> bool:
        return bool(offers_at(env.game(0)))

    def refused(env_index: int, action: cpp.Action, message: str) -> None:
        env.set_action(env_index, action)
        with pytest.raises(ValueError, match=message):
            env.step()
        assert all(env.game(e).search_equivalent(games[e]) for e in range(2))
        assert not env.active(2)
        env.clear_action(env_index)

    def replayed(e: int, limit: int = 200) -> cpp.Game:
        """Env e's game after the index step the policy pending in its row."""
        game = games[e].copy()
        row = e * env.max_players + game.current_player_index
        game.execute(space.decode(game, int(buffers["actions"][row])))
        forced_moves(game, limit)
        return game

    driver.until(pre_roll)
    games = [env.game(e) for e in range(2)]
    game = games[0]
    seat = game.current_player_index
    give = next(k for k, n in enumerate(game.players[seat].resources) if n > 0)
    get = (give + 1) % 5
    driver.policy()  # env 1's index stays pending through every refusal
    refused(0, one_for_one(game, give, get), "offer_trade is not playable")  # before the roll
    refused(0, cpp.Action(game.current_color, cpp.ActionType.REJECT_TRADE, (0,) * 10 + (seat,)),
            "not playable now")
    refused(2, one_for_one(game, give, get), "idle env")
    with pytest.raises(IndexError):
        env.set_action(3, one_for_one(game, give, get))
    env.clear_action(2)  # nothing pending is fine

    driver.until(post_roll)
    games = [env.game(e) for e in range(2)]
    game = games[0]
    seat = game.current_player_index
    give, get, _ = offers_at(game)[0]
    driver.policy()
    too_many = list(bundle(give, get))
    too_many[give] = game.players[seat].resources[give] + 1
    refused(0, cpp.Action(game.current_color, cpp.ActionType.OFFER_TRADE, tuple(too_many)),
            "more than the seat holds")
    refused(0, cpp.Action(game.colors[(seat + 1) % 3], cpp.ActionType.OFFER_TRADE,
                          bundle(give, get)), "not the seat to move")
    # Expert labels ignore a pending override.
    labels = env.expert_actions()
    env.set_action(0, one_for_one(game, give, get))
    assert np.array_equal(env.expert_actions(), labels)
    assert labels[seat] == space.index(game, cpp.value_action(game))
    # Fixing the action lets the same step go through, env 1 by its index.
    driver.step()
    expected = games[0].copy()
    expected.execute(one_for_one(game, give, get))
    forced_moves(expected, 200)
    assert env.game(0).search_equivalent(expected)
    assert env.game(1).search_equivalent(replayed(1))
    while env.game(0).current_prompt in (DECIDE_TRADE, DECIDE_ACCEPTEES):
        env.set_action(0, env.game(0).playable_actions[0])  # REJECT, then CANCEL
        driver.policy()
        driver.step()

    # A cleared override leaves the index path alone, even one that would be refused.
    games = [env.game(e) for e in range(2)]
    game = games[0]
    env.set_action(0, cpp.Action(game.current_color, cpp.ActionType.REJECT_TRADE,
                                 (0,) * 10 + (game.current_player_index,)))
    env.clear_action(0)
    driver.policy()
    driver.step()
    assert env.game(0).current_prompt not in (DECIDE_TRADE, DECIDE_ACCEPTEES)
    assert all(env.game(e).search_equivalent(replayed(e)) for e in range(2))


def test_trackers_observe_an_injected_negotiation():
    env, buffers = scheduled(["learner"] * 3, seed=19, turn_limit=60, trackers=True)
    driver = Driver(env, buffers)
    driver.until(lambda: any(payers for *_, payers in offers_at(env.game(0))))
    game = env.game(0)
    give, get, payers = next(o for o in offers_at(game) if o[2])
    proposer = game.current_player_index
    trackers = [env.tracker(0, seat) for seat in range(3)]

    def play(action: cpp.Action) -> None:
        before = game.copy()
        game.execute(action)
        for tracker in trackers:
            tracker.observe(before, action, game)
        if game.winning_color is None:
            trackers[game.current_player_index].observe_options(game, game.playable_actions)

    def inject(action: cpp.Action) -> cpp.Game:
        env.set_action(0, action)
        driver.step()
        play(action)
        post = game.copy()
        while game.winning_color is None and len(game.playable_actions) == 1:
            play(game.playable_actions[0])
        assert env.game(0).search_equivalent(game)
        for seat in range(3):
            assert env.tracker(0, seat) == trackers[seat]
            assert env.tracker(0, seat).consistent_with(game)
            env.tracker(0, seat).check_public(game)
        return post

    inject(one_for_one(game, give, get))
    while game.current_prompt == DECIDE_TRADE:
        inject(answer(game, accept=game.current_player_index != proposer))
    assert game.current_prompt == DECIDE_ACCEPTEES
    partner = payers[0]
    held = game.players[partner].resources[give]
    post = inject(cpp.Action(game.colors[proposer], cpp.ActionType.CONFIRM_TRADE,
                             (*bundle(give, get), game.colors[partner])))
    assert post.current_prompt == PLAY_TURN
    assert post.players[partner].resources[give] == held + 1


def test_f_responders_reject_every_injected_offer():
    """F keeps the listed REJECT (ACCEPT moves no cards until the confirm, so
    it scores no higher): the step lands back on the proposer's PLAY_TURN with
    the game as it was, and the offer counts as the proposer's decision."""
    env, buffers = scheduled(["learner", "F", "F"], seed=23, turn_limit=40)
    driver = Driver(env, buffers)
    driver.until(lambda: any(payers for *_, payers in offers_at(env.game(0))))
    game = env.game(0)
    give, get, payers = next(o for o in offers_at(game) if o[2])
    assert payers  # an F that could have accepted
    buffers["actions"][:] = -1
    env.set_action(0, one_for_one(game, give, get))
    driver.step()
    after = env.game(0)
    assert after.current_prompt == PLAY_TURN and after.current_player_index == 0
    assert after.search_equivalent(game)  # a rejected offer restores the state, no RNG moves
    assert buffers["acting"].tolist() == [1, 0, 0]
    assert np.array_equal(buffers["observations"][0], env.encoder.encode(after, 0))
    driver.finish()
    (finished,) = env.pop_finished()
    assert finished["decisions"] == driver.decisions[0]
    assert finished["decisions"][1:] == [0, 0]


def test_last_outcome_is_the_game_an_env_finished_last():
    """Each seat's victory points at the end (hidden cards counted), the turns played, whether
    the game was cut, and how many games the env has finished; num_turns is the current game's."""
    env = cpp.VecEnv([cpp.Scenario(num_players=3, rules=cpp.GameRules(vps_to_win=5),
                                   turn_limit=150)], num_envs=3, seed=9)
    buffers = env.allocate()
    env.reset()
    rng = np.random.default_rng(1)
    assert all(env.last_outcome(e)["games"] == 0 for e in range(3))
    finished = [0, 0, 0]
    won = cut = 0
    for _ in range(4_000):
        sample_actions(env, buffers, rng)
        before = [env.num_turns(e) for e in range(3)]
        assert before == [env.game(e).num_turns for e in range(3)]
        env.step()
        for e in range(3):
            last = env.last_outcome(e)
            if not buffers["terminals"][e * 3]:
                assert last["games"] == finished[e]
                continue
            finished[e] += 1
            points = last["victory_points"]
            assert last["games"] == finished[e] and last["vps_to_win"] == 5 and len(points) == 3
            assert last["truncated"] == bool(buffers["truncations"][e * 3]) and last["turns"] >= before[e]
            rewards = buffers["rewards"][e * 3: e * 3 + 3]
            if last["truncated"]:
                assert last["turns"] == 150 and max(points) < 5
                cut += 1
            else:
                winner = int(np.argmax(rewards))  # the winner's seat, as the rewards have it
                assert rewards[winner] == 1.0 and last["turns"] < 150
                assert [p >= 5 for p in points] == [seat == winner for seat in range(3)]
                won += 1
    assert won >= 5 and cut >= 1


def award_holders(game: cpp.Game) -> tuple[int | None, int | None]:
    """The seats holding Largest Army and Longest Road, None while nobody does."""
    army = [seat for seat, player in enumerate(game.players) if player.has_army]
    road = [seat for seat, player in enumerate(game.players) if player.has_road]
    return (army[0] if army else None, road[0] if road else None)


def changes_of_hands(holders: list[int | None]) -> int:
    """Times an award passed from one seat to another in a sequence of its holders."""
    held = [holder for holder in holders if holder is not None]
    return sum(a != b for a, b in itertools.pairwise(held))


@pytest.mark.parametrize("seats,vps,turn_limit,deck,trackers", [
    (["learner"] * 2, 10, 300, [14, 2, 2, 2, 5], False),
    (["learner"] * 3, 8, 300, [14, 0, 1, 1, 2], True),
    (["learner"] * 4, 6, 200, [12, 1, 1, 1, 1], False),
    (["learner", "value"], 10, 70, [8, 1, 1, 1, 2], False),
    (["value", "learner", "value"], 10, 70, [8, 1, 1, 1, 2], True)])
def test_style_stats_match_a_recount_of_every_move(seats, vps, turn_limit, deck, trackers):
    """pop_stats' style means against a recount on copies of the games, which replay each
    step's action, the forced moves and F's: the award holders after every move, the
    development cards bought and the final victory points. Truncated games count everywhere
    but in the victory points; the first award of Largest Army or Longest Road is not a
    change of hands."""
    players = len(seats)
    rules = cpp.GameRules(vps_to_win=vps, development_deck=deck)
    env = cpp.VecEnv([cpp.Scenario(num_players=players, rules=rules, turn_limit=turn_limit,
                                   seats=seats)],
                     num_envs=4, seed=3, dims=cpp.EncoderDims(max_players=players),
                     trackers=trackers)
    buffers = env.allocate()
    env.reset()
    space = env.encoder.action_space
    rng = np.random.default_rng(1)
    holders = [[] for _ in range(env.num_envs)]  # after every move of each env's current game
    bought = [0] * env.num_envs
    seen = dict.fromkeys(("won", "cut", "army_changes", "road_changes", "dev_deck_empty"), 0)
    for _ in range(600 if "value" in seats else 3_000):
        sample_actions(env, buffers, rng)
        ended = []
        for e in range(env.num_envs):
            game = env.game(e)
            row = e * players + game.current_player_index
            action = space.decode(game, int(buffers["actions"][row]))
            while True:
                bought[e] += action.action_type == cpp.ActionType.BUY_DEVELOPMENT_CARD
                game.execute(action)
                holders[e].append(award_holders(game))
                over = game.winning_color is not None or game.num_turns >= turn_limit
                if over:
                    break
                if seats[game.current_player_index] == "value":
                    action = cpp.value_action(game)
                elif len(game.playable_actions) == 1:
                    action = game.playable_actions[0]
                else:
                    break
            if over:
                ended.append((e, game))
        env.step()
        stats = env.pop_stats()
        assert stats["episodes"] == len(ended)
        expected = dict.fromkeys(("dev_bought", "dev_deck_empty", "army_changes", "road_changes",
                                  "army_awarded", "road_awarded"), 0)
        points = dict.fromkeys(("winner_vp", "loser_vp", "total_vp"), 0.0)
        won = 0
        for e, game in ended:
            army, road = zip(*holders[e])
            expected["dev_bought"] += bought[e]
            expected["dev_deck_empty"] += game.development_cards_remaining == 0
            expected["army_changes"] += changes_of_hands(army)
            expected["road_changes"] += changes_of_hands(road)
            expected["army_awarded"] += army[-1] is not None
            expected["road_awarded"] += road[-1] is not None
            holders[e], bought[e] = [], 0
            if game.winning_color is None:
                continue
            won += 1
            final = [player.actual_victory_points for player in game.players]
            winner = final.pop(game.winning_player_index)
            assert winner >= vps > max(final)
            points["winner_vp"] += winner
            points["loser_vp"] += sum(final) / len(final)
            points["total_vp"] += winner + sum(final)
        for key, total in expected.items():
            assert stats[key] * len(ended) == pytest.approx(total), key
            seen[key] = seen.get(key, 0) + total
        for key, total in points.items():
            assert stats[key] * won == pytest.approx(total), key
        assert stats["truncated"] * len(ended) == pytest.approx(len(ended) - won)
        seen["won"] += won
        seen["cut"] += len(ended) - won
    # against F here Largest Army stays where it went and the deck never runs out
    rare = ("army_changes", "dev_deck_empty") if "value" in seats else ()
    assert all(count > 0 for key, count in seen.items() if key not in rare), seen


#: The episode means pop_stats gave before it counted the style of play.
EPISODE_KEYS = ("episodes", "truncated", "length", "decisions", "learner_win", "learner_seats",
                "stuck", "seat_win")
ROLLOUTS = {
    # OpenHex's 1v1 rules with the dice input
    "2p friendly robber, balanced dice": lambda: cpp.VecEnv(
        [cpp.Scenario(num_players=2, turn_limit=300, rules=cpp.GameRules(
            vps_to_win=10, discard_limit=9, friendly_robber=True,
            dice_mode=cpp.DiceMode.BALANCED))],
        num_envs=6, seed=11, dice_state=True),
    "3p against F and R": lambda: cpp.VecEnv(
        [cpp.Scenario(num_players=3, seats=["learner", "value", "random"], shuffle_seats=True,
                      turn_limit=80)],
        num_envs=6, seed=12, dims=cpp.EncoderDims(max_players=3), num_threads=2),
    "4p with trackers and shaping": lambda: cpp.VecEnv(
        [cpp.Scenario(num_players=4, rules=cpp.GameRules(vps_to_win=6), turn_limit=120)],
        num_envs=6, seed=13, trackers=True, vp_reward=0.5, critic_level=cpp.InfoLevel.FULL),
}
#: Recorded with the engine at e8edef9, before the style counters.
GOLDEN_ROLLOUTS = {
    "2p friendly robber, balanced dice":
        "3284b33d25f346c7672646abb8d069ec68efa67d22cd8fc4eb77bab7fcd3eaec",
    "3p against F and R": "b04e9282a4de50a1e672b0a75e1b798c43b58ca1b7ae4310184445e910652318",
    "4p with trackers and shaping":
        "4fdcf296fa6d7b8b4488ef101a114f6ab90da70f0136f213ee294abfc37f36d7",
}


def rollout_digest(env: cpp.VecEnv, steps: int = 500) -> str:
    """sha256 over every step's buffers, then the episode means and each env's last outcome."""
    buffers = env.allocate()
    env.reset()
    rng = np.random.default_rng(env.num_envs)
    digest = hashlib.sha256()
    for _ in range(steps):
        sample_actions(env, buffers, rng)
        env.step()
        for name in sorted(buffers):
            digest.update(buffers[name].tobytes())
    stats = env.pop_stats()
    assert stats["episodes"] > env.num_envs and 0 < stats["truncated"] < 1
    summary = [[bucket[key] for key in EPISODE_KEYS] for bucket in (stats, *stats["scenarios"])]
    outcomes = [sorted(env.last_outcome(e).items()) for e in range(env.num_envs)]
    digest.update(repr((summary, outcomes)).encode())
    return digest.hexdigest()


@pytest.mark.parametrize("name", sorted(ROLLOUTS))
def test_counting_the_style_of_play_changes_no_game(name):
    """The style counters are logging only: seeded rollouts give the rows, rewards, flags,
    episode means and outcomes the engine gave before it kept them."""
    assert rollout_digest(ROLLOUTS[name]()) == GOLDEN_ROLLOUTS[name]
