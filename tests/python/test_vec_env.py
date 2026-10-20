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
