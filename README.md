# cppanatron

`cppanatron` is a completely Codex'ed C++20 rewrite of the Catanatron game engine. Its compatibility
target is the Catanatron revision pinned by the parent `catanrl` repository.

The project is split into:

- a dependency-free C++ rules engine;
- `cppanatron`, a Python package whose `_core` extension binds the engine with
  [nanobind](https://github.com/wjakob/nanobind);
- a parity harness and test suite that run the same scenarios on Catanatron and
  on this engine.

Status: the rules engine implements MINI, BASE, and TOURNAMENT maps and
explicit maps of any shape for 2–6 players, complete setup and turn transitions, seven/discard/robber handling,
builds, development cards, maritime and domestic trades, Longest Road, Largest
Army, terminal scoring, the CatanRL flat action table, and the pinned
`ValueFunctionPlayer` heuristic. `GameRules` sets the
victory point target, discard limit, friendly robber, bank, development deck and
pieces per player, e.g. the 5–6 player extension's bank of 24 per resource and
34-card deck. The 5–6 player special building phase is not implemented.

The engine also includes a policy/value-guided stochastic PUCT implementation.
It keeps cloned game states, tree traversal, exact dice/development-card/robber
chance outcomes, and value backup in C++. It is pull-based: the caller evaluates
the root and each selected leaf with its own model, so CatanRL can batch PyTorch
inference across many searches. `SearchPool` (below) supersedes it with
N-player, batched and determinized search.

## Python

```python
import numpy as np
import cppanatron as cpp

game = cpp.Game(2, map_type=cpp.MapType.BASE, seed=7)
game.playable_actions        # [Action(Color.RED, ActionType.BUILD_SETTLEMENT, 0), ...]
game.execute(game.playable_actions[0])
game.execute(cpp.Action(cpp.Color.RED, cpp.ActionType.BUILD_ROAD, (0, 1)))

space = cpp.FlatActionSpace(2, cpp.MapType.BASE)   # CatanRL's fixed action table
index = int(np.flatnonzero(space.mask(game))[0])
game.execute(space.decode(index, game))  # dice=(3, 4) would replay a roll
```

The Python names mirror Catanatron's (`Color.RED`, `ActionType.BUILD_ROAD`,
`ActionPrompt.PLAY_TURN`, `MapType.BASE`), and `Action` unpacks like
Catanatron's `(color, action_type, value)` namedtuple. Rules beyond the base
game go in one object, `cpp.Game(6, rules=cpp.GameRules(resource_bank=(24,) * 5,
development_deck=(20, 3, 3, 3, 5)))`; `value_action(game, enemies=cpp.ValueEnemies.ALL)`
scores every opponent's production rather than only the next seat's.
`game.apply_setup([cpp.PlayerSetup(settlements=[...], roads=[...], resources=...), ...],
current_player=1, has_rolled=True)` puts a fresh game into an exact position;
the bank, deck, victory points and awards are derived. Chance outcomes come from
the game's RNG unless passed to `execute` (`dice`, `development_card`,
`stolen_resource`), which replays recorded games exactly. Unplayable actions
raise `ValueError`, as in Catanatron.

Maps are immutable and shared: `cpp.CatanMap.build(cpp.MapType.BASE, seed)` or
`cpp.CatanMap.from_spec(spec)` builds one, and `cpp.Game(4, map=m, rules=...,
seed=...)` plays on it without copying. A `cpp.MapSpec` lists every tile in
order (`MapSpec.from_dict` / `.to_dict` use the JSON shape `{"tiles":
[{"coordinate": [x, y, z], "type": "land" | "water" | "port", "resource",
"number", "direction", "port_resource"}]}`, None for the desert and 3:1 ports);
node and edge ids follow Catanatron's `from_tiles` numbering, so a spec builds
the same board in both engines. Roads run along land-tile edges.

`cpp.GraphActionSpace(cpp.EncoderDims(max_players=..., max_tiles=..., max_nodes=...,
max_edges=...))` is the map-agnostic action space of graph networks: globals
(roll, end turn, development cards, discards, monopoly, year of plenty,
maritime trades), then a settlement and a city per node slot, a road per edge
slot, a robber move per tile and relative victim seat, and a final no-op for
waiting seats. `index`, `decode`, `mask` and `layout()` work on any map within
the dims; domestic trades are outside it.

`cpp.GraphEncoder(dims, cpp.InfoLevel.PRIVATE)` writes one float32 row per
(game, seat), relative to that seat and padded to the dims: `[global | players
| tiles | nodes | edges | edge_nodes | tile_nodes | action_mask]`, where the
topology blocks hold node slots (-1 padding) and the mask is the action
space's for the seat to act (only the no-op for the others). `layout()` maps
each block to `(offset, shape)`, `feature_names()` names every per-slot
feature, and the header documents them. `PUBLIC` adds opponents' resource
hands and `FULL` their development cards and actual victory points; tests
check that hidden changes leave lower levels' rows unchanged.

`cpp.VecEnv(scenarios, num_envs=..., dims=..., level=..., seed=...,
num_threads=...)` steps many games for training. A `cpp.Scenario(num_players=4,
rules=..., maps=[...] or map_type=..., seats=["learner", "F", "R", ...],
shuffle_seats=True, turn_limit=..., weight=...)` describes each kind of game;
every reset draws one by weight. Each env has a row per seat slot (`row = env
* max_players + seat`); `allocate()` (or `bind(...)` with caller-owned NumPy
arrays: observations float32 `[rows, observation_size]`, actions int32,
rewards float32, terminals/truncations/acting/learner uint8 or bool,
seat/env_index int32, num_players int32 per env) sets up the buffers. `step()` releases the
GIL, applies the acting rows' actions, plays forced moves and bot seats until
a learner must decide, and writes every row; a finished game pays 1 to the
winner's row, flags every row of its env terminal (a game cut off at the turn
limit pays nothing and is flagged both terminal and truncated, since PufferLib
learns from terminals only) and is replaced in the same step. `pop_stats()` returns
episode aggregates. `tools/bench_vec_env.py` measures throughput.

For evaluation, `cpp.VecEnv(..., auto_reset=False)` is schedule mode: envs
sit idle until `start_games([{"env": e, "map": ..., "game_seed": ...,
"seats": ["learner", "F", ...], "bot_seed": ..., "lineup_slots": [...]}])`
puts exact games into them, a finished env stays idle, and `pop_finished()`
reports each game's winner (seat and lineup slot), turns, truncation and
decisions per seat, so the games played do not depend on `num_envs`, threads
or finishing order. Random bots draw from a stream seeded per game, and the
optional `lineup_slot` buffer says who plays each row's seat.

`trackers=True` keeps a `HandTracker` for every learner seat, fed every
transition in the env (learner actions, forced moves, bot moves) and the
seat's own friendly-robber options; `env.tracker(e, seat)` returns a copy for
`SearchPool.add` and `sample_world`, so determinized search and belief agents
run on the threaded VecEnv.

`env.expert_actions(out)` labels every acting row with the index of the action
Catanatron's value function player (F) would choose there (-1 elsewhere),
computed on the env threads, for imitation learning.

`critic_level=cpp.InfoLevel.FULL` (with, say, `level=PRIVATE`) appends a
critic block to every encoded row: the players block at the critic's level,
the only block the levels differ in. `env.layout()` gives its offset as
`critic_players`; `GraphEncoder.encode_players` writes the same block, and
`cpp.ActorCriticEncoder(dims, level, critic_level)` (also `env.row_encoder`)
writes whole rows, so a `SearchPool` built with it evaluates leaves exactly
as the model was trained. With determinizations a leaf is a sampled world,
whose (sampled) hidden cards a full-information critic block shows.

`cppanatron.catanatron` converts actions, colors and boards to and from
Catanatron (`map_spec_from_catanatron`, `catanatron_map_from_spec`) and builds
the observation layout of Catanatron's board tensor. Our Catanatron fork gives
every map its own node graph of land-tile edges, the same adjacency, so both
engines play any shape alike. The older two-player flat environment
(`GameBatch`), search leaves (`MCTSSearch`) and observations write straight
into caller-owned NumPy buffers as well.

## Hidden information and batched search

`HandTracker(game, seat)` follows a game from one seat using only what that
seat sees: hand sizes, a known lower bound per resource and the unknown cards
beyond it (whose per-resource total is known from the bank), development-card
counts and the unseen cards (deck plus other hands). Feed it true transitions
with `observe(before, action, after)`, or, when you only have views (a server
bot), public events: `gain`, `lose`, `transfer`, `steal` (resource only if you
were thief or victim), `monopoly`, `buy_development_card` (card only for
yourself), `play_development_card`, `end_turn`, `public_victory_points` (after
every step: victory point cards are capped by the fewest points a seat held
without winning), and `observe_options(view, legal)` when you are to move (the
friendly robber's options reveal who is under 3 points). `sample_world(game,
tracker, seed)` deals a full game consistent with that knowledge, never
reading the hidden cards of `game`; `expected_resources(seat)` gives the mean
hand.

`SearchPool` runs many N-player searches in lockstep. It takes an
`ActionCodec` and an `ObservationEncoder` (implemented in C++; `FlatActionCodec`
and `FlatObservationEncoder` wrap the v1 flat table and observation) and a
`SearchConfig`:

```python
codec = cpp.FlatActionCodec(4, cpp.MapType.BASE)
encoder = cpp.FlatObservationEncoder(4, observation_layout())
pool = cpp.SearchPool(codec, encoder, cpp.SearchConfig(
    simulations=256, max_players=4, root=cpp.RootSelection.GUMBEL, determinizations=4))
ids = [pool.add(game, seed=i, tracker=trackers[game.current_player_index])
       for i, (game, trackers) in enumerate(boards)]
obs = np.empty((512, pool.observation_size), np.float32)
rows = np.empty(512, np.int64)
while n := pool.collect(obs, rows):          # leaves of every search, GIL released
    logits, values = net(obs[:n])            # [n, A], [n, P] seat-relative win probabilities
    pool.apply(logits, values)
action = pool.select(ids[0])                 # also visits(), q(), policy_target(), value()
```

A rejected `apply` (wrong shape, non-finite values) changes nothing and can be
retried; `cancel()` withdraws the last collect's rows. `advance(id, action)`
reuses a subtree when it is safe; with determinizations it refuses drawn
outcomes, monopolies and anything the worlds disagree on, and the caller then
adds a new search from the real position with the updated tracker.
Determinized roots offer exactly the real game's legal actions.

Values are per-seat win probabilities (slot 0 = the leaf's player to move);
each node picks by its own player's value (max^n), which for two players whose
values sum to one is the usual zero-sum search (`c_puct` is half of
`MCTSSearch`'s). Chance actions sample exact outcomes, and outcomes reaching
the same position share a child. Children and their games are built only when
first selected, and expanded positions drop their games until needed: a
256-simulation tree takes about 4.4 MB on 2-player BASE, versus 29 MB for
`MCTSSearch` (`build/cppanatron_search_bench` measures both).

## Parity

`cppanatron.parity.new_game(engine, ...)` returns a game with the same
Catanatron-shaped interface on either engine (`"catanatron"` or
`"cppanatron"`), with forced chance outcomes and full state snapshots. The suite
in `tests/python` uses it three ways:

- `test_rules.py` asserts Catan rules once and runs them on both engines;
- `test_differential.py` replays random games in lockstep and requires identical
  snapshots, including the exact playable-action set, before every transition,
  on template maps and explicit ones (templates as specs; hexagons, the 5–6
  player board, a rectangle, a random island, a strip, a lake, islands and a
  board without water) for 2–6 players (`CPPANATRON_PARITY_GAMES=50` fuzzes
  more seeds);
- `test_bindings.py` covers the Python API itself.

Every draw (dice, steals, deck shuffles, map generation, VecEnv sampling,
search chance nodes, Dirichlet and Gumbel noise, sampled worlds) goes through
the samplers in `include/cppanatron/random.hpp` over `std::mt19937_64`, not
the standard library's distributions, so a seed gives the same results with
any compiler or standard library (`tests/test_random.cpp` and
`tests/python/test_golden_random.py` pin seeded games, maps, schedules and
worlds; floating-point noise can still differ in the last bit where math
libraries round `log` differently).

Parity means matching legal actions, state transitions, terminal conditions,
observations, rewards and flat action indices. Random streams differ across
languages; replaying recorded outcomes must reproduce identical states. Map and
in-game randomness use independent seeds.

`cpp.build_info()` says how the loaded extension was built: the engine repo's
git commit and whether tracked files were modified (recorded at every build;
`uv sync`/`uv run` rebuild after a commit or an edit), the compiler, build
type and flags, and the nanobind and Python versions.

## Build and test

The C++ engine and its tests:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The Python package builds with scikit-build-core. Inside CatanRL, `uv sync`
installs it in editable mode and rebuilds the extension whenever engine sources
change. Standalone:

```bash
pip install -e '.[catanatron]' pytest
pytest
```

## License

GPL-3.0-or-later, matching Catanatron.
