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
inference across many searches. Native search is perfect-information MCTS;
information-set determinization remains Python-only.

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
rewards float32, terminals/truncations/acting/learner uint8, seat/env_index
int32, num_players int32 per env) sets up the buffers. `step()` releases the
GIL, applies the acting rows' actions, plays forced moves and bot seats until
a learner must decide, and writes every row; a finished game pays 1 to the
winner's row, flags every row of its env terminal (or truncated at the turn
limit, with no reward) and is replaced in the same step. `pop_stats()` returns
episode aggregates. `tools/bench_vec_env.py` measures throughput.

`cppanatron.catanatron` converts actions, colors and boards to and from
Catanatron (`map_spec_from_catanatron`, `catanatron_map_from_spec`) and builds
the observation layout of Catanatron's board tensor. Stock Catanatron takes
node adjacency from the BASE map, so it plays explicit maps with the template
layouts only; `cppanatron.parity.catanatron_unsupported(spec)` probes whether
the installed Catanatron plays a shape. The older two-player flat environment (`GameBatch`), search leaves
(`MCTSSearch`) and observations write straight into caller-owned NumPy buffers
as well.

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
  board without water for 2–6 players, which xfail until Catanatron uses
  per-map node graphs) (`CPPANATRON_PARITY_GAMES=50` fuzzes more seeds);
- `test_bindings.py` covers the Python API itself.

Parity means matching legal actions, state transitions, terminal conditions,
observations, rewards and flat action indices. Random streams differ across
languages; replaying recorded outcomes must reproduce identical states. Map and
in-game randomness use independent seeds.

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
