# cppanatron

`cppanatron` is a completely Codex'ed C++20 rewrite of the Catanatron game engine. Its compatibility
target is the Catanatron revision pinned by the parent `catanrl` repository.

The project is split into:

- a dependency-free C++ rules engine;
- `cppanatron`, a Python package whose `_core` extension binds the engine with
  [nanobind](https://github.com/wjakob/nanobind);
- a parity harness and test suite that run the same scenarios on Catanatron and
  on this engine.

Status: the rules engine implements MINI, BASE, and TOURNAMENT maps for 2–6
players, complete setup and turn transitions, seven/discard/robber handling,
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

`cppanatron.catanatron` converts actions, colors and boards to and from
Catanatron and builds the observation layout of Catanatron's board tensor.
Batched environments (`GameBatch`), search leaves (`MCTSSearch`) and
observations write straight into caller-owned NumPy buffers.

## Hidden information and batched search

`HandTracker(game, seat)` follows a game from one seat using only what that
seat sees: hand sizes, a known lower bound per resource and the unknown cards
beyond it (whose per-resource total is known from the bank), development-card
counts and the unseen cards (deck plus other hands). Feed it true transitions
with `observe(before, action, after)`, or, when you only have views (a server
bot), public events: `gain`, `lose`, `transfer`, `steal` (resource only if you
were thief or victim), `monopoly`, `buy_development_card` (card only for
yourself), `play_development_card`, `end_turn`. `sample_world(game, tracker,
seed)` deals a full game consistent with that knowledge, never reading the
hidden cards of `game`; `expected_resources(seat)` gives the mean hand.

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
  snapshots, including the exact playable-action set, before every transition
  (`CPPANATRON_PARITY_GAMES=50` fuzzes more seeds);
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
