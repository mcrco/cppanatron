# cppanatron

`cppanatron` is a C++20 rewrite of the Catanatron game engine. Its compatibility
target is the Catanatron revision pinned by the parent `catanrl` repository.

The project is intentionally split into:

- a dependency-free C++ rules engine;
- a stable, batched C ABI for Python/PufferLib integration;
- differential parity tests against the pinned Python implementation.

Status: the map topology, board-placement kernel, complete flat action-index
table, initial snake setup, and basic non-seven turns are implemented. The flat
action table has been exhaustively compared with CatanRL for MINI, BASE, and
TOURNAMENT with 2–4 players. Seven/discard/robber transitions, development
cards, trades, complete awards, feature extraction, baseline players, and the
batched training backend are still in progress.

## Build and test

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Compatibility policy

Parity means matching the legal-action set, state transitions, terminal
conditions, observations, rewards, and flat action indices used by `catanrl`.
Random streams do not need to be bit-identical across languages, but replaying
recorded stochastic outcomes must produce identical states.

## License

GPL-3.0-or-later, matching Catanatron.
