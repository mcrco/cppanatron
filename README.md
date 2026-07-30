# cppanatron

`cppanatron` is a C++20 rewrite of the Catanatron game engine. Its compatibility
target is the Catanatron revision pinned by the parent `catanrl` repository.

The project is intentionally split into:

- a dependency-free C++ rules engine;
- a stable, batched C ABI for Python/PufferLib integration;
- differential parity tests against the pinned Python implementation.

Status: the map topology and board-placement kernel are implemented. Full game
state transitions, feature extraction, baseline players, and the batched
training backend are still in progress.

## Build and test

```bash
cmake -S . -B build -G Ninja
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
