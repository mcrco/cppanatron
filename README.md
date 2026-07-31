# cppanatron

`cppanatron` is a C++20 rewrite of the Catanatron game engine. Its compatibility
target is the Catanatron revision pinned by the parent `catanrl` repository.

The project is intentionally split into:

- a dependency-free C++ rules engine;
- a stable C ABI for Python/PufferLib integration;
- differential parity tests against the pinned Python implementation.

Status: the rules engine implements MINI, BASE, and TOURNAMENT maps for 2–4
players, complete setup and turn transitions, seven/discard/robber handling,
builds, development cards, maritime and domestic trades, Longest Road, Largest
Army, terminal scoring, the CatanRL flat action table, and the pinned
`ValueFunctionPlayer` heuristic. The C ABI exposes legal masks, replayable
stochastic transitions, observations, state inspection, independent map/game
seeds, selectable official-spiral or random number placement, and native expert
actions.

The engine also includes a policy/value-guided stochastic PUCT implementation.
It keeps cloned game states, tree traversal, exact dice/development-card/robber
chance outcomes, and value backup in C++. Its pull-based C ABI returns native
root/leaf observations to the caller and accepts flat policy logits plus a
current-player value, allowing CatanRL to retain centrally batched PyTorch
inference. Root visit counts are exposed in the same flat action space used by
the Python AlphaZero replay buffer.

Native search is currently perfect-information MCTS. Information-set
determinization and CatanRL's optional Python action-pruning heuristic remain
Python-only.

The parent CatanRL repository supplies the `ctypes` binding, exact full-feature
adapter, and multiprocessing PufferLib environment. Its differential suite
reconstructs native random boards in Python and compares legal actions, state,
full observations, and expert values after every replayed transition across all
three maps and all supported player counts.

## Build and test

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The shared library used by Python is emitted as `build/libcppanatron.so` on
Linux (with the platform-equivalent extension elsewhere).

## Compatibility policy

Parity means matching the legal-action set, state transitions, terminal
conditions, observations, rewards, and flat action indices used by `catanrl`.
Random streams do not need to be bit-identical across languages, but replaying
recorded stochastic outcomes must produce identical states.

Map and in-game randomness use independent seeds. The legacy single-seed C ABI
remains available and applies that seed to both streams with official-spiral
number placement. CatanRL uses the additive configured constructor with random
number placement to match its Python training environments.

## License

GPL-3.0-or-later, matching Catanatron.
