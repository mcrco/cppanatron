#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include "cppanatron/action_space.hpp"
#include "cppanatron/game.hpp"

namespace cppanatron {

struct ChanceOutcome {
    Game game;
    double probability{};
};

/**
 * Enumerate the random outcomes of an action.
 *
 * Dice totals, development-card draws, and robber steals are represented
 * exactly. Every other action has one deterministic outcome. Probabilities
 * always sum to one.
 */
[[nodiscard]] std::vector<ChanceOutcome> action_outcomes(const Game& game, const Action& action);

/**
 * A policy/value-guided Monte Carlo tree search over cppanatron Game copies.
 *
 * Search traversal and tree mutation live entirely in C++. Neural inference is
 * deliberately pull-based: select_leaf() returns a position that the caller
 * evaluates, and evaluate_leaf() supplies flat policy logits and a scalar value
 * from the leaf's player-to-move perspective. This keeps the search independent
 * of a particular inference runtime and permits Python to batch PyTorch
 * requests across self-play games.
 */
class MCTSSearch {
  public:
    MCTSSearch(const Game& root_game, FlatActionSpace action_space, double c_puct = 1.5,
               std::uint64_t seed = 0);
    ~MCTSSearch();

    MCTSSearch(const MCTSSearch&) = delete;
    MCTSSearch& operator=(const MCTSSearch&) = delete;
    MCTSSearch(MCTSSearch&&) noexcept;
    MCTSSearch& operator=(MCTSSearch&&) noexcept;

    /**
     * Expand the root from flat policy logits without counting a simulation.
     */
    void initialize_root(std::span<const float> policy_logits);

    /**
     * Start one simulation.
     *
     * Returns the unevaluated leaf when neural inference is required. A null
     * result means the selected terminal/dead-end leaf was backed up internally
     * and the simulation is complete.
     */
    [[nodiscard]] const Game* select_leaf();

    /**
     * Expand and back up the leaf returned by select_leaf().
     *
     * value is from the pending leaf's current-player perspective.
     */
    void evaluate_leaf(std::span<const float> policy_logits, double value);

    [[nodiscard]] bool has_pending_leaf() const noexcept;
    [[nodiscard]] int pending_player_index() const;
    [[nodiscard]] const Game& root_game() const noexcept;
    [[nodiscard]] std::vector<std::uint32_t> root_visits() const;

    /**
     * Blend Dirichlet noise into root priors.
     */
    void add_root_dirichlet_noise(double alpha, double fraction);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cppanatron
