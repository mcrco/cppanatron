#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "cppanatron/codec.hpp"
#include "cppanatron/game.hpp"
#include "cppanatron/tracker.hpp"

namespace cppanatron {

enum class RootSelection : std::uint8_t {
    /** PUCT at the root too, with optional Dirichlet noise. */
    puct,
    /** Gumbel-Top-k plus Sequential Halving (Danihelka et al., 2022). */
    gumbel,
};

struct SearchConfig {
    /**
     * PUCT exploration constant. Values are win probabilities in [0, 1], so
     * this is half the constant MCTSSearch (values in [-1, 1]) needs for the
     * same search: v1's c_puct = 1.5 is 0.75 here.
     */
    double c_puct{0.75};
    /** Completed simulations per search (root evaluations not counted). */
    int simulations{256};
    /** Leaves one search may have awaiting evaluation (virtual visits). */
    int max_pending{1};
    RootSelection root{RootSelection::puct};
    /** Dirichlet noise at PUCT roots; alpha <= 0 disables it. */
    double dirichlet_alpha{0.0};
    double dirichlet_fraction{0.25};
    /** Gumbel: actions considered (m), noise scale (0 = deterministic). */
    int gumbel_considered{16};
    double gumbel_scale{1.0};
    /** Gumbel: sigma(q) = (c_visit + max visits) * c_scale * q. */
    double gumbel_c_visit{50.0};
    double gumbel_c_scale{0.1};
    /** Gumbel: min-max rescale completed Q before sigma. */
    bool gumbel_rescale_values{true};
    /**
     * Root worlds sampled from the root player's HandTracker; the simulations
     * are split between them and their root visits summed. 0 searches the
     * game as given (perfect information: an oracle unless it is a sample).
     */
    int determinizations{0};
    /** Width P of the value rows: the most seats the network predicts. */
    int max_players{6};
    /**
     * Value of unvisited actions for the player choosing: fpu_value when it
     * is in [0, 1] (v1 used 0.5), otherwise the node's own mean value for
     * that player minus fpu_reduction.
     */
    double fpu_value{-1.0};
    double fpu_reduction{0.0};
    /** Expand single-action positions below the root without a network call. */
    bool skip_forced{true};
    /**
     * Keep games only where children are still being created: a position's
     * game is dropped once it is evaluated (and a forced position's once its
     * child exists), and rebuilt from its parent's when a child is needed,
     * then kept. Costs about one game copy and action per simulation and
     * cuts tree memory several times.
     */
    bool release_games{true};
    /** Threads for collect/apply across searches; 0 = hardware concurrency. */
    int threads{1};
};

struct SearchStats {
    std::uint64_t simulations{};
    std::uint64_t evaluations{};
    std::uint64_t nodes{};
    std::uint64_t chance_nodes{};
    std::uint64_t games{};
    std::uint64_t edges{};
    std::uint64_t coalesced_outcomes{};
    /** Released games rebuilt from their parents. */
    std::uint64_t rebuilt_games{};
    std::uint32_t worlds{};
    std::uint32_t max_depth{};
    double mean_depth{};
    /** Tree storage without the games' own heap memory. */
    std::uint64_t tree_bytes{};
    /** The worker thread this search runs on. */
    std::uint32_t lane{};
};

/**
 * Many independent searches advanced in lockstep with pull-based inference.
 *
 * collect() advances every unfinished search to its next leaves and writes
 * their observations, each from its leaf's player to move, into one batch;
 * the caller evaluates the batch with its network and hands the results to
 * apply(), which expands the leaves and backs their values up.
 *
 * N players, max^n: the network returns per leaf P win probabilities in
 * seat-relative order (slot 0 = the leaf's player to move); the tree keeps
 * value sums per absolute seat, and each node selects by the component of its
 * own player to move. Terminal positions are worth 1 to the winner and 0 to
 * the others and never reach the network. With two players whose values sum
 * to one this is the zero-sum negamax search of MCTSSearch.
 *
 * Children are created only when first selected, and only then is the game
 * copied and the action applied. Chance actions (dice, development-card
 * draws, robber steals) become chance nodes whose outcomes are sampled from
 * their exact probabilities; an outcome's game is built when first sampled,
 * and outcomes leading to identical positions share one child. A chance
 * action's value is the probability-weighted mean of its visited outcomes,
 * renormalized over the visited probability mass.
 *
 * Searches never share mutable state, so collect and apply run them on
 * several threads when configured; the codec and encoder must allow
 * concurrent const calls.
 */
class SearchPool {
public:
    SearchPool(
        std::shared_ptr<const ActionCodec> codec,
        std::shared_ptr<const ObservationEncoder> encoder,
        SearchConfig config = {});
    ~SearchPool();
    SearchPool(const SearchPool&) = delete;
    SearchPool& operator=(const SearchPool&) = delete;
    SearchPool(SearchPool&&) noexcept;
    SearchPool& operator=(SearchPool&&) noexcept;

    [[nodiscard]] const SearchConfig& config() const noexcept;
    [[nodiscard]] std::size_t observation_size() const noexcept;
    [[nodiscard]] std::size_t action_size() const noexcept;
    [[nodiscard]] int max_players() const noexcept;

    /**
     * Start a search of `game` for its player to move. With determinizations
     * the root worlds are sampled from `tracker`, which must follow the
     * player to move; without them the tracker is ignored. A finished game
     * gives a search that is done at once, with no visits.
     *
     * Determinized roots offer exactly the legal actions `game` gives its
     * player to move, which that player knows even where they depend on
     * hidden cards (the friendly robber): worlds that do not offer them all
     * are resampled (a bounded number of times), and every root is
     * restricted to the real options the worlds share.
     */
    std::int64_t add(const Game& game, std::uint64_t seed, const HandTracker* tracker = nullptr);

    /**
     * Advance every unfinished search to its next leaves (roots first) and
     * write up to `ids.size()` observation rows into `observations`
     * (row-major, observation_size() floats per row) and the owning search
     * ids into `ids`. Returns the number of rows; 0 means every search is
     * done. Each collect must be followed by apply before the next.
     */
    std::size_t collect(std::span<float> observations, std::span<std::int64_t> ids);
    /**
     * Expand and back up the rows of the last collect: `logits` holds
     * action_size() floats per row, `values` max_players() floats per row.
     * Wrong sizes or non-finite values throw std::invalid_argument before
     * anything changes, so apply can be retried. If an expansion fails
     * anyway, the rows not yet applied are withdrawn (see cancel).
     */
    void apply(std::span<const float> logits, std::span<const float> values);
    /**
     * Withdraw the rows of the last collect: their leaves go back to their
     * searches unevaluated, as if never chosen. collect does this itself
     * when it throws.
     */
    void cancel();

    /** Rows of the last collect awaiting apply. */
    [[nodiscard]] std::size_t pending() const noexcept;
    /** The position of a pending row (for evaluators that read games). */
    [[nodiscard]] const Game& pending_game(std::size_t row) const;

    [[nodiscard]] bool contains(std::int64_t id) const;
    [[nodiscard]] std::vector<std::int64_t> ids() const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool done(std::int64_t id) const;
    /** Root visits over the codec's index space, summed over root worlds. */
    [[nodiscard]] std::vector<std::uint32_t> visits(std::int64_t id) const;
    /** Root action values for the root player; NaN where unvisited. */
    [[nodiscard]] std::vector<double> q(std::int64_t id) const;
    /**
     * Visit distribution (PUCT), or Gumbel's improved policy
     * softmax(logits + sigma(completed Q)), over the codec's index space.
     */
    [[nodiscard]] std::vector<float> policy_target(std::int64_t id) const;
    /** Mean root value per seat, seat-relative to the root player. */
    [[nodiscard]] std::vector<float> value(std::int64_t id) const;
    /**
     * The action to play. Temperature 0: most visits (PUCT) or the
     * Sequential Halving winner (Gumbel); otherwise a sample from
     * policy_target ** (1 / temperature).
     */
    [[nodiscard]] std::size_t select(std::int64_t id, double temperature = 0.0);
    /**
     * Re-root every world at the child of a deterministic root action and
     * start a fresh simulation budget, keeping the subtree. Returns false
     * (and changes the search no further) when the action leads to a chance
     * node, a finished game or an unvisited child, and, with
     * determinizations, when the child is another player's decision, was
     * reached through a drawn outcome (even the only possible one), or the
     * worlds disagree on the new root's public state or options, the action
     * reveals hidden cards (a monopoly's per-seat amounts), or (friendly
     * robber) the new root's robber moves depend on hidden cards. Then
     * start a new search from the real position with the tracker updated
     * by the real transition: reused worlds are only valid while nothing
     * the root player learns could contradict them.
     */
    bool advance(std::int64_t id, std::size_t action_index);
    void remove(std::int64_t id);
    [[nodiscard]] SearchStats stats(std::int64_t id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cppanatron
