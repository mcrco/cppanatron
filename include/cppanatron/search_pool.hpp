#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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

/** Which root worlds a determinized search takes from the root player's HandTracker. */
enum class WorldChoice : std::uint8_t {
    /**
     * `determinizations` draws of sample_world. Draws that deal the same
     * hidden cards share a tree and take a draw's simulations each.
     */
    sample,
    /**
     * The `determinizations` most likely deals of the hidden cards
     * (likely_worlds; all of them when there are fewer), each weighted by
     * its probability among the ones kept.
     */
    likely,
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
     * Root worlds taken from the root player's HandTracker (world_choice:
     * sampled, or the most likely ones); the simulations are split between
     * them and their root visits summed. 0 searches the game as given
     * (perfect information: an oracle unless it is a sample).
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
    /**
     * Share one node between every order of actions that reaches the same
     * position (two roads built either way round, one discarder's kinds in
     * any order, a bank trade before or after a build): each world keeps a
     * table of its positions below the root's children, a new child that
     * another path already created links to the existing node (verified
     * exactly), and that subtree is searched once, its statistics serving
     * every parent. Values back up along the path a simulation took. Off,
     * every path has its own nodes.
     */
    bool transpositions{false};
    /**
     * Temperature of a PUCT root's priors: the root selects by the network's
     * priors raised to 1 / temperature and renormalized over its actions
     * (then the Dirichlet noise is mixed in), so a value above 1 flattens
     * them. 1 leaves them as they are. Every search starts from the
     * network's priors, a re-rooted one too, and root_priors keeps them.
     */
    double root_policy_temperature{1.0};
    /**
     * Forced playouts at a PUCT root (KataGo's k, which is 2 there; 0 = off):
     * a root action that has a visit but fewer than sqrt(k * prior * total)
     * of them is selected before PUCT is asked, the one furthest below first
     * (the first of equals). An action without a visit is not forced: PUCT
     * (and the noise) opens it, as without forced playouts. The prior is the
     * one the root selects by (after the temperature and the noise), total
     * the visits of all the root's actions, and both counts include the
     * leaves awaiting evaluation. Only the root selects this way, each root
     * world on its own counts.
     */
    double forced_playouts{0.0};
    /**
     * Leave the trees that remove, advance and reroot discard (a removed
     * search's, and what a new root cuts away) to the pool's threads, so the
     * caller does not pay for freeing them (a tree is many small
     * allocations): the most tree nodes that may wait to be freed. They are
     * freed at the next collect, and before that when more would take them
     * past this bound, by the threads that ran the searches, side by side.
     * What has more nodes than the bound is freed by the call that discards
     * it, as everything is at 0 or with one thread. Results do not depend
     * on it; the default is about 160 MB of two-player base games.
     */
    int free_backlog{1 << 16};
    /**
     * Where the root worlds of a determinized search come from
     * (determinizations > 0). `sample` draws them. `likely` takes the most
     * likely deals instead and weights each world by its probability
     * (SearchPool::world_weights): a world gets that share of the
     * simulations, of any number of them to within one (PUCT roots, where
     * every world runs PUCT on its share), or of every root action's
     * simulations (Gumbel roots, so the actions Sequential Halving compares
     * were searched in the same worlds); a Gumbel root also weights the
     * worlds' log-priors and root values so. The rest of a world (the order
     * of its deck, its random stream) comes from the search's seed either
     * way. `likely` needs a position likely_worlds can list, which every
     * two-player position is; add throws std::invalid_argument otherwise.
     */
    WorldChoice world_choice{WorldChoice::sample};
};

/**
 * One search's own settings in place of the pool's, for add and reroot: an
 * unset field takes its SearchConfig value. They let searches of different
 * budgets share a pool (and a tree its consecutive searches), e.g. the full
 * and fast searches of playout cap randomization.
 */
struct SearchOptions {
    /** SearchConfig::simulations (reroot: the root visit cap). */
    std::optional<int> simulations;
    /** SearchConfig::max_pending. */
    std::optional<int> max_pending;
    /** SearchConfig::dirichlet_alpha (PUCT roots; <= 0: no noise). */
    std::optional<double> dirichlet_alpha;
    /** SearchConfig::root_policy_temperature (PUCT roots; 1: the priors as they are). */
    std::optional<double> root_policy_temperature;
    /** SearchConfig::forced_playouts (PUCT roots; 0: none). */
    std::optional<double> forced_playouts;
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
    /** Edges linked to a node another path had already created. */
    std::uint64_t transpositions{};
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
     * player to move. Without determinizations the search reads `game` as
     * is, hidden cards included (an oracle unless `game` is itself a
     * sample), so passing a tracker then throws std::invalid_argument
     * unless `ignore_tracker` says that is intended. A finished game gives
     * a search that is done at once, with no visits.
     *
     * Determinized roots offer exactly the legal actions `game` gives its
     * player to move, which that player knows even where they depend on
     * hidden cards (the friendly robber): worlds that do not offer them all
     * are resampled (a bounded number of times), and every root is
     * restricted to the real options the worlds share. `deal` picks how the
     * worlds' resource hands are drawn (sample_world's `deal`): the
     * tracker's own mode by default.
     *
     * With SearchConfig::world_choice `likely` nothing is sampled: the worlds
     * are the most likely deals likely_worlds lists for `game`, `tracker`
     * and `deal`, the same for every seed (the seed still shuffles their
     * decks), and std::invalid_argument is thrown when the position has too
     * many deals to list.
     */
    std::int64_t add(
        const Game& game, std::uint64_t seed, const HandTracker* tracker = nullptr,
        bool ignore_tracker = false, std::optional<TrackerMode> deal = std::nullopt);
    /**
     * add with the search's own budget, leaves per round, root noise, root
     * temperature and forced playouts (SearchOptions), which it keeps until
     * a reroot gives it others.
     */
    std::int64_t add(
        const Game& game, std::uint64_t seed, const SearchOptions& options,
        const HandTracker* tracker = nullptr, bool ignore_tracker = false,
        std::optional<TrackerMode> deal = std::nullopt);

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
     * The root's priors as the network gave them, without the root
     * temperature and the Dirichlet noise the search applied: action_size()
     * floats per root world, world after world, zero off the root's actions.
     * They are what the tree stores (float32), so a re-rooted root has them
     * as a fresh one does. All zero until every root world is expanded.
     */
    [[nodiscard]] std::vector<float> root_priors(std::int64_t id) const;
    /**
     * The priors the root selects by in its current search: root_priors at
     * the root temperature with the Dirichlet noise mixed in (root_priors
     * themselves without either, and at a Gumbel root). Same layout.
     */
    [[nodiscard]] std::vector<float> root_search_priors(std::int64_t id) const;
    /**
     * Each root world's share of the search, in the order of root_priors'
     * worlds; they sum to 1. World choice `likely`: the world's probability
     * among the worlds kept, most likely first. `sample`: the share of the
     * draws that dealt it. One world, also without determinizations: {1}.
     * Empty for a finished game.
     */
    [[nodiscard]] std::vector<double> world_weights(std::int64_t id) const;
    /**
     * Root visits per root world: visits() before the worlds are summed,
     * action_size() counts per world, world after world, as root_priors.
     */
    [[nodiscard]] std::vector<std::uint32_t> world_visits(std::int64_t id) const;
    /**
     * The action to play. Temperature 0: most visits (PUCT) or the
     * Sequential Halving winner (Gumbel: the best noise + logits + sigma
     * among the final survivors, which are re-ranked after every phase);
     * otherwise a sample from policy_target ** (1 / temperature).
     */
    [[nodiscard]] std::size_t select(std::int64_t id, double temperature = 0.0);
    /**
     * Re-root every world at the child of a deterministic root action and
     * start a fresh simulation budget (the search's own, see SearchOptions),
     * keeping the subtree; a PUCT root applies its root temperature to the
     * new root's priors and, with noise, draws new Dirichlet noise for them:
     * they are the network's, which the old root's never touched.
     * With transpositions the new root's children count only visits that
     * passed through the new root (SearchConfig::transpositions). Returns false
     * (and changes the search no further) when the action leads to a chance
     * node, a finished game or an unvisited child, and, with
     * determinizations, when the child is another player's decision, was
     * reached through a drawn outcome (even the only possible one), or the
     * worlds disagree on the new root's public state or options, or the
     * action reveals hidden cards (a monopoly's per-seat amounts). Then
     * start a new search from the real position with the tracker updated
     * by the real transition: reused worlds are only valid while nothing
     * the root player learns could contradict them. The worlds and their
     * weights stay as add chose them: what advance accepts (the root
     * player's own decisions, nothing drawn or revealed) leaves the deals
     * and their probabilities as they were.
     */
    bool advance(std::int64_t id, std::size_t action_index);
    /**
     * advance under a visit cap (KataGo's tree reuse): the visits the new
     * root already has (the sum of visits(), over worlds) count towards the
     * search's simulations, so the search only tops the root up to them,
     * and is done at once, with no simulation, when the root has as many
     * or more. `options` are the search's settings from here on (unset
     * fields: the pool's), so consecutive searches of one tree can differ
     * in budget, leaves per round, noise, root temperature and forced
     * playouts. Refuses what advance refuses, leaving the search and its
     * settings as they were. PUCT roots only (std::invalid_argument on a
     * Gumbel root: Sequential Halving plans its visits from an empty root).
     */
    bool reroot(std::int64_t id, std::size_t action_index, const SearchOptions& options = {});
    /**
     * Forget a search (std::out_of_range for an unknown id, std::logic_error
     * while it has leaves awaiting apply). Its trees are freed here or left
     * to the pool's threads (SearchConfig::free_backlog).
     */
    void remove(std::int64_t id);
    /**
     * Tree nodes that remove, advance and reroot discarded and nothing has
     * freed yet: at most SearchConfig::free_backlog.
     */
    [[nodiscard]] std::size_t unfreed_nodes() const noexcept;
    [[nodiscard]] SearchStats stats(std::int64_t id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cppanatron
