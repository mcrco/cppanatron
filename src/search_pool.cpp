#include "cppanatron/search_pool.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>

namespace cppanatron {
namespace {

constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();
constexpr std::size_t kMaxSeats = kColors.size();
constexpr double kTiny = 1e-12;
/** Log-priors of masked-out or underflowed actions, for Gumbel scores. */
constexpr double kMinLogPrior = -69.0;

using Values = std::array<double, kMaxSeats>;

/**
 * A fixed set of threads running index-parallel loops; the caller joins in.
 *
 * Tasks may be pinned to lanes: lane k always runs on the same thread, so a
 * search's games are allocated and freed by one thread (glibc's per-thread
 * arenas otherwise turn every cross-thread free into lock contention).
 */
class WorkerPool {
public:
    explicit WorkerPool(int threads) {
        for (int i = 1; i < threads; ++i) {
            workers_.emplace_back([this, i] { loop(static_cast<std::size_t>(i)); });
        }
    }

    ~WorkerPool() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    [[nodiscard]] std::size_t threads() const noexcept { return workers_.size() + 1; }

    /** Run task(i) for i < count; with `lanes`, task i runs on thread lanes[i] % threads(). */
    void run(
        std::size_t count, const std::function<void(std::size_t)>& task,
        const std::vector<std::size_t>* lanes = nullptr) {
        if (workers_.empty() || count <= 1) {
            for (std::size_t i = 0; i < count; ++i) {
                task(i);
            }
            return;
        }
        {
            std::lock_guard lock(mutex_);
            task_ = &task;
            count_ = count;
            lanes_ = lanes;
            next_.store(0);
            error_ = nullptr;
            busy_ = workers_.size();
            ++generation_;
        }
        wake_.notify_all();
        work(0);
        std::unique_lock lock(mutex_);
        finished_.wait(lock, [this] { return busy_ == 0; });
        task_ = nullptr;
        if (error_) {
            std::rethrow_exception(error_);
        }
    }

private:
    void work(std::size_t thread) {
        const auto attempt = [this](std::size_t index) {
            try {
                (*task_)(index);
            } catch (...) {
                std::lock_guard lock(mutex_);
                if (!error_) {
                    error_ = std::current_exception();
                }
            }
        };
        if (lanes_ != nullptr) {
            for (std::size_t index = 0; index < count_; ++index) {
                if ((*lanes_)[index] % threads() == thread) {
                    attempt(index);
                }
            }
            return;
        }
        while (true) {
            const std::size_t index = next_.fetch_add(1);
            if (index >= count_) {
                return;
            }
            attempt(index);
        }
    }

    void loop(std::size_t thread) {
        std::size_t seen = 0;
        while (true) {
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) {
                    return;
                }
                seen = generation_;
            }
            work(thread);
            std::lock_guard lock(mutex_);
            if (--busy_ == 0) {
                finished_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable finished_;
    const std::function<void(std::size_t)>* task_{};
    const std::vector<std::size_t>* lanes_{};
    std::size_t count_{};
    std::atomic<std::size_t> next_{};
    std::size_t busy_{};
    std::size_t generation_{};
    bool stop_{};
    std::exception_ptr error_;
};

struct Outcome {
    std::uint32_t code{};
    double probability{};
};

/**
 * The random outcomes of an action, coalesced to what changes the game: dice
 * totals, drawn card kinds, stolen resource kinds. Empty for deterministic
 * actions.
 */
std::vector<Outcome> chance_outcomes(const Game& game, const Action& action) {
    std::vector<Outcome> outcomes;
    if (action.type == ActionType::roll) {
        constexpr std::array<int, 11> ways{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
        for (std::uint32_t total = 2; total <= 12; ++total) {
            outcomes.push_back({total, ways[total - 2] / 36.0});
        }
    } else if (action.type == ActionType::buy_development_card) {
        std::array<int, 5> counts{};
        for (DevelopmentCard card : game.development_deck()) {
            ++counts[static_cast<std::size_t>(card)];
        }
        const double size = static_cast<double>(game.development_cards_remaining());
        for (std::uint32_t card = 0; card < counts.size(); ++card) {
            if (counts[card] > 0) {
                outcomes.push_back({card, counts[card] / size});
            }
        }
    } else if (action.type == ActionType::move_robber) {
        const auto& move = std::get<RobberMove>(action.value);
        if (move.victim.has_value()) {
            const auto& hand = game.player(*move.victim).resources;
            const double size = std::accumulate(hand.begin(), hand.end(), 0);
            for (std::uint32_t resource = 0; resource < hand.size(); ++resource) {
                if (hand[resource] > 0) {
                    outcomes.push_back({resource, hand[resource] / size});
                }
            }
        }
    }
    return outcomes;
}

/** Apply an action with an explicit outcome, so the game's RNG is never used. */
void execute_outcome(Game& game, const Action& action, std::optional<std::uint32_t> code) {
    if (!code.has_value()) {
        game.execute(action);
        return;
    }
    switch (action.type) {
        case ActionType::roll: {
            const int total = static_cast<int>(*code);
            game.execute(action, total <= 7 ? Dice{1, total - 1} : Dice{6, total - 6});
            return;
        }
        case ActionType::buy_development_card:
            game.execute(action, std::nullopt, static_cast<DevelopmentCard>(*code));
            return;
        case ActionType::move_robber:
            game.execute(action, std::nullopt, std::nullopt, static_cast<Resource>(*code));
            return;
        default:
            throw std::logic_error("outcome given for a deterministic action");
    }
}

enum class NodeKind : std::uint8_t { decision, chance, terminal };

/**
 * A decision edge (action = codec index, prior) or a chance outcome
 * (action = outcome code, prior = probability). Several outcomes of one
 * chance node may share a child when they lead to the same position.
 */
struct TreeEdge {
    std::uint32_t action{};
    float prior{};
    std::uint32_t child{kNone};
};

struct Node {
    /**
     * The position, for decision and terminal nodes (chance nodes use their
     * parent's). With SearchConfig::release_games it is dropped once the
     * node is expanded and rebuilt from the parent when a child needs it.
     */
    std::unique_ptr<Game> game;
    Values value_sum{};
    /** Outcome children: identifies the position among its siblings. */
    std::uint64_t fingerprint{};
    std::uint32_t visits{};
    std::uint32_t parent{kNone};
    std::uint32_t first_edge{};
    std::uint32_t num_edges{};
    /** Codec index of the action that led here (the drawn action for chance nodes). */
    std::uint32_t action{};
    /** The chance outcome that led here, or kNone. */
    std::uint32_t outcome{kNone};
    std::uint16_t depth{};
    NodeKind kind{NodeKind::decision};
    std::uint8_t to_play{};
    std::uint8_t winner{};
    bool expanded{};
    bool pending{};
};

struct Tree {
    std::vector<Node> nodes;
    std::vector<TreeEdge> edges;
    std::uint64_t coalesced{};
    std::uint64_t rebuilt{};
    int players{};
};

/**
 * What can differ between the outcomes of one chance action from one
 * position: hands, scores, the bank, the deck and whose prompt it is.
 */
std::uint64_t outcome_fingerprint(const Game& game) {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&hash](std::uint64_t value) {
        hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    };
    for (std::size_t seat = 0; seat < game.players().size(); ++seat) {
        const PlayerState& player = game.players()[seat];
        for (std::size_t i = 0; i < 5; ++i) {
            mix(static_cast<std::uint64_t>(player.resources[i]));
            mix(static_cast<std::uint64_t>(player.development_cards[i]));
        }
        mix(static_cast<std::uint64_t>(player.actual_victory_points));
        mix(static_cast<std::uint64_t>(game.remaining_discards(game.colors()[seat])));
    }
    for (int count : game.resource_bank()) {
        mix(static_cast<std::uint64_t>(count));
    }
    mix(game.development_cards_remaining());
    mix(static_cast<std::uint64_t>(game.current_player_index()));
    mix(static_cast<std::uint64_t>(game.current_prompt()));
    mix(static_cast<std::uint64_t>(game.is_road_building()));
    return hash;
}

struct Leaf {
    std::uint32_t world{};
    std::uint32_t node{};
};

struct GumbelRoot {
    std::vector<double> logits;
    std::vector<double> noise;
    std::vector<std::uint32_t> own_visits;
    double root_value{};
    std::vector<std::pair<int, int>> phases;
    int phase{-1};
    std::vector<std::size_t> survivors;
    std::vector<std::size_t> queue;
};

struct Search {
    std::int64_t id{};
    std::mt19937_64 rng;
    int players{};
    int root_player{};
    bool determinized{};
    std::vector<Tree> worlds;
    /** World of each simulation, cyclically; merged samples appear repeatedly. */
    std::vector<std::uint32_t> schedule;
    int budget{};
    int started{};
    int completed{};
    int pending{};
    bool roots_ready{};
    bool done{};
    int roots_evaluated{};
    double root_net_value{};
    std::optional<GumbelRoot> gumbel;
    std::vector<Leaf> issued;
    std::uint64_t simulations{};
    std::uint64_t evaluations{};
    std::uint64_t depth_sum{};
    std::uint32_t max_depth{};
};

/** ``(actions kept, simulations)`` per Sequential Halving phase, as mctx. */
std::vector<std::pair<int, int>> halving_phases(int considered, int simulations) {
    std::vector<std::pair<int, int>> phases;
    if (simulations < 1) {
        return phases;
    }
    if (considered <= 1) {
        phases.emplace_back(1, simulations);
        return phases;
    }
    const int log2max = static_cast<int>(std::ceil(std::log2(considered)));
    int used = 0;
    int kept = considered;
    while (used < simulations) {
        const int extra = std::max(1, simulations / (log2max * kept));
        const int phase = std::min(extra * kept, simulations - used);
        phases.emplace_back(kept, phase);
        used += phase;
        kept = std::max(2, kept / 2);
    }
    return phases;
}

std::vector<double> softmax(const std::vector<double>& logits) {
    std::vector<double> result(logits.size());
    if (logits.empty()) {
        return result;
    }
    const double top = *std::max_element(logits.begin(), logits.end());
    double total = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        result[i] = std::exp(logits[i] - top);
        total += result[i];
    }
    for (double& value : result) {
        value /= total;
    }
    return result;
}

bool same_hidden_state(const Game& lhs, const Game& rhs) {
    for (std::size_t seat = 0; seat < lhs.players().size(); ++seat) {
        const PlayerState& a = lhs.players()[seat];
        const PlayerState& b = rhs.players()[seat];
        if (a.resources != b.resources || a.development_cards != b.development_cards ||
            a.development_card_owned_at_start != b.development_card_owned_at_start) {
            return false;
        }
    }
    return true;
}

}  // namespace

struct SearchPool::Impl {
    Impl(std::shared_ptr<const ActionCodec> action_codec,
         std::shared_ptr<const ObservationEncoder> observation_encoder,
         SearchConfig search_config)
        : codec(std::move(action_codec)),
          encoder(std::move(observation_encoder)),
          config(search_config) {
        if (!codec || !encoder) {
            throw std::invalid_argument("search needs an action codec and an observation encoder");
        }
        if (!std::isfinite(config.c_puct) || config.c_puct < 0.0) {
            throw std::invalid_argument("c_puct must be finite and non-negative");
        }
        if (config.simulations < 1 || config.max_pending < 1) {
            throw std::invalid_argument("simulations and max_pending must be positive");
        }
        if (config.max_players < 2 || config.max_players > static_cast<int>(kMaxSeats)) {
            throw std::invalid_argument("max_players must be between two and six");
        }
        if (config.determinizations < 0 || config.threads < 0) {
            throw std::invalid_argument("determinizations and threads must be non-negative");
        }
        if (config.gumbel_considered < 1 || !(config.gumbel_scale >= 0.0) ||
            !(config.gumbel_c_visit >= 0.0) || !(config.gumbel_c_scale >= 0.0)) {
            throw std::invalid_argument("invalid Gumbel settings");
        }
        if (!(config.dirichlet_fraction >= 0.0 && config.dirichlet_fraction <= 1.0)) {
            throw std::invalid_argument("Dirichlet fraction must be in [0, 1]");
        }
        if (!std::isfinite(config.fpu_value) || !std::isfinite(config.fpu_reduction)) {
            throw std::invalid_argument("first-play urgency settings must be finite");
        }
        if (codec->size() >= kNone) {
            throw std::invalid_argument("action space is too large");
        }
        const int threads = config.threads == 0
            ? static_cast<int>(std::max(1U, std::thread::hardware_concurrency()))
            : config.threads;
        workers = std::make_unique<WorkerPool>(threads);
    }

    std::shared_ptr<const ActionCodec> codec;
    std::shared_ptr<const ObservationEncoder> encoder;
    SearchConfig config;
    std::map<std::int64_t, std::unique_ptr<Search>> searches;
    std::int64_t next_id{};
    std::size_t rotation{};
    struct Row {
        Search* search{};
        Leaf leaf;
    };
    std::vector<Row> batch;
    std::unique_ptr<WorkerPool> workers;

    // ---------------------------------------------------------------- lookup

    Search& find(std::int64_t id) const {
        const auto it = searches.find(id);
        if (it == searches.end()) {
            throw std::out_of_range("no search with this id");
        }
        return *it->second;
    }

    // ------------------------------------------------------------ tree nodes

    std::uint32_t add_node(
        Tree& tree, std::unique_ptr<Game> game, std::uint32_t parent, std::uint32_t action,
        std::uint32_t outcome) const {
        Node node;
        node.parent = parent;
        node.action = action;
        node.outcome = outcome;
        node.depth = parent == kNone ? 0 : static_cast<std::uint16_t>(tree.nodes[parent].depth + 1);
        node.to_play = static_cast<std::uint8_t>(game->current_player_index());
        if (const auto winner = game->winning_color(); winner.has_value()) {
            const auto& colors = game->colors();
            node.kind = NodeKind::terminal;
            node.winner = static_cast<std::uint8_t>(
                std::find(colors.begin(), colors.end(), *winner) - colors.begin());
        }
        node.fingerprint = outcome_fingerprint(*game);
        // A finished game is never expanded; its value is its winner.
        if (!(config.release_games && node.kind == NodeKind::terminal)) {
            node.game = std::move(game);
        }
        if (tree.nodes.size() >= kNone) {
            throw std::length_error("search tree is too large");
        }
        tree.nodes.push_back(std::move(node));
        return static_cast<std::uint32_t>(tree.nodes.size() - 1);
    }

    /** A node's position, rebuilt from its parent's (recursively) if it was released. */
    const Game& game_of(Tree& tree, std::uint32_t node) const {
        if (tree.nodes[node].game) {
            return *tree.nodes[node].game;
        }
        const Node& target = tree.nodes[node];
        const Node& parent = tree.nodes[target.parent];
        const std::uint32_t source = parent.kind == NodeKind::chance ? parent.parent : target.parent;
        const std::uint32_t action = target.action;
        const std::uint32_t outcome = target.outcome;
        const Game& base = game_of(tree, source);
        auto game = std::make_unique<Game>(base);
        execute_outcome(
            *game, codec->decode(base, action),
            outcome == kNone ? std::nullopt : std::optional<std::uint32_t>(outcome));
        ++tree.rebuilt;
        tree.nodes[node].game = std::move(game);
        return *tree.nodes[node].game;
    }

    /** Create the child behind a decision edge: a chance node or a new position. */
    std::uint32_t create_child(Tree& tree, std::uint32_t node, std::uint32_t edge) const {
        const Game& game = game_of(tree, node);
        const std::uint32_t index = tree.edges[edge].action;
        const Action action = codec->decode(game, index);
        const auto outcomes = chance_outcomes(game, action);
        std::uint32_t child = 0;
        if (outcomes.size() > 1) {
            Node chance;
            chance.kind = NodeKind::chance;
            chance.parent = node;
            chance.depth = static_cast<std::uint16_t>(tree.nodes[node].depth + 1);
            chance.to_play = tree.nodes[node].to_play;
            chance.action = index;
            chance.first_edge = static_cast<std::uint32_t>(tree.edges.size());
            chance.num_edges = static_cast<std::uint32_t>(outcomes.size());
            chance.expanded = true;
            for (const Outcome& outcome : outcomes) {
                tree.edges.push_back({outcome.code, static_cast<float>(outcome.probability), kNone});
            }
            tree.nodes.push_back(std::move(chance));
            child = static_cast<std::uint32_t>(tree.nodes.size() - 1);
        } else {
            const std::uint32_t outcome = outcomes.empty() ? kNone : outcomes[0].code;
            auto next = std::make_unique<Game>(game);
            execute_outcome(
                *next, action, outcome == kNone ? std::nullopt : std::optional<std::uint32_t>(outcome));
            child = add_node(tree, std::move(next), node, index, outcome);
            // A forced position's only child is built; nothing needs it again.
            if (config.release_games && node != 0 && tree.nodes[node].num_edges == 1) {
                tree.nodes[node].game.reset();
            }
        }
        tree.edges[edge].child = child;
        return child;
    }

    /**
     * Build the position of one chance outcome. Outcomes reaching the same
     * position (equal fingerprints, and equal games when both are at hand)
     * share one child.
     */
    std::uint32_t create_outcome(Tree& tree, std::uint32_t chance, std::uint32_t edge) const {
        const std::uint32_t parent = tree.nodes[chance].parent;
        const std::uint32_t index = tree.nodes[chance].action;
        const Game& game = game_of(tree, parent);
        auto next = std::make_unique<Game>(game);
        execute_outcome(*next, codec->decode(game, index), tree.edges[edge].action);
        const std::uint64_t fingerprint = outcome_fingerprint(*next);
        const Node& node = tree.nodes[chance];
        for (std::uint32_t e = node.first_edge; e < node.first_edge + node.num_edges; ++e) {
            const std::uint32_t sibling = tree.edges[e].child;
            if (sibling != kNone && tree.nodes[sibling].fingerprint == fingerprint &&
                (!tree.nodes[sibling].game || tree.nodes[sibling].game->search_equivalent(*next))) {
                tree.edges[edge].child = sibling;
                ++tree.coalesced;
                return sibling;
            }
        }
        const std::uint32_t child = add_node(tree, std::move(next), chance, index, tree.edges[edge].action);
        tree.edges[edge].child = child;
        return child;
    }

    /** Expand a single-action position without asking the network. */
    void expand_forced(Tree& tree, std::uint32_t node) const {
        Node& target = tree.nodes[node];
        const Game& game = *target.game;
        target.first_edge = static_cast<std::uint32_t>(tree.edges.size());
        target.num_edges = 1;
        target.expanded = true;
        tree.edges.push_back(
            {static_cast<std::uint32_t>(codec->index(game, game.playable_actions().front())), 1.0F,
             kNone});
    }

    void expand(Tree& tree, std::uint32_t node, std::span<const float> logits) const {
        thread_local std::vector<std::uint8_t> mask;
        mask.assign(codec->size(), 0);
        Node& target = tree.nodes[node];
        codec->mask(*target.game, mask);
        std::vector<std::uint32_t> legal;
        double top = -std::numeric_limits<double>::infinity();
        for (std::uint32_t i = 0; i < mask.size(); ++i) {
            if (mask[i] != 0) {
                legal.push_back(i);
                if (std::isfinite(logits[i])) {
                    top = std::max(top, static_cast<double>(logits[i]));
                }
            }
        }
        std::vector<double> weights(legal.size());
        double total = 0.0;
        for (std::size_t i = 0; i < legal.size(); ++i) {
            const float logit = logits[legal[i]];
            weights[i] = std::isfinite(logit) ? std::exp(static_cast<double>(logit) - top) : 0.0;
            total += weights[i];
        }
        if (!std::isfinite(total) || total <= kTiny) {
            std::fill(weights.begin(), weights.end(), 1.0);
            total = static_cast<double>(weights.size());
        }
        target.first_edge = static_cast<std::uint32_t>(tree.edges.size());
        target.num_edges = static_cast<std::uint32_t>(legal.size());
        target.expanded = true;
        target.pending = false;
        for (std::size_t i = 0; i < legal.size(); ++i) {
            tree.edges.push_back({legal[i], static_cast<float>(weights[i] / total), kNone});
        }
    }

    // ------------------------------------------------------------ statistics

    static double mean(const Node& node, int seat) {
        return node.value_sum[static_cast<std::size_t>(seat)] / node.visits;
    }

    /**
     * A child's value for `seat`; chance children average their visited
     * outcomes, renormalized over the visited probability. nullopt if unvisited.
     */
    static std::optional<double> child_value(const Tree& tree, std::uint32_t child, int seat) {
        if (child == kNone || tree.nodes[child].visits == 0) {
            return std::nullopt;
        }
        const Node& node = tree.nodes[child];
        if (node.kind != NodeKind::chance) {
            return mean(node, seat);
        }
        double weight = 0.0;
        double value = 0.0;
        for (std::uint32_t e = node.first_edge; e < node.first_edge + node.num_edges; ++e) {
            const std::uint32_t outcome = tree.edges[e].child;
            if (outcome != kNone && tree.nodes[outcome].visits > 0) {
                weight += tree.edges[e].prior;
                value += tree.edges[e].prior * mean(tree.nodes[outcome], seat);
            }
        }
        if (weight <= kTiny) {
            return mean(node, seat);
        }
        return value / weight;
    }

    static std::uint32_t edge_visits(const Tree& tree, const TreeEdge& edge) {
        return edge.child == kNone ? 0 : tree.nodes[edge.child].visits;
    }

    std::uint32_t select_edge(const Tree& tree, std::uint32_t node_index) const {
        const Node& node = tree.nodes[node_index];
        const int seat = node.to_play;
        std::uint64_t total = 0;
        for (std::uint32_t e = node.first_edge; e < node.first_edge + node.num_edges; ++e) {
            total += edge_visits(tree, tree.edges[e]);
        }
        double fpu = config.fpu_value;
        if (fpu < 0.0 || fpu > 1.0) {
            fpu = (node.visits > 0 ? mean(node, seat) : 1.0 / tree.players) - config.fpu_reduction;
        }
        const double exploration = config.c_puct * std::sqrt(static_cast<double>(total) + 1.0);
        std::uint32_t best = node.first_edge;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::uint32_t e = node.first_edge; e < node.first_edge + node.num_edges; ++e) {
            const TreeEdge& edge = tree.edges[e];
            const double q = child_value(tree, edge.child, seat).value_or(fpu);
            const double score =
                q + exploration * edge.prior / (1.0 + static_cast<double>(edge_visits(tree, edge)));
            if (score > best_score) {
                best_score = score;
                best = e;
            }
        }
        return best;
    }

    static std::uint32_t sample_outcome(const Tree& tree, const Node& chance, std::mt19937_64& rng) {
        double total = 0.0;
        for (std::uint32_t e = chance.first_edge; e < chance.first_edge + chance.num_edges; ++e) {
            total += tree.edges[e].prior;
        }
        double draw = std::uniform_real_distribution<double>(0.0, total)(rng);
        for (std::uint32_t e = chance.first_edge; e < chance.first_edge + chance.num_edges; ++e) {
            draw -= tree.edges[e].prior;
            if (draw < 0.0) {
                return e;
            }
        }
        return chance.first_edge + chance.num_edges - 1;
    }

    static void backup(Tree& tree, std::uint32_t node, const Values& values, bool count_visit) {
        for (std::uint32_t n = node; n != kNone; n = tree.nodes[n].parent) {
            Node& target = tree.nodes[n];
            if (count_visit) {
                ++target.visits;
            }
            for (std::size_t seat = 0; seat < kMaxSeats; ++seat) {
                target.value_sum[seat] += values[seat];
            }
        }
    }

    static Values terminal_values(const Node& node) {
        Values values{};
        values[node.winner] = 1.0;
        return values;
    }

    // -------------------------------------------------------------- selection

    enum class Stop : std::uint8_t { leaf, terminal, collision, dead_end };

    struct Descent {
        Stop stop{};
        std::uint32_t node{};
    };

    Descent descend(Search& search, Tree& tree, std::optional<std::size_t> root_edge) const {
        std::uint32_t node = 0;
        while (true) {
            Node& current = tree.nodes[node];
            if (current.kind == NodeKind::terminal) {
                return {Stop::terminal, node};
            }
            std::uint32_t edge = 0;
            if (current.kind == NodeKind::chance) {
                edge = sample_outcome(tree, current, search.rng);
                const std::uint32_t child = tree.edges[edge].child;
                node = child != kNone ? child : create_outcome(tree, node, edge);
                continue;
            }
            if (current.pending) {
                return {Stop::collision, node};
            }
            if (!current.expanded) {
                if (config.skip_forced && node != 0 && current.game->playable_actions().size() == 1) {
                    expand_forced(tree, node);
                    continue;
                }
                return {Stop::leaf, node};
            }
            if (current.num_edges == 0) {
                return {Stop::dead_end, node};
            }
            edge = node == 0 && root_edge.has_value()
                ? current.first_edge + static_cast<std::uint32_t>(*root_edge)
                : select_edge(tree, node);
            const std::uint32_t child = tree.edges[edge].child;
            node = child != kNone ? child : create_child(tree, node, edge);
        }
    }

    static void record_depth(Search& search, const Node& node) {
        ++search.simulations;
        search.depth_sum += node.depth;
        search.max_depth = std::max<std::uint32_t>(search.max_depth, node.depth);
    }

    // ----------------------------------------------------------------- Gumbel

    /** Root visits and values for the root player, per root edge, over all worlds. */
    void root_statistics(
        const Search& search, std::vector<double>& visits, std::vector<double>& values) const {
        const std::size_t count = search.worlds.front().nodes.front().num_edges;
        visits.assign(count, 0.0);
        values.assign(count, 0.0);
        for (const Tree& tree : search.worlds) {
            const Node& root = tree.nodes.front();
            for (std::size_t i = 0; i < count; ++i) {
                const TreeEdge& edge = tree.edges[root.first_edge + i];
                const double n = edge_visits(tree, edge);
                if (n > 0) {
                    visits[i] += n;
                    values[i] += n * child_value(tree, edge.child, search.root_player).value_or(0.0);
                }
            }
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (visits[i] > 0) {
                values[i] /= visits[i];
            }
        }
    }

    /** sigma(completed Q) per root edge, as v1's GumbelRoot (mctx defaults). */
    std::vector<double> gumbel_sigma(const Search& search) const {
        const GumbelRoot& gumbel = *search.gumbel;
        std::vector<double> visits;
        std::vector<double> values;
        root_statistics(search, visits, values);
        const std::vector<double> prior = softmax(gumbel.logits);
        double total = 0.0;
        double weight = 0.0;
        double weighted = 0.0;
        double most = 0.0;
        for (std::size_t i = 0; i < visits.size(); ++i) {
            total += visits[i];
            most = std::max(most, visits[i]);
            if (visits[i] > 0) {
                const double p = std::max(prior[i], std::numeric_limits<double>::min());
                weight += p;
                weighted += p * values[i];
            }
        }
        const double mixed = weight > 0.0
            ? (gumbel.root_value + total * weighted / weight) / (total + 1.0)
            : gumbel.root_value;
        std::vector<double> completed(visits.size());
        for (std::size_t i = 0; i < visits.size(); ++i) {
            completed[i] = visits[i] > 0 ? values[i] : mixed;
        }
        if (config.gumbel_rescale_values && !completed.empty()) {
            const auto [low, high] = std::minmax_element(completed.begin(), completed.end());
            const double lowest = *low;
            const double range = std::max(*high - lowest, 1e-8);
            for (double& value : completed) {
                value = (value - lowest) / range;
            }
        } else {
            for (double& value : completed) {
                value = std::clamp(value, 0.0, 1.0);
            }
        }
        const double scale = (config.gumbel_c_visit + most) * config.gumbel_c_scale;
        for (double& value : completed) {
            value *= scale;
        }
        return completed;
    }

    std::optional<std::size_t> next_root_edge(Search& search) const {
        GumbelRoot& gumbel = *search.gumbel;
        if (gumbel.queue.empty()) {
            // Halving ranks finished phases only, so wait for pending leaves.
            if (search.pending > 0 ||
                gumbel.phase + 1 >= static_cast<int>(gumbel.phases.size())) {
                return std::nullopt;
            }
            ++gumbel.phase;
            const auto [kept, simulations] = gumbel.phases[static_cast<std::size_t>(gumbel.phase)];
            if (gumbel.phase > 0 && static_cast<std::size_t>(kept) < gumbel.survivors.size()) {
                const auto sigma = gumbel_sigma(search);
                std::stable_sort(
                    gumbel.survivors.begin(), gumbel.survivors.end(),
                    [&](std::size_t a, std::size_t b) {
                        return gumbel.noise[a] + gumbel.logits[a] + sigma[a] >
                               gumbel.noise[b] + gumbel.logits[b] + sigma[b];
                    });
                gumbel.survivors.resize(static_cast<std::size_t>(kept));
            }
            for (int i = 0; i < simulations; ++i) {
                gumbel.queue.push_back(gumbel.survivors[static_cast<std::size_t>(i) % gumbel.survivors.size()]);
            }
            std::reverse(gumbel.queue.begin(), gumbel.queue.end());
        }
        const std::size_t edge = gumbel.queue.back();
        gumbel.queue.pop_back();
        ++gumbel.own_visits[edge];
        return edge;
    }

    /** Set up the root once every world's root has its priors. */
    void prepare_root(Search& search, std::optional<double> root_value) const {
        const Tree& first = search.worlds.front();
        const Node& root = first.nodes.front();
        const std::size_t count = root.num_edges;
        for (const Tree& tree : search.worlds) {
            const Node& other = tree.nodes.front();
            bool same = other.num_edges == count;
            for (std::size_t i = 0; same && i < count; ++i) {
                same = tree.edges[other.first_edge + i].action == first.edges[root.first_edge + i].action;
            }
            if (!same) {
                throw std::logic_error("determinized root worlds disagree on the legal actions");
            }
        }
        if (count == 0) {
            search.done = true;
            return;
        }
        if (config.root == RootSelection::puct && config.dirichlet_alpha > 0.0 &&
            config.dirichlet_fraction > 0.0) {
            std::gamma_distribution<double> gamma(config.dirichlet_alpha, 1.0);
            std::vector<double> noise(count);
            for (double& value : noise) {
                value = gamma(search.rng);
            }
            double total = std::accumulate(noise.begin(), noise.end(), 0.0);
            if (total <= kTiny) {
                std::fill(noise.begin(), noise.end(), 1.0);
                total = static_cast<double>(count);
            }
            for (Tree& tree : search.worlds) {
                const Node& world_root = tree.nodes.front();
                for (std::size_t i = 0; i < count; ++i) {
                    TreeEdge& edge = tree.edges[world_root.first_edge + i];
                    edge.prior = static_cast<float>(
                        (1.0 - config.dirichlet_fraction) * edge.prior +
                        config.dirichlet_fraction * noise[i] / total);
                }
            }
        }
        if (config.root != RootSelection::gumbel) {
            search.gumbel.reset();
            return;
        }
        GumbelRoot gumbel;
        gumbel.logits.assign(count, 0.0);
        for (const Tree& tree : search.worlds) {
            const Node& world_root = tree.nodes.front();
            for (std::size_t i = 0; i < count; ++i) {
                const double prior = tree.edges[world_root.first_edge + i].prior;
                gumbel.logits[i] += std::max(prior > 0.0 ? std::log(prior) : kMinLogPrior, kMinLogPrior) /
                                    static_cast<double>(search.worlds.size());
            }
        }
        std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), 1.0);
        for (std::size_t i = 0; i < count; ++i) {
            gumbel.noise.push_back(config.gumbel_scale * -std::log(-std::log(uniform(search.rng))));
        }
        gumbel.own_visits.assign(count, 0);
        if (root_value.has_value()) {
            gumbel.root_value = *root_value;
        } else {
            double visits = 0.0;
            double value = 0.0;
            for (const Tree& tree : search.worlds) {
                visits += tree.nodes.front().visits;
                value += tree.nodes.front().value_sum[static_cast<std::size_t>(search.root_player)];
            }
            gumbel.root_value = visits > 0 ? value / visits : 0.0;
        }
        const int considered = std::min(config.gumbel_considered, static_cast<int>(count));
        gumbel.phases = halving_phases(considered, search.budget);
        std::vector<std::size_t> order(count);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return gumbel.noise[a] + gumbel.logits[a] > gumbel.noise[b] + gumbel.logits[b];
        });
        order.resize(static_cast<std::size_t>(considered));
        gumbel.survivors = std::move(order);
        search.gumbel = std::move(gumbel);
    }

    // ---------------------------------------------------------- collect/apply

    int wanted(const Search& search) const {
        if (search.done) {
            return 0;
        }
        if (!search.roots_ready) {
            int roots = 0;
            for (const Tree& tree : search.worlds) {
                const Node& root = tree.nodes.front();
                roots += !root.expanded && !root.pending;
            }
            return roots;
        }
        return std::max(0, std::min(config.max_pending - search.pending, search.budget - search.started));
    }

    void finish_if_done(Search& search) const {
        if (search.roots_ready && search.pending == 0 && search.started >= search.budget) {
            search.done = true;
        }
    }

    /** Issue up to `quota` leaves of one search, backing up terminal ones. */
    void gather(Search& search, int quota) const {
        int issued = 0;
        if (!search.roots_ready) {
            for (std::uint32_t w = 0; w < search.worlds.size() && issued < quota; ++w) {
                Node& root = search.worlds[w].nodes.front();
                if (!root.expanded && !root.pending) {
                    root.pending = true;
                    search.issued.push_back({w, 0});
                    ++search.pending;
                    ++issued;
                }
            }
            return;
        }
        while (issued < quota && search.started < search.budget) {
            const std::uint32_t world =
                search.schedule[static_cast<std::size_t>(search.started) % search.schedule.size()];
            Tree& tree = search.worlds[world];
            std::optional<std::size_t> root_edge;
            if (search.gumbel.has_value()) {
                root_edge = next_root_edge(search);
                if (!root_edge.has_value()) {
                    break;
                }
            }
            const Descent descent = descend(search, tree, root_edge);
            if (descent.stop == Stop::collision) {
                if (root_edge.has_value()) {
                    search.gumbel->queue.push_back(*root_edge);
                    --search.gumbel->own_visits[*root_edge];
                }
                break;
            }
            ++search.started;
            Node& node = tree.nodes[descent.node];
            if (descent.stop == Stop::leaf) {
                node.pending = true;
                for (std::uint32_t n = descent.node; n != kNone; n = tree.nodes[n].parent) {
                    ++tree.nodes[n].visits;
                }
                search.issued.push_back({world, descent.node});
                ++search.pending;
                ++issued;
                continue;
            }
            Values values{};
            if (descent.stop == Stop::terminal) {
                values = terminal_values(node);
            } else {
                for (int seat = 0; seat < search.players; ++seat) {
                    values[static_cast<std::size_t>(seat)] =
                        node.visits > 0 ? mean(node, seat) : 1.0 / search.players;
                }
            }
            backup(tree, descent.node, values, true);
            ++search.completed;
            record_depth(search, node);
        }
        finish_if_done(search);
    }

    void evaluate(Search& search, const Leaf& leaf, std::span<const float> logits,
                  std::span<const float> values) const {
        Tree& tree = search.worlds[leaf.world];
        Node& node = tree.nodes[leaf.node];
        Values absolute{};
        for (int slot = 0; slot < search.players; ++slot) {
            const float value = values[static_cast<std::size_t>(slot)];
            if (!std::isfinite(value)) {
                throw std::invalid_argument("leaf values must be finite");
            }
            absolute[static_cast<std::size_t>((node.to_play + slot) % search.players)] =
                std::clamp(static_cast<double>(value), 0.0, 1.0);
        }
        expand(tree, leaf.node, logits);
        if (config.release_games && leaf.node != 0) {
            tree.nodes[leaf.node].game.reset();
        }
        ++search.evaluations;
        --search.pending;
        if (leaf.node == 0) {
            backup(tree, 0, absolute, true);
            search.root_net_value += absolute[static_cast<std::size_t>(search.root_player)];
            if (++search.roots_evaluated == static_cast<int>(search.worlds.size())) {
                search.roots_ready = true;
                prepare_root(search, search.root_net_value / search.roots_evaluated);
            }
        } else {
            backup(tree, leaf.node, absolute, false);
            ++search.completed;
            record_depth(search, tree.nodes[leaf.node]);
        }
        finish_if_done(search);
    }

    std::size_t collect(std::span<float> observations, std::span<std::int64_t> ids) {
        if (!batch.empty()) {
            throw std::logic_error("apply the last collect before collecting again");
        }
        const std::size_t capacity = ids.size();
        const std::size_t width = encoder->size();
        if (observations.size() < capacity * width) {
            throw std::invalid_argument("observation buffer is smaller than ids.size() rows");
        }
        while (true) {
            std::vector<Search*> active;
            for (auto& [id, search] : searches) {
                if (!search->done) {
                    active.push_back(search.get());
                }
            }
            if (active.empty() || capacity == 0) {
                return 0;
            }
            // Rotate who goes first so a small batch does not starve anyone.
            std::rotate(
                active.begin(),
                active.begin() + static_cast<std::ptrdiff_t>(rotation++ % active.size()),
                active.end());
            std::vector<int> quotas(active.size());
            std::size_t remaining = capacity;
            std::int64_t progress = 0;
            for (std::size_t i = 0; i < active.size(); ++i) {
                quotas[i] = std::min(wanted(*active[i]), static_cast<int>(remaining));
                remaining -= static_cast<std::size_t>(quotas[i]);
                progress -= active[i]->started;
            }
            std::vector<std::size_t> lanes(active.size());
            for (std::size_t i = 0; i < active.size(); ++i) {
                lanes[i] = static_cast<std::size_t>(active[i]->id);
            }
            workers->run(
                active.size(),
                [&](std::size_t i) {
                    if (quotas[i] > 0) {
                        gather(*active[i], quotas[i]);
                    }
                },
                &lanes);
            for (Search* search : active) {
                for (const Leaf& leaf : search->issued) {
                    batch.push_back({search, leaf});
                }
                search->issued.clear();
            }
            if (!batch.empty()) {
                break;
            }
            for (const Search* search : active) {
                progress += search->started;
            }
            if (progress == 0) {
                throw std::logic_error("no search can make progress");
            }
            // Every issued simulation ended in a terminal position: go again.
        }
        workers->run(batch.size(), [&](std::size_t row) {
            const Row& entry = batch[row];
            const Game& game = *entry.search->worlds[entry.leaf.world].nodes[entry.leaf.node].game;
            encoder->encode(game, game.current_player_index(), observations.subspan(row * width, width));
            ids[row] = entry.search->id;
        });
        return batch.size();
    }

    void apply(std::span<const float> logits, std::span<const float> values) {
        const std::size_t rows = batch.size();
        const std::size_t width = codec->size();
        const auto players = static_cast<std::size_t>(config.max_players);
        if (logits.size() != rows * width || values.size() != rows * players) {
            throw std::invalid_argument(
                "apply needs action_size() logits and max_players() values per collected row");
        }
        // Rows of one search are contiguous; each search is one task.
        std::vector<std::pair<std::size_t, std::size_t>> ranges;
        for (std::size_t row = 0; row < rows; ++row) {
            if (row == 0 || batch[row].search != batch[row - 1].search) {
                ranges.emplace_back(row, row);
            }
            ranges.back().second = row + 1;
        }
        std::vector<Row> rows_in_flight;
        rows_in_flight.swap(batch);
        std::vector<std::size_t> lanes(ranges.size());
        for (std::size_t task = 0; task < ranges.size(); ++task) {
            lanes[task] = static_cast<std::size_t>(rows_in_flight[ranges[task].first].search->id);
        }
        workers->run(
            ranges.size(),
            [&](std::size_t task) {
                for (std::size_t row = ranges[task].first; row < ranges[task].second; ++row) {
                    const Row& entry = rows_in_flight[row];
                    evaluate(
                        *entry.search, entry.leaf, logits.subspan(row * width, width),
                        values.subspan(row * players, players));
                }
            },
            &lanes);
    }

    // ------------------------------------------------------------ re-rooting

    static Tree extract(Tree& tree, std::uint32_t root) {
        std::vector<std::uint32_t> remap(tree.nodes.size(), kNone);
        std::vector<std::uint32_t> order{root};
        remap[root] = 0;
        for (std::size_t i = 0; i < order.size(); ++i) {
            const Node& node = tree.nodes[order[i]];
            for (std::uint32_t e = node.first_edge; e < node.first_edge + node.num_edges; ++e) {
                const std::uint32_t child = tree.edges[e].child;
                if (child != kNone && remap[child] == kNone) {
                    remap[child] = static_cast<std::uint32_t>(order.size());
                    order.push_back(child);
                }
            }
        }
        Tree result;
        result.coalesced = tree.coalesced;
        result.rebuilt = tree.rebuilt;
        result.players = tree.players;
        result.nodes.reserve(order.size());
        const std::uint16_t base = tree.nodes[root].depth;
        for (std::uint32_t old : order) {
            Node node = std::move(tree.nodes[old]);
            const std::uint32_t first = node.first_edge;
            node.parent = old == root ? kNone : remap[node.parent];
            node.depth = static_cast<std::uint16_t>(node.depth - base);
            node.first_edge = static_cast<std::uint32_t>(result.edges.size());
            for (std::uint32_t e = first; e < first + node.num_edges; ++e) {
                TreeEdge edge = tree.edges[e];
                edge.child = edge.child == kNone ? kNone : remap[edge.child];
                result.edges.push_back(edge);
            }
            result.nodes.push_back(std::move(node));
        }
        return result;
    }

    bool advance(Search& search, std::size_t action) {
        if (search.pending > 0) {
            throw std::logic_error("cannot advance a search with leaves awaiting apply");
        }
        if (search.worlds.empty() || !search.roots_ready) {
            return false;
        }
        std::vector<std::uint32_t> children;
        for (const Tree& tree : search.worlds) {
            const Node& root = tree.nodes.front();
            std::uint32_t child = kNone;
            for (std::uint32_t e = root.first_edge; e < root.first_edge + root.num_edges; ++e) {
                if (tree.edges[e].action == action) {
                    child = tree.edges[e].child;
                }
            }
            if (child == kNone) {
                return false;
            }
            const Node& next = tree.nodes[child];
            if (next.kind != NodeKind::decision || !next.expanded || next.visits == 0 ||
                (search.determinized && next.to_play != search.root_player)) {
                return false;
            }
            children.push_back(child);
        }
        for (std::size_t w = 0; w < search.worlds.size(); ++w) {
            (void)game_of(search.worlds[w], children[w]);
            search.worlds[w] = extract(search.worlds[w], children[w]);
        }
        search.root_player = search.worlds.front().nodes.front().to_play;
        search.started = 0;
        search.completed = 0;
        search.done = false;
        search.budget = config.simulations;
        search.simulations = 0;
        search.evaluations = 0;
        search.depth_sum = 0;
        search.max_depth = 0;
        prepare_root(search, std::nullopt);
        return true;
    }

    // ---------------------------------------------------------------- results

    std::vector<std::uint32_t> visits(const Search& search) const {
        std::vector<std::uint32_t> result(codec->size(), 0);
        for (const Tree& tree : search.worlds) {
            const Node& root = tree.nodes.front();
            for (std::uint32_t e = root.first_edge; e < root.first_edge + root.num_edges; ++e) {
                result[tree.edges[e].action] += edge_visits(tree, tree.edges[e]);
            }
        }
        return result;
    }

    std::vector<double> q(const Search& search) const {
        std::vector<double> result(codec->size(), std::numeric_limits<double>::quiet_NaN());
        if (search.worlds.empty() || !search.roots_ready) {
            return result;
        }
        std::vector<double> counts;
        std::vector<double> values;
        root_statistics(search, counts, values);
        const Tree& first = search.worlds.front();
        for (std::size_t i = 0; i < counts.size(); ++i) {
            if (counts[i] > 0) {
                result[first.edges[first.nodes.front().first_edge + i].action] = values[i];
            }
        }
        return result;
    }

    std::vector<float> policy_target(const Search& search) const {
        std::vector<float> result(codec->size(), 0.0F);
        if (search.worlds.empty() || !search.roots_ready) {
            return result;
        }
        const Tree& first = search.worlds.front();
        const Node& root = first.nodes.front();
        if (search.gumbel.has_value()) {
            const auto sigma = gumbel_sigma(search);
            std::vector<double> scores(sigma.size());
            for (std::size_t i = 0; i < scores.size(); ++i) {
                scores[i] = search.gumbel->logits[i] + sigma[i];
            }
            const auto policy = softmax(scores);
            for (std::size_t i = 0; i < policy.size(); ++i) {
                result[first.edges[root.first_edge + i].action] = static_cast<float>(policy[i]);
            }
            return result;
        }
        const auto counts = visits(search);
        const double total = std::accumulate(counts.begin(), counts.end(), 0.0);
        if (total > 0) {
            for (std::size_t i = 0; i < counts.size(); ++i) {
                result[i] = static_cast<float>(counts[i] / total);
            }
        }
        return result;
    }

    std::vector<float> value(const Search& search) const {
        std::vector<float> result(static_cast<std::size_t>(config.max_players), 0.0F);
        double visits = 0.0;
        Values sums{};
        for (const Tree& tree : search.worlds) {
            const Node& root = tree.nodes.front();
            visits += root.visits;
            for (std::size_t seat = 0; seat < kMaxSeats; ++seat) {
                sums[seat] += root.value_sum[seat];
            }
        }
        if (visits > 0) {
            for (int slot = 0; slot < search.players; ++slot) {
                result[static_cast<std::size_t>(slot)] = static_cast<float>(
                    sums[static_cast<std::size_t>((search.root_player + slot) % search.players)] / visits);
            }
        }
        return result;
    }

    std::size_t select(Search& search, double temperature) const {
        if (!(temperature >= 0.0) || !std::isfinite(temperature)) {
            throw std::invalid_argument("temperature must be finite and non-negative");
        }
        if (search.worlds.empty() || !search.roots_ready) {
            throw std::logic_error("the search has no root actions to select from");
        }
        const Tree& first = search.worlds.front();
        const Node& root = first.nodes.front();
        if (root.num_edges == 0) {
            throw std::logic_error("the search has no root actions to select from");
        }
        const auto action_of = [&](std::size_t i) {
            return static_cast<std::size_t>(first.edges[root.first_edge + i].action);
        };
        if (temperature == 0.0) {
            if (search.gumbel.has_value()) {
                const GumbelRoot& gumbel = *search.gumbel;
                const auto sigma = gumbel_sigma(search);
                const std::uint32_t most = *std::max_element(gumbel.own_visits.begin(), gumbel.own_visits.end());
                std::size_t best = 0;
                double best_score = -std::numeric_limits<double>::infinity();
                for (std::size_t i = 0; i < sigma.size(); ++i) {
                    const double score = gumbel.noise[i] + gumbel.logits[i] + sigma[i];
                    if (gumbel.own_visits[i] == most && score > best_score) {
                        best_score = score;
                        best = i;
                    }
                }
                return action_of(best);
            }
            const auto counts = visits(search);
            std::size_t best = 0;
            double best_key = -1.0;
            for (std::size_t i = 0; i < root.num_edges; ++i) {
                // Most visits; the prior breaks ties (e.g. before any simulation).
                const double key = counts[action_of(i)] + first.edges[root.first_edge + i].prior * 1e-3;
                if (key > best_key) {
                    best_key = key;
                    best = i;
                }
            }
            return action_of(best);
        }
        const auto policy = policy_target(search);
        std::vector<double> weights(root.num_edges);
        double top = 0.0;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            top = std::max(top, static_cast<double>(policy[action_of(i)]));
        }
        for (std::size_t i = 0; i < weights.size(); ++i) {
            const double p = policy[action_of(i)];
            weights[i] = top > 0.0 ? std::pow(p / top, 1.0 / temperature) : 1.0;
        }
        if (std::accumulate(weights.begin(), weights.end(), 0.0) <= 0.0) {
            std::fill(weights.begin(), weights.end(), 1.0);
        }
        std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
        return action_of(distribution(search.rng));
    }

    SearchStats stats(const Search& search) const {
        SearchStats result;
        result.simulations = search.simulations;
        result.evaluations = search.evaluations;
        result.worlds = static_cast<std::uint32_t>(search.worlds.size());
        result.max_depth = search.max_depth;
        result.mean_depth = search.simulations == 0
            ? 0.0
            : static_cast<double>(search.depth_sum) / static_cast<double>(search.simulations);
        for (const Tree& tree : search.worlds) {
            result.nodes += tree.nodes.size();
            result.edges += tree.edges.size();
            result.coalesced_outcomes += tree.coalesced;
            result.rebuilt_games += tree.rebuilt;
            for (const Node& node : tree.nodes) {
                result.chance_nodes += node.kind == NodeKind::chance;
                result.games += node.game != nullptr;
            }
            result.tree_bytes += tree.nodes.capacity() * sizeof(Node) + tree.edges.capacity() * sizeof(TreeEdge);
        }
        result.tree_bytes += result.games * sizeof(Game);
        return result;
    }
};

SearchPool::SearchPool(
    std::shared_ptr<const ActionCodec> codec,
    std::shared_ptr<const ObservationEncoder> encoder,
    SearchConfig config)
    : impl_(std::make_unique<Impl>(std::move(codec), std::move(encoder), config)) {}

SearchPool::~SearchPool() = default;
SearchPool::SearchPool(SearchPool&&) noexcept = default;
SearchPool& SearchPool::operator=(SearchPool&&) noexcept = default;

const SearchConfig& SearchPool::config() const noexcept { return impl_->config; }
std::size_t SearchPool::observation_size() const noexcept { return impl_->encoder->size(); }
std::size_t SearchPool::action_size() const noexcept { return impl_->codec->size(); }
int SearchPool::max_players() const noexcept { return impl_->config.max_players; }

std::int64_t SearchPool::add(const Game& game, std::uint64_t seed, const HandTracker* tracker) {
    const int players = static_cast<int>(game.colors().size());
    if (players > impl_->config.max_players) {
        throw std::invalid_argument("the game has more seats than max_players");
    }
    auto search = std::make_unique<Search>();
    search->id = impl_->next_id;
    search->rng.seed(seed);
    search->players = players;
    search->root_player = game.current_player_index();
    search->budget = impl_->config.simulations;
    if (game.winning_color().has_value()) {
        search->done = true;
    } else {
        std::vector<Game> worlds;
        if (impl_->config.determinizations > 0) {
            if (tracker == nullptr) {
                throw std::invalid_argument("determinized search needs the root player's HandTracker");
            }
            if (tracker->observer() != search->root_player) {
                throw std::invalid_argument("the tracker must follow the player to move");
            }
            search->determinized = true;
            for (int k = 0; k < impl_->config.determinizations; ++k) {
                Game world = sample_world(game, *tracker, search->rng);
                const auto same = std::find_if(worlds.begin(), worlds.end(), [&](const Game& other) {
                    return same_hidden_state(other, world);
                });
                if (same != worlds.end()) {
                    search->schedule.push_back(static_cast<std::uint32_t>(same - worlds.begin()));
                } else {
                    search->schedule.push_back(static_cast<std::uint32_t>(worlds.size()));
                    worlds.push_back(std::move(world));
                }
            }
        } else {
            worlds.push_back(game);
            search->schedule.push_back(0);
        }
        for (Game& world : worlds) {
            Tree tree;
            tree.players = players;
            impl_->add_node(tree, std::make_unique<Game>(std::move(world)), kNone, kNone, kNone);
            search->worlds.push_back(std::move(tree));
        }
    }
    impl_->searches.emplace(impl_->next_id, std::move(search));
    return impl_->next_id++;
}

std::size_t SearchPool::collect(std::span<float> observations, std::span<std::int64_t> ids) {
    return impl_->collect(observations, ids);
}

void SearchPool::apply(std::span<const float> logits, std::span<const float> values) {
    impl_->apply(logits, values);
}

std::size_t SearchPool::pending() const noexcept { return impl_->batch.size(); }

const Game& SearchPool::pending_game(std::size_t row) const {
    const auto& entry = impl_->batch.at(row);
    return *entry.search->worlds[entry.leaf.world].nodes[entry.leaf.node].game;
}

bool SearchPool::contains(std::int64_t id) const { return impl_->searches.contains(id); }

std::vector<std::int64_t> SearchPool::ids() const {
    std::vector<std::int64_t> result;
    for (const auto& [id, search] : impl_->searches) {
        result.push_back(id);
    }
    return result;
}

std::size_t SearchPool::size() const noexcept { return impl_->searches.size(); }
bool SearchPool::done(std::int64_t id) const { return impl_->find(id).done; }

std::vector<std::uint32_t> SearchPool::visits(std::int64_t id) const {
    return impl_->visits(impl_->find(id));
}

std::vector<double> SearchPool::q(std::int64_t id) const { return impl_->q(impl_->find(id)); }

std::vector<float> SearchPool::policy_target(std::int64_t id) const {
    return impl_->policy_target(impl_->find(id));
}

std::vector<float> SearchPool::value(std::int64_t id) const {
    return impl_->value(impl_->find(id));
}

std::size_t SearchPool::select(std::int64_t id, double temperature) {
    return impl_->select(impl_->find(id), temperature);
}

bool SearchPool::advance(std::int64_t id, std::size_t action_index) {
    return impl_->advance(impl_->find(id), action_index);
}

void SearchPool::remove(std::int64_t id) {
    Search& search = impl_->find(id);
    if (search.pending > 0) {
        throw std::logic_error("cannot remove a search with leaves awaiting apply");
    }
    impl_->searches.erase(id);
}

SearchStats SearchPool::stats(std::int64_t id) const { return impl_->stats(impl_->find(id)); }

}  // namespace cppanatron
