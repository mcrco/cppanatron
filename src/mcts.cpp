#include "cppanatron/mcts.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cppanatron {
namespace {

constexpr double kProbabilityTolerance = 1e-12;

std::size_t color_index(const Game& game, Color color) {
    const auto it = std::find(game.colors().begin(), game.colors().end(), color);
    if (it == game.colors().end()) {
        throw std::logic_error("game current color is absent from its color list");
    }
    return static_cast<std::size_t>(std::distance(game.colors().begin(), it));
}

void normalize_probabilities(std::vector<ChanceOutcome>& outcomes) {
    const double total = std::accumulate(
        outcomes.begin(), outcomes.end(), 0.0,
        [](double sum, const ChanceOutcome& outcome) { return sum + outcome.probability; });
    if (outcomes.empty() || total <= kProbabilityTolerance) {
        throw std::logic_error("action produced no positive-probability outcomes");
    }
    for (auto& outcome : outcomes) {
        outcome.probability /= total;
    }
}

} // namespace

std::vector<ChanceOutcome> action_outcomes(const Game& game, const Action& action) {
    std::vector<ChanceOutcome> outcomes;

    if (action.type == ActionType::roll) {
        constexpr std::array<int, 11> multiplicities{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
        for (int total = 2; total <= 12; ++total) {
            Game next = game;
            const Dice dice = total <= 7 ? Dice{1, total - 1} : Dice{6, total - 6};
            next.execute(action, dice);
            outcomes.push_back(
                {std::move(next),
                 static_cast<double>(multiplicities[static_cast<std::size_t>(total - 2)]) / 36.0});
        }
        return outcomes;
    }

    if (action.type == ActionType::buy_development_card) {
        std::map<DevelopmentCard, int> counts;
        for (DevelopmentCard card : game.development_deck()) {
            ++counts[card];
        }
        const double total = static_cast<double>(game.development_cards_remaining());
        for (const auto& [card, count] : counts) {
            Game next = game;
            next.execute(action, std::nullopt, card);
            outcomes.push_back({std::move(next), static_cast<double>(count) / total});
        }
        normalize_probabilities(outcomes);
        return outcomes;
    }

    if (action.type == ActionType::move_robber) {
        const auto& move = std::get<RobberMove>(action.value);
        if (move.victim.has_value()) {
            const auto& resources = game.player(*move.victim).resources;
            const int total = std::accumulate(resources.begin(), resources.end(), 0);
            if (total > 0) {
                for (std::size_t index = 0; index < resources.size(); ++index) {
                    if (resources[index] <= 0) {
                        continue;
                    }
                    Game next = game;
                    next.execute(action, std::nullopt, std::nullopt, static_cast<Resource>(index));
                    outcomes.push_back({std::move(next), static_cast<double>(resources[index]) /
                                                             static_cast<double>(total)});
                }
                normalize_probabilities(outcomes);
                return outcomes;
            }
        }
    }

    Game next = game;
    next.execute(action);
    outcomes.push_back({std::move(next), 1.0});
    return outcomes;
}

struct MCTSSearch::Impl {
    struct Node;

    struct OutcomeEdge {
        std::unique_ptr<Node> child;
        double probability{};
    };

    struct ActionEdge {
        std::size_t action_index{};
        double prior{};
        bool stochastic{};
        std::vector<OutcomeEdge> outcomes;

        [[nodiscard]] std::uint32_t visits() const {
            return std::accumulate(outcomes.begin(), outcomes.end(), std::uint32_t{0},
                                   [](std::uint32_t sum, const OutcomeEdge& outcome) {
                                       return sum + outcome.child->visits;
                                   });
        }
    };

    struct Node {
        explicit Node(Game state, int minimum_discard = -1)
            : game(std::move(state)), to_play(game.current_color()),
              min_discard_resource(minimum_discard) {}

        Game game;
        Color to_play;
        std::uint32_t visits{};
        double value_sum{};
        bool expanded{};
        int min_discard_resource{-1};
        std::vector<ActionEdge> actions;

        [[nodiscard]] double value() const noexcept {
            return visits == 0 ? 0.0 : value_sum / static_cast<double>(visits);
        }
    };

    Impl(const Game& game, FlatActionSpace flat_action_space, double exploration,
         std::uint64_t seed, bool prune)
        : root(game), action_space(std::move(flat_action_space)), c_puct(exploration),
          random(seed), canonical_pruning(prune) {
        if (!std::isfinite(c_puct) || c_puct < 0.0) {
            throw std::invalid_argument("c_puct must be finite and non-negative");
        }
    }

    Node root;
    FlatActionSpace action_space;
    double c_puct;
    std::mt19937_64 random;
    bool canonical_pruning{};
    Node* pending_leaf{};
    std::vector<Node*> pending_path;
    std::uint64_t completed_simulations{};
    std::uint64_t depth_sum{};
    std::uint32_t maximum_depth{};
    std::uint32_t retained_root_visits{};
    std::uint64_t pruned_actions{};
    std::uint64_t coalesced_outcomes{};
    bool tree_reused{};

    [[nodiscard]] static bool is_stochastic(const Game& game, const Action& action) {
        if (action.type == ActionType::roll ||
            action.type == ActionType::buy_development_card) {
            return true;
        }
        if (action.type != ActionType::move_robber) {
            return false;
        }
        const auto& move = std::get<RobberMove>(action.value);
        if (!move.victim.has_value()) {
            return false;
        }
        const auto& resources = game.player(*move.victim).resources;
        return std::accumulate(resources.begin(), resources.end(), 0) > 0;
    }

    [[nodiscard]] static int child_minimum_discard(
        const Node& parent,
        const Action& action,
        const Game& child) {
        if (action.type != ActionType::discard_resource ||
            child.current_prompt() != ActionPrompt::discard ||
            child.current_color() != parent.to_play) {
            return -1;
        }
        return std::max(
            parent.min_discard_resource,
            static_cast<int>(std::get<Resource>(action.value)));
    }

    [[nodiscard]] static bool equivalent_edges(
        const ActionEdge& lhs,
        const ActionEdge& rhs) {
        if (lhs.stochastic != rhs.stochastic ||
            lhs.outcomes.size() != rhs.outcomes.size()) {
            return false;
        }
        for (std::size_t i = 0; i < lhs.outcomes.size(); ++i) {
            if (std::abs(lhs.outcomes[i].probability - rhs.outcomes[i].probability) >
                    kProbabilityTolerance ||
                !lhs.outcomes[i].child->game.search_equivalent(
                    rhs.outcomes[i].child->game)) {
                return false;
            }
        }
        return true;
    }

    void clear_metrics() noexcept {
        completed_simulations = 0;
        depth_sum = 0;
        maximum_depth = 0;
        retained_root_visits = root.visits;
        pruned_actions = 0;
        coalesced_outcomes = 0;
    }

    [[nodiscard]] static bool canonical_discard_action_allowed(
        const Node& node,
        const Action& action) {
        const auto resource = static_cast<std::size_t>(std::get<Resource>(action.value));
        if (static_cast<int>(resource) < node.min_discard_resource) {
            return false;
        }
        const auto& hand = node.game.player(action.color).resources;
        const int suffix_cards = std::accumulate(
            hand.begin() + static_cast<std::ptrdiff_t>(resource),
            hand.end(),
            0);
        return suffix_cards >= node.game.remaining_discards(action.color);
    }

    void validate_logits(std::span<const float> logits) const {
        if (logits.size() != action_space.size()) {
            throw std::invalid_argument("policy logits have incorrect size");
        }
    }

    void expand(Node& node, std::span<const float> logits) {
        validate_logits(logits);
        if (node.expanded) {
            throw std::logic_error("cannot expand a node twice");
        }
        if (node.game.winning_color().has_value()) {
            throw std::logic_error("cannot expand a terminal node");
        }

        const auto& all_playable_actions = node.game.playable_actions();
        if (all_playable_actions.empty()) {
            node.expanded = true;
            return;
        }

        std::vector<const Action*> playable_actions;
        playable_actions.reserve(all_playable_actions.size());
        for (const Action& action : all_playable_actions) {
            if (canonical_pruning &&
                node.game.current_prompt() == ActionPrompt::discard &&
                action.type == ActionType::discard_resource &&
                !canonical_discard_action_allowed(node, action)) {
                ++pruned_actions;
                continue;
            }
            playable_actions.push_back(&action);
        }
        if (playable_actions.empty()) {
            throw std::logic_error("canonical pruning removed every legal action");
        }

        std::vector<std::size_t> indices;
        indices.reserve(playable_actions.size());
        float max_logit = -std::numeric_limits<float>::infinity();
        for (const Action* action : playable_actions) {
            const std::size_t index = action_space.index(*action, node.game.colors());
            indices.push_back(index);
            max_logit = std::max(max_logit, logits[index]);
        }

        std::vector<double> weights;
        weights.reserve(indices.size());
        double weight_sum = 0.0;
        for (std::size_t index : indices) {
            const double weight =
                std::isfinite(logits[index])
                    ? std::exp(static_cast<double>(logits[index]) - static_cast<double>(max_logit))
                    : 0.0;
            weights.push_back(weight);
            weight_sum += weight;
        }
        if (!std::isfinite(weight_sum) || weight_sum <= kProbabilityTolerance) {
            std::fill(weights.begin(), weights.end(), 1.0);
            weight_sum = static_cast<double>(weights.size());
        }

        node.actions.reserve(playable_actions.size());
        for (std::size_t i = 0; i < playable_actions.size(); ++i) {
            ActionEdge edge;
            edge.action_index = indices[i];
            edge.prior = weights[i] / weight_sum;
            const Action& action = *playable_actions[i];
            edge.stochastic = is_stochastic(node.game, action);
            auto outcomes = action_outcomes(node.game, action);
            edge.outcomes.reserve(outcomes.size());
            for (auto& outcome : outcomes) {
                const auto duplicate = std::find_if(
                    edge.outcomes.begin(),
                    edge.outcomes.end(),
                    [&](const OutcomeEdge& existing) {
                        return existing.child->game.search_equivalent(outcome.game);
                    });
                if (canonical_pruning && duplicate != edge.outcomes.end()) {
                    duplicate->probability += outcome.probability;
                    ++coalesced_outcomes;
                    continue;
                }
                const int minimum_discard = canonical_pruning
                    ? child_minimum_discard(node, action, outcome.game)
                    : -1;
                edge.outcomes.push_back(
                    {std::make_unique<Node>(std::move(outcome.game), minimum_discard),
                     outcome.probability});
            }
            if (canonical_pruning) {
                const auto duplicate = std::find_if(
                    node.actions.begin(),
                    node.actions.end(),
                    [&](const ActionEdge& existing) {
                        return equivalent_edges(existing, edge);
                    });
                if (duplicate != node.actions.end()) {
                    duplicate->prior += edge.prior;
                    ++pruned_actions;
                    continue;
                }
            }
            node.actions.push_back(std::move(edge));
        }
        node.expanded = true;
    }

    [[nodiscard]] Node& select_child(Node& node) {
        const std::uint32_t total_visits = std::accumulate(
            node.actions.begin(), node.actions.end(), std::uint32_t{0},
            [](std::uint32_t sum, const ActionEdge& action) { return sum + action.visits(); });
        ActionEdge* best_action = nullptr;
        double best_score = -std::numeric_limits<double>::infinity();
        for (auto& action : node.actions) {
            const std::uint32_t action_visits = action.visits();
            double q_value = 0.0;
            if (action_visits > 0) {
                for (const auto& outcome : action.outcomes) {
                    const Node& child = *outcome.child;
                    const double oriented_value =
                        child.to_play == node.to_play ? child.value() : -child.value();
                    q_value += outcome.probability * oriented_value;
                }
            }
            const double u_value = c_puct * action.prior *
                                   std::sqrt(static_cast<double>(total_visits) + 1.0) /
                                   (1.0 + static_cast<double>(action_visits));
            const double score = q_value + u_value;
            if (score > best_score) {
                best_score = score;
                best_action = &action;
            }
        }
        if (best_action == nullptr || best_action->outcomes.empty()) {
            throw std::logic_error("expanded search node has no outcomes");
        }

        std::vector<double> probabilities;
        probabilities.reserve(best_action->outcomes.size());
        for (const auto& outcome : best_action->outcomes) {
            probabilities.push_back(outcome.probability);
        }
        std::discrete_distribution<std::size_t> distribution(probabilities.begin(),
                                                             probabilities.end());
        return *best_action->outcomes[distribution(random)].child;
    }

    void backup(const std::vector<Node*>& path, double leaf_value) {
        double value = std::clamp(leaf_value, -1.0, 1.0);
        for (std::size_t index = path.size(); index-- > 0;) {
            Node& node = *path[index];
            ++node.visits;
            node.value_sum += value;
            if (index > 0 && path[index - 1]->to_play != node.to_play) {
                value = -value;
            }
        }
    }

    void record_simulation(const std::vector<Node*>& path) noexcept {
        const auto depth =
            path.empty() ? std::uint32_t{0} : static_cast<std::uint32_t>(path.size() - 1);
        ++completed_simulations;
        depth_sum += depth;
        maximum_depth = std::max(maximum_depth, depth);
    }

    [[nodiscard]] std::uint32_t principal_variation_depth() const noexcept {
        const Node* node = &root;
        std::uint32_t depth = 0;
        while (node->expanded && !node->actions.empty()) {
            const auto best_action = std::max_element(
                node->actions.begin(), node->actions.end(),
                [](const ActionEdge& lhs, const ActionEdge& rhs) {
                    return lhs.visits() < rhs.visits();
                });
            if (best_action == node->actions.end() || best_action->visits() == 0) {
                break;
            }
            const auto best_outcome = std::max_element(
                best_action->outcomes.begin(), best_action->outcomes.end(),
                [](const OutcomeEdge& lhs, const OutcomeEdge& rhs) {
                    return lhs.child->visits < rhs.child->visits;
                });
            if (best_outcome == best_action->outcomes.end() || best_outcome->child->visits == 0) {
                break;
            }
            node = best_outcome->child.get();
            ++depth;
        }
        return depth;
    }
};

MCTSSearch::MCTSSearch(const Game& root_game, FlatActionSpace action_space, double c_puct,
                       std::uint64_t seed, bool canonical_pruning)
    : impl_(std::make_unique<Impl>(
          root_game,
          std::move(action_space),
          c_puct,
          seed,
          canonical_pruning)) {}

MCTSSearch::~MCTSSearch() = default;
MCTSSearch::MCTSSearch(MCTSSearch&&) noexcept = default;
MCTSSearch& MCTSSearch::operator=(MCTSSearch&&) noexcept = default;

void MCTSSearch::initialize_root(std::span<const float> policy_logits) {
    if (impl_->pending_leaf != nullptr) {
        throw std::logic_error("cannot initialize root with a pending leaf");
    }
    impl_->expand(impl_->root, policy_logits);
}

const Game* MCTSSearch::select_leaf() {
    if (impl_->pending_leaf != nullptr) {
        throw std::logic_error("previous leaf has not been evaluated");
    }
    if (!impl_->root.expanded) {
        throw std::logic_error("search root has not been initialized");
    }

    Impl::Node* node = &impl_->root;
    std::vector<Impl::Node*> path{node};
    while (node->expanded && !node->game.winning_color().has_value() && !node->actions.empty()) {
        node = &impl_->select_child(*node);
        path.push_back(node);
    }

    const auto winner = node->game.winning_color();
    if (winner.has_value()) {
        impl_->record_simulation(path);
        impl_->backup(path, *winner == node->to_play ? 1.0 : -1.0);
        return nullptr;
    }
    if (node->expanded && node->actions.empty()) {
        impl_->record_simulation(path);
        impl_->backup(path, 0.0);
        return nullptr;
    }

    impl_->pending_leaf = node;
    impl_->pending_path = std::move(path);
    return &node->game;
}

void MCTSSearch::evaluate_leaf(std::span<const float> policy_logits, double value) {
    if (impl_->pending_leaf == nullptr) {
        throw std::logic_error("no pending search leaf");
    }
    if (!std::isfinite(value)) {
        throw std::invalid_argument("leaf value must be finite");
    }
    impl_->expand(*impl_->pending_leaf, policy_logits);
    impl_->record_simulation(impl_->pending_path);
    impl_->backup(impl_->pending_path, value);
    impl_->pending_leaf = nullptr;
    impl_->pending_path.clear();
}

bool MCTSSearch::has_pending_leaf() const noexcept { return impl_->pending_leaf != nullptr; }

bool MCTSSearch::root_expanded() const noexcept { return impl_->root.expanded; }

int MCTSSearch::pending_player_index() const {
    if (impl_->pending_leaf == nullptr) {
        throw std::logic_error("no pending search leaf");
    }
    return static_cast<int>(color_index(impl_->pending_leaf->game, impl_->pending_leaf->to_play));
}

const Game& MCTSSearch::root_game() const noexcept { return impl_->root.game; }

std::vector<std::uint32_t> MCTSSearch::root_visits() const {
    std::vector<std::uint32_t> visits(impl_->action_space.size(), 0);
    for (const auto& action : impl_->root.actions) {
        visits[action.action_index] = action.visits();
    }
    return visits;
}

std::vector<double> MCTSSearch::root_action_values() const {
    std::vector<double> values(
        impl_->action_space.size(),
        std::numeric_limits<double>::quiet_NaN());
    for (const auto& action : impl_->root.actions) {
        if (action.visits() == 0) {
            continue;
        }
        double expected_value = 0.0;
        for (const auto& outcome : action.outcomes) {
            const auto& child = *outcome.child;
            const double oriented_value =
                child.to_play == impl_->root.to_play ? child.value() : -child.value();
            expected_value += outcome.probability * oriented_value;
        }
        values[action.action_index] = expected_value;
    }
    return values;
}

MCTSSearchMetrics MCTSSearch::metrics() const noexcept {
    return {
        impl_->completed_simulations,
        impl_->principal_variation_depth(),
        impl_->maximum_depth,
        impl_->completed_simulations == 0
            ? 0.0
            : static_cast<double>(impl_->depth_sum) /
                  static_cast<double>(impl_->completed_simulations),
        impl_->root.value(),
        impl_->retained_root_visits,
        impl_->pruned_actions,
        impl_->coalesced_outcomes,
        impl_->tree_reused,
    };
}

void MCTSSearch::reset_metrics() noexcept { impl_->clear_metrics(); }

bool MCTSSearch::advance(std::size_t action_index) {
    if (impl_->pending_leaf != nullptr) {
        throw std::logic_error("cannot advance with a pending leaf");
    }
    if (!impl_->root.expanded) {
        return false;
    }
    const auto edge = std::find_if(
        impl_->root.actions.begin(),
        impl_->root.actions.end(),
        [action_index](const Impl::ActionEdge& action) {
            return action.action_index == action_index;
        });
    if (edge == impl_->root.actions.end() || edge->stochastic ||
        edge->outcomes.size() != 1) {
        return false;
    }
    std::unique_ptr<Impl::Node> next = std::move(edge->outcomes.front().child);
    impl_->root = std::move(*next);
    impl_->tree_reused = true;
    impl_->clear_metrics();
    return true;
}

void MCTSSearch::add_root_dirichlet_noise(double alpha, double fraction) {
    if (!impl_->root.expanded) {
        throw std::logic_error("cannot add noise before root initialization");
    }
    if (!std::isfinite(alpha) || alpha <= 0.0) {
        throw std::invalid_argument("Dirichlet alpha must be positive and finite");
    }
    if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0) {
        throw std::invalid_argument("Dirichlet fraction must be in [0, 1]");
    }
    if (impl_->root.actions.empty() || fraction == 0.0) {
        return;
    }

    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> noise;
    noise.reserve(impl_->root.actions.size());
    for (std::size_t i = 0; i < impl_->root.actions.size(); ++i) {
        noise.push_back(gamma(impl_->random));
    }
    const double total = std::accumulate(noise.begin(), noise.end(), 0.0);
    if (total <= kProbabilityTolerance) {
        std::fill(noise.begin(), noise.end(), 1.0);
    }
    const double denominator = std::accumulate(noise.begin(), noise.end(), 0.0);
    for (std::size_t i = 0; i < impl_->root.actions.size(); ++i) {
        auto& action = impl_->root.actions[i];
        action.prior = (1.0 - fraction) * action.prior + fraction * noise[i] / denominator;
    }
}

} // namespace cppanatron
