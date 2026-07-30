#include "cppanatron/value_player.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>

namespace cppanatron {
namespace {

constexpr double kTranslateVariety = 4.0;
constexpr double kProbabilityPoint = 2.778 / 100.0;

double number_probability(int number) {
    return static_cast<double>(6 - std::abs(7 - number)) / 36.0;
}

std::array<double, 5> production_for(
    const Game& game,
    Color color,
    bool consider_robber) {
    std::array<double, 5> result{};
    for (const auto& [coordinate, tile_index] :
         game.board().map().coordinate_to_tile()) {
        const Tile& tile =
            game.board().map().tiles().at(static_cast<std::size_t>(tile_index));
        if (tile.kind != TileKind::land || !tile.resource.has_value() ||
            (consider_robber && coordinate == game.board().robber_coordinate())) {
            continue;
        }
        const double probability = number_probability(*tile.number);
        for (int node : tile.nodes) {
            const auto building = game.board().buildings().find(node);
            if (building == game.board().buildings().end() ||
                building->second.color != color) {
                continue;
            }
            const double multiplier =
                building->second.building == Building::settlement ? 1.0 : 2.0;
            result[static_cast<std::size_t>(*tile.resource)] +=
                multiplier * probability;
        }
    }
    return result;
}

double production_value(
    const std::array<double, 5>& production,
    bool include_variety) {
    const double sum =
        std::accumulate(production.begin(), production.end(), 0.0);
    const int variety = static_cast<int>(std::count_if(
        production.begin(), production.end(), [](double value) {
            return value != 0.0;
        }));
    return sum +
           (include_variety
                ? variety * kTranslateVariety * kProbabilityPoint
                : 0.0);
}

std::set<int> network_nodes(const Game& game, Color color) {
    std::set<int> result;
    const PlayerState& player = game.player(color);
    result.insert(player.settlements.begin(), player.settlements.end());
    result.insert(player.cities.begin(), player.cities.end());
    for (Edge edge : player.roads) {
        result.insert(edge.a);
        result.insert(edge.b);
    }
    return result;
}

std::array<double, 5> production_at_nodes(
    const Game& game,
    const std::set<int>& nodes) {
    std::array<double, 5> result{};
    for (const auto& [coordinate, tile_index] :
         game.board().map().coordinate_to_tile()) {
        (void)coordinate;
        const Tile& tile =
            game.board().map().tiles().at(static_cast<std::size_t>(tile_index));
        if (tile.kind != TileKind::land || !tile.resource.has_value()) {
            continue;
        }
        const double probability = number_probability(*tile.number);
        for (int node : tile.nodes) {
            if (nodes.contains(node)) {
                result[static_cast<std::size_t>(*tile.resource)] += probability;
            }
        }
    }
    return result;
}

std::pair<double, double> reachable_production(const Game& game, Color color) {
    const std::set<int> zero = network_nodes(game, color);
    const std::vector<int> globally_buildable =
        game.board().buildable_node_ids(color, true);
    std::set<int> owned_or_buildable(
        globally_buildable.begin(), globally_buildable.end());
    const PlayerState& player = game.player(color);
    owned_or_buildable.insert(
        player.settlements.begin(), player.settlements.end());
    owned_or_buildable.insert(player.cities.begin(), player.cities.end());

    std::set<int> zero_buildable;
    std::set_intersection(
        zero.begin(),
        zero.end(),
        owned_or_buildable.begin(),
        owned_or_buildable.end(),
        std::inserter(zero_buildable, zero_buildable.end()));

    std::set<int> level_one = zero;
    for (int node : zero) {
        if (game.board().is_enemy_node(node, color)) {
            continue;
        }
        for (Edge edge : game.board().map().land_edges()) {
            if (edge.a != node && edge.b != node) {
                continue;
            }
            if (game.board().is_enemy_road(edge, color)) {
                continue;
            }
            level_one.insert(edge.a == node ? edge.b : edge.a);
        }
    }
    std::set<int> one_buildable;
    std::set_intersection(
        level_one.begin(),
        level_one.end(),
        owned_or_buildable.begin(),
        owned_or_buildable.end(),
        std::inserter(one_buildable, one_buildable.end()));

    const auto zero_production = production_at_nodes(game, zero_buildable);
    const auto one_production = production_at_nodes(game, one_buildable);
    return {
        std::accumulate(zero_production.begin(), zero_production.end(), 0.0),
        std::accumulate(one_production.begin(), one_production.end(), 0.0),
    };
}

int owned_tile_count(const Game& game, Color color) {
    std::set<int> tiles;
    const PlayerState& player = game.player(color);
    std::set<int> nodes(player.settlements.begin(), player.settlements.end());
    nodes.insert(player.cities.begin(), player.cities.end());
    for (int tile_index : game.board().map().land_tile_indices()) {
        const Tile& tile =
            game.board().map().tiles().at(static_cast<std::size_t>(tile_index));
        if (std::any_of(tile.nodes.begin(), tile.nodes.end(), [&](int node) {
                return nodes.contains(node);
            })) {
            tiles.insert(tile.id);
        }
    }
    return static_cast<int>(tiles.size());
}

}  // namespace

double value_score(
    const Game& game,
    Color perspective,
    const ValueWeights& weights) {
    const PlayerState& player = game.player(perspective);
    const auto perspective_it =
        std::find(game.colors().begin(), game.colors().end(), perspective);
    const int perspective_index = static_cast<int>(
        std::distance(game.colors().begin(), perspective_it));
    const Color enemy = game.colors()[static_cast<std::size_t>(
        (perspective_index + 1) % static_cast<int>(game.colors().size()))];

    const double production =
        production_value(production_for(game, perspective, true), true);
    const double enemy_production =
        production_value(production_for(game, enemy, true), false);
    const auto [reachable_zero, reachable_one] =
        reachable_production(game, perspective);

    const double distance_to_city =
        (std::max(2 - player.resources[3], 0) +
         std::max(3 - player.resources[4], 0)) /
        5.0;
    const double distance_to_settlement =
        (std::max(1 - player.resources[3], 0) +
         std::max(1 - player.resources[2], 0) +
         std::max(1 - player.resources[1], 0) +
         std::max(1 - player.resources[0], 0)) /
        4.0;
    const double hand_synergy =
        (2.0 - distance_to_city - distance_to_settlement) / 2.0;
    const int resources =
        std::accumulate(player.resources.begin(), player.resources.end(), 0);
    const int development_cards = std::accumulate(
        player.development_cards.begin(), player.development_cards.end(), 0);
    const int buildable_nodes =
        static_cast<int>(game.board().buildable_node_ids(perspective).size());
    const double longest_road_factor =
        buildable_nodes == 0 ? weights.longest_road : 0.1;

    return player.victory_points * weights.public_vps +
           production * weights.production +
           enemy_production * weights.enemy_production +
           reachable_zero * weights.reachable_production_0 +
           reachable_one * weights.reachable_production_1 +
           hand_synergy * weights.hand_synergy +
           buildable_nodes * weights.buildable_nodes +
           owned_tile_count(game, perspective) * weights.num_tiles +
           resources * weights.hand_resources +
           (resources > 7 ? weights.discard_penalty : 0.0) +
           player.longest_road_length * longest_road_factor +
           development_cards * weights.hand_devs +
           player.played_development_cards[
               static_cast<std::size_t>(DevelopmentCard::knight)] *
               weights.army_size;
}

Action value_action(const Game& game, const ValueWeights& weights) {
    if (game.playable_actions().empty()) {
        throw std::logic_error("value player has no playable actions");
    }
    if (game.playable_actions().size() == 1) {
        return game.playable_actions().front();
    }

    double best_value = -std::numeric_limits<double>::infinity();
    Action best_action = game.playable_actions().front();
    for (const Action& action : game.playable_actions()) {
        Game candidate = game;
        candidate.execute(action);
        const double value = value_score(candidate, action.color, weights);
        if (value > best_value) {
            best_value = value;
            best_action = action;
        }
    }
    return best_action;
}

}  // namespace cppanatron
