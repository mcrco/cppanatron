#include "cppanatron/value_player.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace cppanatron {
namespace {

constexpr double kTranslateVariety = 4.0;
constexpr double kProbabilityPoint = 2.778 / 100.0;

double number_probability(int number) {
    return static_cast<double>(6 - std::abs(7 - number)) / 36.0;
}

/** Flat per-node scratch reused by every evaluation on a thread. */
struct Scratch {
    std::vector<std::uint8_t> open;      // the distance rule allows a settlement
    std::vector<std::uint8_t> zero;      // in the network (buildings, road ends)
    std::vector<std::uint8_t> one;       // one road away from the network
    std::vector<std::uint8_t> ends;      // ends of the player's roads
    std::vector<std::uint8_t> tiles;     // land tiles touched, by tiles() index
};

Scratch& scratch(const CatanMap& map) {
    thread_local Scratch result;
    const auto nodes = static_cast<std::size_t>(map.num_node_ids());
    result.open.assign(nodes, 0);
    result.zero.assign(nodes, 0);
    result.one.assign(nodes, 0);
    result.ends.assign(nodes, 0);
    result.tiles.assign(map.tiles().size(), 0);
    return result;
}

/**
 * Pips of a node by resource as Catanatron's get_node_production adds them:
 * adjacent land tiles in tile order, the robber's tile left out if given.
 */
std::array<double, 5> node_production(const CatanMap& map, int node, int robber_tile) {
    std::array<double, 5> result{};
    for (int tile_index : map.adjacent_land_tiles(node)) {
        const Tile& tile = map.tiles()[static_cast<std::size_t>(tile_index)];
        if (tile.resource.has_value() && tile_index != robber_tile) {
            result[static_cast<std::size_t>(*tile.resource)] += number_probability(*tile.number);
        }
    }
    return result;
}

/** Catanatron's build_production_features: settlements, then cities twice. */
std::array<double, 5> production_for(
    const Game& game,
    Color color,
    bool consider_robber) {
    const CatanMap& map = game.board().map();
    const int robber_tile =
        consider_robber ? map.coordinate_to_tile().at(game.board().robber_coordinate()) : -1;
    const PlayerState& player = game.player(color);
    std::array<double, 5> result{};
    for (int node : player.settlements) {
        const auto production = node_production(map, node, robber_tile);
        for (std::size_t r = 0; r < result.size(); ++r) {
            result[r] += production[r];
        }
    }
    for (int node : player.cities) {
        const auto production = node_production(map, node, robber_tile);
        for (std::size_t r = 0; r < result.size(); ++r) {
            result[r] += 2.0 * production[r];
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

/** Total pips (robber ignored) of the marked land nodes. */
double production_at_nodes(const CatanMap& map, const std::vector<std::uint8_t>& marked) {
    std::array<double, 5> result{};
    for (int node : map.land_node_list()) {
        if (marked[static_cast<std::size_t>(node)] == 0) {
            continue;
        }
        const auto production = node_production(map, node, -1);
        for (std::size_t r = 0; r < result.size(); ++r) {
            result[r] += production[r];
        }
    }
    return std::accumulate(result.begin(), result.end(), 0.0);
}

struct Reach {
    double zero{};
    double one{};
    /** Open nodes on the player's roads: Board::buildable_node_ids(color).size(). */
    int buildable_nodes{};
};

/**
 * Catanatron's reachability_features at levels 0 and 1 and the buildable
 * node count, from flat per-node marks instead of std::sets. Openness is the
 * distance rule (Board's buildable ids); the network is the player's
 * buildings and road ends (its road components).
 */
Reach reachable_production(const Game& game, Color color) {
    const Board& board = game.board();
    const CatanMap& map = board.map();
    Scratch& marks = scratch(map);
    for (int node : map.land_node_list()) {
        marks.open[static_cast<std::size_t>(node)] = 1;
    }
    for (const auto& [node, building] : board.buildings()) {
        (void)building;
        marks.open[static_cast<std::size_t>(node)] = 0;
        for (int neighbor : map.neighbors(node)) {
            marks.open[static_cast<std::size_t>(neighbor)] = 0;
        }
    }
    const PlayerState& player = game.player(color);
    for (int node : player.settlements) {
        marks.zero[static_cast<std::size_t>(node)] = 1;
    }
    for (int node : player.cities) {
        marks.zero[static_cast<std::size_t>(node)] = 1;
    }
    for (Edge edge : player.roads) {
        for (int node : {edge.a, edge.b}) {
            marks.zero[static_cast<std::size_t>(node)] = 1;
            marks.ends[static_cast<std::size_t>(node)] = 1;
        }
    }

    Reach result;
    for (int node : map.land_node_list()) {
        const auto index = static_cast<std::size_t>(node);
        if (marks.zero[index] == 0) {
            continue;
        }
        marks.one[index] = 1;
        if (board.is_enemy_node(node, color)) {
            continue;
        }
        for (int neighbor : map.neighbors(node)) {
            if (!board.is_enemy_road(Edge{node, neighbor}, color)) {
                marks.one[static_cast<std::size_t>(neighbor)] = 1;
            }
        }
    }
    // Owned (buildings) or buildable (open) nodes count toward production.
    const auto owned_or_open = [&](std::size_t node) {
        return marks.open[node] != 0 || board.is_friendly_node(static_cast<int>(node), color);
    };
    for (int node : map.land_node_list()) {
        const auto index = static_cast<std::size_t>(node);
        const bool counted = owned_or_open(index);
        marks.zero[index] = marks.zero[index] != 0 && counted ? 1 : 0;
        marks.one[index] = marks.one[index] != 0 && counted ? 1 : 0;
        if (marks.open[index] != 0 && marks.ends[index] != 0) {
            ++result.buildable_nodes;
        }
    }
    result.zero = production_at_nodes(map, marks.zero);
    result.one = production_at_nodes(map, marks.one);
    return result;
}

int owned_tile_count(const Game& game, Color color) {
    const CatanMap& map = game.board().map();
    Scratch& marks = scratch(map);
    const PlayerState& player = game.player(color);
    int count = 0;
    const auto touch = [&](int node) {
        for (int tile_index : map.adjacent_land_tiles(node)) {
            auto& seen = marks.tiles[static_cast<std::size_t>(tile_index)];
            count += seen == 0 ? 1 : 0;
            seen = 1;
        }
    };
    for (int node : player.settlements) {
        touch(node);
    }
    for (int node : player.cities) {
        touch(node);
    }
    return count;
}

}  // namespace

double value_score(
    const Game& game,
    Color perspective,
    const ValueWeights& weights,
    ValueEnemies enemies) {
    const PlayerState& player = game.player(perspective);
    const auto perspective_it =
        std::find(game.colors().begin(), game.colors().end(), perspective);
    const int perspective_index = static_cast<int>(
        std::distance(game.colors().begin(), perspective_it));
    const int num_players = static_cast<int>(game.colors().size());
    const auto enemy_at = [&](int offset) {
        return game.colors()[static_cast<std::size_t>(
            (perspective_index + offset) % num_players)];
    };

    const double production =
        production_value(production_for(game, perspective, true), true);
    double enemy_production = 0.0;
    if (enemies == ValueEnemies::next) {
        enemy_production =
            production_value(production_for(game, enemy_at(1), true), false);
    } else {
        // Same summation order as the Python reference, then the mean.
        for (int offset = 1; offset < num_players; ++offset) {
            enemy_production +=
                production_value(production_for(game, enemy_at(offset), true), false);
        }
        enemy_production /= static_cast<double>(std::max(num_players - 1, 1));
    }
    const Reach reach = reachable_production(game, perspective);
    const double reachable_zero = reach.zero;
    const double reachable_one = reach.one;

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
    const int buildable_nodes = reach.buildable_nodes;
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

Action value_action(
    const Game& game,
    const ValueWeights& weights,
    ValueEnemies enemies) {
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
        const double value = value_score(candidate, action.color, weights, enemies);
        if (value > best_value) {
            best_value = value;
            best_action = action;
        }
    }
    return best_action;
}

}  // namespace cppanatron
