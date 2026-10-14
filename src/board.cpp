#include "cppanatron/board.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>

namespace cppanatron {
namespace {

bool contains(const std::vector<int>& values, int value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool contains(const std::vector<Edge>& values, Edge value) {
    value = value.normalized();
    return std::find(values.begin(), values.end(), value) != values.end();
}

}  // namespace

Board::Board(std::shared_ptr<const CatanMap> map) : map_(std::move(map)) {
    if (map_ == nullptr) {
        throw std::invalid_argument("board needs a map");
    }
    board_buildable_ids_ = map_->land_nodes();
    // Like Catanatron, the robber starts on the first desert in tile order.
    for (int tile_index : map_->land_tile_indices()) {
        const auto index = static_cast<std::size_t>(tile_index);
        if (!map_->tiles().at(index).resource.has_value()) {
            robber_coordinate_ = map_->spec().tiles.at(index).coordinate;
            return;
        }
    }
    throw std::invalid_argument("map must contain a desert tile");
}

Board::Board(CatanMap map) : Board(std::make_shared<const CatanMap>(std::move(map))) {}

void Board::move_robber(Coordinate coordinate) {
    const Tile& tile = map_->tile_at(coordinate);
    if (tile.kind != TileKind::land) {
        throw std::invalid_argument("robber must move to a land tile");
    }
    if (coordinate == robber_coordinate_) {
        throw std::invalid_argument("robber must move to a different tile");
    }
    robber_coordinate_ = coordinate;
}

std::optional<Color> Board::node_color(int node_id) const {
    const auto it = buildings_.find(node_id);
    return it == buildings_.end() ? std::nullopt : std::optional(it->second.color);
}

std::optional<Color> Board::edge_color(Edge edge) const {
    const auto it = roads_.find(edge.normalized());
    return it == roads_.end() ? std::nullopt : std::optional(it->second);
}

bool Board::is_enemy_node(int node_id, Color color) const {
    const auto owner = node_color(node_id);
    return owner.has_value() && owner != color;
}

bool Board::is_enemy_road(Edge edge, Color color) const {
    const auto owner = edge_color(edge);
    return owner.has_value() && owner != color;
}

bool Board::is_friendly_node(int node_id, Color color) const {
    return node_color(node_id) == color;
}

bool Board::is_friendly_road(Edge edge, Color color) const {
    return edge_color(edge) == color;
}

bool Board::player_network_contains(Color color, int node_id) const {
    const auto components = connected_components_.find(color);
    if (components == connected_components_.end()) {
        return false;
    }
    return std::any_of(
        components->second.begin(),
        components->second.end(),
        [node_id](const std::set<int>& component) {
            return component.contains(node_id);
        });
}

std::vector<int> Board::buildable_node_ids(Color color, bool initial_build_phase) const {
    std::vector<int> result;
    for (int node_id : board_buildable_ids_) {
        if (initial_build_phase || player_network_contains(color, node_id)) {
            result.push_back(node_id);
        }
    }
    return result;
}

std::vector<Edge> Board::buildable_edges(Color color) const {
    std::vector<Edge> result;
    for (Edge edge : map_->land_edges()) {
        if (roads_.contains(edge)) {
            continue;
        }
        // Roads extend from the player's network but never through an
        // opponent's building.
        const auto expandable = [&](int node_id) {
            return player_network_contains(color, node_id) &&
                   !is_enemy_node(node_id, color);
        };
        if (expandable(edge.a) || expandable(edge.b)) {
            result.push_back(edge);
        }
    }
    return result;
}

std::set<std::optional<Resource>> Board::player_port_resources(Color color) const {
    std::set<std::optional<Resource>> result;
    for (const Tile& tile : map_->tiles()) {
        if (tile.kind != TileKind::port || !tile.port_direction.has_value()) {
            continue;
        }
        const auto [first, second] = port_nodes(tile);
        if (is_friendly_node(first, color) || is_friendly_node(second, color)) {
            result.insert(tile.resource);
        }
    }
    return result;
}

void Board::build_settlement(Color color, int node_id, bool initial_build_phase) {
    const auto buildable = buildable_node_ids(color, initial_build_phase);
    if (!contains(buildable, node_id)) {
        throw std::invalid_argument("invalid settlement placement");
    }
    if (buildings_.contains(node_id)) {
        throw std::invalid_argument("building already exists at node");
    }
    buildings_.emplace(node_id, BuildingState{color, Building::settlement});
    if (initial_build_phase) {
        connected_components_[color].push_back({node_id});
    } else {
        std::map<Color, std::vector<Edge>> adjacent_roads;
        for (int neighbor : neighbors(node_id)) {
            const Edge edge = Edge{node_id, neighbor}.normalized();
            const auto owner = edge_color(edge);
            if (owner.has_value() && *owner != color) {
                adjacent_roads[*owner].push_back(edge);
            }
        }
        // An opponent's settlement can split a road network. Rebuild the cut
        // player's components from the authoritative roads and buildings rather
        // than editing the cache incrementally, which earlier cuts and cycles
        // can leave with overlapping or missing nodes.
        for (const auto& [road_color, edges] : adjacent_roads) {
            if (edges.size() < 2) {
                continue;
            }
            rebuild_connected_components(road_color);

            int best = 0;
            for (const auto& component : connected_components_[road_color]) {
                best = std::max(
                    best, longest_acyclic_path(component, road_color));
            }
            road_lengths_[road_color] = best;
            // Catanatron's max(road_lengths.items()): the first longest in the
            // order the players built their first road, even below five.
            road_color_.reset();
            road_length_ = 0;
            for (Color candidate : road_order_) {
                const int length = road_lengths_.at(candidate);
                if (!road_color_.has_value() || length > road_length_) {
                    road_color_ = candidate;
                    road_length_ = length;
                }
            }
        }
    }
    board_buildable_ids_.erase(node_id);
    for (int neighbor : neighbors(node_id)) {
        board_buildable_ids_.erase(neighbor);
    }
}

void Board::build_road(Color color, Edge edge) {
    edge = edge.normalized();
    if (!contains(buildable_edges(color), edge)) {
        throw std::invalid_argument("invalid road placement");
    }
    roads_.emplace(edge, color);

    const auto a_index = connected_component_index(edge.a, color);
    const auto b_index = connected_component_index(edge.b, color);
    auto& components = connected_components_[color];
    std::size_t chosen_index{};
    if (!a_index.has_value() && !is_enemy_node(edge.a, color)) {
        if (!b_index.has_value()) {
            throw std::logic_error("road is disconnected from player network");
        }
        chosen_index = *b_index;
        components[chosen_index].insert(edge.a);
    } else if (!b_index.has_value() && !is_enemy_node(edge.b, color)) {
        if (!a_index.has_value()) {
            throw std::logic_error("road is disconnected from player network");
        }
        chosen_index = *a_index;
        components[chosen_index].insert(edge.b);
    } else if (
        a_index.has_value() && b_index.has_value() && *a_index != *b_index) {
        chosen_index = *a_index;
        components[chosen_index].insert(
            components[*b_index].begin(), components[*b_index].end());
        components.erase(
            components.begin() + static_cast<std::ptrdiff_t>(*b_index));
        if (*b_index < chosen_index) {
            --chosen_index;
        }
    } else {
        const auto index = a_index.has_value() ? a_index : b_index;
        if (!index.has_value()) {
            throw std::logic_error("road is disconnected from player network");
        }
        chosen_index = *index;
    }

    const int candidate_length =
        longest_acyclic_path(components[chosen_index], color);
    if (!road_lengths_.contains(color)) {
        road_order_.push_back(color);
    }
    int& cached_length = road_lengths_[color];
    cached_length = std::max(cached_length, candidate_length);
    if (candidate_length >= 5 && candidate_length > road_length_) {
        road_color_ = color;
        road_length_ = candidate_length;
    }
}

void Board::build_city(Color color, int node_id) {
    const auto it = buildings_.find(node_id);
    if (it == buildings_.end() || it->second.color != color ||
        it->second.building != Building::settlement) {
        throw std::invalid_argument("city requires a friendly settlement");
    }
    it->second.building = Building::city;
}

int Board::longest_road(Color color) const {
    const auto length = road_lengths_.find(color);
    return length == road_lengths_.end() ? 0 : length->second;
}

std::optional<std::size_t> Board::connected_component_index(
    int node_id,
    Color color) const {
    const auto components = connected_components_.find(color);
    if (components == connected_components_.end()) {
        return std::nullopt;
    }
    for (std::size_t i = 0; i < components->second.size(); ++i) {
        if (components->second[i].contains(node_id)) {
            return i;
        }
    }
    return std::nullopt;
}

void Board::rebuild_connected_components(Color color) {
    std::set<Edge> unvisited;
    for (const auto& [edge, owner] : roads_) {
        if (owner == color) {
            unvisited.insert(edge);
        }
    }

    std::vector<std::set<int>> components;
    while (!unvisited.empty()) {
        std::vector<Edge> agenda{*unvisited.begin()};
        unvisited.erase(unvisited.begin());
        std::set<int> component{agenda.front().a, agenda.front().b};
        while (!agenda.empty()) {
            const Edge edge = agenda.back();
            agenda.pop_back();
            for (int node_id : {edge.a, edge.b}) {
                if (is_enemy_node(node_id, color)) {
                    continue;
                }
                for (int neighbor : neighbors(node_id)) {
                    const auto next = unvisited.find(Edge{node_id, neighbor}.normalized());
                    if (next == unvisited.end()) {
                        continue;
                    }
                    component.insert({next->a, next->b});
                    agenda.push_back(*next);
                    unvisited.erase(next);
                }
            }
        }
        components.push_back(std::move(component));
    }

    // Isolated friendly buildings still anchor future roads.
    for (const auto& [node_id, building] : buildings_) {
        const bool connected = std::any_of(
            components.begin(),
            components.end(),
            [node_id](const std::set<int>& component) {
                return component.contains(node_id);
            });
        if (building.color == color && !connected) {
            components.push_back({node_id});
        }
    }
    connected_components_[color] = std::move(components);
}

int Board::longest_acyclic_path(
    const std::set<int>& component,
    Color color) const {
    int best = 0;
    std::set<Edge> used;
    std::function<void(int, int)> visit = [&](int node, int length) {
        bool able_to_navigate = false;
        for (int neighbor : neighbors(node)) {
            const Edge edge = Edge{node, neighbor}.normalized();
            if (!is_friendly_road(edge, color) ||
                is_enemy_node(neighbor, color) || used.contains(edge)) {
                continue;
            }
            used.insert(edge);
            visit(neighbor, length + 1);
            used.erase(edge);
            able_to_navigate = true;
        }
        if (!able_to_navigate) {
            best = std::max(best, length);
        }
    };
    for (int node : component) {
        visit(node, 0);
    }
    return best;
}

void Board::restore(
    const std::map<int, BuildingState>& buildings,
    const std::map<Edge, Color>& roads,
    Coordinate robber,
    const std::vector<Color>& road_order) {
    if (map_->tile_at(robber).kind != TileKind::land) {
        throw std::invalid_argument("robber must be on a land tile");
    }
    const auto edge_name = [](Edge edge) {
        return "(" + std::to_string(edge.a) + ", " + std::to_string(edge.b) + ")";
    };
    const std::set<Edge> land_edges(map_->land_edges().begin(), map_->land_edges().end());
    for (const auto& [node_id, building] : buildings) {
        (void)building;
        if (!map_->land_nodes().contains(node_id)) {
            throw std::invalid_argument("node " + std::to_string(node_id) + " is not on land");
        }
        for (int neighbor : neighbors(node_id)) {
            if (buildings.contains(neighbor)) {
                throw std::invalid_argument(
                    "buildings at nodes " + std::to_string(node_id) + " and " +
                    std::to_string(neighbor) + " break the distance rule");
            }
        }
    }
    for (const auto& [edge, color] : roads) {
        (void)color;
        if (edge != edge.normalized() || !land_edges.contains(edge)) {
            throw std::invalid_argument("road " + edge_name(edge) + " is not a land edge");
        }
    }
    // Every road must reach a building of its color. Walk through any node:
    // an opponent's later settlement may have cut the network in two.
    std::set<Edge> unreached;
    for (const auto& [edge, color] : roads) {
        (void)color;
        unreached.insert(edge);
    }
    for (const auto& [node_id, building] : buildings) {
        std::vector<int> agenda{node_id};
        std::set<int> seen{node_id};
        while (!agenda.empty()) {
            const int node = agenda.back();
            agenda.pop_back();
            for (int neighbor : neighbors(node)) {
                const Edge edge = Edge{node, neighbor}.normalized();
                const auto road = roads.find(edge);
                if (road == roads.end() || road->second != building.color) {
                    continue;
                }
                unreached.erase(edge);
                if (seen.insert(neighbor).second) {
                    agenda.push_back(neighbor);
                }
            }
        }
    }
    if (!unreached.empty()) {
        throw std::invalid_argument(
            "road " + edge_name(*unreached.begin()) +
            " does not connect to a building of its color");
    }

    buildings_ = buildings;
    roads_ = roads;
    robber_coordinate_ = robber;
    board_buildable_ids_ = map_->land_nodes();
    for (const auto& [node_id, building] : buildings_) {
        (void)building;
        board_buildable_ids_.erase(node_id);
        for (int neighbor : neighbors(node_id)) {
            board_buildable_ids_.erase(neighbor);
        }
    }
    connected_components_.clear();
    road_lengths_.clear();
    std::set<Color> colors;
    for (const auto& [node_id, building] : buildings_) {
        (void)node_id;
        colors.insert(building.color);
    }
    for (Color color : colors) {
        rebuild_connected_components(color);
    }
    // As in play, only players who have built a road have a cached length,
    // kept in the order they built their first one: `road_order` first, then
    // any other color with roads.
    std::vector<Color> order = road_order;
    for (const auto& [edge, color] : roads_) {
        (void)edge;
        if (std::find(order.begin(), order.end(), color) == order.end()) {
            order.push_back(color);
        }
    }
    road_order_.clear();
    for (Color color : order) {
        const bool has_road = std::any_of(roads_.begin(), roads_.end(), [color](const auto& road) {
            return road.second == color;
        });
        if (!has_road) {
            continue;
        }
        int best = 0;
        for (const auto& component : connected_components_[color]) {
            best = std::max(best, longest_acyclic_path(component, color));
        }
        road_lengths_[color] = best;
        road_order_.push_back(color);
    }
    road_color_.reset();
    road_length_ = 0;
}

void Board::set_longest_road_holder(std::optional<Color> color) {
    road_color_ = color;
    road_length_ = color.has_value() ? longest_road(*color) : 0;
}

}  // namespace cppanatron
