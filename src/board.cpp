#include "cppanatron/board.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>

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

Board::Board(CatanMap map) : map_(std::move(map)), board_buildable_ids_(map_.land_nodes()) {
    bool found_desert = false;
    for (const auto& [coordinate, tile_index] : map_.coordinate_to_tile()) {
        const Tile& tile = map_.tiles().at(static_cast<std::size_t>(tile_index));
        if (tile.kind == TileKind::land && !tile.resource.has_value()) {
            robber_coordinate_ = coordinate;
            found_desert = true;
            break;
        }
    }
    if (!found_desert) {
        throw std::invalid_argument("map must contain a desert tile");
    }
}

void Board::move_robber(Coordinate coordinate) {
    const Tile& tile = map_.tile_at(coordinate);
    if (tile.kind != TileKind::land) {
        throw std::invalid_argument("robber must move to a land tile");
    }
    if (coordinate == robber_coordinate_) {
        throw std::invalid_argument("robber must move to a different tile");
    }
    robber_coordinate_ = coordinate;
}

std::vector<int> Board::neighbors(int node_id) const {
    std::vector<int> result;
    for (Edge edge : map_.land_edges()) {
        if (edge.a == node_id) {
            result.push_back(edge.b);
        } else if (edge.b == node_id) {
            result.push_back(edge.a);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
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
    for (Edge edge : map_.land_edges()) {
        if (roads_.contains(edge)) {
            continue;
        }
        if (player_network_contains(color, edge.a) ||
            player_network_contains(color, edge.b)) {
            result.push_back(edge);
        }
    }
    return result;
}

std::set<std::optional<Resource>> Board::player_port_resources(Color color) const {
    std::set<std::optional<Resource>> result;
    for (const Tile& tile : map_.tiles()) {
        if (tile.kind != TileKind::port || !tile.port_direction.has_value()) {
            continue;
        }
        std::pair<int, int> port_nodes;
        switch (*tile.port_direction) {
            case Direction::west:
                port_nodes = {tile.nodes[5], tile.nodes[4]};
                break;
            case Direction::northwest:
                port_nodes = {tile.nodes[0], tile.nodes[5]};
                break;
            case Direction::northeast:
                port_nodes = {tile.nodes[1], tile.nodes[0]};
                break;
            case Direction::east:
                port_nodes = {tile.nodes[2], tile.nodes[1]};
                break;
            case Direction::southeast:
                port_nodes = {tile.nodes[3], tile.nodes[2]};
                break;
            case Direction::southwest:
                port_nodes = {tile.nodes[4], tile.nodes[3]};
                break;
        }
        if (is_friendly_node(port_nodes.first, color) ||
            is_friendly_node(port_nodes.second, color)) {
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
        for (const auto& [road_color, edges] : adjacent_roads) {
            if (edges.size() != 2) {
                continue;
            }
            const int a = edges[0].a == node_id ? edges[0].b : edges[0].a;
            const int c = edges[1].a == node_id ? edges[1].b : edges[1].a;
            const auto component_index =
                connected_component_index(node_id, road_color);
            if (!component_index.has_value()) {
                throw std::logic_error(
                    "road component is missing settlement node");
            }
            auto& components = connected_components_[road_color];
            components.erase(
                components.begin() +
                static_cast<std::ptrdiff_t>(*component_index));
            components.push_back(dfs_walk(a, road_color));
            components.push_back(dfs_walk(c, road_color));

            int best = 0;
            for (const auto& component : components) {
                best = std::max(
                    best, longest_acyclic_path(component, road_color));
            }
            road_lengths_[road_color] = best;
            if (!road_lengths_.empty()) {
                const auto winner = std::max_element(
                    road_lengths_.begin(),
                    road_lengths_.end(),
                    [](const auto& lhs, const auto& rhs) {
                        return lhs.second < rhs.second;
                    });
                road_color_ = winner->first;
                road_length_ = winner->second;
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

std::set<int> Board::dfs_walk(int node_id, Color color) const {
    std::vector<int> agenda{node_id};
    std::set<int> visited;
    while (!agenda.empty()) {
        const int current = agenda.back();
        agenda.pop_back();
        if (!visited.insert(current).second || is_enemy_node(current, color)) {
            continue;
        }
        for (int neighbor : neighbors(current)) {
            if (!visited.contains(neighbor) &&
                is_friendly_road({current, neighbor}, color)) {
                agenda.push_back(neighbor);
            }
        }
    }
    return visited;
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

}  // namespace cppanatron
