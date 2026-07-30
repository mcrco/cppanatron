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
    if (is_friendly_node(node_id, color)) {
        return true;
    }
    return std::any_of(
        roads_.begin(),
        roads_.end(),
        [color, node_id](const auto& item) {
            return item.second == color &&
                   (item.first.a == node_id || item.first.b == node_id);
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

        const bool from_a =
            player_network_contains(color, edge.a) && !is_enemy_node(edge.a, color);
        const bool from_b =
            player_network_contains(color, edge.b) && !is_enemy_node(edge.b, color);
        if (from_a || from_b) {
            result.push_back(edge);
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
    int best = 0;
    std::set<Edge> used;
    std::function<void(int, int)> visit = [&](int node, int length) {
        best = std::max(best, length);
        if (length > 0 && is_enemy_node(node, color)) {
            return;
        }
        for (int neighbor : neighbors(node)) {
            const Edge edge = Edge{node, neighbor}.normalized();
            if (!is_friendly_road(edge, color) || used.contains(edge)) {
                continue;
            }
            used.insert(edge);
            visit(neighbor, length + 1);
            used.erase(edge);
        }
    };
    for (Edge edge : map_.land_edges()) {
        if (is_friendly_road(edge, color)) {
            visit(edge.a, 0);
            visit(edge.b, 0);
        }
    }
    return best;
}

}  // namespace cppanatron
