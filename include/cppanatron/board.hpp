#pragma once

#include <array>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "cppanatron/map.hpp"

namespace cppanatron {

struct BuildingState {
    Color color{};
    Building building{};

    auto operator<=>(const BuildingState&) const = default;
};

class Board {
public:
    explicit Board(CatanMap map);

    [[nodiscard]] const CatanMap& map() const noexcept { return map_; }
    [[nodiscard]] const std::map<int, BuildingState>& buildings() const noexcept {
        return buildings_;
    }
    [[nodiscard]] const std::map<Edge, Color>& roads() const noexcept { return roads_; }
    [[nodiscard]] Coordinate robber_coordinate() const noexcept { return robber_coordinate_; }
    void move_robber(Coordinate coordinate);

    [[nodiscard]] std::vector<int> buildable_node_ids(
        Color color,
        bool initial_build_phase = false) const;
    [[nodiscard]] std::vector<Edge> buildable_edges(Color color) const;

    void build_settlement(Color color, int node_id, bool initial_build_phase = false);
    void build_road(Color color, Edge edge);
    void build_city(Color color, int node_id);

    [[nodiscard]] std::optional<Color> node_color(int node_id) const;
    [[nodiscard]] std::optional<Color> edge_color(Edge edge) const;
    [[nodiscard]] bool is_enemy_node(int node_id, Color color) const;
    [[nodiscard]] bool is_enemy_road(Edge edge, Color color) const;
    [[nodiscard]] bool is_friendly_node(int node_id, Color color) const;
    [[nodiscard]] bool is_friendly_road(Edge edge, Color color) const;
    [[nodiscard]] int longest_road(Color color) const;

private:
    [[nodiscard]] std::vector<int> neighbors(int node_id) const;
    [[nodiscard]] bool player_network_contains(Color color, int node_id) const;

    CatanMap map_;
    std::map<int, BuildingState> buildings_;
    std::map<Edge, Color> roads_;
    std::set<int> board_buildable_ids_;
    Coordinate robber_coordinate_{};
};

}  // namespace cppanatron
