#pragma once

#include <array>
#include <map>
#include <memory>
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
    /** A board on a shared, immutable map. */
    explicit Board(std::shared_ptr<const CatanMap> map);
    /** A board owning its own copy of `map`. */
    explicit Board(CatanMap map);

    [[nodiscard]] const CatanMap& map() const noexcept { return *map_; }
    [[nodiscard]] const std::shared_ptr<const CatanMap>& shared_map() const noexcept {
        return map_;
    }
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
    [[nodiscard]] std::set<std::optional<Resource>> player_port_resources(
        Color color) const;

    void build_settlement(Color color, int node_id, bool initial_build_phase = false);
    void build_road(Color color, Edge edge);
    void build_city(Color color, int node_id);

    /**
     * Scenario setup: replace every building and road and move the robber.
     *
     * Validates the distance rule and that every road connects to a building
     * of its color (through any nodes, since a later settlement may have cut
     * the road), then rebuilds road networks and lengths from scratch. Nobody
     * holds Longest Road until set_longest_road_holder.
     */
    void restore(
        const std::map<int, BuildingState>& buildings,
        const std::map<Edge, Color>& roads,
        Coordinate robber);
    /** Scenario setup: who holds Longest Road after restore. */
    void set_longest_road_holder(std::optional<Color> color);

    [[nodiscard]] std::optional<Color> node_color(int node_id) const;
    [[nodiscard]] std::optional<Color> edge_color(Edge edge) const;
    [[nodiscard]] bool is_enemy_node(int node_id, Color color) const;
    [[nodiscard]] bool is_enemy_road(Edge edge, Color color) const;
    [[nodiscard]] bool is_friendly_node(int node_id, Color color) const;
    [[nodiscard]] bool is_friendly_road(Edge edge, Color color) const;
    [[nodiscard]] int longest_road(Color color) const;
    [[nodiscard]] std::optional<Color> longest_road_color() const noexcept {
        return road_color_;
    }

private:
    [[nodiscard]] const std::vector<int>& neighbors(int node_id) const {
        return map_->neighbors(node_id);
    }
    [[nodiscard]] bool player_network_contains(Color color, int node_id) const;
    [[nodiscard]] std::optional<std::size_t> connected_component_index(
        int node_id,
        Color color) const;
    void rebuild_connected_components(Color color);
    [[nodiscard]] int longest_acyclic_path(
        const std::set<int>& component,
        Color color) const;

    std::shared_ptr<const CatanMap> map_;
    std::map<int, BuildingState> buildings_;
    std::map<Edge, Color> roads_;
    std::map<Color, std::vector<std::set<int>>> connected_components_;
    std::map<Color, int> road_lengths_;
    std::optional<Color> road_color_;
    int road_length_{};
    std::set<int> board_buildable_ids_;
    Coordinate robber_coordinate_{};
};

}  // namespace cppanatron
