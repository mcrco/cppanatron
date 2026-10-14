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
     * holds Longest Road until set_longest_road_holder. `road_order` is the
     * order the colors built their first road (seating order in a game played
     * from the start), which breaks ties when a cut road hands the award on.
     */
    void restore(
        const std::map<int, BuildingState>& buildings,
        const std::map<Edge, Color>& roads,
        Coordinate robber,
        const std::vector<Color>& road_order = {});
    /**
     * Scenario setup: who holds Longest Road after restore, and the length
     * the award was won with (by default the holder's cached length).
     */
    void set_longest_road_holder(
        std::optional<Color> color, std::optional<int> length = std::nullopt);
    /**
     * Scenario setup: a color's cached road length (Catanatron's
     * board.road_lengths entry), which play only ever raises until a cut. The
     * color must have roads.
     */
    void set_road_length(Color color, int length);
    /**
     * Scenario setup: a color's road networks as play keeps them (Catanatron's
     * board.connected_components), replacing the ones restore rebuilt, and
     * its cached road length recomputed from them. They differ from rebuilt
     * ones only at a road built into an opponent's existing building: play
     * leaves that building out of the network, so no road walk starts there.
     * Every node must be on land, and every building of the color in one.
     */
    void set_road_components(Color color, std::vector<std::set<int>> components);
    /** A color's road networks (node sets), in order. */
    [[nodiscard]] const std::vector<std::set<int>>& road_components(Color color) const;
    /** Cached road lengths in the order colors built their first road. */
    [[nodiscard]] std::vector<std::pair<Color, int>> road_lengths() const;
    /** The length Longest Road was last won with (0 without a holder). */
    [[nodiscard]] int longest_road_holder_length() const noexcept { return road_length_; }

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
    /**
     * Colors in the order they built their first road: Catanatron's
     * road_lengths dict order, which decides ties after a cut.
     */
    std::vector<Color> road_order_;
    std::optional<Color> road_color_;
    int road_length_{};
    std::set<int> board_buildable_ids_;
    Coordinate robber_coordinate_{};
};

}  // namespace cppanatron
