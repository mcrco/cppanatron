#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "cppanatron/types.hpp"

namespace cppanatron {

enum class Direction : std::uint8_t {
    east,
    southeast,
    southwest,
    west,
    northwest,
    northeast,
};

enum class NodeRef : std::uint8_t {
    north,
    northeast,
    southeast,
    south,
    southwest,
    northwest,
};

enum class EdgeRef : std::uint8_t {
    east,
    southeast,
    southwest,
    west,
    northwest,
    northeast,
};

enum class TileKind : std::uint8_t { land, water, port };

inline constexpr std::array<Direction, 6> kDirections{
    Direction::east,
    Direction::southeast,
    Direction::southwest,
    Direction::west,
    Direction::northwest,
    Direction::northeast,
};

struct Tile {
    TileKind kind{TileKind::water};
    int id{-1};
    std::optional<Resource> resource;
    std::optional<int> number;
    std::optional<Direction> port_direction;
    std::array<int, 6> nodes{};
    std::array<Edge, 6> edges{};
};

/**
 * One tile of an explicit map. Which fields apply depends on the kind:
 * land tiles have a resource (none for the desert) and a number (none
 * exactly for the desert); ports face land in `direction` and trade
 * `port_resource` 2:1 (none for 3:1); water has neither.
 */
struct TileSpec {
    Coordinate coordinate{};
    TileKind kind{TileKind::water};
    std::optional<Resource> resource;
    std::optional<int> number;
    std::optional<Direction> direction;
    std::optional<Resource> port_resource;

    bool operator==(const TileSpec&) const = default;
};

/**
 * An explicit map: every tile, in order. The order is part of the map, as in
 * Catanatron: node ids, edge ids, land tile ids and port ids are assigned
 * while walking the tiles in this order (see CatanMap::from_spec).
 */
struct MapSpec {
    std::vector<TileSpec> tiles;

    bool operator==(const MapSpec&) const = default;
};

struct MapTemplate {
    std::vector<int> numbers;
    std::vector<std::optional<Resource>> port_resources;
    std::vector<std::optional<Resource>> tile_resources;
    std::vector<std::tuple<Coordinate, TileKind, std::optional<Direction>>> topology;
};

/**
 * An immutable board layout: tiles, node and edge ids, and the graph caches
 * the rules and encoders query. Games share one map through
 * std::shared_ptr<const CatanMap>; enable_shared_from_this lets a reference
 * handed out by a board (or to Python) recover that shared ownership.
 *
 * Roads run along the edges of land tiles, and two nodes are neighbors when
 * a land edge joins them. That is Catanatron's graph restricted to land on
 * every template map; irregular maps (lakes, notches) must use the same rule
 * in the reference, see cppanatron.catanatron.map_graph.
 */
class CatanMap : public std::enable_shared_from_this<CatanMap> {
public:
    static CatanMap build(
        MapType map_type,
        std::uint64_t seed = 0,
        NumberPlacement number_placement = NumberPlacement::official_spiral);

    static CatanMap from_template(
        const MapTemplate& map_template,
        std::uint64_t seed,
        NumberPlacement number_placement);

    /**
     * Build a map from explicit tiles, numbering nodes and edges like
     * Catanatron's initialize_tiles + CatanMap.from_tiles walk, so the same
     * spec yields the same ids in both engines. Throws std::invalid_argument
     * for specs no game can use: duplicate or invalid coordinates, fields
     * that do not fit the tile kind, numbers outside 2-12 or 7, no land or
     * no desert (the robber starts on the first desert).
     */
    static CatanMap from_spec(const MapSpec& spec);

    /** The tiles of a template map as an explicit spec (same draws as build). */
    static MapSpec build_spec(
        MapType map_type,
        std::uint64_t seed = 0,
        NumberPlacement number_placement = NumberPlacement::official_spiral);

    /** The explicit tiles this map was built from; from_spec(spec()) rebuilds it. */
    [[nodiscard]] const MapSpec& spec() const noexcept { return spec_; }

    [[nodiscard]] const std::vector<Tile>& tiles() const noexcept { return tiles_; }
    [[nodiscard]] const std::vector<int>& land_tile_indices() const noexcept {
        return land_tile_indices_;
    }
    [[nodiscard]] const std::set<int>& land_nodes() const noexcept { return land_nodes_; }
    [[nodiscard]] const std::vector<Edge>& land_edges() const noexcept { return land_edges_; }
    [[nodiscard]] const std::map<Coordinate, int>& coordinate_to_tile() const noexcept {
        return coordinate_to_tile_;
    }
    [[nodiscard]] const Tile& tile_at(Coordinate coordinate) const;
    [[nodiscard]] int num_nodes() const noexcept {
        return static_cast<int>(land_nodes_.size());
    }

    // Compact slots: land nodes by ascending id, land edges in land_edges()
    // order (ascending), land tiles in land_tile_indices() order. Encoders and
    // action codecs index nodes, edges and tiles by these slots.

    /** Land node ids in ascending order; a node's position is its slot. */
    [[nodiscard]] const std::vector<int>& land_node_list() const noexcept {
        return land_node_list_;
    }
    /** Slot of a land node, or -1 for any other id. */
    [[nodiscard]] int node_slot(int node_id) const noexcept;
    /** Slot of a land edge (either orientation), or -1. */
    [[nodiscard]] int edge_slot(Edge edge) const noexcept;
    /** Slot of a land tile by tiles() index, or -1 for water and ports. */
    [[nodiscard]] int land_tile_slot(int tile_index) const noexcept;
    /** One past the largest node id on the map, land or not. */
    [[nodiscard]] int num_node_ids() const noexcept {
        return static_cast<int>(neighbors_.size());
    }
    /** Land neighbors of a node in ascending order (empty off land). */
    [[nodiscard]] const std::vector<int>& neighbors(int node_id) const;
    /** tiles() indices of the land tiles touching a node. */
    [[nodiscard]] const std::vector<int>& adjacent_land_tiles(int node_id) const;
    /**
     * Ports at a node as a bit set: bit r for a 2:1 port of Resource r and
     * bit 5 for a 3:1 port.
     */
    [[nodiscard]] std::uint8_t node_ports(int node_id) const;

private:
    void index_graph();

    MapSpec spec_;
    std::vector<Tile> tiles_;
    std::vector<int> land_tile_indices_;
    std::set<int> land_nodes_;
    std::vector<Edge> land_edges_;
    std::map<Coordinate, int> coordinate_to_tile_;
    std::vector<int> land_node_list_;
    std::vector<int> node_slots_;
    std::vector<int> land_tile_slots_;
    std::vector<std::vector<int>> neighbors_;
    std::vector<std::vector<int>> adjacent_land_tiles_;
    std::vector<std::uint8_t> node_ports_;
};

/** The port bit for "3:1" in CatanMap::node_ports. */
inline constexpr std::uint8_t kGenericPortBit = 1U << 5U;

/** The two nodes of a port tile that trade at it, by its direction. */
[[nodiscard]] std::pair<int, int> port_nodes(const Tile& tile);

[[nodiscard]] const MapTemplate& base_map_template();
[[nodiscard]] const MapTemplate& mini_map_template();
[[nodiscard]] Coordinate direction_vector(Direction direction) noexcept;

}  // namespace cppanatron
