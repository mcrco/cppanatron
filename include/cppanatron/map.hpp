#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <set>
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

struct MapTemplate {
    std::vector<int> numbers;
    std::vector<std::optional<Resource>> port_resources;
    std::vector<std::optional<Resource>> tile_resources;
    std::vector<std::tuple<Coordinate, TileKind, std::optional<Direction>>> topology;
};

class CatanMap {
public:
    static CatanMap build(
        MapType map_type,
        std::uint64_t seed = 0,
        NumberPlacement number_placement = NumberPlacement::official_spiral);

    static CatanMap from_template(
        const MapTemplate& map_template,
        std::uint64_t seed,
        NumberPlacement number_placement);

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

private:
    std::vector<Tile> tiles_;
    std::vector<int> land_tile_indices_;
    std::set<int> land_nodes_;
    std::vector<Edge> land_edges_;
    std::map<Coordinate, int> coordinate_to_tile_;
};

[[nodiscard]] const MapTemplate& base_map_template();
[[nodiscard]] const MapTemplate& mini_map_template();
[[nodiscard]] Coordinate direction_vector(Direction direction) noexcept;

}  // namespace cppanatron
