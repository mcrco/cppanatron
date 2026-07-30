#include "cppanatron/map.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <tuple>

namespace cppanatron {
namespace {

using TopologyEntry =
    std::tuple<Coordinate, TileKind, std::optional<Direction>>;

constexpr std::array<int, 18> kBaseNumbersInSpiralOrder{
    5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11,
};

constexpr std::size_t index(NodeRef ref) noexcept {
    return static_cast<std::size_t>(ref);
}

constexpr std::size_t index(EdgeRef ref) noexcept {
    return static_cast<std::size_t>(ref);
}

std::pair<NodeRef, NodeRef> edge_nodes(EdgeRef edge_ref) {
    switch (edge_ref) {
        case EdgeRef::east:
            return {NodeRef::northeast, NodeRef::southeast};
        case EdgeRef::southeast:
            return {NodeRef::southeast, NodeRef::south};
        case EdgeRef::southwest:
            return {NodeRef::south, NodeRef::southwest};
        case EdgeRef::west:
            return {NodeRef::southwest, NodeRef::northwest};
        case EdgeRef::northwest:
            return {NodeRef::northwest, NodeRef::north};
        case EdgeRef::northeast:
            return {NodeRef::north, NodeRef::northeast};
    }
    throw std::logic_error("unknown edge reference");
}

struct NodeEdgeResult {
    std::array<int, 6> nodes{};
    std::array<Edge, 6> edges{};
    int next_node_id{};
};

NodeEdgeResult get_nodes_and_edges(
    const std::vector<Tile>& tiles,
    const std::map<Coordinate, int>& coordinate_to_tile,
    Coordinate coordinate,
    int next_node_id) {
    std::array<int, 6> nodes{};
    nodes.fill(-1);
    std::array<Edge, 6> edges{};
    edges.fill(Edge{-1, -1});

    for (Direction direction : kDirections) {
        const auto neighbor_it =
            coordinate_to_tile.find(coordinate + direction_vector(direction));
        if (neighbor_it == coordinate_to_tile.end()) {
            continue;
        }
        const Tile& neighbor = tiles.at(static_cast<std::size_t>(neighbor_it->second));
        switch (direction) {
            case Direction::east:
                nodes[index(NodeRef::northeast)] = neighbor.nodes[index(NodeRef::northwest)];
                nodes[index(NodeRef::southeast)] = neighbor.nodes[index(NodeRef::southwest)];
                edges[index(EdgeRef::east)] = neighbor.edges[index(EdgeRef::west)];
                break;
            case Direction::southeast:
                nodes[index(NodeRef::south)] = neighbor.nodes[index(NodeRef::northwest)];
                nodes[index(NodeRef::southeast)] = neighbor.nodes[index(NodeRef::north)];
                edges[index(EdgeRef::southeast)] = neighbor.edges[index(EdgeRef::northwest)];
                break;
            case Direction::southwest:
                nodes[index(NodeRef::south)] = neighbor.nodes[index(NodeRef::northeast)];
                nodes[index(NodeRef::southwest)] = neighbor.nodes[index(NodeRef::north)];
                edges[index(EdgeRef::southwest)] = neighbor.edges[index(EdgeRef::northeast)];
                break;
            case Direction::west:
                nodes[index(NodeRef::northwest)] = neighbor.nodes[index(NodeRef::northeast)];
                nodes[index(NodeRef::southwest)] = neighbor.nodes[index(NodeRef::southeast)];
                edges[index(EdgeRef::west)] = neighbor.edges[index(EdgeRef::east)];
                break;
            case Direction::northwest:
                nodes[index(NodeRef::north)] = neighbor.nodes[index(NodeRef::southeast)];
                nodes[index(NodeRef::northwest)] = neighbor.nodes[index(NodeRef::south)];
                edges[index(EdgeRef::northwest)] = neighbor.edges[index(EdgeRef::southeast)];
                break;
            case Direction::northeast:
                nodes[index(NodeRef::north)] = neighbor.nodes[index(NodeRef::southwest)];
                nodes[index(NodeRef::northeast)] = neighbor.nodes[index(NodeRef::south)];
                edges[index(EdgeRef::northeast)] = neighbor.edges[index(EdgeRef::southwest)];
                break;
        }
    }

    for (int& node : nodes) {
        if (node < 0) {
            node = next_node_id++;
        }
    }
    for (EdgeRef ref : {
             EdgeRef::east,
             EdgeRef::southeast,
             EdgeRef::southwest,
             EdgeRef::west,
             EdgeRef::northwest,
             EdgeRef::northeast,
         }) {
        Edge& edge = edges[index(ref)];
        if (edge.a < 0) {
            const auto [a_ref, b_ref] = edge_nodes(ref);
            edge = Edge{nodes[index(a_ref)], nodes[index(b_ref)]}.normalized();
        }
    }
    return {nodes, edges, next_node_id};
}

std::vector<Coordinate> land_coordinates(bool mini) {
    std::vector<Coordinate> result{
        {0, 0, 0},
        {1, -1, 0},
        {0, -1, 1},
        {-1, 0, 1},
        {-1, 1, 0},
        {0, 1, -1},
        {1, 0, -1},
    };
    if (!mini) {
        result.insert(
            result.end(),
            {
                {2, -2, 0},
                {1, -2, 1},
                {0, -2, 2},
                {-1, -1, 2},
                {-2, 0, 2},
                {-2, 1, 1},
                {-2, 2, 0},
                {-1, 2, -1},
                {0, 2, -2},
                {1, 1, -2},
                {2, 0, -2},
                {2, -1, -1},
            });
    }
    return result;
}

std::vector<TopologyEntry> make_topology(bool mini) {
    std::vector<TopologyEntry> result;
    for (Coordinate coordinate : land_coordinates(mini)) {
        result.emplace_back(coordinate, TileKind::land, std::nullopt);
    }

    if (mini) {
        for (Coordinate coordinate : {
                 Coordinate{2, -2, 0},
                 {1, -2, 1},
                 {0, -2, 2},
                 {-1, -1, 2},
                 {-2, 0, 2},
                 {-2, 1, 1},
                 {-2, 2, 0},
                 {-1, 2, -1},
                 {0, 2, -2},
                 {1, 1, -2},
                 {2, 0, -2},
                 {2, -1, -1},
             }) {
            result.emplace_back(coordinate, TileKind::water, std::nullopt);
        }
        return result;
    }

    const std::array<TopologyEntry, 18> water_ring{{
        {{3, -3, 0}, TileKind::port, Direction::west},
        {{2, -3, 1}, TileKind::water, std::nullopt},
        {{1, -3, 2}, TileKind::port, Direction::northwest},
        {{0, -3, 3}, TileKind::water, std::nullopt},
        {{-1, -2, 3}, TileKind::port, Direction::northwest},
        {{-2, -1, 3}, TileKind::water, std::nullopt},
        {{-3, 0, 3}, TileKind::port, Direction::northeast},
        {{-3, 1, 2}, TileKind::water, std::nullopt},
        {{-3, 2, 1}, TileKind::port, Direction::east},
        {{-3, 3, 0}, TileKind::water, std::nullopt},
        {{-2, 3, -1}, TileKind::port, Direction::east},
        {{-1, 3, -2}, TileKind::water, std::nullopt},
        {{0, 3, -3}, TileKind::port, Direction::southeast},
        {{1, 2, -3}, TileKind::water, std::nullopt},
        {{2, 1, -3}, TileKind::port, Direction::southwest},
        {{3, 0, -3}, TileKind::water, std::nullopt},
        {{3, -1, -2}, TileKind::port, Direction::southwest},
        {{3, -2, -1}, TileKind::water, std::nullopt},
    }};
    result.insert(result.end(), water_ring.begin(), water_ring.end());
    return result;
}

template <typename T>
void shuffle(std::vector<T>& values, std::mt19937_64& random) {
    std::shuffle(values.begin(), values.end(), random);
}

std::vector<Coordinate> official_spiral_coordinates(bool mini) {
    if (mini) {
        return {
            {1, -1, 0},
            {1, 0, -1},
            {0, 1, -1},
            {-1, 1, 0},
            {-1, 0, 1},
            {0, -1, 1},
            {0, 0, 0},
        };
    }
    return {
        {2, -2, 0},
        {2, -1, -1},
        {2, 0, -2},
        {1, 1, -2},
        {0, 2, -2},
        {-1, 2, -1},
        {-2, 2, 0},
        {-2, 1, 1},
        {-2, 0, 2},
        {-1, -1, 2},
        {0, -2, 2},
        {1, -2, 1},
        {1, -1, 0},
        {1, 0, -1},
        {0, 1, -1},
        {-1, 1, 0},
        {-1, 0, 1},
        {0, -1, 1},
        {0, 0, 0},
    };
}

}  // namespace

Coordinate direction_vector(Direction direction) noexcept {
    switch (direction) {
        case Direction::east:
            return {1, -1, 0};
        case Direction::southeast:
            return {0, -1, 1};
        case Direction::southwest:
            return {-1, 0, 1};
        case Direction::west:
            return {-1, 1, 0};
        case Direction::northwest:
            return {0, 1, -1};
        case Direction::northeast:
            return {1, 0, -1};
    }
    return {};
}

const MapTemplate& mini_map_template() {
    static const MapTemplate value{
        .numbers = {3, 4, 5, 6, 8, 9, 10},
        .port_resources = {},
        .tile_resources = {
            Resource::wood,
            std::nullopt,
            Resource::brick,
            Resource::sheep,
            Resource::wheat,
            Resource::wheat,
            Resource::ore,
        },
        .topology = make_topology(true),
    };
    return value;
}

const MapTemplate& base_map_template() {
    static const MapTemplate value{
        .numbers = {2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12},
        .port_resources = {
            Resource::wood,
            Resource::brick,
            Resource::sheep,
            Resource::wheat,
            Resource::ore,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
        },
        .tile_resources = {
            Resource::wood,
            Resource::wood,
            Resource::wood,
            Resource::wood,
            Resource::brick,
            Resource::brick,
            Resource::brick,
            Resource::sheep,
            Resource::sheep,
            Resource::sheep,
            Resource::sheep,
            Resource::wheat,
            Resource::wheat,
            Resource::wheat,
            Resource::wheat,
            Resource::ore,
            Resource::ore,
            Resource::ore,
            std::nullopt,
        },
        .topology = make_topology(false),
    };
    return value;
}

CatanMap CatanMap::build(
    MapType map_type,
    std::uint64_t seed,
    NumberPlacement number_placement) {
    const MapTemplate& map_template =
        map_type == MapType::mini ? mini_map_template() : base_map_template();
    CatanMap result = from_template(map_template, seed, number_placement);
    if (map_type != MapType::tournament) {
        return result;
    }

    constexpr std::array<std::optional<Resource>, 19> kTournamentResources{
        std::nullopt,
        Resource::ore,
        Resource::wood,
        Resource::ore,
        Resource::brick,
        Resource::ore,
        Resource::wheat,
        Resource::wheat,
        Resource::sheep,
        Resource::brick,
        Resource::sheep,
        Resource::brick,
        Resource::wheat,
        Resource::wood,
        Resource::wheat,
        Resource::wood,
        Resource::sheep,
        Resource::sheep,
        Resource::wood,
    };
    constexpr std::array<std::optional<int>, 19> kTournamentNumbers{
        std::nullopt, 6, 3, 11, 9, 4, 5, 9, 12, 11, 4, 8, 10, 5, 2, 6, 3, 8, 10,
    };
    constexpr std::array<std::optional<Resource>, 9> kTournamentPorts{
        std::nullopt,
        Resource::brick,
        Resource::wood,
        std::nullopt,
        Resource::wheat,
        Resource::ore,
        std::nullopt,
        Resource::sheep,
        std::nullopt,
    };
    for (Tile& tile : result.tiles_) {
        if (tile.kind == TileKind::land) {
            const auto tile_id = static_cast<std::size_t>(tile.id);
            tile.resource = kTournamentResources.at(tile_id);
            tile.number = kTournamentNumbers.at(tile_id);
        } else if (tile.kind == TileKind::port) {
            tile.resource = kTournamentPorts.at(static_cast<std::size_t>(tile.id));
        }
    }
    return result;
}

CatanMap CatanMap::from_template(
    const MapTemplate& map_template,
    std::uint64_t seed,
    NumberPlacement number_placement) {
    CatanMap result;
    std::mt19937_64 random(seed);
    auto resources = map_template.tile_resources;
    auto numbers = map_template.numbers;
    auto ports = map_template.port_resources;
    shuffle(resources, random);
    shuffle(numbers, random);
    shuffle(ports, random);

    int next_node_id = 0;
    int next_land_id = 0;
    int next_port_id = 0;
    for (const auto& [coordinate, kind, port_direction] : map_template.topology) {
        NodeEdgeResult node_edges = get_nodes_and_edges(
            result.tiles_, result.coordinate_to_tile_, coordinate, next_node_id);
        next_node_id = node_edges.next_node_id;

        Tile tile;
        tile.kind = kind;
        tile.nodes = node_edges.nodes;
        tile.edges = node_edges.edges;
        tile.port_direction = port_direction;
        if (kind == TileKind::land) {
            tile.id = next_land_id++;
            tile.resource = resources.back();
            resources.pop_back();
            if (tile.resource.has_value()) {
                tile.number = numbers.back();
                numbers.pop_back();
            }
        } else if (kind == TileKind::port) {
            tile.id = next_port_id++;
            tile.resource = ports.back();
            ports.pop_back();
        }

        const int tile_index = static_cast<int>(result.tiles_.size());
        result.coordinate_to_tile_.emplace(coordinate, tile_index);
        result.tiles_.push_back(tile);
        if (kind == TileKind::land) {
            result.land_tile_indices_.push_back(tile_index);
            result.land_nodes_.insert(tile.nodes.begin(), tile.nodes.end());
        }
    }

    if (number_placement == NumberPlacement::official_spiral) {
        const bool mini = map_template.topology.size() == mini_map_template().topology.size();
        std::size_t number_index = 0;
        for (Coordinate coordinate : official_spiral_coordinates(mini)) {
            Tile& tile = result.tiles_.at(
                static_cast<std::size_t>(result.coordinate_to_tile_.at(coordinate)));
            if (!tile.resource.has_value()) {
                tile.number = std::nullopt;
                continue;
            }
            if (number_index >= kBaseNumbersInSpiralOrder.size()) {
                throw std::logic_error("official spiral has too many numbered tiles");
            }
            tile.number = kBaseNumbersInSpiralOrder[number_index++];
        }
    }
    std::set<Edge> unique_edges;
    for (int tile_index : result.land_tile_indices_) {
        const Tile& tile = result.tiles_[static_cast<std::size_t>(tile_index)];
        unique_edges.insert(tile.edges.begin(), tile.edges.end());
    }
    result.land_edges_.assign(unique_edges.begin(), unique_edges.end());
    return result;
}

const Tile& CatanMap::tile_at(Coordinate coordinate) const {
    return tiles_.at(static_cast<std::size_t>(coordinate_to_tile_.at(coordinate)));
}

}  // namespace cppanatron
