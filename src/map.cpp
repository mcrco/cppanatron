#include "cppanatron/map.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
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

/** Far beyond any playable map; keeps coordinate arithmetic overflow-free. */
constexpr std::int64_t kMaxCoordinate = 1 << 20;

std::string coordinate_name(Coordinate coordinate) {
    return "(" + std::to_string(coordinate.x) + ", " + std::to_string(coordinate.y) + ", " +
           std::to_string(coordinate.z) + ")";
}

void validate_spec(const MapSpec& spec) {
    std::set<Coordinate> seen;
    bool has_land = false;
    bool has_desert = false;
    for (const TileSpec& tile : spec.tiles) {
        const std::string where = "tile at " + coordinate_name(tile.coordinate);
        const auto [x, y, z] = tile.coordinate;
        // Bounded first, so neighbor coordinates (one step away) cannot overflow.
        if (std::max({std::abs(static_cast<std::int64_t>(x)), std::abs(static_cast<std::int64_t>(y)),
                      std::abs(static_cast<std::int64_t>(z))}) > kMaxCoordinate) {
            throw std::invalid_argument(
                where + ": cube coordinates must lie within " + std::to_string(kMaxCoordinate));
        }
        if (static_cast<std::int64_t>(x) + y + z != 0) {
            throw std::invalid_argument(where + ": cube coordinates must sum to zero");
        }
        if (!seen.insert(tile.coordinate).second) {
            throw std::invalid_argument(where + ": coordinate appears twice");
        }
        switch (tile.kind) {
            case TileKind::land:
                has_land = true;
                if (tile.direction.has_value() || tile.port_resource.has_value()) {
                    throw std::invalid_argument(where + ": land has no port direction or resource");
                }
                if (!tile.resource.has_value()) {
                    has_desert = true;
                    if (tile.number.has_value()) {
                        throw std::invalid_argument(where + ": the desert has no number");
                    }
                } else if (
                    !tile.number.has_value() || *tile.number < 2 || *tile.number > 12 ||
                    *tile.number == 7) {
                    throw std::invalid_argument(
                        where + ": a resource tile needs a number in 2-6 or 8-12");
                }
                break;
            case TileKind::port:
                if (!tile.direction.has_value()) {
                    throw std::invalid_argument(where + ": a port needs a direction");
                }
                if (tile.resource.has_value() || tile.number.has_value()) {
                    throw std::invalid_argument(
                        where + ": a port has a port_resource, not a resource or number");
                }
                break;
            case TileKind::water:
                if (tile.resource.has_value() || tile.number.has_value() ||
                    tile.direction.has_value() || tile.port_resource.has_value()) {
                    throw std::invalid_argument(where + ": water has no resource, number or port");
                }
                break;
        }
    }
    if (!has_land) {
        throw std::invalid_argument("a map needs at least one land tile");
    }
    if (!has_desert) {
        throw std::invalid_argument("a map needs a desert, where the robber starts");
    }
}

/**
 * Draw a template's terrain, numbers and ports like Catanatron's
 * initialize_tiles: shuffle each list, then pop from the back while walking
 * the topology, and finally overwrite numbers along the official spiral.
 */
MapSpec spec_from_template(
    const MapTemplate& map_template,
    std::uint64_t seed,
    NumberPlacement number_placement) {
    std::mt19937_64 random(seed);
    auto resources = map_template.tile_resources;
    auto numbers = map_template.numbers;
    auto ports = map_template.port_resources;
    shuffle(resources, random);
    shuffle(numbers, random);
    shuffle(ports, random);

    MapSpec result;
    std::map<Coordinate, std::size_t> positions;
    for (const auto& [coordinate, kind, port_direction] : map_template.topology) {
        TileSpec tile;
        tile.coordinate = coordinate;
        tile.kind = kind;
        if (kind == TileKind::land) {
            tile.resource = resources.back();
            resources.pop_back();
            if (tile.resource.has_value()) {
                tile.number = numbers.back();
                numbers.pop_back();
            }
        } else if (kind == TileKind::port) {
            tile.direction = port_direction;
            tile.port_resource = ports.back();
            ports.pop_back();
        }
        positions.emplace(coordinate, result.tiles.size());
        result.tiles.push_back(tile);
    }

    if (number_placement == NumberPlacement::official_spiral) {
        const bool mini = map_template.topology.size() == mini_map_template().topology.size();
        std::size_t number_index = 0;
        for (Coordinate coordinate : official_spiral_coordinates(mini)) {
            TileSpec& tile = result.tiles.at(positions.at(coordinate));
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
    return result;
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

MapSpec CatanMap::build_spec(
    MapType map_type,
    std::uint64_t seed,
    NumberPlacement number_placement) {
    const MapTemplate& map_template =
        map_type == MapType::mini ? mini_map_template() : base_map_template();
    MapSpec result = spec_from_template(map_template, seed, number_placement);
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
    std::size_t land_id = 0;
    std::size_t port_id = 0;
    for (TileSpec& tile : result.tiles) {
        if (tile.kind == TileKind::land) {
            tile.resource = kTournamentResources.at(land_id);
            tile.number = kTournamentNumbers.at(land_id);
            ++land_id;
        } else if (tile.kind == TileKind::port) {
            tile.port_resource = kTournamentPorts.at(port_id++);
        }
    }
    return result;
}

CatanMap CatanMap::build(
    MapType map_type,
    std::uint64_t seed,
    NumberPlacement number_placement) {
    return from_spec(build_spec(map_type, seed, number_placement));
}

CatanMap CatanMap::from_template(
    const MapTemplate& map_template,
    std::uint64_t seed,
    NumberPlacement number_placement) {
    return from_spec(spec_from_template(map_template, seed, number_placement));
}

CatanMap CatanMap::from_spec(const MapSpec& spec) {
    validate_spec(spec);
    CatanMap result;
    result.spec_ = spec;

    int next_node_id = 0;
    int next_land_id = 0;
    int next_port_id = 0;
    for (const TileSpec& tile_spec : spec.tiles) {
        const NodeEdgeResult node_edges = get_nodes_and_edges(
            result.tiles_, result.coordinate_to_tile_, tile_spec.coordinate, next_node_id);
        next_node_id = node_edges.next_node_id;

        Tile tile;
        tile.kind = tile_spec.kind;
        tile.nodes = node_edges.nodes;
        tile.edges = node_edges.edges;
        if (tile_spec.kind == TileKind::land) {
            tile.id = next_land_id++;
            tile.resource = tile_spec.resource;
            tile.number = tile_spec.number;
        } else if (tile_spec.kind == TileKind::port) {
            tile.id = next_port_id++;
            tile.resource = tile_spec.port_resource;
            tile.port_direction = tile_spec.direction;
        }

        const int tile_index = static_cast<int>(result.tiles_.size());
        result.coordinate_to_tile_.emplace(tile_spec.coordinate, tile_index);
        result.tiles_.push_back(tile);
        if (tile.kind == TileKind::land) {
            result.land_tile_indices_.push_back(tile_index);
            result.land_nodes_.insert(tile.nodes.begin(), tile.nodes.end());
        }
    }

    std::set<Edge> unique_edges;
    for (int tile_index : result.land_tile_indices_) {
        const Tile& tile = result.tiles_[static_cast<std::size_t>(tile_index)];
        unique_edges.insert(tile.edges.begin(), tile.edges.end());
    }
    result.land_edges_.assign(unique_edges.begin(), unique_edges.end());
    result.index_graph();
    return result;
}

void CatanMap::index_graph() {
    int num_node_ids = 0;
    for (const Tile& tile : tiles_) {
        for (int node : tile.nodes) {
            num_node_ids = std::max(num_node_ids, node + 1);
        }
    }
    const auto size = static_cast<std::size_t>(num_node_ids);
    land_node_list_.assign(land_nodes_.begin(), land_nodes_.end());
    node_slots_.assign(size, -1);
    for (std::size_t slot = 0; slot < land_node_list_.size(); ++slot) {
        node_slots_[static_cast<std::size_t>(land_node_list_[slot])] = static_cast<int>(slot);
    }
    land_tile_slots_.assign(tiles_.size(), -1);
    for (std::size_t slot = 0; slot < land_tile_indices_.size(); ++slot) {
        land_tile_slots_[static_cast<std::size_t>(land_tile_indices_[slot])] =
            static_cast<int>(slot);
    }

    neighbors_.assign(size, {});
    for (Edge edge : land_edges_) {
        neighbors_[static_cast<std::size_t>(edge.a)].push_back(edge.b);
        neighbors_[static_cast<std::size_t>(edge.b)].push_back(edge.a);
    }
    for (auto& neighbors : neighbors_) {
        std::sort(neighbors.begin(), neighbors.end());
    }

    adjacent_land_tiles_.assign(size, {});
    for (int tile_index : land_tile_indices_) {
        for (int node : tiles_[static_cast<std::size_t>(tile_index)].nodes) {
            adjacent_land_tiles_[static_cast<std::size_t>(node)].push_back(tile_index);
        }
    }

    node_ports_.assign(size, 0);
    for (const Tile& tile : tiles_) {
        if (tile.kind != TileKind::port) {
            continue;
        }
        const std::uint8_t bit =
            tile.resource.has_value()
                ? static_cast<std::uint8_t>(1U << static_cast<unsigned>(*tile.resource))
                : kGenericPortBit;
        const auto [first, second] = port_nodes(tile);
        node_ports_[static_cast<std::size_t>(first)] |= bit;
        node_ports_[static_cast<std::size_t>(second)] |= bit;
    }
}

int CatanMap::node_slot(int node_id) const noexcept {
    if (node_id < 0 || node_id >= num_node_ids()) {
        return -1;
    }
    return node_slots_[static_cast<std::size_t>(node_id)];
}

int CatanMap::edge_slot(Edge edge) const noexcept {
    edge = edge.normalized();
    const auto it = std::lower_bound(land_edges_.begin(), land_edges_.end(), edge);
    if (it == land_edges_.end() || *it != edge) {
        return -1;
    }
    return static_cast<int>(std::distance(land_edges_.begin(), it));
}

int CatanMap::land_tile_slot(int tile_index) const noexcept {
    if (tile_index < 0 || static_cast<std::size_t>(tile_index) >= land_tile_slots_.size()) {
        return -1;
    }
    return land_tile_slots_[static_cast<std::size_t>(tile_index)];
}

const std::vector<int>& CatanMap::neighbors(int node_id) const {
    return neighbors_.at(static_cast<std::size_t>(node_id));
}

const std::vector<int>& CatanMap::adjacent_land_tiles(int node_id) const {
    return adjacent_land_tiles_.at(static_cast<std::size_t>(node_id));
}

std::uint8_t CatanMap::node_ports(int node_id) const {
    return node_ports_.at(static_cast<std::size_t>(node_id));
}

std::pair<int, int> port_nodes(const Tile& tile) {
    if (!tile.port_direction.has_value()) {
        throw std::invalid_argument("tile is not a port");
    }
    // Catanatron's PORT_DIRECTION_TO_NODEREFS.
    switch (*tile.port_direction) {
        case Direction::west:
            return {tile.nodes[index(NodeRef::northwest)], tile.nodes[index(NodeRef::southwest)]};
        case Direction::northwest:
            return {tile.nodes[index(NodeRef::north)], tile.nodes[index(NodeRef::northwest)]};
        case Direction::northeast:
            return {tile.nodes[index(NodeRef::northeast)], tile.nodes[index(NodeRef::north)]};
        case Direction::east:
            return {tile.nodes[index(NodeRef::southeast)], tile.nodes[index(NodeRef::northeast)]};
        case Direction::southeast:
            return {tile.nodes[index(NodeRef::south)], tile.nodes[index(NodeRef::southeast)]};
        case Direction::southwest:
            return {tile.nodes[index(NodeRef::southwest)], tile.nodes[index(NodeRef::south)]};
    }
    throw std::logic_error("unknown port direction");
}

const Tile& CatanMap::tile_at(Coordinate coordinate) const {
    return tiles_.at(static_cast<std::size_t>(coordinate_to_tile_.at(coordinate)));
}

}  // namespace cppanatron
