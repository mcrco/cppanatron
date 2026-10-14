#include "cppanatron/graph_encoder.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace cppanatron {
namespace {

constexpr std::array<int, 10> kNumbers{2, 3, 4, 5, 6, 8, 9, 10, 11, 12};
constexpr std::array<const char*, 5> kResourceNames{"wood", "brick", "sheep", "wheat", "ore"};
constexpr std::array<const char*, 5> kCardNames{
    "knight", "year_of_plenty", "monopoly", "road_building", "victory_point"};
constexpr std::array<const char*, 7> kPromptNames{
    "build_initial_settlement", "build_initial_road", "play_turn", "discard",
    "move_robber", "decide_trade", "decide_acceptees"};
constexpr int kUnreached = 1'000;

int pips(int number) { return 6 - std::abs(7 - number); }

float ratio(double value, double denominator) {
    return denominator > 0.0 ? static_cast<float>(value / denominator) : 0.0F;
}

float capped(double value, double denominator) {
    return std::min(ratio(value, denominator), 1.0F);
}

void add_resources(std::vector<std::string>& names, const std::string& prefix) {
    for (const char* resource : kResourceNames) {
        names.push_back(prefix + resource);
    }
}

void add_seats(std::vector<std::string>& names, const std::string& prefix, int seats) {
    for (int k = 0; k < seats; ++k) {
        names.push_back(prefix + std::to_string(k));
    }
}

void add_ports(std::vector<std::string>& names) {
    add_resources(names, "port_");
    names.push_back("port_3to1");
}

/** Sequential writer that checks each slot gets exactly its feature count. */
class Cursor {
public:
    explicit Cursor(float* begin) : position_(begin) {}

    void put(float value) { *position_++ = value; }
    void put(bool value) { put(value ? 1.0F : 0.0F); }
    void skip(std::size_t count) { position_ += count; }
    void put_bits(std::uint8_t bits) {
        for (unsigned bit = 0; bit < 6; ++bit) {
            put(((bits >> bit) & 1U) != 0U);
        }
    }
    [[nodiscard]] float* position() const noexcept { return position_; }

private:
    float* position_;
};

void expect(const Cursor& cursor, const float* slot, std::size_t size, const char* block) {
    if (cursor.position() != slot + size) {
        throw std::logic_error(std::string("graph encoder wrote the wrong number of ") + block +
                               " features");
    }
}

}  // namespace

GraphEncoder::GraphEncoder(EncoderDims dims, InfoLevel level)
    : dims_(dims), level_(level), action_space_(dims) {
    const int seats = dims_.max_players;

    for (const char* prompt : kPromptNames) {
        global_names_.push_back(std::string("prompt_") + prompt);
    }
    global_names_.push_back("road_building");
    add_resources(global_names_, "bank_");
    for (const char* name : {"deck", "turn", "vps_to_win", "discard_limit", "friendly_robber"}) {
        global_names_.push_back(name);
    }
    for (int count = 1; count <= seats; ++count) {
        global_names_.push_back("players_" + std::to_string(count));
    }
    add_resources(global_names_, "board_pips_");

    for (const char* name : {"present", "acting", "turn_owner", "public_vps", "actual_vps"}) {
        player_names_.push_back(name);
    }
    add_resources(player_names_, "resource_");
    for (const char* name : {"hand_size", "over_discard_limit", "discards_left", "dev_hand_size"}) {
        player_names_.push_back(name);
    }
    for (const char* card : kCardNames) {
        player_names_.push_back(std::string("dev_") + card);
    }
    for (std::size_t i = 0; i < 4; ++i) {
        player_names_.push_back(std::string("dev_playable_") + kCardNames[i]);
    }
    player_names_.push_back("played_dev_this_turn");
    for (std::size_t i = 0; i < 4; ++i) {
        player_names_.push_back(std::string("played_") + kCardNames[i]);
    }
    for (const char* name : {"longest_road", "largest_army", "road_length", "roads_left",
                             "settlements_left", "cities_left", "has_rolled"}) {
        player_names_.push_back(name);
    }
    add_resources(player_names_, "production_");
    add_ports(player_names_);

    tile_names_.push_back("valid");
    add_resources(tile_names_, "resource_");
    tile_names_.push_back("desert");
    for (int number : kNumbers) {
        tile_names_.push_back("number_" + std::to_string(number));
    }
    tile_names_.push_back("pips");
    tile_names_.push_back("robber");
    add_seats(tile_names_, "buildings_", seats);

    node_names_.push_back("valid");
    add_seats(node_names_, "settlement_", seats);
    add_seats(node_names_, "city_", seats);
    add_ports(node_names_);
    add_resources(node_names_, "pips_");
    for (const char* name : {"robber", "tiles", "open", "buildable"}) {
        node_names_.push_back(name);
    }
    add_seats(node_names_, "distance_", seats);

    edge_names_.push_back("valid");
    add_seats(edge_names_, "road_", seats);
    edge_names_.push_back("buildable");

    const auto players = static_cast<std::size_t>(seats);
    const auto tiles = static_cast<std::size_t>(dims_.max_tiles);
    const auto nodes = static_cast<std::size_t>(dims_.max_nodes);
    const auto edges = static_cast<std::size_t>(dims_.max_edges);
    std::size_t offset = 0;
    const auto add_block = [&](const char* name, std::vector<std::size_t> shape) {
        const std::size_t count = std::accumulate(
            shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
        layout_.push_back({name, offset, std::move(shape)});
        offset += count;
    };
    add_block("global", {global_names_.size()});
    add_block("players", {players, player_names_.size()});
    add_block("tiles", {tiles, tile_names_.size()});
    add_block("nodes", {nodes, node_names_.size()});
    add_block("edges", {edges, edge_names_.size()});
    add_block("edge_nodes", {edges, 2});
    add_block("tile_nodes", {tiles, 6});
    add_block("action_mask", {action_space_.size()});
    size_ = offset;

    const auto at = [](const std::vector<std::string>& names, const std::string& name) {
        return static_cast<std::size_t>(
            std::find(names.begin(), names.end(), name) - names.begin());
    };
    offsets_ = {
        .tile_robber = at(tile_names_, "robber"),
        .tile_buildings = at(tile_names_, "buildings_0"),
        .node_settlement = at(node_names_, "settlement_0"),
        .node_city = at(node_names_, "city_0"),
        .node_pips = at(node_names_, "pips_wood"),
        .node_robber = at(node_names_, "robber"),
        .node_open = at(node_names_, "open"),
        .node_buildable = at(node_names_, "buildable"),
        .node_distance = at(node_names_, "distance_0"),
        .edge_road = at(edge_names_, "road_0"),
        .edge_buildable = at(edge_names_, "buildable"),
    };
}

const std::vector<std::string>& GraphEncoder::feature_names(const std::string& block) const {
    if (block == "global") {
        return global_names_;
    }
    if (block == "players") {
        return player_names_;
    }
    if (block == "tiles") {
        return tile_names_;
    }
    if (block == "nodes") {
        return node_names_;
    }
    if (block == "edges") {
        return edge_names_;
    }
    throw std::invalid_argument("no per-slot feature names for block " + block);
}


/**
 * Map-only data, in slots: adjacency for the per-call passes and the row with
 * every feature that depends on the map alone already written (validity,
 * terrain, numbers, ports, tile counts, board pips, topology), so encode
 * starts from a copy of it. Built per thread, per live map and dims.
 */
struct GraphEncoder::MapCache {
    const CatanMap* address{};
    std::weak_ptr<const CatanMap> map;
    EncoderDims dims;
    std::vector<float> base;
    /** Per node slot: up to 3 neighbor node slots and adjacent land tile slots (-1 pads). */
    std::vector<std::array<int, 3>> neighbors;
    std::vector<std::array<int, 3>> tiles;
    /** Per node slot: pips of adjacent tiles by resource, robber ignored. */
    std::vector<std::array<int, 5>> pips;
    /** Per land tile slot: the corner node slots, resource (-1 desert) and pips. */
    std::vector<std::array<int, 6>> corners;
    std::vector<int> tile_resource;
    std::vector<int> tile_pips;
    /** Per land tile index (tiles()): its land tile slot, or -1. */
    std::vector<int> tile_slot;
};

const GraphEncoder::MapCache& GraphEncoder::map_cache(
    const std::shared_ptr<const CatanMap>& map) const {
    // A handful of recent maps per thread: VecEnvs and searches revisit the
    // same few maps, and an entry is only reused while its map is alive.
    constexpr std::size_t kRecent = 8;
    thread_local std::array<std::shared_ptr<MapCache>, kRecent> recent{};
    for (std::size_t i = 0; i < kRecent; ++i) {
        const auto& entry = recent[i];
        if (entry != nullptr && entry->address == map.get() && entry->dims == dims_ &&
            !entry->map.expired()) {
            std::rotate(recent.begin(), recent.begin() + static_cast<std::ptrdiff_t>(i),
                        recent.begin() + static_cast<std::ptrdiff_t>(i) + 1);
            return *recent.front();
        }
    }

    auto cache = std::make_shared<MapCache>();
    cache->address = map.get();
    cache->map = map;
    cache->dims = dims_;
    const auto& nodes = map->land_node_list();
    const auto& land_tiles = map->land_tile_indices();
    const auto& edges = map->land_edges();
    cache->tile_slot.assign(map->tiles().size(), -1);
    for (std::size_t slot = 0; slot < land_tiles.size(); ++slot) {
        cache->tile_slot[static_cast<std::size_t>(land_tiles[slot])] = static_cast<int>(slot);
        const Tile& tile = map->tiles()[static_cast<std::size_t>(land_tiles[slot])];
        std::array<int, 6> corners{};
        for (std::size_t corner = 0; corner < 6; ++corner) {
            corners[corner] = map->node_slot(tile.nodes[corner]);
        }
        cache->corners.push_back(corners);
        cache->tile_resource.push_back(
            tile.resource.has_value() ? static_cast<int>(*tile.resource) : -1);
        cache->tile_pips.push_back(tile.resource.has_value() ? pips(*tile.number) : 0);
    }
    for (int node : nodes) {
        std::array<int, 3> neighbors{-1, -1, -1};
        std::array<int, 3> tiles{-1, -1, -1};
        std::array<int, 5> node_pips{};
        const auto& adjacent = map->neighbors(node);
        for (std::size_t i = 0; i < adjacent.size() && i < 3; ++i) {
            neighbors[i] = map->node_slot(adjacent[i]);
        }
        const auto& touching = map->adjacent_land_tiles(node);
        for (std::size_t i = 0; i < touching.size() && i < 3; ++i) {
            const int tile = cache->tile_slot[static_cast<std::size_t>(touching[i])];
            tiles[i] = tile;
            if (cache->tile_resource[static_cast<std::size_t>(tile)] >= 0) {
                node_pips[static_cast<std::size_t>(
                    cache->tile_resource[static_cast<std::size_t>(tile)])] +=
                    cache->tile_pips[static_cast<std::size_t>(tile)];
            }
        }
        cache->neighbors.push_back(neighbors);
        cache->tiles.push_back(tiles);
        cache->pips.push_back(node_pips);
    }

    // The map-only features, in the order the names list them.
    std::vector<float>& row = cache->base;
    row.assign(size_, 0.0F);
    const auto block = [&](std::size_t index) { return row.data() + layout_[index].offset; };
    {
        std::array<int, 5> board_pips{};
        for (std::size_t slot = 0; slot < land_tiles.size(); ++slot) {
            if (cache->tile_resource[slot] >= 0) {
                board_pips[static_cast<std::size_t>(cache->tile_resource[slot])] +=
                    cache->tile_pips[slot];
            }
        }
        float* const pips_begin = block(0) + global_names_.size() - board_pips.size();
        for (std::size_t r = 0; r < board_pips.size(); ++r) {
            pips_begin[r] = ratio(board_pips[r], 30.0);
        }
    }
    for (std::size_t slot = 0; slot < land_tiles.size(); ++slot) {
        float* const begin = block(2) + slot * tile_names_.size();
        Cursor cursor(begin);
        const Tile& tile = map->tiles()[static_cast<std::size_t>(land_tiles[slot])];
        cursor.put(true);
        for (Resource resource : kResources) {
            cursor.put(tile.resource == resource);
        }
        cursor.put(!tile.resource.has_value());
        for (int number : kNumbers) {
            cursor.put(tile.number == number);
        }
        cursor.put(ratio(cache->tile_pips[slot], 5.0));
        cursor.skip(1 + static_cast<std::size_t>(dims_.max_players));  // robber, buildings
        expect(cursor, begin, tile_names_.size(), "tile");
    }
    for (std::size_t slot = 0; slot < nodes.size(); ++slot) {
        float* const begin = block(3) + slot * node_names_.size();
        Cursor cursor(begin);
        cursor.put(true);
        cursor.skip(2 * static_cast<std::size_t>(dims_.max_players));  // buildings
        cursor.put_bits(map->node_ports(nodes[slot]));
        for (int value : cache->pips[slot]) {
            cursor.put(ratio(value, 13.0));
        }
        cursor.skip(1);  // robber
        cursor.put(ratio(static_cast<double>(map->adjacent_land_tiles(nodes[slot]).size()), 3.0));
        cursor.skip(2 + static_cast<std::size_t>(dims_.max_players));  // open .. distances
        expect(cursor, begin, node_names_.size(), "node");
    }
    for (std::size_t slot = 0; slot < edges.size(); ++slot) {
        block(4)[slot * edge_names_.size()] = 1.0F;  // valid
    }
    float* const edge_nodes = block(5);
    for (std::size_t slot = 0; slot < static_cast<std::size_t>(dims_.max_edges); ++slot) {
        const bool valid = slot < edges.size();
        edge_nodes[2 * slot] = valid ? static_cast<float>(map->node_slot(edges[slot].a)) : -1.0F;
        edge_nodes[2 * slot + 1] =
            valid ? static_cast<float>(map->node_slot(edges[slot].b)) : -1.0F;
    }
    float* const tile_nodes = block(6);
    for (std::size_t slot = 0; slot < static_cast<std::size_t>(dims_.max_tiles); ++slot) {
        for (std::size_t corner = 0; corner < 6; ++corner) {
            tile_nodes[6 * slot + corner] =
                slot < land_tiles.size() ? static_cast<float>(cache->corners[slot][corner]) : -1.0F;
        }
    }

    std::rotate(recent.begin(), recent.end() - 1, recent.end());
    recent.front() = std::move(cache);
    return *recent.front();
}

void GraphEncoder::encode_waiting(std::span<float> out) const {
    if (out.size() != size_) {
        throw std::invalid_argument("observation buffer must match the encoder size");
    }
    std::fill(out.begin(), out.end(), 0.0F);
    out[mask_offset() + action_space_.noop()] = 1.0F;
}

void GraphEncoder::encode(const Game& game, int observer, std::span<float> out) const {
    if (out.size() != size_) {
        throw std::invalid_argument("observation buffer must match the encoder size");
    }
    dims_.require_fits(game);
    const int num_players = static_cast<int>(game.colors().size());
    if (observer < 0 || observer >= num_players) {
        throw std::out_of_range("observer is not a seat of this game");
    }

    const Board& board = game.board();
    const CatanMap& map = board.map();
    const MapCache& cache = map_cache(board.shared_map());
    std::memcpy(out.data(), cache.base.data(), size_ * sizeof(float));

    const GameRules& rules = game.rules();
    const auto seat_of_slot = [&](int slot) { return (observer + slot) % num_players; };
    // Relative slot of each color seated in this game.
    std::array<int, kColors.size()> slot_of{};
    slot_of.fill(-1);
    for (int slot = 0; slot < num_players; ++slot) {
        slot_of[static_cast<std::size_t>(game.colors()[static_cast<std::size_t>(
            seat_of_slot(slot))])] = slot;
    }
    const auto slot_of_color = [&](Color color) {
        return slot_of[static_cast<std::size_t>(color)];
    };
    const int robber_slot =
        cache.tile_slot[static_cast<std::size_t>(map.coordinate_to_tile().at(board.robber_coordinate()))];
    const int deck_size = std::accumulate(
        rules.development_deck.begin(), rules.development_deck.end(), 0);

    float* const row = out.data();
    const auto block = [&](std::size_t index) { return row + layout_[index].offset; };
    const auto players = static_cast<std::size_t>(dims_.max_players);
    const std::size_t num_nodes = map.land_node_list().size();

    // ---- global (the board pips are the map's)
    {
        Cursor cursor(block(0));
        for (std::size_t prompt = 0; prompt < kPromptNames.size(); ++prompt) {
            cursor.put(static_cast<std::size_t>(game.current_prompt()) == prompt);
        }
        cursor.put(game.is_road_building());
        for (std::size_t r = 0; r < kResources.size(); ++r) {
            cursor.put(ratio(game.resource_bank()[r], rules.resource_bank[r]));
        }
        cursor.put(ratio(static_cast<double>(game.development_cards_remaining()), deck_size));
        cursor.put(capped(game.num_turns(), 500.0));
        cursor.put(ratio(game.victory_points_to_win(), 20.0));
        cursor.put(ratio(game.discard_limit(), 20.0));
        cursor.put(game.friendly_robber());
        for (int count = 1; count <= dims_.max_players; ++count) {
            cursor.put(count == num_players);
        }
        cursor.skip(kResources.size());
        expect(cursor, block(0), global_names_.size(), "global");
    }

    // ---- players
    for (int slot = 0; slot < num_players; ++slot) {
        float* const begin = block(1) + static_cast<std::size_t>(slot) * player_names_.size();
        Cursor cursor(begin);
        const int seat = seat_of_slot(slot);
        const Color color = game.colors()[static_cast<std::size_t>(seat)];
        const PlayerState& player = game.players()[static_cast<std::size_t>(seat)];
        const bool own = slot == 0;
        const bool hand_visible = own || level_ != InfoLevel::private_;
        const bool cards_visible = own || level_ == InfoLevel::full;
        const double vps = game.victory_points_to_win();
        const int hand = std::accumulate(player.resources.begin(), player.resources.end(), 0);
        const int dev_hand = std::accumulate(
            player.development_cards.begin(), player.development_cards.end(), 0);

        cursor.put(true);
        cursor.put(seat == game.current_player_index());
        cursor.put(seat == game.current_turn_index());
        cursor.put(ratio(player.victory_points, vps));
        cursor.put(cards_visible ? ratio(player.actual_victory_points, vps) : 0.0F);
        for (std::size_t r = 0; r < kResources.size(); ++r) {
            cursor.put(hand_visible ? ratio(player.resources[r], rules.resource_bank[r]) : 0.0F);
        }
        cursor.put(capped(hand, 20.0));
        cursor.put(hand > game.discard_limit());
        cursor.put(capped(game.remaining_discards(color), 10.0));
        cursor.put(capped(dev_hand, 10.0));
        for (std::size_t d = 0; d < 5; ++d) {
            cursor.put(
                cards_visible ? ratio(player.development_cards[d], rules.development_deck[d])
                              : 0.0F);
        }
        for (std::size_t d = 0; d < 4; ++d) {
            cursor.put(
                cards_visible && player.development_cards[d] > 0 &&
                player.development_card_owned_at_start[d]);
        }
        cursor.put(player.has_played_development_card_in_turn);
        for (std::size_t d = 0; d < 4; ++d) {
            cursor.put(ratio(player.played_development_cards[d], rules.development_deck[d]));
        }
        cursor.put(player.has_road);
        cursor.put(player.has_army);
        cursor.put(capped(player.longest_road_length, 15.0));
        cursor.put(ratio(player.roads_available, rules.roads_per_player));
        cursor.put(ratio(player.settlements_available, rules.settlements_per_player));
        cursor.put(ratio(player.cities_available, rules.cities_per_player));
        cursor.put(player.has_rolled);

        std::array<int, 5> production{};
        std::uint8_t ports = 0;
        const auto collect = [&](int node, int multiplier) {
            ports |= map.node_ports(node);
            for (int tile : cache.tiles[static_cast<std::size_t>(map.node_slot(node))]) {
                if (tile >= 0 && tile != robber_slot &&
                    cache.tile_resource[static_cast<std::size_t>(tile)] >= 0) {
                    production[static_cast<std::size_t>(
                        cache.tile_resource[static_cast<std::size_t>(tile)])] +=
                        multiplier * cache.tile_pips[static_cast<std::size_t>(tile)];
                }
            }
        };
        for (int node : player.settlements) {
            collect(node, 1);
        }
        for (int node : player.cities) {
            collect(node, 2);
        }
        for (int value : production) {
            cursor.put(ratio(value, 15.0));
        }
        cursor.put_bits(ports);
        expect(cursor, begin, player_names_.size(), "player");
    }

    // ---- buildings and roads, scattered once into flat per-slot arrays
    struct Scratch {
        std::vector<int> owner;           // relative seat of the building at a node slot, or -1
        std::vector<std::uint8_t> open;   // the distance rule allows a settlement
        std::vector<std::uint8_t> mine;   // in the observer's road network
        std::vector<int> tile_weight;     // settlements + 2 x cities per (tile slot, seat)
        std::vector<int> distance;        // per (seat, node slot)
        std::vector<int> frontier;
    };
    thread_local Scratch scratch;
    scratch.owner.assign(num_nodes, -1);
    scratch.open.assign(num_nodes, 1);
    scratch.mine.assign(num_nodes, 0);
    scratch.tile_weight.assign(map.land_tile_indices().size() * players, 0);
    float* const node_block = block(3);
    float* const tile_block = block(2);
    const std::size_t node_width = node_names_.size();
    const std::size_t tile_width = tile_names_.size();
    for (const auto& [node, building] : board.buildings()) {
        const auto slot = static_cast<std::size_t>(map.node_slot(node));
        const int owner = slot_of_color(building.color);
        const bool city = building.building == Building::city;
        scratch.owner[slot] = owner;
        scratch.open[slot] = 0;
        for (int neighbor : cache.neighbors[slot]) {
            if (neighbor >= 0) {
                scratch.open[static_cast<std::size_t>(neighbor)] = 0;
            }
        }
        node_block[slot * node_width + (city ? offsets_.node_city : offsets_.node_settlement) +
                   static_cast<std::size_t>(owner)] = 1.0F;
        for (int tile : cache.tiles[slot]) {
            if (tile >= 0) {
                scratch.tile_weight[static_cast<std::size_t>(tile) * players +
                                    static_cast<std::size_t>(owner)] += city ? 2 : 1;
            }
        }
    }
    for (std::size_t tile = 0; tile < map.land_tile_indices().size(); ++tile) {
        for (std::size_t seat = 0; seat < players; ++seat) {
            if (const int weight = scratch.tile_weight[tile * players + seat]; weight != 0) {
                tile_block[tile * tile_width + offsets_.tile_buildings + seat] =
                    static_cast<float>(weight) / 6.0F;
            }
        }
    }
    tile_block[static_cast<std::size_t>(robber_slot) * tile_width + offsets_.tile_robber] = 1.0F;

    // ---- nodes: the robber's pips, openness, buildability, network distances
    for (int node_slot : cache.corners[static_cast<std::size_t>(robber_slot)]) {
        const auto slot = static_cast<std::size_t>(node_slot);
        float* const node = node_block + slot * node_width;
        node[offsets_.node_robber] = 1.0F;
        const int resource = cache.tile_resource[static_cast<std::size_t>(robber_slot)];
        if (resource >= 0) {
            const auto r = static_cast<std::size_t>(resource);
            node[offsets_.node_pips + r] =
                ratio(cache.pips[slot][r] - cache.tile_pips[static_cast<std::size_t>(robber_slot)],
                      13.0);
        }
    }
    // A seat's network: its buildings and the ends of its roads.
    scratch.distance.assign(players * num_nodes, kUnreached);
    for (int slot = 0; slot < num_players; ++slot) {
        const PlayerState& player = game.players()[static_cast<std::size_t>(seat_of_slot(slot))];
        int* const distance = scratch.distance.data() + static_cast<std::size_t>(slot) * num_nodes;
        scratch.frontier.clear();
        const auto reach = [&](int node) {
            const auto node_slot = static_cast<std::size_t>(map.node_slot(node));
            if (distance[node_slot] != 0) {
                distance[node_slot] = 0;
                scratch.frontier.push_back(static_cast<int>(node_slot));
            }
        };
        for (int node : player.settlements) {
            reach(node);
        }
        for (int node : player.cities) {
            reach(node);
        }
        for (Edge edge : player.roads) {
            reach(edge.a);
            reach(edge.b);
            if (slot == 0) {
                scratch.mine[static_cast<std::size_t>(map.node_slot(edge.a))] = 1;
                scratch.mine[static_cast<std::size_t>(map.node_slot(edge.b))] = 1;
            }
        }
        for (std::size_t head = 0; head < scratch.frontier.size(); ++head) {
            const auto node_slot = static_cast<std::size_t>(scratch.frontier[head]);
            const int next = distance[node_slot] + 1;
            for (int neighbor : cache.neighbors[node_slot]) {
                if (neighbor >= 0 && distance[static_cast<std::size_t>(neighbor)] > next) {
                    distance[static_cast<std::size_t>(neighbor)] = next;
                    scratch.frontier.push_back(neighbor);
                }
            }
        }
    }
    // Settlements extend the network from its road ends (Board::buildable_node_ids);
    // during the initial placement every open node is buildable.
    const bool initial = game.is_initial_build_phase();
    for (std::size_t slot = 0; slot < num_nodes; ++slot) {
        float* const node = node_block + slot * node_width;
        const bool open = scratch.open[slot] != 0;
        node[offsets_.node_open] = open ? 1.0F : 0.0F;
        node[offsets_.node_buildable] = open && (initial || scratch.mine[slot] != 0) ? 1.0F : 0.0F;
        for (int seat = 0; seat < num_players; ++seat) {
            node[offsets_.node_distance + static_cast<std::size_t>(seat)] =
                capped(scratch.distance[static_cast<std::size_t>(seat) * num_nodes + slot], 6.0);
        }
    }

    // ---- edges: roads by seat, and where the observer may build one
    float* const edge_block = block(4);
    const std::size_t edge_width = edge_names_.size();
    const auto& edges = map.land_edges();
    for (const auto& [edge, color] : board.roads()) {
        edge_block[static_cast<std::size_t>(map.edge_slot(edge)) * edge_width +
                   offsets_.edge_road + static_cast<std::size_t>(slot_of_color(color))] = 1.0F;
    }
    // Board::buildable_edges: free edges with an end in the observer's network
    // (its buildings and road ends) that holds no opponent's building.
    const PlayerState& me = game.players()[static_cast<std::size_t>(observer)];
    for (int node : me.settlements) {
        scratch.mine[static_cast<std::size_t>(map.node_slot(node))] = 1;
    }
    for (int node : me.cities) {
        scratch.mine[static_cast<std::size_t>(map.node_slot(node))] = 1;
    }
    const auto expandable = [&](int node) {
        const auto slot = static_cast<std::size_t>(map.node_slot(node));
        return scratch.mine[slot] != 0 && (scratch.owner[slot] < 0 || scratch.owner[slot] == 0);
    };
    // The initial road must touch the settlement just placed.
    const int initial_road_at =
        game.current_prompt() == ActionPrompt::build_initial_road &&
                observer == game.current_player_index()
            ? me.settlements.back()
            : -1;
    for (std::size_t slot = 0; slot < edges.size(); ++slot) {
        const Edge edge = edges[slot];
        float* const features = edge_block + slot * edge_width;
        const bool free = std::none_of(
            features + offsets_.edge_road, features + offsets_.edge_road + players,
            [](float road) { return road != 0.0F; });
        const bool reaches = initial_road_at < 0 || edge.a == initial_road_at ||
                             edge.b == initial_road_at;
        features[offsets_.edge_buildable] =
            free && reaches && (expandable(edge.a) || expandable(edge.b)) ? 1.0F : 0.0F;
    }

    // ---- action mask
    float* const mask = block(7);
    if (observer != game.current_player_index() || game.winning_color().has_value()) {
        mask[action_space_.noop()] = 1.0F;
        return;
    }
    for (const Action& action : game.playable_actions()) {
        switch (action.type) {
            case ActionType::offer_trade:
            case ActionType::accept_trade:
            case ActionType::reject_trade:
            case ActionType::confirm_trade:
            case ActionType::cancel_trade:
                continue;
            default:
                mask[action_space_.index(game, action)] = 1.0F;
        }
    }
}

}  // namespace cppanatron
