#include "cppanatron/graph_encoder.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
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

/** Multi-source BFS over land edges from a seat's buildings and road ends. */
std::vector<int> network_distances(const Game& game, Color color) {
    const CatanMap& map = game.board().map();
    std::vector<int> distance(static_cast<std::size_t>(map.num_node_ids()), kUnreached);
    std::vector<int> frontier;
    const auto reach = [&](int node) {
        if (distance[static_cast<std::size_t>(node)] != 0) {
            distance[static_cast<std::size_t>(node)] = 0;
            frontier.push_back(node);
        }
    };
    const PlayerState& player = game.player(color);
    for (int node : player.settlements) {
        reach(node);
    }
    for (int node : player.cities) {
        reach(node);
    }
    for (Edge edge : player.roads) {
        reach(edge.a);
        reach(edge.b);
    }
    for (std::size_t head = 0; head < frontier.size(); ++head) {
        const int node = frontier[head];
        const int next = distance[static_cast<std::size_t>(node)] + 1;
        for (int neighbor : map.neighbors(node)) {
            int& known = distance[static_cast<std::size_t>(neighbor)];
            if (known > next) {
                known = next;
                frontier.push_back(neighbor);
            }
        }
    }
    return distance;
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
    std::fill(out.begin(), out.end(), 0.0F);

    const Board& board = game.board();
    const CatanMap& map = board.map();
    const GameRules& rules = game.rules();
    const Color me = game.colors()[static_cast<std::size_t>(observer)];
    const Coordinate robber = board.robber_coordinate();
    const int robber_tile = map.coordinate_to_tile().at(robber);
    const auto seat_color = [&](int slot) {
        return game.colors()[static_cast<std::size_t>((observer + slot) % num_players)];
    };
    // Relative slot of each color seated in this game.
    std::array<int, kColors.size()> slot_of{};
    slot_of.fill(-1);
    for (int slot = 0; slot < num_players; ++slot) {
        slot_of[static_cast<std::size_t>(seat_color(slot))] = slot;
    }
    const auto slot_of_color = [&](Color color) { return slot_of[static_cast<std::size_t>(color)]; };
    const auto tile_pips = [&](int tile_index) {
        const Tile& tile = map.tiles()[static_cast<std::size_t>(tile_index)];
        return tile.resource.has_value() ? pips(*tile.number) : 0;
    };
    const int deck_size = std::accumulate(
        rules.development_deck.begin(), rules.development_deck.end(), 0);

    float* const row = out.data();
    const auto block = [&](std::size_t index) { return row + layout_[index].offset; };

    // ---- global
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
        std::array<int, 5> board_pips{};
        for (int tile_index : map.land_tile_indices()) {
            const Tile& tile = map.tiles()[static_cast<std::size_t>(tile_index)];
            if (tile.resource.has_value()) {
                board_pips[static_cast<std::size_t>(*tile.resource)] += pips(*tile.number);
            }
        }
        for (int value : board_pips) {
            cursor.put(ratio(value, 30.0));
        }
        expect(cursor, block(0), global_names_.size(), "global");
    }

    // ---- players
    for (int slot = 0; slot < num_players; ++slot) {
        float* const begin = block(1) + static_cast<std::size_t>(slot) * player_names_.size();
        Cursor cursor(begin);
        const Color color = seat_color(slot);
        const int seat = (observer + slot) % num_players;
        const PlayerState& player = game.player(color);
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
            for (int tile_index : map.adjacent_land_tiles(node)) {
                const Tile& tile = map.tiles()[static_cast<std::size_t>(tile_index)];
                if (tile.resource.has_value() && tile_index != robber_tile) {
                    production[static_cast<std::size_t>(*tile.resource)] +=
                        multiplier * pips(*tile.number);
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

    // ---- tiles
    const auto& land_tiles = map.land_tile_indices();
    for (std::size_t slot = 0; slot < land_tiles.size(); ++slot) {
        float* const begin = block(2) + slot * tile_names_.size();
        Cursor cursor(begin);
        const Tile& tile = map.tiles()[static_cast<std::size_t>(land_tiles[slot])];
        cursor.put(true);
        for (Resource resource : kResources) {
            cursor.put(tile.resource == resource);
        }
        cursor.put(!tile.resource.has_value());
        for (int number : kNumbers) {
            cursor.put(tile.number == number);
        }
        cursor.put(ratio(tile_pips(land_tiles[slot]), 5.0));
        cursor.put(land_tiles[slot] == robber_tile);
        float* const buildings = cursor.position();
        cursor.skip(static_cast<std::size_t>(dims_.max_players));
        for (int node : tile.nodes) {
            const auto building = board.buildings().find(node);
            if (building != board.buildings().end()) {
                const int multiplier = building->second.building == Building::city ? 2 : 1;
                buildings[slot_of_color(building->second.color)] +=
                    static_cast<float>(multiplier) / 6.0F;
            }
        }
        expect(cursor, begin, tile_names_.size(), "tile");
    }

    // ---- nodes
    std::vector<std::vector<int>> distances;
    for (int slot = 0; slot < num_players; ++slot) {
        distances.push_back(network_distances(game, seat_color(slot)));
    }
    const std::vector<int> open = board.buildable_node_ids(me, true);
    const std::vector<int> buildable =
        board.buildable_node_ids(me, game.is_initial_build_phase());
    const auto& nodes = map.land_node_list();
    const auto players = static_cast<std::size_t>(dims_.max_players);
    for (std::size_t slot = 0; slot < nodes.size(); ++slot) {
        float* const begin = block(3) + slot * node_names_.size();
        Cursor cursor(begin);
        const int node = nodes[slot];
        cursor.put(true);
        float* const owners = cursor.position();
        cursor.skip(2 * players);
        if (const auto building = board.buildings().find(node);
            building != board.buildings().end()) {
            const auto owner = static_cast<std::size_t>(slot_of_color(building->second.color));
            owners[(building->second.building == Building::city ? players : 0) + owner] = 1.0F;
        }
        cursor.put_bits(map.node_ports(node));
        std::array<int, 5> node_pips{};
        bool robbed = false;
        for (int tile_index : map.adjacent_land_tiles(node)) {
            const Tile& tile = map.tiles()[static_cast<std::size_t>(tile_index)];
            if (tile_index == robber_tile) {
                robbed = true;
            } else if (tile.resource.has_value()) {
                node_pips[static_cast<std::size_t>(*tile.resource)] += pips(*tile.number);
            }
        }
        for (int value : node_pips) {
            cursor.put(ratio(value, 13.0));
        }
        cursor.put(robbed);
        cursor.put(ratio(static_cast<double>(map.adjacent_land_tiles(node).size()), 3.0));
        cursor.put(std::binary_search(open.begin(), open.end(), node));
        cursor.put(std::binary_search(buildable.begin(), buildable.end(), node));
        for (std::size_t seat = 0; seat < players; ++seat) {
            cursor.put(
                seat < distances.size()
                    ? capped(distances[seat][static_cast<std::size_t>(node)], 6.0)
                    : 0.0F);
        }
        expect(cursor, begin, node_names_.size(), "node");
    }

    // ---- edges
    std::vector<Edge> buildable_edges = board.buildable_edges(me);
    std::sort(buildable_edges.begin(), buildable_edges.end());
    const auto& edges = map.land_edges();
    for (std::size_t slot = 0; slot < edges.size(); ++slot) {
        float* const begin = block(4) + slot * edge_names_.size();
        Cursor cursor(begin);
        cursor.put(true);
        float* const roads = cursor.position();
        cursor.skip(players);
        if (const auto owner = board.edge_color(edges[slot]); owner.has_value()) {
            roads[slot_of_color(*owner)] = 1.0F;
        }
        cursor.put(std::binary_search(buildable_edges.begin(), buildable_edges.end(), edges[slot]));
        expect(cursor, begin, edge_names_.size(), "edge");
    }

    // ---- topology
    float* edge_nodes = block(5);
    for (std::size_t slot = 0; slot < static_cast<std::size_t>(dims_.max_edges); ++slot) {
        const bool valid = slot < edges.size();
        edge_nodes[2 * slot] = valid ? static_cast<float>(map.node_slot(edges[slot].a)) : -1.0F;
        edge_nodes[2 * slot + 1] = valid ? static_cast<float>(map.node_slot(edges[slot].b)) : -1.0F;
    }
    float* tile_nodes = block(6);
    for (std::size_t slot = 0; slot < static_cast<std::size_t>(dims_.max_tiles); ++slot) {
        for (std::size_t corner = 0; corner < 6; ++corner) {
            tile_nodes[6 * slot + corner] =
                slot < land_tiles.size()
                    ? static_cast<float>(map.node_slot(
                          map.tiles()[static_cast<std::size_t>(land_tiles[slot])].nodes[corner]))
                    : -1.0F;
        }
    }

    // ---- action mask
    float* const mask = block(7);
    if (observer != game.current_player_index() || game.winning_color().has_value()) {
        mask[action_space_.noop()] = 1.0F;
        return;
    }
    thread_local std::vector<std::uint8_t> legal;
    legal.resize(action_space_.size());
    action_space_.mask(game, legal);
    for (std::size_t i = 0; i < legal.size(); ++i) {
        mask[i] = legal[i] != 0 ? 1.0F : 0.0F;
    }
}

}  // namespace cppanatron
