#include "cppanatron/observation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace cppanatron {
namespace {

constexpr int kPlayerFullFeatureCount = 32;
constexpr int kPlayerPublicFeatureCount = 16;
constexpr int kPlayerPrivateFeatureCount = 16;

int turns_since(int completed_turns, int last_completed_turn) {
    return last_completed_turn < 0 ? -1 : completed_turns - last_completed_turn;
}

int card_count(const PlayerState& state, DevelopmentCard card) {
    return state.development_cards[static_cast<std::size_t>(card)];
}

int played_card_count(const PlayerState& state, DevelopmentCard card) {
    return state.played_development_cards[static_cast<std::size_t>(card)];
}

int playable_card(const PlayerState& state, DevelopmentCard card) {
    const auto index = static_cast<std::size_t>(card);
    return state.development_cards[index] > 0 &&
                   state.development_card_owned_at_start[index] &&
                   !state.has_played_development_card_in_turn
               ? 1
               : 0;
}

int resource_count(const PlayerState& state, Resource resource) {
    return state.resources[static_cast<std::size_t>(resource)];
}

int total_resources(const PlayerState& state) {
    return std::accumulate(state.resources.begin(), state.resources.end(), 0);
}

int total_development_cards(const PlayerState& state) {
    return std::accumulate(
        state.development_cards.begin(),
        state.development_cards.end(),
        0);
}

template <std::size_t Size>
void append_features(
    float*& destination,
    const std::array<float, Size>& features) {
    std::copy(features.begin(), features.end(), destination);
    destination += Size;
}

void write_full_player_features(
    const Game& game,
    const PlayerState& state,
    float*& output) {
    const int completed_turns = game.completed_turns();
    append_features(
        output,
        std::array<float, kPlayerFullFeatureCount>{
            static_cast<float>(state.actual_victory_points),
            static_cast<float>(resource_count(state, Resource::brick)),
            static_cast<float>(state.cities_available),
            static_cast<float>(state.has_army),
            static_cast<float>(state.has_played_development_card_in_turn),
            static_cast<float>(state.has_road),
            static_cast<float>(state.has_rolled),
            static_cast<float>(card_count(state, DevelopmentCard::knight)),
            static_cast<float>(playable_card(state, DevelopmentCard::knight)),
            static_cast<float>(played_card_count(state, DevelopmentCard::knight)),
            static_cast<float>(state.longest_road_length),
            static_cast<float>(card_count(state, DevelopmentCard::monopoly)),
            static_cast<float>(playable_card(state, DevelopmentCard::monopoly)),
            static_cast<float>(played_card_count(state, DevelopmentCard::monopoly)),
            static_cast<float>(total_development_cards(state)),
            static_cast<float>(total_resources(state)),
            static_cast<float>(resource_count(state, Resource::ore)),
            static_cast<float>(state.victory_points),
            static_cast<float>(state.roads_available),
            static_cast<float>(card_count(state, DevelopmentCard::road_building)),
            static_cast<float>(
                playable_card(state, DevelopmentCard::road_building)),
            static_cast<float>(
                played_card_count(state, DevelopmentCard::road_building)),
            static_cast<float>(state.settlements_available),
            static_cast<float>(resource_count(state, Resource::sheep)),
            static_cast<float>(turns_since(
                completed_turns,
                state.last_development_card_bought_completed_turn)),
            static_cast<float>(
                turns_since(completed_turns, state.last_knight_completed_turn)),
            static_cast<float>(
                card_count(state, DevelopmentCard::victory_point)),
            static_cast<float>(resource_count(state, Resource::wheat)),
            static_cast<float>(resource_count(state, Resource::wood)),
            static_cast<float>(
                card_count(state, DevelopmentCard::year_of_plenty)),
            static_cast<float>(
                playable_card(state, DevelopmentCard::year_of_plenty)),
            static_cast<float>(
                played_card_count(state, DevelopmentCard::year_of_plenty)),
        });
}

void write_public_player_features(
    const Game& game,
    const PlayerState& state,
    float*& output) {
    const int completed_turns = game.completed_turns();
    append_features(
        output,
        std::array<float, kPlayerPublicFeatureCount>{
            static_cast<float>(state.cities_available),
            static_cast<float>(state.has_army),
            static_cast<float>(state.has_road),
            static_cast<float>(state.has_rolled),
            static_cast<float>(played_card_count(state, DevelopmentCard::knight)),
            static_cast<float>(state.longest_road_length),
            static_cast<float>(played_card_count(state, DevelopmentCard::monopoly)),
            static_cast<float>(total_development_cards(state)),
            static_cast<float>(total_resources(state)),
            static_cast<float>(state.victory_points),
            static_cast<float>(state.roads_available),
            static_cast<float>(
                played_card_count(state, DevelopmentCard::road_building)),
            static_cast<float>(state.settlements_available),
            static_cast<float>(turns_since(
                completed_turns,
                state.last_development_card_bought_completed_turn)),
            static_cast<float>(
                turns_since(completed_turns, state.last_knight_completed_turn)),
            static_cast<float>(
                played_card_count(state, DevelopmentCard::year_of_plenty)),
        });
}

void write_private_player_features(
    const PlayerState& state,
    float*& output) {
    append_features(
        output,
        std::array<float, kPlayerPrivateFeatureCount>{
            static_cast<float>(state.actual_victory_points),
            static_cast<float>(resource_count(state, Resource::brick)),
            static_cast<float>(state.has_played_development_card_in_turn),
            static_cast<float>(card_count(state, DevelopmentCard::knight)),
            static_cast<float>(playable_card(state, DevelopmentCard::knight)),
            static_cast<float>(card_count(state, DevelopmentCard::monopoly)),
            static_cast<float>(playable_card(state, DevelopmentCard::monopoly)),
            static_cast<float>(resource_count(state, Resource::ore)),
            static_cast<float>(card_count(state, DevelopmentCard::road_building)),
            static_cast<float>(
                playable_card(state, DevelopmentCard::road_building)),
            static_cast<float>(resource_count(state, Resource::sheep)),
            static_cast<float>(
                card_count(state, DevelopmentCard::victory_point)),
            static_cast<float>(resource_count(state, Resource::wheat)),
            static_cast<float>(resource_count(state, Resource::wood)),
            static_cast<float>(
                card_count(state, DevelopmentCard::year_of_plenty)),
            static_cast<float>(
                playable_card(state, DevelopmentCard::year_of_plenty)),
        });
}

int color_index(Color color) {
    return static_cast<int>(color);
}

}  // namespace

ObservationLayout::ObservationLayout(
    int width,
    int height,
    std::vector<NodePosition> node_positions,
    std::vector<EdgePosition> edge_positions,
    std::vector<TilePosition> tile_positions)
    : width_(width), height_(height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("observation dimensions must be positive");
    }
    for (const auto& position : node_positions) {
        node_positions_.emplace(
            position.node,
            std::pair{position.x, position.y});
    }
    for (const auto& position : edge_positions) {
        edge_positions_.emplace(
            position.edge.normalized(),
            std::pair{position.x, position.y});
    }
    for (const auto& position : tile_positions) {
        tile_positions_.emplace(
            position.coordinate,
            std::pair{position.x, position.y});
    }
}

const std::pair<int, int>& ObservationLayout::node_position(int node) const {
    const auto position = node_positions_.find(node);
    if (position == node_positions_.end()) {
        throw std::out_of_range(
            "missing observation position for node " + std::to_string(node));
    }
    return position->second;
}

const std::pair<int, int>& ObservationLayout::edge_position(Edge edge) const {
    const auto normalized = edge.normalized();
    const auto position = edge_positions_.find(normalized);
    if (position == edge_positions_.end()) {
        throw std::out_of_range(
            "missing observation position for edge " +
            std::to_string(normalized.a) + "-" +
            std::to_string(normalized.b));
    }
    return position->second;
}

const std::pair<int, int>& ObservationLayout::tile_position(
    Coordinate coordinate) const {
    const auto position = tile_positions_.find(coordinate);
    if (position == tile_positions_.end()) {
        throw std::out_of_range(
            "missing observation position for tile " +
            std::to_string(coordinate.x) + "," +
            std::to_string(coordinate.y) + "," +
            std::to_string(coordinate.z));
    }
    return position->second;
}

std::size_t full_numeric_observation_size(int num_players) {
    if (num_players < 2 || num_players > 4) {
        throw std::invalid_argument("observation supports two to four players");
    }
    return static_cast<std::size_t>(32 * num_players + 10);
}

std::size_t board_observation_size(
    int num_players,
    const ObservationLayout& layout) {
    if (num_players < 2 || num_players > 4) {
        throw std::invalid_argument("observation supports two to four players");
    }
    const int channels = 2 * num_players + 12;
    return static_cast<std::size_t>(
        layout.width() * layout.height() * channels);
}

std::size_t full_observation_size(
    int num_players,
    const ObservationLayout& layout) {
    return full_numeric_observation_size(num_players) +
           board_observation_size(num_players, layout);
}

void write_full_observation(
    const Game& game,
    int base_player,
    const ObservationLayout& layout,
    float* output,
    std::size_t output_size) {
    const int num_players = static_cast<int>(game.players().size());
    if (output == nullptr) {
        throw std::invalid_argument("observation output is null");
    }
    if (base_player < 0 || base_player >= num_players) {
        throw std::out_of_range("base player is out of range");
    }
    if (output_size != full_observation_size(num_players, layout)) {
        throw std::invalid_argument("observation output has incorrect size");
    }
    std::fill(output, output + output_size, 0.0F);

    float* cursor = output;
    const auto& bank = game.resource_bank();
    append_features(
        cursor,
        std::array<float, 9>{
            static_cast<float>(bank[static_cast<std::size_t>(Resource::brick)]),
            static_cast<float>(game.development_cards_remaining()),
            static_cast<float>(bank[static_cast<std::size_t>(Resource::ore)]),
            static_cast<float>(bank[static_cast<std::size_t>(Resource::sheep)]),
            static_cast<float>(bank[static_cast<std::size_t>(Resource::wheat)]),
            static_cast<float>(bank[static_cast<std::size_t>(Resource::wood)]),
            static_cast<float>(game.is_discarding()),
            static_cast<float>(game.is_initial_build_phase()),
            static_cast<float>(game.is_moving_robber()),
        });

    const auto& players = game.players();
    write_full_player_features(
        game,
        players[static_cast<std::size_t>(base_player)],
        cursor);
    for (int offset = 1; offset < num_players; ++offset) {
        const int player = (base_player + offset) % num_players;
        write_public_player_features(
            game,
            players[static_cast<std::size_t>(player)],
            cursor);
    }
    *cursor++ = static_cast<float>(game.num_turns());
    for (int offset = 1; offset < num_players; ++offset) {
        const int player = (base_player + offset) % num_players;
        write_private_player_features(
            players[static_cast<std::size_t>(player)],
            cursor);
    }

    const std::size_t numeric_size = full_numeric_observation_size(num_players);
    if (cursor != output + numeric_size) {
        throw std::logic_error("numeric observation writer size mismatch");
    }
    const int channels = 2 * num_players + 12;
    auto board_index = [&](int x, int y, int channel) -> std::size_t {
        if (x < 0 || x >= layout.width() || y < 0 || y >= layout.height() ||
            channel < 0 || channel >= channels) {
            throw std::out_of_range("board observation coordinate is out of range");
        }
        return numeric_size +
               static_cast<std::size_t>(
                   (x * layout.height() + y) * channels + channel);
    };

    for (const auto& [node, building] : game.board().buildings()) {
        const int relative_player =
            (color_index(building.color) - base_player + num_players) %
            num_players;
        const auto [x, y] = layout.node_position(node);
        output[board_index(x, y, 2 * relative_player)] =
            building.building == Building::settlement ? 1.0F : 2.0F;
    }
    for (const auto& [edge, color] : game.board().roads()) {
        const int relative_player =
            (color_index(color) - base_player + num_players) % num_players;
        const auto [x, y] = layout.edge_position(edge);
        output[board_index(x, y, 2 * relative_player + 1)] = 1.0F;
    }

    const auto& map = game.board().map();
    for (const auto& [coordinate, tile_index] : map.coordinate_to_tile()) {
        const auto& tile =
            map.tiles().at(static_cast<std::size_t>(tile_index));
        if (tile.kind == TileKind::water) {
            continue;
        }
        if (tile.kind == TileKind::land) {
            const auto [x, y] = layout.tile_position(coordinate);
            if (tile.resource.has_value()) {
                const int number = tile.number.value();
                const float probability =
                    static_cast<float>(6 - std::abs(7 - number)) / 36.0F;
                const int channel =
                    2 * num_players + static_cast<int>(*tile.resource);
                for (int x_delta : {0, 2, 4}) {
                    for (int y_delta : {0, 2}) {
                        output[board_index(
                            x + x_delta,
                            y + y_delta,
                            channel)] += probability;
                    }
                }
            }

            if (coordinate == game.board().robber_coordinate()) {
                const int channel = 2 * num_players + 5;
                for (int x_delta : {0, 2, 4}) {
                    for (int y_delta : {0, 2}) {
                        output[board_index(
                            x + x_delta,
                            y + y_delta,
                            channel)] = 1.0F;
                    }
                }
            }
        }

        if (tile.kind == TileKind::port) {
            constexpr std::array<std::array<int, 2>, 6> kPortNodeIndices{{
                {{2, 1}},
                {{3, 2}},
                {{4, 3}},
                {{5, 4}},
                {{0, 5}},
                {{1, 0}},
            }};
            const int direction = static_cast<int>(tile.port_direction.value());
            const int resource =
                tile.resource.has_value() ? static_cast<int>(*tile.resource) : 5;
            const int channel = 2 * num_players + 6 + resource;
            for (int node_index :
                 kPortNodeIndices[static_cast<std::size_t>(direction)]) {
                const auto [node_x, node_y] =
                    layout.node_position(
                        tile.nodes[static_cast<std::size_t>(node_index)]);
                output[board_index(node_x, node_y, channel)] = 1.0F;
            }
        }
    }
}

double production_sum(const Game& game, Color color) {
    double total = 0.0;
    const auto& board = game.board();
    const auto& map = board.map();
    for (int tile_index : map.land_tile_indices()) {
        const auto& tile =
            map.tiles().at(static_cast<std::size_t>(tile_index));
        const auto coordinate = std::find_if(
            map.coordinate_to_tile().begin(),
            map.coordinate_to_tile().end(),
            [tile_index](const auto& item) { return item.second == tile_index; });
        if (!tile.resource.has_value() || !tile.number.has_value() ||
            coordinate == map.coordinate_to_tile().end() ||
            coordinate->first == board.robber_coordinate()) {
            continue;
        }
        const double probability =
            static_cast<double>(6 - std::abs(7 - *tile.number)) / 36.0;
        for (int node : tile.nodes) {
            const auto building = board.buildings().find(node);
            if (building == board.buildings().end() ||
                building->second.color != color) {
                continue;
            }
            total += probability *
                     (building->second.building == Building::settlement ? 1.0
                                                                        : 2.0);
        }
    }
    return total;
}

}  // namespace cppanatron
