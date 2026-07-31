#include "cppanatron/c_api.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cppanatron/action_space.hpp"
#include "cppanatron/batch.hpp"
#include "cppanatron/game.hpp"
#include "cppanatron/mcts.hpp"
#include "cppanatron/value_player.hpp"

using cppanatron::Color;
using cppanatron::BatchConfig;
using cppanatron::BatchBuffers;
using cppanatron::FlatActionSpace;
using cppanatron::Game;
using cppanatron::GameBatch;
using cppanatron::MapType;
using cppanatron::MCTSSearch;
using cppanatron::NumberPlacement;
using cppanatron::ObservationLayout;
using cppanatron::RewardFunction;

struct cppanatron_game {
    int num_players{};
    MapType map_type{};
    NumberPlacement number_placement{};
    int discard_limit{};
    bool friendly_robber{};
    int victory_points_to_win{};
    std::vector<Color> colors;
    FlatActionSpace action_space;
    std::unique_ptr<Game> game;

    cppanatron_game(
        int players,
        MapType map,
        NumberPlacement placement,
        std::uint64_t map_seed,
        std::uint64_t game_seed,
        int discard,
        bool friendly,
        int victory_points)
        : num_players(players),
          map_type(map),
          number_placement(placement),
          discard_limit(discard),
          friendly_robber(friendly),
          victory_points_to_win(victory_points),
          colors(cppanatron::kColors.begin(), cppanatron::kColors.begin() + players),
          action_space(players, map),
          game(std::make_unique<Game>(
              colors,
              map,
              game_seed,
              discard,
              friendly,
              victory_points,
              placement,
              map_seed)) {}

    void reset(std::uint64_t map_seed, std::uint64_t game_seed) {
        game = std::make_unique<Game>(
            colors,
            map_type,
            game_seed,
            discard_limit,
            friendly_robber,
            victory_points_to_win,
            number_placement,
            map_seed);
    }
};

struct cppanatron_batch {
    GameBatch batch;

    cppanatron_batch(BatchConfig config, ObservationLayout layout)
        : batch(config, std::move(layout)) {}
};

struct cppanatron_search {
    ObservationLayout observation_layout;
    MCTSSearch search;
    std::size_t observation_size{};

    cppanatron_search(
        const cppanatron_game& game,
        double c_puct,
        std::uint64_t search_seed,
        ObservationLayout layout)
        : observation_layout(std::move(layout)),
          search(*game.game, game.action_space, c_puct, search_seed),
          observation_size(cppanatron::full_observation_size(
              game.num_players,
              observation_layout)) {}
};

namespace {

thread_local std::string last_error;

MapType parse_map_type(int value) {
    switch (value) {
        case CPPANATRON_MAP_BASE:
            return MapType::base;
        case CPPANATRON_MAP_MINI:
            return MapType::mini;
        case CPPANATRON_MAP_TOURNAMENT:
            return MapType::tournament;
        default:
            throw std::invalid_argument("invalid map type");
    }
}

NumberPlacement parse_number_placement(int value) {
    switch (value) {
        case CPPANATRON_NUMBER_PLACEMENT_OFFICIAL_SPIRAL:
            return NumberPlacement::official_spiral;
        case CPPANATRON_NUMBER_PLACEMENT_RANDOM:
            return NumberPlacement::random;
        default:
            throw std::invalid_argument("invalid number placement");
    }
}

RewardFunction parse_reward_function(int value) {
    switch (value) {
        case 0:
            return RewardFunction::shaped;
        case 1:
            return RewardFunction::win;
        default:
            throw std::invalid_argument("invalid reward function");
    }
}

ObservationLayout make_observation_layout(
    int board_width,
    int board_height,
    const cppanatron_node_position* node_positions,
    std::size_t node_position_count,
    const cppanatron_edge_position* edge_positions,
    std::size_t edge_position_count,
    const cppanatron_tile_position* tile_positions,
    std::size_t tile_position_count) {
    if ((node_position_count > 0 && node_positions == nullptr) ||
        (edge_position_count > 0 && edge_positions == nullptr) ||
        (tile_position_count > 0 && tile_positions == nullptr)) {
        throw std::invalid_argument("null observation position array");
    }
    std::vector<cppanatron::NodePosition> nodes;
    nodes.reserve(node_position_count);
    for (std::size_t index = 0; index < node_position_count; ++index) {
        nodes.push_back({
            node_positions[index].node,
            node_positions[index].x,
            node_positions[index].y,
        });
    }
    std::vector<cppanatron::EdgePosition> edges;
    edges.reserve(edge_position_count);
    for (std::size_t index = 0; index < edge_position_count; ++index) {
        edges.push_back({
            {
                edge_positions[index].a,
                edge_positions[index].b,
            },
            edge_positions[index].x,
            edge_positions[index].y,
        });
    }
    std::vector<cppanatron::TilePosition> tiles;
    tiles.reserve(tile_position_count);
    for (std::size_t index = 0; index < tile_position_count; ++index) {
        tiles.push_back({
            {
                tile_positions[index].x,
                tile_positions[index].y,
                tile_positions[index].z,
            },
            tile_positions[index].board_x,
            tile_positions[index].board_y,
        });
    }
    return {
        board_width,
        board_height,
        std::move(nodes),
        std::move(edges),
        std::move(tiles),
    };
}

template <typename Callable>
int guard(Callable&& callable) noexcept {
    try {
        callable();
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    } catch (...) {
        last_error = "unknown C++ exception";
        return -1;
    }
}

template <typename Callable>
int guarded_value(Callable&& callable, int error_value = -1) noexcept {
    try {
        const int result = callable();
        last_error.clear();
        return result;
    } catch (const std::exception& error) {
        last_error = error.what();
        return error_value;
    } catch (...) {
        last_error = "unknown C++ exception";
        return error_value;
    }
}

}  // namespace

extern "C" {

const char* cppanatron_version(void) {
    return "0.3.0";
}

const char* cppanatron_last_error(void) {
    return last_error.c_str();
}

cppanatron_game* cppanatron_game_create(
    int32_t num_players,
    int32_t map_type,
    uint64_t seed,
    int32_t discard_limit,
    int32_t friendly_robber,
    int32_t victory_points_to_win) {
    try {
        if (num_players < 1 || num_players > 4) {
            throw std::invalid_argument("num_players must be between one and four");
        }
        auto* result = new cppanatron_game(
            num_players,
            parse_map_type(map_type),
            NumberPlacement::official_spiral,
            seed,
            seed,
            discard_limit,
            friendly_robber != 0,
            victory_points_to_win);
        last_error.clear();
        return result;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "unknown C++ exception";
        return nullptr;
    }
}

cppanatron_game* cppanatron_game_create_seeded(
    int32_t num_players,
    int32_t map_type,
    uint64_t map_seed,
    uint64_t game_seed,
    int32_t discard_limit,
    int32_t friendly_robber,
    int32_t victory_points_to_win) {
    try {
        if (num_players < 1 || num_players > 4) {
            throw std::invalid_argument("num_players must be between one and four");
        }
        auto* result = new cppanatron_game(
            num_players,
            parse_map_type(map_type),
            NumberPlacement::official_spiral,
            map_seed,
            game_seed,
            discard_limit,
            friendly_robber != 0,
            victory_points_to_win);
        last_error.clear();
        return result;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "unknown C++ exception";
        return nullptr;
    }
}

cppanatron_game* cppanatron_game_create_seeded_with_number_placement(
    int32_t num_players,
    int32_t map_type,
    uint64_t map_seed,
    uint64_t game_seed,
    int32_t discard_limit,
    int32_t friendly_robber,
    int32_t victory_points_to_win,
    int32_t number_placement) {
    try {
        if (num_players < 1 || num_players > 4) {
            throw std::invalid_argument("num_players must be between one and four");
        }
        auto* result = new cppanatron_game(
            num_players,
            parse_map_type(map_type),
            parse_number_placement(number_placement),
            map_seed,
            game_seed,
            discard_limit,
            friendly_robber != 0,
            victory_points_to_win);
        last_error.clear();
        return result;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "unknown C++ exception";
        return nullptr;
    }
}

void cppanatron_game_destroy(cppanatron_game* handle) {
    delete handle;
}

int32_t cppanatron_game_reset(cppanatron_game* handle, uint64_t seed) {
    return guard([&] {
        if (handle == nullptr) {
            throw std::invalid_argument("null game handle");
        }
        handle->reset(seed, seed);
    });
}

int32_t cppanatron_game_reset_seeded(
    cppanatron_game* handle,
    uint64_t map_seed,
    uint64_t game_seed) {
    return guard([&] {
        if (handle == nullptr) {
            throw std::invalid_argument("null game handle");
        }
        handle->reset(map_seed, game_seed);
    });
}

int32_t cppanatron_game_action_space_size(const cppanatron_game* handle) {
    return guarded_value([&] {
        if (handle == nullptr) {
            throw std::invalid_argument("null game handle");
        }
        return static_cast<int>(handle->action_space.size());
    });
}

int32_t cppanatron_game_valid_action_mask(
    const cppanatron_game* handle,
    uint8_t* mask,
    size_t mask_size) {
    return guard([&] {
        if (handle == nullptr || mask == nullptr) {
            throw std::invalid_argument("null mask or game handle");
        }
        if (mask_size != handle->action_space.size()) {
            throw std::invalid_argument("action mask has incorrect size");
        }
        std::memset(mask, 0, mask_size);
        for (const auto& action : handle->game->playable_actions()) {
            mask[handle->action_space.index(action, handle->colors)] = 1;
        }
    });
}

int32_t cppanatron_game_step(cppanatron_game* handle, int32_t flat_action) {
    return guard([&] {
        if (handle == nullptr) {
            throw std::invalid_argument("null game handle");
        }
        if (flat_action < 0 ||
            static_cast<std::size_t>(flat_action) >= handle->action_space.size()) {
            throw std::out_of_range("flat action is out of range");
        }
        const auto action = handle->action_space.decode(
            static_cast<std::size_t>(flat_action),
            handle->game->current_color(),
            handle->colors);
        handle->game->execute(action);
    });
}

int32_t cppanatron_game_step_replay(
    cppanatron_game* handle,
    int32_t flat_action,
    int32_t die_one,
    int32_t die_two,
    int32_t development_card,
    int32_t stolen_resource) {
    return guard([&] {
        if (handle == nullptr) {
            throw std::invalid_argument("null game handle");
        }
        if (flat_action < 0 ||
            static_cast<std::size_t>(flat_action) >= handle->action_space.size()) {
            throw std::out_of_range("flat action is out of range");
        }
        std::optional<cppanatron::Dice> dice;
        if (die_one >= 0 || die_two >= 0) {
            if (die_one < 1 || die_one > 6 || die_two < 1 || die_two > 6) {
                throw std::invalid_argument("replay dice must both be in [1, 6]");
            }
            dice = cppanatron::Dice{die_one, die_two};
        }
        std::optional<cppanatron::DevelopmentCard> card;
        if (development_card >= 0) {
            if (development_card > 4) {
                throw std::invalid_argument("invalid replay development card");
            }
            card = static_cast<cppanatron::DevelopmentCard>(development_card);
        }
        std::optional<cppanatron::Resource> resource;
        if (stolen_resource >= 0) {
            if (stolen_resource > 4) {
                throw std::invalid_argument("invalid replay stolen resource");
            }
            resource = static_cast<cppanatron::Resource>(stolen_resource);
        }
        const auto action = handle->action_space.decode(
            static_cast<std::size_t>(flat_action),
            handle->game->current_color(),
            handle->colors);
        handle->game->execute(action, dice, card, resource);
    });
}

int32_t cppanatron_game_value_action(const cppanatron_game* handle) {
    return guarded_value([&] {
        if (handle == nullptr) {
            throw std::invalid_argument("null game handle");
        }
        const auto action = cppanatron::value_action(*handle->game);
        return static_cast<int>(
            handle->action_space.index(action, handle->colors));
    });
}

int32_t cppanatron_game_value_score(
    const cppanatron_game* handle,
    int32_t player,
    double* output) {
    return guard([&] {
        if (handle == nullptr || output == nullptr) {
            throw std::invalid_argument("null output or game handle");
        }
        if (player < 0 || player >= handle->num_players) {
            throw std::out_of_range("player index is out of range");
        }
        *output = cppanatron::value_score(
            *handle->game,
            handle->colors[static_cast<std::size_t>(player)]);
    });
}

int32_t cppanatron_game_num_players(const cppanatron_game* handle) {
    return handle == nullptr ? -1 : handle->num_players;
}

int32_t cppanatron_game_current_player(const cppanatron_game* handle) {
    return handle == nullptr ? -1 : handle->game->current_player_index();
}

int32_t cppanatron_game_current_prompt(const cppanatron_game* handle) {
    return handle == nullptr ? -1 : static_cast<int>(handle->game->current_prompt());
}

int32_t cppanatron_game_num_turns(const cppanatron_game* handle) {
    return handle == nullptr ? -1 : handle->game->num_turns();
}

int32_t cppanatron_game_winner(const cppanatron_game* handle) {
    if (handle == nullptr) {
        return -2;
    }
    const auto winner = handle->game->winning_color();
    if (!winner.has_value()) {
        return -1;
    }
    return static_cast<int>(std::distance(
        handle->colors.begin(),
        std::find(handle->colors.begin(), handle->colors.end(), *winner)));
}

int32_t cppanatron_game_flags(
    const cppanatron_game* handle,
    int32_t output[7]) {
    return guard([&] {
        if (handle == nullptr || output == nullptr) {
            throw std::invalid_argument("null output or game handle");
        }
        output[0] = handle->game->is_initial_build_phase();
        output[1] = handle->game->is_discarding();
        output[2] = handle->game->is_moving_robber();
        output[3] = handle->game->is_road_building();
        output[4] = handle->game->current_player_index();
        output[5] = handle->game->current_turn_index();
        output[6] = handle->game->completed_turns();
    });
}

int32_t cppanatron_game_robber_coordinate(
    const cppanatron_game* handle,
    int32_t output[3]) {
    return guard([&] {
        if (handle == nullptr || output == nullptr) {
            throw std::invalid_argument("null output or game handle");
        }
        const auto coordinate = handle->game->board().robber_coordinate();
        output[0] = coordinate.x;
        output[1] = coordinate.y;
        output[2] = coordinate.z;
    });
}

int32_t cppanatron_game_development_cards_remaining(const cppanatron_game* handle) {
    return handle == nullptr
               ? -1
               : static_cast<int>(handle->game->development_cards_remaining());
}

int32_t cppanatron_game_resource_bank(
    const cppanatron_game* handle,
    int32_t output[5]) {
    return guard([&] {
        if (handle == nullptr || output == nullptr) {
            throw std::invalid_argument("null output or game handle");
        }
        std::copy(
            handle->game->resource_bank().begin(),
            handle->game->resource_bank().end(),
            output);
    });
}

int32_t cppanatron_game_player_state(
    const cppanatron_game* handle,
    int32_t player,
    cppanatron_player_state* output) {
    return guard([&] {
        if (handle == nullptr || output == nullptr) {
            throw std::invalid_argument("null output or game handle");
        }
        const auto& state =
            handle->game->players().at(static_cast<std::size_t>(player));
        *output = {};
        output->victory_points = state.victory_points;
        output->actual_victory_points = state.actual_victory_points;
        output->roads_available = state.roads_available;
        output->settlements_available = state.settlements_available;
        output->cities_available = state.cities_available;
        output->has_road = state.has_road;
        output->has_army = state.has_army;
        output->has_rolled = state.has_rolled;
        output->has_played_development_card_in_turn =
            state.has_played_development_card_in_turn;
        output->longest_road_length = state.longest_road_length;
        std::copy(state.resources.begin(), state.resources.end(), output->resources);
        std::copy(
            state.development_cards.begin(),
            state.development_cards.end(),
            output->development_cards);
        std::copy(
            state.played_development_cards.begin(),
            state.played_development_cards.end(),
            output->played_development_cards);
        std::copy(
            state.development_card_owned_at_start.begin(),
            state.development_card_owned_at_start.end(),
            output->development_card_owned_at_start);
        output->turns_since_last_knight =
            state.last_knight_completed_turn < 0
                ? -1
                : handle->game->completed_turns() -
                      state.last_knight_completed_turn;
        output->turns_since_last_development_card_bought =
            state.last_development_card_bought_completed_turn < 0
                ? -1
                : handle->game->completed_turns() -
                      state.last_development_card_bought_completed_turn;
    });
}

int32_t cppanatron_game_buildings(
    const cppanatron_game* handle,
    cppanatron_building* output,
    size_t capacity) {
    return guarded_value([&] {
        if (handle == nullptr) {
            throw std::invalid_argument("null game handle");
        }
        const auto& buildings = handle->game->board().buildings();
        if (output == nullptr) {
            return static_cast<int>(buildings.size());
        }
        if (capacity < buildings.size()) {
            throw std::invalid_argument("building output capacity is too small");
        }
        std::size_t i = 0;
        for (const auto& [node, building] : buildings) {
            output[i++] = {
                node,
                static_cast<int32_t>(building.color),
                static_cast<int32_t>(building.building)};
        }
        return static_cast<int>(buildings.size());
    });
}

int32_t cppanatron_game_roads(
    const cppanatron_game* handle,
    cppanatron_road* output,
    size_t capacity) {
    return guarded_value([&] {
        if (handle == nullptr) {
            throw std::invalid_argument("null game handle");
        }
        const auto& roads = handle->game->board().roads();
        if (output == nullptr) {
            return static_cast<int>(roads.size());
        }
        if (capacity < roads.size()) {
            throw std::invalid_argument("road output capacity is too small");
        }
        std::size_t i = 0;
        for (const auto& [edge, color] : roads) {
            output[i++] = {edge.a, edge.b, static_cast<int32_t>(color)};
        }
        return static_cast<int>(roads.size());
    });
}

int32_t cppanatron_game_tiles(
    const cppanatron_game* handle,
    cppanatron_tile* output,
    size_t capacity) {
    return guarded_value([&] {
        if (handle == nullptr) {
            throw std::invalid_argument("null game handle");
        }
        const auto& map = handle->game->board().map();
        const auto& tiles = map.tiles();
        if (output == nullptr) {
            return static_cast<int>(tiles.size());
        }
        if (capacity < tiles.size()) {
            throw std::invalid_argument("tile output capacity is too small");
        }
        for (const auto& [coordinate, tile_index] : map.coordinate_to_tile()) {
            const auto& tile = tiles[static_cast<std::size_t>(tile_index)];
            auto& target = output[static_cast<std::size_t>(tile_index)];
            target = {};
            target.x = coordinate.x;
            target.y = coordinate.y;
            target.z = coordinate.z;
            target.id = tile.id;
            target.kind = static_cast<int32_t>(tile.kind);
            target.resource =
                tile.resource.has_value() ? static_cast<int32_t>(*tile.resource) : -1;
            target.number = tile.number.value_or(-1);
            target.port_direction =
                tile.port_direction.has_value()
                    ? static_cast<int32_t>(*tile.port_direction)
                    : -1;
            std::copy(tile.nodes.begin(), tile.nodes.end(), target.nodes);
        }
        return static_cast<int>(tiles.size());
    });
}

cppanatron_search* cppanatron_search_create(
    const cppanatron_game* game,
    double c_puct,
    uint64_t search_seed,
    int32_t board_width,
    int32_t board_height,
    const cppanatron_node_position* node_positions,
    size_t node_position_count,
    const cppanatron_edge_position* edge_positions,
    size_t edge_position_count,
    const cppanatron_tile_position* tile_positions,
    size_t tile_position_count) {
    try {
        if (game == nullptr) {
            throw std::invalid_argument("null game handle");
        }
        auto* result = new cppanatron_search(
            *game,
            c_puct,
            search_seed,
            make_observation_layout(
                board_width,
                board_height,
                node_positions,
                node_position_count,
                edge_positions,
                edge_position_count,
                tile_positions,
                tile_position_count));
        last_error.clear();
        return result;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "unknown C++ exception";
        return nullptr;
    }
}

void cppanatron_search_destroy(cppanatron_search* handle) {
    delete handle;
}

int32_t cppanatron_search_initialize_root(
    cppanatron_search* handle,
    const float* policy_logits,
    size_t policy_size) {
    return guard([&] {
        if (handle == nullptr || policy_logits == nullptr) {
            throw std::invalid_argument("null search handle or policy logits");
        }
        handle->search.initialize_root({policy_logits, policy_size});
    });
}

int32_t cppanatron_search_add_root_dirichlet_noise(
    cppanatron_search* handle,
    double alpha,
    double fraction) {
    return guard([&] {
        if (handle == nullptr) {
            throw std::invalid_argument("null search handle");
        }
        handle->search.add_root_dirichlet_noise(alpha, fraction);
    });
}

int32_t cppanatron_search_root_observation(
    const cppanatron_search* handle,
    float* observation,
    size_t observation_size,
    int32_t* player) {
    return guard([&] {
        if (handle == nullptr || observation == nullptr || player == nullptr) {
            throw std::invalid_argument("null search handle or root output");
        }
        if (observation_size != handle->observation_size) {
            throw std::invalid_argument("root observation has incorrect size");
        }
        const Game& root = handle->search.root_game();
        *player = root.current_player_index();
        cppanatron::write_full_observation(
            root,
            *player,
            handle->observation_layout,
            observation,
            observation_size);
    });
}

int32_t cppanatron_search_select_leaf(
    cppanatron_search* handle,
    float* observation,
    size_t observation_size,
    int32_t* player) {
    return guarded_value([&] {
        if (handle == nullptr || observation == nullptr || player == nullptr) {
            throw std::invalid_argument("null search handle or leaf output");
        }
        if (observation_size != handle->observation_size) {
            throw std::invalid_argument("leaf observation has incorrect size");
        }
        const Game* leaf = handle->search.select_leaf();
        if (leaf == nullptr) {
            return 0;
        }
        *player = handle->search.pending_player_index();
        cppanatron::write_full_observation(
            *leaf,
            *player,
            handle->observation_layout,
            observation,
            observation_size);
        return 1;
    });
}

int32_t cppanatron_search_evaluate_leaf(
    cppanatron_search* handle,
    const float* policy_logits,
    size_t policy_size,
    double value) {
    return guard([&] {
        if (handle == nullptr || policy_logits == nullptr) {
            throw std::invalid_argument("null search handle or policy logits");
        }
        handle->search.evaluate_leaf({policy_logits, policy_size}, value);
    });
}

int32_t cppanatron_search_root_visits(
    const cppanatron_search* handle,
    uint32_t* visits,
    size_t visit_count) {
    return guard([&] {
        if (handle == nullptr || visits == nullptr) {
            throw std::invalid_argument("null search handle or visit output");
        }
        const auto root_visits = handle->search.root_visits();
        if (visit_count != root_visits.size()) {
            throw std::invalid_argument("root visit output has incorrect size");
        }
        std::copy(root_visits.begin(), root_visits.end(), visits);
    });
}

cppanatron_batch* cppanatron_batch_create(
    int32_t num_envs,
    int32_t num_players,
    int32_t map_type,
    int32_t discard_limit,
    int32_t friendly_robber,
    int32_t victory_points_to_win,
    int32_t number_placement,
    int32_t reward_function,
    int32_t turns_limit,
    int32_t board_width,
    int32_t board_height,
    const cppanatron_node_position* node_positions,
    size_t node_position_count,
    const cppanatron_edge_position* edge_positions,
    size_t edge_position_count,
    const cppanatron_tile_position* tile_positions,
    size_t tile_position_count) {
    try {
        auto* result = new cppanatron_batch(
            BatchConfig{
                num_envs,
                num_players,
                parse_map_type(map_type),
                parse_number_placement(number_placement),
                discard_limit,
                friendly_robber != 0,
                victory_points_to_win,
                parse_reward_function(reward_function),
                turns_limit,
            },
            make_observation_layout(
                board_width,
                board_height,
                node_positions,
                node_position_count,
                edge_positions,
                edge_position_count,
                tile_positions,
                tile_position_count));
        last_error.clear();
        return result;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "unknown C++ exception";
        return nullptr;
    }
}

void cppanatron_batch_destroy(cppanatron_batch* handle) {
    delete handle;
}

int32_t cppanatron_batch_bind_buffers(
    cppanatron_batch* handle,
    uint8_t* observations,
    size_t observation_row_stride,
    size_t action_mask_offset,
    size_t observation_offset,
    int32_t* actions,
    float* rewards,
    uint8_t* terminals,
    uint8_t* truncations,
    uint8_t* masks) {
    return guard([&] {
        if (handle == nullptr) {
            throw std::invalid_argument("null batch handle");
        }
        handle->batch.bind(BatchBuffers{
            observations,
            observation_row_stride,
            action_mask_offset,
            observation_offset,
            actions,
            rewards,
            terminals,
            truncations,
            masks,
        });
    });
}

int32_t cppanatron_batch_reset_all(
    cppanatron_batch* handle,
    const uint64_t* map_seeds,
    const uint64_t* game_seeds,
    size_t seed_count) {
    return guard([&] {
        if (handle == nullptr) {
            throw std::invalid_argument("null batch handle");
        }
        handle->batch.reset_all(map_seeds, game_seeds, seed_count);
    });
}

int32_t cppanatron_batch_reset_at(
    cppanatron_batch* handle,
    int32_t env_index,
    uint64_t map_seed,
    uint64_t game_seed,
    int32_t preserve_transition) {
    return guard([&] {
        if (handle == nullptr) {
            throw std::invalid_argument("null batch handle");
        }
        handle->batch.reset_at(
            env_index,
            map_seed,
            game_seed,
            preserve_transition != 0);
    });
}

int32_t cppanatron_batch_step(cppanatron_batch* handle) {
    return guard([&] {
        if (handle == nullptr) {
            throw std::invalid_argument("null batch handle");
        }
        handle->batch.step();
    });
}

}  // extern "C"
