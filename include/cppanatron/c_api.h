#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define CPPANATRON_API __declspec(dllexport)
#else
#define CPPANATRON_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cppanatron_game cppanatron_game;

enum cppanatron_map_type {
    CPPANATRON_MAP_BASE = 0,
    CPPANATRON_MAP_MINI = 1,
    CPPANATRON_MAP_TOURNAMENT = 2,
};

typedef struct cppanatron_player_state {
    int32_t victory_points;
    int32_t actual_victory_points;
    int32_t roads_available;
    int32_t settlements_available;
    int32_t cities_available;
    int32_t has_road;
    int32_t has_army;
    int32_t has_rolled;
    int32_t has_played_development_card_in_turn;
    int32_t longest_road_length;
    int32_t resources[5];
    int32_t development_cards[5];
    int32_t played_development_cards[5];
    int32_t development_card_owned_at_start[4];
    int32_t turns_since_last_knight;
    int32_t turns_since_last_development_card_bought;
} cppanatron_player_state;

typedef struct cppanatron_building {
    int32_t node;
    int32_t color;
    int32_t building;
} cppanatron_building;

typedef struct cppanatron_road {
    int32_t a;
    int32_t b;
    int32_t color;
} cppanatron_road;

typedef struct cppanatron_tile {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t id;
    int32_t kind;
    int32_t resource;
    int32_t number;
    int32_t port_direction;
    int32_t nodes[6];
} cppanatron_tile;

CPPANATRON_API const char* cppanatron_version(void);
CPPANATRON_API const char* cppanatron_last_error(void);
CPPANATRON_API cppanatron_game* cppanatron_game_create(
    int32_t num_players,
    int32_t map_type,
    uint64_t seed,
    int32_t discard_limit,
    int32_t friendly_robber,
    int32_t victory_points_to_win);
CPPANATRON_API void cppanatron_game_destroy(cppanatron_game* handle);
CPPANATRON_API int32_t cppanatron_game_reset(cppanatron_game* handle, uint64_t seed);
CPPANATRON_API int32_t cppanatron_game_action_space_size(const cppanatron_game* handle);
CPPANATRON_API int32_t cppanatron_game_valid_action_mask(
    const cppanatron_game* handle,
    uint8_t* mask,
    size_t mask_size);
CPPANATRON_API int32_t cppanatron_game_step(cppanatron_game* handle, int32_t flat_action);
CPPANATRON_API int32_t cppanatron_game_step_replay(
    cppanatron_game* handle,
    int32_t flat_action,
    int32_t die_one,
    int32_t die_two,
    int32_t development_card,
    int32_t stolen_resource);
CPPANATRON_API int32_t cppanatron_game_value_action(
    const cppanatron_game* handle);
CPPANATRON_API int32_t cppanatron_game_value_score(
    const cppanatron_game* handle,
    int32_t player,
    double* output);
CPPANATRON_API int32_t cppanatron_game_num_players(const cppanatron_game* handle);
CPPANATRON_API int32_t cppanatron_game_current_player(const cppanatron_game* handle);
CPPANATRON_API int32_t cppanatron_game_current_prompt(const cppanatron_game* handle);
CPPANATRON_API int32_t cppanatron_game_num_turns(const cppanatron_game* handle);
CPPANATRON_API int32_t cppanatron_game_winner(const cppanatron_game* handle);
CPPANATRON_API int32_t cppanatron_game_flags(
    const cppanatron_game* handle,
    int32_t output[7]);
CPPANATRON_API int32_t cppanatron_game_robber_coordinate(
    const cppanatron_game* handle,
    int32_t output[3]);
CPPANATRON_API int32_t cppanatron_game_development_cards_remaining(
    const cppanatron_game* handle);
CPPANATRON_API int32_t cppanatron_game_resource_bank(
    const cppanatron_game* handle,
    int32_t output[5]);
CPPANATRON_API int32_t cppanatron_game_player_state(
    const cppanatron_game* handle,
    int32_t player,
    cppanatron_player_state* output);
CPPANATRON_API int32_t cppanatron_game_buildings(
    const cppanatron_game* handle,
    cppanatron_building* output,
    size_t capacity);
CPPANATRON_API int32_t cppanatron_game_roads(
    const cppanatron_game* handle,
    cppanatron_road* output,
    size_t capacity);
CPPANATRON_API int32_t cppanatron_game_tiles(
    const cppanatron_game* handle,
    cppanatron_tile* output,
    size_t capacity);

#ifdef __cplusplus
}
#endif
