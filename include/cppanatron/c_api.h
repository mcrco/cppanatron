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
typedef struct cppanatron_batch cppanatron_batch;
typedef struct cppanatron_search cppanatron_search;

typedef struct cppanatron_search_metrics {
    uint64_t simulations;
    uint32_t principal_variation_depth;
    uint32_t maximum_depth;
    double mean_depth;
    double root_value;
    uint32_t retained_root_visits;
    uint64_t pruned_actions;
    uint64_t coalesced_outcomes;
    int32_t tree_reused;
} cppanatron_search_metrics;

enum cppanatron_map_type {
    CPPANATRON_MAP_BASE = 0,
    CPPANATRON_MAP_MINI = 1,
    CPPANATRON_MAP_TOURNAMENT = 2,
};

enum cppanatron_number_placement {
    CPPANATRON_NUMBER_PLACEMENT_OFFICIAL_SPIRAL = 0,
    CPPANATRON_NUMBER_PLACEMENT_RANDOM = 1,
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

typedef struct cppanatron_node_position {
    int32_t node;
    int32_t x;
    int32_t y;
} cppanatron_node_position;

typedef struct cppanatron_edge_position {
    int32_t a;
    int32_t b;
    int32_t x;
    int32_t y;
} cppanatron_edge_position;

typedef struct cppanatron_tile_position {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t board_x;
    int32_t board_y;
} cppanatron_tile_position;

CPPANATRON_API const char* cppanatron_version(void);
CPPANATRON_API const char* cppanatron_last_error(void);
CPPANATRON_API cppanatron_game* cppanatron_game_create(
    int32_t num_players,
    int32_t map_type,
    uint64_t seed,
    int32_t discard_limit,
    int32_t friendly_robber,
    int32_t victory_points_to_win);
CPPANATRON_API cppanatron_game* cppanatron_game_create_seeded(
    int32_t num_players,
    int32_t map_type,
    uint64_t map_seed,
    uint64_t game_seed,
    int32_t discard_limit,
    int32_t friendly_robber,
    int32_t victory_points_to_win);
CPPANATRON_API cppanatron_game*
cppanatron_game_create_seeded_with_number_placement(
    int32_t num_players,
    int32_t map_type,
    uint64_t map_seed,
    uint64_t game_seed,
    int32_t discard_limit,
    int32_t friendly_robber,
    int32_t victory_points_to_win,
    int32_t number_placement);
CPPANATRON_API void cppanatron_game_destroy(cppanatron_game* handle);
CPPANATRON_API int32_t cppanatron_game_reset(cppanatron_game* handle, uint64_t seed);
CPPANATRON_API int32_t cppanatron_game_reset_seeded(
    cppanatron_game* handle,
    uint64_t map_seed,
    uint64_t game_seed);
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

CPPANATRON_API cppanatron_search* cppanatron_search_create(
    const cppanatron_game* game,
    double c_puct,
    uint64_t search_seed,
    int32_t canonical_pruning,
    int32_t board_width,
    int32_t board_height,
    const cppanatron_node_position* node_positions,
    size_t node_position_count,
    const cppanatron_edge_position* edge_positions,
    size_t edge_position_count,
    const cppanatron_tile_position* tile_positions,
    size_t tile_position_count);
CPPANATRON_API void cppanatron_search_destroy(cppanatron_search* handle);
CPPANATRON_API int32_t cppanatron_search_initialize_root(
    cppanatron_search* handle,
    const float* policy_logits,
    size_t policy_size);
CPPANATRON_API int32_t cppanatron_search_add_root_dirichlet_noise(
    cppanatron_search* handle,
    double alpha,
    double fraction);
CPPANATRON_API int32_t cppanatron_search_root_observation(
    const cppanatron_search* handle,
    float* observation,
    size_t observation_size,
    int32_t* player);
/**
 * Select one simulation leaf.
 *
 * Returns 1 and writes a full observation when neural evaluation is needed,
 * returns 0 when a terminal/dead-end simulation was completed internally, and
 * returns -1 on error.
 */
CPPANATRON_API int32_t cppanatron_search_select_leaf(
    cppanatron_search* handle,
    float* observation,
    size_t observation_size,
    int32_t* player);
CPPANATRON_API int32_t cppanatron_search_evaluate_leaf(
    cppanatron_search* handle,
    const float* policy_logits,
    size_t policy_size,
    double value);
CPPANATRON_API int32_t cppanatron_search_root_visits(
    const cppanatron_search* handle,
    uint32_t* visits,
    size_t visit_count);
CPPANATRON_API int32_t cppanatron_search_get_metrics(
    const cppanatron_search* handle,
    cppanatron_search_metrics* output);
CPPANATRON_API int32_t cppanatron_search_root_expanded(
    const cppanatron_search* handle);
CPPANATRON_API int32_t cppanatron_search_reset_metrics(
    cppanatron_search* handle);
/** Returns 1 when the tree was reused, 0 when it must be rebuilt, -1 on error. */
CPPANATRON_API int32_t cppanatron_search_advance(
    cppanatron_search* handle,
    size_t action_index);

CPPANATRON_API cppanatron_batch* cppanatron_batch_create(
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
    size_t tile_position_count);
CPPANATRON_API void cppanatron_batch_destroy(cppanatron_batch* handle);
CPPANATRON_API int32_t cppanatron_batch_bind_buffers(
    cppanatron_batch* handle,
    uint8_t* observations,
    size_t observation_row_stride,
    size_t action_mask_offset,
    size_t observation_offset,
    int32_t* actions,
    float* rewards,
    uint8_t* terminals,
    uint8_t* truncations,
    uint8_t* masks);
CPPANATRON_API int32_t cppanatron_batch_reset_all(
    cppanatron_batch* handle,
    const uint64_t* map_seeds,
    const uint64_t* game_seeds,
    size_t seed_count);
CPPANATRON_API int32_t cppanatron_batch_reset_at(
    cppanatron_batch* handle,
    int32_t env_index,
    uint64_t map_seed,
    uint64_t game_seed,
    int32_t preserve_transition);
CPPANATRON_API int32_t cppanatron_batch_step(cppanatron_batch* handle);

#ifdef __cplusplus
}
#endif
