#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cppanatron/action_space.hpp"
#include "cppanatron/game.hpp"
#include "cppanatron/observation.hpp"

namespace cppanatron {

enum class RewardFunction {
    shaped = 0,
    win = 1,
};

struct BatchConfig {
    int num_envs{};
    int num_players{};
    MapType map_type{};
    NumberPlacement number_placement{};
    int discard_limit{};
    bool friendly_robber{};
    int victory_points_to_win{};
    RewardFunction reward_function{};
    int turns_limit{};
};

struct BatchBuffers {
    std::uint8_t* observations{};
    std::size_t observation_row_stride{};
    std::size_t action_mask_offset{};
    std::size_t observation_offset{};
    std::int32_t* actions{};
    float* rewards{};
    std::uint8_t* terminals{};
    std::uint8_t* truncations{};
    std::uint8_t* masks{};
};

class GameBatch {
public:
    GameBatch(BatchConfig config, ObservationLayout observation_layout);

    void bind(BatchBuffers buffers);
    void reset_all(
        const std::uint64_t* map_seeds,
        const std::uint64_t* game_seeds,
        std::size_t seed_count);
    void reset_at(
        int env_index,
        std::uint64_t map_seed,
        std::uint64_t game_seed,
        bool preserve_transition);
    void step();

private:
    struct Environment {
        std::unique_ptr<Game> game;
        std::vector<int> previous_victory_points;
        std::vector<double> previous_production;
    };

    [[nodiscard]] std::size_t row_index(int env_index, int player) const;
    [[nodiscard]] std::uint8_t* row(int env_index, int player) const;
    void create_game(
        Environment& environment,
        std::uint64_t map_seed,
        std::uint64_t game_seed);
    void write_environment(int env_index);
    void write_action_masks(int env_index);
    void clear_transition(int env_index);
    void validate_bound() const;

    BatchConfig config_;
    ObservationLayout observation_layout_;
    std::vector<Color> colors_;
    FlatActionSpace action_space_;
    std::vector<Environment> environments_;
    BatchBuffers buffers_;
};

}  // namespace cppanatron
