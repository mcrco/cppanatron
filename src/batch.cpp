#include "cppanatron/batch.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace cppanatron {

GameBatch::GameBatch(
    BatchConfig config,
    ObservationLayout observation_layout)
    : config_(config),
      observation_layout_(std::move(observation_layout)),
      colors_(kColors.begin(), kColors.begin() + config.num_players),
      action_space_(config.num_players, config.map_type),
      environments_(static_cast<std::size_t>(config.num_envs)) {
    if (config.num_envs <= 0) {
        throw std::invalid_argument("num_envs must be positive");
    }
    if (config.num_players < 2 || config.num_players > 4) {
        throw std::invalid_argument("num_players must be between two and four");
    }
    if (config.turns_limit <= 0) {
        throw std::invalid_argument("turns_limit must be positive");
    }
}

void GameBatch::bind(BatchBuffers buffers) {
    if (buffers.observations == nullptr || buffers.actions == nullptr ||
        buffers.rewards == nullptr || buffers.terminals == nullptr ||
        buffers.truncations == nullptr || buffers.masks == nullptr) {
        throw std::invalid_argument("batch buffer pointer is null");
    }
    if (buffers.observation_row_stride == 0 ||
        buffers.observation_offset % alignof(float) != 0) {
        throw std::invalid_argument("invalid observation buffer layout");
    }
    buffers_ = buffers;
}

void GameBatch::validate_bound() const {
    if (buffers_.observations == nullptr) {
        throw std::logic_error("batch buffers have not been bound");
    }
}

std::size_t GameBatch::row_index(int env_index, int player) const {
    return static_cast<std::size_t>(
        env_index * config_.num_players + player);
}

std::uint8_t* GameBatch::row(int env_index, int player) const {
    return buffers_.observations +
           row_index(env_index, player) * buffers_.observation_row_stride;
}

void GameBatch::create_game(
    Environment& environment,
    std::uint64_t map_seed,
    std::uint64_t game_seed) {
    environment.game = std::make_unique<Game>(
        colors_,
        config_.map_type,
        game_seed,
        config_.discard_limit,
        config_.friendly_robber,
        config_.victory_points_to_win,
        config_.number_placement,
        map_seed);
    environment.previous_victory_points.assign(
        static_cast<std::size_t>(config_.num_players),
        0);
    environment.previous_production.assign(
        static_cast<std::size_t>(config_.num_players),
        0.0);
}

void GameBatch::clear_transition(int env_index) {
    for (int player = 0; player < config_.num_players; ++player) {
        const auto index = row_index(env_index, player);
        buffers_.rewards[index] = 0.0F;
        buffers_.terminals[index] = 0;
        buffers_.truncations[index] = 0;
        buffers_.masks[index] = 1;
    }
}

void GameBatch::write_action_masks(int env_index) {
    const auto& environment =
        environments_.at(static_cast<std::size_t>(env_index));
    const int current_player = environment.game->current_player_index();
    for (int player = 0; player < config_.num_players; ++player) {
        auto* action_mask = row(env_index, player) +
                            buffers_.action_mask_offset;
        std::memset(action_mask, 0, action_space_.size());
        if (player != current_player) {
            action_mask[action_space_.size() - 1] = 1;
            continue;
        }
        for (const auto& action : environment.game->playable_actions()) {
            action_mask[action_space_.index(action, colors_)] = 1;
        }
    }
}

void GameBatch::write_environment(int env_index) {
    const auto& environment =
        environments_.at(static_cast<std::size_t>(env_index));
    for (int player = 0; player < config_.num_players; ++player) {
        auto* observation = reinterpret_cast<float*>(
            row(env_index, player) + buffers_.observation_offset);
        write_full_observation(
            *environment.game,
            player,
            observation_layout_,
            observation,
            full_observation_size(
                config_.num_players,
                observation_layout_));
    }
    write_action_masks(env_index);
}

void GameBatch::reset_all(
    const std::uint64_t* map_seeds,
    const std::uint64_t* game_seeds,
    std::size_t seed_count) {
    validate_bound();
    if (map_seeds == nullptr || game_seeds == nullptr ||
        seed_count != environments_.size()) {
        throw std::invalid_argument("batch reset seed array has incorrect size");
    }
    for (int env_index = 0; env_index < config_.num_envs; ++env_index) {
        create_game(
            environments_[static_cast<std::size_t>(env_index)],
            map_seeds[env_index],
            game_seeds[env_index]);
        clear_transition(env_index);
        write_environment(env_index);
    }
}

void GameBatch::reset_at(
    int env_index,
    std::uint64_t map_seed,
    std::uint64_t game_seed,
    bool preserve_transition) {
    validate_bound();
    if (env_index < 0 || env_index >= config_.num_envs) {
        throw std::out_of_range("batch environment index is out of range");
    }
    create_game(
        environments_[static_cast<std::size_t>(env_index)],
        map_seed,
        game_seed);
    if (!preserve_transition) {
        clear_transition(env_index);
    } else {
        for (int player = 0; player < config_.num_players; ++player) {
            buffers_.masks[row_index(env_index, player)] = 1;
        }
    }
    write_environment(env_index);
}

void GameBatch::step() {
    validate_bound();
    for (int env_index = 0; env_index < config_.num_envs; ++env_index) {
        auto& environment =
            environments_.at(static_cast<std::size_t>(env_index));
        if (environment.game == nullptr) {
            throw std::logic_error("batch step called before reset");
        }
        clear_transition(env_index);
        const int current_player = environment.game->current_player_index();
        const auto action_index =
            buffers_.actions[row_index(env_index, current_player)];
        if (action_index < 0 ||
            static_cast<std::size_t>(action_index) >= action_space_.size()) {
            throw std::out_of_range(
                "action for environment " + std::to_string(env_index) +
                " is out of range");
        }
        const auto action = action_space_.decode(
            static_cast<std::size_t>(action_index),
            environment.game->current_color(),
            colors_);
        const auto valid = std::find(
            environment.game->playable_actions().begin(),
            environment.game->playable_actions().end(),
            action);
        if (valid == environment.game->playable_actions().end()) {
            throw std::invalid_argument(
                "invalid action for environment " +
                std::to_string(env_index));
        }
        environment.game->execute(action);

        const auto winner = environment.game->winning_color();
        for (int player = 0; player < config_.num_players; ++player) {
            const auto index = row_index(env_index, player);
            float reward = 0.0F;
            if (winner.has_value() &&
                *winner == colors_[static_cast<std::size_t>(player)]) {
                reward = 1.0F;
            } else if (
                config_.reward_function == RewardFunction::win &&
                winner.has_value()) {
                reward = -1.0F;
            } else if (config_.reward_function == RewardFunction::shaped) {
                const auto& state =
                    environment.game->players().at(
                        static_cast<std::size_t>(player));
                const double current_production = production_sum(
                    *environment.game,
                    colors_[static_cast<std::size_t>(player)]);
                reward = static_cast<float>(
                    0.01 *
                        static_cast<double>(
                            state.actual_victory_points -
                            environment.previous_victory_points[
                                static_cast<std::size_t>(player)]) /
                        static_cast<double>(config_.victory_points_to_win) +
                    0.0025 *
                        (current_production -
                         environment.previous_production[
                             static_cast<std::size_t>(player)]));
                environment.previous_victory_points[
                    static_cast<std::size_t>(player)] =
                    state.actual_victory_points;
                environment.previous_production[
                    static_cast<std::size_t>(player)] =
                    current_production;
            }
            buffers_.rewards[index] = reward;
            buffers_.terminals[index] = winner.has_value() ? 1 : 0;
            buffers_.truncations[index] =
                environment.game->num_turns() >= config_.turns_limit ? 1 : 0;
        }

        const bool done = winner.has_value() ||
                          environment.game->num_turns() >= config_.turns_limit;
        if (done) {
            for (int player = 0; player < config_.num_players; ++player) {
                const auto index = row_index(env_index, player);
                std::memset(
                    row(env_index, player),
                    0,
                    buffers_.observation_row_stride);
                buffers_.masks[index] = 0;
            }
        } else {
            write_environment(env_index);
        }
    }
}

}  // namespace cppanatron
