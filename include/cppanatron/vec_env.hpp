#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "cppanatron/game.hpp"
#include "cppanatron/graph_encoder.hpp"
#include "cppanatron/value_player.hpp"

namespace cppanatron {

/** Who plays a seat: the learner (rows the caller acts for) or a built-in bot. */
enum class SeatKind : std::uint8_t {
    learner,
    /** Catanatron's ValueFunctionPlayer ("F"), see value_action. */
    value,
    /** Uniformly random among the playable actions. */
    random,
};

/** One kind of game a VecEnv plays; each reset draws a scenario by weight. */
struct Scenario {
    int num_players{2};
    GameRules rules{};
    /** Shared maps drawn uniformly per reset; when empty, a fresh map_type board. */
    std::vector<std::shared_ptr<const CatanMap>> maps;
    MapType map_type{MapType::base};
    NumberPlacement number_placement{NumberPlacement::official_spiral};
    /** One kind per seat in seating order; empty means every seat learns. */
    std::vector<SeatKind> seats;
    /** Shuffle which seat gets which kind at every reset. */
    bool shuffle_seats{false};
    /**
     * A game is truncated once num_turns reaches this. It must exceed the
     * 2 (num_players - 1) turns of the initial placement.
     */
    int turn_limit{500};
    double weight{1.0};
    ValueEnemies value_enemies{ValueEnemies::next};
};

struct VecEnvConfig {
    std::vector<Scenario> scenarios;
    int num_envs{1};
    EncoderDims dims{};
    InfoLevel level{InfoLevel::private_};
    std::uint64_t seed{0};
    int num_threads{1};
    /**
     * Optional shaping: each learner row also gets vp_reward times its change
     * in actual victory points / victory points to win. 0 is win/loss only.
     */
    float vp_reward{0.0F};
    /**
     * Encode the observations of learner seats that wait. Off by default:
     * waiting rows then hold zeros and the no-op mask, since networks only
     * act (and estimate every seat's value) from the acting row.
     */
    bool encode_waiting{false};
};

/**
 * Caller-owned buffers, rows = num_envs x max_players, row = env x
 * max_players + seat (absolute seat index; seats past the game's player
 * count are padding).
 */
struct VecEnvBuffers {
    float* observations{};        // [rows, encoder size]
    std::int32_t* actions{};      // [rows], read for acting rows only
    float* rewards{};             // [rows]
    std::uint8_t* terminals{};    // [rows]
    std::uint8_t* truncations{};  // [rows]
    std::uint8_t* acting{};       // [rows] 1 if the row must act now
    std::uint8_t* learner{};      // [rows] 1 if a learner plays the row's seat
    std::int32_t* seat{};         // [rows] absolute seat, -1 for padding
    std::int32_t* env_index{};    // [rows]
    std::int32_t* num_players{};  // [num_envs]
};

/** Episode aggregates since the last pop. */
struct EpisodeStats {
    struct Bucket {
        std::uint64_t episodes{};
        std::uint64_t truncated{};
        std::uint64_t turns{};
        std::uint64_t decisions{};
        /** Episodes won by a learner seat. */
        std::uint64_t learner_wins{};
        /** Learner seats summed over episodes (for each seat's expected share). */
        std::uint64_t learner_seats{};
        /** Truncated episodes that ended with no playable action (crowded maps). */
        std::uint64_t stuck{};
        std::array<std::uint64_t, 6> seat_wins{};

        void add(const Bucket& other);
    };

    Bucket total;
    std::vector<Bucket> scenarios;
};

/**
 * Many games stepped together for reinforcement learning (DESIGN 1.4).
 *
 * Every env has one row per seat slot up to the dims' max_players. Only the
 * row of the learner seat to move acts; step() applies its action, then plays
 * forced moves (a single legal action) and bot seats until a learner seat
 * must decide or the game ends, and writes every row. A finished game writes
 * its rewards (1 to the winner's row, 0 to the others) and sets terminal
 * flags on all of the env's rows. A game cut off at the turn limit pays 0
 * and sets both the terminal and the truncation flags: PufferLib 3 learns
 * from terminals only (its GAE ignores truncations, and its own C envs report
 * timeouts as terminals), so a truncation must also end the trajectory
 * there; the truncation flag tells the two apart for logging. Either way the
 * game is replaced at once, so the same step's observations are the next
 * game's, as in PufferLib's C environments (e.g. breakout). A game left with
 * no playable action (initial placements can crowd a small map) ends the
 * same way as a truncation.
 *
 * Each env draws scenarios, maps, seats and game seeds from its own random
 * stream seeded from the config seed, so results do not depend on threading.
 */
class VecEnv {
public:
    explicit VecEnv(VecEnvConfig config);
    ~VecEnv();

    VecEnv(const VecEnv&) = delete;
    VecEnv& operator=(const VecEnv&) = delete;

    [[nodiscard]] const VecEnvConfig& config() const noexcept { return config_; }
    [[nodiscard]] const GraphEncoder& encoder() const noexcept { return encoder_; }
    [[nodiscard]] int num_envs() const noexcept { return config_.num_envs; }
    [[nodiscard]] int max_players() const noexcept { return config_.dims.max_players; }
    [[nodiscard]] std::size_t num_rows() const noexcept {
        return static_cast<std::size_t>(num_envs()) * static_cast<std::size_t>(max_players());
    }
    [[nodiscard]] std::size_t observation_size() const noexcept { return encoder_.size(); }

    void bind(const VecEnvBuffers& buffers);
    /**
     * Start a fresh game in every env. A `seed` (or, on the first reset, the
     * config seed) reseeds every env's random stream, so equal seeds replay
     * equal runs; without one the streams continue and the games are new,
     * as in Gymnasium.
     */
    void reset(std::optional<std::uint64_t> seed = std::nullopt);
    /**
     * Throws std::invalid_argument, before changing any env, if an acting
     * row's action is not playable. If the engine fails later in the step,
     * some envs may have advanced: the VecEnv then refuses to step until the
     * next reset().
     */
    void step();
    /** Aggregates of the episodes finished since the last call. */
    [[nodiscard]] EpisodeStats pop_stats();

    [[nodiscard]] const Game& game(int env_index) const;
    [[nodiscard]] int scenario_index(int env_index) const;
    [[nodiscard]] const std::vector<SeatKind>& seat_kinds(int env_index) const;

private:
    struct Env;
    class ThreadPool;

    void for_each_env(void (VecEnv::*method)(int));
    void start_env(int env_index);
    void check_action(int env_index);
    void step_env(int env_index);
    void start_game(Env& env);
    void play_until_learner(int env_index);
    void next_game(int env_index);
    void finish_game(int env_index, bool truncated, bool stuck = false);
    void shape_rewards(int env_index, bool pay);
    void write_rows(int env_index);
    [[nodiscard]] std::size_t row(int env_index, int seat) const noexcept;

    VecEnvConfig config_;
    GraphEncoder encoder_;
    std::vector<Env> envs_;
    std::unique_ptr<ThreadPool> pool_;
    VecEnvBuffers buffers_{};
    bool seeded_{false};
    /** Every env holds a game waiting for a learner: set by reset, cleared by failures. */
    bool ready_{false};
};

}  // namespace cppanatron
