#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
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
    /**
     * Asymmetric actor/critic: when set, every encoded row is followed by a
     * critic block, the players block encoded at this level (the information
     * levels differ in that block alone), e.g. a full-information critic for
     * a private actor. Unset: rows are the actor's encoding only.
     */
    std::optional<InfoLevel> critic_level;
    std::uint64_t seed{0};
    int num_threads{1};
    /**
     * Optional shaping: each learner row also gets vp_reward times its change
     * in actual victory points / victory points to win. 0 is win/loss only.
     * The winner's reward on its winning step is capped at 1 (PufferLib
     * clamps rewards to [-1, 1]).
     */
    float vp_reward{0.0F};
    /**
     * Encode the observations of learner seats that wait. Off by default:
     * waiting rows then hold zeros and the no-op mask, since networks only
     * act (and estimate every seat's value) from the acting row.
     */
    bool encode_waiting{false};
    /**
     * Start the next game as soon as one ends (training). When false
     * (schedule mode, for evaluation) a finished env stays idle until the
     * caller puts an exact game into it with start_games, so the games
     * played do not depend on num_envs, threads or finishing order.
     */
    bool auto_reset{true};
};

/** An exact game for VecEnv::start_games (schedule mode). */
struct GameStart {
    int env{};
    /** The scenario whose player count, rules, turn limit and F settings apply. */
    int scenario{};
    std::shared_ptr<const CatanMap> map;
    /** Dice, development cards and steals. */
    std::uint64_t game_seed{};
    /** This game's random bots. */
    std::uint64_t bot_seed{};
    /** Who plays each seat, in seating order; every seat may be a bot. */
    std::vector<SeatKind> seats;
    /** The caller's label for each seat (its lineup slot); default 0..n-1. */
    std::vector<int> lineup_slots;
};

/** How a game ended, as schedule mode reports it. */
struct FinishedGame {
    int env{};
    int scenario{};
    std::uint64_t game_seed{};
    std::uint64_t bot_seed{};
    std::vector<SeatKind> seats;
    std::vector<int> lineup_slots;
    /** The winning seat, or -1 for a truncated game. */
    int winner{-1};
    int turns{};
    bool truncated{};
    /** Truncated because no action was playable (crowded maps). */
    bool stuck{};
    /** Learner decisions per seat (forced moves are not decisions). */
    std::vector<std::uint64_t> decisions;
};

/**
 * Caller-owned buffers, rows = num_envs x max_players, row = env x
 * max_players + seat (absolute seat index; seats past the game's player
 * count are padding). Callers read but do not write the observations: the
 * VecEnv only rewrites a waiting row's zeros and no-op when it changes.
 */
struct VecEnvBuffers {
    float* observations{};        // [rows, observation_size()]
    std::int32_t* actions{};      // [rows], read for acting rows only
    float* rewards{};             // [rows]
    std::uint8_t* terminals{};    // [rows]
    std::uint8_t* truncations{};  // [rows]
    std::uint8_t* acting{};       // [rows] 1 if the row must act now
    std::uint8_t* learner{};      // [rows] 1 if a learner plays the row's seat
    std::int32_t* seat{};         // [rows] absolute seat, -1 for padding
    std::int32_t* env_index{};    // [rows]
    std::int32_t* num_players{};  // [num_envs], 0 for an idle env
    /** Optional: [rows] the lineup slot of the row's seat, -1 for padding. */
    std::int32_t* lineup_slot{};
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
 * In schedule mode (auto_reset false) envs instead wait idle (no rows act)
 * until start_games gives them an exact game, and finished games are
 * reported by pop_finished.
 *
 * Each env draws scenarios, maps, seats, game seeds and bot seeds from its
 * own random stream seeded from the config seed, so results do not depend on
 * threading. Random bots play from a stream seeded per game by its bot seed,
 * so a game's play depends on its seeds alone.
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
    /** Row size: the encoder's, plus the critic block when critic_level is set. */
    [[nodiscard]] std::size_t observation_size() const noexcept {
        return encoder_.size() + (critic_encoder_.has_value() ? critic_encoder_->players_size() : 0);
    }
    /** The critic block's encoder (players block at critic_level), if any. */
    [[nodiscard]] const GraphEncoder* critic_encoder() const noexcept {
        return critic_encoder_.has_value() ? &*critic_encoder_ : nullptr;
    }

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

    /**
     * Schedule mode: put exact games into idle envs and play each until a
     * learner must decide (or, with bots only, to the end), in parallel, then
     * write those envs' rows. Throws std::invalid_argument, before starting
     * any, for a busy or repeated env, a bad scenario, a map that does not
     * fit, or seats that do not match the scenario's player count.
     */
    void start_games(const std::vector<GameStart>& games);
    /**
     * The expert's labels for imitation: for every acting row, the action
     * index Catanatron's value function player (value_action) would choose
     * in that game; -1 for every other row. Computed on the env threads.
     */
    void expert_actions(
        std::span<std::int32_t> out, ValueEnemies enemies = ValueEnemies::next,
        const ValueWeights& weights = {}) const;
    /** Schedule mode: the games finished since the last call, by env. */
    [[nodiscard]] std::vector<FinishedGame> pop_finished();
    /** Whether an env holds a game (always, unless in schedule mode). */
    [[nodiscard]] bool active(int env_index) const;
    [[nodiscard]] const std::vector<int>& lineup_slots(int env_index) const;

    [[nodiscard]] const Game& game(int env_index) const;
    [[nodiscard]] int scenario_index(int env_index) const;
    [[nodiscard]] const std::vector<SeatKind>& seat_kinds(int env_index) const;
    /** Seeds of an env's current (or last) game: dice, cards and steals, and bots. */
    [[nodiscard]] std::uint64_t game_seed(int env_index) const;
    [[nodiscard]] std::uint64_t bot_seed(int env_index) const;

private:
    struct Env;
    class ThreadPool;

    void for_each_env(void (VecEnv::*method)(int));
    void start_env(int env_index);
    void check_action(int env_index);
    void step_env(int env_index);
    void start_game(Env& env);
    void begin_game(
        Env& env, int scenario, std::shared_ptr<const CatanMap> map, std::uint64_t game_seed,
        std::vector<SeatKind> seats, std::vector<int> lineup_slots, std::uint64_t bot_seed);
    void clear_transition(int env_index);
    void play_until_learner(int env_index);
    void next_game(int env_index);
    void finish_game(int env_index, bool truncated, bool stuck = false);
    void shape_rewards(int env_index, bool pay);
    void write_rows(int env_index);
    [[nodiscard]] std::size_t row(int env_index, int seat) const noexcept;

    VecEnvConfig config_;
    GraphEncoder encoder_;
    std::optional<GraphEncoder> critic_encoder_;
    std::vector<Env> envs_;
    std::unique_ptr<ThreadPool> pool_;
    VecEnvBuffers buffers_{};
    bool seeded_{false};
    /** Every env holds a game waiting for a learner: set by reset, cleared by failures. */
    bool ready_{false};
};

}  // namespace cppanatron
