#include "cppanatron/vec_env.hpp"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace cppanatron {
namespace {

/** SplitMix64: decorrelates the per-env streams drawn from one seed. */
std::uint64_t mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

template <typename T>
std::size_t pick(std::size_t count, T& random) {
    return std::uniform_int_distribution<std::size_t>(0, count - 1)(random);
}

void validate(const VecEnvConfig& config) {
    if (config.num_envs <= 0) {
        throw std::invalid_argument("num_envs must be positive");
    }
    if (config.num_threads <= 0) {
        throw std::invalid_argument("num_threads must be positive");
    }
    if (config.scenarios.empty()) {
        throw std::invalid_argument("a VecEnv needs at least one scenario");
    }
    config.dims.validate();
    double total_weight = 0.0;
    for (std::size_t i = 0; i < config.scenarios.size(); ++i) {
        const Scenario& scenario = config.scenarios[i];
        const std::string where = "scenario " + std::to_string(i) + ": ";
        validate_rules(scenario.rules);
        if (scenario.num_players < 2 || scenario.num_players > config.dims.max_players) {
            throw std::invalid_argument(
                where + "num_players must be between two and the dims' max_players");
        }
        if (!scenario.seats.empty()) {
            if (static_cast<int>(scenario.seats.size()) != scenario.num_players) {
                throw std::invalid_argument(where + "give one seat kind per player");
            }
            if (std::none_of(scenario.seats.begin(), scenario.seats.end(),
                             [](SeatKind kind) { return kind == SeatKind::learner; })) {
                throw std::invalid_argument(where + "at least one seat must be a learner");
            }
        }
        if (scenario.turn_limit <= 0) {
            throw std::invalid_argument(where + "turn_limit must be positive");
        }
        if (!(scenario.weight >= 0.0)) {
            throw std::invalid_argument(where + "weight must be non-negative");
        }
        total_weight += scenario.weight;
        std::vector<std::shared_ptr<const CatanMap>> maps = scenario.maps;
        if (maps.empty()) {
            maps.push_back(std::make_shared<const CatanMap>(
                CatanMap::build(scenario.map_type, 0, scenario.number_placement)));
        }
        for (const auto& map : maps) {
            if (map == nullptr) {
                throw std::invalid_argument(where + "maps must not be null");
            }
            if (!config.dims.fits(*map, scenario.num_players)) {
                throw std::invalid_argument(where + "a map does not fit the dims");
            }
        }
    }
    if (!(total_weight > 0.0)) {
        throw std::invalid_argument("scenario weights must not all be zero");
    }
}

}  // namespace

void EpisodeStats::Bucket::add(const Bucket& other) {
    episodes += other.episodes;
    truncated += other.truncated;
    turns += other.turns;
    decisions += other.decisions;
    learner_wins += other.learner_wins;
    learner_seats += other.learner_seats;
    for (std::size_t seat = 0; seat < seat_wins.size(); ++seat) {
        seat_wins[seat] += other.seat_wins[seat];
    }
}

struct VecEnv::Env {
    std::mt19937_64 random;
    std::unique_ptr<Game> game;
    int scenario{};
    std::vector<SeatKind> seats;
    /** Actual victory points at the last reward, for shaping. */
    std::vector<int> victory_points;
    std::uint64_t decisions{};
    EpisodeStats stats;
};

/**
 * Persistent workers that split a range of envs into one contiguous chunk per
 * thread; the calling thread runs the first chunk.
 */
class VecEnv::ThreadPool {
public:
    explicit ThreadPool(int threads) {
        for (int index = 1; index < threads; ++index) {
            workers_.emplace_back([this, index] { work(index); });
        }
        errors_.resize(static_cast<std::size_t>(threads));
    }

    ~ThreadPool() {
        {
            const std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        start_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void run(int tasks, const std::function<void(int)>& task) {
        {
            const std::lock_guard lock(mutex_);
            task_ = &task;
            tasks_ = tasks;
            pending_ = static_cast<int>(workers_.size());
            ++generation_;
        }
        start_.notify_all();
        run_chunk(0);
        {
            std::unique_lock lock(mutex_);
            done_.wait(lock, [this] { return pending_ == 0; });
            task_ = nullptr;
        }
        for (std::exception_ptr& error : errors_) {
            if (error) {
                std::exception_ptr first = error;
                std::fill(errors_.begin(), errors_.end(), nullptr);
                std::rethrow_exception(first);
            }
        }
    }

private:
    void run_chunk(int index) {
        const int threads = static_cast<int>(workers_.size()) + 1;
        const int begin = tasks_ * index / threads;
        const int end = tasks_ * (index + 1) / threads;
        try {
            for (int task = begin; task < end; ++task) {
                (*task_)(task);
            }
        } catch (...) {
            errors_[static_cast<std::size_t>(index)] = std::current_exception();
        }
    }

    void work(int index) {
        std::uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock lock(mutex_);
                start_.wait(lock, [&] { return stopping_ || generation_ != seen; });
                if (stopping_) {
                    return;
                }
                seen = generation_;
            }
            run_chunk(index);
            {
                const std::lock_guard lock(mutex_);
                if (--pending_ == 0) {
                    done_.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers_;
    std::vector<std::exception_ptr> errors_;
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable done_;
    const std::function<void(int)>* task_{};
    int tasks_{};
    int pending_{};
    std::uint64_t generation_{};
    bool stopping_{false};
};

VecEnv::VecEnv(VecEnvConfig config)
    : config_((validate(config), std::move(config))),
      encoder_(config_.dims, config_.level),
      envs_(static_cast<std::size_t>(config_.num_envs)),
      pool_(std::make_unique<ThreadPool>(std::min(config_.num_threads, config_.num_envs))) {}

VecEnv::~VecEnv() = default;

void VecEnv::bind(const VecEnvBuffers& buffers) {
    if (buffers.observations == nullptr || buffers.actions == nullptr ||
        buffers.rewards == nullptr || buffers.terminals == nullptr ||
        buffers.truncations == nullptr || buffers.acting == nullptr ||
        buffers.learner == nullptr || buffers.seat == nullptr ||
        buffers.env_index == nullptr || buffers.num_players == nullptr) {
        throw std::invalid_argument("every VecEnv buffer must be bound");
    }
    buffers_ = buffers;
}

std::size_t VecEnv::row(int env_index, int seat) const noexcept {
    return static_cast<std::size_t>(env_index) * static_cast<std::size_t>(max_players()) +
           static_cast<std::size_t>(seat);
}

void VecEnv::for_each_env(void (VecEnv::*method)(int)) {
    if (buffers_.observations == nullptr) {
        throw std::logic_error("bind the VecEnv buffers first");
    }
    const std::function<void(int)> task = [this, method](int env_index) {
        (this->*method)(env_index);
    };
    pool_->run(config_.num_envs, task);
}

void VecEnv::reset(std::optional<std::uint64_t> seed) {
    if (seed.has_value()) {
        config_.seed = *seed;
    }
    for (std::size_t index = 0; index < envs_.size(); ++index) {
        envs_[index].random.seed(mix(config_.seed ^ mix(index)));
    }
    for_each_env(&VecEnv::start_env);
    started_ = true;
}

void VecEnv::step() {
    if (!started_) {
        throw std::logic_error("reset the VecEnv before stepping it");
    }
    for_each_env(&VecEnv::step_env);
}

void VecEnv::start_game(Env& env) {
    std::vector<double> weights;
    weights.reserve(config_.scenarios.size());
    for (const Scenario& scenario : config_.scenarios) {
        weights.push_back(scenario.weight);
    }
    env.scenario = std::discrete_distribution<int>(weights.begin(), weights.end())(env.random);
    const Scenario& scenario = config_.scenarios[static_cast<std::size_t>(env.scenario)];

    std::shared_ptr<const CatanMap> map;
    if (!scenario.maps.empty()) {
        map = scenario.maps[pick(scenario.maps.size(), env.random)];
    } else {
        map = std::make_shared<const CatanMap>(
            CatanMap::build(scenario.map_type, env.random(), scenario.number_placement));
    }
    const auto num_players = static_cast<std::size_t>(scenario.num_players);
    env.game = std::make_unique<Game>(
        std::vector<Color>(kColors.begin(), kColors.begin() + scenario.num_players), map,
        env.random(), scenario.rules);
    env.seats = scenario.seats.empty()
                    ? std::vector<SeatKind>(num_players, SeatKind::learner)
                    : scenario.seats;
    if (scenario.shuffle_seats) {
        std::shuffle(env.seats.begin(), env.seats.end(), env.random);
    }
    env.victory_points.assign(num_players, 0);
    env.decisions = 0;
}

void VecEnv::finish_game(int env_index, bool truncated) {
    Env& env = envs_[static_cast<std::size_t>(env_index)];
    const Game& game = *env.game;
    shape_rewards(env_index, true);

    EpisodeStats::Bucket episode;
    episode.episodes = 1;
    episode.turns = static_cast<std::uint64_t>(game.num_turns());
    episode.decisions = env.decisions;
    episode.learner_seats = static_cast<std::uint64_t>(
        std::count(env.seats.begin(), env.seats.end(), SeatKind::learner));
    if (truncated) {
        episode.truncated = 1;
    } else {
        const auto& colors = game.colors();
        const auto winner = static_cast<std::size_t>(
            std::find(colors.begin(), colors.end(), *game.winning_color()) - colors.begin());
        buffers_.rewards[row(env_index, static_cast<int>(winner))] += 1.0F;
        episode.seat_wins[winner] = 1;
        episode.learner_wins = env.seats[winner] == SeatKind::learner ? 1 : 0;
    }
    for (int seat = 0; seat < max_players(); ++seat) {
        const std::size_t index = row(env_index, seat);
        // Terminal either way: PufferLib's advantages only read terminals.
        buffers_.terminals[index] = 1;
        buffers_.truncations[index] = truncated ? 1 : 0;
    }
    env.stats.total.add(episode);
    if (env.stats.scenarios.size() < config_.scenarios.size()) {
        env.stats.scenarios.resize(config_.scenarios.size());
    }
    env.stats.scenarios[static_cast<std::size_t>(env.scenario)].add(episode);
}

void VecEnv::shape_rewards(int env_index, bool pay) {
    Env& env = envs_[static_cast<std::size_t>(env_index)];
    const Game& game = *env.game;
    for (std::size_t seat = 0; seat < env.seats.size(); ++seat) {
        const int points = game.players()[seat].actual_victory_points;
        if (pay && config_.vp_reward != 0.0F && env.seats[seat] == SeatKind::learner) {
            buffers_.rewards[row(env_index, static_cast<int>(seat))] +=
                config_.vp_reward * static_cast<float>(points - env.victory_points[seat]) /
                static_cast<float>(game.victory_points_to_win());
        }
        env.victory_points[seat] = points;
    }
}

void VecEnv::play_until_learner(int env_index) {
    Env& env = envs_[static_cast<std::size_t>(env_index)];
    for (;;) {
        Game& game = *env.game;
        const Scenario& scenario = config_.scenarios[static_cast<std::size_t>(env.scenario)];
        const bool won = game.winning_color().has_value();
        if (won || game.num_turns() >= scenario.turn_limit) {
            finish_game(env_index, !won);
            start_game(env);
            shape_rewards(env_index, false);
            continue;
        }
        const auto& actions = game.playable_actions();
        if (actions.empty()) {
            throw std::logic_error("a game in progress has no playable action");
        }
        switch (env.seats[static_cast<std::size_t>(game.current_player_index())]) {
            case SeatKind::learner:
                if (actions.size() > 1) {
                    return;
                }
                game.execute(actions.front());
                break;
            case SeatKind::random:
                game.execute(actions[pick(actions.size(), env.random)]);
                break;
            case SeatKind::value:
                game.execute(value_action(game, ValueWeights{}, scenario.value_enemies));
                break;
        }
    }
}

void VecEnv::start_env(int env_index) {
    Env& env = envs_[static_cast<std::size_t>(env_index)];
    for (int seat = 0; seat < max_players(); ++seat) {
        const std::size_t index = row(env_index, seat);
        buffers_.rewards[index] = 0.0F;
        buffers_.terminals[index] = 0;
        buffers_.truncations[index] = 0;
    }
    start_game(env);
    play_until_learner(env_index);
    // Pieces placed before the first decision are not rewarded.
    shape_rewards(env_index, false);
    write_rows(env_index);
}

void VecEnv::step_env(int env_index) {
    Env& env = envs_[static_cast<std::size_t>(env_index)];
    for (int seat = 0; seat < max_players(); ++seat) {
        const std::size_t index = row(env_index, seat);
        buffers_.rewards[index] = 0.0F;
        buffers_.terminals[index] = 0;
        buffers_.truncations[index] = 0;
    }
    Game& game = *env.game;
    const int seat = game.current_player_index();
    const std::int32_t index = buffers_.actions[row(env_index, seat)];
    const GraphActionSpace& space = encoder_.action_space();
    const auto refuse = [&](const std::string& why) {
        throw std::invalid_argument(
            "env " + std::to_string(env_index) + ", seat " + std::to_string(seat) +
            ": action " + std::to_string(index) + " " + why);
    };
    if (index < 0 || static_cast<std::size_t>(index) >= space.noop()) {
        refuse("is not a game action");
    }
    Action action;
    try {
        action = space.decode(game, static_cast<std::size_t>(index));
    } catch (const std::exception& error) {
        refuse(std::string("does not decode: ") + error.what());
    }
    const auto& playable = game.playable_actions();
    if (std::find(playable.begin(), playable.end(), action) == playable.end()) {
        refuse("(" + space.name(static_cast<std::size_t>(index)) + ") is not playable");
    }
    game.execute(action);
    ++env.decisions;

    const std::uint64_t episodes = env.stats.total.episodes;
    play_until_learner(env_index);
    // A finished game was paid in finish_game; the next one starts unpaid.
    shape_rewards(env_index, env.stats.total.episodes == episodes);
    write_rows(env_index);
}

void VecEnv::write_rows(int env_index) {
    const Env& env = envs_[static_cast<std::size_t>(env_index)];
    const Game& game = *env.game;
    const int num_players = static_cast<int>(game.colors().size());
    const int current = game.current_player_index();
    const std::size_t size = encoder_.size();
    buffers_.num_players[env_index] = num_players;
    for (int seat = 0; seat < max_players(); ++seat) {
        const std::size_t index = row(env_index, seat);
        const bool present = seat < num_players;
        const bool learns =
            present && env.seats[static_cast<std::size_t>(seat)] == SeatKind::learner;
        const bool acts = learns && seat == current;
        buffers_.seat[index] = present ? seat : -1;
        buffers_.env_index[index] = env_index;
        buffers_.learner[index] = learns ? 1 : 0;
        buffers_.acting[index] = acts ? 1 : 0;
        const std::span<float> observation(buffers_.observations + index * size, size);
        if (acts || (learns && config_.encode_waiting)) {
            encoder_.encode(game, seat, observation);
        } else {
            encoder_.encode_waiting(observation);
        }
    }
}

EpisodeStats VecEnv::pop_stats() {
    EpisodeStats result;
    result.scenarios.resize(config_.scenarios.size());
    for (Env& env : envs_) {
        result.total.add(env.stats.total);
        for (std::size_t i = 0; i < env.stats.scenarios.size(); ++i) {
            result.scenarios[i].add(env.stats.scenarios[i]);
        }
        env.stats = {};
    }
    return result;
}

const Game& VecEnv::game(int env_index) const {
    const Env& env = envs_.at(static_cast<std::size_t>(env_index));
    if (env.game == nullptr) {
        throw std::logic_error("reset the VecEnv first");
    }
    return *env.game;
}

int VecEnv::scenario_index(int env_index) const {
    return envs_.at(static_cast<std::size_t>(env_index)).scenario;
}

const std::vector<SeatKind>& VecEnv::seat_kinds(int env_index) const {
    return envs_.at(static_cast<std::size_t>(env_index)).seats;
}

}  // namespace cppanatron
