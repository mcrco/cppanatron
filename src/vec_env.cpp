#include "cppanatron/vec_env.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <iterator>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "cppanatron/random.hpp"

namespace cppanatron {
namespace {

/** SplitMix64: decorrelates the per-env streams drawn from one seed. */
/**
 * Games an env may finish in a row without a learner decision before step()
 * gives up: bots alone, or crowded maps sticking before a learner's turn,
 * would otherwise loop forever with the GIL released.
 */
constexpr int kMaxGamesWithoutLearner = 64;

std::uint64_t mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

template <typename T>
std::size_t pick(std::size_t count, T& random) {
    return random::index(random, count);
}

/**
 * Settlements the distance rule lets a map hold, by greedily placing on the
 * open node with the fewest open neighbors: a lower bound on the maximum,
 * and usually equal to it on hex maps.
 */
int settlement_capacity(const CatanMap& map) {
    std::vector<bool> open(static_cast<std::size_t>(map.num_node_ids()), false);
    for (int node : map.land_node_list()) {
        open[static_cast<std::size_t>(node)] = true;
    }
    const auto open_neighbors = [&](int node) {
        return std::count_if(map.neighbors(node).begin(), map.neighbors(node).end(),
                             [&](int neighbor) { return open[static_cast<std::size_t>(neighbor)]; });
    };
    int placed = 0;
    for (;;) {
        int best = -1;
        std::ptrdiff_t fewest = 0;
        for (int node : map.land_node_list()) {
            if (open[static_cast<std::size_t>(node)] &&
                (best < 0 || open_neighbors(node) < fewest)) {
                best = node;
                fewest = open_neighbors(node);
            }
        }
        if (best < 0) {
            return placed;
        }
        ++placed;
        open[static_cast<std::size_t>(best)] = false;
        for (int neighbor : map.neighbors(best)) {
            open[static_cast<std::size_t>(neighbor)] = false;
        }
    }
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
            // Schedule mode gives every game its seats, bots-only included.
            if (config.auto_reset &&
                std::none_of(scenario.seats.begin(), scenario.seats.end(),
                             [](SeatKind kind) { return kind == SeatKind::learner; })) {
                throw std::invalid_argument(where + "at least one seat must be a learner");
            }
        }
        // The initial placements take 2 (n - 1) turns; a game cut off before
        // every learner has placed could end before any learner decides.
        if (scenario.turn_limit <= 2 * (scenario.num_players - 1)) {
            throw std::invalid_argument(
                where + "turn_limit must exceed the " +
                std::to_string(2 * (scenario.num_players - 1)) + " turns of initial placement");
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
            // Every seat places two settlements to start.
            if (settlement_capacity(*map) < 2 * scenario.num_players) {
                throw std::invalid_argument(
                    where + "a map cannot seat " + std::to_string(scenario.num_players) +
                    " players: the distance rule leaves room for " +
                    std::to_string(settlement_capacity(*map)) + " settlements");
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
    stuck += other.stuck;
    for (std::size_t seat = 0; seat < seat_wins.size(); ++seat) {
        seat_wins[seat] += other.seat_wins[seat];
    }
}

struct VecEnv::Env {
    /** Draws each new game's scenario, map, seeds and seating (auto-reset). */
    std::mt19937_64 random;
    /**
     * The current game's bot stream, seeded per game: what bots play depends
     * on that game's bot seed alone, not on the games the env played before.
     */
    std::mt19937_64 bots;
    std::uint64_t bot_seed{};
    std::uint64_t game_seed{};
    /** Null while idle (schedule mode). */
    std::unique_ptr<Game> game;
    int scenario{};
    std::vector<SeatKind> seats;
    std::vector<int> lineup_slots;
    std::vector<std::uint64_t> seat_decisions;
    /** With config.trackers: each learner seat's HandTracker of this game. */
    std::map<std::size_t, HandTracker> trackers;
    /** Schedule mode: games finished since the last pop_finished. */
    std::vector<FinishedGame> finished;
    /** Actual victory points at the last reward, for shaping. */
    std::vector<int> victory_points;
    std::uint64_t decisions{};
    EpisodeStats stats;
    /**
     * Per seat slot: the row already holds the waiting observation (zeros
     * and the no-op), which does not depend on the game, so it need not be
     * written again until the row is encoded.
     */
    std::vector<bool> waiting_written;
    /** Games finished in a row without a learner decision. */
    int games_without_learner{};
    /** The acting row's decoded action, or why it cannot be played. */
    Action pending;
    std::string refusal;
};

/**
 * Persistent workers that share a range of envs: each takes the next few
 * (kGrain) from an atomic counter, so slow envs (bot seats, six players) do
 * not leave other threads idle. The calling thread works too. Which thread
 * steps an env never matters, since every env has its own random stream.
 */
class VecEnv::ThreadPool {
public:
    static constexpr int kGrain = 4;

    explicit ThreadPool(int threads) : errors_(static_cast<std::size_t>(threads)) {
        for (int index = 1; index < threads; ++index) {
            workers_.emplace_back([this, index] { work(index); });
        }
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

    /** Runs task(0..tasks-1) once each; rethrows the failure of the lowest task. */
    void run(int tasks, const std::function<void(int)>& task) {
        {
            const std::lock_guard lock(mutex_);
            task_ = &task;
            tasks_ = tasks;
            next_.store(0, std::memory_order_relaxed);
            pending_ = static_cast<int>(workers_.size());
            ++generation_;
        }
        start_.notify_all();
        run_tasks(0);
        {
            std::unique_lock lock(mutex_);
            done_.wait(lock, [this] { return pending_ == 0; });
            task_ = nullptr;
        }
        const Failure* first = nullptr;
        for (const Failure& failure : errors_) {
            if (failure.error && (first == nullptr || failure.task < first->task)) {
                first = &failure;
            }
        }
        if (first != nullptr) {
            const std::exception_ptr error = first->error;
            std::fill(errors_.begin(), errors_.end(), Failure{});
            std::rethrow_exception(error);
        }
    }

private:
    struct Failure {
        int task{};
        std::exception_ptr error;
    };

    void run_tasks(int thread) {
        Failure& failure = errors_[static_cast<std::size_t>(thread)];
        for (;;) {
            const int begin = next_.fetch_add(kGrain, std::memory_order_relaxed);
            if (begin >= tasks_) {
                return;
            }
            const int end = std::min(begin + kGrain, tasks_);
            for (int task = begin; task < end; ++task) {
                try {
                    (*task_)(task);
                } catch (...) {
                    if (!failure.error || task < failure.task) {
                        failure = {task, std::current_exception()};
                    }
                }
            }
        }
    }

    void work(int thread) {
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
            run_tasks(thread);
            {
                const std::lock_guard lock(mutex_);
                if (--pending_ == 0) {
                    done_.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers_;
    std::vector<Failure> errors_;
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable done_;
    const std::function<void(int)>* task_{};
    int tasks_{};
    std::atomic<int> next_{0};
    int pending_{};
    std::uint64_t generation_{};
    bool stopping_{false};
};

VecEnv::VecEnv(VecEnvConfig config)
    : config_((validate(config), std::move(config))),
      encoder_(config_.dims, config_.level),
      critic_encoder_(
          config_.critic_level.has_value()
              ? std::optional<ActorCriticEncoder>(
                    std::in_place, config_.dims, config_.level, *config_.critic_level)
              : std::nullopt),
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
    // New buffers hold anything.
    for (Env& env : envs_) {
        env.waiting_written.assign(static_cast<std::size_t>(max_players()), false);
    }
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
    // Like Gymnasium: a seed (or the first reset) reseeds every env's stream;
    // otherwise the streams continue, so the new games are not replays.
    if (seed.has_value() || !seeded_) {
        config_.seed = seed.value_or(config_.seed);
        for (std::size_t index = 0; index < envs_.size(); ++index) {
            envs_[index].random.seed(mix(config_.seed ^ mix(index)));
        }
        seeded_ = true;
    }
    ready_ = false;
    for_each_env(&VecEnv::start_env);
    ready_ = true;
}

void VecEnv::step() {
    if (!ready_) {
        throw std::logic_error(
            "reset the VecEnv before stepping it (first, or after a step failed mid-way)");
    }
    // Every action is checked before any env changes, so a bad action leaves
    // all of them as they were and the caller may fix it and step again.
    for_each_env(&VecEnv::check_action);
    for (const Env& env : envs_) {
        if (!env.refusal.empty()) {
            throw std::invalid_argument(env.refusal);
        }
    }
    try {
        for_each_env(&VecEnv::step_env);
    } catch (...) {
        ready_ = false;
        throw;
    }
}

void VecEnv::start_game(Env& env) {
    std::vector<double> weights;
    weights.reserve(config_.scenarios.size());
    for (const Scenario& scenario : config_.scenarios) {
        weights.push_back(scenario.weight);
    }
    const int index = static_cast<int>(random::discrete(env.random, weights));
    const Scenario& scenario = config_.scenarios[static_cast<std::size_t>(index)];

    std::shared_ptr<const CatanMap> map;
    if (!scenario.maps.empty()) {
        map = scenario.maps[pick(scenario.maps.size(), env.random)];
    } else {
        map = std::make_shared<const CatanMap>(
            CatanMap::build(scenario.map_type, env.random(), scenario.number_placement));
    }
    const auto num_players = static_cast<std::size_t>(scenario.num_players);
    const std::uint64_t game_seed = env.random();
    // Lineup slots are positions in the scenario's seat list, so they tell
    // who plays where after a shuffle.
    std::vector<int> slots(num_players);
    std::iota(slots.begin(), slots.end(), 0);
    if (scenario.shuffle_seats) {
        random::shuffle(slots.begin(), slots.end(), env.random);
    }
    std::vector<SeatKind> seats(num_players, SeatKind::learner);
    if (!scenario.seats.empty()) {
        for (std::size_t seat = 0; seat < num_players; ++seat) {
            seats[seat] = scenario.seats[static_cast<std::size_t>(slots[seat])];
        }
    }
    const std::uint64_t bot_seed = env.random();
    begin_game(env, index, std::move(map), game_seed, std::move(seats), std::move(slots), bot_seed);
}

void VecEnv::begin_game(
    Env& env, int scenario_index, std::shared_ptr<const CatanMap> map, std::uint64_t game_seed,
    std::vector<SeatKind> seats, std::vector<int> lineup_slots, std::uint64_t bot_seed) {
    const Scenario& scenario = config_.scenarios[static_cast<std::size_t>(scenario_index)];
    env.scenario = scenario_index;
    env.game_seed = game_seed;
    env.game = std::make_unique<Game>(
        std::vector<Color>(kColors.begin(), kColors.begin() + scenario.num_players),
        std::move(map), game_seed, scenario.rules);
    env.seats = std::move(seats);
    env.lineup_slots = std::move(lineup_slots);
    env.bot_seed = bot_seed;
    env.bots.seed(bot_seed);
    env.victory_points.assign(env.seats.size(), 0);
    env.seat_decisions.assign(env.seats.size(), 0);
    env.decisions = 0;
    env.trackers.clear();
    if (config_.trackers) {
        for (std::size_t seat = 0; seat < env.seats.size(); ++seat) {
            if (env.seats[seat] == SeatKind::learner) {
                env.trackers.emplace(seat, HandTracker(*env.game, static_cast<int>(seat)));
            }
        }
        observe_options(env);
    }
}

// By value: the action may live in the game's playable actions, which
// executing it replaces.
void VecEnv::play(Env& env, const Action action) {
    if (env.trackers.empty()) {
        env.game->execute(action);
        return;
    }
    const Game before = *env.game;
    env.game->execute(action);
    for (auto& [seat, tracker] : env.trackers) {
        (void)seat;
        tracker.observe(before, action, *env.game);
    }
    observe_options(env);
}

void VecEnv::observe_options(Env& env) {
    // What the seat to move learns from its own legal actions (the friendly
    // robber spares seats under 3 points, bounding their victory point cards).
    const Game& game = *env.game;
    const auto tracker = env.trackers.find(static_cast<std::size_t>(game.current_player_index()));
    if (tracker != env.trackers.end() && !game.winning_color().has_value()) {
        tracker->second.observe_options(game, game.playable_actions());
    }
}

void VecEnv::finish_game(int env_index, bool truncated, bool stuck) {
    Env& env = envs_[static_cast<std::size_t>(env_index)];
    const Game& game = *env.game;
    shape_rewards(env_index, true);

    EpisodeStats::Bucket episode;
    episode.episodes = 1;
    episode.turns = static_cast<std::uint64_t>(game.num_turns());
    episode.decisions = env.decisions;
    episode.learner_seats = static_cast<std::uint64_t>(
        std::count(env.seats.begin(), env.seats.end(), SeatKind::learner));
    episode.stuck = stuck ? 1 : 0;
    if (truncated) {
        episode.truncated = 1;
    } else {
        const auto& colors = game.colors();
        const auto winner = static_cast<std::size_t>(
            std::find(colors.begin(), colors.end(), *game.winning_color()) - colors.begin());
        // The win plus this step's shaping stays within PufferLib's [-1, 1] clamp.
        float& reward = buffers_.rewards[row(env_index, static_cast<int>(winner))];
        reward = std::min(reward + 1.0F, 1.0F);
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

    if (!config_.auto_reset) {
        FinishedGame finished;
        finished.env = env_index;
        finished.scenario = env.scenario;
        finished.game_seed = env.game_seed;
        finished.bot_seed = env.bot_seed;
        finished.seats = env.seats;
        finished.lineup_slots = env.lineup_slots;
        if (!truncated) {
            const auto& colors = game.colors();
            finished.winner = static_cast<int>(
                std::find(colors.begin(), colors.end(), *game.winning_color()) - colors.begin());
        }
        finished.turns = game.num_turns();
        finished.truncated = truncated;
        finished.stuck = stuck;
        finished.decisions = env.seat_decisions;
        env.finished.push_back(std::move(finished));
    }
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

void VecEnv::next_game(int env_index) {
    Env& env = envs_[static_cast<std::size_t>(env_index)];
    if (++env.games_without_learner > kMaxGamesWithoutLearner) {
        throw std::runtime_error(
            "env " + std::to_string(env_index) + ": " +
            std::to_string(kMaxGamesWithoutLearner) +
            " games in a row ended before any learner decided; check the scenarios");
    }
    start_game(env);
    shape_rewards(env_index, false);
}

void VecEnv::play_until_learner(int env_index) {
    Env& env = envs_[static_cast<std::size_t>(env_index)];
    // After a game ends: the next one (auto-reset), or an idle env.
    const auto end_game = [&](bool truncated, bool stuck) {
        finish_game(env_index, truncated, stuck);
        if (config_.auto_reset) {
            next_game(env_index);
            return true;
        }
        env.game.reset();
        return false;
    };
    for (;;) {
        Game& game = *env.game;
        const Scenario& scenario = config_.scenarios[static_cast<std::size_t>(env.scenario)];
        const bool won = game.winning_color().has_value();
        if (won || game.num_turns() >= scenario.turn_limit) {
            if (end_game(!won, false)) {
                continue;
            }
            return;
        }
        const auto& actions = game.playable_actions();
        if (actions.empty()) {
            // Possible on crowded maps: the initial placements can leave a
            // seat no open node. The game cannot go on; count it as a
            // truncated (stuck) episode.
            if (end_game(true, true)) {
                continue;
            }
            return;
        }
        switch (env.seats[static_cast<std::size_t>(game.current_player_index())]) {
            case SeatKind::learner:
                if (actions.size() > 1) {
                    env.games_without_learner = 0;
                    return;
                }
                play(env, actions.front());
                break;
            case SeatKind::random:
                play(env, actions[pick(actions.size(), env.bots)]);
                break;
            case SeatKind::value:
                play(env, value_action(game, ValueWeights{}, scenario.value_enemies));
                break;
        }
    }
}

void VecEnv::clear_transition(int env_index) {
    for (int seat = 0; seat < max_players(); ++seat) {
        const std::size_t index = row(env_index, seat);
        buffers_.rewards[index] = 0.0F;
        buffers_.terminals[index] = 0;
        buffers_.truncations[index] = 0;
    }
}

void VecEnv::start_env(int env_index) {
    Env& env = envs_[static_cast<std::size_t>(env_index)];
    env.waiting_written.assign(static_cast<std::size_t>(max_players()), false);
    clear_transition(env_index);
    if (config_.auto_reset) {
        start_game(env);
        play_until_learner(env_index);
        // Pieces placed before the first decision are not rewarded.
        shape_rewards(env_index, false);
    } else {
        env.game.reset();
        env.finished.clear();
    }
    write_rows(env_index);
}

void VecEnv::check_action(int env_index) {
    Env& env = envs_[static_cast<std::size_t>(env_index)];
    env.refusal.clear();
    if (env.game == nullptr) {
        return;  // idle: no row acts
    }
    const Game& game = *env.game;
    const int seat = game.current_player_index();
    const std::int32_t index = buffers_.actions[row(env_index, seat)];
    const GraphActionSpace& space = encoder_.action_space();
    const auto refuse = [&](const std::string& why) {
        env.refusal = "env " + std::to_string(env_index) + ", seat " + std::to_string(seat) +
                      ": action " + std::to_string(index) + " " + why;
    };
    if (index < 0 || static_cast<std::size_t>(index) >= space.noop()) {
        refuse("is not a game action");
        return;
    }
    try {
        env.pending = space.decode(game, static_cast<std::size_t>(index));
    } catch (const std::exception& error) {
        refuse(std::string("does not decode: ") + error.what());
        return;
    }
    const auto& playable = game.playable_actions();
    if (std::find(playable.begin(), playable.end(), env.pending) == playable.end()) {
        refuse("(" + space.name(static_cast<std::size_t>(index)) + ") is not playable");
    }
}

void VecEnv::step_env(int env_index) {
    Env& env = envs_[static_cast<std::size_t>(env_index)];
    clear_transition(env_index);
    if (env.game == nullptr) {
        return;  // idle since its game ended; its rows already say so
    }
    ++env.seat_decisions[static_cast<std::size_t>(env.game->current_player_index())];
    play(env, env.pending);
    ++env.decisions;

    const std::uint64_t episodes = env.stats.total.episodes;
    play_until_learner(env_index);
    // A finished game was paid in finish_game; the next one starts unpaid.
    if (env.game != nullptr) {
        shape_rewards(env_index, env.stats.total.episodes == episodes);
    }
    write_rows(env_index);
}

void VecEnv::start_games(const std::vector<GameStart>& games) {
    if (config_.auto_reset) {
        throw std::logic_error("start_games needs a VecEnv in schedule mode (auto_reset off)");
    }
    if (!ready_) {
        throw std::logic_error(
            "reset the VecEnv before starting games (first, or after a step failed mid-way)");
    }
    std::vector<bool> taken(envs_.size(), false);
    for (const GameStart& start : games) {
        const std::string where = "game for env " + std::to_string(start.env) + ": ";
        if (start.env < 0 || start.env >= num_envs()) {
            throw std::invalid_argument(where + "no such env");
        }
        if (taken[static_cast<std::size_t>(start.env)] ||
            envs_[static_cast<std::size_t>(start.env)].game != nullptr) {
            throw std::invalid_argument(where + "the env is busy");
        }
        taken[static_cast<std::size_t>(start.env)] = true;
        if (start.scenario < 0 ||
            static_cast<std::size_t>(start.scenario) >= config_.scenarios.size()) {
            throw std::invalid_argument(where + "no such scenario");
        }
        const Scenario& scenario = config_.scenarios[static_cast<std::size_t>(start.scenario)];
        const auto num_players = static_cast<std::size_t>(scenario.num_players);
        if (start.map == nullptr || !config_.dims.fits(*start.map, scenario.num_players)) {
            throw std::invalid_argument(where + "the map is missing or does not fit the dims");
        }
        if (settlement_capacity(*start.map) < 2 * scenario.num_players) {
            throw std::invalid_argument(where + "the map cannot seat the scenario's players");
        }
        if (start.seats.size() != num_players ||
            (!start.lineup_slots.empty() && start.lineup_slots.size() != num_players)) {
            throw std::invalid_argument(
                where + "give a seat kind (and lineup slot) for each of the scenario's " +
                std::to_string(num_players) + " players");
        }
    }
    const std::function<void(int)> task = [&](int index) {
        const GameStart& start = games[static_cast<std::size_t>(index)];
        Env& env = envs_[static_cast<std::size_t>(start.env)];
        std::vector<int> slots = start.lineup_slots;
        if (slots.empty()) {
            slots.resize(start.seats.size());
            std::iota(slots.begin(), slots.end(), 0);
        }
        clear_transition(start.env);
        begin_game(env, start.scenario, start.map, start.game_seed, start.seats,
                   std::move(slots), start.bot_seed);
        play_until_learner(start.env);
        if (env.game != nullptr) {
            shape_rewards(start.env, false);
        }
        write_rows(start.env);
    };
    try {
        pool_->run(static_cast<int>(games.size()), task);
    } catch (...) {
        ready_ = false;
        throw;
    }
}

void VecEnv::expert_actions(
    std::span<std::int32_t> out, ValueEnemies enemies, const ValueWeights& weights) const {
    if (out.size() != num_rows()) {
        throw std::invalid_argument("expert_actions needs one entry per row");
    }
    std::fill(out.begin(), out.end(), -1);
    const GraphActionSpace& space = encoder_.action_space();
    const std::function<void(int)> task = [&](int env_index) {
        const Env& env = envs_[static_cast<std::size_t>(env_index)];
        if (env.game == nullptr) {
            return;
        }
        const Game& game = *env.game;
        const int seat = game.current_player_index();
        if (env.seats[static_cast<std::size_t>(seat)] != SeatKind::learner) {
            return;
        }
        out[row(env_index, seat)] =
            static_cast<std::int32_t>(space.index(game, value_action(game, weights, enemies)));
    };
    pool_->run(config_.num_envs, task);
}

std::vector<FinishedGame> VecEnv::pop_finished() {
    std::vector<FinishedGame> result;
    for (Env& env : envs_) {
        std::move(env.finished.begin(), env.finished.end(), std::back_inserter(result));
        env.finished.clear();
    }
    return result;
}

bool VecEnv::active(int env_index) const {
    return envs_.at(static_cast<std::size_t>(env_index)).game != nullptr;
}

const std::vector<int>& VecEnv::lineup_slots(int env_index) const {
    return envs_.at(static_cast<std::size_t>(env_index)).lineup_slots;
}

void VecEnv::write_rows(int env_index) {
    Env& env = envs_[static_cast<std::size_t>(env_index)];
    // An idle env (schedule mode) writes padding rows.
    const Game* const game = env.game.get();
    const int num_players = game == nullptr ? 0 : static_cast<int>(game->colors().size());
    const int current = game == nullptr ? -1 : game->current_player_index();
    const std::size_t stride = observation_size();
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
        if (buffers_.lineup_slot != nullptr) {
            buffers_.lineup_slot[index] =
                present ? env.lineup_slots[static_cast<std::size_t>(seat)] : -1;
        }
        const std::span<float> observation(buffers_.observations + index * stride, stride);
        const auto slot = static_cast<std::size_t>(seat);
        if (acts || (learns && config_.encode_waiting)) {
            if (critic_encoder_.has_value()) {
                critic_encoder_->encode(*game, seat, observation);
            } else {
                encoder_.encode(*game, seat, observation);
            }
            env.waiting_written[slot] = false;
        } else if (!env.waiting_written[slot]) {
            // Waiting and padded rows only change when a row stops acting.
            if (critic_encoder_.has_value()) {
                critic_encoder_->encode_waiting(observation);
            } else {
                encoder_.encode_waiting(observation);
            }
            env.waiting_written[slot] = true;
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
        throw std::logic_error("env " + std::to_string(env_index) + " holds no game");
    }
    return *env.game;
}

int VecEnv::scenario_index(int env_index) const {
    return envs_.at(static_cast<std::size_t>(env_index)).scenario;
}

const HandTracker& VecEnv::tracker(int env_index, int seat) const {
    const Env& env = envs_.at(static_cast<std::size_t>(env_index));
    if (!config_.trackers) {
        throw std::logic_error("this VecEnv keeps no trackers (trackers is off)");
    }
    if (env.game == nullptr) {
        throw std::logic_error("env " + std::to_string(env_index) + " holds no game");
    }
    const auto tracker = env.trackers.find(static_cast<std::size_t>(seat));
    if (tracker == env.trackers.end()) {
        throw std::out_of_range(
            "env " + std::to_string(env_index) + " keeps no tracker for seat " +
            std::to_string(seat) + " (trackers follow learner seats)");
    }
    return tracker->second;
}

std::uint64_t VecEnv::game_seed(int env_index) const {
    return envs_.at(static_cast<std::size_t>(env_index)).game_seed;
}

std::uint64_t VecEnv::bot_seed(int env_index) const {
    return envs_.at(static_cast<std::size_t>(env_index)).bot_seed;
}

const std::vector<SeatKind>& VecEnv::seat_kinds(int env_index) const {
    return envs_.at(static_cast<std::size_t>(env_index)).seats;
}

}  // namespace cppanatron
