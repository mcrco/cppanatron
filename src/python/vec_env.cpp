#include "bindings.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "cppanatron/value_player.hpp"
#include "cppanatron/vec_env.hpp"

namespace cppanatron::python {
namespace {

using SeatSpec = std::variant<SeatKind, std::string>;

SeatKind seat_kind(const SeatSpec& spec) {
    if (const auto* kind = std::get_if<SeatKind>(&spec)) {
        return *kind;
    }
    const std::string& name = std::get<std::string>(spec);
    if (name == "learner") {
        return SeatKind::learner;
    }
    if (name == "value" || name == "F") {
        return SeatKind::value;
    }
    if (name == "random" || name == "R") {
        return SeatKind::random;
    }
    throw std::invalid_argument("seat kinds are learner, value (F) or random (R), not " + name);
}

template <typename T, int Dims>
using Buffer = nb::ndarray<T, nb::ndim<Dims>, nb::c_contig, nb::device::cpu>;

/** A shared map from a CatanMap, a MapSpec or its dict. */
std::shared_ptr<const CatanMap> map_from_python(nb::handle value) {
    if (nb::isinstance<CatanMap>(value)) {
        return nb::cast<std::shared_ptr<const CatanMap>>(value);
    }
    nb::object module = nb::module_::import_("cppanatron._core");
    return nb::cast<std::shared_ptr<const CatanMap>>(module.attr("CatanMap").attr("from_spec")(value));
}

GameStart game_start(const nb::dict& game) {
    for (auto [key, value] : game) {
        (void)value;
        const std::string name = nb::cast<std::string>(key);
        if (name != "env" && name != "map" && name != "game_seed" && name != "seats" &&
            name != "bot_seed" && name != "scenario" && name != "lineup_slots") {
            throw std::invalid_argument("unknown scheduled game field " + name);
        }
    }
    for (const char* required : {"env", "map", "game_seed", "seats"}) {
        if (!game.contains(required)) {
            throw std::invalid_argument(std::string("a scheduled game needs ") + required);
        }
    }
    GameStart start;
    start.env = nb::cast<int>(game["env"]);
    start.map = map_from_python(game["map"]);
    start.game_seed = nb::cast<std::uint64_t>(game["game_seed"]);
    for (const SeatSpec& seat : nb::cast<std::vector<SeatSpec>>(game["seats"])) {
        start.seats.push_back(seat_kind(seat));
    }
    if (game.contains("bot_seed")) {
        start.bot_seed = nb::cast<std::uint64_t>(game["bot_seed"]);
    }
    if (game.contains("scenario")) {
        start.scenario = nb::cast<int>(game["scenario"]);
    }
    if (game.contains("lineup_slots") && !game["lineup_slots"].is_none()) {
        start.lineup_slots = nb::cast<std::vector<int>>(game["lineup_slots"]);
    }
    return start;
}

/** A VecEnv and the NumPy buffers it writes, which it keeps alive. */
struct PyVecEnv {
    explicit PyVecEnv(VecEnvConfig config) : env(std::move(config)) {}

    VecEnv env;
    nb::dict buffers;
};

template <typename T, int Dims>
T* checked(nb::handle array, const char* name, std::size_t rows, std::size_t columns = 0) {
    Buffer<T, Dims> buffer;
    try {
        buffer = nb::cast<Buffer<T, Dims>>(array, false);
    } catch (const nb::cast_error&) {
        throw std::invalid_argument(
            std::string(name) + " must be a C-contiguous " + std::to_string(Dims) +
            "-d NumPy array of the documented dtype");
    }
    if (buffer.shape(0) != rows || (Dims == 2 && buffer.shape(Dims - 1) != columns)) {
        throw std::invalid_argument(std::string(name) + " has the wrong shape");
    }
    return buffer.data();
}

/**
 * A flag buffer: uint8, or NumPy bool (one byte holding 0 or 1, as PufferLib
 * allocates terminals and truncations), which the engine writes as uint8.
 */
std::uint8_t* checked_flags(nb::handle array, const char* name, std::size_t rows) {
    if (nb::isinstance<Buffer<bool, 1>>(array)) {
        return reinterpret_cast<std::uint8_t*>(checked<bool, 1>(array, name, rows));
    }
    return checked<std::uint8_t, 1>(array, name, rows);
}

nb::dict bucket_to_python(const EpisodeStats::Bucket& bucket, std::size_t seats) {
    nb::dict result;
    const auto episodes = static_cast<double>(bucket.episodes);
    const auto mean = [&](std::uint64_t value) -> nb::object {
        return bucket.episodes == 0 ? nb::cast(0.0)
                                    : nb::cast(static_cast<double>(value) / episodes);
    };
    result["episodes"] = bucket.episodes;
    result["truncated"] = mean(bucket.truncated);
    result["length"] = mean(bucket.turns);
    result["decisions"] = mean(bucket.decisions);
    result["learner_win"] = mean(bucket.learner_wins);
    result["learner_seats"] = mean(bucket.learner_seats);
    result["stuck"] = mean(bucket.stuck);
    nb::list seat_wins;
    for (std::size_t seat = 0; seat < seats; ++seat) {
        seat_wins.append(mean(bucket.seat_wins[seat]));
    }
    result["seat_win"] = seat_wins;
    return result;
}

}  // namespace

void bind_vec_env(nb::module_& module) {
    nb::enum_<SeatKind>(
        module, "SeatKind",
        "Who plays a seat: the LEARNER, the VALUE expert (Catanatron's F) or\n"
        "RANDOM. Strings 'learner', 'value'/'F', 'random'/'R' work too.")
        .value("LEARNER", SeatKind::learner)
        .value("VALUE", SeatKind::value)
        .value("RANDOM", SeatKind::random);

    const Scenario defaults;
    nb::class_<Scenario>(
        module, "Scenario",
        "One kind of game a VecEnv plays: seats, rules, maps, opponents, turn limit\n"
        "and sampling weight.")
        .def(
            "__init__",
            [](Scenario* self, int num_players, std::optional<GameRules> rules,
               std::vector<std::shared_ptr<const CatanMap>> maps, MapType map_type,
               NumberPlacement number_placement, std::optional<std::vector<SeatSpec>> seats,
               bool shuffle_seats, int turn_limit, double weight, ValueEnemies value_enemies) {
                Scenario scenario;
                scenario.num_players = num_players;
                scenario.rules = rules.value_or(GameRules{});
                scenario.maps = std::move(maps);
                scenario.map_type = map_type;
                scenario.number_placement = number_placement;
                if (seats.has_value()) {
                    for (const SeatSpec& seat : *seats) {
                        scenario.seats.push_back(seat_kind(seat));
                    }
                }
                scenario.shuffle_seats = shuffle_seats;
                scenario.turn_limit = turn_limit;
                scenario.weight = weight;
                scenario.value_enemies = value_enemies;
                new (self) Scenario(std::move(scenario));
            },
            nb::kw_only(), nb::arg("num_players") = defaults.num_players,
            nb::arg("rules") = nb::none(),
            nb::arg("maps") = std::vector<std::shared_ptr<const CatanMap>>{},
            nb::arg("map_type") = defaults.map_type,
            nb::arg("number_placement") = defaults.number_placement,
            nb::arg("seats") = nb::none(), nb::arg("shuffle_seats") = defaults.shuffle_seats,
            nb::arg("turn_limit") = defaults.turn_limit, nb::arg("weight") = defaults.weight,
            nb::arg("value_enemies") = defaults.value_enemies,
            "``maps`` is a pool of shared CatanMaps drawn uniformly at each reset;\n"
            "without it every reset draws a fresh ``map_type`` board. ``seats`` lists\n"
            "a SeatKind per seat (default: all learners), shuffled per reset with\n"
            "``shuffle_seats``. Games are truncated at ``turn_limit`` turns.")
        .def_ro("num_players", &Scenario::num_players)
        .def_ro("rules", &Scenario::rules)
        .def_ro("maps", &Scenario::maps)
        .def_ro("map_type", &Scenario::map_type)
        .def_ro("number_placement", &Scenario::number_placement)
        .def_ro("seats", &Scenario::seats)
        .def_ro("shuffle_seats", &Scenario::shuffle_seats)
        .def_ro("turn_limit", &Scenario::turn_limit)
        .def_ro("weight", &Scenario::weight)
        .def_ro("value_enemies", &Scenario::value_enemies);

    nb::class_<PyVecEnv>(
        module, "VecEnv",
        "Many games stepped together for PufferLib-style training (DESIGN 1.4).\n\n"
        "Each env has ``max_players`` rows (row = env * max_players + seat). After\n"
        "``bind`` (or ``allocate``) and ``reset``, every ``step`` reads the actions\n"
        "of acting rows, plays forced moves and bot seats until a learner must\n"
        "decide, and writes every row. A finished game sets rewards (1 to the\n"
        "winner) and terminal flags on all of the env's rows; a game cut off at\n"
        "the turn limit pays nothing and sets both terminal and truncation flags\n"
        "(PufferLib's GAE reads terminals only). Either way the same step's\n"
        "observations are already the next game's.")
        .def(
            "__init__",
            [](PyVecEnv* self, std::vector<Scenario> scenarios, int num_envs,
               std::optional<EncoderDims> dims, InfoLevel level, std::uint64_t seed,
               int num_threads, float vp_reward, bool encode_waiting, bool auto_reset,
               std::optional<InfoLevel> critic_level) {
                VecEnvConfig config;
                if (!dims.has_value()) {
                    std::vector<std::shared_ptr<const CatanMap>> maps;
                    int players = 2;
                    for (const Scenario& scenario : scenarios) {
                        players = std::max(players, scenario.num_players);
                        maps.insert(maps.end(), scenario.maps.begin(), scenario.maps.end());
                        if (scenario.maps.empty()) {
                            maps.push_back(std::make_shared<const CatanMap>(CatanMap::build(
                                scenario.map_type, 0, scenario.number_placement)));
                        }
                    }
                    dims = EncoderDims::covering(maps, players);
                }
                config.scenarios = std::move(scenarios);
                config.num_envs = num_envs;
                config.dims = *dims;
                config.level = level;
                config.seed = seed;
                config.num_threads = num_threads;
                config.vp_reward = vp_reward;
                config.encode_waiting = encode_waiting;
                config.auto_reset = auto_reset;
                config.critic_level = critic_level;
                new (self) PyVecEnv(std::move(config));
            },
            nb::arg("scenarios"), nb::kw_only(), nb::arg("num_envs") = 1,
            nb::arg("dims") = nb::none(), nb::arg("level") = InfoLevel::private_,
            nb::arg("seed") = 0, nb::arg("num_threads") = 1, nb::arg("vp_reward") = 0.0F,
            nb::arg("encode_waiting") = false, nb::arg("auto_reset") = true,
            nb::arg("critic_level") = nb::none(),
            "``dims`` defaults to the smallest covering every scenario's maps and\n"
            "players. ``vp_reward`` > 0 adds that much per victory point to win\n"
            "gained as shaping. ``encode_waiting`` also encodes learner rows that\n"
            "wait (otherwise zeros with the no-op mask).\n\n"
            "``auto_reset=False`` is schedule mode, for evaluation: ``reset`` leaves\n"
            "every env idle, ``start_games`` puts exact games (map, seeds, seats) into\n"
            "idle envs, a finished env stays idle, and ``pop_finished`` reports how\n"
            "each game ended. Pass ``dims`` covering the maps you will schedule.\n\n"
            "``critic_level`` (an InfoLevel) appends a critic block to every encoded\n"
            "row: the players block at that level, the only block that differs\n"
            "between levels (see ``layout()``).")
        .def_prop_ro("num_envs", [](const PyVecEnv& self) { return self.env.num_envs(); })
        .def_prop_ro("max_players", [](const PyVecEnv& self) { return self.env.max_players(); })
        .def_prop_ro("num_rows", [](const PyVecEnv& self) { return self.env.num_rows(); })
        .def_prop_ro(
            "observation_size", [](const PyVecEnv& self) { return self.env.observation_size(); })
        .def_prop_ro(
            "action_size",
            [](const PyVecEnv& self) { return self.env.encoder().action_space().size(); })
        .def_prop_ro("dims", [](const PyVecEnv& self) { return self.env.config().dims; })
        .def_prop_ro("level", [](const PyVecEnv& self) { return self.env.config().level; })
        .def_prop_ro(
            "critic_level", [](const PyVecEnv& self) { return self.env.config().critic_level; })
        .def(
            "layout",
            [](const PyVecEnv& self) {
                nb::dict result = nb::cast<nb::dict>(
                    nb::cast(self.env.encoder()).attr("layout")());
                if (const GraphEncoder* critic = self.env.critic_encoder()) {
                    result["critic_players"] = nb::make_tuple(
                        self.env.encoder().size(),
                        nb::make_tuple(self.env.max_players(), critic->player_size()));
                }
                return result;
            },
            "``{block: (offset, shape)}`` of a row: the encoder's blocks, then\n"
            "``critic_players`` (max_players x player features at critic_level) when\n"
            "a critic level is set. Its feature names are the players block's.")
        .def_prop_ro(
            "encoder", [](const PyVecEnv& self) { return self.env.encoder(); },
            "A copy of the env's GraphEncoder (layout, feature names, action space).")
        .def_prop_ro(
            "buffers", [](const PyVecEnv& self) { return self.buffers; },
            "The bound arrays by name.")
        .def(
            "bind",
            [](PyVecEnv& self, nb::handle observations, nb::handle actions, nb::handle rewards,
               nb::handle terminals, nb::handle truncations, nb::handle acting,
               nb::handle learner, nb::handle seat, nb::handle env_index,
               nb::handle num_players, nb::handle lineup_slot) {
                const std::size_t rows = self.env.num_rows();
                VecEnvBuffers buffers;
                buffers.observations = checked<float, 2>(
                    observations, "observations (float32)", rows, self.env.observation_size());
                buffers.actions = checked<std::int32_t, 1>(actions, "actions (int32)", rows);
                buffers.rewards = checked<float, 1>(rewards, "rewards (float32)", rows);
                buffers.terminals = checked_flags(terminals, "terminals (uint8 or bool)", rows);
                buffers.truncations =
                    checked_flags(truncations, "truncations (uint8 or bool)", rows);
                buffers.acting = checked_flags(acting, "acting (uint8 or bool)", rows);
                buffers.learner = checked_flags(learner, "learner (uint8 or bool)", rows);
                buffers.seat = checked<std::int32_t, 1>(seat, "seat (int32)", rows);
                buffers.env_index =
                    checked<std::int32_t, 1>(env_index, "env_index (int32)", rows);
                buffers.num_players = checked<std::int32_t, 1>(
                    num_players, "num_players (int32)",
                    static_cast<std::size_t>(self.env.num_envs()));
                if (!lineup_slot.is_none()) {
                    buffers.lineup_slot =
                        checked<std::int32_t, 1>(lineup_slot, "lineup_slot (int32)", rows);
                }
                self.env.bind(buffers);
                nb::dict bound;
                bound["observations"] = observations;
                bound["actions"] = actions;
                bound["rewards"] = rewards;
                bound["terminals"] = terminals;
                bound["truncations"] = truncations;
                bound["acting"] = acting;
                bound["learner"] = learner;
                bound["seat"] = seat;
                bound["env_index"] = env_index;
                bound["num_players"] = num_players;
                if (!lineup_slot.is_none()) {
                    bound["lineup_slot"] = lineup_slot;
                }
                self.buffers = bound;
            },
            nb::kw_only(), nb::arg("observations"), nb::arg("actions"), nb::arg("rewards"),
            nb::arg("terminals"), nb::arg("truncations"), nb::arg("acting"),
            nb::arg("learner"), nb::arg("seat"), nb::arg("env_index"),
            nb::arg("num_players"), nb::arg("lineup_slot") = nb::none(),
            "Bind caller-owned arrays (no conversion; wrong dtypes raise):\n"
            "observations float32 [rows, observation_size], actions int32 [rows],\n"
            "rewards float32 [rows], terminals/truncations/acting/learner uint8 or\n"
            "bool [rows], seat/env_index int32 [rows], num_players int32 [num_envs];\n"
            "optionally lineup_slot int32 [rows] (who plays each row's seat, -1 for\n"
            "padding).")
        .def(
            "allocate",
            [](nb::handle self) {
                const PyVecEnv& env = nb::cast<const PyVecEnv&>(self);
                nb::object numpy = nb::module_::import_("numpy");
                const auto zeros = [&](nb::object shape, const char* dtype) {
                    return numpy.attr("zeros")(shape, nb::arg("dtype") = dtype);
                };
                const std::size_t rows = env.env.num_rows();
                nb::dict arrays;
                arrays["observations"] =
                    zeros(nb::make_tuple(rows, env.env.observation_size()), "float32");
                arrays["actions"] = zeros(nb::make_tuple(rows), "int32");
                arrays["rewards"] = zeros(nb::make_tuple(rows), "float32");
                for (const char* name : {"terminals", "truncations", "acting", "learner"}) {
                    arrays[name] = zeros(nb::make_tuple(rows), "uint8");
                }
                arrays["seat"] = zeros(nb::make_tuple(rows), "int32");
                arrays["env_index"] = zeros(nb::make_tuple(rows), "int32");
                arrays["num_players"] = zeros(nb::make_tuple(env.env.num_envs()), "int32");
                arrays["lineup_slot"] = zeros(nb::make_tuple(rows), "int32");
                self.attr("bind")(**arrays);
                return arrays;
            },
            "Allocate and bind fresh arrays; returns them by name.")
        .def(
            "reset",
            [](PyVecEnv& self, std::optional<std::uint64_t> seed) {
                nb::gil_scoped_release release;
                self.env.reset(seed);
            },
            nb::arg("seed") = nb::none(),
            "Start a fresh game in every env and write every row. A ``seed`` (or the\n"
            "constructor's, on the first reset) reseeds every env, so equal seeds\n"
            "replay equal runs; ``seed=None`` later continues the streams and starts\n"
            "new games (Gymnasium semantics).")
        .def(
            "step",
            [](PyVecEnv& self) {
                nb::gil_scoped_release release;
                self.env.step();
            },
            "Apply the acting rows' actions and advance to the next learner decisions.")
        .def(
            "pop_stats",
            [](PyVecEnv& self) {
                const EpisodeStats stats = self.env.pop_stats();
                const auto seats = static_cast<std::size_t>(self.env.max_players());
                nb::dict result = bucket_to_python(stats.total, seats);
                nb::list scenarios;
                for (const auto& bucket : stats.scenarios) {
                    scenarios.append(bucket_to_python(bucket, seats));
                }
                result["scenarios"] = scenarios;
                return result;
            },
            "Episodes finished since the last call: count, and means of truncated,\n"
            "stuck (truncated with no playable action), length (turns), learner\n"
            "decisions, learner_win (won by a learner seat), learner_seats and\n"
            "seat_win per seat; the same per scenario.")
        .def(
            "game", [](const PyVecEnv& self, int env_index) { return Game(self.env.game(env_index)); },
            nb::arg("env_index"), "A copy of an env's current game.")
        .def(
            "scenario_index",
            [](const PyVecEnv& self, int env_index) { return self.env.scenario_index(env_index); },
            nb::arg("env_index"))
        .def(
            "seat_kinds",
            [](const PyVecEnv& self, int env_index) { return self.env.seat_kinds(env_index); },
            nb::arg("env_index"))
        .def(
            "start_games",
            [](PyVecEnv& self, const std::vector<nb::dict>& games) {
                std::vector<GameStart> starts;
                for (const nb::dict& game : games) {
                    starts.push_back(game_start(game));
                }
                nb::gil_scoped_release release;
                self.env.start_games(starts);
            },
            nb::arg("games"),
            "Schedule mode: start exact games in idle envs, in parallel. Each game is a\n"
            "dict: ``env``, ``map`` (CatanMap, MapSpec or its dict), ``game_seed``,\n"
            "``seats`` (SeatKind or 'learner'/'value'/'F'/'random'/'R' per seat),\n"
            "and optionally ``bot_seed`` (0), ``scenario`` (0: players, rules, turn\n"
            "limit) and ``lineup_slots`` (who plays each seat; default 0..n-1). Each\n"
            "plays until a learner must decide (bots-only games play to the end).")
        .def(
            "start_game",
            [](PyVecEnv& self, int env_index, nb::handle map, std::uint64_t game_seed,
               std::vector<SeatSpec> seats, std::uint64_t bot_seed, int scenario,
               std::optional<std::vector<int>> lineup_slots) {
                GameStart start;
                start.env = env_index;
                start.scenario = scenario;
                start.map = map_from_python(map);
                start.game_seed = game_seed;
                start.bot_seed = bot_seed;
                for (const SeatSpec& seat : seats) {
                    start.seats.push_back(seat_kind(seat));
                }
                start.lineup_slots = lineup_slots.value_or(std::vector<int>{});
                nb::gil_scoped_release release;
                self.env.start_games({start});
            },
            nb::arg("env_index"), nb::kw_only(), nb::arg("map"), nb::arg("game_seed"),
            nb::arg("seats"), nb::arg("bot_seed") = 0, nb::arg("scenario") = 0,
            nb::arg("lineup_slots") = nb::none(),
            "Schedule mode: start one exact game (see start_games).")
        .def(
            "expert_actions",
            [](const PyVecEnv& self, nb::handle out, ValueEnemies enemies) -> nb::object {
                nb::object result = nb::borrow(out);
                if (out.is_none()) {
                    result = nb::module_::import_("numpy").attr("empty")(
                        self.env.num_rows(), nb::arg("dtype") = "int32");
                }
                std::int32_t* data = checked<std::int32_t, 1>(
                    result, "out (int32)", self.env.num_rows());
                {
                    nb::gil_scoped_release release;
                    self.env.expert_actions(std::span<std::int32_t>(data, self.env.num_rows()),
                                            enemies);
                }
                return result;
            },
            nb::arg("out") = nb::none(), nb::kw_only(), nb::arg("enemies") = ValueEnemies::next,
            "Imitation labels: the action index Catanatron's value function player\n"
            "(F, value_action) chooses for every acting row, -1 elsewhere, computed\n"
            "on the env threads into ``out`` (int32 [rows], allocated if None).")
        .def(
            "pop_finished",
            [](PyVecEnv& self) {
                nb::list result;
                for (const FinishedGame& game : self.env.pop_finished()) {
                    nb::dict row;
                    row["env"] = game.env;
                    row["scenario"] = game.scenario;
                    row["game_seed"] = game.game_seed;
                    row["bot_seed"] = game.bot_seed;
                    row["seats"] = nb::cast(game.seats);
                    row["lineup_slots"] = nb::cast(game.lineup_slots);
                    row["winner"] = game.winner;
                    row["winner_slot"] =
                        game.winner < 0
                            ? -1
                            : game.lineup_slots[static_cast<std::size_t>(game.winner)];
                    row["turns"] = game.turns;
                    row["truncated"] = game.truncated;
                    row["stuck"] = game.stuck;
                    row["decisions"] = nb::cast(game.decisions);
                    result.append(row);
                }
                return result;
            },
            "Schedule mode: games finished since the last call, by env: ``env``,\n"
            "``scenario``, ``game_seed``, ``bot_seed``, ``seats``, ``lineup_slots``,\n"
            "``winner`` (seat, -1 if truncated) and ``winner_slot``, ``turns``,\n"
            "``truncated``, ``stuck`` and learner ``decisions`` per seat.")
        .def(
            "active", [](const PyVecEnv& self, int env_index) { return self.env.active(env_index); },
            nb::arg("env_index"), "Whether the env holds a game (always, outside schedule mode).")
        .def(
            "idle_envs",
            [](const PyVecEnv& self) {
                std::vector<int> idle;
                for (int env = 0; env < self.env.num_envs(); ++env) {
                    if (!self.env.active(env)) {
                        idle.push_back(env);
                    }
                }
                return idle;
            },
            "Schedule mode: envs without a game, ready for start_games.")
        .def(
            "lineup_slots",
            [](const PyVecEnv& self, int env_index) { return self.env.lineup_slots(env_index); },
            nb::arg("env_index"),
            "Who plays each seat of the env's game: the scheduled lineup slots, or\n"
            "positions in the scenario's seat list for auto-reset games.")
        .def(
            "game_seed",
            [](const PyVecEnv& self, int env_index) { return self.env.game_seed(env_index); },
            nb::arg("env_index"), "Seed of the env's current game (dice, cards, steals).")
        .def(
            "bot_seed",
            [](const PyVecEnv& self, int env_index) { return self.env.bot_seed(env_index); },
            nb::arg("env_index"), "Seed of the current game's bot stream (random bots).");
}

}  // namespace cppanatron::python
