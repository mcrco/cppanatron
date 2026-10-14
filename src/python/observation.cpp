#include "bindings.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>

#include "cppanatron/action_space.hpp"
#include "cppanatron/batch.hpp"
#include "cppanatron/observation.hpp"

namespace cppanatron::python {
namespace {

using Position = std::pair<int, int>;

ObservationLayout make_layout(
    int width,
    int height,
    const std::map<int, Position>& nodes,
    const std::map<std::array<int, 2>, Position>& edges,
    const std::map<std::array<int, 3>, Position>& tiles) {
    std::vector<NodePosition> node_positions;
    for (const auto& [node, position] : nodes) {
        node_positions.push_back({node, position.first, position.second});
    }
    std::vector<EdgePosition> edge_positions;
    for (const auto& [edge, position] : edges) {
        edge_positions.push_back({{edge[0], edge[1]}, position.first, position.second});
    }
    std::vector<TilePosition> tile_positions;
    for (const auto& [coordinate, position] : tiles) {
        tile_positions.push_back(
            {{coordinate[0], coordinate[1], coordinate[2]}, position.first, position.second});
    }
    return {
        width, height, std::move(node_positions), std::move(edge_positions),
        std::move(tile_positions)};
}

/** GameBatch plus the sizes needed to validate caller-owned buffers. */
struct Batch {
    Batch(BatchConfig batch_config, ObservationLayout layout)
        : config(batch_config),
          action_space_size(FlatActionSpace(config.num_players, config.map_type).size()),
          observation_size(full_observation_size(config.num_players, layout)),
          batch(config, std::move(layout)) {}

    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;

    BatchConfig config;
    std::size_t action_space_size;
    std::size_t observation_size;
    GameBatch batch;
};

template <typename Array>
void require_rows(const Array& array, std::size_t rows, const char* name) {
    if (array.shape(0) != rows) {
        throw std::invalid_argument(
            std::string(name) + " must have one row per agent (" + std::to_string(rows) + ")");
    }
}

using Rows = nb::ndarray<std::uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

void bind_buffers(
    Batch& self,
    Rows observations,
    std::size_t action_mask_offset,
    std::size_t observation_offset,
    OutputVector<std::int32_t> actions,
    OutputVector<float> rewards,
    OutputVector<bool> terminals,
    OutputVector<bool> truncations,
    OutputVector<bool> masks) {
    const auto agents =
        static_cast<std::size_t>(self.config.num_envs) * static_cast<std::size_t>(self.config.num_players);
    require_rows(observations, agents, "observations");
    require_rows(actions, agents, "actions");
    require_rows(rewards, agents, "rewards");
    require_rows(terminals, agents, "terminals");
    require_rows(truncations, agents, "truncations");
    require_rows(masks, agents, "masks");
    const std::size_t stride = observations.shape(1);
    if (action_mask_offset + self.action_space_size > stride ||
        observation_offset + self.observation_size * sizeof(float) > stride) {
        throw std::invalid_argument("observation rows are too small for the mask and features");
    }
    self.batch.bind({
        observations.data(),
        stride,
        action_mask_offset,
        observation_offset,
        actions.data(),
        rewards.data(),
        reinterpret_cast<std::uint8_t*>(terminals.data()),
        reinterpret_cast<std::uint8_t*>(truncations.data()),
        reinterpret_cast<std::uint8_t*>(masks.data()),
    });
}

}  // namespace

void bind_observation(nb::module_& module) {
    nb::class_<ObservationLayout>(
        module, "ObservationLayout",
        "Board-tensor cell of every node, edge and tile for full observations.")
        .def(
            "__init__",
            [](ObservationLayout* self, int width, int height, const std::map<int, Position>& nodes,
               const std::map<std::array<int, 2>, Position>& edges,
               const std::map<std::array<int, 3>, Position>& tiles) {
                new (self) ObservationLayout(make_layout(width, height, nodes, edges, tiles));
            },
            nb::arg("width"), nb::arg("height"), nb::arg("nodes"), nb::arg("edges"),
            nb::arg("tiles"),
            "Positions map ids to ``(x, y)`` cells: ``nodes`` by node id, ``edges`` by\n"
            "``(a, b)`` (either orientation) and ``tiles`` by cube coordinate.")
        .def_prop_ro("width", &ObservationLayout::width)
        .def_prop_ro("height", &ObservationLayout::height);

    module.def("full_numeric_observation_size", &full_numeric_observation_size, nb::arg("num_players"));
    module.def(
        "board_observation_size", &board_observation_size, nb::arg("num_players"),
        nb::arg("layout"));
    module.def(
        "full_observation_size", &full_observation_size, nb::arg("num_players"), nb::arg("layout"));
    module.def(
        "write_full_observation",
        [](const Game& game, int base_player, const ObservationLayout& layout,
           OutputVector<float> out) {
            write_full_observation(game, base_player, layout, out.data(), out.shape(0));
        },
        nb::arg("game"), nb::arg("base_player"), nb::arg("layout"), nb::arg("out").noconvert(),
        nb::call_guard<nb::gil_scoped_release>(),
        "Write numeric features then the flattened board tensor from base_player's view.");
    module.def(
        "full_observation",
        [](const Game& game, int base_player, const ObservationLayout& layout) {
            std::vector<float> features(full_observation_size(
                static_cast<int>(game.colors().size()), layout));
            write_full_observation(game, base_player, layout, features.data(), features.size());
            return to_numpy(std::move(features));
        },
        nb::arg("game"), nb::arg("base_player"), nb::arg("layout"));
    module.def("production_sum", &production_sum, nb::arg("game"), nb::arg("color"));

    nb::enum_<RewardFunction>(module, "RewardFunction")
        .value("SHAPED", RewardFunction::shaped)
        .value("WIN", RewardFunction::win);

    nb::class_<Batch>(
        module, "GameBatch",
        "Many games stepped in one call, writing straight into caller buffers.")
        .def(
            "__init__",
            [](Batch* self, int num_envs, int num_players, MapType map_type,
               const ObservationLayout& layout, NumberPlacement number_placement,
               std::optional<int> discard_limit, std::optional<bool> friendly_robber,
               std::optional<int> vps_to_win, std::optional<GameRules> rules,
               RewardFunction reward_function, int turns_limit) {
                BatchConfig config;
                config.num_envs = num_envs;
                config.num_players = num_players;
                config.map_type = map_type;
                config.number_placement = number_placement;
                config.rules = resolve_rules(rules, discard_limit, friendly_robber, vps_to_win);
                config.reward_function = reward_function;
                config.turns_limit = turns_limit;
                validate_rules(config.rules);
                new (self) Batch(config, layout);
            },
            nb::arg("num_envs"), nb::arg("num_players"), nb::arg("map_type"), nb::arg("layout"),
            nb::kw_only(), nb::arg("number_placement") = NumberPlacement::official_spiral,
            nb::arg("discard_limit") = nb::none(), nb::arg("friendly_robber") = nb::none(),
            nb::arg("vps_to_win") = nb::none(), nb::arg("rules") = nb::none(),
            nb::arg("reward_function") = RewardFunction::shaped, nb::arg("turns_limit") = 1000,
            "``rules`` sets every game rule; otherwise the base game's apply, with\n"
            "``discard_limit`` (7), ``friendly_robber`` (False) and ``vps_to_win`` (10)\n"
            "as overrides, as for Game.")
        .def_prop_ro("action_space_size", [](const Batch& self) { return self.action_space_size; })
        .def_prop_ro("observation_size", [](const Batch& self) { return self.observation_size; })
        .def(
            "bind", &bind_buffers, nb::arg("observations").noconvert(),
            nb::arg("action_mask_offset"), nb::arg("observation_offset"),
            nb::arg("actions").noconvert(), nb::arg("rewards").noconvert(),
            nb::arg("terminals").noconvert(), nb::arg("truncations").noconvert(),
            nb::arg("masks").noconvert(),
            nb::keep_alive<1, 2>(), nb::keep_alive<1, 5>(), nb::keep_alive<1, 6>(),
            nb::keep_alive<1, 7>(), nb::keep_alive<1, 8>(), nb::keep_alive<1, 9>(),
            "Bind per-agent buffers (one row per env and seat). ``observations`` holds\n"
            "raw rows with a bool action mask at ``action_mask_offset`` and float32\n"
            "features at ``observation_offset``. The batch keeps the buffers alive.")
        .def(
            "reset_all",
            [](Batch& self, InputVector<std::uint64_t> map_seeds,
               InputVector<std::uint64_t> game_seeds) {
                if (map_seeds.shape(0) != game_seeds.shape(0)) {
                    throw std::invalid_argument("map and game seeds must have the same length");
                }
                self.batch.reset_all(map_seeds.data(), game_seeds.data(), map_seeds.shape(0));
            },
            nb::arg("map_seeds"), nb::arg("game_seeds"), nb::call_guard<nb::gil_scoped_release>())
        .def(
            "reset_at",
            [](Batch& self, int env_index, std::uint64_t map_seed, std::uint64_t game_seed,
               bool preserve_transition) {
                self.batch.reset_at(env_index, map_seed, game_seed, preserve_transition);
            },
            nb::arg("env_index"), nb::arg("map_seed"), nb::arg("game_seed"), nb::kw_only(),
            nb::arg("preserve_transition") = false)
        .def(
            "step", [](Batch& self) { self.batch.step(); },
            nb::call_guard<nb::gil_scoped_release>(),
            "Apply ``actions`` to every game and write rewards, flags, masks and features.");
}

}  // namespace cppanatron::python
