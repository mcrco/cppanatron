#include "bindings.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

#include <nanobind/stl/optional.h>

#include "cppanatron/action_space.hpp"
#include "cppanatron/mcts.hpp"
#include "cppanatron/observation.hpp"

namespace cppanatron::python {
namespace {

/** A search tree plus the layout its root and leaf observations use. */
struct Search {
    Search(
        const Game& game,
        FlatActionSpace action_space,
        ObservationLayout observation_layout,
        double c_puct,
        std::uint64_t seed,
        bool canonical_pruning)
        : layout(std::move(observation_layout)),
          observation_size(full_observation_size(static_cast<int>(game.colors().size()), layout)),
          action_space_size(action_space.size()),
          search(game, std::move(action_space), c_puct, seed, canonical_pruning) {}

    void write(const Game& game, int player, OutputVector<float>& out) const {
        if (out.shape(0) != observation_size) {
            throw std::invalid_argument("observation buffer has the wrong size");
        }
        write_full_observation(game, player, layout, out.data(), out.shape(0));
    }

    ObservationLayout layout;
    std::size_t observation_size;
    std::size_t action_space_size;
    MCTSSearch search;
};

std::span<const float> logits(const Search& self, const InputVector<float>& policy_logits) {
    if (policy_logits.shape(0) != self.action_space_size) {
        throw std::invalid_argument("policy logits must cover the flat action space");
    }
    return {policy_logits.data(), policy_logits.shape(0)};
}

}  // namespace

void bind_search(nb::module_& module) {
    nb::class_<MCTSSearchMetrics>(module, "MCTSSearchMetrics")
        .def_ro("simulations", &MCTSSearchMetrics::simulations)
        .def_ro("principal_variation_depth", &MCTSSearchMetrics::principal_variation_depth)
        .def_ro("maximum_depth", &MCTSSearchMetrics::maximum_depth)
        .def_ro("mean_depth", &MCTSSearchMetrics::mean_depth)
        .def_ro("root_value", &MCTSSearchMetrics::root_value)
        .def_ro("retained_root_visits", &MCTSSearchMetrics::retained_root_visits)
        .def_ro("pruned_actions", &MCTSSearchMetrics::pruned_actions)
        .def_ro("coalesced_outcomes", &MCTSSearchMetrics::coalesced_outcomes)
        .def_ro("tree_reused", &MCTSSearchMetrics::tree_reused);

    nb::class_<Search>(
        module, "MCTSSearch",
        "Pull-based policy/value MCTS. The caller evaluates the root and every\n"
        "selected leaf with its own inference backend, so PyTorch never runs in C++\n"
        "and leaves from many searches can share one batched model call.")
        .def(
            nb::init<const Game&, FlatActionSpace, ObservationLayout, double, std::uint64_t, bool>(),
            nb::arg("game"), nb::arg("action_space"), nb::arg("layout"), nb::kw_only(),
            nb::arg("c_puct") = 1.5, nb::arg("seed") = 0, nb::arg("canonical_pruning") = false)
        .def_prop_ro("observation_size", [](const Search& self) { return self.observation_size; })
        .def_prop_ro("action_space_size", [](const Search& self) { return self.action_space_size; })
        .def(
            "initialize_root",
            [](Search& self, InputVector<float> policy_logits) {
                self.search.initialize_root(logits(self, policy_logits));
            },
            nb::arg("policy_logits"), nb::call_guard<nb::gil_scoped_release>(),
            "Expand the root from flat policy logits without counting a simulation.")
        .def(
            "add_root_dirichlet_noise",
            [](Search& self, double alpha, double fraction) {
                self.search.add_root_dirichlet_noise(alpha, fraction);
            },
            nb::arg("alpha"), nb::arg("fraction"))
        .def(
            "root_observation",
            [](const Search& self, OutputVector<float> out) {
                const Game& root = self.search.root_game();
                self.write(root, root.current_player_index(), out);
                return root.current_player_index();
            },
            nb::arg("out").noconvert(), nb::call_guard<nb::gil_scoped_release>(),
            "Write the root observation into ``out`` and return the player to move.")
        .def(
            "select_leaf",
            [](Search& self, OutputVector<float> out) -> std::optional<int> {
                if (out.shape(0) != self.observation_size) {
                    throw std::invalid_argument("observation buffer has the wrong size");
                }
                const Game* leaf = self.search.select_leaf();
                if (leaf == nullptr) {
                    return std::nullopt;
                }
                const int player = self.search.pending_player_index();
                self.write(*leaf, player, out);
                return player;
            },
            nb::arg("out").noconvert(), nb::call_guard<nb::gil_scoped_release>(),
            "Start a simulation. Returns the leaf's player to move after writing its\n"
            "observation into ``out``, or None when the simulation ended at a\n"
            "terminal or dead-end node and was backed up internally.")
        .def(
            "evaluate_leaf",
            [](Search& self, InputVector<float> policy_logits, double value) {
                self.search.evaluate_leaf(logits(self, policy_logits), value);
            },
            nb::arg("policy_logits"), nb::arg("value"), nb::call_guard<nb::gil_scoped_release>(),
            "Expand and back up the pending leaf; value is from its player's view.")
        .def_prop_ro("has_pending_leaf", [](const Search& self) { return self.search.has_pending_leaf(); })
        .def_prop_ro("root_expanded", [](const Search& self) { return self.search.root_expanded(); })
        .def_prop_ro(
            "root_game", [](const Search& self) { return Game(self.search.root_game()); },
            "A copy of the current root position.")
        .def(
            "root_visits", [](const Search& self) { return to_numpy(self.search.root_visits()); },
            "Visit counts over the flat action space.")
        .def(
            "root_action_values",
            [](const Search& self) { return to_numpy(self.search.root_action_values()); },
            "Root action Q from the root player's view; NaN where unvisited.")
        .def("metrics", [](const Search& self) { return self.search.metrics(); })
        .def("reset_metrics", [](Search& self) { self.search.reset_metrics(); })
        .def(
            "advance",
            [](Search& self, std::size_t action_index) { return self.search.advance(action_index); },
            nb::arg("action_index"),
            "Re-root at a deterministic child. False means the caller must build a\n"
            "fresh search from the actual post-action game.");
}

}  // namespace cppanatron::python
