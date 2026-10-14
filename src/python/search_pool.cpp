#include "bindings.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "cppanatron/codec.hpp"
#include "cppanatron/flat_codec.hpp"
#include "cppanatron/observation.hpp"
#include "cppanatron/search_pool.hpp"
#include "cppanatron/tracker.hpp"

namespace cppanatron::python {
namespace {

template <typename T>
using OutputMatrix = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
template <typename T>
using InputMatrix = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

}  // namespace

void bind_codecs(nb::module_& module) {
    nb::class_<ActionCodec>(
        module, "ActionCodec",
        "Maps legal actions to a fixed index space (implemented in C++).")
        .def_prop_ro("size", &ActionCodec::size)
        .def("__len__", &ActionCodec::size)
        .def("index", &ActionCodec::index, nb::arg("game"), nb::arg("action"),
             "Index of a legal action of ``game``.")
        .def("decode", &ActionCodec::decode, nb::arg("game"), nb::arg("index"),
             "The action ``index`` means for ``game``'s player to move.")
        .def(
            "mask",
            [](const ActionCodec& codec, const Game& game) {
                std::vector<std::uint8_t> mask(codec.size());
                codec.mask(game, mask);
                return to_numpy(std::move(mask));
            },
            nb::arg("game"), "1 at every playable action's index (uint8).");
    nb::class_<ObservationEncoder>(
        module, "ObservationEncoder",
        "Writes a network input for one game as one seat sees it (implemented in C++).")
        .def_prop_ro("size", &ObservationEncoder::size)
        .def(
            "encode",
            [](const ObservationEncoder& encoder, const Game& game, int observer) {
                std::vector<float> row(encoder.size());
                encoder.encode(game, observer, row);
                return to_numpy(std::move(row));
            },
            nb::arg("game"), nb::arg("observer"));
}

void bind_search_pool(nb::module_& module) {
    nb::class_<FlatActionCodec, ActionCodec>(
        module, "FlatActionCodec",
        "The v1 flat action table (FlatActionSpace) as an ActionCodec, for one map\n"
        "type. A stand-in until the graph action space lands.")
        .def(nb::init<int, MapType>(), nb::arg("num_players"), nb::arg("map_type"));

    nb::class_<FlatObservationEncoder, ObservationEncoder>(
        module, "FlatObservationEncoder",
        "The v1 full-information observation (write_full_observation) as an\n"
        "ObservationEncoder. Perfect information and no action mask: a stand-in\n"
        "until the graph encoder lands.")
        .def(nb::init<int, ObservationLayout>(), nb::arg("num_players"), nb::arg("layout"));

    nb::enum_<RootSelection>(module, "RootSelection", "How a search chooses root actions.")
        .value("PUCT", RootSelection::puct, "PUCT, with optional Dirichlet noise.")
        .value(
            "GUMBEL", RootSelection::gumbel,
            "Gumbel-Top-k with Sequential Halving (Danihelka et al., 2022).");

    const SearchConfig defaults;
    nb::class_<SearchConfig>(
        module, "SearchConfig",
        "Settings shared by every search of a SearchPool.\n\n"
        "c_puct: PUCT constant for values in [0, 1] (half of MCTSSearch's; v1's 1.5\n"
        "  is 0.75). simulations: per search, root evaluations excluded.\n"
        "max_pending: leaves one search may have in flight (virtual visits).\n"
        "root: RootSelection. dirichlet_alpha (<= 0 disables), dirichlet_fraction.\n"
        "gumbel_considered (m), gumbel_scale (0 = no noise), gumbel_c_visit,\n"
        "  gumbel_c_scale (sigma = (c_visit + max visits) * c_scale * q),\n"
        "  gumbel_rescale_values (min-max rescale completed Q).\n"
        "determinizations: root worlds sampled from a HandTracker (0 = search the\n"
        "  given game as is). max_players: width P of the value rows.\n"
        "fpu_value: value of unvisited actions when in [0, 1] (v1: 0.5); otherwise\n"
        "  the node's mean value minus fpu_reduction. skip_forced: single-action\n"
        "  positions below the root need no network call. release_games: keep games\n"
        "  only where children are still created (rebuilt on demand).\n"
        "threads: for collect/apply across searches (0 = all cores).")
        .def(
            "__init__",
            [](SearchConfig* self, double c_puct, int simulations, int max_pending, RootSelection root,
               double dirichlet_alpha, double dirichlet_fraction, int gumbel_considered,
               double gumbel_scale, double gumbel_c_visit, double gumbel_c_scale,
               bool gumbel_rescale_values, int determinizations, int max_players, double fpu_value,
               double fpu_reduction, bool skip_forced, bool release_games, int threads) {
                new (self) SearchConfig{
                    c_puct, simulations, max_pending, root, dirichlet_alpha, dirichlet_fraction,
                    gumbel_considered, gumbel_scale, gumbel_c_visit, gumbel_c_scale,
                    gumbel_rescale_values, determinizations, max_players, fpu_value, fpu_reduction,
                    skip_forced, release_games, threads};
            },
            nb::kw_only(), nb::arg("c_puct") = defaults.c_puct,
            nb::arg("simulations") = defaults.simulations,
            nb::arg("max_pending") = defaults.max_pending, nb::arg("root") = defaults.root,
            nb::arg("dirichlet_alpha") = defaults.dirichlet_alpha,
            nb::arg("dirichlet_fraction") = defaults.dirichlet_fraction,
            nb::arg("gumbel_considered") = defaults.gumbel_considered,
            nb::arg("gumbel_scale") = defaults.gumbel_scale,
            nb::arg("gumbel_c_visit") = defaults.gumbel_c_visit,
            nb::arg("gumbel_c_scale") = defaults.gumbel_c_scale,
            nb::arg("gumbel_rescale_values") = defaults.gumbel_rescale_values,
            nb::arg("determinizations") = defaults.determinizations,
            nb::arg("max_players") = defaults.max_players, nb::arg("fpu_value") = defaults.fpu_value,
            nb::arg("fpu_reduction") = defaults.fpu_reduction,
            nb::arg("skip_forced") = defaults.skip_forced,
            nb::arg("release_games") = defaults.release_games, nb::arg("threads") = defaults.threads)
        .def_rw("c_puct", &SearchConfig::c_puct)
        .def_rw("simulations", &SearchConfig::simulations)
        .def_rw("max_pending", &SearchConfig::max_pending)
        .def_rw("root", &SearchConfig::root)
        .def_rw("dirichlet_alpha", &SearchConfig::dirichlet_alpha)
        .def_rw("dirichlet_fraction", &SearchConfig::dirichlet_fraction)
        .def_rw("gumbel_considered", &SearchConfig::gumbel_considered)
        .def_rw("gumbel_scale", &SearchConfig::gumbel_scale)
        .def_rw("gumbel_c_visit", &SearchConfig::gumbel_c_visit)
        .def_rw("gumbel_c_scale", &SearchConfig::gumbel_c_scale)
        .def_rw("gumbel_rescale_values", &SearchConfig::gumbel_rescale_values)
        .def_rw("determinizations", &SearchConfig::determinizations)
        .def_rw("max_players", &SearchConfig::max_players)
        .def_rw("fpu_value", &SearchConfig::fpu_value)
        .def_rw("fpu_reduction", &SearchConfig::fpu_reduction)
        .def_rw("skip_forced", &SearchConfig::skip_forced)
        .def_rw("release_games", &SearchConfig::release_games)
        .def_rw("threads", &SearchConfig::threads)
        .def("__repr__", [](const SearchConfig& c) {
            return nb::str(
                       "SearchConfig(c_puct={}, simulations={}, max_pending={}, root={!r}, "
                       "dirichlet_alpha={}, dirichlet_fraction={}, gumbel_considered={}, "
                       "gumbel_scale={}, gumbel_c_visit={}, gumbel_c_scale={}, "
                       "gumbel_rescale_values={}, determinizations={}, max_players={}, "
                       "fpu_value={}, fpu_reduction={}, skip_forced={}, release_games={}, threads={})")
                .format(
                    c.c_puct, c.simulations, c.max_pending, c.root, c.dirichlet_alpha,
                    c.dirichlet_fraction, c.gumbel_considered, c.gumbel_scale, c.gumbel_c_visit,
                    c.gumbel_c_scale, c.gumbel_rescale_values, c.determinizations, c.max_players,
                    c.fpu_value, c.fpu_reduction, c.skip_forced, c.release_games, c.threads);
        });

    nb::class_<SearchStats>(module, "SearchStats", "Counters of one search.")
        .def_ro("simulations", &SearchStats::simulations)
        .def_ro("evaluations", &SearchStats::evaluations)
        .def_ro("nodes", &SearchStats::nodes)
        .def_ro("chance_nodes", &SearchStats::chance_nodes)
        .def_ro("games", &SearchStats::games, "Games currently held by the tree.")
        .def_ro("edges", &SearchStats::edges)
        .def_ro("coalesced_outcomes", &SearchStats::coalesced_outcomes)
        .def_ro("rebuilt_games", &SearchStats::rebuilt_games)
        .def_ro("worlds", &SearchStats::worlds)
        .def_ro("max_depth", &SearchStats::max_depth)
        .def_ro("mean_depth", &SearchStats::mean_depth)
        .def_ro("tree_bytes", &SearchStats::tree_bytes, "Tree storage excluding the games' heap.")
        .def_ro("lane", &SearchStats::lane, "The worker thread the search runs on.");

    nb::class_<SearchPool>(
        module, "SearchPool",
        "Many independent searches advanced in lockstep with batched inference.\n\n"
        "    while n := pool.collect(obs, ids):\n"
        "        logits, values = net(obs[:n])   # [n, A] and [n, P] float32\n"
        "        pool.apply(logits, values)\n\n"
        "Observations are written from each leaf's player to move. ``values`` are\n"
        "P win probabilities in seat-relative order (slot 0 = the leaf's player to\n"
        "move); the tree backs up a value per seat and every node picks by its own\n"
        "player's (max^n). Terminal positions never reach the network; chance\n"
        "actions (dice, card draws, steals) are sampled exactly. The codec's mask\n"
        "restricts priors to legal actions. collect and apply release the GIL.")
        .def(
            "__init__",
            [](SearchPool* self, std::shared_ptr<ActionCodec> codec,
               std::shared_ptr<ObservationEncoder> encoder, const SearchConfig& config) {
                new (self) SearchPool(std::move(codec), std::move(encoder), config);
            },
            nb::arg("codec"), nb::arg("encoder"), nb::arg("config") = SearchConfig{})
        .def_prop_ro("config", &SearchPool::config, nb::rv_policy::copy)
        .def_prop_ro("observation_size", &SearchPool::observation_size)
        .def_prop_ro("action_size", &SearchPool::action_size)
        .def_prop_ro("max_players", &SearchPool::max_players)
        .def("__len__", &SearchPool::size)
        .def("__contains__", &SearchPool::contains, nb::arg("id"))
        .def_prop_ro("ids", &SearchPool::ids)
        .def_prop_ro("pending", &SearchPool::pending, "Rows of the last collect awaiting apply.")
        .def(
            "add",
            [](SearchPool& pool, const Game& game, std::uint64_t seed, const HandTracker* tracker,
               bool ignore_tracker) { return pool.add(game, seed, tracker, ignore_tracker); },
            nb::arg("game"), nb::arg("seed") = 0, nb::arg("tracker") = nb::none(), nb::kw_only(),
            nb::arg("ignore_tracker") = false,
            "Start a search for ``game``'s player to move; returns its id. With\n"
            "determinizations the root worlds are sampled from ``tracker``, which\n"
            "must follow that player, and the root offers exactly ``game``'s legal\n"
            "actions (the player's own knowledge): worlds that do not offer them all\n"
            "are resampled, and every root is restricted to them. Without\n"
            "determinizations the search reads ``game`` as is, hidden cards included\n"
            "(an oracle), so a ``tracker`` raises ValueError unless\n"
            "``ignore_tracker=True``. A finished game's search is done at once.")
        .def(
            "collect",
            [](SearchPool& pool, OutputMatrix<float> observations, OutputVector<std::int64_t> ids) {
                if (observations.shape(1) != pool.observation_size()) {
                    throw std::invalid_argument("observation rows must be observation_size wide");
                }
                if (ids.shape(0) > observations.shape(0)) {
                    throw std::invalid_argument("ids has more rows than observations");
                }
                nb::gil_scoped_release release;
                return pool.collect(
                    std::span<float>(observations.data(), observations.shape(0) * observations.shape(1)),
                    std::span<std::int64_t>(ids.data(), ids.shape(0)));
            },
            nb::arg("observations").noconvert(), nb::arg("ids").noconvert(),
            "Advance every unfinished search to its next leaves (roots first) and\n"
            "write up to ``len(ids)`` rows: observations into ``observations``\n"
            "(float32 [B, observation_size], C-contiguous) and search ids into ``ids``\n"
            "(int64 [B]). Returns the number of rows n; 0 once every search is done.")
        .def(
            "apply",
            [](SearchPool& pool, InputMatrix<float> logits, InputMatrix<float> values) {
                if (logits.shape(0) != pool.pending() || logits.shape(1) != pool.action_size()) {
                    throw std::invalid_argument("logits must be [n, action_size] for the n collected rows");
                }
                if (values.shape(0) != pool.pending() ||
                    values.shape(1) != static_cast<std::size_t>(pool.max_players())) {
                    throw std::invalid_argument("values must be [n, max_players] for the n collected rows");
                }
                nb::gil_scoped_release release;
                pool.apply(
                    std::span<const float>(logits.data(), logits.shape(0) * logits.shape(1)),
                    std::span<const float>(values.data(), values.shape(0) * values.shape(1)));
            },
            nb::arg("logits"), nb::arg("values"),
            "Expand and back up the last collect's rows: ``logits`` float32\n"
            "[n, action_size], ``values`` float32 [n, max_players] (seat-relative win\n"
            "probabilities).")
        .def(
            "cancel", &SearchPool::cancel,
            "Withdraw the last collect's rows unevaluated (e.g. after a failed\n"
            "network call); the next collect chooses leaves afresh. apply rejects bad\n"
            "rows (wrong shape, non-finite values) without changing anything, so it can\n"
            "also simply be retried.")
        .def(
            "pending_game", [](const SearchPool& pool, std::size_t row) { return Game(pool.pending_game(row)); },
            nb::arg("row"), "A copy of a pending row's position.")
        .def("done", &SearchPool::done, nb::arg("id"))
        .def(
            "visits", [](const SearchPool& pool, std::int64_t id) { return to_numpy(pool.visits(id)); },
            nb::arg("id"), "Root visits over the codec's index space (uint32), summed over worlds.")
        .def(
            "q", [](const SearchPool& pool, std::int64_t id) { return to_numpy(pool.q(id)); },
            nb::arg("id"), "Root action values for the root player (float64); NaN where unvisited.")
        .def(
            "policy_target",
            [](const SearchPool& pool, std::int64_t id) { return to_numpy(pool.policy_target(id)); },
            nb::arg("id"),
            "Visit distribution (PUCT) or Gumbel's softmax(logits + sigma(completed Q))\n"
            "over the codec's index space (float32).")
        .def(
            "value", [](const SearchPool& pool, std::int64_t id) { return to_numpy(pool.value(id)); },
            nb::arg("id"), "Mean root value per seat, seat-relative to the root player (float32 [P]).")
        .def(
            "select", &SearchPool::select, nb::arg("id"), nb::arg("temperature") = 0.0,
            "The action index to play: most visits (PUCT) or the Sequential Halving\n"
            "winner (Gumbel) at temperature 0, else a sample from\n"
            "policy_target ** (1 / temperature).")
        .def(
            "advance", &SearchPool::advance, nb::arg("id"), nb::arg("action_index"),
            "Re-root at a deterministic root action's child, keep the subtree and start\n"
            "a fresh budget. False for chance actions, finished or unvisited children\n"
            "and, with determinizations, another player's decision, a child reached\n"
            "through a drawn outcome, an action revealing hidden cards (a monopoly), or\n"
            "worlds disagreeing on the new root's public state or options. On False,\n"
            "add() a new search from the real position with the updated tracker.")
        .def("remove", &SearchPool::remove, nb::arg("id"))
        .def("stats", &SearchPool::stats, nb::arg("id"));
}

}  // namespace cppanatron::python
