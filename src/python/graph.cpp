#include "bindings.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cppanatron/graph_action_space.hpp"
#include "cppanatron/graph_dims.hpp"

namespace cppanatron::python {
namespace {

nb::tuple span(std::size_t offset, nb::tuple shape) { return nb::make_tuple(offset, shape); }

nb::dict action_layout(const GraphActionSpace& space) {
    const EncoderDims& dims = space.dims();
    nb::dict result;
    result["globals"] = span(0, nb::make_tuple(GraphActionSpace::kNumGlobals));
    result["settlement"] = span(space.settlement_offset(), nb::make_tuple(dims.max_nodes));
    result["city"] = span(space.city_offset(), nb::make_tuple(dims.max_nodes));
    result["road"] = span(space.road_offset(), nb::make_tuple(dims.max_edges));
    result["robber"] =
        span(space.robber_offset(), nb::make_tuple(dims.max_tiles, dims.max_players));
    result["noop"] = span(space.noop(), nb::make_tuple(1));
    return result;
}

}  // namespace

void bind_graph(nb::module_& module) {
    const EncoderDims defaults;
    nb::class_<EncoderDims>(
        module, "EncoderDims",
        "Maxima of a graph encoder and action space: seats, land tiles, land nodes\n"
        "and land edges. Rows and action spaces are padded to them. The defaults\n"
        "fit BASE for up to six players.")
        .def(
            "__init__",
            [](EncoderDims* self, int max_players, int max_tiles, int max_nodes, int max_edges) {
                EncoderDims dims{max_players, max_tiles, max_nodes, max_edges};
                dims.validate();
                new (self) EncoderDims(dims);
            },
            nb::kw_only(), nb::arg("max_players") = defaults.max_players,
            nb::arg("max_tiles") = defaults.max_tiles, nb::arg("max_nodes") = defaults.max_nodes,
            nb::arg("max_edges") = defaults.max_edges)
        .def_ro("max_players", &EncoderDims::max_players)
        .def_ro("max_tiles", &EncoderDims::max_tiles)
        .def_ro("max_nodes", &EncoderDims::max_nodes)
        .def_ro("max_edges", &EncoderDims::max_edges)
        .def_static(
            "covering", &EncoderDims::covering, nb::arg("maps"), nb::arg("max_players"),
            "The smallest dims fitting every map for up to ``max_players`` seats.")
        .def(
            "fits",
            [](const EncoderDims& dims, const CatanMap& map, int num_players) {
                return dims.fits(map, num_players);
            },
            nb::arg("map"), nb::arg("num_players"))
        .def(
            "__eq__", [](const EncoderDims& lhs, const EncoderDims& rhs) { return lhs == rhs; },
            nb::is_operator())
        .def("__repr__", [](const EncoderDims& dims) {
            return nb::str("EncoderDims(max_players={}, max_tiles={}, max_nodes={}, max_edges={})")
                .format(dims.max_players, dims.max_tiles, dims.max_nodes, dims.max_edges);
        });

    nb::class_<GraphActionSpace>(
        module, "GraphActionSpace",
        "Map-agnostic action indices: [globals | settlement@node | city@node |\n"
        "road@edge | robber@tile x victim-slot | no-op]. Robber victims are seats\n"
        "relative to the actor; the last index is the no-op of waiting seats.\n"
        "Domestic trades are not in the space.")
        .def(nb::init<EncoderDims>(), nb::arg("dims"))
        .def("__len__", &GraphActionSpace::size)
        .def_prop_ro("size", &GraphActionSpace::size)
        .def_prop_ro("dims", &GraphActionSpace::dims)
        .def_prop_ro("noop", &GraphActionSpace::noop)
        .def(
            "layout", &action_layout,
            "``{block: (offset, shape)}`` for globals, settlement, city, road, robber\n"
            "(tiles x victim slots) and noop.")
        .def(
            "index", &GraphActionSpace::index, nb::arg("game"), nb::arg("action"),
            "Index of a game action; IndexError for actions outside the space.")
        .def(
            "decode", &GraphActionSpace::decode, nb::arg("game"), nb::arg("index"),
            "The game action an index means for the player to move.")
        .def(
            "mask",
            [](const GraphActionSpace& space, const Game& game) {
                std::vector<std::uint8_t> out(space.size());
                space.mask(game, out);
                // NumPy bool shares uint8's layout.
                auto* owned = new std::vector<std::uint8_t>(std::move(out));
                nb::capsule deleter(owned, [](void* pointer) noexcept {
                    delete static_cast<std::vector<std::uint8_t>*>(pointer);
                });
                return Vector<bool>(reinterpret_cast<bool*>(owned->data()), {owned->size()}, deleter);
            },
            nb::arg("game"), "Boolean legality mask for the player to move.")
        .def("name", &GraphActionSpace::name, nb::arg("index"))
        .def(
            "names",
            [](const GraphActionSpace& space) {
                std::vector<std::string> names;
                names.reserve(space.size());
                for (std::size_t i = 0; i < space.size(); ++i) {
                    names.push_back(space.name(i));
                }
                return names;
            },
            "A readable name for every index.");
}

}  // namespace cppanatron::python
