#include "bindings.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cppanatron/graph_action_space.hpp"
#include "cppanatron/graph_dims.hpp"
#include "cppanatron/graph_encoder.hpp"

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

nb::dict encoder_layout(const GraphEncoder& encoder) {
    nb::dict result;
    for (const auto& block : encoder.layout()) {
        nb::list shape;
        for (std::size_t extent : block.shape) {
            shape.append(extent);
        }
        result[block.name.c_str()] = span(block.offset, nb::tuple(shape));
    }
    return result;
}

void check_observer(const Game& game, int observer) {
    if (observer < 0 || static_cast<std::size_t>(observer) >= game.colors().size()) {
        throw std::out_of_range("observer is not a seat of this game");
    }
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

    nb::class_<GraphActionSpace, ActionCodec>(
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

    nb::enum_<InfoLevel>(
        module, "InfoLevel",
        "What an encoded seat sees of the others: PRIVATE (its own hand and the\n"
        "table), PUBLIC (plus everyone's resource hands), FULL (plus development\n"
        "cards and actual victory points).")
        .value("PRIVATE", InfoLevel::private_)
        .value("PUBLIC", InfoLevel::public_)
        .value("FULL", InfoLevel::full);

    nb::class_<GraphEncoder, ObservationEncoder>(
        module, "GraphEncoder",
        "Seat-relative graph observations padded to EncoderDims. A row is\n"
        "[global | players | tiles | nodes | edges | edge_nodes | tile_nodes |\n"
        "action_mask]; layout() gives every block's (offset, shape) and\n"
        "feature_names(block) the per-slot features. The mask block is the\n"
        "action space's mask for the seat to act and the no-op for the others.")
        .def(nb::init<EncoderDims, InfoLevel>(), nb::arg("dims"),
             nb::arg("level") = InfoLevel::private_)
        .def_prop_ro("size", &GraphEncoder::size)
        .def("__len__", &GraphEncoder::size)
        .def_prop_ro("dims", &GraphEncoder::dims)
        .def_prop_ro("level", &GraphEncoder::level)
        .def_prop_ro(
            "action_space", [](const GraphEncoder& encoder) { return encoder.action_space(); },
            "The GraphActionSpace whose mask ends each row.")
        .def("layout", &encoder_layout, "``{block: (offset, shape)}`` in row order.")
        .def(
            "feature_names",
            [](const GraphEncoder& encoder, std::optional<std::string> block) -> nb::object {
                if (block.has_value()) {
                    return nb::cast(encoder.feature_names(*block));
                }
                nb::dict result;
                for (const char* name : {"global", "players", "tiles", "nodes", "edges"}) {
                    result[name] = nb::cast(encoder.feature_names(name));
                }
                return result;
            },
            nb::arg("block") = nb::none(),
            "Per-slot feature names of one block, or a dict of every block's.")
        .def(
            "encode",
            [](const GraphEncoder& encoder, const Game& game, int observer) {
                check_observer(game, observer);
                std::vector<float> row(encoder.size());
                {
                    nb::gil_scoped_release release;
                    encoder.encode(game, observer, row);
                }
                return to_numpy(std::move(row));
            },
            nb::arg("game"), nb::arg("observer"),
            "A fresh float32 row for seat ``observer`` (an index into game.colors).")
        .def(
            "encode_into",
            [](const GraphEncoder& encoder, const Game& game, int observer,
               OutputVector<float> out) {
                check_observer(game, observer);
                nb::gil_scoped_release release;
                encoder.encode(game, observer, std::span<float>(out.data(), out.shape(0)));
            },
            nb::arg("game"), nb::arg("observer"), nb::arg("out").noconvert(),
            "Write the row into a caller-owned float32 buffer of size ``size``.");
}

}  // namespace cppanatron::python
