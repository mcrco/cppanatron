#include "bindings.hpp"

#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cppanatron/map.hpp"

namespace cppanatron::python {
namespace {

nb::tuple edge_to_python(Edge edge) { return nb::make_tuple(edge.a, edge.b); }

/** Every tile keyed by cube coordinate, in tile (spec) order like Catanatron's dict. */
nb::dict tiles_to_python(const CatanMap& map, bool land_only) {
    nb::dict result;
    for (std::size_t index = 0; index < map.tiles().size(); ++index) {
        const Tile& tile = map.tiles()[index];
        if (!land_only || tile.kind == TileKind::land) {
            result[coordinate_to_python(map.spec().tiles[index].coordinate)] =
                nb::cast(tile, nb::rv_policy::copy);
        }
    }
    return result;
}

constexpr std::string_view kind_name(TileKind kind) {
    switch (kind) {
        case TileKind::land:
            return "land";
        case TileKind::water:
            return "water";
        case TileKind::port:
            return "port";
    }
    return "";
}

/** Enum members by their Python name ("WOOD", "EAST"), None for null. */
template <typename Enum>
std::optional<Enum> enum_from_python(nb::handle value, const char* field) {
    if (value.is_none()) {
        return std::nullopt;
    }
    if (!nb::isinstance<nb::str>(value)) {
        return nb::cast<Enum>(value);
    }
    nb::object type = nb::borrow(nb::type<Enum>());
    const std::string name = nb::cast<std::string>(value);
    if (!nb::hasattr(type, name.c_str())) {
        throw std::invalid_argument(std::string("unknown ") + field + " " + name);
    }
    return nb::cast<Enum>(type.attr(name.c_str()));
}

template <typename Enum>
nb::object enum_to_python(const std::optional<Enum>& value) {
    if (!value.has_value()) {
        return nb::none();
    }
    return nb::cast(*value).attr("name");
}

TileSpec tile_from_dict(nb::handle item) {
    if (!nb::isinstance<nb::dict>(item)) {
        throw std::invalid_argument("each tile must be a dict");
    }
    const nb::dict tile_dict = nb::borrow<nb::dict>(item);
    for (auto [key, value] : tile_dict) {
        (void)value;
        const std::string name = nb::cast<std::string>(key);
        if (name != "coordinate" && name != "type" && name != "resource" && name != "number" &&
            name != "direction" && name != "port_resource") {
            throw std::invalid_argument("unknown tile field " + name);
        }
    }
    if (!tile_dict.contains("coordinate") || !tile_dict.contains("type")) {
        throw std::invalid_argument("each tile needs a coordinate and a type");
    }
    const auto field = [&tile_dict](const char* name) -> nb::object {
        return tile_dict.contains(name) ? nb::borrow(tile_dict[name]) : nb::none();
    };

    TileSpec tile;
    tile.coordinate = coordinate_from_python(tile_dict["coordinate"]);
    const std::string type = nb::cast<std::string>(tile_dict["type"]);
    if (type == "land") {
        tile.kind = TileKind::land;
    } else if (type == "water") {
        tile.kind = TileKind::water;
    } else if (type == "port") {
        tile.kind = TileKind::port;
    } else {
        throw std::invalid_argument("tile type must be land, water or port, not " + type);
    }
    tile.resource = enum_from_python<Resource>(field("resource"), "resource");
    if (nb::object number = field("number"); !number.is_none()) {
        tile.number = nb::cast<int>(number);
    }
    tile.direction = enum_from_python<Direction>(field("direction"), "direction");
    tile.port_resource = enum_from_python<Resource>(field("port_resource"), "port resource");
    return tile;
}

nb::dict tile_to_dict(const TileSpec& tile) {
    nb::dict result;
    nb::list coordinate;  // JSON-shaped: a list, like the input.
    coordinate.append(tile.coordinate.x);
    coordinate.append(tile.coordinate.y);
    coordinate.append(tile.coordinate.z);
    result["coordinate"] = coordinate;
    result["type"] = nb::str(kind_name(tile.kind).data(), kind_name(tile.kind).size());
    result["resource"] = enum_to_python(tile.resource);
    result["number"] = tile.number.has_value() ? nb::cast(*tile.number) : nb::none();
    result["direction"] = enum_to_python(tile.direction);
    result["port_resource"] = enum_to_python(tile.port_resource);
    return result;
}

MapSpec spec_from_dict(const nb::dict& value) {
    for (auto [key, item] : value) {
        (void)item;
        if (nb::cast<std::string>(key) != "tiles") {
            throw std::invalid_argument("a map spec has only a tiles list");
        }
    }
    if (!value.contains("tiles")) {
        throw std::invalid_argument("a map spec needs a tiles list");
    }
    MapSpec spec;
    for (nb::handle item : nb::iter(value["tiles"])) {
        spec.tiles.push_back(tile_from_dict(item));
    }
    return spec;
}

nb::dict spec_to_dict(const MapSpec& spec) {
    nb::list tiles;
    for (const TileSpec& tile : spec.tiles) {
        tiles.append(tile_to_dict(tile));
    }
    nb::dict result;
    result["tiles"] = tiles;
    return result;
}

std::shared_ptr<const CatanMap> share(CatanMap map) {
    return std::make_shared<const CatanMap>(std::move(map));
}

}  // namespace

void bind_maps(nb::module_& module) {
    nb::class_<TileSpec>(
        module, "TileSpec",
        "One tile of a MapSpec. Land: ``resource`` (None for the desert) and\n"
        "``number``; port: ``direction`` (the side facing land) and\n"
        "``port_resource`` (None for 3:1); water: neither.")
        .def(
            "__init__",
            [](TileSpec* self, nb::handle coordinate, TileKind kind,
               std::optional<Resource> resource, std::optional<int> number,
               std::optional<Direction> direction, std::optional<Resource> port_resource) {
                new (self) TileSpec{
                    coordinate_from_python(coordinate), kind, resource, number, direction,
                    port_resource};
            },
            nb::arg("coordinate"), nb::arg("kind"), nb::kw_only(),
            nb::arg("resource") = nb::none(), nb::arg("number") = nb::none(),
            nb::arg("direction") = nb::none(), nb::arg("port_resource") = nb::none())
        .def_prop_ro(
            "coordinate", [](const TileSpec& tile) { return coordinate_to_python(tile.coordinate); })
        .def_ro("kind", &TileSpec::kind)
        .def_ro("resource", &TileSpec::resource)
        .def_ro("number", &TileSpec::number)
        .def_ro("direction", &TileSpec::direction)
        .def_ro("port_resource", &TileSpec::port_resource)
        .def("to_dict", &tile_to_dict)
        .def(
            "__eq__", [](const TileSpec& lhs, const TileSpec& rhs) { return lhs == rhs; },
            nb::is_operator())
        .def("__repr__", [](const TileSpec& tile) {
            return nb::str("TileSpec({!r})").format(tile_to_dict(tile));
        });

    nb::class_<MapSpec>(
        module, "MapSpec",
        "An explicit map: every tile in order. Tile order defines node, edge,\n"
        "land tile and port ids exactly as in Catanatron's from_tiles walk, so a\n"
        "spec builds the same board in both engines.\n\n"
        "JSON shape (``to_dict``/``from_dict``)::\n\n"
        "    {\"tiles\": [{\"coordinate\": [x, y, z],\n"
        "                \"type\": \"land\" | \"water\" | \"port\",\n"
        "                \"resource\": \"WOOD\" | ... | None,  # land; None = desert\n"
        "                \"number\": int | None,\n"
        "                \"direction\": \"EAST\" | ... | None,  # port\n"
        "                \"port_resource\": \"WOOD\" | ... | None}]}  # None = 3:1")
        .def(
            "__init__",
            [](MapSpec* self, std::vector<TileSpec> tiles) {
                new (self) MapSpec{std::move(tiles)};
            },
            nb::arg("tiles"))
        .def_prop_ro("tiles", [](const MapSpec& spec) { return spec.tiles; })
        .def("__len__", [](const MapSpec& spec) { return spec.tiles.size(); })
        .def_static(
            "from_dict", &spec_from_dict, nb::arg("value"),
            "Parse the JSON shape; missing optional fields are None.")
        .def("to_dict", &spec_to_dict, "The JSON shape, every field present.")
        .def_static(
            "build",
            [](MapType map_type, std::uint64_t seed, NumberPlacement number_placement) {
                return CatanMap::build_spec(map_type, seed, number_placement);
            },
            nb::arg("map_type"), nb::arg("seed") = 0,
            nb::arg("number_placement") = NumberPlacement::official_spiral,
            "A template map's tiles, drawn as CatanMap.build draws them.")
        .def(
            "__eq__", [](const MapSpec& lhs, const MapSpec& rhs) { return lhs == rhs; },
            nb::is_operator())
        .def("__repr__", [](const MapSpec& spec) {
            return nb::str("MapSpec(<{} tiles>)").format(spec.tiles.size());
        });

    nb::class_<Tile>(module, "Tile", "A board hexagon: land, water or port.")
        .def_ro("kind", &Tile::kind)
        .def_ro("id", &Tile::id)
        .def_ro("resource", &Tile::resource)
        .def_ro("number", &Tile::number)
        .def_ro("port_direction", &Tile::port_direction)
        .def_prop_ro(
            "nodes", [](const Tile& tile) { return to_tuple(tile.nodes); },
            "Node ids ordered NORTH, NORTHEAST, SOUTHEAST, SOUTH, SOUTHWEST, NORTHWEST.")
        .def_prop_ro(
            "edges",
            [](const Tile& tile) {
                nb::list edges;
                for (Edge edge : tile.edges) {
                    edges.append(edge_to_python(edge));
                }
                return nb::tuple(edges);
            },
            "Edges ordered EAST, SOUTHEAST, SOUTHWEST, WEST, NORTHWEST, NORTHEAST.")
        .def("__repr__", [](const Tile& tile) {
            return nb::str("Tile({!r}, id={}, resource={!r}, number={!r})")
                .format(tile.kind, tile.id, nb::cast(tile.resource), nb::cast(tile.number));
        });

    // Maps only ever reach Python as shared pointers owned by C++, so passing
    // one back (to a Game, a VecEnv) shares that ownership and never ties the
    // map's lifetime to a Python reference count.
    nb::class_<CatanMap>(
        module, "CatanMap",
        "An immutable board layout, shared by every game built on it.")
        .def_static(
            "build",
            [](MapType map_type, std::uint64_t seed, NumberPlacement number_placement) {
                return share(CatanMap::build(map_type, seed, number_placement));
            },
            nb::arg("map_type"), nb::arg("seed") = 0,
            nb::arg("number_placement") = NumberPlacement::official_spiral)
        .def_static(
            "from_spec",
            [](const nb::handle& spec) {
                if (nb::isinstance<nb::dict>(spec)) {
                    return share(CatanMap::from_spec(spec_from_dict(nb::borrow<nb::dict>(spec))));
                }
                return share(CatanMap::from_spec(nb::cast<const MapSpec&>(spec)));
            },
            nb::arg("spec"),
            "Build a map from a MapSpec (or its dict form). Raises ValueError for\n"
            "specs no game can use (see MapSpec).")
        .def("spec", &CatanMap::spec, nb::rv_policy::copy, "The explicit tiles of this map.")
        .def_prop_ro(
            "tiles", [](const CatanMap& map) { return tiles_to_python(map, false); },
            "Every tile keyed by cube coordinate, like Catanatron's CatanMap.tiles.")
        .def_prop_ro(
            "land_tiles", [](const CatanMap& map) { return tiles_to_python(map, true); })
        .def_prop_ro(
            "land_nodes",
            [](const CatanMap& map) {
                return nb::frozenset(nb::cast(map.land_node_list()));
            })
        .def_prop_ro(
            "land_edges",
            [](const CatanMap& map) {
                nb::list edges;
                for (Edge edge : map.land_edges()) {
                    edges.append(edge_to_python(edge));
                }
                return edges;
            })
        .def(
            "tile_at",
            [](const CatanMap& map, nb::handle coordinate) {
                return map.tile_at(coordinate_from_python(coordinate));
            },
            nb::arg("coordinate"))
        .def_prop_ro("num_nodes", &CatanMap::num_nodes)
        .def_prop_ro(
            "num_edges", [](const CatanMap& map) { return map.land_edges().size(); })
        .def_prop_ro(
            "num_land_tiles", [](const CatanMap& map) { return map.land_tile_indices().size(); })
        .def_prop_ro(
            "node_ids", [](const CatanMap& map) { return map.land_node_list(); },
            "Land node ids in slot order (ascending), as encoders and codecs index them.")
        .def_prop_ro(
            "land_tile_coordinates",
            [](const CatanMap& map) {
                nb::list result;
                for (int index : map.land_tile_indices()) {
                    result.append(coordinate_to_python(
                        map.spec().tiles[static_cast<std::size_t>(index)].coordinate));
                }
                return result;
            },
            "Land tile coordinates in slot order (tile order).")
        .def(
            "neighbors", &CatanMap::neighbors, nb::arg("node"),
            "Land neighbors of a node, ascending.")
        .def_prop_ro(
            "port_nodes",
            [](const CatanMap& map) {
                // Catanatron's CatanMap.port_nodes: resource (None = 3:1) -> nodes.
                std::map<std::optional<Resource>, std::vector<int>> ports;
                for (const Tile& tile : map.tiles()) {
                    if (tile.kind == TileKind::port) {
                        const auto [first, second] = port_nodes(tile);
                        ports[tile.resource].push_back(first);
                        ports[tile.resource].push_back(second);
                    }
                }
                nb::dict result;
                for (const auto& [resource, nodes] : ports) {
                    result[resource.has_value() ? nb::cast(*resource) : nb::none()] =
                        nb::frozenset(nb::cast(nodes));
                }
                return result;
            },
            "``{Resource | None: frozenset(nodes)}``, None for 3:1 ports.");
}

}  // namespace cppanatron::python
