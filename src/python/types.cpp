#include "bindings.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <variant>

#include <nanobind/stl/string.h>

#include "cppanatron/map.hpp"

namespace cppanatron::python {
namespace {

template <typename... Visitors>
struct Overloaded : Visitors... {
    using Visitors::operator()...;
};

nb::object optional_color(std::optional<Color> color) {
    return color.has_value() ? nb::cast(*color) : nb::none();
}

nb::object optional_resource(std::optional<Resource> resource) {
    return resource.has_value() ? nb::cast(*resource) : nb::none();
}

std::vector<nb::handle> items(nb::handle value, std::size_t expected, const char* shape) {
    nb::tuple sequence = nb::tuple(nb::borrow(value));
    if (sequence.size() != expected) {
        throw std::invalid_argument(std::string("action value must be ") + shape);
    }
    std::vector<nb::handle> result;
    result.reserve(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        result.push_back(sequence[i]);
    }
    return result;
}

DomesticTrade trade_from_python(const std::vector<nb::handle>& counts) {
    DomesticTrade trade;
    for (std::size_t i = 0; i < 5; ++i) {
        trade.offering[i] = nb::cast<int>(counts[i]);
        trade.asking[i] = nb::cast<int>(counts[5 + i]);
    }
    return trade;
}

void append_trade(nb::list& result, const DomesticTrade& trade) {
    for (int count : trade.offering) {
        result.append(count);
    }
    for (int count : trade.asking) {
        result.append(count);
    }
}

/**
 * Drop state a caller cannot express, so equal decisions compare equal.
 *
 * Like Catanatron's, offers and confirmations name no offering seat: the game
 * fills it in. Accept and reject responses carry it as their 11th element.
 */
Action public_form(Action action) {
    if (auto* trade = std::get_if<DomesticTrade>(&action.value);
        trade != nullptr && action.type == ActionType::offer_trade) {
        trade->offering_player_index = -1;
    } else if (auto* confirmed = std::get_if<ConfirmedTrade>(&action.value)) {
        confirmed->trade.offering_player_index = -1;
    }
    return action;
}

void hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
}

std::size_t hash_action(const Action& action) {
    std::size_t seed = std::hash<int>{}(static_cast<int>(action.color));
    hash_combine(seed, std::hash<int>{}(static_cast<int>(action.type)));
    const auto combine = [&seed](int value) { hash_combine(seed, std::hash<int>{}(value)); };
    const auto combine_trade = [&combine](const DomesticTrade& trade) {
        for (int count : trade.offering) {
            combine(count);
        }
        for (int count : trade.asking) {
            combine(count);
        }
    };
    std::visit(
        Overloaded{
            [](std::monostate) {},
            [&](int node) { combine(node); },
            [&](const Edge& edge) {
                combine(edge.a);
                combine(edge.b);
            },
            [&](Resource resource) { combine(static_cast<int>(resource)); },
            [&](const Dice& dice) {
                combine(dice.first);
                combine(dice.second);
            },
            [&](const RobberMove& move) {
                combine(move.coordinate.x);
                combine(move.coordinate.y);
                combine(move.coordinate.z);
                combine(move.victim.has_value() ? static_cast<int>(*move.victim) : -1);
            },
            [&](const FlatRobberMove& move) {
                combine(move.coordinate.x);
                combine(move.coordinate.y);
                combine(move.coordinate.z);
                combine(move.victim_slot.value_or(-1));
            },
            [&](const MaritimeTrade& trade) {
                for (const auto& card : trade.cards) {
                    combine(card.has_value() ? static_cast<int>(*card) : -1);
                }
            },
            [&](const DomesticTrade& trade) { combine_trade(trade); },
            [&](const ConfirmedTrade& trade) {
                combine_trade(trade.trade);
                combine(static_cast<int>(trade.partner));
            },
            [&](const std::vector<Resource>& resources) {
                for (Resource resource : resources) {
                    combine(static_cast<int>(resource));
                }
            },
        },
        action.value);
    return seed;
}



}  // namespace

nb::tuple coordinate_to_python(Coordinate coordinate) {
    return nb::make_tuple(coordinate.x, coordinate.y, coordinate.z);
}

Coordinate coordinate_from_python(nb::handle value) {
    const auto xyz = items(value, 3, "a cube coordinate (x, y, z)");
    return {nb::cast<int>(xyz[0]), nb::cast<int>(xyz[1]), nb::cast<int>(xyz[2])};
}

nb::object action_value_to_python(ActionType type, const ActionValue& value) {
    return std::visit(
        Overloaded{
            [](std::monostate) -> nb::object { return nb::none(); },
            [](int node) -> nb::object { return nb::int_(node); },
            [](const Edge& edge) -> nb::object { return nb::make_tuple(edge.a, edge.b); },
            [](Resource resource) -> nb::object { return nb::cast(resource); },
            [](const Dice& dice) -> nb::object {
                return nb::make_tuple(dice.first, dice.second);
            },
            [](const RobberMove& move) -> nb::object {
                return nb::make_tuple(
                    coordinate_to_python(move.coordinate), optional_color(move.victim));
            },
            [](const FlatRobberMove& move) -> nb::object {
                return nb::make_tuple(
                    coordinate_to_python(move.coordinate),
                    move.victim_slot.has_value() ? nb::cast(*move.victim_slot) : nb::none());
            },
            [](const MaritimeTrade& trade) -> nb::object {
                nb::list cards;
                for (const auto& card : trade.cards) {
                    cards.append(optional_resource(card));
                }
                return nb::tuple(cards);
            },
            [type](const DomesticTrade& trade) -> nb::object {
                nb::list counts;
                append_trade(counts, trade);
                if (type != ActionType::offer_trade) {
                    counts.append(trade.offering_player_index);
                }
                return nb::tuple(counts);
            },
            [](const ConfirmedTrade& trade) -> nb::object {
                nb::list counts;
                append_trade(counts, trade.trade);
                counts.append(nb::cast(trade.partner));
                return nb::tuple(counts);
            },
            [](const std::vector<Resource>& resources) -> nb::object {
                nb::list result;
                for (Resource resource : resources) {
                    result.append(nb::cast(resource));
                }
                return nb::tuple(result);
            },
        },
        value);
}

ActionValue action_value_from_python(ActionType type, nb::handle value) {
    switch (type) {
        case ActionType::roll:
            if (value.is_none()) {
                return std::monostate{};
            } else {
                const auto dice = items(value, 2, "two dice");
                return Dice{nb::cast<int>(dice[0]), nb::cast<int>(dice[1])};
            }
        case ActionType::end_turn:
        case ActionType::buy_development_card:
        case ActionType::play_knight_card:
        case ActionType::play_road_building:
        case ActionType::cancel_trade:
            if (!value.is_none()) {
                throw std::invalid_argument("this action type takes no value");
            }
            return std::monostate{};
        case ActionType::build_settlement:
        case ActionType::build_city:
            return nb::cast<int>(value);
        case ActionType::build_road: {
            const auto nodes = items(value, 2, "an edge (a, b)");
            return Edge{nb::cast<int>(nodes[0]), nb::cast<int>(nodes[1])}.normalized();
        }
        case ActionType::discard_resource:
        case ActionType::play_monopoly:
            return nb::cast<Resource>(value);
        case ActionType::play_year_of_plenty: {
            std::vector<Resource> resources;
            for (nb::handle resource : nb::tuple(nb::borrow(value))) {
                resources.push_back(nb::cast<Resource>(resource));
            }
            if (resources.empty() || resources.size() > 2) {
                throw std::invalid_argument("year of plenty takes one or two resources");
            }
            return resources;
        }
        case ActionType::move_robber: {
            const auto move = items(value, 2, "(coordinate, victim)");
            RobberMove result{coordinate_from_python(move[0]), std::nullopt};
            if (!move[1].is_none()) {
                result.victim = nb::cast<Color>(move[1]);
            }
            return result;
        }
        case ActionType::maritime_trade: {
            const auto cards = items(value, 5, "five cards (given ... given, asked)");
            MaritimeTrade trade;
            for (std::size_t i = 0; i < cards.size(); ++i) {
                if (!cards[i].is_none()) {
                    trade.cards[i] = nb::cast<Resource>(cards[i]);
                }
            }
            return trade;
        }
        case ActionType::offer_trade:
            return trade_from_python(items(value, 10, "ten counts (offered, asked)"));
        case ActionType::accept_trade:
        case ActionType::reject_trade: {
            const auto counts = items(value, 11, "ten counts and the offering seat");
            DomesticTrade trade = trade_from_python(counts);
            trade.offering_player_index = nb::cast<int>(counts[10]);
            return trade;
        }
        case ActionType::confirm_trade: {
            const auto counts = items(value, 11, "ten counts and the accepting color");
            return ConfirmedTrade{trade_from_python(counts), nb::cast<Color>(counts[10])};
        }
    }
    throw std::invalid_argument("unknown action type");
}

const Action& resolve_playable(const Game& game, const Action& action) {
    const auto& playable = game.playable_actions();
    const auto exact = std::find(playable.begin(), playable.end(), action);
    if (exact != playable.end()) {
        return *exact;
    }
    if (action.type == ActionType::confirm_trade) {
        const Action wanted = public_form(action);
        const auto equivalent = std::find_if(
            playable.begin(), playable.end(),
            [&wanted](const Action& candidate) { return public_form(candidate) == wanted; });
        if (equivalent != playable.end()) {
            return *equivalent;
        }
    }
    return action;
}

void bind_types(nb::module_& module) {
    nb::enum_<Color>(module, "Color")
        .value("RED", Color::red)
        .value("BLUE", Color::blue)
        .value("WHITE", Color::white)
        .value("ORANGE", Color::orange);
    nb::enum_<Resource>(module, "Resource")
        .value("WOOD", Resource::wood)
        .value("BRICK", Resource::brick)
        .value("SHEEP", Resource::sheep)
        .value("WHEAT", Resource::wheat)
        .value("ORE", Resource::ore);
    nb::enum_<DevelopmentCard>(module, "DevelopmentCard")
        .value("KNIGHT", DevelopmentCard::knight)
        .value("YEAR_OF_PLENTY", DevelopmentCard::year_of_plenty)
        .value("MONOPOLY", DevelopmentCard::monopoly)
        .value("ROAD_BUILDING", DevelopmentCard::road_building)
        .value("VICTORY_POINT", DevelopmentCard::victory_point);
    nb::enum_<Building>(module, "Building")
        .value("SETTLEMENT", Building::settlement)
        .value("CITY", Building::city);
    nb::enum_<MapType>(module, "MapType")
        .value("BASE", MapType::base)
        .value("MINI", MapType::mini)
        .value("TOURNAMENT", MapType::tournament);
    nb::enum_<NumberPlacement>(module, "NumberPlacement")
        .value("OFFICIAL_SPIRAL", NumberPlacement::official_spiral)
        .value("RANDOM", NumberPlacement::random);
    nb::enum_<TileKind>(module, "TileKind")
        .value("LAND", TileKind::land)
        .value("WATER", TileKind::water)
        .value("PORT", TileKind::port);
    nb::enum_<Direction>(module, "Direction")
        .value("EAST", Direction::east)
        .value("SOUTHEAST", Direction::southeast)
        .value("SOUTHWEST", Direction::southwest)
        .value("WEST", Direction::west)
        .value("NORTHWEST", Direction::northwest)
        .value("NORTHEAST", Direction::northeast);
    nb::enum_<ActionPrompt>(module, "ActionPrompt")
        .value("BUILD_INITIAL_SETTLEMENT", ActionPrompt::build_initial_settlement)
        .value("BUILD_INITIAL_ROAD", ActionPrompt::build_initial_road)
        .value("PLAY_TURN", ActionPrompt::play_turn)
        .value("DISCARD", ActionPrompt::discard)
        .value("MOVE_ROBBER", ActionPrompt::move_robber)
        .value("DECIDE_TRADE", ActionPrompt::decide_trade)
        .value("DECIDE_ACCEPTEES", ActionPrompt::decide_acceptees);
    nb::enum_<ActionType>(module, "ActionType")
        .value("ROLL", ActionType::roll)
        .value("MOVE_ROBBER", ActionType::move_robber)
        .value("DISCARD_RESOURCE", ActionType::discard_resource)
        .value("BUILD_ROAD", ActionType::build_road)
        .value("BUILD_SETTLEMENT", ActionType::build_settlement)
        .value("BUILD_CITY", ActionType::build_city)
        .value("BUY_DEVELOPMENT_CARD", ActionType::buy_development_card)
        .value("PLAY_KNIGHT_CARD", ActionType::play_knight_card)
        .value("PLAY_YEAR_OF_PLENTY", ActionType::play_year_of_plenty)
        .value("PLAY_MONOPOLY", ActionType::play_monopoly)
        .value("PLAY_ROAD_BUILDING", ActionType::play_road_building)
        .value("MARITIME_TRADE", ActionType::maritime_trade)
        .value("OFFER_TRADE", ActionType::offer_trade)
        .value("ACCEPT_TRADE", ActionType::accept_trade)
        .value("REJECT_TRADE", ActionType::reject_trade)
        .value("CONFIRM_TRADE", ActionType::confirm_trade)
        .value("CANCEL_TRADE", ActionType::cancel_trade)
        .value("END_TURN", ActionType::end_turn);

    nb::class_<Action>(
        module, "Action",
        "An immutable engine action, shaped like Catanatron's Action namedtuple.\n\n"
        "``value`` depends on ``action_type``: None; a node id (settlements, cities);\n"
        "an edge ``(a, b)`` with ``a < b``; a Resource (discard, monopoly); a tuple of\n"
        "one or two Resources (year of plenty); ``(coordinate, Color | None)``\n"
        "(robber); five ``Resource | None`` with the asked resource last (maritime);\n"
        "ten counts, offered then asked (offers), followed by the offering seat\n"
        "(accept and reject) or the accepting Color (confirm). The flat action\n"
        "space stores robber victims as relative seat offsets instead of colors.")
        .def(
            "__init__",
            [](Action* self, Color color, ActionType action_type, nb::handle value) {
                new (self) Action{color, action_type, action_value_from_python(action_type, value)};
            },
            nb::arg("color"), nb::arg("action_type"), nb::arg("value") = nb::none())
        .def_prop_ro("color", [](const Action& action) { return action.color; })
        .def_prop_ro("action_type", [](const Action& action) { return action.type; })
        .def_prop_ro(
            "value", [](const Action& action) { return action_value_to_python(action.type, action.value); })
        .def(
            "__eq__",
            [](const Action& lhs, const Action& rhs) { return public_form(lhs) == public_form(rhs); },
            nb::is_operator())
        .def("__hash__", [](const Action& action) { return hash_action(public_form(action)); })
        .def("__iter__", [](const Action& action) {
            return nb::iter(nb::make_tuple(
                action.color, action.type, action_value_to_python(action.type, action.value)));
        })
        .def("__repr__", [](const Action& action) {
            return nb::str("Action({!r}, {!r}, {!r})")
                .format(action.color, action.type, action_value_to_python(action.type, action.value));
        });
}

}  // namespace cppanatron::python
