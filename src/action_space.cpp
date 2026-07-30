#include "cppanatron/action_space.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cppanatron {
namespace {

std::string resource_repr(Resource resource) {
    return "'" + std::string(to_string(resource)) + "'";
}

std::string coordinate_repr(Coordinate coordinate) {
    return "(" + std::to_string(coordinate.x) + ", " + std::to_string(coordinate.y) +
           ", " + std::to_string(coordinate.z) + ")";
}

std::string action_type_name(ActionType type) {
    switch (type) {
        case ActionType::roll:
            return "ROLL";
        case ActionType::move_robber:
            return "MOVE_ROBBER";
        case ActionType::discard_resource:
            return "DISCARD_RESOURCE";
        case ActionType::build_road:
            return "BUILD_ROAD";
        case ActionType::build_settlement:
            return "BUILD_SETTLEMENT";
        case ActionType::build_city:
            return "BUILD_CITY";
        case ActionType::buy_development_card:
            return "BUY_DEVELOPMENT_CARD";
        case ActionType::play_knight_card:
            return "PLAY_KNIGHT_CARD";
        case ActionType::play_year_of_plenty:
            return "PLAY_YEAR_OF_PLENTY";
        case ActionType::play_monopoly:
            return "PLAY_MONOPOLY";
        case ActionType::play_road_building:
            return "PLAY_ROAD_BUILDING";
        case ActionType::maritime_trade:
            return "MARITIME_TRADE";
        case ActionType::offer_trade:
            return "OFFER_TRADE";
        case ActionType::accept_trade:
            return "ACCEPT_TRADE";
        case ActionType::reject_trade:
            return "REJECT_TRADE";
        case ActionType::confirm_trade:
            return "CONFIRM_TRADE";
        case ActionType::cancel_trade:
            return "CANCEL_TRADE";
        case ActionType::end_turn:
            return "END_TURN";
    }
    return "";
}

std::string value_repr(const ActionValue& value) {
    if (std::holds_alternative<std::monostate>(value)) {
        return "None";
    }
    if (const auto* node = std::get_if<int>(&value)) {
        return std::to_string(*node);
    }
    if (const auto* edge = std::get_if<Edge>(&value)) {
        const Edge normalized = edge->normalized();
        return "(" + std::to_string(normalized.a) + ", " + std::to_string(normalized.b) +
               ")";
    }
    if (const auto* resource = std::get_if<Resource>(&value)) {
        return resource_repr(*resource);
    }
    if (const auto* move = std::get_if<FlatRobberMove>(&value)) {
        return "(" + coordinate_repr(move->coordinate) + ", " +
               (move->victim_slot.has_value() ? std::to_string(*move->victim_slot)
                                              : "None") +
               ")";
    }
    if (const auto* trade = std::get_if<MaritimeTrade>(&value)) {
        std::string result = "(";
        for (std::size_t i = 0; i < trade->cards.size(); ++i) {
            if (i != 0) {
                result += ", ";
            }
            result += trade->cards[i].has_value() ? resource_repr(*trade->cards[i]) : "None";
        }
        return result + ")";
    }
    if (const auto* cards = std::get_if<std::vector<Resource>>(&value)) {
        std::string result = "(";
        for (std::size_t i = 0; i < cards->size(); ++i) {
            if (i != 0) {
                result += ", ";
            }
            result += resource_repr((*cards)[i]);
        }
        if (cards->size() == 1) {
            result += ",";
        }
        return result + ")";
    }
    throw std::logic_error("unsupported flat action value");
}

}  // namespace

std::string flat_action_key(const Action& action) {
    // CatanRL sorts `(ActionType, value)` by its Python string form. The type
    // prefix is equivalent to sorting by the enum member name; value_repr
    // reproduces tuple/string/None formatting for the supported value shapes.
    return action_type_name(action.type) + "|" + value_repr(action.value);
}

FlatActionSpace::FlatActionSpace(int num_players, MapType map_type) {
    if (num_players < 1 || num_players > 4) {
        throw std::invalid_argument("num_players must be between one and four");
    }
    const CatanMap map = CatanMap::build(map_type, 0);
    const Color placeholder = Color::red;

    actions_.push_back({placeholder, ActionType::roll, std::monostate{}});
    for (Resource resource : kResources) {
        actions_.push_back({placeholder, ActionType::discard_resource, resource});
    }
    for (Edge edge : map.land_edges()) {
        actions_.push_back({placeholder, ActionType::build_road, edge});
    }
    for (int node = 0; node < map.num_nodes(); ++node) {
        actions_.push_back({placeholder, ActionType::build_settlement, node});
        actions_.push_back({placeholder, ActionType::build_city, node});
    }
    actions_.push_back(
        {placeholder, ActionType::buy_development_card, std::monostate{}});
    actions_.push_back({placeholder, ActionType::play_knight_card, std::monostate{}});

    for (std::size_t first = 0; first < kResources.size(); ++first) {
        for (std::size_t second = first; second < kResources.size(); ++second) {
            actions_.push_back(
                {placeholder,
                 ActionType::play_year_of_plenty,
                 std::vector<Resource>{kResources[first], kResources[second]}});
        }
        actions_.push_back(
            {placeholder,
             ActionType::play_year_of_plenty,
             std::vector<Resource>{kResources[first]}});
    }
    actions_.push_back(
        {placeholder, ActionType::play_road_building, std::monostate{}});
    for (Resource resource : kResources) {
        actions_.push_back({placeholder, ActionType::play_monopoly, resource});
    }

    for (int tile_index : map.land_tile_indices()) {
        Coordinate coordinate{};
        for (const auto& [candidate, candidate_index] : map.coordinate_to_tile()) {
            if (candidate_index == tile_index) {
                coordinate = candidate;
                break;
            }
        }
        actions_.push_back(
            {placeholder, ActionType::move_robber, FlatRobberMove{coordinate, std::nullopt}});
        for (int slot = 1; slot < num_players; ++slot) {
            actions_.push_back(
                {placeholder, ActionType::move_robber, FlatRobberMove{coordinate, slot}});
        }
    }

    for (Resource offered : kResources) {
        for (Resource received : kResources) {
            if (offered == received) {
                continue;
            }
            for (int rate : {4, 3, 2}) {
                MaritimeTrade trade;
                for (int i = 0; i < rate; ++i) {
                    trade.cards[static_cast<std::size_t>(i)] = offered;
                }
                trade.cards[4] = received;
                actions_.push_back({placeholder, ActionType::maritime_trade, trade});
            }
        }
    }
    actions_.push_back({placeholder, ActionType::end_turn, std::monostate{}});

    std::sort(
        actions_.begin(),
        actions_.end(),
        [](const Action& lhs, const Action& rhs) {
            return flat_action_key(lhs) < flat_action_key(rhs);
        });
}

std::size_t FlatActionSpace::index(const Action& action) const {
    Action normalized = action;
    normalized.color = Color::red;
    if (auto* edge = std::get_if<Edge>(&normalized.value)) {
        *edge = edge->normalized();
    }
    const auto it = std::find(actions_.begin(), actions_.end(), normalized);
    if (it == actions_.end()) {
        throw std::out_of_range("action is not in flat action space");
    }
    return static_cast<std::size_t>(std::distance(actions_.begin(), it));
}

std::size_t FlatActionSpace::index(
    const Action& action,
    const std::vector<Color>& game_colors) const {
    if (action.type != ActionType::move_robber) {
        return index(action);
    }
    const RobberMove& move = std::get<RobberMove>(action.value);
    std::optional<int> slot;
    if (move.victim.has_value()) {
        const auto actor = std::find(
            game_colors.begin(), game_colors.end(), action.color);
        const auto victim = std::find(
            game_colors.begin(), game_colors.end(), *move.victim);
        if (actor == game_colors.end() || victim == game_colors.end()) {
            throw std::out_of_range("robber actor or victim is not in game");
        }
        const int actor_index =
            static_cast<int>(std::distance(game_colors.begin(), actor));
        const int victim_index =
            static_cast<int>(std::distance(game_colors.begin(), victim));
        const int count = static_cast<int>(game_colors.size());
        slot = (victim_index - actor_index + count) % count;
        if (*slot == 0) {
            throw std::logic_error("robber cannot target its actor");
        }
    }
    return index(
        {Color::red,
         ActionType::move_robber,
         FlatRobberMove{move.coordinate, slot}});
}

Action FlatActionSpace::decode(
    std::size_t action_index,
    Color actor,
    const std::vector<Color>& game_colors) const {
    Action result = at(action_index);
    result.color = actor;
    if (result.type != ActionType::move_robber) {
        return result;
    }
    const FlatRobberMove flat = std::get<FlatRobberMove>(result.value);
    std::optional<Color> victim;
    if (flat.victim_slot.has_value()) {
        const auto actor_it =
            std::find(game_colors.begin(), game_colors.end(), actor);
        if (actor_it == game_colors.end()) {
            throw std::out_of_range("robber actor is not in game");
        }
        const int actor_index =
            static_cast<int>(std::distance(game_colors.begin(), actor_it));
        const int victim_index =
            (actor_index + *flat.victim_slot) %
            static_cast<int>(game_colors.size());
        victim = game_colors[static_cast<std::size_t>(victim_index)];
    }
    result.value = RobberMove{flat.coordinate, victim};
    return result;
}

}  // namespace cppanatron
