#include "cppanatron/graph_action_space.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <variant>

namespace cppanatron {
namespace {

constexpr std::array<int, 3> kTradeRatios{4, 3, 2};

std::size_t resource_index(Resource resource) { return static_cast<std::size_t>(resource); }

/** Year-of-plenty pair (a <= b) to its position among the 15 pairs. */
std::size_t pair_position(std::size_t first, std::size_t second) {
    std::size_t position = 0;
    for (std::size_t row = 0; row < first; ++row) {
        position += kResources.size() - row;
    }
    return position + (second - first);
}

int seat_of(const Game& game, Color color) {
    const auto& colors = game.colors();
    const auto it = std::find(colors.begin(), colors.end(), color);
    if (it == colors.end()) {
        throw std::out_of_range("color is not seated in this game");
    }
    return static_cast<int>(std::distance(colors.begin(), it));
}

[[noreturn]] void outside(const std::string& what) {
    throw std::out_of_range(what + " is outside the graph action space");
}

}  // namespace

GraphActionSpace::GraphActionSpace(EncoderDims dims) : dims_(dims) {
    dims_.validate();
    size_ = robber_offset() +
            static_cast<std::size_t>(dims_.max_tiles) * static_cast<std::size_t>(dims_.max_players) +
            1;
}

std::size_t GraphActionSpace::index(const Game& game, const Action& action) const {
    const CatanMap& map = game.board().map();
    switch (action.type) {
        case ActionType::roll:
            return kRoll;
        case ActionType::end_turn:
            return kEndTurn;
        case ActionType::buy_development_card:
            return kBuyDevelopmentCard;
        case ActionType::play_knight_card:
            return kPlayKnight;
        case ActionType::play_road_building:
            return kPlayRoadBuilding;
        case ActionType::discard_resource:
            return kDiscard + resource_index(std::get<Resource>(action.value));
        case ActionType::play_monopoly:
            return kMonopoly + resource_index(std::get<Resource>(action.value));
        case ActionType::play_year_of_plenty: {
            const auto& cards = std::get<std::vector<Resource>>(action.value);
            if (cards.size() == 1) {
                return kYearOfPlenty + 15 + resource_index(cards[0]);
            }
            if (cards.size() != 2) {
                outside("a year of plenty of " + std::to_string(cards.size()) + " cards");
            }
            const std::size_t first = resource_index(cards[0]);
            const std::size_t second = resource_index(cards[1]);
            return kYearOfPlenty + pair_position(std::min(first, second), std::max(first, second));
        }
        case ActionType::maritime_trade: {
            const auto& cards = std::get<MaritimeTrade>(action.value).cards;
            if (!cards[0].has_value() || !cards[4].has_value()) {
                outside("a maritime trade without a given and an asked card");
            }
            const std::size_t give = resource_index(*cards[0]);
            const std::size_t get = resource_index(*cards[4]);
            const auto ratio = static_cast<int>(std::count(cards.begin(), cards.begin() + 4, cards[0]));
            const auto ratio_it = std::find(kTradeRatios.begin(), kTradeRatios.end(), ratio);
            if (give == get || ratio_it == kTradeRatios.end()) {
                outside("a maritime trade of " + std::to_string(ratio) + " for 1");
            }
            const std::size_t pair = give * 4 + (get < give ? get : get - 1);
            return kMaritimeTrade + pair * kTradeRatios.size() +
                   static_cast<std::size_t>(std::distance(kTradeRatios.begin(), ratio_it));
        }
        case ActionType::build_settlement:
        case ActionType::build_city: {
            const int slot = map.node_slot(std::get<int>(action.value));
            if (slot < 0 || slot >= dims_.max_nodes) {
                outside("node " + std::to_string(std::get<int>(action.value)));
            }
            return (action.type == ActionType::build_settlement ? settlement_offset()
                                                                : city_offset()) +
                   static_cast<std::size_t>(slot);
        }
        case ActionType::build_road: {
            const int slot = map.edge_slot(std::get<Edge>(action.value));
            if (slot < 0 || slot >= dims_.max_edges) {
                outside("an edge");
            }
            return road_offset() + static_cast<std::size_t>(slot);
        }
        case ActionType::move_robber: {
            const RobberMove& move = std::get<RobberMove>(action.value);
            const auto tile = map.coordinate_to_tile().find(move.coordinate);
            const int slot = tile == map.coordinate_to_tile().end()
                                 ? -1
                                 : map.land_tile_slot(tile->second);
            if (slot < 0 || slot >= dims_.max_tiles) {
                outside("a robber tile");
            }
            int victim = 0;
            if (move.victim.has_value()) {
                const int count = static_cast<int>(game.colors().size());
                victim = (seat_of(game, *move.victim) - seat_of(game, action.color) + count) % count;
                if (victim == 0 || victim >= dims_.max_players) {
                    outside("a robber victim");
                }
            }
            return robber_offset() +
                   static_cast<std::size_t>(slot) * static_cast<std::size_t>(dims_.max_players) +
                   static_cast<std::size_t>(victim);
        }
        case ActionType::offer_trade:
        case ActionType::accept_trade:
        case ActionType::reject_trade:
        case ActionType::confirm_trade:
        case ActionType::cancel_trade:
            outside("a domestic trade");
    }
    outside("an unknown action type");
}

Action GraphActionSpace::decode(const Game& game, std::size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("action index " + std::to_string(index) + " is out of range");
    }
    if (index == noop()) {
        throw std::invalid_argument("the no-op has no game action");
    }
    const Color color = game.current_color();
    const CatanMap& map = game.board().map();
    if (index < kNumGlobals) {
        if (index == kRoll) {
            return {color, ActionType::roll, std::monostate{}};
        }
        if (index == kEndTurn) {
            return {color, ActionType::end_turn, std::monostate{}};
        }
        if (index == kBuyDevelopmentCard) {
            return {color, ActionType::buy_development_card, std::monostate{}};
        }
        if (index == kPlayKnight) {
            return {color, ActionType::play_knight_card, std::monostate{}};
        }
        if (index == kPlayRoadBuilding) {
            return {color, ActionType::play_road_building, std::monostate{}};
        }
        if (index < kMonopoly) {
            return {color, ActionType::discard_resource, kResources[index - kDiscard]};
        }
        if (index < kYearOfPlenty) {
            return {color, ActionType::play_monopoly, kResources[index - kMonopoly]};
        }
        if (index < kMaritimeTrade) {
            std::size_t position = index - kYearOfPlenty;
            if (position >= 15) {
                return {
                    color, ActionType::play_year_of_plenty,
                    std::vector<Resource>{kResources[position - 15]}};
            }
            for (std::size_t first = 0; first < kResources.size(); ++first) {
                const std::size_t row = kResources.size() - first;
                if (position < row) {
                    return {
                        color, ActionType::play_year_of_plenty,
                        std::vector<Resource>{kResources[first], kResources[first + position]}};
                }
                position -= row;
            }
        }
        const std::size_t position = index - kMaritimeTrade;
        const std::size_t pair = position / kTradeRatios.size();
        const int ratio = kTradeRatios[position % kTradeRatios.size()];
        const std::size_t give = pair / 4;
        std::size_t get = pair % 4;
        if (get >= give) {
            ++get;
        }
        MaritimeTrade trade;
        for (int i = 0; i < ratio; ++i) {
            trade.cards[static_cast<std::size_t>(i)] = kResources[give];
        }
        trade.cards[4] = kResources[get];
        return {color, ActionType::maritime_trade, trade};
    }

    const auto require_slot = [](int slot, std::size_t count, const char* what) {
        if (slot < 0 || static_cast<std::size_t>(slot) >= count) {
            throw std::out_of_range(std::string("this map has no ") + what + " slot " +
                                    std::to_string(slot));
        }
        return static_cast<std::size_t>(slot);
    };
    if (index < road_offset()) {
        const bool settlement = index < city_offset();
        const int slot = static_cast<int>(index - (settlement ? settlement_offset() : city_offset()));
        const int node =
            map.land_node_list()[require_slot(slot, map.land_node_list().size(), "node")];
        return {color, settlement ? ActionType::build_settlement : ActionType::build_city, node};
    }
    if (index < robber_offset()) {
        const int slot = static_cast<int>(index - road_offset());
        return {
            color, ActionType::build_road,
            map.land_edges()[require_slot(slot, map.land_edges().size(), "edge")]};
    }
    const std::size_t position = index - robber_offset();
    const auto players = static_cast<std::size_t>(dims_.max_players);
    const int slot = static_cast<int>(position / players);
    const int tile_index =
        map.land_tile_indices()[require_slot(slot, map.land_tile_indices().size(), "tile")];
    const Coordinate coordinate = map.spec().tiles[static_cast<std::size_t>(tile_index)].coordinate;
    const auto victim_slot = static_cast<int>(position % players);
    std::optional<Color> victim;
    if (victim_slot != 0) {
        const int count = static_cast<int>(game.colors().size());
        if (victim_slot >= count) {
            throw std::out_of_range("robber victim slot " + std::to_string(victim_slot) +
                                    " has no seat in this game");
        }
        victim = game.colors()[static_cast<std::size_t>(
            (game.current_player_index() + victim_slot) % count)];
    }
    return {color, ActionType::move_robber, RobberMove{coordinate, victim}};
}

void GraphActionSpace::mask(const Game& game, std::span<std::uint8_t> out) const {
    if (out.size() != size_) {
        throw std::invalid_argument("mask buffer must have one entry per action");
    }
    dims_.require_fits(game);
    std::fill(out.begin(), out.end(), std::uint8_t{0});
    for (const Action& action : game.playable_actions()) {
        switch (action.type) {
            case ActionType::offer_trade:
            case ActionType::accept_trade:
            case ActionType::reject_trade:
            case ActionType::confirm_trade:
            case ActionType::cancel_trade:
                continue;
            default:
                out[index(game, action)] = 1;
        }
    }
}

std::string GraphActionSpace::name(std::size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("action index " + std::to_string(index) + " is out of range");
    }
    const auto resource = [](std::size_t i) { return std::string(to_string(kResources[i])); };
    if (index == noop()) {
        return "NOOP";
    }
    if (index < kDiscard) {
        constexpr std::array<const char*, 5> kNames{
            "ROLL", "END_TURN", "BUY_DEVELOPMENT_CARD", "PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING"};
        return kNames[index];
    }
    if (index < kMonopoly) {
        return "DISCARD_" + resource(index - kDiscard);
    }
    if (index < kYearOfPlenty) {
        return "MONOPOLY_" + resource(index - kMonopoly);
    }
    if (index < kMaritimeTrade) {
        std::size_t position = index - kYearOfPlenty;
        if (position >= 15) {
            return "YEAR_OF_PLENTY_" + resource(position - 15);
        }
        for (std::size_t first = 0;; ++first) {
            const std::size_t row = kResources.size() - first;
            if (position < row) {
                return "YEAR_OF_PLENTY_" + resource(first) + "_" + resource(first + position);
            }
            position -= row;
        }
    }
    if (index < kNumGlobals) {
        const std::size_t position = index - kMaritimeTrade;
        const std::size_t pair = position / kTradeRatios.size();
        const std::size_t give = pair / 4;
        std::size_t get = pair % 4;
        if (get >= give) {
            ++get;
        }
        return "MARITIME_" + resource(give) + "_" + resource(get) + "_" +
               std::to_string(kTradeRatios[position % kTradeRatios.size()]);
    }
    if (index < city_offset()) {
        return "SETTLEMENT@" + std::to_string(index - settlement_offset());
    }
    if (index < road_offset()) {
        return "CITY@" + std::to_string(index - city_offset());
    }
    if (index < robber_offset()) {
        return "ROAD@" + std::to_string(index - road_offset());
    }
    const std::size_t position = index - robber_offset();
    const auto players = static_cast<std::size_t>(dims_.max_players);
    return "ROBBER@" + std::to_string(position / players) + "/" +
           std::to_string(position % players);
}

}  // namespace cppanatron
