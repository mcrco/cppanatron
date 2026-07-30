#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "cppanatron/types.hpp"

namespace cppanatron {

enum class ActionPrompt : std::uint8_t {
    build_initial_settlement,
    build_initial_road,
    play_turn,
    discard,
    move_robber,
    decide_trade,
    decide_acceptees,
};

enum class ActionType : std::uint8_t {
    roll,
    move_robber,
    discard_resource,
    build_road,
    build_settlement,
    build_city,
    buy_development_card,
    play_knight_card,
    play_year_of_plenty,
    play_monopoly,
    play_road_building,
    maritime_trade,
    offer_trade,
    accept_trade,
    reject_trade,
    confirm_trade,
    cancel_trade,
    end_turn,
};

struct Dice {
    int first{};
    int second{};

    auto operator<=>(const Dice&) const = default;
};

struct RobberMove {
    Coordinate coordinate{};
    std::optional<Color> victim;

    auto operator<=>(const RobberMove&) const = default;
};

struct FlatRobberMove {
    Coordinate coordinate{};
    std::optional<int> victim_slot;

    auto operator<=>(const FlatRobberMove&) const = default;
};

struct MaritimeTrade {
    std::array<std::optional<Resource>, 5> cards{};

    auto operator<=>(const MaritimeTrade&) const = default;
};

using ActionValue =
    std::variant<
        std::monostate,
        int,
        Edge,
        Resource,
        Dice,
        RobberMove,
        FlatRobberMove,
        MaritimeTrade,
        std::vector<Resource>>;

struct Action {
    Color color{};
    ActionType type{};
    ActionValue value{};

    auto operator<=>(const Action&) const = default;
};

}  // namespace cppanatron
