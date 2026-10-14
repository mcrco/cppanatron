#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>

namespace cppanatron {

/**
 * Rule settings that vary between games. The defaults are the base game's;
 * the 5-6 player extension uses a bank of 24 per resource and a 34-card deck
 * (20 knights, 3 of each progress card, 5 victory points).
 */
struct GameRules {
    int victory_points_to_win{10};
    int discard_limit{7};
    bool friendly_robber{false};
    /** Starting bank, ordered WOOD, BRICK, SHEEP, WHEAT, ORE. */
    std::array<int, 5> resource_bank{19, 19, 19, 19, 19};
    /**
     * Starting deck in DevelopmentCard order: knight, year of plenty,
     * monopoly, road building, victory point.
     */
    std::array<int, 5> development_deck{14, 2, 2, 2, 5};
    int roads_per_player{15};
    int settlements_per_player{5};
    int cities_per_player{4};

    auto operator<=>(const GameRules&) const = default;
};

/** Throws std::invalid_argument for rules no game can be played under. */
inline void validate_rules(const GameRules& rules) {
    if (rules.victory_points_to_win < 1) {
        throw std::invalid_argument("victory points to win must be positive");
    }
    if (rules.discard_limit < 0) {
        throw std::invalid_argument("discard limit must be non-negative");
    }
    const auto negative = [](const std::array<int, 5>& counts) {
        return std::any_of(counts.begin(), counts.end(), [](int count) { return count < 0; });
    };
    if (negative(rules.resource_bank) || negative(rules.development_deck)) {
        throw std::invalid_argument("bank and deck counts must be non-negative");
    }
    // The initial build phase places two settlements and two roads each.
    if (rules.roads_per_player < 2 || rules.settlements_per_player < 2 ||
        rules.cities_per_player < 0) {
        throw std::invalid_argument(
            "players need at least two roads and two settlements, and no negative cities");
    }
}

}  // namespace cppanatron
