#pragma once

#include <array>
#include <optional>
#include <vector>

#include "cppanatron/types.hpp"

namespace cppanatron {

/** One seat's pieces and cards in a GameSetup. */
struct PlayerSetup {
    std::vector<int> settlements;
    std::vector<int> cities;
    std::vector<Edge> roads;
    /** Hand ordered WOOD, BRICK, SHEEP, WHEAT, ORE. */
    std::array<int, 5> resources{};
    /** Unplayed cards in DevelopmentCard order, victory points included. */
    std::array<int, 5> development_cards{};
    std::array<int, 5> played_development_cards{};
};

/**
 * An exact position, applied to a freshly created game by Game::apply_setup.
 *
 * The bank and deck are what the game's rules start with minus every hand
 * and played card; victory points, Longest Road and Largest Army follow from
 * the pieces and cards. Positions are at the current player's turn, before
 * or after the roll; every held development card is playable.
 */
struct GameSetup {
    /** One entry per seat, in seating order. */
    std::vector<PlayerSetup> players;
    /** Robber tile; the board's starting tile (the desert) when unset. */
    std::optional<Coordinate> robber;
    int current_player{0};
    bool has_rolled{false};
    bool has_played_development_card{false};
    int completed_turns{0};
    /**
     * Award holders by seat; -1 for nobody. Unset means the unique leader
     * (a road of five or more, three or more knights), and a tie must be
     * settled explicitly. An explicit holder must lead, ties allowed.
     */
    std::optional<int> longest_road;
    std::optional<int> largest_army;
};

}  // namespace cppanatron
