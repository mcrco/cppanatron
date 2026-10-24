#pragma once

#include <array>
#include <optional>
#include <vector>

#include "cppanatron/action.hpp"
#include "cppanatron/dice.hpp"
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
    /**
     * Which playable kinds (knight, year of plenty, monopoly, road building)
     * the seat held when its turn began; unset: every kind it holds. A kind
     * played this turn stays set with none left.
     */
    std::optional<std::array<bool, 4>> development_card_owned_at_start;
    /** Cards still to discard (the discard prompt only). */
    int discards{0};
    /**
     * Catanatron's per-player LONGEST_ROAD_LENGTH, as last published; unset:
     * the cached length (0 during the initial placement, when play does not
     * publish it yet).
     */
    std::optional<int> longest_road_length;
    /**
     * Catanatron's board.road_lengths entry for a seat with roads (play only
     * raises it until a cut recomputes it); unset: the longest road now.
     */
    std::optional<int> cached_road_length;
    /**
     * The seat's road networks as play keeps them (Catanatron's
     * board.connected_components for its color, in order); unset: rebuilt
     * from the pieces. They differ only where a road was built into an
     * opponent's existing building, which play leaves out of the network.
     */
    std::optional<std::vector<std::vector<int>>> road_components;
    /** Completed turns when the seat last played a knight / bought a card; -1: never. */
    int last_knight_completed_turn{-1};
    int last_development_card_bought_completed_turn{-1};
};

/**
 * An exact position, applied to a freshly created game by Game::apply_setup.
 *
 * The bank and deck are what the game's rules start with minus every hand
 * and played card; victory points, Longest Road and Largest Army follow from
 * the pieces and cards. Positions are at `current_player`'s turn at any
 * prompt but trade negotiations:
 *
 *   play_turn                 before or after the roll; `free_roads` > 0
 *                             while a road building card is being played
 *   discard                   after a 7: `acting_player` is the first seat
 *                             (in seat order) with `discards` left
 *   move_robber               after a 7 (rolled) or a knight
 *   build_initial_settlement  the initial placement, its state read from the
 *   build_initial_road        pieces: the seats placed so far in snake order,
 *                             the last of a seat's settlements being its newest
 *
 * By default every held development card counts as held since the turn began.
 *
 * Road lengths count a road's last edge into an opponent's building, as the
 * rules do. In play the engine, like Catanatron, counts that edge only if the
 * road was built before the building, so a declared copy of a played
 * position can have a road one longer, unless it also declares the seats'
 * road networks (road_components) and cached and published lengths: with
 * every optional field given, a declaration reproduces a played position
 * exactly (Catanatron's views carry all of them).
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
    ActionPrompt prompt{ActionPrompt::play_turn};
    /** The seat to act when it is not the turn owner: a discarding seat. */
    std::optional<int> acting_player;
    /** Free roads left of a road building card (1 or 2), at play_turn. */
    int free_roads{0};
    /**
     * The length Longest Road was won with (Catanatron's board.road_length);
     * unset: the holder's cached length.
     */
    std::optional<int> longest_road_length;
    /**
     * The balanced dice (DiceMode::balanced only; its `seats` the game's);
     * unset: a full deck and no history. Declared at an initial-placement
     * prompt, its deck is refilled when the main phase starts.
     */
    std::optional<BalancedDice> dice;
};

}  // namespace cppanatron
