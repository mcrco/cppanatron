#include <algorithm>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "cppanatron/game.hpp"

namespace cppanatron {
namespace {

constexpr std::array<const char*, 5> kResourceNames{"WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"};
constexpr std::array<const char*, 5> kCardNames{
    "KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"};

/**
 * The seat holding an award: ``requested`` when set (-1 for nobody), else the
 * unique leader at or above ``threshold``. An explicit holder must lead with
 * at least ``minimum_holder_score``.
 */
std::optional<int> resolve_award(
    std::optional<int> requested,
    const std::vector<int>& scores,
    int threshold,
    int minimum_holder_score,
    const std::string& award) {
    const int best = scores.empty() ? 0 : *std::max_element(scores.begin(), scores.end());
    if (requested.has_value()) {
        if (*requested == -1) {
            if (best >= threshold) {
                throw std::invalid_argument(award + " must have a holder once someone qualifies");
            }
            return std::nullopt;
        }
        if (*requested < 0 || *requested >= static_cast<int>(scores.size())) {
            throw std::invalid_argument(award + " holder is not a seat");
        }
        const int score = scores[static_cast<std::size_t>(*requested)];
        if (score != best || score < minimum_holder_score) {
            throw std::invalid_argument(award + " holder must lead");
        }
        return requested;
    }
    if (best < threshold) {
        return std::nullopt;
    }
    std::optional<int> leader;
    for (std::size_t seat = 0; seat < scores.size(); ++seat) {
        if (scores[seat] != best) {
            continue;
        }
        if (leader.has_value()) {
            throw std::invalid_argument(award + " is tied; say who holds it");
        }
        leader = static_cast<int>(seat);
    }
    return leader;
}

}  // namespace

void Game::apply_setup(const GameSetup& setup) {
    if (!is_initial_build_phase_ || num_turns_ != 0 || !board_.buildings().empty() ||
        !board_.roads().empty()) {
        throw std::logic_error("apply_setup needs a freshly created game");
    }
    const std::size_t num_players = colors_.size();
    if (setup.players.size() != num_players) {
        throw std::invalid_argument("setup needs one player entry per seat");
    }
    if (setup.current_player < 0 || setup.current_player >= static_cast<int>(num_players)) {
        throw std::invalid_argument("current player is not a seat");
    }
    if (setup.completed_turns < 0) {
        throw std::invalid_argument("completed turns must be non-negative");
    }

    std::map<int, BuildingState> buildings;
    std::map<Edge, Color> roads;
    std::array<int, 5> bank = rules_.resource_bank;
    std::array<int, 5> deck = rules_.development_deck;
    constexpr auto victory_point = static_cast<std::size_t>(DevelopmentCard::victory_point);
    for (std::size_t seat = 0; seat < num_players; ++seat) {
        const PlayerSetup& player = setup.players[seat];
        const Color color = colors_[seat];
        const std::string who = "seat " + std::to_string(seat);
        if (static_cast<int>(player.settlements.size()) > rules_.settlements_per_player ||
            static_cast<int>(player.cities.size()) > rules_.cities_per_player ||
            static_cast<int>(player.roads.size()) > rules_.roads_per_player) {
            throw std::invalid_argument(who + " has more pieces than the rules allow");
        }
        const auto place = [&](int node_id, Building building) {
            if (!buildings.emplace(node_id, BuildingState{color, building}).second) {
                throw std::invalid_argument(
                    "node " + std::to_string(node_id) + " has more than one building");
            }
        };
        for (int node_id : player.settlements) {
            place(node_id, Building::settlement);
        }
        for (int node_id : player.cities) {
            place(node_id, Building::city);
        }
        for (Edge edge : player.roads) {
            if (!roads.emplace(edge.normalized(), color).second) {
                throw std::invalid_argument(
                    "edge (" + std::to_string(edge.a) + ", " + std::to_string(edge.b) +
                    ") has more than one road");
            }
        }
        for (std::size_t i = 0; i < 5; ++i) {
            if (player.resources[i] < 0 || player.development_cards[i] < 0 ||
                player.played_development_cards[i] < 0) {
                throw std::invalid_argument(who + " has a negative card count");
            }
            bank[i] -= player.resources[i];
            deck[i] -= player.development_cards[i] + player.played_development_cards[i];
        }
        if (player.played_development_cards[victory_point] != 0) {
            throw std::invalid_argument("victory point cards are never played");
        }
    }
    for (std::size_t i = 0; i < 5; ++i) {
        if (bank[i] < 0) {
            throw std::invalid_argument(
                std::string("hands hold more ") + kResourceNames[i] + " than the bank has");
        }
        if (deck[i] < 0) {
            throw std::invalid_argument(
                std::string("players hold more ") + kCardNames[i] + " cards than the deck has");
        }
    }

    board_.restore(buildings, roads, setup.robber.value_or(board_.robber_coordinate()));
    std::vector<int> road_lengths(num_players);
    std::vector<int> knights(num_players);
    for (std::size_t seat = 0; seat < num_players; ++seat) {
        road_lengths[seat] = board_.longest_road(colors_[seat]);
        knights[seat] = setup.players[seat].played_development_cards[
            static_cast<std::size_t>(DevelopmentCard::knight)];
    }
    // Longest Road can stay with a road shorter than five after a cut.
    const auto road_holder =
        resolve_award(setup.longest_road, road_lengths, 5, 1, "Longest Road");
    const auto army_holder = resolve_award(setup.largest_army, knights, 3, 3, "Largest Army");
    board_.set_longest_road_holder(
        road_holder.has_value()
            ? std::optional<Color>(colors_[static_cast<std::size_t>(*road_holder)])
            : std::nullopt);

    for (std::size_t seat = 0; seat < num_players; ++seat) {
        const PlayerSetup& setup_player = setup.players[seat];
        PlayerState& player = players_[seat];
        player.settlements = setup_player.settlements;
        player.cities = setup_player.cities;
        player.roads.clear();
        for (Edge edge : setup_player.roads) {
            player.roads.push_back(edge.normalized());
        }
        player.settlements_available =
            rules_.settlements_per_player - static_cast<int>(player.settlements.size());
        player.cities_available =
            rules_.cities_per_player - static_cast<int>(player.cities.size());
        player.roads_available = rules_.roads_per_player - static_cast<int>(player.roads.size());
        player.resources = setup_player.resources;
        player.development_cards = setup_player.development_cards;
        player.played_development_cards = setup_player.played_development_cards;
        for (std::size_t i = 0; i < player.development_card_owned_at_start.size(); ++i) {
            player.development_card_owned_at_start[i] = player.development_cards[i] > 0;
        }
        const bool current = static_cast<int>(seat) == setup.current_player;
        player.has_rolled = current && setup.has_rolled;
        player.has_played_development_card_in_turn = current && setup.has_played_development_card;
        player.longest_road_length = road_lengths[seat];
        player.has_road = road_holder == static_cast<int>(seat);
        player.has_army = army_holder == static_cast<int>(seat);
        player.victory_points = static_cast<int>(player.settlements.size()) +
                                2 * static_cast<int>(player.cities.size()) +
                                (player.has_road ? 2 : 0) + (player.has_army ? 2 : 0);
        player.actual_victory_points =
            player.victory_points + player.development_cards[victory_point];
    }

    resource_bank_ = bank;
    development_deck_.clear();
    for (std::size_t i = 0; i < deck.size(); ++i) {
        development_deck_.insert(
            development_deck_.end(), static_cast<std::size_t>(deck[i]),
            static_cast<DevelopmentCard>(i));
    }
    std::shuffle(development_deck_.begin(), development_deck_.end(), random_);

    is_initial_build_phase_ = false;
    current_prompt_ = ActionPrompt::play_turn;
    current_player_index_ = setup.current_player;
    current_turn_index_ = setup.current_player;
    completed_turns_ = setup.completed_turns;
    // The initial build phase passes the turn 2 * (players - 1) times.
    num_turns_ = setup.completed_turns + 2 * (static_cast<int>(num_players) - 1);
    playable_actions_ = generate_playable_actions();
}

}  // namespace cppanatron
