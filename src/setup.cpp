#include <algorithm>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "cppanatron/game.hpp"
#include "cppanatron/random.hpp"

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
    const int n = static_cast<int>(num_players);
    if (setup.players.size() != num_players) {
        throw std::invalid_argument("setup needs one player entry per seat");
    }
    if (setup.current_player < 0 || setup.current_player >= n) {
        throw std::invalid_argument("current player is not a seat");
    }
    if (setup.completed_turns < 0) {
        throw std::invalid_argument("completed turns must be non-negative");
    }
    const ActionPrompt prompt = setup.prompt;
    const bool initial = prompt == ActionPrompt::build_initial_settlement ||
                         prompt == ActionPrompt::build_initial_road;
    if (prompt == ActionPrompt::decide_trade || prompt == ActionPrompt::decide_acceptees) {
        throw std::invalid_argument("positions inside a trade negotiation cannot be declared");
    }
    const int acting = setup.acting_player.value_or(setup.current_player);
    if (acting < 0 || acting >= n) {
        throw std::invalid_argument("acting player is not a seat");
    }
    if (prompt != ActionPrompt::discard && acting != setup.current_player) {
        throw std::invalid_argument("only a discarding seat acts out of turn");
    }
    if (setup.free_roads < 0 || setup.free_roads > 2 ||
        (setup.free_roads > 0 && prompt != ActionPrompt::play_turn)) {
        throw std::invalid_argument("free roads are 0, 1 or 2, and only at play_turn");
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
        const int hand = std::accumulate(player.resources.begin(), player.resources.end(), 0);
        if (player.discards < 0 || player.discards > hand ||
            (player.discards > 0 && prompt != ActionPrompt::discard)) {
            throw std::invalid_argument(
                who + " can only have cards to discard at the discard prompt, at most its hand");
        }
        const auto turn = [&](int completed) {
            return completed >= -1 && completed <= setup.completed_turns;
        };
        if (!turn(player.last_knight_completed_turn) ||
            !turn(player.last_development_card_bought_completed_turn)) {
            throw std::invalid_argument(who + " played or bought a card after the current turn");
        }
        if ((player.longest_road_length.has_value() && *player.longest_road_length < 0) ||
            (player.cached_road_length.has_value() && *player.cached_road_length < 0)) {
            throw std::invalid_argument(who + " has a negative road length");
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

    // The initial placement: who places next follows from the pieces, in
    // snake order (seats 0 .. n-1, then n-1 .. 0), each settlement then road.
    int placements = 0;  // settlement-and-road pairs completed
    if (initial) {
        int settlements = 0;
        for (const PlayerSetup& player : setup.players) {
            settlements += static_cast<int>(player.settlements.size());
            if (!player.cities.empty() ||
                std::accumulate(player.development_cards.begin(), player.development_cards.end(), 0) +
                        std::accumulate(player.played_development_cards.begin(),
                                        player.played_development_cards.end(), 0) >
                    0) {
                throw std::invalid_argument("the initial placement has no cities or development cards");
            }
        }
        const bool road_next = prompt == ActionPrompt::build_initial_road;
        placements = settlements - (road_next ? 1 : 0);
        if (placements < 0 || placements >= 2 * n) {
            throw std::invalid_argument("the pieces do not fit the initial placement");
        }
        const int placer = placements < n ? placements : 2 * n - 1 - placements;
        const int second_round = std::max(0, settlements - n);
        for (int seat = 0; seat < n; ++seat) {
            const PlayerSetup& player = setup.players[static_cast<std::size_t>(seat)];
            const int expected = (seat < std::min(settlements, n) ? 1 : 0) +
                                 (seat >= n - second_round ? 1 : 0);
            const int expected_roads =
                static_cast<int>(player.settlements.size()) - (road_next && seat == placer ? 1 : 0);
            if (static_cast<int>(player.settlements.size()) != expected ||
                static_cast<int>(player.roads.size()) != expected_roads) {
                throw std::invalid_argument(
                    "seat " + std::to_string(seat) +
                    "'s pieces do not fit the initial placement in snake order");
            }
        }
        if (setup.current_player != placer || setup.completed_turns != 0 || setup.has_rolled ||
            setup.has_played_development_card) {
            throw std::invalid_argument(
                "in the initial placement seat " + std::to_string(placer) +
                " places, on turn 0, before any roll");
        }
    }
    if (prompt == ActionPrompt::discard) {
        if (!setup.has_rolled ||
            setup.players[static_cast<std::size_t>(acting)].discards == 0) {
            throw std::invalid_argument("the discard prompt needs a roll and a seat to discard");
        }
        for (int seat = 0; seat < acting; ++seat) {
            if (setup.players[static_cast<std::size_t>(seat)].discards > 0) {
                throw std::invalid_argument("seats discard in seat order: an earlier one is left");
            }
        }
    }
    if (prompt == ActionPrompt::move_robber && !setup.has_rolled &&
        !setup.has_played_development_card) {
        throw std::invalid_argument("the robber moves after a roll of 7 or a knight");
    }
    if (setup.free_roads > 0 && !setup.has_played_development_card) {
        throw std::invalid_argument("free roads come from a road building card played this turn");
    }
    if (setup.largest_army_taken_from.has_value() &&
        (!friendly_robber_ || prompt != ActionPrompt::move_robber || !setup.has_played_development_card)) {
        throw std::invalid_argument(
            "largest_army_taken_from is for a knight's robber prompt under the friendly robber");
    }

    // Played from the start, everyone builds a first road in seating order.
    board_.restore(buildings, roads, setup.robber.value_or(board_.robber_coordinate()), colors_);
    for (std::size_t seat = 0; seat < num_players; ++seat) {
        const PlayerSetup& player = setup.players[seat];
        if (player.road_components.has_value()) {
            std::vector<std::set<int>> components;
            for (const auto& component : *player.road_components) {
                components.emplace_back(component.begin(), component.end());
            }
            board_.set_road_components(colors_[seat], std::move(components));
        }
        if (player.cached_road_length.has_value()) {
            board_.set_road_length(colors_[seat], *player.cached_road_length);
        }
    }
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
    if (setup.largest_army_taken_from.has_value()) {
        // The knight just played took the award: its former holder trails
        // the current player by that one knight.
        const int loser = *setup.largest_army_taken_from;
        if (loser < 0 || loser >= n || loser == setup.current_player ||
            army_holder != setup.current_player ||
            knights[static_cast<std::size_t>(loser)] < 3 ||
            knights[static_cast<std::size_t>(loser)] !=
                knights[static_cast<std::size_t>(setup.current_player)] - 1) {
            throw std::invalid_argument(
                "largest_army_taken_from must be a seat the current player's last knight took Largest Army from");
        }
    }
    board_.set_longest_road_holder(
        road_holder.has_value()
            ? std::optional<Color>(colors_[static_cast<std::size_t>(*road_holder)])
            : std::nullopt,
        setup.longest_road_length);

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
            player.development_card_owned_at_start[i] =
                setup_player.development_card_owned_at_start.has_value()
                    ? (*setup_player.development_card_owned_at_start)[i]
                    : player.development_cards[i] > 0;
        }
        const bool current = static_cast<int>(seat) == setup.current_player;
        player.has_rolled = current && setup.has_rolled;
        player.has_played_development_card_in_turn = current && setup.has_played_development_card;
        // Play publishes lengths only from the first build after the placement.
        player.longest_road_length =
            setup_player.longest_road_length.value_or(initial ? 0 : road_lengths[seat]);
        player.has_road = road_holder == static_cast<int>(seat);
        player.has_army = army_holder == static_cast<int>(seat);
        player.last_knight_completed_turn = setup_player.last_knight_completed_turn;
        player.last_development_card_bought_completed_turn =
            setup_player.last_development_card_bought_completed_turn;
        player.victory_points = static_cast<int>(player.settlements.size()) +
                                2 * static_cast<int>(player.cities.size()) +
                                (player.has_road ? 2 : 0) + (player.has_army ? 2 : 0);
        player.actual_victory_points =
            player.victory_points + player.development_cards[victory_point];
        discard_counts_[seat] = setup_player.discards;
    }
    if (setup.free_roads > 0) {
        const Color color = colors_[static_cast<std::size_t>(setup.current_player)];
        if (players_[static_cast<std::size_t>(setup.current_player)].roads_available == 0 ||
            board_.buildable_edges(color).empty()) {
            throw std::invalid_argument("road building ends when no road can be built");
        }
    }

    resource_bank_ = bank;
    development_deck_.clear();
    for (std::size_t i = 0; i < deck.size(); ++i) {
        development_deck_.insert(
            development_deck_.end(), static_cast<std::size_t>(deck[i]),
            static_cast<DevelopmentCard>(i));
    }
    random::shuffle(development_deck_.begin(), development_deck_.end(), random_);

    is_initial_build_phase_ = initial;
    current_prompt_ = prompt;
    current_player_index_ = acting;
    current_turn_index_ = setup.current_player;
    completed_turns_ = setup.completed_turns;
    is_discarding_ = prompt == ActionPrompt::discard;
    is_moving_knight_ = prompt == ActionPrompt::move_robber;
    largest_army_taken_from_ = setup.largest_army_taken_from;
    is_road_building_ = setup.free_roads > 0;
    free_roads_available_ = setup.free_roads;
    if (initial) {
        // The turn passes after every pair but the n-th (the snake's turn).
        num_turns_ = placements < n ? placements : placements - 1;
    } else {
        // The initial build phase passes the turn 2 * (players - 1) times.
        num_turns_ = setup.completed_turns + 2 * (n - 1);
    }
    playable_actions_ = generate_playable_actions();
}

}  // namespace cppanatron
