#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cppanatron/board.hpp"
#include "cppanatron/action_space.hpp"
#include "cppanatron/game.hpp"

namespace {

using cppanatron::Board;
using cppanatron::CatanMap;
using cppanatron::Color;
using cppanatron::Edge;
using cppanatron::MapType;
using cppanatron::Resource;
using cppanatron::ActionPrompt;
using cppanatron::ActionType;
using cppanatron::Dice;
using cppanatron::Game;
using cppanatron::FlatActionSpace;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void require_throws(Callable callable, const std::string& message) {
    try {
        callable();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_topology_contract() {
    const auto mini = CatanMap::build(MapType::mini, 7);
    require(mini.num_nodes() == 24, "MINI must have 24 nodes");
    require(mini.land_edges().size() == 30, "MINI must have 30 edges");
    require(mini.land_tile_indices().size() == 7, "MINI must have 7 land tiles");

    const auto base = CatanMap::build(MapType::base, 7);
    require(base.num_nodes() == 54, "BASE must have 54 nodes");
    require(base.land_edges().size() == 72, "BASE must have 72 edges");
    require(base.land_tile_indices().size() == 19, "BASE must have 19 land tiles");

    const std::vector<Edge> required_edges{
        {0, 1},
        {0, 5},
        {0, 20},
        {23, 52},
        {24, 53},
        {52, 53},
    };
    for (Edge edge : required_edges) {
        require(
            std::binary_search(base.land_edges().begin(), base.land_edges().end(), edge),
            "BASE edge identity differs from Python Catanatron");
    }
}

void test_number_and_tournament_contract() {
    const auto mini = CatanMap::build(MapType::mini, 9);
    require(mini.tile_at({1, -1, 0}).number == 5, "MINI spiral starts at A=5");

    const auto base = CatanMap::build(MapType::base, 9);
    require(base.tile_at({2, -2, 0}).number == 5, "BASE spiral starts at A=5");
    require(base.tile_at({2, -1, -1}).number == 2, "BASE spiral second value is 2");

    const auto tournament = CatanMap::build(MapType::tournament, 9);
    require(
        !tournament.tile_at({0, 0, 0}).resource.has_value(),
        "tournament desert must be centered");
    require(
        tournament.tile_at({1, -1, 0}).resource == Resource::ore &&
            tournament.tile_at({1, -1, 0}).number == 6,
        "tournament tile 1 must match pinned Python map");
    require(
        tournament.tile_at({2, -1, -1}).resource == Resource::wood &&
            tournament.tile_at({2, -1, -1}).number == 10,
        "tournament tile 18 must match pinned Python map");
}

void test_settlement_distance_rule() {
    Board board(CatanMap::build(MapType::base, 3));
    require(board.buildable_node_ids(Color::red, true).size() == 54, "initial nodes");
    require_throws(
        [&] { board.build_settlement(Color::red, 3); },
        "non-initial disconnected settlement must fail");

    board.build_settlement(Color::red, 3, true);
    require(
        board.buildable_node_ids(Color::blue, true).size() == 50,
        "settlement must remove itself and three neighbors");
    require_throws(
        [&] { board.build_settlement(Color::blue, 4, true); },
        "adjacent settlement must fail");
    board.build_settlement(Color::blue, 1, true);
}

void test_connected_roads_and_city() {
    Board board(CatanMap::build(MapType::base, 11));
    board.build_settlement(Color::red, 3, true);
    require(board.buildable_edges(Color::red).size() == 3, "node 3 has three edges");
    require_throws(
        [&] { board.build_road(Color::red, {1, 2}); },
        "disconnected road must fail");

    board.build_road(Color::red, {3, 2});
    board.build_road(Color::red, {2, 1});
    board.build_settlement(Color::red, 1);
    board.build_city(Color::red, 1);
    require_throws(
        [&] { board.build_city(Color::red, 0); },
        "city requires an existing settlement");
}

void test_two_player_initial_setup_and_turn() {
    Game game({Color::red, Color::blue}, MapType::base, 17);
    require(game.playable_actions().size() == 54, "setup starts with every node legal");

    while (game.is_initial_build_phase()) {
        const auto action = game.playable_actions().front();
        game.execute(action);
    }

    require(game.current_color() == Color::red, "snake setup returns to first player");
    require(game.current_prompt() == ActionPrompt::play_turn, "setup enters play turn");
    require(game.player(Color::red).settlements.size() == 2, "red has two settlements");
    require(game.player(Color::blue).settlements.size() == 2, "blue has two settlements");
    require(game.player(Color::red).roads.size() == 2, "red has two roads");
    require(game.player(Color::blue).roads.size() == 2, "blue has two roads");
    require(game.player(Color::red).actual_victory_points == 2, "red has two VP");
    require(game.player(Color::blue).actual_victory_points == 2, "blue has two VP");
    require(
        game.playable_actions().size() == 1 &&
            game.playable_actions().front().type == ActionType::roll,
        "first normal decision is roll");

    game.execute(game.playable_actions().front(), Dice{1, 1});
    require(game.player(Color::red).has_rolled, "roll updates player state");
    require(
        std::any_of(
            game.playable_actions().begin(),
            game.playable_actions().end(),
            [](const auto& action) { return action.type == ActionType::end_turn; }),
        "after a non-seven roll end turn is legal");
}

void test_flat_action_space_contract() {
    const FlatActionSpace mini(2, MapType::mini);
    const FlatActionSpace base(2, MapType::base);
    const FlatActionSpace base_four(4, MapType::base);
    require(mini.size() == 187, "MINI 2p action space size");
    require(base.size() == 313, "BASE 2p action space size");
    require(base_four.size() == 351, "BASE 4p action space size");

    require(
        base.index({Color::blue, ActionType::build_city, 0}) == 0,
        "BUILD_CITY node 0 index");
    require(
        base.index({Color::blue, ActionType::build_city, 10}) == 2,
        "integer values use Python lexicographic ordering");
    require(
        base.index({Color::blue, ActionType::build_city, 2}) == 12,
        "BUILD_CITY node 2 index");
    require(
        base.index({Color::blue, ActionType::build_road, Edge{0, 20}}) == 55,
        "BUILD_ROAD edge index");
    require(
        base.index({Color::blue, ActionType::build_settlement, 2}) == 138,
        "BUILD_SETTLEMENT node index");
    require(
        base.index({Color::blue, ActionType::discard_resource, Resource::wood}) == 185,
        "resource values use Python string ordering");
    require(
        base.index({Color::blue, ActionType::end_turn, std::monostate{}}) == 186,
        "END_TURN index");
    require(
        base.index({Color::blue, ActionType::roll, std::monostate{}}) == 312,
        "ROLL index");
}

}  // namespace

int main() {
    try {
        test_topology_contract();
        test_number_and_tournament_contract();
        test_settlement_distance_rule();
        test_connected_roads_and_city();
        test_two_player_initial_setup_and_turn();
        test_flat_action_space_contract();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "cppanatron board tests passed\n";
    return 0;
}
