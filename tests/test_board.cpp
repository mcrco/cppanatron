#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cppanatron/board.hpp"

namespace {

using cppanatron::Board;
using cppanatron::CatanMap;
using cppanatron::Color;
using cppanatron::Edge;
using cppanatron::MapType;
using cppanatron::Resource;

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

}  // namespace

int main() {
    try {
        test_topology_contract();
        test_number_and_tournament_contract();
        test_settlement_distance_rule();
        test_connected_roads_and_city();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "cppanatron board tests passed\n";
    return 0;
}
