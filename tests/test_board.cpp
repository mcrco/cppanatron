#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cppanatron/board.hpp"
#include "cppanatron/action_space.hpp"
#include "cppanatron/game.hpp"
#include "cppanatron/mcts.hpp"

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
using cppanatron::GameRules;
using cppanatron::MCTSSearch;
using cppanatron::Coordinate;
using cppanatron::GameSetup;
using cppanatron::PlayerSetup;
using cppanatron::RobberMove;
using cppanatron::TileKind;

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

cppanatron::Action find_action(const Game& game, ActionType type) {
    const auto it = std::find_if(
        game.playable_actions().begin(),
        game.playable_actions().end(),
        [type](const auto& action) { return action.type == type; });
    if (it == game.playable_actions().end()) {
        throw std::runtime_error("expected action type is not playable");
    }
    return *it;
}

void finish_turn(Game& game, Dice dice = Dice{1, 1}) {
    if (!game.player(game.current_color()).has_rolled) {
        game.execute(find_action(game, ActionType::roll), dice);
    }
    game.execute(find_action(game, ActionType::end_turn));
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
    const std::vector<Color> colors{Color::red, Color::blue};
    const cppanatron::Action robber{
        Color::red,
        ActionType::move_robber,
        cppanatron::RobberMove{{0, 0, 0}, Color::blue}};
    require(
        base.index(robber, colors) == 265,
        "absolute robber victim maps to relative action slot");
    require(
        base.decode(265, Color::red, colors) == robber,
        "relative robber action decodes to absolute victim");
}

void test_mcts_dice_outcomes_are_exact() {
    Game game({Color::red, Color::blue}, MapType::mini, 43);
    while (game.is_initial_build_phase()) {
        game.execute(game.playable_actions().front());
    }
    const auto roll = find_action(game, ActionType::roll);
    const auto outcomes = cppanatron::action_outcomes(game, roll);
    require(outcomes.size() == 11, "roll search edge must enumerate every dice total");
    const double total = std::accumulate(
        outcomes.begin(),
        outcomes.end(),
        0.0,
        [](double sum, const auto& outcome) {
            return sum + outcome.probability;
        });
    require(
        std::abs(total - 1.0) < 1e-12,
        "roll search outcome probabilities must sum to one");
    require(
        std::abs(outcomes.front().probability - 1.0 / 36.0) < 1e-12 &&
            std::abs(outcomes[5].probability - 6.0 / 36.0) < 1e-12,
        "roll search outcome probabilities must match two dice");
}

void test_mcts_card_and_robber_outcomes_are_exact() {
    Game game({Color::red, Color::blue}, MapType::mini, 45);
    while (game.is_initial_build_phase()) {
        game.execute(game.playable_actions().front());
    }
    game.player(Color::red).resources = {0, 0, 1, 1, 1};
    game.execute(find_action(game, ActionType::roll), Dice{1, 1});
    const auto buy = find_action(game, ActionType::buy_development_card);
    const auto card_outcomes = cppanatron::action_outcomes(game, buy);
    require(
        card_outcomes.size() == 5,
        "development-card search edge must enumerate every remaining card type");
    const auto knight = std::find_if(
        card_outcomes.begin(),
        card_outcomes.end(),
        [](const auto& outcome) {
            return outcome.game.player(Color::red)
                       .development_cards[static_cast<std::size_t>(
                           cppanatron::DevelopmentCard::knight)] == 1;
        });
    require(
        knight != card_outcomes.end() &&
            std::abs(knight->probability - 14.0 / 25.0) < 1e-12,
        "development-card outcome probability must match deck composition");

    Game robber_game({Color::red, Color::blue}, MapType::mini, 46);
    while (robber_game.is_initial_build_phase()) {
        robber_game.execute(robber_game.playable_actions().front());
    }
    robber_game.player(Color::red).resources = {};
    robber_game.player(Color::blue).resources = {2, 1, 0, 0, 0};
    robber_game.execute(find_action(robber_game, ActionType::roll), Dice{3, 4});
    const auto robber = std::find_if(
        robber_game.playable_actions().begin(),
        robber_game.playable_actions().end(),
        [](const auto& action) {
            const auto& move = std::get<cppanatron::RobberMove>(action.value);
            return move.victim == Color::blue;
        });
    require(
        robber != robber_game.playable_actions().end(),
        "robber fixture must expose a steal action");
    const auto robber_outcomes =
        cppanatron::action_outcomes(robber_game, *robber);
    require(
        robber_outcomes.size() == 2,
        "robber search edge must enumerate every resource type held by victim");
    std::vector<double> robber_probabilities;
    for (const auto& outcome : robber_outcomes) {
        robber_probabilities.push_back(outcome.probability);
    }
    std::sort(robber_probabilities.begin(), robber_probabilities.end());
    require(
        std::abs(robber_probabilities[0] - 1.0 / 3.0) < 1e-12 &&
            std::abs(robber_probabilities[1] - 2.0 / 3.0) < 1e-12,
        "robber outcome probability must match victim resource counts");
}

void test_mcts_visits_only_legal_actions() {
    Game game({Color::red, Color::blue}, MapType::mini, 47);
    const FlatActionSpace action_space(2, MapType::mini);
    MCTSSearch search(game, action_space, 1.5, 101);
    std::vector<float> logits(action_space.size(), 0.0F);
    search.initialize_root(logits);
    search.add_root_dirichlet_noise(0.3, 0.25);

    constexpr int kSimulations = 32;
    for (int simulation = 0; simulation < kSimulations; ++simulation) {
        const Game* leaf = search.select_leaf();
        if (leaf != nullptr) {
            require(search.has_pending_leaf(), "selected leaf must become pending");
            require(
                search.pending_player_index() == leaf->current_player_index(),
                "pending leaf player index must match its game");
            search.evaluate_leaf(logits, 0.0);
        }
    }

    const auto visits = search.root_visits();
    const auto action_values = search.root_action_values();
    require(
        action_values.size() == visits.size(),
        "root action values must cover the flat action space");
    const auto total = std::accumulate(
        visits.begin(), visits.end(), std::uint32_t{0});
    require(total == kSimulations, "every MCTS simulation must reach one root action");
    for (std::size_t index = 0; index < visits.size(); ++index) {
        if (visits[index] == 0) {
            require(
                std::isnan(action_values[index]),
                "unvisited root actions must not masquerade as zero-value estimates");
            continue;
        }
        require(
            std::abs(action_values[index]) < 1e-12,
            "zero-valued leaf evaluations must produce zero root action values");
        const auto decoded =
            action_space.decode(index, game.current_color(), game.colors());
        require(
            std::find(
                game.playable_actions().begin(),
                game.playable_actions().end(),
                decoded) != game.playable_actions().end(),
            "MCTS root visits must only target legal actions");
    }
}

void test_mcts_chance_values_ignore_unvisited_outcomes() {
    Game game({Color::red, Color::blue}, MapType::mini, 53);
    while (game.is_initial_build_phase()) {
        game.execute(game.playable_actions().front());
    }
    require(
        game.playable_actions().size() == 1 &&
            game.playable_actions().front().type == ActionType::roll,
        "chance-value fixture must start with ROLL as the only legal action");
    const FlatActionSpace action_space(2, MapType::mini);
    const std::vector<float> logits(action_space.size(), 0.0F);
    MCTSSearch search(game, action_space, 1.5, 109);
    search.initialize_root(logits);

    constexpr double kLeafValue = 0.5;
    require(search.select_leaf() != nullptr, "first simulation must reach a dice outcome");
    search.evaluate_leaf(logits, kLeafValue);

    const auto roll_index =
        action_space.index(game.playable_actions().front(), game.colors());
    require(
        std::abs(std::abs(search.root_action_values()[roll_index]) - kLeafValue) < 1e-12,
        "one visited dice outcome must define the roll Q instead of being "
        "diluted by the ten unvisited outcomes");
}

void test_mcts_reuses_only_deterministic_subtrees() {
    Game game({Color::red, Color::blue}, MapType::mini, 49);
    const FlatActionSpace action_space(2, MapType::mini);
    std::vector<float> logits(action_space.size(), 0.0F);
    MCTSSearch search(game, action_space, 1.5, 103, true);
    search.initialize_root(logits);
    for (int simulation = 0; simulation < 8; ++simulation) {
        const Game* leaf = search.select_leaf();
        if (leaf != nullptr) {
            search.evaluate_leaf(logits, 0.0);
        }
    }

    const auto deterministic = game.playable_actions().front();
    const auto deterministic_index = action_space.index(deterministic, game.colors());
    require(search.advance(deterministic_index), "setup action must reuse its subtree");
    game.execute(deterministic);
    require(
        search.root_game().search_equivalent(game),
        "reused root must exactly match the played deterministic state");
    require(search.metrics().tree_reused, "reuse must be visible in search metrics");

    Game roll_game({Color::red, Color::blue}, MapType::mini, 51);
    while (roll_game.is_initial_build_phase()) {
        roll_game.execute(roll_game.playable_actions().front());
    }
    MCTSSearch roll_search(roll_game, action_space, 1.5, 107, true);
    roll_search.initialize_root(logits);
    const auto roll = find_action(roll_game, ActionType::roll);
    require(
        !roll_search.advance(action_space.index(roll, roll_game.colors())),
        "dice action must invalidate rather than reuse a sampled chance child");
}

void test_mcts_canonical_discard_prunes_infeasible_suffixes() {
    Game game({Color::red, Color::blue}, MapType::mini, 53);
    while (game.is_initial_build_phase()) {
        game.execute(game.playable_actions().front());
    }
    const Color roller = game.current_color();
    game.player(roller).resources = {6, 0, 0, 0, 2};
    game.execute(find_action(game, ActionType::roll), Dice{3, 4});

    const FlatActionSpace action_space(2, MapType::mini);
    const auto ore_discard = std::find_if(
        game.playable_actions().begin(),
        game.playable_actions().end(),
        [](const auto& action) {
            return action.type == ActionType::discard_resource &&
                   std::get<Resource>(action.value) == Resource::ore;
        });
    require(ore_discard != game.playable_actions().end(), "fixture must permit discarding ore");
    const auto ore_index = action_space.index(*ore_discard, game.colors());
    std::vector<float> logits(action_space.size(), -100.0F);
    logits[ore_index] = 100.0F;
    MCTSSearch search(game, action_space, 1.5, 109, true);
    search.initialize_root(logits);
    for (int simulation = 0; simulation < 8; ++simulation) {
        const Game* leaf = search.select_leaf();
        if (leaf != nullptr) {
            search.evaluate_leaf(logits, 0.0);
        }
    }

    require(
        search.root_visits()[ore_index] == 0,
        "discard ordering must reject a suffix that cannot finish the discard");
    require(search.metrics().pruned_actions > 0, "discard pruning must be observable");
}

void test_seven_discard_and_robber_transition() {
    Game game({Color::red, Color::blue}, MapType::base, 31);
    while (game.is_initial_build_phase()) {
        game.execute(game.playable_actions().front());
    }

    game.player(Color::red).resources = {8, 0, 0, 0, 0};
    game.execute(game.playable_actions().front(), Dice{3, 4});
    require(game.current_prompt() == ActionPrompt::discard, "eight cards require four discards");
    for (int i = 0; i < 4; ++i) {
        require(
            game.playable_actions().size() == 1 &&
                game.playable_actions().front().type == ActionType::discard_resource,
            "only held resources can be discarded");
        game.execute(game.playable_actions().front());
    }
    require(
        game.current_prompt() == ActionPrompt::move_robber,
        "last discard advances to robber movement");
    require(!game.playable_actions().empty(), "robber has legal destination actions");
    game.execute(game.playable_actions().front());
    require(
        game.current_prompt() == ActionPrompt::play_turn,
        "robber movement returns to play turn");
    require(game.player(Color::red).has_rolled, "rolling player remains marked rolled");
}

// ---- friendly robber (OpenHex's rule) ----------------------------------------

/** The tiles the robber may move to now. */
std::set<Coordinate> robber_tiles(const Game& game) {
    std::set<Coordinate> tiles;
    for (const auto& action : game.playable_actions()) {
        require(action.type == ActionType::move_robber, "only robber moves at the robber prompt");
        tiles.insert(std::get<RobberMove>(action.value).coordinate);
    }
    return tiles;
}

/** Every land tile but the robber's. */
std::set<Coordinate> robber_candidates(const Game& game) {
    std::set<Coordinate> tiles;
    for (const auto& [coordinate, index] : game.board().map().coordinate_to_tile()) {
        if (game.board().map().tiles().at(static_cast<std::size_t>(index)).kind == TileKind::land &&
            coordinate != game.board().robber_coordinate()) {
            tiles.insert(coordinate);
        }
    }
    return tiles;
}

bool has_building_of(const Game& game, Coordinate coordinate, Color color) {
    for (int node : game.board().map().tile_at(coordinate).nodes) {
        const auto building = game.board().buildings().find(node);
        if (building != game.board().buildings().end() && building->second.color == color) {
            return true;
        }
    }
    return false;
}

/** Every seat's public victory points now. */
std::vector<int> public_points(const Game& game) {
    std::vector<int> points;
    for (const auto& state : game.players()) {
        points.push_back(state.victory_points);
    }
    return points;
}

/**
 * OpenHex's friendly robber from public state alone (open-hex engine.py
 * _validate_robber_placement, _friendly_robber_shields): a tile is shielded
 * when any seat with at most 2 public points has a building on it, the
 * mover included, and the robber may go to the other tiles only. With none
 * (empty here) no robber move is owed (_robber_has_legal_hex).
 */
std::set<Coordinate> openhex_friendly_tiles(const Game& game) {
    const std::vector<int> points = public_points(game);
    std::set<Coordinate> allowed;
    for (const Coordinate& coordinate : robber_candidates(game)) {
        bool shielded = false;
        for (std::size_t seat = 0; seat < game.colors().size(); ++seat) {
            shielded = shielded ||
                       (points[seat] <= 2 && has_building_of(game, coordinate, game.colors()[seat]));
        }
        if (!shielded) {
            allowed.insert(coordinate);
        }
    }
    return allowed;
}

GameRules friendly_rules(int players) {
    GameRules rules;
    rules.friendly_robber = true;
    if (players == 2) {  // OpenHex's 1v1
        rules.victory_points_to_win = 15;
        rules.discard_limit = 9;
    }
    return rules;
}

/**
 * `players` seats placed by a seeded random initial phase, then Red at the
 * robber prompt after a seven, with `edit` applied to the seats.
 */
Game at_friendly_robber(int players, std::uint64_t seed,
                        const std::function<void(std::vector<PlayerSetup>&)>& edit) {
    const std::vector<Color> colors(cppanatron::kColors.begin(), cppanatron::kColors.begin() + players);
    Game placed(colors, MapType::base, seed, friendly_rules(players));
    std::mt19937_64 rng(seed);
    while (placed.is_initial_build_phase()) {
        const auto& actions = placed.playable_actions();
        placed.execute(actions[std::uniform_int_distribution<std::size_t>(0, actions.size() - 1)(rng)]);
    }
    std::vector<PlayerSetup> seats;
    for (const auto& state : placed.players()) {
        PlayerSetup seat;
        seat.settlements = state.settlements;
        seat.roads = state.roads;
        seat.resources = {1, 1, 0, 0, 0};
        seats.push_back(seat);
    }
    edit(seats);
    GameSetup setup;
    setup.players = seats;
    setup.has_rolled = true;
    setup.prompt = ActionPrompt::move_robber;
    Game game(colors, MapType::base, seed, friendly_rules(players));
    game.apply_setup(setup);
    return game;
}

std::set<Coordinate> tiles_without(const Game& game, const std::set<Coordinate>& tiles, Color color) {
    std::set<Coordinate> result;
    for (const Coordinate& coordinate : tiles) {
        if (!has_building_of(game, coordinate, color)) {
            result.insert(coordinate);
        }
    }
    return result;
}

/** Turns a seat's newest settlement into a city: 3 public points from two buildings. */
void upgrade_newest(PlayerSetup& seat) {
    seat.cities = {seat.settlements.back()};
    seat.settlements.pop_back();
}

/**
 * Blue shows 2 points and holds a victory point card (3 actual points): its
 * tiles are shielded, since the rule reads public points. Red, at 3 points,
 * shields nothing.
 */
void test_friendly_robber_shields_low_public_points() {
    const Game game = at_friendly_robber(2, 37, [](std::vector<PlayerSetup>& seats) {
        upgrade_newest(seats[0]);
        seats[1].development_cards = {0, 0, 0, 0, 1};
    });
    const auto& blue = game.player(Color::blue);
    require(game.player(Color::red).victory_points == 3, "Red shows 3 points");
    require(blue.victory_points == 2 && blue.actual_victory_points == 3, "2 public points, 3 actual");
    const auto candidates = robber_candidates(game);
    const auto legal = robber_tiles(game);
    require(legal == tiles_without(game, candidates, Color::blue), "every tile but Blue's is open");
    require(legal.size() < candidates.size(), "Blue's tiles are candidates");
    require(legal == openhex_friendly_tiles(game), "OpenHex's rule");
    for (const auto& action : game.playable_actions()) {
        require(std::get<RobberMove>(action.value).victim != Color::blue, "Blue cannot be robbed");
    }
}

/**
 * The mover's own buildings shield like anyone's: at 2 points Red may not
 * move the robber to a tile with only its own settlement on it; at 3 points
 * (a city) it may.
 */
void test_friendly_robber_shields_the_movers_own_tiles() {
    const auto only_reds = [](const Game& game) {
        std::set<Coordinate> tiles;
        for (const Coordinate& coordinate : robber_candidates(game)) {
            if (has_building_of(game, coordinate, Color::red) && !has_building_of(game, coordinate, Color::blue)) {
                tiles.insert(coordinate);
            }
        }
        return tiles;
    };
    const auto contains_any = [](const std::set<Coordinate>& tiles, const std::set<Coordinate>& of) {
        return std::any_of(of.begin(), of.end(), [&](const Coordinate& c) { return tiles.contains(c); });
    };
    // Blue at 3 points shields nothing, so only Red's points matter.
    const Game low = at_friendly_robber(2, 37, [](std::vector<PlayerSetup>& seats) { upgrade_newest(seats[1]); });
    require(low.player(Color::red).victory_points == 2 && low.player(Color::blue).victory_points == 3,
            "the mover at 2 points, Blue at 3");
    const auto own = only_reds(low);
    require(!own.empty(), "Red has tiles of its own");
    require(robber_tiles(low) == tiles_without(low, robber_candidates(low), Color::red),
            "at 2 points the mover's own tiles are shielded");
    require(!contains_any(robber_tiles(low), own), "also those with no other building");
    require(robber_tiles(low) == openhex_friendly_tiles(low), "OpenHex's rule");

    const Game high = at_friendly_robber(2, 37, [](std::vector<PlayerSetup>& seats) {
        upgrade_newest(seats[0]);
        upgrade_newest(seats[1]);
    });
    require(high.player(Color::red).victory_points == 3, "the mover at 3 points");
    require(only_reds(high) == own, "the same tiles");
    require(robber_tiles(high) == robber_candidates(high), "at 3 points they are open");
    require(robber_tiles(high) == openhex_friendly_tiles(high), "OpenHex's rule");

    // Both at 2 points: only the tiles with no building at all are open.
    const Game both = at_friendly_robber(2, 37, [](std::vector<PlayerSetup>&) {});
    require(robber_tiles(both) ==
                tiles_without(both, tiles_without(both, robber_candidates(both), Color::red), Color::blue),
            "everyone's tiles are shielded");
    for (const auto& action : both.playable_actions()) {
        require(!std::get<RobberMove>(action.value).victim.has_value(), "and nobody can be robbed");
    }
}

/** At 3 public points (a city, or an award) a seat is robbable. */
void test_friendly_robber_spares_no_one_at_three_public_points() {
    const Game city = at_friendly_robber(2, 37, [](std::vector<PlayerSetup>& seats) {
        upgrade_newest(seats[0]);
        upgrade_newest(seats[1]);
    });
    require(city.player(Color::blue).victory_points == 3, "a city and a settlement show 3 points");
    require(robber_tiles(city) == robber_candidates(city), "every tile is open at 3 public points");
    bool robs_blue = false;
    for (const auto& action : city.playable_actions()) {
        robs_blue = robs_blue || std::get<RobberMove>(action.value).victim == Color::blue;
    }
    require(robs_blue, "Blue can be robbed");

    const Game army = at_friendly_robber(2, 37, [](std::vector<PlayerSetup>& seats) {
        upgrade_newest(seats[0]);
        seats[1].played_development_cards = {3, 0, 0, 0, 0};
    });
    require(army.player(Color::blue).has_army && army.player(Color::blue).victory_points == 4,
            "Largest Army is public");
    require(robber_tiles(army) == robber_candidates(army), "an award counts toward public points");
}

/** The legal set does not depend on anyone's hidden victory point cards. */
void test_friendly_robber_ignores_hidden_victory_point_cards() {
    const std::vector<std::array<int, 3>> deals{{0, 0, 0}, {1, 0, 0}, {0, 2, 1}, {2, 1, 2}, {1, 1, 1}};
    std::vector<cppanatron::Action> first;
    for (const auto& deal : deals) {
        const Game game = at_friendly_robber(4, 52, [&](std::vector<PlayerSetup>& seats) {
            for (std::size_t seat = 1; seat < 4; ++seat) {
                seats[seat].development_cards = {1, 0, 0, 0, deal[seat - 1]};
            }
        });
        require(robber_tiles(game) == openhex_friendly_tiles(game), "OpenHex's rule");
        require(robber_tiles(game) != robber_candidates(game), "the seats at 2 points shield tiles");
        if (first.empty()) {
            first = game.playable_actions();
        }
        require(game.playable_actions() == first, "the same robber moves whatever the hidden cards");
    }
}

/**
 * Three land tiles: the desert (with the robber), ore and wheat, so two
 * candidate tiles. A settlement on the node where all three meet (`corner`)
 * touches both candidates; `ore_only` and `desert_only` touch one tile.
 */
struct ThreeTiles {
    std::shared_ptr<const CatanMap> map;
    int corner{-1};
    int ore_only{-1};
    int desert_only{-1};
    static constexpr Coordinate kDesert{0, 0, 0};
    static constexpr Coordinate kOre{1, -1, 0};
    static constexpr Coordinate kWheat{0, -1, 1};

    ThreeTiles() {
        cppanatron::MapSpec spec;
        const auto land = [&](Coordinate coordinate, std::optional<Resource> resource, std::optional<int> number) {
            spec.tiles.push_back({.coordinate = coordinate, .kind = TileKind::land, .resource = resource,
                                  .number = number, .direction = std::nullopt, .port_resource = std::nullopt});
        };
        land(kDesert, std::nullopt, std::nullopt);
        land(kOre, Resource::ore, 8);
        land(kWheat, Resource::wheat, 6);
        map = std::make_shared<const CatanMap>(CatanMap::from_spec(spec));
        const auto nodes_of = [&](Coordinate coordinate) {
            const auto& nodes = map->tile_at(coordinate).nodes;
            return std::set<int>(nodes.begin(), nodes.end());
        };
        const auto desert = nodes_of(kDesert);
        const auto ore = nodes_of(kOre);
        const auto wheat = nodes_of(kWheat);
        for (int node : ore) {
            if (desert.contains(node) && wheat.contains(node)) {
                corner = node;
            }
        }
        // The single-tile nodes farthest from the corner: not next to it.
        const auto apart = [&](int node) {
            const auto& near = map->neighbors(corner);
            return node != corner && std::find(near.begin(), near.end(), node) == near.end();
        };
        for (int node : ore) {
            if (!wheat.contains(node) && !desert.contains(node) && apart(node)) {
                ore_only = node;
            }
        }
        for (int node : desert) {
            if (!wheat.contains(node) && !ore.contains(node) && apart(node)) {
                const auto& near = map->neighbors(node);
                if (std::find(near.begin(), near.end(), ore_only) == near.end()) {
                    desert_only = node;
                }
            }
        }
        require(corner >= 0 && ore_only >= 0 && desert_only >= 0, "the three tiles meet at a node");
    }

    /** Red's and Blue's one settlement each, Red to play (before the roll unless `setup` says otherwise). */
    [[nodiscard]] Game game(int red_node, int blue_node, const std::function<void(GameSetup&)>& edit = {}) const {
        GameSetup setup;
        setup.players.resize(2);
        setup.players[0].settlements = {red_node};
        setup.players[1].settlements = {blue_node};
        if (edit) {
            edit(setup);
        }
        Game result({Color::red, Color::blue}, map, 5, friendly_rules(2));
        result.apply_setup(setup);
        return result;
    }
};

bool offers(const Game& game, ActionType type) {
    return std::any_of(game.playable_actions().begin(), game.playable_actions().end(),
                       [type](const auto& action) { return action.type == type; });
}

/**
 * No fallback: with every tile but the robber's shielded, a 7 owes no robber
 * move. The roller plays on with the robber where it was and nobody robbed,
 * also after the discards, which still happen. One shielded tile of two
 * leaves the other as the only move.
 */
void test_friendly_robber_stays_after_a_seven_when_every_tile_is_shielded() {
    const ThreeTiles tiles;
    const auto hands = [](const Game& game) {
        return std::pair{game.player(Color::red).resources, game.player(Color::blue).resources};
    };
    const auto require_plays_on = [&](const Game& game, const std::string& what) {
        require(game.current_prompt() == ActionPrompt::play_turn && game.current_color() == Color::red &&
                    game.current_turn_index() == 0,
                what + ": the roller's play turn");
        require(game.player(Color::red).has_rolled && !game.is_moving_robber(), what + ": after the roll");
        require(game.board().robber_coordinate() == ThreeTiles::kDesert, what + ": the robber has not moved");
        require(offers(game, ActionType::end_turn) && !offers(game, ActionType::move_robber) &&
                    !offers(game, ActionType::roll),
                what + ": the turn goes on");
    };

    // Blue's settlement at the corner (1 point) shields both candidates.
    Game shielded = tiles.game(tiles.desert_only, tiles.corner, [](GameSetup& setup) {
        setup.players[1].resources = {0, 0, 0, 3, 0};
    });
    require(robber_candidates(shielded) == std::set<Coordinate>{ThreeTiles::kOre, ThreeTiles::kWheat},
            "two candidates");
    require(openhex_friendly_tiles(shielded).empty(), "both shielded");
    const auto before = hands(shielded);
    shielded.execute(find_action(shielded, ActionType::roll), Dice{3, 4});
    require_plays_on(shielded, "a 7");
    require(hands(shielded) == before, "nobody is robbed");
    shielded.execute(find_action(shielded, ActionType::end_turn));
    require(shielded.current_color() == Color::blue && offers(shielded, ActionType::roll), "and the turn passes");

    // The mover's own settlement shields as well: Red at the corner, Blue off both candidates.
    Game own = tiles.game(tiles.corner, tiles.desert_only, [](GameSetup& setup) {
        setup.players[1].resources = {0, 0, 0, 3, 0};
    });
    require(openhex_friendly_tiles(own).empty(), "both shielded by the mover");
    own.execute(find_action(own, ActionType::roll), Dice{1, 6});
    require_plays_on(own, "a 7 with the mover's own shield");

    // Discards first: both seats over the limit of 9 discard half, then the roller plays on.
    Game discards = tiles.game(tiles.desert_only, tiles.corner, [](GameSetup& setup) {
        setup.players[0].resources = {4, 3, 2, 1, 0};
        setup.players[1].resources = {0, 0, 0, 11, 0};
    });
    discards.execute(find_action(discards, ActionType::roll), Dice{2, 5});
    require(discards.current_prompt() == ActionPrompt::discard && discards.current_color() == Color::red,
            "the discards still happen, in seat order");
    int discarded = 0;
    while (discards.current_prompt() == ActionPrompt::discard) {
        require(discarded < 5 ? discards.current_color() == Color::red : discards.current_color() == Color::blue,
                "Red discards 5, then Blue 5");
        discards.execute(discards.playable_actions().front());
        ++discarded;
    }
    require(discarded == 10, "10 cards and 11 cards: 5 each");
    require_plays_on(discards, "a 7 after discards");
    const auto [red_hand, blue_hand] = hands(discards);
    require(std::accumulate(red_hand.begin(), red_hand.end(), 0) == 5 &&
                std::accumulate(blue_hand.begin(), blue_hand.end(), 0) == 6,
            "the discards are the only cards that moved");

    // One of the two shielded: after a 7 (and after discards) the other is the only move.
    for (bool with_discards : {false, true}) {
        Game partial = tiles.game(tiles.desert_only, tiles.ore_only, [&](GameSetup& setup) {
            setup.players[1].resources = {0, 0, 0, with_discards ? 10 : 3, 0};
        });
        partial.execute(find_action(partial, ActionType::roll), Dice{3, 4});
        while (partial.current_prompt() == ActionPrompt::discard) {
            partial.execute(partial.playable_actions().front());
        }
        require(partial.current_prompt() == ActionPrompt::move_robber && partial.current_color() == Color::red,
                "an open tile: the robber prompt");
        require(robber_tiles(partial) == std::set<Coordinate>{ThreeTiles::kWheat}, "the open tile only");
        require(robber_tiles(partial) == openhex_friendly_tiles(partial), "OpenHex's rule");
    }

    // A declared robber prompt with nowhere to go is not a position of the game.
    require_throws(
        [&] {
            (void)tiles.game(tiles.desert_only, tiles.corner, [](GameSetup& setup) {
                setup.has_rolled = true;
                setup.prompt = ActionPrompt::move_robber;
            });
        },
        "a robber prompt with every tile shielded must be refused");

    // The normal robber is not concerned: the same position owes its robber move.
    GameSetup setup;
    setup.players.resize(2);
    setup.players[0].settlements = {tiles.desert_only};
    setup.players[1].settlements = {tiles.corner};
    GameRules normal_rules = friendly_rules(2);
    normal_rules.friendly_robber = false;
    Game normal({Color::red, Color::blue}, tiles.map, 5, normal_rules);
    normal.apply_setup(setup);
    normal.execute(find_action(normal, ActionType::roll), Dice{3, 4});
    require(normal.current_prompt() == ActionPrompt::move_robber &&
                robber_tiles(normal) == robber_candidates(normal),
            "the normal robber moves anywhere");
}

/**
 * A knight with every tile shielded is still played: it leaves the hand,
 * counts for Largest Army (awarded if due) and uses the turn's development
 * card, but no robber prompt follows; before the roll, the roll is next.
 */
void test_friendly_robber_knight_without_an_open_tile() {
    const ThreeTiles tiles;
    constexpr auto knight = static_cast<std::size_t>(cppanatron::DevelopmentCard::knight);
    for (int played : {0, 2}) {
        for (bool rolled : {false, true}) {
            Game game = tiles.game(tiles.desert_only, tiles.corner, [&](GameSetup& setup) {
                setup.players[0].development_cards = {2, 0, 0, 0, 0};
                setup.players[0].played_development_cards = {played, 0, 0, 0, 0};
                setup.players[1].resources = {0, 0, 0, 3, 0};
                setup.has_rolled = rolled;
            });
            require(openhex_friendly_tiles(game).empty(), "Blue's corner shields both tiles");
            const auto blue_hand = game.player(Color::blue).resources;
            game.execute(find_action(game, ActionType::play_knight_card));
            const auto& red = game.player(Color::red);
            require(game.current_prompt() == ActionPrompt::play_turn && game.current_color() == Color::red &&
                        !game.is_moving_robber() && !offers(game, ActionType::move_robber),
                    "no robber prompt");
            require(game.board().robber_coordinate() == ThreeTiles::kDesert &&
                        game.player(Color::blue).resources == blue_hand,
                    "the robber stays and nobody is robbed");
            require(red.development_cards[knight] == 1 && red.played_development_cards[knight] == played + 1,
                    "the knight left the hand and counts");
            require(red.has_played_development_card_in_turn && !offers(game, ActionType::play_knight_card),
                    "it was the turn's development card");
            require(red.has_army == (played == 2) && red.victory_points == (played == 2 ? 3 : 1),
                    "Largest Army is awarded with the third knight");
            require(!game.largest_army_taken_from().has_value(), "no record is kept");
            require(offers(game, ActionType::roll) == !rolled && offers(game, ActionType::end_turn) == rolled,
                    "the turn goes on where it was");
        }
    }
}

/**
 * A knight's robber is judged on the points after the knight's Largest Army
 * award (OpenHex's _with_knight_counted), both ways: the award lifts the
 * mover past 2 points and opens its own tiles, and a holder that loses the
 * award to the knight is shielded on the points it is left with.
 */
void test_friendly_robber_knight_is_judged_after_the_award() {
    const ThreeTiles tiles;
    // Red's corner settlement (1 point) shields both tiles until its third knight brings 2 more.
    Game lifted = tiles.game(tiles.corner, tiles.desert_only, [](GameSetup& setup) {
        setup.players[0].development_cards = {1, 0, 0, 0, 0};
        setup.players[0].played_development_cards = {2, 0, 0, 0, 0};
        setup.has_rolled = true;
    });
    require(lifted.player(Color::red).victory_points == 1 && openhex_friendly_tiles(lifted).empty(),
            "before the knight every tile is shielded");
    lifted.execute(find_action(lifted, ActionType::play_knight_card));
    require(lifted.player(Color::red).has_army && lifted.player(Color::red).victory_points == 3,
            "the knight brings Largest Army: 3 points");
    require(lifted.current_prompt() == ActionPrompt::move_robber &&
                robber_tiles(lifted) == robber_candidates(lifted),
            "judged after the award, both tiles are open");
    require(robber_tiles(lifted) == openhex_friendly_tiles(lifted), "OpenHex's rule");
    // One knight short of the award, the same knight moves no robber.
    Game short_of = tiles.game(tiles.corner, tiles.desert_only, [](GameSetup& setup) {
        setup.players[0].development_cards = {1, 0, 0, 0, 0};
        setup.players[0].played_development_cards = {1, 0, 0, 0, 0};
        setup.has_rolled = true;
    });
    short_of.execute(find_action(short_of, ActionType::play_knight_card));
    require(short_of.current_prompt() == ActionPrompt::play_turn && !short_of.player(Color::red).has_army,
            "without the award the tiles stay shielded");

    // Blue's corner settlement and Largest Army: 3 points, shielding nothing, until Red's
    // fourth knight takes the award and leaves it 1 point. Then no tile is open.
    Game taken = tiles.game(tiles.desert_only, tiles.corner, [](GameSetup& setup) {
        setup.players[0].development_cards = {1, 0, 0, 0, 0};
        setup.players[0].played_development_cards = {3, 0, 0, 0, 0};
        setup.players[1].played_development_cards = {3, 0, 0, 0, 0};
        setup.players[1].resources = {0, 0, 0, 3, 0};
        setup.largest_army = 1;
        setup.has_rolled = true;
    });
    require(taken.player(Color::blue).has_army && taken.player(Color::blue).victory_points == 3 &&
                taken.player(Color::red).victory_points == 1,
            "Blue holds Largest Army at 3 points");
    require(openhex_friendly_tiles(taken) == robber_candidates(taken), "before the knight both tiles are open");
    taken.execute(find_action(taken, ActionType::play_knight_card));
    require(taken.player(Color::red).has_army && taken.player(Color::blue).victory_points == 1,
            "the knight takes Largest Army");
    require(taken.current_prompt() == ActionPrompt::play_turn && !taken.largest_army_taken_from().has_value(),
            "judged after the award, Blue shields both tiles: no robber prompt");
    require(taken.board().robber_coordinate() == ThreeTiles::kDesert &&
                taken.player(Color::blue).resources == (std::array<int, 5>{0, 0, 0, 3, 0}),
            "and Blue is not robbed");
}

/**
 * On the base map: Red plays its fourth knight and takes Largest Army from
 * Blue, which shows two settlements and the award (4 public points before
 * the knight, 2 after). The robber is judged after the award, so Blue's
 * tiles are shielded and Blue cannot be robbed. largest_army_taken_from,
 * which named Blue under the earlier rule (judged before the award), is
 * never set by play; a declared position may still carry it, checked as
 * before and then ignored.
 */
void test_friendly_robber_after_a_knight_takes_largest_army() {
    const std::vector<Color> colors{Color::red, Color::blue};
    const auto rules_with = [](bool friendly) {
        GameRules rules = friendly_rules(2);
        rules.friendly_robber = friendly;
        return rules;
    };
    Game placed(colors, MapType::base, 37, rules_with(true));
    std::mt19937_64 rng(37);
    while (placed.is_initial_build_phase()) {
        const auto& actions = placed.playable_actions();
        placed.execute(actions[std::uniform_int_distribution<std::size_t>(0, actions.size() - 1)(rng)]);
    }
    std::vector<PlayerSetup> seats;
    for (const auto& state : placed.players()) {
        PlayerSetup seat;
        seat.settlements = state.settlements;
        seat.roads = state.roads;
        seat.resources = {1, 1, 0, 0, 0};
        seats.push_back(seat);
    }
    // Red to play its fourth knight; Blue holds Largest Army at 3 knights (or has none).
    const auto before_knight = [&](bool friendly, int blue_knights) {
        std::vector<PlayerSetup> players = seats;
        players[0].played_development_cards = {3, 0, 0, 0, 0};
        players[0].development_cards = {1, 0, 0, 0, 0};
        players[1].played_development_cards = {blue_knights, 0, 0, 0, 0};
        GameSetup setup;
        setup.players = players;
        setup.has_rolled = true;
        if (blue_knights == 3) {
            setup.largest_army = 1;
        }
        Game game(colors, MapType::base, 37, rules_with(friendly));
        game.apply_setup(setup);
        return game;
    };
    const auto robs = [](const Game& game, Color victim) {
        for (const auto& action : game.playable_actions()) {
            if (std::get<RobberMove>(action.value).victim == victim) {
                return std::optional<cppanatron::Action>(action);
            }
        }
        return std::optional<cppanatron::Action>();
    };

    Game game = before_knight(true, 3);
    require(game.player(Color::blue).has_army && game.player(Color::blue).victory_points == 4,
            "Blue holds Largest Army at 4 public points");
    game.execute(find_action(game, ActionType::play_knight_card));
    require(game.current_prompt() == ActionPrompt::move_robber, "the knight's robber");
    require(game.player(Color::red).has_army && game.player(Color::red).victory_points == 4 &&
                game.player(Color::blue).victory_points == 2,
            "the knight takes Largest Army: Blue shows 2 points, Red 4");
    require(!game.largest_army_taken_from().has_value(), "play keeps no record of it");
    require(robber_tiles(game) == openhex_friendly_tiles(game), "OpenHex's rule, on the points after the award");
    require(robber_tiles(game) == tiles_without(game, robber_candidates(game), Color::blue),
            "Blue at 2 points is shielded, Red at 4 shields nothing");
    require(!robs(game, Color::blue).has_value(), "and Blue cannot be robbed");

    // A declared copy of this prompt may carry the old record: it is checked as it always
    // was (refused where no knight took the award, or under the normal robber) and ignored.
    std::vector<PlayerSetup> after = seats;
    after[0].played_development_cards = {4, 0, 0, 0, 0};
    after[1].played_development_cards = {3, 0, 0, 0, 0};
    GameSetup declared;
    declared.players = after;
    declared.has_rolled = true;
    declared.has_played_development_card = true;
    declared.prompt = ActionPrompt::move_robber;
    Game plain(colors, MapType::base, 37, rules_with(true));
    plain.apply_setup(declared);
    require(plain.playable_actions() == game.playable_actions(),
            "a declared position reproduces the knight's robber moves");
    declared.largest_army_taken_from = 1;
    Game copy(colors, MapType::base, 37, rules_with(true));
    copy.apply_setup(declared);
    require(!copy.largest_army_taken_from().has_value() && copy.playable_actions() == game.playable_actions() &&
                copy.search_equivalent(plain),
            "with the old record it loads to the same position");
    for (const auto& [taken, friendly] : std::vector<std::pair<int, bool>>{{0, true}, {1, false}}) {
        declared.largest_army_taken_from = taken;
        Game refused(colors, MapType::base, 37, rules_with(friendly));
        require_throws([&] { refused.apply_setup(declared); }, "largest_army_taken_from is validated");
    }

    game.execute(game.playable_actions().front());
    require(!game.largest_army_taken_from().has_value() && game.current_prompt() == ActionPrompt::play_turn,
            "the robber moves and the turn goes on");

    Game normal = before_knight(false, 3);
    normal.execute(find_action(normal, ActionType::play_knight_card));
    require(normal.player(Color::red).has_army && !normal.largest_army_taken_from().has_value(),
            "the normal robber keeps no record either");
    require(robber_tiles(normal) == robber_candidates(normal), "and shields nothing");
    require(robs(normal, Color::blue).has_value(), "Blue can be robbed");

    Game kept = before_knight(true, 0);
    require(kept.player(Color::red).has_army, "Red already holds Largest Army");
    kept.execute(find_action(kept, ActionType::play_knight_card));
    require(robber_tiles(kept) == tiles_without(kept, robber_candidates(kept), Color::blue),
            "Blue at 2 points is shielded");
    require(!robs(kept, Color::blue).has_value(), "and cannot be robbed");
}

/**
 * Random 2-4 player games under the friendly robber, on the base map and on
 * a three-tile map (where shields often cover every tile). The robber prompt
 * follows a 7 (after its discards) or a knight exactly when OpenHex's rule
 * leaves a tile open on the points of that moment (a knight's after its
 * award), its legal tiles are OpenHex's, and re-dealing a seat's hidden
 * development cards (as many victory point cards as the unseen cards allow,
 * or none) leaves the legal moves as they were.
 */
void test_random_friendly_games_follow_openhex() {
    int prompts = 0;
    int shielding = 0;
    int redeals = 0;
    int stays = 0;
    int knight_stays = 0;
    const ThreeTiles small;
    for (int players = 2; players <= 4; ++players) {
        for (std::uint64_t seed = 1; seed <= 12; ++seed) {
            const bool tiny = seed > 6;
            if (tiny && players > 2) {
                continue;  // (three tiles seat two)
            }
            const std::vector<Color> colors(cppanatron::kColors.begin(), cppanatron::kColors.begin() + players);
            Game game = tiny ? Game(colors, small.map, seed, friendly_rules(players))
                             : Game(colors, MapType::base, seed, friendly_rules(players),
                                    cppanatron::NumberPlacement::random);
            std::mt19937_64 rng(seed + 100 * static_cast<std::uint64_t>(players));
            for (int step = 0; step < 4000 && !game.winning_color().has_value(); ++step) {
                const auto& actions = game.playable_actions();
                if (actions.empty()) {
                    require(game.is_initial_build_phase(), "only a crowded initial placement has no action");
                    break;
                }
                if (game.current_prompt() == ActionPrompt::move_robber) {
                    ++prompts;
                    require(!actions.empty() && robber_tiles(game) == openhex_friendly_tiles(game),
                            "OpenHex's rule in play");
                    shielding += robber_tiles(game) != robber_candidates(game);
                    for (Color color : colors) {
                        const auto& hand = game.player(color).development_cards;
                        const int size = std::accumulate(hand.begin(), hand.end(), 0);
                        if (color == game.current_color() || size == 0) {
                            continue;
                        }
                        std::array<int, 5> unseen = hand;
                        for (auto card : game.development_deck()) {
                            ++unseen[static_cast<std::size_t>(card)];
                        }
                        for (bool most : {true, false}) {
                            std::array<int, 5> cards{};
                            int left = size;
                            for (std::size_t kind : most ? std::vector<std::size_t>{4, 0, 1, 2, 3}
                                                         : std::vector<std::size_t>{0, 1, 2, 3, 4}) {
                                cards[kind] = std::min(left, unseen[kind]);
                                left -= cards[kind];
                            }
                            Game redealt = game;
                            redealt.determinize_development_cards(color, cards, seed);
                            require(redealt.playable_actions() == actions, "hidden cards do not move the legal set");
                            ++redeals;
                        }
                    }
                }
                const cppanatron::Action chosen =
                    actions[std::uniform_int_distribution<std::size_t>(0, actions.size() - 1)(rng)];
                const Coordinate robber = game.board().robber_coordinate();
                const int turn = game.current_turn_index();
                bool owed = false;  // a robber move is due unless every tile is shielded
                if (chosen.type == ActionType::roll) {
                    const Dice dice{std::uniform_int_distribution<int>(1, 6)(rng),
                                    std::uniform_int_distribution<int>(1, 6)(rng)};
                    game.execute(chosen, dice);
                    owed = dice.first + dice.second == 7 && game.current_prompt() != ActionPrompt::discard;
                } else {
                    game.execute(chosen);
                    owed = chosen.type == ActionType::play_knight_card ||
                           (chosen.type == ActionType::discard_resource &&
                            game.current_prompt() != ActionPrompt::discard);
                }
                if (owed && !game.winning_color().has_value()) {
                    const bool open = !openhex_friendly_tiles(game).empty();
                    require((game.current_prompt() == ActionPrompt::move_robber) == open,
                            "the robber prompt follows exactly when a tile is open");
                    if (!open) {
                        ++stays;
                        knight_stays += chosen.type == ActionType::play_knight_card;
                        require(game.current_prompt() == ActionPrompt::play_turn &&
                                    game.current_turn_index() == turn && game.current_player_index() == turn &&
                                    game.board().robber_coordinate() == robber,
                                "the robber stays and the turn's owner plays on");
                    }
                }
            }
        }
    }
    std::cout << "friendly robber in random games: " << prompts << " prompts, " << shielding
              << " with shielded tiles, " << stays << " moves not owed (" << knight_stays << " knights)\n";
    require(prompts > 300 && shielding > 50 && redeals > 50 && stays > 20, "the games exercise the rule");
}

/**
 * A roll the bank cannot pay in full, per resource (OpenHex's
 * _distribute_resources): one seat claiming the resource takes what the
 * bank has left, however many buildings the claim comes from; with two or
 * more seats claiming it, nobody gets any. The bank never goes negative.
 */
void test_short_bank_pays_a_sole_claimant_the_remainder() {
    const ThreeTiles tiles;
    constexpr auto ore = static_cast<std::size_t>(Resource::ore);
    constexpr auto wheat = static_cast<std::size_t>(Resource::wheat);
    // The ore tile rolls on 8. `red_city` makes Red's corner building a city.
    const auto rolled = [&](int bank_ore, std::vector<int> red, bool red_city, int blue, Dice dice = Dice{4, 4}) {
        GameRules rules;
        rules.resource_bank[ore] = bank_ore;
        GameSetup setup;
        setup.players.resize(2);
        if (red_city) {
            require(red.front() == tiles.corner, "the city is the corner building");
            setup.players[0].cities = {red.front()};
            red.erase(red.begin());
        }
        setup.players[0].settlements = red;
        setup.players[1].settlements = {blue};
        Game game({Color::red, Color::blue}, tiles.map, 5, rules);
        game.apply_setup(setup);
        game.execute(find_action(game, ActionType::roll), dice);
        return game;
    };
    const auto ore_of = [&](const Game& game) {
        return std::array<int, 3>{game.player(Color::red).resources[ore], game.player(Color::blue).resources[ore],
                                  game.resource_bank()[ore]};
    };
    using Ore = std::array<int, 3>;  // Red's, Blue's, the bank's

    // A sole claimant: a city and a settlement on the tile claim 3 as one seat.
    require(ore_of(rolled(3, {tiles.corner, tiles.ore_only}, true, tiles.desert_only)) == Ore{3, 0, 0},
            "a bank that covers the claim pays it");
    require(ore_of(rolled(2, {tiles.corner, tiles.ore_only}, true, tiles.desert_only)) == Ore{2, 0, 0},
            "a sole claimant with two buildings takes the 2 that are left");
    require(ore_of(rolled(1, {tiles.corner}, true, tiles.desert_only)) == Ore{1, 0, 0},
            "a sole claimant's city takes the 1 that is left");
    require(ore_of(rolled(0, {tiles.ore_only}, false, tiles.desert_only)) == Ore{0, 0, 0},
            "an empty bank pays nothing");

    // Two claimants: all or nothing.
    require(ore_of(rolled(2, {tiles.ore_only}, false, tiles.corner)) == Ore{1, 1, 0},
            "two claimants the bank can pay are paid");
    require(ore_of(rolled(1, {tiles.ore_only}, false, tiles.corner)) == Ore{0, 0, 1},
            "two claimants and a short bank: nobody is paid and the bank keeps its card");
    require(ore_of(rolled(2, {tiles.corner}, true, tiles.ore_only)) == Ore{0, 0, 2},
            "also when one of them could have been paid in full");

    // Each resource on its own: the wheat tile (6) pays while ore is short.
    const Game wheat_roll = rolled(0, {tiles.corner}, true, tiles.ore_only, Dice{3, 3});
    require(wheat_roll.player(Color::red).resources[wheat] == 2 && wheat_roll.resource_bank()[wheat] == 17 &&
                ore_of(wheat_roll) == Ore{0, 0, 0},
            "the wheat roll pays from its own bank");
}

void test_tournament_setup_differential_fixture() {
    Game game({Color::red, Color::blue}, MapType::tournament, 42);
    const FlatActionSpace action_space(2, MapType::tournament);
    std::vector<std::size_t> played;
    while (game.is_initial_build_phase()) {
        const auto selected = std::min_element(
            game.playable_actions().begin(),
            game.playable_actions().end(),
            [&](const auto& lhs, const auto& rhs) {
                return action_space.index(lhs) < action_space.index(rhs);
            });
        played.push_back(action_space.index(*selected));
        game.execute(*selected);
    }

    require(
        played == std::vector<std::size_t>{126, 54, 128, 59, 130, 61, 132, 64},
        "setup action trace must match pinned Python Catanatron");
    require(
        game.player(Color::red).resources == std::array<int, 5>{0, 1, 0, 1, 1},
        "red setup resources must match Python");
    require(
        game.player(Color::blue).resources == std::array<int, 5>{1, 0, 1, 0, 1},
        "blue setup resources must match Python");
    require(
        game.player(Color::red).settlements == std::vector<int>{0, 14} &&
            game.player(Color::blue).settlements == std::vector<int>{10, 12},
        "setup settlements must match Python");
}

void test_development_card_lifecycle_and_knight() {
    Game game({Color::red, Color::blue}, MapType::tournament, 41);
    while (game.is_initial_build_phase()) {
        game.execute(game.playable_actions().front());
    }
    game.player(Color::red).resources = {0, 0, 1, 1, 1};
    game.execute(find_action(game, ActionType::roll), Dice{1, 1});
    game.execute(
        find_action(game, ActionType::buy_development_card),
        std::nullopt,
        cppanatron::DevelopmentCard::knight);
    require(
        game.player(Color::red).development_cards[0] == 1,
        "bought knight enters hand");
    require(
        std::none_of(
            game.playable_actions().begin(),
            game.playable_actions().end(),
            [](const auto& action) {
                return action.type == ActionType::play_knight_card;
            }),
        "development card cannot be played on purchase turn");

    game.execute(find_action(game, ActionType::end_turn));
    finish_turn(game);
    game.execute(find_action(game, ActionType::play_knight_card));
    require(
        game.player(Color::red).played_development_cards[0] == 1 &&
            game.player(Color::red).development_cards[0] == 0,
        "playing knight transfers it from hand to played count");
    require(
        game.current_prompt() == ActionPrompt::move_robber,
        "knight requires robber movement");
}

void test_year_of_plenty_and_monopoly() {
    Game game({Color::red, Color::blue}, MapType::tournament, 47);
    while (game.is_initial_build_phase()) {
        game.execute(game.playable_actions().front());
    }
    game.player(Color::red).resources = {0, 0, 2, 2, 2};
    game.execute(find_action(game, ActionType::roll), Dice{1, 1});
    game.execute(
        find_action(game, ActionType::buy_development_card),
        std::nullopt,
        cppanatron::DevelopmentCard::year_of_plenty);
    game.execute(
        find_action(game, ActionType::buy_development_card),
        std::nullopt,
        cppanatron::DevelopmentCard::monopoly);
    game.execute(find_action(game, ActionType::end_turn));
    finish_turn(game);

    const auto plenty = std::find_if(
        game.playable_actions().begin(),
        game.playable_actions().end(),
        [](const auto& action) {
            return action.type == ActionType::play_year_of_plenty &&
                   std::get<std::vector<Resource>>(action.value) ==
                       std::vector<Resource>{Resource::wood, Resource::ore};
        });
    require(plenty != game.playable_actions().end(), "year of plenty pair is legal");
    const auto hand_before = game.player(Color::red).resources;
    game.execute(*plenty);
    require(
        game.player(Color::red).resources[0] == hand_before[0] + 1 &&
            game.player(Color::red).resources[4] == hand_before[4] + 1,
        "year of plenty draws both selected resources");
    require(
        std::none_of(
            game.playable_actions().begin(),
            game.playable_actions().end(),
            [](const auto& action) {
                return action.type == ActionType::play_monopoly;
            }),
        "only one development card can be played per turn");

    finish_turn(game);
    finish_turn(game);
    game.player(Color::blue).resources[0] = 3;
    const auto monopoly = std::find_if(
        game.playable_actions().begin(),
        game.playable_actions().end(),
        [](const auto& action) {
            return action.type == ActionType::play_monopoly &&
                   std::get<Resource>(action.value) == Resource::wood;
        });
    require(monopoly != game.playable_actions().end(), "wood monopoly is legal");
    const int red_wood_before = game.player(Color::red).resources[0];
    game.execute(*monopoly);
    require(
        game.player(Color::blue).resources[0] == 0 &&
            game.player(Color::red).resources[0] == red_wood_before + 3,
        "monopoly transfers every selected opponent resource");
}

void test_largest_army_award() {
    Game game({Color::red, Color::blue}, MapType::tournament, 49);
    while (game.is_initial_build_phase()) {
        game.execute(game.playable_actions().front());
    }
    game.player(Color::red).resources = {0, 0, 3, 3, 3};
    game.execute(find_action(game, ActionType::roll), Dice{1, 1});
    for (int i = 0; i < 3; ++i) {
        game.execute(
            find_action(game, ActionType::buy_development_card),
            std::nullopt,
            cppanatron::DevelopmentCard::knight);
    }
    game.execute(find_action(game, ActionType::end_turn));

    for (int played = 1; played <= 3; ++played) {
        finish_turn(game);
        game.execute(find_action(game, ActionType::play_knight_card));
        require(
            game.current_prompt() == ActionPrompt::move_robber,
            "each knight requires robber movement");
        game.execute(find_action(game, ActionType::move_robber));
        require(
            game.player(Color::red).played_development_cards[0] == played,
            "played knight count advances");
        if (played < 3) {
            require(
                !game.player(Color::red).has_army,
                "fewer than three knights do not receive largest army");
            finish_turn(game);
        }
    }

    require(
        game.player(Color::red).has_army &&
            game.player(Color::red).victory_points == 4 &&
            game.player(Color::red).actual_victory_points == 4,
        "third knight awards largest army and two victory points");
}

void test_victory_point_card_is_hidden_and_terminal() {
    Game game(
        {Color::red, Color::blue},
        MapType::tournament,
        50,
        7,
        false,
        3);
    while (game.is_initial_build_phase()) {
        game.execute(game.playable_actions().front());
    }
    game.player(Color::red).resources = {0, 0, 1, 1, 1};
    game.execute(find_action(game, ActionType::roll), Dice{1, 1});
    game.execute(
        find_action(game, ActionType::buy_development_card),
        std::nullopt,
        cppanatron::DevelopmentCard::victory_point);

    require(
        game.player(Color::red).victory_points == 2 &&
            game.player(Color::red).actual_victory_points == 3,
        "victory point cards remain hidden from the public score");
    require(
        game.winning_color() == Color::red,
        "a hidden victory point card can end the game");
}

void test_longest_road_award() {
    Game game({Color::red, Color::blue}, MapType::tournament, 53);
    while (game.is_initial_build_phase()) {
        game.execute(game.playable_actions().front());
    }
    game.player(Color::red).resources = {15, 15, 0, 0, 0};
    game.execute(find_action(game, ActionType::roll), Dice{1, 1});

    int roads_built = 0;
    while (game.player(Color::red).longest_road_length < 5) {
        const auto road = std::find_if(
            game.playable_actions().begin(),
            game.playable_actions().end(),
            [](const auto& action) {
                return action.type == ActionType::build_road;
            });
        require(
            road != game.playable_actions().end(),
            "road network can extend to award length");
        game.execute(*road);
        require(++roads_built <= 13, "road award fixture must converge");
    }

    require(
        game.player(Color::red).has_road &&
            game.player(Color::red).victory_points == 4 &&
            game.player(Color::red).actual_victory_points == 4,
        "a five-edge path awards longest road and two victory points");
    require(
        !game.player(Color::blue).has_road,
        "only the longest qualifying network owns the award");
}

void test_enemy_endpoint_road_matches_reference_component_cache() {
    Board board(CatanMap::build(
        MapType::mini, 17, cppanatron::NumberPlacement::random));
    board.build_settlement(Color::blue, 0, true);
    board.build_settlement(Color::red, 2, true);

    board.build_road(Color::blue, {0, 1});
    board.build_road(Color::blue, {1, 2});
    board.build_road(Color::blue, {0, 20});
    board.build_road(Color::blue, {20, 22});
    board.build_road(Color::blue, {0, 5});

    require(
        board.is_friendly_road({1, 2}, Color::blue),
        "road ending at an enemy settlement remains on the board");
    require(
        board.longest_road(Color::blue) == 3,
        "road built into an existing enemy node is excluded from the "
        "reference connected-component cache");
}

std::vector<Edge> sorted_buildable_edges(const Board& board, Color color) {
    auto edges = board.buildable_edges(color);
    std::sort(edges.begin(), edges.end());
    return edges;
}

void test_settlement_cut_rebuilds_road_components() {
    Board board(CatanMap::build(
        MapType::base, 19, cppanatron::NumberPlacement::official_spiral));
    board.build_settlement(Color::red, 3, true);
    board.build_road(Color::red, {3, 4});
    board.build_road(Color::red, {4, 15});
    board.build_road(Color::red, {15, 17});
    board.build_settlement(Color::blue, 13, true);
    board.build_road(Color::blue, {13, 14});
    board.build_road(Color::blue, {14, 15});

    board.build_settlement(Color::blue, 15);

    require(
        board.longest_road(Color::red) == 2,
        "a settlement cut splits the road network at the enemy node");
    require(
        sorted_buildable_edges(board, Color::red) ==
            std::vector<Edge>{{2, 3}, {3, 12}, {4, 5}, {17, 18}, {17, 39}},
        "both halves of a cut network stay buildable, matching the reference");
}

void test_cannot_build_road_through_enemy_settlement() {
    Board board(CatanMap::build(
        MapType::base, 23, cppanatron::NumberPlacement::official_spiral));
    board.build_settlement(Color::red, 3, true);
    board.build_road(Color::red, {2, 3});
    board.build_road(Color::red, {1, 2});
    board.build_settlement(Color::blue, 1, true);

    require(
        sorted_buildable_edges(board, Color::red) ==
            std::vector<Edge>{{2, 9}, {3, 4}, {3, 12}},
        "roads cannot extend through an opponent's settlement");
}

void test_maritime_trade_and_other_development_cards() {
    Game game({Color::red, Color::blue}, MapType::tournament, 51);
    while (game.is_initial_build_phase()) {
        game.execute(game.playable_actions().front());
    }
    game.player(Color::red).resources = {4, 0, 1, 1, 1};
    game.execute(find_action(game, ActionType::roll), Dice{1, 1});

    const auto trade_it = std::find_if(
        game.playable_actions().begin(),
        game.playable_actions().end(),
        [](const auto& action) {
            if (action.type != ActionType::maritime_trade) {
                return false;
            }
            const auto& trade = std::get<cppanatron::MaritimeTrade>(action.value);
            return trade.cards[0] == Resource::wood &&
                   trade.cards[1] == Resource::wood &&
                   trade.cards[2] == Resource::wood &&
                   trade.cards[3] == Resource::wood &&
                   trade.cards[4] == Resource::ore;
        });
    require(trade_it != game.playable_actions().end(), "4:1 bank trade is legal");
    const auto before_trade = game.player(Color::red).resources;
    game.execute(*trade_it);
    require(
        game.player(Color::red).resources[0] == before_trade[0] - 4 &&
            game.player(Color::red).resources[4] == before_trade[4] + 1,
        "maritime trade exchanges exact resources");

    game.execute(
        find_action(game, ActionType::buy_development_card),
        std::nullopt,
        cppanatron::DevelopmentCard::road_building);
    game.execute(find_action(game, ActionType::end_turn));
    finish_turn(game);
    const std::size_t roads_before = game.player(Color::red).roads.size();
    game.execute(find_action(game, ActionType::play_road_building));
    game.execute(find_action(game, ActionType::build_road));
    if (std::any_of(
            game.playable_actions().begin(),
            game.playable_actions().end(),
            [](const auto& action) { return action.type == ActionType::build_road; })) {
        game.execute(find_action(game, ActionType::build_road));
    }
    require(
        game.player(Color::red).roads.size() >= roads_before + 1,
        "road building places free roads");
}

void test_domestic_trade_negotiation() {
    Game game({Color::red, Color::blue}, MapType::tournament, 61);
    while (game.is_initial_build_phase()) {
        game.execute(game.playable_actions().front());
    }
    game.player(Color::red).resources = {1, 0, 0, 0, 0};
    game.player(Color::blue).resources = {0, 1, 0, 0, 0};
    game.execute(find_action(game, ActionType::roll), Dice{1, 1});

    cppanatron::DomesticTrade offer;
    offer.offering[0] = 1;
    offer.asking[1] = 1;
    game.execute({Color::red, ActionType::offer_trade, offer});
    require(
        game.current_prompt() == ActionPrompt::decide_trade &&
            game.current_color() == Color::blue,
        "offer asks the opponent");
    game.execute(find_action(game, ActionType::accept_trade));
    require(
        game.current_prompt() == ActionPrompt::decide_acceptees &&
            game.current_color() == Color::red,
        "accepted offer returns to offering player");
    game.execute(find_action(game, ActionType::confirm_trade));
    require(
        game.current_prompt() == ActionPrompt::play_turn &&
            game.player(Color::red).resources[0] == 0 &&
            game.player(Color::red).resources[1] == 1 &&
            game.player(Color::blue).resources[0] == 1 &&
            game.player(Color::blue).resources[1] == 0,
        "confirmed domestic trade exchanges both hands");
}

void test_domestic_trade_reject_and_cancel() {
    Game game(
        {Color::red, Color::blue, Color::white},
        MapType::tournament,
        67);
    while (game.is_initial_build_phase()) {
        game.execute(game.playable_actions().front());
    }
    game.player(Color::red).resources = {1, 0, 0, 0, 0};
    game.player(Color::blue).resources = {0, 1, 0, 0, 0};
    game.player(Color::white).resources = {0, 1, 0, 0, 0};
    game.execute(find_action(game, ActionType::roll), Dice{1, 1});

    cppanatron::DomesticTrade offer;
    offer.offering[0] = 1;
    offer.asking[1] = 1;
    game.execute({Color::red, ActionType::offer_trade, offer});
    require(
        game.current_color() == Color::blue,
        "first opponent receives the trade offer");
    game.execute(find_action(game, ActionType::reject_trade));
    require(
        game.current_color() == Color::white &&
            game.current_prompt() == ActionPrompt::decide_trade,
        "a rejection advances to the next opponent");
    game.execute(find_action(game, ActionType::accept_trade));
    require(
        game.current_color() == Color::red &&
            game.current_prompt() == ActionPrompt::decide_acceptees,
        "the offer returns after every opponent answers");

    const auto red_before = game.player(Color::red).resources;
    const auto white_before = game.player(Color::white).resources;
    game.execute(find_action(game, ActionType::cancel_trade));
    require(
        game.current_color() == Color::red &&
            game.current_prompt() == ActionPrompt::play_turn &&
            game.player(Color::red).resources == red_before &&
            game.player(Color::white).resources == white_before,
        "cancelling an accepted offer preserves both hands");
}

}  // namespace

void test_six_player_games_and_rules() {
    require(FlatActionSpace(5, MapType::base).size() == 370, "BASE 5p action space size");
    require(FlatActionSpace(6, MapType::base).size() == 389, "BASE 6p action space size");
    require_throws(
        [] { (void)FlatActionSpace(7, MapType::base); }, "seven players have no colors");

    const std::vector<Color> six(cppanatron::kColors.begin(), cppanatron::kColors.end());
    Game game(six, MapType::base, 71);
    while (game.is_initial_build_phase()) {
        game.execute(game.playable_actions().front());
    }
    for (const auto& player : game.players()) {
        require(player.settlements.size() == 2, "six players each place two settlements");
    }

    // The default rules must leave every seeded game unchanged.
    const Game loose(six, MapType::base, 5);
    const Game explicit_rules(six, MapType::base, 5, GameRules{});
    require(loose.search_equivalent(explicit_rules), "default rules match the loose constructor");
    require(loose.development_cards_remaining() == 25, "base deck has 25 cards");

    GameRules extension;
    extension.resource_bank = {24, 24, 24, 24, 24};
    extension.development_deck = {20, 3, 3, 3, 5};
    extension.roads_per_player = 13;
    extension.settlements_per_player = 4;
    extension.cities_per_player = 3;
    const Game extended(six, MapType::base, 5, extension);
    require(extended.rules() == extension, "a game keeps its rules");
    require(extended.resource_bank() == extension.resource_bank, "bank follows the rules");
    require(extended.development_cards_remaining() == 34, "extension deck has 34 cards");
    const auto& deck = extended.development_deck();
    require(
        std::count(deck.begin(), deck.end(), cppanatron::DevelopmentCard::knight) == 20,
        "extension deck has 20 knights");
    for (const auto& player : extended.players()) {
        require(
            player.roads_available == 13 && player.settlements_available == 4 &&
                player.cities_available == 3,
            "pieces follow the rules");
    }

    GameRules too_few_settlements;
    too_few_settlements.settlements_per_player = 1;
    require_throws(
        [&] { (void)Game(six, MapType::base, 5, too_few_settlements); },
        "a player needs two settlements for the initial build phase");
}

void test_explicit_maps_and_shared_boards() {
    for (MapType map_type : {MapType::mini, MapType::base, MapType::tournament}) {
        const CatanMap built = CatanMap::build(map_type, 13);
        const CatanMap rebuilt = CatanMap::from_spec(built.spec());
        require(rebuilt.spec() == built.spec(), "a map's spec rebuilds the same spec");
        require(rebuilt.land_edges() == built.land_edges(), "a spec rebuilds the same edges");
        for (std::size_t i = 0; i < built.tiles().size(); ++i) {
            require(
                rebuilt.tiles()[i].nodes == built.tiles()[i].nodes &&
                    rebuilt.tiles()[i].id == built.tiles()[i].id &&
                    rebuilt.tiles()[i].resource == built.tiles()[i].resource,
                "a spec rebuilds the same tiles");
        }
        for (int node : built.land_node_list()) {
            require(
                built.land_node_list()[static_cast<std::size_t>(built.node_slot(node))] == node,
                "node slots invert the land node list");
            for (int neighbor : built.neighbors(node)) {
                require(
                    built.edge_slot({neighbor, node}) >= 0,
                    "neighbors are joined by a land edge");
            }
        }
    }

    cppanatron::MapSpec spec;
    spec.tiles.push_back({.coordinate = {0, 0, 0}, .kind = cppanatron::TileKind::land,
                          .resource = std::nullopt, .number = std::nullopt,
                          .direction = std::nullopt, .port_resource = std::nullopt});
    spec.tiles.push_back({.coordinate = {1, -1, 0}, .kind = cppanatron::TileKind::land,
                          .resource = Resource::ore, .number = 8,
                          .direction = std::nullopt, .port_resource = std::nullopt});
    spec.tiles.push_back({.coordinate = {2, -2, 0}, .kind = cppanatron::TileKind::port,
                          .resource = std::nullopt, .number = std::nullopt,
                          .direction = cppanatron::Direction::west,
                          .port_resource = Resource::wheat});
    const auto shared = std::make_shared<const CatanMap>(CatanMap::from_spec(spec));
    require(shared->num_nodes() == 10 && shared->land_edges().size() == 11,
            "two land tiles share an edge");
    const auto [first, second] = cppanatron::port_nodes(shared->tiles()[2]);
    require(
        shared->node_ports(first) == (1U << static_cast<unsigned>(Resource::wheat)) &&
            shared->node_ports(second) == shared->node_ports(first),
        "port nodes carry their port");

    Game game({Color::red, Color::blue}, shared, 3);
    const Game copy = game;
    require(
        &game.board().map() == shared.get() && &copy.board().map() == shared.get(),
        "games and their copies share the map");
    require(game.playable_actions().size() == 10, "every land node is open at the start");

    spec.tiles[1].number = 7;
    require_throws([&] { (void)CatanMap::from_spec(spec); }, "seven is not a tile number");
    spec.tiles[1].number = 8;
    spec.tiles[0].resource = Resource::wood;
    spec.tiles[0].number = 4;
    require_throws([&] { (void)CatanMap::from_spec(spec); }, "a map needs a desert");
}

int main() {
    try {
        test_explicit_maps_and_shared_boards();
        test_topology_contract();
        test_number_and_tournament_contract();
        test_settlement_distance_rule();
        test_connected_roads_and_city();
        test_two_player_initial_setup_and_turn();
        test_flat_action_space_contract();
        test_mcts_dice_outcomes_are_exact();
        test_mcts_card_and_robber_outcomes_are_exact();
        test_mcts_visits_only_legal_actions();
        test_mcts_chance_values_ignore_unvisited_outcomes();
        test_mcts_reuses_only_deterministic_subtrees();
        test_mcts_canonical_discard_prunes_infeasible_suffixes();
        test_seven_discard_and_robber_transition();
        test_friendly_robber_shields_low_public_points();
        test_friendly_robber_spares_no_one_at_three_public_points();
        test_friendly_robber_ignores_hidden_victory_point_cards();
        test_friendly_robber_shields_the_movers_own_tiles();
        test_friendly_robber_stays_after_a_seven_when_every_tile_is_shielded();
        test_friendly_robber_knight_without_an_open_tile();
        test_friendly_robber_knight_is_judged_after_the_award();
        test_short_bank_pays_a_sole_claimant_the_remainder();
        test_friendly_robber_after_a_knight_takes_largest_army();
        test_random_friendly_games_follow_openhex();
        test_tournament_setup_differential_fixture();
        test_development_card_lifecycle_and_knight();
        test_year_of_plenty_and_monopoly();
        test_largest_army_award();
        test_victory_point_card_is_hidden_and_terminal();
        test_longest_road_award();
        test_enemy_endpoint_road_matches_reference_component_cache();
        test_settlement_cut_rebuilds_road_components();
        test_cannot_build_road_through_enemy_settlement();
        test_maritime_trade_and_other_development_cards();
        test_domestic_trade_negotiation();
        test_domestic_trade_reject_and_cancel();
        test_six_player_games_and_rules();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "cppanatron board tests passed\n";
    return 0;
}
