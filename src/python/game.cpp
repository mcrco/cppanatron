#include "bindings.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <variant>

#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "cppanatron/action_space.hpp"
#include "cppanatron/board.hpp"
#include "cppanatron/game.hpp"
#include "cppanatron/map.hpp"
#include "cppanatron/value_player.hpp"

namespace cppanatron::python {
namespace {

using PlayerRef = std::variant<int, Color>;

/** PlayerState plus the turn-relative counters observations report. */
struct PlayerSnapshot {
    PlayerState state;
    int turns_since_last_knight{};
    int turns_since_last_development_card_bought{};
};

Color resolve_color(const Game& game, const PlayerRef& player) {
    if (const auto* color = std::get_if<Color>(&player)) {
        return *color;
    }
    const int index = std::get<int>(player);
    if (index < 0 || static_cast<std::size_t>(index) >= game.colors().size()) {
        throw std::out_of_range("player index is out of range");
    }
    return game.colors()[static_cast<std::size_t>(index)];
}

int turns_since(const Game& game, int completed_turn) {
    return completed_turn < 0 ? -1 : game.completed_turns() - completed_turn;
}

PlayerSnapshot snapshot(const Game& game, Color color) {
    const PlayerState& state = game.player(color);
    return {
        state,
        turns_since(game, state.last_knight_completed_turn),
        turns_since(game, state.last_development_card_bought_completed_turn),
    };
}

std::optional<int> player_index(const Game& game, std::optional<Color> color) {
    if (!color.has_value()) {
        return std::nullopt;
    }
    const auto& colors = game.colors();
    return static_cast<int>(std::find(colors.begin(), colors.end(), *color) - colors.begin());
}

std::vector<Color> game_colors(const std::variant<int, std::vector<Color>>& players) {
    if (const auto* colors = std::get_if<std::vector<Color>>(&players)) {
        return *colors;
    }
    const int count = std::get<int>(players);
    if (count < 1 || count > static_cast<int>(kColors.size())) {
        throw std::invalid_argument("a game needs between one and six players");
    }
    return {kColors.begin(), kColors.begin() + count};
}

nb::tuple edge_to_python(Edge edge) { return nb::make_tuple(edge.a, edge.b); }

Edge edge_from_python(const std::array<int, 2>& edge) { return Edge{edge[0], edge[1]}.normalized(); }

nb::dict buildings_to_python(const Board& board) {
    nb::dict result;
    for (const auto& [node, building] : board.buildings()) {
        result[nb::int_(node)] = nb::make_tuple(building.color, building.building);
    }
    return result;
}

nb::dict roads_to_python(const Board& board) {
    nb::dict result;
    for (const auto& [edge, color] : board.roads()) {
        result[edge_to_python(edge)] = nb::cast(color);
    }
    return result;
}

nb::dict tiles_to_python(const CatanMap& map, bool land_only) {
    nb::dict result;
    for (const auto& [coordinate, index] : map.coordinate_to_tile()) {
        const Tile& tile = map.tiles()[static_cast<std::size_t>(index)];
        if (!land_only || tile.kind == TileKind::land) {
            result[coordinate_to_python(coordinate)] = nb::cast(tile, nb::rv_policy::copy);
        }
    }
    return result;
}

Vector<bool> action_mask(const FlatActionSpace& space, const Game& game) {
    // std::vector<bool> is bit-packed, so NumPy needs its own byte-per-entry buffer.
    auto* legal = new bool[space.size()]();
    nb::capsule owner(legal, [](void* pointer) noexcept { delete[] static_cast<bool*>(pointer); });
    for (const Action& action : game.playable_actions()) {
        legal[space.index(action, game.colors())] = true;
    }
    return Vector<bool>(legal, {space.size()}, owner);
}

}  // namespace

GameRules resolve_rules(
    const std::optional<GameRules>& rules,
    std::optional<int> discard_limit,
    std::optional<bool> friendly_robber,
    std::optional<int> vps_to_win) {
    if (rules.has_value()) {
        if (discard_limit.has_value() || friendly_robber.has_value() || vps_to_win.has_value()) {
            throw std::invalid_argument(
                "set discard_limit, friendly_robber and vps_to_win inside rules, not beside it");
        }
        return *rules;
    }
    GameRules result;
    result.discard_limit = discard_limit.value_or(result.discard_limit);
    result.friendly_robber = friendly_robber.value_or(result.friendly_robber);
    result.victory_points_to_win = vps_to_win.value_or(result.victory_points_to_win);
    return result;
}

void bind_game(nb::module_& module) {
    const GameRules defaults;
    nb::class_<GameRules>(
        module, "GameRules",
        "Rule settings: scoring, discards, the robber, and the bank, deck and pieces.\n"
        "Defaults are the base game's. Counts follow Resource and DevelopmentCard order.")
        .def(
            "__init__",
            [](GameRules* self, int vps_to_win, int discard_limit, bool friendly_robber,
               const std::array<int, 5>& resource_bank, const std::array<int, 5>& development_deck,
               int roads_per_player, int settlements_per_player, int cities_per_player) {
                GameRules rules;
                rules.victory_points_to_win = vps_to_win;
                rules.discard_limit = discard_limit;
                rules.friendly_robber = friendly_robber;
                rules.resource_bank = resource_bank;
                rules.development_deck = development_deck;
                rules.roads_per_player = roads_per_player;
                rules.settlements_per_player = settlements_per_player;
                rules.cities_per_player = cities_per_player;
                validate_rules(rules);
                new (self) GameRules(rules);
            },
            nb::kw_only(), nb::arg("vps_to_win") = defaults.victory_points_to_win,
            nb::arg("discard_limit") = defaults.discard_limit,
            nb::arg("friendly_robber") = defaults.friendly_robber,
            nb::arg("resource_bank") = defaults.resource_bank,
            nb::arg("development_deck") = defaults.development_deck,
            nb::arg("roads_per_player") = defaults.roads_per_player,
            nb::arg("settlements_per_player") = defaults.settlements_per_player,
            nb::arg("cities_per_player") = defaults.cities_per_player)
        .def_ro("vps_to_win", &GameRules::victory_points_to_win)
        .def_ro("discard_limit", &GameRules::discard_limit)
        .def_ro("friendly_robber", &GameRules::friendly_robber)
        .def_prop_ro(
            "resource_bank", [](const GameRules& rules) { return to_tuple(rules.resource_bank); },
            "Starting bank ordered WOOD, BRICK, SHEEP, WHEAT, ORE.")
        .def_prop_ro(
            "development_deck",
            [](const GameRules& rules) { return to_tuple(rules.development_deck); },
            "Starting deck counts in DevelopmentCard order.")
        .def_ro("roads_per_player", &GameRules::roads_per_player)
        .def_ro("settlements_per_player", &GameRules::settlements_per_player)
        .def_ro("cities_per_player", &GameRules::cities_per_player)
        .def(
            "__eq__", [](const GameRules& lhs, const GameRules& rhs) { return lhs == rhs; },
            nb::is_operator())
        .def("__repr__", [](const GameRules& rules) {
            return nb::str(
                       "GameRules(vps_to_win={}, discard_limit={}, friendly_robber={}, "
                       "resource_bank={}, development_deck={}, roads_per_player={}, "
                       "settlements_per_player={}, cities_per_player={})")
                .format(
                    rules.victory_points_to_win, rules.discard_limit, rules.friendly_robber,
                    to_tuple(rules.resource_bank), to_tuple(rules.development_deck),
                    rules.roads_per_player, rules.settlements_per_player,
                    rules.cities_per_player);
        });

    nb::class_<PlayerSetup>(
        module, "PlayerSetup",
        "One seat's pieces and cards for Game.apply_setup. Counts follow Resource and\n"
        "DevelopmentCard order; development_cards includes victory point cards.")
        .def(
            "__init__",
            [](PlayerSetup* self, std::vector<int> settlements, std::vector<int> cities,
               const std::vector<std::array<int, 2>>& roads, const std::array<int, 5>& resources,
               const std::array<int, 5>& development_cards,
               const std::array<int, 5>& played_development_cards) {
                PlayerSetup setup;
                setup.settlements = std::move(settlements);
                setup.cities = std::move(cities);
                for (const auto& road : roads) {
                    setup.roads.push_back(edge_from_python(road));
                }
                setup.resources = resources;
                setup.development_cards = development_cards;
                setup.played_development_cards = played_development_cards;
                new (self) PlayerSetup(std::move(setup));
            },
            nb::kw_only(), nb::arg("settlements") = std::vector<int>{},
            nb::arg("cities") = std::vector<int>{},
            nb::arg("roads") = std::vector<std::array<int, 2>>{},
            nb::arg("resources") = std::array<int, 5>{},
            nb::arg("development_cards") = std::array<int, 5>{},
            nb::arg("played_development_cards") = std::array<int, 5>{})
        .def_ro("settlements", &PlayerSetup::settlements)
        .def_ro("cities", &PlayerSetup::cities)
        .def_prop_ro(
            "roads",
            [](const PlayerSetup& setup) {
                nb::list roads;
                for (Edge edge : setup.roads) {
                    roads.append(edge_to_python(edge));
                }
                return roads;
            })
        .def_prop_ro("resources", [](const PlayerSetup& setup) { return to_tuple(setup.resources); })
        .def_prop_ro(
            "development_cards",
            [](const PlayerSetup& setup) { return to_tuple(setup.development_cards); })
        .def_prop_ro(
            "played_development_cards",
            [](const PlayerSetup& setup) { return to_tuple(setup.played_development_cards); });

    nb::class_<Tile>(module, "Tile", "A board hexagon: land, water or port.")
        .def_ro("kind", &Tile::kind)
        .def_ro("id", &Tile::id)
        .def_ro("resource", &Tile::resource)
        .def_ro("number", &Tile::number)
        .def_ro("port_direction", &Tile::port_direction)
        .def_prop_ro(
            "nodes", [](const Tile& tile) { return to_tuple(tile.nodes); },
            "Node ids ordered NORTH, NORTHEAST, SOUTHEAST, SOUTH, SOUTHWEST, NORTHWEST.")
        .def_prop_ro(
            "edges",
            [](const Tile& tile) {
                nb::list edges;
                for (Edge edge : tile.edges) {
                    edges.append(edge_to_python(edge));
                }
                return nb::tuple(edges);
            },
            "Edges ordered EAST, SOUTHEAST, SOUTHWEST, WEST, NORTHWEST, NORTHEAST.")
        .def("__repr__", [](const Tile& tile) {
            return nb::str("Tile({!r}, id={}, resource={!r}, number={!r})")
                .format(tile.kind, tile.id, nb::cast(tile.resource), nb::cast(tile.number));
        });

    nb::class_<CatanMap>(module, "CatanMap")
        .def_static(
            "build", &CatanMap::build, nb::arg("map_type"), nb::arg("seed") = 0,
            nb::arg("number_placement") = NumberPlacement::official_spiral)
        .def_prop_ro(
            "tiles", [](const CatanMap& map) { return tiles_to_python(map, false); },
            "Every tile keyed by cube coordinate, like Catanatron's CatanMap.tiles.")
        .def_prop_ro(
            "land_tiles", [](const CatanMap& map) { return tiles_to_python(map, true); })
        .def_prop_ro(
            "land_nodes",
            [](const CatanMap& map) {
                return nb::frozenset(nb::cast(std::vector<int>(
                    map.land_nodes().begin(), map.land_nodes().end())));
            })
        .def_prop_ro(
            "land_edges",
            [](const CatanMap& map) {
                nb::list edges;
                for (Edge edge : map.land_edges()) {
                    edges.append(edge_to_python(edge));
                }
                return edges;
            })
        .def(
            "tile_at",
            [](const CatanMap& map, nb::handle coordinate) {
                return map.tile_at(coordinate_from_python(coordinate));
            },
            nb::arg("coordinate"))
        .def_prop_ro("num_nodes", &CatanMap::num_nodes);

    nb::class_<Board>(module, "Board", "Live, read-only view of a game's board.")
        .def_prop_ro("map", &Board::map, nb::rv_policy::reference_internal)
        .def_prop_ro(
            "buildings", &buildings_to_python, "``{node: (Color, Building)}`` snapshot.")
        .def_prop_ro("roads", &roads_to_python, "``{(a, b): Color}`` snapshot with ``a < b``.")
        .def_prop_ro(
            "robber_coordinate",
            [](const Board& board) { return coordinate_to_python(board.robber_coordinate()); })
        .def(
            "buildable_node_ids", &Board::buildable_node_ids, nb::arg("color"),
            nb::arg("initial_build_phase") = false)
        .def(
            "buildable_edges",
            [](const Board& board, Color color) {
                nb::list edges;
                for (Edge edge : board.buildable_edges(color)) {
                    edges.append(edge_to_python(edge));
                }
                return edges;
            },
            nb::arg("color"))
        .def(
            "player_port_resources",
            [](const Board& board, Color color) {
                nb::set result;
                for (const auto& resource : board.player_port_resources(color)) {
                    result.add(nb::cast(resource));
                }
                return result;
            },
            nb::arg("color"), "Port resources a player can trade at; None is a 3:1 port.")
        .def("node_color", &Board::node_color, nb::arg("node"))
        .def(
            "edge_color",
            [](const Board& board, const std::array<int, 2>& edge) {
                return board.edge_color(edge_from_python(edge));
            },
            nb::arg("edge"))
        .def("longest_road", &Board::longest_road, nb::arg("color"))
        .def_prop_ro("longest_road_color", &Board::longest_road_color);

    nb::class_<PlayerSnapshot>(module, "PlayerState", "Snapshot of one player's state.")
        .def_prop_ro("victory_points", [](const PlayerSnapshot& p) { return p.state.victory_points; })
        .def_prop_ro(
            "actual_victory_points",
            [](const PlayerSnapshot& p) { return p.state.actual_victory_points; })
        .def_prop_ro("roads_available", [](const PlayerSnapshot& p) { return p.state.roads_available; })
        .def_prop_ro(
            "settlements_available",
            [](const PlayerSnapshot& p) { return p.state.settlements_available; })
        .def_prop_ro(
            "cities_available", [](const PlayerSnapshot& p) { return p.state.cities_available; })
        .def_prop_ro("has_road", [](const PlayerSnapshot& p) { return p.state.has_road; })
        .def_prop_ro("has_army", [](const PlayerSnapshot& p) { return p.state.has_army; })
        .def_prop_ro("has_rolled", [](const PlayerSnapshot& p) { return p.state.has_rolled; })
        .def_prop_ro(
            "has_played_development_card_in_turn",
            [](const PlayerSnapshot& p) { return p.state.has_played_development_card_in_turn; })
        .def_prop_ro(
            "longest_road_length",
            [](const PlayerSnapshot& p) { return p.state.longest_road_length; })
        .def_prop_ro(
            "resources", [](const PlayerSnapshot& p) { return to_tuple(p.state.resources); },
            "Hand counts ordered WOOD, BRICK, SHEEP, WHEAT, ORE.")
        .def_prop_ro(
            "development_cards",
            [](const PlayerSnapshot& p) { return to_tuple(p.state.development_cards); },
            "Unplayed card counts in DevelopmentCard order.")
        .def_prop_ro(
            "played_development_cards",
            [](const PlayerSnapshot& p) { return to_tuple(p.state.played_development_cards); })
        .def_prop_ro(
            "development_card_owned_at_start",
            [](const PlayerSnapshot& p) {
                return to_tuple(p.state.development_card_owned_at_start);
            },
            "Whether each playable card type was held at the start of the turn.")
        .def_prop_ro("settlements", [](const PlayerSnapshot& p) { return p.state.settlements; })
        .def_prop_ro("cities", [](const PlayerSnapshot& p) { return p.state.cities; })
        .def_prop_ro(
            "roads",
            [](const PlayerSnapshot& p) {
                nb::list roads;
                for (Edge edge : p.state.roads) {
                    roads.append(edge_to_python(edge));
                }
                return roads;
            })
        .def_ro("turns_since_last_knight", &PlayerSnapshot::turns_since_last_knight)
        .def_ro(
            "turns_since_last_development_card_bought",
            &PlayerSnapshot::turns_since_last_development_card_bought);

    nb::class_<Game>(
        module, "Game",
        "A complete game state. Copying a game copies its RNG, so copies replay\n"
        "identically; pass explicit chance outcomes to execute() to diverge.")
        .def(
            "__init__",
            [](Game* self, const std::variant<int, std::vector<Color>>& players, MapType map_type,
               std::uint64_t seed, std::optional<std::uint64_t> map_seed,
               NumberPlacement number_placement, std::optional<int> discard_limit,
               std::optional<bool> friendly_robber, std::optional<int> vps_to_win,
               std::optional<GameRules> rules) {
                new (self) Game(
                    game_colors(players), map_type, seed,
                    resolve_rules(rules, discard_limit, friendly_robber, vps_to_win),
                    number_placement, map_seed);
            },
            nb::arg("players") = 2, nb::kw_only(), nb::arg("map_type") = MapType::base,
            nb::arg("seed") = 0, nb::arg("map_seed") = nb::none(),
            nb::arg("number_placement") = NumberPlacement::official_spiral,
            nb::arg("discard_limit") = nb::none(), nb::arg("friendly_robber") = nb::none(),
            nb::arg("vps_to_win") = nb::none(), nb::arg("rules") = nb::none(),
            "Create a game for ``players`` (a count up to six, or explicit seating).\n\n"
            "``seed`` drives dice, cards and steals; ``map_seed`` (default ``seed``)\n"
            "drives the board. ``rules`` sets every rule at once; otherwise the base\n"
            "game's apply, with ``discard_limit`` (7), ``friendly_robber`` (False) and\n"
            "``vps_to_win`` (10) as overrides.")
        .def("__repr__", [](const Game& game) {
            return nb::str("Game(colors={!r}, current_color={!r}, current_prompt={!r}, num_turns={})")
                .format(game.colors(), game.current_color(), game.current_prompt(), game.num_turns());
        })
        .def("copy", [](const Game& game) { return Game(game); })
        .def("__copy__", [](const Game& game) { return Game(game); })
        .def("__deepcopy__", [](const Game& game, nb::handle) { return Game(game); }, nb::arg("memo"))
        .def_prop_ro("colors", &Game::colors, nb::rv_policy::copy)
        .def_prop_ro("num_players", [](const Game& game) { return game.colors().size(); })
        .def_prop_ro("current_color", &Game::current_color)
        .def_prop_ro("current_player_index", &Game::current_player_index)
        .def_prop_ro("current_turn_index", &Game::current_turn_index)
        .def_prop_ro("current_prompt", &Game::current_prompt)
        .def_prop_ro("num_turns", &Game::num_turns)
        .def_prop_ro("completed_turns", &Game::completed_turns)
        .def_prop_ro("is_initial_build_phase", &Game::is_initial_build_phase)
        .def_prop_ro("is_discarding", &Game::is_discarding)
        .def_prop_ro("is_moving_robber", &Game::is_moving_robber)
        .def_prop_ro("is_road_building", &Game::is_road_building)
        .def_prop_ro("vps_to_win", &Game::victory_points_to_win)
        .def_prop_ro("discard_limit", &Game::discard_limit)
        .def_prop_ro("friendly_robber", &Game::friendly_robber)
        .def_prop_ro("rules", &Game::rules, nb::rv_policy::copy)
        .def_prop_ro(
            "resource_bank", [](const Game& game) { return to_tuple(game.resource_bank()); })
        .def_prop_ro("development_cards_remaining", &Game::development_cards_remaining)
        // Copies: the game regenerates this vector on every transition.
        .def_prop_ro("playable_actions", &Game::playable_actions, nb::rv_policy::copy)
        .def_prop_ro("winning_color", &Game::winning_color)
        .def_prop_ro(
            "winning_player_index",
            [](const Game& game) { return player_index(game, game.winning_color()); })
        .def_prop_ro("board", &Game::board, nb::rv_policy::reference_internal)
        .def(
            "player",
            [](const Game& game, const PlayerRef& player) {
                return snapshot(game, resolve_color(game, player));
            },
            nb::arg("player"), "Snapshot of a player by seat index or Color.")
        .def_prop_ro(
            "players",
            [](const Game& game) {
                std::vector<PlayerSnapshot> players;
                for (Color color : game.colors()) {
                    players.push_back(snapshot(game, color));
                }
                return players;
            })
        .def(
            "remaining_discards",
            [](const Game& game, const PlayerRef& player) {
                return game.remaining_discards(resolve_color(game, player));
            },
            nb::arg("player"))
        .def(
            "execute",
            [](Game& game, const Action& action, std::optional<std::array<int, 2>> dice,
               std::optional<DevelopmentCard> development_card,
               std::optional<Resource> stolen_resource) {
                std::optional<Dice> replay_dice;
                if (dice.has_value()) {
                    const auto [first, second] = *dice;
                    if (first < 1 || first > 6 || second < 1 || second > 6) {
                        throw std::invalid_argument("dice must both be in [1, 6]");
                    }
                    replay_dice = Dice{first, second};
                }
                game.execute(
                    resolve_playable(game, action), replay_dice, development_card, stolen_resource);
            },
            nb::arg("action"), nb::kw_only(), nb::arg("dice") = nb::none(),
            nb::arg("development_card") = nb::none(), nb::arg("stolen_resource") = nb::none(),
            "Apply an action. Chance outcomes (dice, drawn card, stolen resource)\n"
            "come from the game's RNG unless given, as when replaying a record.\n"
            "Raises ValueError for actions that are not currently playable.")
        .def(
            "search_equivalent", &Game::search_equivalent, nb::arg("other"),
            "Exact state equality, including hidden state and RNG.")
        .def(
            "set_resources",
            [](Game& game, const PlayerRef& player, const std::array<int, 5>& resources) {
                game.set_resources(resolve_color(game, player), resources);
            },
            nb::arg("player"), nb::arg("resources"),
            "Scenario setup: overwrite a hand without debiting the bank.")
        .def(
            "set_development_cards",
            [](Game& game, const PlayerRef& player, const std::array<int, 5>& cards) {
                game.set_development_cards(resolve_color(game, player), cards);
            },
            nb::arg("player"), nb::arg("cards"),
            "Scenario setup: overwrite unplayed development cards, playable this turn.\n"
            "Victory point cards must be bought instead.")
        .def(
            "apply_setup",
            [](Game& game, const std::vector<PlayerSetup>& players,
               std::optional<std::array<int, 3>> robber, int current_player, bool has_rolled,
               bool has_played_development_card, int completed_turns,
               std::optional<int> longest_road, std::optional<int> largest_army) {
                GameSetup setup;
                setup.players = players;
                if (robber.has_value()) {
                    setup.robber = Coordinate{(*robber)[0], (*robber)[1], (*robber)[2]};
                }
                setup.current_player = current_player;
                setup.has_rolled = has_rolled;
                setup.has_played_development_card = has_played_development_card;
                setup.completed_turns = completed_turns;
                setup.longest_road = longest_road;
                setup.largest_army = largest_army;
                game.apply_setup(setup);
            },
            nb::arg("players"), nb::kw_only(), nb::arg("robber") = nb::none(),
            nb::arg("current_player") = 0, nb::arg("has_rolled") = false,
            nb::arg("has_played_development_card") = false, nb::arg("completed_turns") = 0,
            nb::arg("longest_road") = nb::none(), nb::arg("largest_army") = nb::none(),
            "Scenario setup: put a freshly created game into an exact position.\n\n"
            "``players`` holds one PlayerSetup per seat. The bank and deck are the\n"
            "rules' starting counts minus every hand and played card, and the deck\n"
            "is reshuffled. Victory points and awards follow from the pieces and\n"
            "cards; ``longest_road``/``largest_army`` name the holder's seat (-1 for\n"
            "nobody) when the leader is tied or must be explicit. The position is at\n"
            "``current_player``'s turn, before or after the roll, with every held\n"
            "card playable. Raises ValueError for positions the rules cannot produce.")
        .def(
            "determinize_development_cards",
            [](Game& game, const PlayerRef& player, const std::array<int, 5>& cards,
               std::uint64_t seed) {
                game.determinize_development_cards(resolve_color(game, player), cards, seed);
            },
            nb::arg("player"), nb::arg("cards"), nb::arg("seed") = 0,
            "Information-set determinization: deal a player's hidden development cards\n"
            "from the cards they could hold (their hand plus the deck), return the rest\n"
            "to a reshuffled deck, and keep victory points consistent.");

    nb::class_<FlatActionSpace>(
        module, "FlatActionSpace",
        "CatanRL's fixed action table. Robber victims are relative seat offsets,\n"
        "so one table serves every seat.")
        .def(nb::init<int, MapType>(), nb::arg("num_players"), nb::arg("map_type"))
        .def("__len__", &FlatActionSpace::size)
        .def_prop_ro("size", &FlatActionSpace::size)
        .def_prop_ro("actions", &FlatActionSpace::actions, nb::rv_policy::copy)
        .def("__getitem__", &FlatActionSpace::at, nb::arg("index"), nb::rv_policy::copy)
        .def(
            "index",
            [](const FlatActionSpace& space, const Action& action, const Game& game) {
                return space.index(action, game.colors());
            },
            nb::arg("action"), nb::arg("game"), "Flat index of a game action.")
        .def(
            "decode",
            [](const FlatActionSpace& space, std::size_t index, const Game& game) {
                return space.decode(index, game.current_color(), game.colors());
            },
            nb::arg("index"), nb::arg("game"),
            "The action a flat index means for the game's current player.")
        .def("mask", &action_mask, nb::arg("game"), "Boolean legality mask over the table.");

    nb::class_<ValueWeights>(module, "ValueWeights", "Weights of Catanatron's value heuristic.")
        .def(nb::init<>())
        .def_rw("public_vps", &ValueWeights::public_vps)
        .def_rw("production", &ValueWeights::production)
        .def_rw("enemy_production", &ValueWeights::enemy_production)
        .def_rw("num_tiles", &ValueWeights::num_tiles)
        .def_rw("reachable_production_0", &ValueWeights::reachable_production_0)
        .def_rw("reachable_production_1", &ValueWeights::reachable_production_1)
        .def_rw("buildable_nodes", &ValueWeights::buildable_nodes)
        .def_rw("longest_road", &ValueWeights::longest_road)
        .def_rw("hand_synergy", &ValueWeights::hand_synergy)
        .def_rw("hand_resources", &ValueWeights::hand_resources)
        .def_rw("discard_penalty", &ValueWeights::discard_penalty)
        .def_rw("hand_devs", &ValueWeights::hand_devs)
        .def_rw("army_size", &ValueWeights::army_size);

    nb::enum_<ValueEnemies>(
        module, "ValueEnemies",
        "Whose production the value heuristic counts against a player: the next\n"
        "seat (Catanatron's original) or the mean over all opponents.")
        .value("NEXT", ValueEnemies::next)
        .value("ALL", ValueEnemies::all);

    module.def(
        "value_action", &value_action, nb::arg("game"), nb::arg("weights") = ValueWeights{},
        nb::arg("enemies") = ValueEnemies::next, nb::call_guard<nb::gil_scoped_release>(),
        "The action Catanatron's ValueFunctionPlayer would choose.");
    module.def(
        "value_score",
        [](const Game& game, const PlayerRef& player, const ValueWeights& weights,
           ValueEnemies enemies) {
            return value_score(game, resolve_color(game, player), weights, enemies);
        },
        nb::arg("game"), nb::arg("player"), nb::arg("weights") = ValueWeights{},
        nb::arg("enemies") = ValueEnemies::next,
        "Catanatron's value heuristic for a player by seat index or Color.");
}

}  // namespace cppanatron::python
