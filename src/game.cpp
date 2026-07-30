#include "cppanatron/game.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace cppanatron {
namespace {

constexpr std::array<int, 5> kRoadCost{1, 1, 0, 0, 0};
constexpr std::array<int, 5> kSettlementCost{1, 1, 1, 1, 0};
constexpr std::array<int, 5> kCityCost{0, 0, 0, 2, 3};

std::size_t resource_index(Resource resource) {
    return static_cast<std::size_t>(resource);
}

}  // namespace

Game::Game(
    std::vector<Color> colors,
    MapType map_type,
    std::uint64_t seed,
    int discard_limit,
    bool friendly_robber,
    int victory_points_to_win,
    NumberPlacement number_placement)
    : colors_(std::move(colors)),
      players_(colors_.size()),
      board_(CatanMap::build(map_type, seed, number_placement)),
      random_(seed),
      discard_limit_(discard_limit),
      friendly_robber_(friendly_robber),
      victory_points_to_win_(victory_points_to_win) {
    if (colors_.empty() || colors_.size() > kColors.size()) {
        throw std::invalid_argument("game requires one to four colors");
    }
    auto unique_colors = colors_;
    std::sort(unique_colors.begin(), unique_colors.end());
    if (std::adjacent_find(unique_colors.begin(), unique_colors.end()) !=
        unique_colors.end()) {
        throw std::invalid_argument("player colors must be unique");
    }
    playable_actions_ = generate_playable_actions();
}

int Game::player_index(Color color) const {
    const auto it = std::find(colors_.begin(), colors_.end(), color);
    if (it == colors_.end()) {
        throw std::out_of_range("color is not in this game");
    }
    return static_cast<int>(std::distance(colors_.begin(), it));
}

const PlayerState& Game::player(Color color) const {
    return players_.at(static_cast<std::size_t>(player_index(color)));
}

PlayerState& Game::player(Color color) {
    return players_.at(static_cast<std::size_t>(player_index(color)));
}

std::optional<Color> Game::winning_color() const {
    std::optional<Color> winner;
    for (std::size_t i = 0; i < colors_.size(); ++i) {
        if (players_[i].actual_victory_points >= victory_points_to_win_) {
            winner = colors_[i];
        }
    }
    return winner;
}

bool Game::can_afford(Color color, const std::array<int, 5>& cost) const {
    const auto& hand = player(color).resources;
    for (std::size_t i = 0; i < hand.size(); ++i) {
        if (hand[i] < cost[i]) {
            return false;
        }
    }
    return true;
}

void Game::pay(Color color, const std::array<int, 5>& cost) {
    auto& hand = player(color).resources;
    for (std::size_t i = 0; i < hand.size(); ++i) {
        hand[i] -= cost[i];
        resource_bank_[i] += cost[i];
    }
}

std::vector<Action> Game::generate_playable_actions() const {
    const Color color = current_color();
    std::vector<Action> result;
    if (current_prompt_ == ActionPrompt::build_initial_settlement) {
        for (int node : board_.buildable_node_ids(color, true)) {
            result.push_back({color, ActionType::build_settlement, node});
        }
        return result;
    }
    if (current_prompt_ == ActionPrompt::build_initial_road) {
        const auto& settlements = player(color).settlements;
        if (settlements.empty()) {
            throw std::logic_error("initial road prompt without a settlement");
        }
        const int last_settlement = settlements.back();
        for (Edge edge : board_.buildable_edges(color)) {
            if (edge.a == last_settlement || edge.b == last_settlement) {
                result.push_back({color, ActionType::build_road, edge});
            }
        }
        return result;
    }
    if (current_prompt_ != ActionPrompt::play_turn) {
        return result;
    }

    const PlayerState& state = player(color);
    if (!state.has_rolled) {
        result.push_back({color, ActionType::roll, std::monostate{}});
        return result;
    }

    result.push_back({color, ActionType::end_turn, std::monostate{}});
    if (state.roads_available > 0 && can_afford(color, kRoadCost)) {
        for (Edge edge : board_.buildable_edges(color)) {
            result.push_back({color, ActionType::build_road, edge});
        }
    }
    if (state.settlements_available > 0 && can_afford(color, kSettlementCost)) {
        for (int node : board_.buildable_node_ids(color)) {
            result.push_back({color, ActionType::build_settlement, node});
        }
    }
    if (state.cities_available > 0 && can_afford(color, kCityCost)) {
        for (int node : state.settlements) {
            result.push_back({color, ActionType::build_city, node});
        }
    }
    return result;
}

void Game::execute(const Action& action, std::optional<Dice> replay_dice) {
    if (std::find(playable_actions_.begin(), playable_actions_.end(), action) ==
        playable_actions_.end()) {
        throw std::invalid_argument("action is not currently playable");
    }
    switch (action.type) {
        case ActionType::build_settlement:
            apply_build_settlement(action);
            break;
        case ActionType::build_road:
            apply_build_road(action);
            break;
        case ActionType::build_city:
            apply_build_city(action);
            break;
        case ActionType::roll:
            apply_roll(action, replay_dice);
            break;
        case ActionType::end_turn:
            apply_end_turn(action);
            break;
        default:
            throw std::logic_error("playable action handler is not implemented");
    }
    playable_actions_ = generate_playable_actions();
}

void Game::apply_build_settlement(const Action& action) {
    const int node_id = std::get<int>(action.value);
    PlayerState& state = player(action.color);
    if (is_initial_build_phase_) {
        board_.build_settlement(action.color, node_id, true);
        state.settlements.push_back(node_id);
        --state.settlements_available;
        ++state.victory_points;
        ++state.actual_victory_points;

        if (state.settlements.size() == 2) {
            for (int tile_index : board_.map().land_tile_indices()) {
                const Tile& tile =
                    board_.map().tiles().at(static_cast<std::size_t>(tile_index));
                if (!tile.resource.has_value() ||
                    std::find(tile.nodes.begin(), tile.nodes.end(), node_id) ==
                        tile.nodes.end()) {
                    continue;
                }
                const std::size_t index = resource_index(*tile.resource);
                if (resource_bank_[index] <= 0) {
                    throw std::logic_error("resource bank depleted during setup");
                }
                --resource_bank_[index];
                ++state.resources[index];
            }
        }
        current_prompt_ = ActionPrompt::build_initial_road;
        return;
    }

    board_.build_settlement(action.color, node_id);
    state.settlements.push_back(node_id);
    --state.settlements_available;
    ++state.victory_points;
    ++state.actual_victory_points;
    pay(action.color, kSettlementCost);
}

void Game::apply_build_road(const Action& action) {
    const Edge edge = std::get<Edge>(action.value).normalized();
    PlayerState& state = player(action.color);
    board_.build_road(action.color, edge);
    state.roads.push_back(edge);
    --state.roads_available;

    if (is_initial_build_phase_) {
        const int num_buildings = std::accumulate(
            players_.begin(),
            players_.end(),
            0,
            [](int sum, const PlayerState& candidate) {
                return sum + static_cast<int>(candidate.settlements.size());
            });
        const int num_players = static_cast<int>(players_.size());
        if (num_buildings < num_players) {
            advance_turn();
            current_prompt_ = ActionPrompt::build_initial_settlement;
        } else if (num_buildings == num_players) {
            current_prompt_ = ActionPrompt::build_initial_settlement;
        } else if (num_buildings == 2 * num_players) {
            is_initial_build_phase_ = false;
            current_prompt_ = ActionPrompt::play_turn;
        } else {
            advance_turn(-1);
            current_prompt_ = ActionPrompt::build_initial_settlement;
        }
        return;
    }

    pay(action.color, kRoadCost);
    state.longest_road_length = board_.longest_road(action.color);
    // Longest-road ownership transfer is added with full award parity; keeping
    // the exact length now makes observations and later award logic stable.
}

void Game::apply_build_city(const Action& action) {
    const int node_id = std::get<int>(action.value);
    PlayerState& state = player(action.color);
    board_.build_city(action.color, node_id);
    const auto settlement = std::find(state.settlements.begin(), state.settlements.end(), node_id);
    if (settlement == state.settlements.end()) {
        throw std::logic_error("board/player building caches disagree");
    }
    state.settlements.erase(settlement);
    state.cities.push_back(node_id);
    ++state.settlements_available;
    --state.cities_available;
    ++state.victory_points;
    ++state.actual_victory_points;
    pay(action.color, kCityCost);
}

void Game::yield_resources(int number) {
    std::vector<std::array<int, 5>> payouts(players_.size());
    std::array<int, 5> totals{};
    for (const auto& [coordinate, tile_index] : board_.map().coordinate_to_tile()) {
        const Tile& tile =
            board_.map().tiles().at(static_cast<std::size_t>(tile_index));
        if (tile.kind != TileKind::land || tile.number != number ||
            coordinate == board_.robber_coordinate() || !tile.resource.has_value()) {
            continue;
        }
        const std::size_t resource = resource_index(*tile.resource);
        for (int node : tile.nodes) {
            const auto building = board_.buildings().find(node);
            if (building == board_.buildings().end()) {
                continue;
            }
            const int count =
                building->second.building == Building::settlement ? 1 : 2;
            payouts[static_cast<std::size_t>(player_index(building->second.color))][resource] +=
                count;
            totals[resource] += count;
        }
    }
    for (std::size_t resource = 0; resource < totals.size(); ++resource) {
        if (totals[resource] > resource_bank_[resource]) {
            continue;
        }
        for (std::size_t player_id = 0; player_id < players_.size(); ++player_id) {
            players_[player_id].resources[resource] += payouts[player_id][resource];
            resource_bank_[resource] -= payouts[player_id][resource];
        }
    }
}

void Game::apply_roll(const Action& action, std::optional<Dice> replay_dice) {
    std::uniform_int_distribution<int> die(1, 6);
    const Dice dice = replay_dice.value_or(Dice{die(random_), die(random_)});
    const int number = dice.first + dice.second;
    if (number == 7) {
        // Keeping seven unavailable is safer than producing a legal-action set
        // with incomplete discard/robber semantics.
        throw std::logic_error("discard and robber transition is not implemented yet");
    }
    PlayerState& state = player(action.color);
    state.has_rolled = true;
    yield_resources(number);
    current_prompt_ = ActionPrompt::play_turn;
}

void Game::apply_end_turn(const Action& action) {
    PlayerState& state = player(action.color);
    state.has_played_development_card_in_turn = false;
    state.has_rolled = false;
    for (std::size_t i = 0; i < state.development_card_owned_at_start.size(); ++i) {
        state.development_card_owned_at_start[i] = state.development_cards[i] > 0;
    }
    advance_turn();
    current_prompt_ = ActionPrompt::play_turn;
}

void Game::advance_turn(int direction) {
    const int count = static_cast<int>(colors_.size());
    current_player_index_ = (current_player_index_ + direction + count) % count;
    current_turn_index_ = current_player_index_;
    ++num_turns_;
}

}  // namespace cppanatron
