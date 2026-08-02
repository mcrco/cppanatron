#include "cppanatron/game.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace cppanatron {
namespace {

constexpr std::array<int, 5> kRoadCost{1, 1, 0, 0, 0};
constexpr std::array<int, 5> kSettlementCost{1, 1, 1, 1, 0};
constexpr std::array<int, 5> kCityCost{0, 0, 0, 2, 3};
constexpr std::array<int, 5> kDevelopmentCardCost{0, 0, 1, 1, 1};

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
    NumberPlacement number_placement,
    std::optional<std::uint64_t> map_seed)
    : colors_(std::move(colors)),
      players_(colors_.size()),
      board_(CatanMap::build(
          map_type,
          map_seed.value_or(seed),
          number_placement)),
      random_(seed),
      discard_limit_(discard_limit),
      friendly_robber_(friendly_robber),
      victory_points_to_win_(victory_points_to_win),
      acceptees_(colors_.size()),
      discard_counts_(colors_.size()) {
    if (colors_.empty() || colors_.size() > kColors.size()) {
        throw std::invalid_argument("game requires one to four colors");
    }
    auto unique_colors = colors_;
    std::sort(unique_colors.begin(), unique_colors.end());
    if (std::adjacent_find(unique_colors.begin(), unique_colors.end()) !=
        unique_colors.end()) {
        throw std::invalid_argument("player colors must be unique");
    }
    development_deck_.insert(
        development_deck_.end(), 14, DevelopmentCard::knight);
    development_deck_.insert(
        development_deck_.end(), 2, DevelopmentCard::year_of_plenty);
    development_deck_.insert(
        development_deck_.end(), 2, DevelopmentCard::road_building);
    development_deck_.insert(
        development_deck_.end(), 2, DevelopmentCard::monopoly);
    development_deck_.insert(
        development_deck_.end(), 5, DevelopmentCard::victory_point);
    std::shuffle(development_deck_.begin(), development_deck_.end(), random_);
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

bool Game::search_equivalent(const Game& other) const noexcept {
    return colors_ == other.colors_ && players_ == other.players_ &&
           board_.buildings() == other.board_.buildings() &&
           board_.roads() == other.board_.roads() &&
           board_.robber_coordinate() == other.board_.robber_coordinate() &&
           board_.longest_road_color() == other.board_.longest_road_color() &&
           random_ == other.random_ && discard_limit_ == other.discard_limit_ &&
           friendly_robber_ == other.friendly_robber_ &&
           victory_points_to_win_ == other.victory_points_to_win_ &&
           resource_bank_ == other.resource_bank_ &&
           development_deck_ == other.development_deck_ &&
           current_player_index_ == other.current_player_index_ &&
           current_turn_index_ == other.current_turn_index_ &&
           num_turns_ == other.num_turns_ && completed_turns_ == other.completed_turns_ &&
           current_prompt_ == other.current_prompt_ &&
           is_initial_build_phase_ == other.is_initial_build_phase_ &&
           is_discarding_ == other.is_discarding_ &&
           is_moving_knight_ == other.is_moving_knight_ &&
           is_road_building_ == other.is_road_building_ &&
           free_roads_available_ == other.free_roads_available_ &&
           is_resolving_trade_ == other.is_resolving_trade_ &&
           current_trade_ == other.current_trade_ && acceptees_ == other.acceptees_ &&
           discard_counts_ == other.discard_counts_ &&
           playable_actions_ == other.playable_actions_;
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

int Game::num_resource_cards(Color color) const {
    const auto& resources = player(color).resources;
    return std::accumulate(resources.begin(), resources.end(), 0);
}

bool Game::can_play_development_card(
    Color color,
    DevelopmentCard card) const {
    if (card == DevelopmentCard::victory_point) {
        return false;
    }
    const PlayerState& state = player(color);
    const std::size_t index = static_cast<std::size_t>(card);
    return !state.has_played_development_card_in_turn &&
           state.development_cards[index] > 0 &&
           state.development_card_owned_at_start[index];
}

void Game::consume_development_card(Color color, DevelopmentCard card) {
    if (!can_play_development_card(color, card)) {
        throw std::logic_error("development card is not playable");
    }
    PlayerState& state = player(color);
    const std::size_t index = static_cast<std::size_t>(card);
    --state.development_cards[index];
    ++state.played_development_cards[index];
    state.has_played_development_card_in_turn = true;
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
        if (current_prompt_ == ActionPrompt::decide_trade) {
            result.push_back(
                {color, ActionType::reject_trade, current_trade_});
            if (can_afford(color, current_trade_.asking)) {
                result.push_back(
                    {color, ActionType::accept_trade, current_trade_});
            }
            return result;
        }
        if (current_prompt_ == ActionPrompt::decide_acceptees) {
            result.push_back(
                {color, ActionType::cancel_trade, std::monostate{}});
            for (std::size_t i = 0; i < acceptees_.size(); ++i) {
                if (acceptees_[i]) {
                    result.push_back(
                        {color,
                         ActionType::confirm_trade,
                         ConfirmedTrade{current_trade_, colors_[i]}});
                }
            }
            return result;
        }
        if (current_prompt_ == ActionPrompt::discard) {
            const auto& hand = player(color).resources;
            for (std::size_t i = 0; i < hand.size(); ++i) {
                if (hand[i] > 0) {
                    result.push_back(
                        {color, ActionType::discard_resource, kResources[i]});
                }
            }
            return result;
        }
        if (current_prompt_ == ActionPrompt::move_robber) {
            std::vector<Action> unfiltered;
            for (const auto& [coordinate, tile_index] :
                 board_.map().coordinate_to_tile()) {
                const Tile& tile =
                    board_.map().tiles().at(static_cast<std::size_t>(tile_index));
                if (tile.kind != TileKind::land ||
                    coordinate == board_.robber_coordinate()) {
                    continue;
                }
                std::vector<Color> victims;
                for (int node : tile.nodes) {
                    const auto building = board_.buildings().find(node);
                    if (building == board_.buildings().end() ||
                        building->second.color == color ||
                        num_resource_cards(building->second.color) == 0) {
                        continue;
                    }
                    if (std::find(
                            victims.begin(),
                            victims.end(),
                            building->second.color) == victims.end()) {
                        victims.push_back(building->second.color);
                    }
                }
                if (victims.empty()) {
                    unfiltered.push_back(
                        {color,
                         ActionType::move_robber,
                         RobberMove{coordinate, std::nullopt}});
                } else {
                    for (Color victim : victims) {
                        unfiltered.push_back(
                            {color,
                             ActionType::move_robber,
                             RobberMove{coordinate, victim}});
                    }
                }
            }
            if (!friendly_robber_) {
                return unfiltered;
            }
            for (const Action& action : unfiltered) {
                const RobberMove& move = std::get<RobberMove>(action.value);
                const Tile& tile = board_.map().tile_at(move.coordinate);
                const bool blocks_low_vp_enemy = std::any_of(
                    tile.nodes.begin(),
                    tile.nodes.end(),
                    [&](int node) {
                        const auto building = board_.buildings().find(node);
                        return building != board_.buildings().end() &&
                               building->second.color != color &&
                               player(building->second.color).actual_victory_points < 3;
                    });
                if (!blocks_low_vp_enemy) {
                    result.push_back(action);
                }
            }
            return result.empty() ? unfiltered : result;
        }
        return result;
    }

    const PlayerState& state = player(color);
    if (is_road_building_) {
        if (state.roads_available > 0) {
            for (Edge edge : board_.buildable_edges(color)) {
                result.push_back({color, ActionType::build_road, edge});
            }
        }
        return result;
    }

    if (can_play_development_card(color, DevelopmentCard::year_of_plenty)) {
        std::set<std::vector<Resource>> options;
        for (std::size_t first = 0; first < kResources.size(); ++first) {
            for (std::size_t second = first; second < kResources.size(); ++second) {
                const bool pair_available =
                    first == second
                        ? resource_bank_[first] >= 2
                        : (resource_bank_[first] >= 1 &&
                           resource_bank_[second] >= 1);
                if (pair_available) {
                    options.insert({kResources[first], kResources[second]});
                } else {
                    if (resource_bank_[first] > 0) {
                        options.insert({kResources[first]});
                    }
                    if (resource_bank_[second] > 0) {
                        options.insert({kResources[second]});
                    }
                }
            }
        }
        for (const auto& cards : options) {
            result.push_back(
                {color, ActionType::play_year_of_plenty, cards});
        }
    }
    if (can_play_development_card(color, DevelopmentCard::monopoly)) {
        for (Resource resource : kResources) {
            result.push_back({color, ActionType::play_monopoly, resource});
        }
    }
    if (can_play_development_card(color, DevelopmentCard::knight)) {
        result.push_back(
            {color, ActionType::play_knight_card, std::monostate{}});
    }
    if (can_play_development_card(color, DevelopmentCard::road_building) &&
        state.roads_available > 0 && !board_.buildable_edges(color).empty()) {
        result.push_back(
            {color, ActionType::play_road_building, std::monostate{}});
    }

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
    if (!development_deck_.empty() &&
        can_afford(color, kDevelopmentCardCost)) {
        result.push_back(
            {color, ActionType::buy_development_card, std::monostate{}});
    }

    std::array<int, 5> rates{4, 4, 4, 4, 4};
    const auto ports = board_.player_port_resources(color);
    if (ports.contains(std::nullopt)) {
        rates.fill(3);
    }
    for (const auto port : ports) {
        if (port.has_value()) {
            rates[resource_index(*port)] = 2;
        }
    }
    for (std::size_t offered = 0; offered < kResources.size(); ++offered) {
        if (state.resources[offered] < rates[offered]) {
            continue;
        }
        for (std::size_t received = 0; received < kResources.size(); ++received) {
            if (offered == received || resource_bank_[received] <= 0) {
                continue;
            }
            MaritimeTrade trade;
            for (int i = 0; i < rates[offered]; ++i) {
                trade.cards[static_cast<std::size_t>(i)] = kResources[offered];
            }
            trade.cards[4] = kResources[received];
            result.push_back({color, ActionType::maritime_trade, trade});
        }
    }
    return result;
}

void Game::execute(
    const Action& action,
    std::optional<Dice> replay_dice,
    std::optional<DevelopmentCard> replay_development_card,
    std::optional<Resource> replay_stolen_resource) {
    const bool is_offer =
        action.type == ActionType::offer_trade &&
        action.color == current_color() &&
        current_prompt_ == ActionPrompt::play_turn &&
        player(action.color).has_rolled;
    bool valid_offer = false;
    if (is_offer) {
        const DomesticTrade& trade = std::get<DomesticTrade>(action.value);
        const int offered =
            std::accumulate(trade.offering.begin(), trade.offering.end(), 0);
        const int asked =
            std::accumulate(trade.asking.begin(), trade.asking.end(), 0);
        valid_offer = offered > 0 && asked > 0;
        for (std::size_t i = 0; i < trade.offering.size(); ++i) {
            valid_offer =
                valid_offer &&
                !(trade.offering[i] > 0 && trade.asking[i] > 0);
        }
    }
    if (!valid_offer &&
        std::find(playable_actions_.begin(), playable_actions_.end(), action) ==
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
        case ActionType::buy_development_card:
            apply_buy_development_card(action, replay_development_card);
            break;
        case ActionType::roll:
            apply_roll(action, replay_dice);
            break;
        case ActionType::discard_resource:
            apply_discard(action);
            break;
        case ActionType::move_robber:
            apply_move_robber(action, replay_stolen_resource);
            break;
        case ActionType::play_knight_card:
            apply_play_knight(action);
            break;
        case ActionType::play_year_of_plenty:
            apply_play_year_of_plenty(action);
            break;
        case ActionType::play_monopoly:
            apply_play_monopoly(action);
            break;
        case ActionType::play_road_building:
            apply_play_road_building(action);
            break;
        case ActionType::maritime_trade:
            apply_maritime_trade(action);
            break;
        case ActionType::offer_trade:
            apply_offer_trade(action);
            break;
        case ActionType::accept_trade:
            apply_accept_trade(action);
            break;
        case ActionType::reject_trade:
            apply_reject_trade(action);
            break;
        case ActionType::confirm_trade:
            apply_confirm_trade(action);
            break;
        case ActionType::cancel_trade:
            apply_cancel_trade(action);
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
    update_longest_road_award();
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

    if (is_road_building_ && free_roads_available_ > 0) {
        --free_roads_available_;
        if (free_roads_available_ == 0 || state.roads_available == 0 ||
            board_.buildable_edges(action.color).empty()) {
            is_road_building_ = false;
            free_roads_available_ = 0;
        }
    } else {
        pay(action.color, kRoadCost);
    }
    update_longest_road_award();
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

void Game::apply_buy_development_card(
    const Action& action,
    std::optional<DevelopmentCard> replay_card) {
    if (development_deck_.empty() ||
        !can_afford(action.color, kDevelopmentCardCost)) {
        throw std::logic_error("development card cannot be bought");
    }
    DevelopmentCard card;
    if (replay_card.has_value()) {
        const auto it = std::find(
            development_deck_.begin(), development_deck_.end(), *replay_card);
        if (it == development_deck_.end()) {
            throw std::logic_error("replay development card is absent");
        }
        card = *it;
        development_deck_.erase(it);
    } else {
        card = development_deck_.back();
        development_deck_.pop_back();
    }
    PlayerState& state = player(action.color);
    ++state.development_cards[static_cast<std::size_t>(card)];
    state.last_development_card_bought_completed_turn = completed_turns_;
    if (card == DevelopmentCard::victory_point) {
        ++state.actual_victory_points;
    }
    pay(action.color, kDevelopmentCardCost);
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
    PlayerState& state = player(action.color);
    state.has_rolled = true;
    if (number == 7) {
        int first_discarder = -1;
        for (std::size_t i = 0; i < colors_.size(); ++i) {
            const int cards = num_resource_cards(colors_[i]);
            discard_counts_[i] = cards > discard_limit_ ? cards / 2 : 0;
            if (discard_counts_[i] > 0 && first_discarder < 0) {
                first_discarder = static_cast<int>(i);
            }
        }
        if (first_discarder >= 0) {
            current_player_index_ = first_discarder;
            current_prompt_ = ActionPrompt::discard;
            is_discarding_ = true;
        } else {
            std::fill(discard_counts_.begin(), discard_counts_.end(), 0);
            current_prompt_ = ActionPrompt::move_robber;
            is_moving_knight_ = true;
        }
        return;
    }
    yield_resources(number);
    current_prompt_ = ActionPrompt::play_turn;
}

void Game::apply_discard(const Action& action) {
    const int index = player_index(action.color);
    if (discard_counts_.at(static_cast<std::size_t>(index)) <= 0) {
        throw std::logic_error("player is not required to discard");
    }
    const std::size_t resource = resource_index(std::get<Resource>(action.value));
    PlayerState& state = player(action.color);
    if (state.resources[resource] <= 0) {
        throw std::logic_error("cannot discard an absent resource");
    }
    --state.resources[resource];
    ++resource_bank_[resource];
    --discard_counts_[static_cast<std::size_t>(index)];
    if (discard_counts_[static_cast<std::size_t>(index)] > 0) {
        return;
    }

    int next_discarder = -1;
    for (int i = index + 1; i < static_cast<int>(colors_.size()); ++i) {
        if (discard_counts_[static_cast<std::size_t>(i)] > 0) {
            next_discarder = i;
            break;
        }
    }
    if (next_discarder >= 0) {
        current_player_index_ = next_discarder;
        return;
    }
    current_player_index_ = current_turn_index_;
    current_prompt_ = ActionPrompt::move_robber;
    is_discarding_ = false;
    is_moving_knight_ = true;
    std::fill(discard_counts_.begin(), discard_counts_.end(), 0);
}

void Game::apply_move_robber(
    const Action& action,
    std::optional<Resource> replay_stolen_resource) {
    const RobberMove& move = std::get<RobberMove>(action.value);
    if (move.victim.has_value()) {
        PlayerState& victim = player(*move.victim);
        std::vector<Resource> cards;
        for (std::size_t i = 0; i < victim.resources.size(); ++i) {
            cards.insert(
                cards.end(),
                static_cast<std::size_t>(victim.resources[i]),
                kResources[i]);
        }
        if (cards.empty()) {
            throw std::logic_error("robber victim has no resources");
        }
        Resource stolen;
        if (replay_stolen_resource.has_value()) {
            if (victim.resources[resource_index(*replay_stolen_resource)] <= 0) {
                throw std::logic_error("replay stolen resource is absent");
            }
            stolen = *replay_stolen_resource;
        } else {
            std::uniform_int_distribution<std::size_t> choose(0, cards.size() - 1);
            stolen = cards[choose(random_)];
        }
        const std::size_t resource = resource_index(stolen);
        --victim.resources[resource];
        ++player(action.color).resources[resource];
    }
    board_.move_robber(move.coordinate);
    current_prompt_ = ActionPrompt::play_turn;
    is_moving_knight_ = false;
}

void Game::apply_play_knight(const Action& action) {
    consume_development_card(action.color, DevelopmentCard::knight);
    player(action.color).last_knight_completed_turn = completed_turns_;
    update_largest_army_award(action.color);
    current_prompt_ = ActionPrompt::move_robber;
    is_moving_knight_ = true;
}

void Game::apply_play_year_of_plenty(const Action& action) {
    const auto& cards = std::get<std::vector<Resource>>(action.value);
    if (cards.empty() || cards.size() > 2) {
        throw std::logic_error("year of plenty must draw one or two resources");
    }
    std::array<int, 5> requested{};
    for (Resource card : cards) {
        ++requested[resource_index(card)];
    }
    for (std::size_t i = 0; i < requested.size(); ++i) {
        if (requested[i] > resource_bank_[i]) {
            throw std::logic_error("bank cannot fulfill year of plenty");
        }
    }
    consume_development_card(action.color, DevelopmentCard::year_of_plenty);
    PlayerState& state = player(action.color);
    for (std::size_t i = 0; i < requested.size(); ++i) {
        state.resources[i] += requested[i];
        resource_bank_[i] -= requested[i];
    }
    current_prompt_ = ActionPrompt::play_turn;
}

void Game::apply_play_monopoly(const Action& action) {
    const Resource card = std::get<Resource>(action.value);
    consume_development_card(action.color, DevelopmentCard::monopoly);
    const std::size_t resource = resource_index(card);
    int stolen = 0;
    for (Color color : colors_) {
        if (color == action.color) {
            continue;
        }
        PlayerState& victim = player(color);
        stolen += victim.resources[resource];
        victim.resources[resource] = 0;
    }
    player(action.color).resources[resource] += stolen;
    current_prompt_ = ActionPrompt::play_turn;
}

void Game::apply_play_road_building(const Action& action) {
    consume_development_card(action.color, DevelopmentCard::road_building);
    is_road_building_ = true;
    free_roads_available_ = 2;
    current_prompt_ = ActionPrompt::play_turn;
}

void Game::apply_maritime_trade(const Action& action) {
    const MaritimeTrade& trade = std::get<MaritimeTrade>(action.value);
    if (!trade.cards[4].has_value()) {
        throw std::logic_error("maritime trade must request a resource");
    }
    std::array<int, 5> offered{};
    for (std::size_t i = 0; i < 4; ++i) {
        if (trade.cards[i].has_value()) {
            ++offered[resource_index(*trade.cards[i])];
        }
    }
    const std::size_t received = resource_index(*trade.cards[4]);
    if (!can_afford(action.color, offered) || resource_bank_[received] <= 0) {
        throw std::logic_error("maritime trade cannot be fulfilled");
    }
    pay(action.color, offered);
    --resource_bank_[received];
    ++player(action.color).resources[received];
    current_prompt_ = ActionPrompt::play_turn;
}

void Game::apply_offer_trade(const Action& action) {
    current_trade_ = std::get<DomesticTrade>(action.value);
    current_trade_.offering_player_index = current_turn_index_;
    is_resolving_trade_ = true;
    const auto it = std::find_if(
        colors_.begin(),
        colors_.end(),
        [&](Color color) { return color != action.color; });
    current_player_index_ = static_cast<int>(std::distance(colors_.begin(), it));
    current_prompt_ = ActionPrompt::decide_trade;
}

void Game::apply_accept_trade(const Action& action) {
    const int index = player_index(action.color);
    acceptees_[static_cast<std::size_t>(index)] = true;
    int next = -1;
    for (int i = current_player_index_ + 1;
         i < static_cast<int>(colors_.size());
         ++i) {
        if (colors_[static_cast<std::size_t>(i)] != action.color) {
            next = i;
            break;
        }
    }
    if (next >= 0) {
        current_player_index_ = next;
    } else {
        current_player_index_ = current_turn_index_;
        current_prompt_ = ActionPrompt::decide_acceptees;
    }
}

void Game::apply_reject_trade(const Action& action) {
    int next = -1;
    for (int i = current_player_index_ + 1;
         i < static_cast<int>(colors_.size());
         ++i) {
        if (colors_[static_cast<std::size_t>(i)] != action.color) {
            next = i;
            break;
        }
    }
    if (next >= 0) {
        current_player_index_ = next;
        return;
    }
    current_player_index_ = current_turn_index_;
    if (std::none_of(acceptees_.begin(), acceptees_.end(), [](bool value) {
            return value;
        })) {
        reset_trading_state();
        current_prompt_ = ActionPrompt::play_turn;
    } else {
        current_prompt_ = ActionPrompt::decide_acceptees;
    }
}

void Game::apply_confirm_trade(const Action& action) {
    const ConfirmedTrade& confirmed = std::get<ConfirmedTrade>(action.value);
    PlayerState& offering = player(action.color);
    PlayerState& partner = player(confirmed.partner);
    for (std::size_t i = 0; i < offering.resources.size(); ++i) {
        offering.resources[i] -= confirmed.trade.offering[i];
        offering.resources[i] += confirmed.trade.asking[i];
        partner.resources[i] -= confirmed.trade.asking[i];
        partner.resources[i] += confirmed.trade.offering[i];
    }
    reset_trading_state();
    current_player_index_ = current_turn_index_;
    current_prompt_ = ActionPrompt::play_turn;
}

void Game::apply_cancel_trade(const Action&) {
    reset_trading_state();
    current_player_index_ = current_turn_index_;
    current_prompt_ = ActionPrompt::play_turn;
}

void Game::reset_trading_state() {
    is_resolving_trade_ = false;
    current_trade_ = DomesticTrade{};
    std::fill(acceptees_.begin(), acceptees_.end(), false);
}

void Game::update_longest_road_award() {
    std::optional<int> previous_owner;
    for (std::size_t i = 0; i < players_.size(); ++i) {
        players_[i].longest_road_length = board_.longest_road(colors_[i]);
        if (players_[i].has_road) {
            previous_owner = static_cast<int>(i);
        }
    }
    const auto winner_color = board_.longest_road_color();
    const std::optional<int> winner =
        winner_color.has_value()
            ? std::optional<int>(player_index(*winner_color))
            : std::nullopt;
    if (winner == previous_owner) {
        return;
    }
    if (previous_owner.has_value()) {
        PlayerState& previous = players_[static_cast<std::size_t>(*previous_owner)];
        previous.has_road = false;
        previous.victory_points -= 2;
        previous.actual_victory_points -= 2;
    }
    if (winner.has_value()) {
        PlayerState& next = players_[static_cast<std::size_t>(*winner)];
        next.has_road = true;
        next.victory_points += 2;
        next.actual_victory_points += 2;
    }
}

void Game::update_largest_army_award(Color candidate_color) {
    const int candidate_index = player_index(candidate_color);
    PlayerState& candidate = players_[static_cast<std::size_t>(candidate_index)];
    const int candidate_size = candidate.played_development_cards[
        static_cast<std::size_t>(DevelopmentCard::knight)];
    if (candidate_size < 3) {
        return;
    }
    std::optional<int> previous_owner;
    for (std::size_t i = 0; i < players_.size(); ++i) {
        if (players_[i].has_army) {
            previous_owner = static_cast<int>(i);
            break;
        }
    }
    if (!previous_owner.has_value()) {
        candidate.has_army = true;
        candidate.victory_points += 2;
        candidate.actual_victory_points += 2;
        return;
    }
    if (*previous_owner == candidate_index) {
        return;
    }
    const PlayerState& previous =
        players_[static_cast<std::size_t>(*previous_owner)];
    const int previous_size = previous.played_development_cards[
        static_cast<std::size_t>(DevelopmentCard::knight)];
    if (candidate_size <= previous_size) {
        return;
    }
    PlayerState& mutable_previous =
        players_[static_cast<std::size_t>(*previous_owner)];
    mutable_previous.has_army = false;
    mutable_previous.victory_points -= 2;
    mutable_previous.actual_victory_points -= 2;
    candidate.has_army = true;
    candidate.victory_points += 2;
    candidate.actual_victory_points += 2;
}

void Game::apply_end_turn(const Action& action) {
    PlayerState& state = player(action.color);
    state.has_played_development_card_in_turn = false;
    state.has_rolled = false;
    for (std::size_t i = 0; i < state.development_card_owned_at_start.size(); ++i) {
        state.development_card_owned_at_start[i] = state.development_cards[i] > 0;
    }
    ++completed_turns_;
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
