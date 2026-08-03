#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "cppanatron/action.hpp"
#include "cppanatron/board.hpp"

namespace cppanatron {

struct PlayerState {
    int victory_points{};
    int actual_victory_points{};
    int roads_available{15};
    int settlements_available{5};
    int cities_available{4};
    bool has_road{};
    bool has_army{};
    bool has_rolled{};
    bool has_played_development_card_in_turn{};
    int longest_road_length{};
    std::array<int, 5> resources{};
    std::array<int, 5> development_cards{};
    std::array<int, 5> played_development_cards{};
    std::array<bool, 4> development_card_owned_at_start{};
    std::vector<int> settlements;
    std::vector<int> cities;
    std::vector<Edge> roads;
    int last_knight_completed_turn{-1};
    int last_development_card_bought_completed_turn{-1};

    auto operator<=>(const PlayerState&) const = default;
};

class Game {
public:
    Game(
        std::vector<Color> colors,
        MapType map_type = MapType::base,
        std::uint64_t seed = 0,
        int discard_limit = 7,
        bool friendly_robber = false,
        int victory_points_to_win = 10,
        NumberPlacement number_placement = NumberPlacement::official_spiral,
        std::optional<std::uint64_t> map_seed = std::nullopt);

    [[nodiscard]] const Board& board() const noexcept { return board_; }
    [[nodiscard]] const std::vector<Color>& colors() const noexcept { return colors_; }
    [[nodiscard]] const std::vector<PlayerState>& players() const noexcept {
        return players_;
    }
    [[nodiscard]] const PlayerState& player(Color color) const;
    [[nodiscard]] PlayerState& player(Color color);
    [[nodiscard]] Color current_color() const noexcept {
        return colors_[static_cast<std::size_t>(current_player_index_)];
    }
    [[nodiscard]] int current_player_index() const noexcept {
        return current_player_index_;
    }
    [[nodiscard]] int current_turn_index() const noexcept { return current_turn_index_; }
    [[nodiscard]] int num_turns() const noexcept { return num_turns_; }
    [[nodiscard]] ActionPrompt current_prompt() const noexcept { return current_prompt_; }
    [[nodiscard]] bool is_initial_build_phase() const noexcept {
        return is_initial_build_phase_;
    }
    [[nodiscard]] bool is_discarding() const noexcept { return is_discarding_; }
    [[nodiscard]] bool is_moving_robber() const noexcept {
        return current_prompt_ == ActionPrompt::move_robber;
    }
    [[nodiscard]] bool is_road_building() const noexcept {
        return is_road_building_;
    }
    [[nodiscard]] int remaining_discards(Color color) const;
    [[nodiscard]] int completed_turns() const noexcept { return completed_turns_; }
    [[nodiscard]] const std::array<int, 5>& resource_bank() const noexcept {
        return resource_bank_;
    }
    [[nodiscard]] std::size_t development_cards_remaining() const noexcept {
        return development_deck_.size();
    }
    [[nodiscard]] const std::vector<DevelopmentCard>& development_deck() const noexcept {
        return development_deck_;
    }
    [[nodiscard]] const std::vector<Action>& playable_actions() const noexcept {
        return playable_actions_;
    }
    [[nodiscard]] std::optional<Color> winning_color() const;

    /**
     * Exact logical-state equality for search deduplication.
     *
     * This deliberately includes hidden state and RNG state. It is stricter
     * than equality of observations, so merging equivalent search branches
     * cannot change any future transition distribution.
     */
    [[nodiscard]] bool search_equivalent(const Game& other) const noexcept;

    void execute(
        const Action& action,
        std::optional<Dice> replay_dice = std::nullopt,
        std::optional<DevelopmentCard> replay_development_card = std::nullopt,
        std::optional<Resource> replay_stolen_resource = std::nullopt);

private:
    [[nodiscard]] int player_index(Color color) const;
    [[nodiscard]] std::vector<Action> generate_playable_actions() const;
    void apply_build_settlement(const Action& action);
    void apply_build_road(const Action& action);
    void apply_build_city(const Action& action);
    void apply_buy_development_card(
        const Action& action,
        std::optional<DevelopmentCard> replay_card);
    void apply_roll(const Action& action, std::optional<Dice> replay_dice);
    void apply_discard(const Action& action);
    void apply_move_robber(
        const Action& action,
        std::optional<Resource> replay_stolen_resource);
    void apply_play_knight(const Action& action);
    void apply_play_year_of_plenty(const Action& action);
    void apply_play_monopoly(const Action& action);
    void apply_play_road_building(const Action& action);
    void apply_maritime_trade(const Action& action);
    void apply_offer_trade(const Action& action);
    void apply_accept_trade(const Action& action);
    void apply_reject_trade(const Action& action);
    void apply_confirm_trade(const Action& action);
    void apply_cancel_trade(const Action& action);
    void apply_end_turn(const Action& action);
    void advance_turn(int direction = 1);
    void yield_resources(int number);
    [[nodiscard]] bool can_afford(Color color, const std::array<int, 5>& cost) const;
    [[nodiscard]] int num_resource_cards(Color color) const;
    [[nodiscard]] bool can_play_development_card(
        Color color,
        DevelopmentCard card) const;
    void consume_development_card(Color color, DevelopmentCard card);
    void update_longest_road_award();
    void update_largest_army_award(Color candidate);
    void reset_trading_state();
    void pay(Color color, const std::array<int, 5>& cost);

    std::vector<Color> colors_;
    std::vector<PlayerState> players_;
    Board board_;
    std::mt19937_64 random_;
    int discard_limit_{};
    bool friendly_robber_{};
    int victory_points_to_win_{};
    std::array<int, 5> resource_bank_{19, 19, 19, 19, 19};
    std::vector<DevelopmentCard> development_deck_;
    int current_player_index_{};
    int current_turn_index_{};
    int num_turns_{};
    int completed_turns_{};
    ActionPrompt current_prompt_{ActionPrompt::build_initial_settlement};
    bool is_initial_build_phase_{true};
    bool is_discarding_{};
    bool is_moving_knight_{};
    bool is_road_building_{};
    int free_roads_available_{};
    bool is_resolving_trade_{};
    DomesticTrade current_trade_{};
    std::vector<bool> acceptees_;
    std::vector<int> discard_counts_;
    std::vector<Action> playable_actions_;
};

}  // namespace cppanatron
