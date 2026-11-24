#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "cppanatron/action.hpp"
#include "cppanatron/board.hpp"
#include "cppanatron/dice.hpp"
#include "cppanatron/rules.hpp"
#include "cppanatron/setup.hpp"

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

/**
 * One seat's hidden cards for Game::redeal_hidden_cards. Hand sizes must not
 * change, and per resource the hands must hold what they held before (the
 * bank stays as it is); development cards come from the deck plus every
 * hand. `development_card_owned_at_start` says which kinds the seat held
 * when its turn began, i.e. may play this turn.
 */
struct HiddenHand {
    std::array<int, 5> resources{};
    std::array<int, 5> development_cards{};
    std::array<bool, 4> development_card_owned_at_start{};
};

class Game {
public:
    /** A base-game game; GameRules sets everything else to the base game's. */
    Game(
        std::vector<Color> colors,
        MapType map_type = MapType::base,
        std::uint64_t seed = 0,
        int discard_limit = 7,
        bool friendly_robber = false,
        int victory_points_to_win = 10,
        NumberPlacement number_placement = NumberPlacement::official_spiral,
        std::optional<std::uint64_t> map_seed = std::nullopt);

    /** A game under explicit rules: bank, deck and pieces included. */
    Game(
        std::vector<Color> colors,
        MapType map_type,
        std::uint64_t seed,
        const GameRules& rules,
        NumberPlacement number_placement = NumberPlacement::official_spiral,
        std::optional<std::uint64_t> map_seed = std::nullopt);

    /**
     * A game on a shared, immutable map. Many games (and their copies) can
     * share one map; `seed` drives dice, cards and steals.
     */
    Game(
        std::vector<Color> colors,
        std::shared_ptr<const CatanMap> map,
        std::uint64_t seed,
        const GameRules& rules = {});

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
    [[nodiscard]] int free_roads_available() const noexcept { return free_roads_available_; }
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
    [[nodiscard]] int victory_points_to_win() const noexcept {
        return victory_points_to_win_;
    }
    [[nodiscard]] int discard_limit() const noexcept { return discard_limit_; }
    /**
     * Whether OpenHex's friendly robber is on. A tile is then shielded when
     * any player with at most 2 public points (buildings and awards, never
     * hidden victory point cards) has a building on it, the mover included,
     * and the robber may not move to a shielded tile. There is no fallback:
     * when every tile but the robber's is shielded the robber stays and
     * nobody is robbed. A roll of 7 then owes no robber move (after the
     * discards, which still happen, the prompt is the roller's play_turn),
     * and a knight is played without a robber prompt (it leaves the hand,
     * counts for Largest Army and uses the turn's development card). A
     * knight's robber is judged on the points after the knight's Largest
     * Army award, like any other: the shields always read the current
     * public points, so the legal set holds no hidden information.
     */
    [[nodiscard]] bool friendly_robber() const noexcept { return friendly_robber_; }
    /**
     * No longer part of the rules: play never sets it, and nothing reads it.
     * OpenHex used to judge a knight's robber on the points before the
     * knight's Largest Army award, and this named the seat the knight took
     * the award from, whose tiles counted 2 more points. The shields now read
     * the points after the award. Always unset, also after apply_setup with
     * GameSetup::largest_army_taken_from (accepted so saved positions load).
     */
    [[nodiscard]] std::optional<int> largest_army_taken_from() const noexcept {
        return largest_army_taken_from_;
    }
    [[nodiscard]] const GameRules& rules() const noexcept { return rules_; }
    [[nodiscard]] DiceMode dice_mode() const noexcept { return rules_.dice_mode; }
    /**
     * The balanced dice between two rolls (DiceMode::balanced). Under random
     * dice it stays a fresh deck and plays no part.
     */
    [[nodiscard]] const BalancedDice& balanced_dice() const noexcept { return dice_; }
    /**
     * The seat that rolls next: the turn owner until it has rolled, then the
     * next seat in turn order.
     */
    [[nodiscard]] int next_roller() const noexcept;
    /**
     * P(next total = 2..12) when the seat `roller` rolls: 2d6 under random
     * dice, the balanced dice's distribution otherwise. Search chance nodes
     * use exactly these.
     */
    [[nodiscard]] std::array<double, 11> roll_probabilities(int roller) const;

    /**
     * Scenario setup: overwrite a player's resource hand.
     *
     * Like Catanatron's player_deck_replenish-style test setup, the bank is not
     * debited. Playable actions are regenerated for the new hand.
     */
    void set_resources(Color color, const std::array<int, 5>& resources);

    /**
     * Scenario setup: overwrite a player's development-card hand.
     *
     * Cards held this way count as owned at the start of the turn, so they are
     * playable immediately. Victory point cards must be bought instead, which
     * keeps actual victory points consistent.
     */
    void set_development_cards(Color color, const std::array<int, 5>& cards);

    /**
     * Information-set determinization: replace a player's hidden development
     * cards with `cards` drawn from the cards that player could be holding.
     *
     * The player's current hand is pooled with the remaining deck, `cards` is
     * dealt from that pool, and the rest becomes the deck in an order shuffled
     * by `seed`. Victory points follow any change in victory point cards, so
     * the deck, hands and scores stay a consistent game. Meant for players
     * other than the one to move: the new cards count as owned at the start
     * of that player's next turn, when they could first be played.
     */
    void determinize_development_cards(
        Color color, const std::array<int, 5>& cards, std::uint64_t seed);

    /**
     * Information-set determinization of every hidden card at once (see
     * HiddenHand): every seat's hand is replaced, the deck becomes the unseen
     * development cards no hand holds, reshuffled, and the game's random
     * stream is reseeded, so nothing hidden in the old state survives.
     */
    void redeal_hidden_cards(const std::vector<HiddenHand>& hands, std::uint64_t seed);

    /**
     * Scenario setup: put a freshly created game into an exact position.
     *
     * See GameSetup. Throws std::invalid_argument for positions the rules
     * cannot produce (buildings too close, unconnected roads, more pieces or
     * cards than exist) and std::logic_error once the game has started. The
     * deck is reshuffled from the game's random stream.
     */
    void apply_setup(const GameSetup& setup);

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
    /** Friendly robber: whether a player with at most 2 public points has a building on `tile`. */
    [[nodiscard]] bool robber_tile_shielded(const Tile& tile) const;
    /** Whether the robber has a tile to go to: always, unless the friendly robber shields them all. */
    [[nodiscard]] bool robber_can_move() const;
    /** After a 7's discards or a knight: the robber prompt, or play_turn when the robber cannot move. */
    void prompt_robber_move();
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
    GameRules rules_;
    /** Public balanced-dice state; changes only under DiceMode::balanced. */
    BalancedDice dice_;
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
    std::optional<int> largest_army_taken_from_;
    bool is_road_building_{};
    int free_roads_available_{};
    bool is_resolving_trade_{};
    DomesticTrade current_trade_{};
    std::vector<bool> acceptees_;
    std::vector<int> discard_counts_;
    std::vector<Action> playable_actions_;
};

}  // namespace cppanatron
