// Balanced dice (OpenHex's 1v1 dice): the state's odds on hand-computed
// cases, its long-run behaviour, games under DiceMode::balanced, and searches
// whose chance nodes follow the state's own odds. tests/test_dice_reference.py
// in catanrl checks the odds against an exact reference over many states.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cppanatron/dice.hpp"
#include "cppanatron/flat_codec.hpp"
#include "cppanatron/game.hpp"
#include "cppanatron/mcts.hpp"
#include "cppanatron/search_pool.hpp"

namespace {

using namespace cppanatron;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void require_invalid(Callable callable, const std::string& message) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

/** P(total) as an exact fraction, compared to one rounding. */
void require_probability(double actual, std::int64_t numerator, std::int64_t denominator,
                         const std::string& what) {
    const double expected = static_cast<double>(numerator) / static_cast<double>(denominator);
    require(std::abs(actual - expected) <= 1e-15, what + ": " + std::to_string(actual) + " != " +
                                                      std::to_string(numerator) + "/" +
                                                      std::to_string(denominator));
}

constexpr std::array<int, 11> kWays{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};

std::vector<Color> seats(int n) {
    return {kColors.begin(), kColors.begin() + n};
}

GameRules balanced_rules() {
    GameRules rules;
    rules.dice_mode = DiceMode::balanced;
    return rules;
}

void test_a_fresh_deck_is_two_fair_dice() {
    const BalancedDice dice = BalancedDice::fresh(2);
    require(dice.remaining() == 36 && dice.drawable() == BalancedDice::kFullDeck, "full deck");
    for (int roller : {-1, 0, 1}) {
        const auto probabilities = dice.sum_probabilities(roller);
        for (std::size_t i = 0; i < 11; ++i) {
            require_probability(probabilities[i], kWays[i], 36, "fresh deck");
        }
    }
    require(dice.counts() == kWays, "cards per total");
    require(dice.card_for_total(7) == (Dice{1, 6}) && dice.card_for_total(12) == (Dice{6, 6}),
            "first card of a total, faces ascending");
}

void test_hand_computed_odds() {
    // Seat 0 rolls (3, 4): seat 1's 7 gets the streak (+0.4) and the window (0.66).
    BalancedDice dice = BalancedDice::fresh(2);
    dice.apply(Dice{3, 4}, 0);
    require(dice.remaining() == 35 && dice.recent_count == 1 && dice.recent[0] == 7 &&
                dice.sevens[0] == 1 && dice.streak_player == 0 && dice.streak_count == 1,
            "state after an opening 7");
    require(dice.seven_multiplier(1) == std::pair<std::int64_t, std::int64_t>{7, 5}, "x 7/5");
    const auto other = dice.sum_probabilities(1);
    require_probability(other[5], 231, 1731, "the other seat's 7");
    require_probability(other[0], 50, 1731, "the other seat's 2");
    require_probability(other[4], 250, 1731, "the other seat's 6");
    // The streak's owner rolling again (not a turn order a game produces): x 3/5.
    require_probability(dice.sum_probabilities(0)[5], 99, 1599, "the owner's 7");

    // Imbalance once every seat could have a 7: seat 0 holds 3 of 4.
    BalancedDice uneven = BalancedDice::fresh(2);
    uneven.sevens = {3, 1};
    require_probability(uneven.sum_probabilities(0)[5], 3, 33, "imbalance against");
    require_probability(uneven.sum_probabilities(0)[6], 5, 33, "imbalance against, an 8");
    require_probability(uneven.sum_probabilities(1)[5], 9, 39, "imbalance for");
    BalancedDice few = BalancedDice::fresh(2);
    few.sevens = {1, 0};
    require(few.seven_multiplier(0).first == few.seven_multiplier(0).second,
            "fewer 7s than seats: no imbalance term");
    BalancedDice four = BalancedDice::fresh(4);
    four.sevens = {2, 1, 1, 0};
    require(four.seven_multiplier(0).first == 0, "four seats, half the 7s: x 0");
    require(four.seven_multiplier(3).first == 2 * four.seven_multiplier(3).second,
            "four seats, no 7s: x 2");

    // Clamped to [0, 2]: no 7s of the roller's and the other seat's streak, 2.4 -> 2.
    BalancedDice high = BalancedDice::fresh(2);
    high.sevens = {0, 2};
    high.streak_player = 1;
    high.streak_count = 1;
    require(high.seven_multiplier(0).first == 2 * high.seven_multiplier(0).second, "clamped to 2");
    require_probability(high.sum_probabilities(0)[5], 12, 42, "clamped to 2, P(7)");
    // Every 7 the roller's and its own streak: -0.4 -> 0.
    BalancedDice low = BalancedDice::fresh(2);
    low.sevens = {2, 0};
    low.streak_player = 0;
    low.streak_count = 1;
    require(low.seven_multiplier(0).first == 0 && low.sum_probabilities(0)[5] == 0.0,
            "clamped to 0");

    // Five rolls: recency (7 twice, 6 twice, 8 once), imbalance x 2 for seat 1.
    BalancedDice window = BalancedDice::fresh(2);
    const std::array<std::pair<Dice, int>, 5> rolls{{
        {{1, 6}, 0}, {{1, 5}, 1}, {{2, 5}, 0}, {{2, 6}, 1}, {{2, 4}, 0}}};
    for (const auto& [card, roller] : rolls) {
        window.apply(card, roller);
    }
    const std::array<std::int64_t, 11> numerators{25, 50, 75, 100, 24, 64, 66, 100, 75, 50, 25};
    const auto probabilities = window.sum_probabilities(1);
    for (std::size_t i = 0; i < 11; ++i) {
        require_probability(probabilities[i], numerators[i], 654, "window, seat 1");
    }
    // A sixth roll drops the oldest total: 6, 7, 8, 6, 7.
    window.apply(Dice{3, 4}, 1);
    require(window.recent_count == 5 && window.recent == std::array<std::int8_t, 5>{6, 7, 8, 6, 7},
            "the window keeps the last five");
    // Three appearances zero a total: 7, 4, 7, 6, 7.
    BalancedDice thrice = BalancedDice::fresh(2);
    for (const auto& [card, roller] : std::array<std::pair<Dice, int>, 5>{{
             {{1, 6}, 0}, {{2, 2}, 1}, {{2, 5}, 0}, {{3, 3}, 1}, {{3, 4}, 0}}}) {
        thrice.apply(card, roller);
    }
    require(thrice.sum_probabilities(0)[5] == 0.0 && thrice.sum_probabilities(1)[5] == 0.0,
            "three 7s in the window: no 7");
    require(thrice.sum_weights(1)[5] == 0 && thrice.sum_weights(1)[0] > 0, "only the 7 is out");
}

void test_the_deck_refills_lazily_and_keeps_its_history() {
    BalancedDice dice = BalancedDice::fresh(2);
    std::mt19937_64 rng(5);
    for (int draw = 0; draw < 23; ++draw) {
        dice.apply(dice.draw(rng, draw % 2), draw % 2);
    }
    require(dice.remaining() == 13 && dice.drawable() == BalancedDice::kFullDeck,
            "23 draws leave 13 cards, and the next draw refills");
    const BalancedDice before = dice;
    const Dice card = BalancedDice::card_of_bit(std::countr_zero(~dice.deck & BalancedDice::kFullDeck));
    require(dice.can_draw(card), "a card drawn before the refill can be drawn again");
    dice.apply(card, 1);
    require(dice.remaining() == 35, "refilled, then one card drawn");
    require(dice.sevens == before.sevens && dice.recent[3] == before.recent[4],
            "the history survives the refill");
    // A card drawn since the refill is refused, and nothing changes.
    const BalancedDice snapshot = dice;
    require_invalid([&] { dice.apply(card, 0); }, "a drawn card must be refused");
    require_invalid([&] { dice.apply(Dice{0, 7}, 0); }, "faces outside [1, 6] must be refused");
    require_invalid([&] { dice.apply(Dice{1, 1}, 2); }, "a roller must be a seat");
    require(dice == snapshot, "a refused card changes nothing");
}

/**
 * Mean |7s of seat 0 - 7s of seat 1| after `rolls` alternating rolls, over
 * `games` games, under balanced or fair dice. Balanced: a seat holding every
 * 7 so far (two or more) never gets a chance at one.
 */
double mean_seven_gap(bool balanced, int games, int rolls, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    long long gaps = 0;
    for (int game = 0; game < games; ++game) {
        BalancedDice dice = BalancedDice::fresh(2);
        std::array<int, 2> sevens{};
        for (int roll = 0; roll < rolls; ++roll) {
            const int roller = roll % 2;
            Dice card{};
            if (balanced) {
                const auto own = static_cast<std::size_t>(roller);
                if (dice.sevens[own] >= 2 && dice.sevens[1 - own] == 0) {
                    require(dice.sum_probabilities(roller)[5] == 0.0,
                            "a seat with every 7 cannot roll one");
                }
                card = dice.draw(rng, roller);
                dice.apply(card, roller);
            } else {
                card = Dice{random::between(rng, 1, 6), random::between(rng, 1, 6)};
            }
            sevens[static_cast<std::size_t>(roller)] += card.first + card.second == 7;
        }
        gaps += std::abs(sevens[0] - sevens[1]);
    }
    return static_cast<double>(gaps) / games;
}

void test_long_run_frequencies_and_seven_balance() {
    // Two seats alternating (as turns do), 200000 draws.
    BalancedDice dice = BalancedDice::fresh(2);
    std::mt19937_64 rng(2026);
    constexpr int kDraws = 200000;
    std::array<int, 11> totals{};
    int sevens_after_seven = 0;
    int after_seven = 0;
    int no_seven_possible = 0;
    bool previous_seven = false;
    for (int draw = 0; draw < kDraws; ++draw) {
        const int roller = draw % 2;
        if (dice.sum_probabilities(roller)[5] == 0.0) {
            ++no_seven_possible;
        }
        const Dice card = dice.draw(rng, roller);
        require(dice.can_draw(card), "draws come from the deck");
        dice.apply(card, roller);
        const int total = card.first + card.second;
        ++totals[static_cast<std::size_t>(total - 2)];
        if (previous_seven) {
            ++after_seven;
            sevens_after_seven += total == 7;
        }
        previous_seven = total == 7;
    }
    // Simulated with the exact reference (200k rolls): .0301 .0594 .0853 .1108 .1340 .1587
    // .1348 .1113 .0860 .0591 .0305.
    const std::array<double, 11> expected{.0301, .0594, .0853, .1108, .1340, .1587,
                                          .1348, .1113, .0860, .0591, .0305};
    for (std::size_t i = 0; i < 11; ++i) {
        const double frequency = totals[i] / static_cast<double>(kDraws);
        require(std::abs(frequency - expected[i]) < 0.004,
                "total " + std::to_string(i + 2) + " comes up " + std::to_string(frequency));
        require(std::abs(frequency - kWays[i] / 36.0) < 0.012, "close to 2d6");
    }
    const double repeat = sevens_after_seven / static_cast<double>(after_seven);
    require(std::abs(repeat - 0.128) < 0.01, "P(7 after a 7) " + std::to_string(repeat));
    const double blocked = no_seven_possible / static_cast<double>(kDraws);
    require(blocked > 0.005 && blocked < 0.02, "P(no 7 possible) " + std::to_string(blocked));

    // Over a game's worth of rolls the seats' 7s stay closer than with fair dice.
    const double balanced_gap = mean_seven_gap(true, 4000, 60, 7);
    const double fair_gap = mean_seven_gap(false, 4000, 60, 7);
    std::cout << "mean 7s gap after 60 rolls: balanced " << balanced_gap << ", fair " << fair_gap << '\n';
    require(balanced_gap < 0.7 * fair_gap, "balanced dice keep the seats' 7s close");
}

const Action& find_action(const Game& game, ActionType type) {
    const auto& actions = game.playable_actions();
    const auto it = std::find_if(actions.begin(), actions.end(),
                                 [type](const Action& action) { return action.type == type; });
    require(it != actions.end(), "expected action is not playable");
    return *it;
}

bool can_roll(const Game& game) {
    const auto& actions = game.playable_actions();
    return game.current_prompt() == ActionPrompt::play_turn &&
           std::any_of(actions.begin(), actions.end(),
                       [](const Action& action) { return action.type == ActionType::roll; });
}

/** Play on (ending turns when possible, else the first action) until someone must roll. */
void to_next_roll(Game& game) {
    while (!can_roll(game)) {
        const auto& actions = game.playable_actions();
        const auto end = std::find_if(actions.begin(), actions.end(), [](const Action& action) {
            return action.type == ActionType::end_turn;
        });
        game.execute(end != actions.end() ? *end : actions.front());
    }
}

void test_random_dice_ignore_the_state() {
    Game game(seats(3), MapType::base, 9);
    require(game.dice_mode() == DiceMode::random, "random dice by default");
    std::mt19937_64 rng(9);
    for (int step = 0; step < 2000 && !game.winning_color().has_value(); ++step) {
        const auto& actions = game.playable_actions();
        game.execute(actions[rng() % actions.size()]);
    }
    require(game.balanced_dice() == BalancedDice::fresh(3), "random dice never touch the state");
    const auto probabilities = game.roll_probabilities(0);
    for (std::size_t i = 0; i < 11; ++i) {
        require(probabilities[i] == kWays[i] / 36.0, "random dice roll 2d6");
    }
}

void test_balanced_games_draw_from_the_state() {
    Game game(seats(2), MapType::base, 4, balanced_rules());
    std::mt19937_64 rng(4);
    int rolls = 0;
    for (int step = 0; step < 4000 && !game.winning_color().has_value(); ++step) {
        const auto& actions = game.playable_actions();
        const Action action = actions[rng() % actions.size()];
        if (action.type != ActionType::roll) {
            game.execute(action);
            continue;
        }
        const BalancedDice before = game.balanced_dice();
        const int roller = game.current_player_index();
        require(game.next_roller() == roller, "the turn owner rolls next");
        const auto probabilities = game.roll_probabilities(roller);
        require(probabilities == before.sum_probabilities(roller), "the game's odds are the state's");
        game.execute(action);
        ++rolls;
        const BalancedDice& after = game.balanced_dice();
        const std::uint64_t drawn = before.drawable() & ~after.deck;
        require(std::popcount(drawn) == 1 && (after.deck & ~before.drawable()) == 0,
                "a roll draws one card");
        const Dice card = BalancedDice::card_of_bit(std::countr_zero(drawn));
        BalancedDice expected = before;
        expected.apply(card, roller);
        require(after == expected, "a roll updates the state as BalancedDice::apply does");
        require(probabilities[static_cast<std::size_t>(card.first + card.second - 2)] > 0.0,
                "only totals with a chance come up");
        if (game.current_prompt() == ActionPrompt::play_turn) {
            require(game.next_roller() == (roller + 1) % 2, "after the roll, the next seat");
        }
    }
    require(rolls > 100, "the game rolled");

    // Copies replay identically, and the dice state is part of equivalence.
    Game copy = game;
    require(copy.search_equivalent(game), "a copy is equivalent");
}

void test_replayed_rolls_update_and_refuse() {
    Game game(seats(2), MapType::base, 12, balanced_rules());
    to_next_roll(game);
    game.execute(find_action(game, ActionType::roll), Dice{3, 4});
    require(game.balanced_dice().remaining() == 35 && game.balanced_dice().sevens[0] == 1,
            "a replayed roll draws that card");
    to_next_roll(game);
    const Game before = game;
    require_invalid([&] { game.execute(find_action(game, ActionType::roll), Dice{3, 4}); },
                    "a card drawn since the last refill must be refused");
    require(game.search_equivalent(before), "a refused roll changes nothing");
    game.execute(find_action(game, ActionType::roll), Dice{4, 3});
    require(game.balanced_dice().streak_player == 1 && game.balanced_dice().sevens[1] == 1,
            "another 7 card is fine");

    // Random dice take any faces, as before.
    Game random_game(seats(2), MapType::base, 12);
    to_next_roll(random_game);
    random_game.execute(find_action(random_game, ActionType::roll), Dice{3, 4});
    to_next_roll(random_game);
    random_game.execute(find_action(random_game, ActionType::roll), Dice{3, 4});
}

void test_positions_declare_and_keep_the_dice() {
    Game played(seats(2), MapType::base, 8, balanced_rules());
    to_next_roll(played);
    for (Dice card : {Dice{1, 6}, Dice{5, 5}, Dice{6, 1}}) {
        played.execute(find_action(played, ActionType::roll), card);
        to_next_roll(played);
    }
    GameSetup setup;
    for (const PlayerState& state : played.players()) {
        PlayerSetup player;
        player.settlements = state.settlements;
        player.cities = state.cities;
        player.roads = state.roads;
        player.resources = state.resources;
        setup.players.push_back(player);
    }
    setup.robber = played.board().robber_coordinate();
    setup.current_player = played.current_player_index();
    setup.completed_turns = played.completed_turns();
    setup.dice = played.balanced_dice();

    Game declared(seats(2), MapType::base, 8, balanced_rules());
    declared.apply_setup(setup);
    require(declared.balanced_dice() == played.balanced_dice(), "a declared dice state");
    require(declared.roll_probabilities(declared.current_player_index()) ==
                played.roll_probabilities(played.current_player_index()),
            "the same odds");

    GameSetup fresh_setup = setup;
    fresh_setup.dice.reset();
    Game fresh(seats(2), MapType::base, 8, balanced_rules());
    fresh.apply_setup(fresh_setup);
    require(fresh.balanced_dice() == BalancedDice::fresh(2), "undeclared: a fresh deck");

    Game random_game(seats(2), MapType::base, 8);
    require_invalid([&] { random_game.apply_setup(setup); }, "random dice take no dice state");
    GameSetup wrong = setup;
    wrong.dice = BalancedDice::fresh(3);
    Game three(seats(2), MapType::base, 8, balanced_rules());
    require_invalid([&] { three.apply_setup(wrong); }, "one sevens count per seat");
    wrong.dice = played.balanced_dice();
    wrong.dice->streak_count = 0;
    require_invalid([&] { three.apply_setup(wrong); }, "a streak needs a count");

    // Redealing hidden cards keeps the (public) dice.
    Game redealt = played;
    std::vector<HiddenHand> hands;
    for (const PlayerState& state : played.players()) {
        hands.push_back({state.resources, state.development_cards, state.development_card_owned_at_start});
    }
    redealt.redeal_hidden_cards(hands, 99);
    require(redealt.balanced_dice() == played.balanced_dice(), "redeal keeps the dice");
}

/** Seat 0: 7, seat 1: 6, seat 0: 7, seat 1: 8, seat 0: 6; seat 1 to roll. */
Game window_root() {
    Game game(seats(2), MapType::base, 31, balanced_rules());
    to_next_roll(game);
    for (Dice card : {Dice{1, 6}, Dice{1, 5}, Dice{2, 5}, Dice{2, 6}, Dice{2, 4}}) {
        game.execute(find_action(game, ActionType::roll), card);
        to_next_roll(game);
    }
    require(game.current_player_index() == 1 && game.balanced_dice().recent_count == 5,
            "seat 1 to roll after five rolls");
    return game;
}

void test_mcts_outcomes_follow_the_state() {
    const Game game = window_root();
    const Action& roll = find_action(game, ActionType::roll);
    const auto probabilities = game.roll_probabilities(1);
    const auto outcomes = action_outcomes(game, roll);
    std::size_t next = 0;
    for (int total = 2; total <= 12; ++total) {
        const double probability = probabilities[static_cast<std::size_t>(total - 2)];
        if (probability == 0.0) {
            continue;
        }
        require(next < outcomes.size() && outcomes[next].probability == probability,
                "an MCTS roll outcome per total with the state's odds");
        BalancedDice expected = game.balanced_dice();
        expected.apply(game.balanced_dice().card_for_total(total), 1);
        require(outcomes[next].game.balanced_dice() == expected, "the outcome draws a card of it");
        ++next;
    }
    require(next == outcomes.size(), "no outcome without a chance");

    // A state where a total cannot come up: 7 three times in the window.
    Game blocked(seats(2), MapType::base, 31, balanced_rules());
    to_next_roll(blocked);
    for (Dice card : {Dice{1, 6}, Dice{6, 1}, Dice{2, 5}}) {
        blocked.execute(find_action(blocked, ActionType::roll), card);
        to_next_roll(blocked);
    }
    const auto blocked_outcomes = action_outcomes(blocked, find_action(blocked, ActionType::roll));
    require(blocked_outcomes.size() == 10, "no outcome for a total that cannot come up");
    for (const auto& outcome : blocked_outcomes) {
        require(outcome.game.balanced_dice().recent[3] != 7, "never the blocked 7");
    }
}

/** Fills one row: logits over the codec and seat-relative values. */
using Evaluator = std::function<void(const Game&, std::span<float>, std::span<float>)>;

void run(SearchPool& pool, const Evaluator& evaluate, std::size_t batch = 64) {
    std::vector<float> observations(batch * pool.observation_size());
    std::vector<std::int64_t> ids(batch);
    const std::size_t width = pool.action_size();
    const auto players = static_cast<std::size_t>(pool.max_players());
    while (const std::size_t rows = pool.collect(observations, ids)) {
        std::vector<float> logits(rows * width);
        std::vector<float> values(rows * players);
        for (std::size_t row = 0; row < rows; ++row) {
            evaluate(pool.pending_game(row), std::span<float>(logits).subspan(row * width, width),
                     std::span<float>(values).subspan(row * players, players));
        }
        pool.apply(logits, values);
    }
}

class ToyEncoder final : public ObservationEncoder {
public:
    [[nodiscard]] std::size_t size() const noexcept override { return 2; }
    void encode(const Game& game, int observer, std::span<float> out) const override {
        out[0] = static_cast<float>(observer);
        out[1] = static_cast<float>(game.num_turns());
    }
};

/**
 * The search's chance node for the root's roll weighs its outcomes by the
 * state's odds: with every position after a roll of `marked` worth 0.9 to
 * the root player and every other 0.5, the roll's value is
 * 0.5 + 0.4 P(marked) (every outcome is visited; edge priors are floats). Two totals that produce
 * the same position but different decks must stay separate outcomes.
 */
void test_search_pool_chance_nodes_follow_the_state() {
    const Game game = window_root();
    const auto codec = std::make_shared<FlatActionCodec>(2, MapType::base);
    const std::size_t roll_index = codec->index(game, find_action(game, ActionType::roll));
    const auto probabilities = game.roll_probabilities(1);
    double mass = 0.0;
    for (double probability : probabilities) {
        mass += static_cast<float>(probability);
    }
    const int root_remaining = game.balanced_dice().remaining();
    for (bool transpositions : {false, true}) {
        for (int marked = 2; marked <= 12; ++marked) {
            SearchConfig config;
            // Every position costs an evaluation, so no path gets five rolls past
            // the root (the window remembers the first only that long).
            config.simulations = 400;
            config.skip_forced = false;
            config.max_players = 2;
            config.transpositions = transpositions;
            SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
            const auto id = pool.add(game, static_cast<std::uint64_t>(marked));
            run(pool, [&](const Game& leaf, std::span<float> logits, std::span<float> values) {
                std::fill(logits.begin(), logits.end(), 0.0F);
                const int rolled = root_remaining - leaf.balanced_dice().remaining();
                require(rolled >= 0 && rolled <= 5,
                        "the search stays within five rolls, not " + std::to_string(rolled));
                double value = 0.5;
                if (rolled > 0) {
                    const int first = leaf.balanced_dice().recent[static_cast<std::size_t>(5 - rolled)];
                    value = first == marked ? 0.9 : 0.5;
                }
                const bool root_to_move = leaf.current_player_index() == 1;
                values[0] = static_cast<float>(root_to_move ? value : 1.0 - value);
                values[1] = static_cast<float>(root_to_move ? 1.0 - value : value);
            });
            const double expected =
                0.5 + 0.4 * static_cast<float>(probabilities[static_cast<std::size_t>(marked - 2)]) / mass;
            const double actual = pool.q(id)[roll_index];
            require(std::abs(actual - expected) < 1e-6,
                    "the roll's value with " + std::to_string(marked) + " marked: " +
                        std::to_string(actual) + " != " + std::to_string(expected));
            require(pool.stats(id).coalesced_outcomes == 0, "balanced outcomes never coalesce");
        }
    }
}

}  // namespace

int main() {
    try {
        test_a_fresh_deck_is_two_fair_dice();
        test_hand_computed_odds();
        test_the_deck_refills_lazily_and_keeps_its_history();
        test_long_run_frequencies_and_seven_balance();
        test_random_dice_ignore_the_state();
        test_balanced_games_draw_from_the_state();
        test_replayed_rolls_update_and_refuse();
        test_positions_declare_and_keep_the_dice();
        test_mcts_outcomes_follow_the_state();
        test_search_pool_chance_nodes_follow_the_state();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "cppanatron dice tests passed\n";
    return 0;
}
