// HandTracker and sample_world: soundness, independence from hidden truth,
// and legality of sampled worlds over random games of 2-6 players.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cppanatron/game.hpp"
#include "cppanatron/tracker.hpp"

namespace {

using namespace cppanatron;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<Color> seats(int n) {
    return {kColors.begin(), kColors.begin() + n};
}

GameRules rules_for(int n) {
    GameRules rules;
    if (n >= 5) {
        rules.resource_bank = {24, 24, 24, 24, 24};
        rules.development_deck = {20, 3, 3, 3, 5};
    }
    return rules;
}

Action random_action(const Game& game, std::mt19937_64& rng) {
    const auto& actions = game.playable_actions();
    std::uniform_int_distribution<std::size_t> pick(0, actions.size() - 1);
    return actions[pick(rng)];
}

/** Every card is somewhere: hands plus bank, hands plus deck plus played. */
void require_conserved(const Game& game) {
    std::array<int, 5> resources = game.resource_bank();
    std::array<int, 5> cards{};
    for (DevelopmentCard card : game.development_deck()) {
        ++cards[static_cast<std::size_t>(card)];
    }
    for (const PlayerState& player : game.players()) {
        for (std::size_t i = 0; i < 5; ++i) {
            resources[i] += player.resources[i];
            cards[i] += player.development_cards[i] + player.played_development_cards[i];
        }
        require(
            player.actual_victory_points ==
                player.victory_points +
                    player.development_cards[static_cast<std::size_t>(DevelopmentCard::victory_point)],
            "actual victory points must be public points plus victory point cards");
    }
    require(resources == game.rules().resource_bank, "resources must be conserved");
    require(cards == game.rules().development_deck, "development cards must be conserved");
}

struct Counts {
    int hidden_steals{};
    int seen_steals{};
    int monopolies{};
    int samples{};
    int continuations{};
};

void count_events(const Game& before, const Action& action, const Game& after, int n, Counts& counts) {
    for (int observer = 0; observer < n; ++observer) {
        for (const auto& event : HandTracker::events(before, action, after, observer)) {
            if (event.type == TrackerEventType::steal) {
                ++(event.resource.has_value() ? counts.seen_steals : counts.hidden_steals);
            }
            if (event.type == TrackerEventType::monopoly && observer == 0) {
                ++counts.monopolies;
            }
        }
    }
}

/** (b) and (c) at one position of one observer. */
void check_samples(const Game& truth, const HandTracker& tracker, std::uint64_t seed, Counts& counts) {
    std::mt19937_64 alt_rng(seed);
    const Game alternative = sample_world(truth, tracker, alt_rng);
    require(tracker.consistent_with(alternative), "a sampled world must satisfy the tracker");
    require_conserved(alternative);
    for (std::uint64_t k = 0; k < 3; ++k) {
        std::mt19937_64 first(seed * 7919 + k);
        std::mt19937_64 second(seed * 7919 + k);
        const Game from_truth = sample_world(truth, tracker, first);
        const Game from_alternative = sample_world(alternative, tracker, second);
        require(
            from_truth.search_equivalent(from_alternative),
            "samples must not depend on the hidden state of the game they start from");
        ++counts.samples;

        // (c) the sample is a legal game: play it on.
        Game world = from_truth;
        std::mt19937_64 play_rng(seed + k);
        for (int step = 0; step < 80 && !world.winning_color().has_value(); ++step) {
            world.execute(random_action(world, play_rng));
            require_conserved(world);
        }
        ++counts.continuations;
    }
}

void test_tracker_follows_random_games() {
    Counts counts;
    for (int n = 2; n <= 6; ++n) {
        for (std::uint64_t seed = 1; seed <= 4; ++seed) {
            Game game(seats(n), MapType::base, seed * 101 + static_cast<std::uint64_t>(n), rules_for(n));
            std::vector<HandTracker> trackers;
            for (int seat = 0; seat < n; ++seat) {
                trackers.emplace_back(game, seat);
            }
            std::mt19937_64 rng(seed);
            for (int step = 0; step < 1500 && !game.winning_color().has_value(); ++step) {
                const Game before = game;
                const Action action = random_action(game, rng);
                game.execute(action);
                count_events(before, action, game, n, counts);
                for (int seat = 0; seat < n; ++seat) {
                    trackers[static_cast<std::size_t>(seat)].observe(before, action, game);
                    require(
                        trackers[static_cast<std::size_t>(seat)].consistent_with(game),
                        "the tracker excluded the true state (n=" + std::to_string(n) +
                            ", seed=" + std::to_string(seed) + ", step=" + std::to_string(step) + ")");
                    if (action.type == ActionType::play_monopoly) {
                        const auto r = static_cast<std::size_t>(std::get<Resource>(action.value));
                        require(trackers[static_cast<std::size_t>(seat)].unknown_pool()[r] == 0,
                                "after a monopoly everyone knows where the resource is");
                    }
                }
                // Like the late trackers below: whether a finished game is won
                // decides how sample_world caps victory point cards, so worlds
                // sampled from a finished game may differ.
                if (step % 37 == 0 && !game.winning_color().has_value()) {
                    const int observer = static_cast<int>(rng() % static_cast<std::uint64_t>(n));
                    check_samples(
                        game, trackers[static_cast<std::size_t>(observer)],
                        seed * 1000003 + static_cast<std::uint64_t>(step), counts);
                }
            }
            // A tracker started mid-game knows only public state; it is sound too.
            for (int seat = 0; seat < n; ++seat) {
                const HandTracker late(game, seat);
                require(late.consistent_with(game), "a tracker started mid-game must be sound");
                if (!game.winning_color().has_value()) {
                    check_samples(game, late, seed + 17 * static_cast<std::uint64_t>(seat), counts);
                }
            }
        }
    }
    require(counts.hidden_steals > 0, "random games must exercise hidden steals");
    require(counts.seen_steals > 0, "random games must exercise seen steals");
    std::cout << "tracker: " << counts.hidden_steals << " hidden steals, " << counts.seen_steals
              << " seen steals, " << counts.monopolies << " monopolies, " << counts.samples
              << " sample pairs, " << counts.continuations << " continuations\n";
}

/**
 * Two games the observer cannot tell apart, played on with the same public
 * actions and dice, keep identical trackers.
 */
void test_sample_world_refuses_finished_games() {
    Game game(seats(2), MapType::base, 5);
    std::mt19937_64 rng(5);
    while (!game.winning_color().has_value()) {
        game.execute(random_action(game, rng));
    }
    const HandTracker tracker(game, 0);
    bool refused = false;
    try {
        (void)sample_world(game, tracker, rng);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "sample_world must refuse a finished game");
}

void test_tracker_ignores_hidden_differences() {
    int compared = 0;
    for (int n = 3; n <= 4; ++n) {
        for (std::uint64_t seed = 1; seed <= 12; ++seed) {
            Game truth(seats(n), MapType::base, seed * 13 + static_cast<std::uint64_t>(n), rules_for(n));
            std::mt19937_64 rng(seed);
            // Play into the game first so hands hold something.
            for (int step = 0; step < 250 && !truth.winning_color().has_value(); ++step) {
                truth.execute(random_action(truth, rng));
            }
            if (truth.winning_color().has_value()) {
                continue;
            }
            const int observer = static_cast<int>(seed % static_cast<std::uint64_t>(n));
            HandTracker truth_tracker(truth, observer);
            std::mt19937_64 sample_rng(seed);
            Game other = sample_world(truth, truth_tracker, sample_rng);
            HandTracker other_tracker = truth_tracker;
            for (int step = 0; step < 300 && !truth.winning_color().has_value(); ++step) {
                const Action action = random_action(truth, rng);
                const auto& legal = other.playable_actions();
                if (std::find(legal.begin(), legal.end(), action) == legal.end()) {
                    break;
                }
                std::optional<Dice> dice;
                if (action.type == ActionType::roll) {
                    dice = Dice{static_cast<int>(rng() % 6) + 1, static_cast<int>(rng() % 6) + 1};
                }
                const Game truth_before = truth;
                const Game other_before = other;
                truth.execute(action, dice);
                other.execute(action, dice);
                truth_tracker.observe(truth_before, action, truth);
                other_tracker.observe(other_before, action, other);
                // Stop once the games become distinguishable to the observer.
                const auto& own_truth = truth.players()[static_cast<std::size_t>(observer)];
                const auto& own_other = other.players()[static_cast<std::size_t>(observer)];
                bool same = own_truth.resources == own_other.resources &&
                            own_truth.development_cards == own_other.development_cards;
                for (int seat = 0; seat < n; ++seat) {
                    const auto& a = truth.players()[static_cast<std::size_t>(seat)];
                    const auto& b = other.players()[static_cast<std::size_t>(seat)];
                    same = same &&
                           std::accumulate(a.resources.begin(), a.resources.end(), 0) ==
                               std::accumulate(b.resources.begin(), b.resources.end(), 0) &&
                           a.victory_points == b.victory_points;
                }
                if (!same) {
                    break;
                }
                require(
                    truth_tracker == other_tracker,
                    "observationally identical games must give identical trackers");
                ++compared;
            }
        }
    }
    require(compared > 200, "the dual-trajectory test must compare many steps");
    std::cout << "tracker: " << compared << " dual-trajectory steps compared\n";
}

/** An initial build phase played by random choices. */
Game setup_game(int n, std::uint64_t seed, const GameRules& rules) {
    Game game(seats(n), MapType::base, seed, rules);
    std::mt19937_64 rng(seed);
    while (game.is_initial_build_phase()) {
        game.execute(random_action(game, rng));
    }
    return game;
}

void test_victory_point_cards_respect_the_target() {
    // Three points win: seats with two public points cannot hold a VP card.
    GameRules rules;
    rules.victory_points_to_win = 3;
    const Game placed = setup_game(3, 5, rules);
    std::vector<PlayerSetup> players;
    for (const PlayerState& state : placed.players()) {
        PlayerSetup setup;
        setup.settlements = state.settlements;
        setup.roads = state.roads;
        players.push_back(setup);
    }
    players[1].development_cards = {1, 0, 0, 0, 0};
    players[2].development_cards = {0, 0, 1, 1, 0};
    Game game(seats(3), MapType::base, 5, rules);
    GameSetup setup;
    setup.players = players;
    game.apply_setup(setup);
    const HandTracker tracker(game, 0);
    std::mt19937_64 rng(3);
    for (int i = 0; i < 300; ++i) {
        const Game world = sample_world(game, tracker, rng);
        for (int seat = 1; seat < 3; ++seat) {
            const auto& cards = world.players()[static_cast<std::size_t>(seat)].development_cards;
            require(cards[4] == 0, "a sampled seat must not hold a winning VP card");
            require(std::accumulate(cards.begin(), cards.end(), 0) ==
                        tracker.development_card_count(seat),
                    "sampled development hands keep their size");
        }
        require(!world.winning_color().has_value(), "a sampled world must not be over");
    }

    // With the usual target, VP cards are dealt like any other.
    GameRules usual;
    Game open(seats(3), MapType::base, 5, usual);
    open.apply_setup(setup);
    const HandTracker open_tracker(open, 0);
    int victory_cards = 0;
    for (int i = 0; i < 300; ++i) {
        const Game world = sample_world(open, open_tracker, rng);
        victory_cards += world.players()[1].development_cards[4] + world.players()[2].development_cards[4];
    }
    // 3 hidden cards from 25 unseen, 5 of them VP: about 0.6 per world.
    require(victory_cards > 100 && victory_cards < 260, "VP cards must be dealt uniformly");
}

/** Sampled hands average to the tracker's expected hands. */
void test_samples_match_expected_hands() {
    const int n = 4;
    Game game(seats(n), MapType::base, 77);
    std::mt19937_64 rng(77);
    std::vector<HandTracker> trackers;
    for (int seat = 0; seat < n; ++seat) {
        trackers.emplace_back(game, seat);
    }
    for (int step = 0; step < 400 && !game.winning_color().has_value(); ++step) {
        const Game before = game;
        const Action action = random_action(game, rng);
        game.execute(action);
        for (auto& tracker : trackers) {
            tracker.observe(before, action, game);
        }
    }
    require(!game.winning_color().has_value(), "the position must still be in play");
    const HandTracker late(game, 0);
    for (const HandTracker* tracker : std::vector<const HandTracker*>{&trackers[0], &late}) {
        const int samples = 4000;
        std::vector<std::array<double, 5>> resources(n);
        std::vector<std::array<double, 5>> cards(n);
        for (int i = 0; i < samples; ++i) {
            const Game world = sample_world(game, *tracker, rng);
            for (int seat = 1; seat < n; ++seat) {
                for (std::size_t r = 0; r < 5; ++r) {
                    resources[static_cast<std::size_t>(seat)][r] +=
                        world.players()[static_cast<std::size_t>(seat)].resources[r];
                    cards[static_cast<std::size_t>(seat)][r] +=
                        world.players()[static_cast<std::size_t>(seat)].development_cards[r];
                }
            }
        }
        for (int seat = 1; seat < n; ++seat) {
            const auto expected = tracker->expected_resources(seat);
            const auto expected_cards = tracker->expected_development_cards(seat);
            for (std::size_t r = 0; r < 5; ++r) {
                const double mean = resources[static_cast<std::size_t>(seat)][r] / samples;
                require(std::abs(mean - expected[r]) < 0.1, "sampled hands must average to the expected hand");
                // VP caps may only lower the VP share a little.
                const double card_mean = cards[static_cast<std::size_t>(seat)][r] / samples;
                require(std::abs(card_mean - expected_cards[r]) < 0.1 || r == 4,
                        "sampled development hands must average to the expected hand");
            }
        }
    }
}

void test_steal_and_monopoly_events() {
    // A fresh game: every hand is empty, so the trackers start exact.
    const Game fresh(seats(3), MapType::base, 11);

    HandTracker tracker(fresh, 0);
    tracker.gain(1, {2, 1, 0, 0, 0});
    tracker.gain(2, {0, 0, 1, 0, 0});
    tracker.steal(2, 1, std::nullopt);
    require(tracker.hand_size(1) == 2 && tracker.hand_size(2) == 2, "a steal moves one card");
    require(tracker.known_resources(1) == std::array<int, 5>{1, 0, 0, 0, 0},
            "an unseen steal lowers every known bound of the victim");
    require(tracker.unknown_count(1) == 1 && tracker.unknown_count(2) == 1,
            "victim and thief each hold one unknown card");
    require(tracker.unknown_pool() == std::array<int, 5>{1, 1, 0, 0, 0},
            "the unknown cards are the wood or brick that moved");
    const auto expected = tracker.expected_resources(2);
    require(expected[0] == 0.5 && expected[1] == 0.5 && expected[2] == 1.0,
            "expected hands split the unknown pool evenly");

    bool threw = false;
    try {
        tracker.steal(0, 1, std::nullopt);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "the observer always sees its own steals");
    threw = false;
    try {
        tracker.buy_development_card(1, DevelopmentCard::knight);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "an opponent's bought card must not reach the tracker");
    threw = false;
    const HandTracker unchanged = tracker;
    try {
        tracker.lose(2, {0, 0, 0, 3, 0});
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw && tracker == unchanged, "impossible events throw and change nothing");

    // A steal from a hand of one known kind stays known.
    HandTracker certain(fresh, 0);
    certain.gain(1, {0, 0, 0, 3, 0});
    certain.steal(2, 1, std::nullopt);
    require(certain.known_resources(2)[3] == 1 && certain.unknown_count(2) == 0,
            "a steal of a certain kind is a known transfer");

    // A monopoly reveals the monopolist's whole stock of the resource.
    HandTracker monopoly(fresh, 0);
    monopoly.gain(0, {0, 0, 1, 0, 0});
    monopoly.gain(1, {0, 0, 2, 0, 0});
    monopoly.gain(2, {0, 0, 3, 1, 0});
    monopoly.steal(1, 2, std::nullopt);
    monopoly.monopoly(1, Resource::sheep, {1, 0, 2});
    require(monopoly.known_resources(1)[2] == 6 && monopoly.unknown_count(1) == 0,
            "the monopolist holds every sheep");
    require(monopoly.unknown_pool() == std::array<int, 5>{0, 0, 0, 1, 0},
            "the victim's last unknown card is the wheat");
}

}  // namespace

/** A game after a random initial phase, rebuilt by apply_setup with edits. */
Game positioned(int n, std::uint64_t seed, const GameRules& rules,
                const std::function<void(std::vector<PlayerSetup>&, GameSetup&)>& edit) {
    const Game placed = setup_game(n, seed, rules);
    std::vector<PlayerSetup> players;
    for (const PlayerState& state : placed.players()) {
        PlayerSetup player;
        player.settlements = state.settlements;
        player.roads = state.roads;
        players.push_back(player);
    }
    GameSetup setup;
    edit(players, setup);
    setup.players = players;
    Game game(seats(n), MapType::base, seed, rules);
    game.apply_setup(setup);
    return game;
}

/** The engine lets an offerer accept and confirm its own offer; nothing moves. */
void test_self_confirmed_trade() {
    Game game = positioned(3, 9, GameRules{}, [](std::vector<PlayerSetup>& players, GameSetup& setup) {
        players[1].resources = {0, 0, 1, 0, 0};
        players[2].resources = {1, 1, 0, 0, 0};
        setup.current_player = 1;
        setup.has_rolled = true;
    });
    std::vector<HandTracker> trackers{HandTracker(game, 0), HandTracker(game, 1), HandTracker(game, 2)};
    const auto step = [&](const Action action) {  // a copy: execute replaces playable actions
        const Game before = game;
        game.execute(action);
        for (auto& tracker : trackers) {
            tracker.observe(before, action, game);
            require(tracker.consistent_with(game), "a self trade must not break the tracker");
        }
    };
    DomesticTrade trade;
    trade.offering = {1, 0, 0, 0, 0};
    trade.asking = {0, 0, 1, 0, 0};
    step({Color::blue, ActionType::offer_trade, trade});
    bool confirmed = false;
    while (game.current_prompt() == ActionPrompt::decide_trade ||
           game.current_prompt() == ActionPrompt::decide_acceptees) {
        const auto& actions = game.playable_actions();
        const bool offerer = game.current_color() == Color::blue;
        const auto wanted = game.current_prompt() == ActionPrompt::decide_acceptees
            ? ActionType::confirm_trade
            : (offerer ? ActionType::accept_trade : ActionType::reject_trade);
        const auto it = std::find_if(actions.begin(), actions.end(), [&](const Action& action) {
            return action.type == wanted &&
                   (wanted != ActionType::confirm_trade ||
                    std::get<ConfirmedTrade>(action.value).partner == Color::blue);
        });
        require(it != actions.end(), "the offerer can accept and confirm its own offer");
        confirmed = confirmed || wanted == ActionType::confirm_trade;
        step(*it);
    }
    require(confirmed, "the self trade was confirmed");
    require(trackers[0].unknown_count(1) == 1 && trackers[0].known_resources(1)[2] == 0,
            "a self trade reveals nothing");

    bool threw = false;
    HandTracker tracker(game, 0);
    const HandTracker unchanged = tracker;
    try {
        tracker.transfer(1, 1, {0, 0, 1, 0, 0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw && tracker == unchanged, "a transfer to oneself is rejected and changes nothing");
}

/**
 * A seat that held its one card at 9 public points (of 10) cannot hold a
 * victory point card, even after losing Largest Army.
 */
void test_victory_point_cap_history() {
    Game game(seats(3), MapType::tournament, 3);
    GameSetup setup;
    setup.players.resize(3);
    setup.players[0].cities = {1, 3, 13};
    setup.players[0].settlements = {20};
    setup.players[0].development_cards = {0, 0, 0, 1, 0};
    setup.players[0].played_development_cards = {3, 0, 0, 0, 0};
    setup.players[1].settlements = {42};
    setup.players[1].development_cards = {1, 0, 0, 0, 0};
    setup.players[1].played_development_cards = {3, 0, 0, 0, 0};
    setup.players[2].settlements = {49};
    setup.current_player = 1;
    setup.largest_army = 0;
    game.apply_setup(setup);
    require(game.players()[0].victory_points == 9, "seat 0 starts at 9 public points");
    HandTracker tracker(game, 2);
    require(tracker.victory_point_card_cap(0) == 0, "at 9 points the card is not a VP card");

    const Game before = game;
    const Action knight{Color::blue, ActionType::play_knight_card, std::monostate{}};
    game.execute(knight);
    tracker.observe(before, knight, game);
    require(game.players()[0].victory_points == 7, "seat 0 lost Largest Army");
    require(tracker.victory_point_card_cap(0) == 0, "the cap remembers the 9 points");
    std::mt19937_64 rng(1);
    for (int i = 0; i < 200; ++i) {
        require(sample_world(game, tracker, rng).players()[0].development_cards[4] == 0,
                "a sample must not give seat 0 a VP card");
    }

    // Each card bought since raises the cap by one.
    HandTracker buying(game, 2);
    buying.public_victory_points({9, 2, 1});
    buying.buy_development_card(0, std::nullopt);
    require(buying.victory_point_card_cap(0) == 1, "a new card may be a VP card");
}

/**
 * Friendly robber: the observer's robber options reveal which seats are
 * under 3 actual points. Seat 1 shows 2 points but holds a VP card.
 */
void test_friendly_robber_options_bound_victory_point_cards() {
    GameRules rules;
    rules.friendly_robber = true;
    Game game = positioned(4, 31, rules, [](std::vector<PlayerSetup>& players, GameSetup&) {
        players[0].development_cards = {1, 0, 0, 0, 0};
        players[1].development_cards = {0, 0, 0, 0, 1};
        players[2].development_cards = {1, 0, 0, 0, 0};
    });
    HandTracker tracker(game, 0);
    require(tracker.victory_point_card_floor(1) == 0, "nothing known before the robber moves");
    const Game before = game;
    const Action knight{Color::red, ActionType::play_knight_card, std::monostate{}};
    game.execute(knight);
    tracker.observe(before, knight, game);
    require(tracker.consistent_with(game), "the bounds hold for the truth");
    require(tracker.victory_point_card_floor(1) == 1, "seat 1 is robbable, so it holds a VP card");
    require(tracker.victory_point_card_cap(2) == 0, "seat 2 is spared, so it holds none");
    std::mt19937_64 rng(5);
    for (int i = 0; i < 100; ++i) {
        const Game world = sample_world(game, tracker, rng);
        require(world.players()[1].development_cards[4] == 1, "samples give seat 1 its VP card");
        require(world.playable_actions() == game.playable_actions(), "samples offer the real robber moves");
    }
    // A tracker started at the robber prompt reads the same options.
    const HandTracker late(game, 0);
    require(late.victory_point_card_floor(1) == 1 && late.victory_point_card_cap(2) == 0,
            "a new tracker reads its options too");
}

int main() {
    try {
        test_tracker_follows_random_games();
        test_tracker_ignores_hidden_differences();
        test_sample_world_refuses_finished_games();
        test_victory_point_cards_respect_the_target();
        test_samples_match_expected_hands();
        test_steal_and_monopoly_events();
        test_self_confirmed_trade();
        test_victory_point_cap_history();
        test_friendly_robber_options_bound_victory_point_cards();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "tracker tests passed\n";
    return 0;
}
