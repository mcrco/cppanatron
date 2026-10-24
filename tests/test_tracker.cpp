// HandTracker and sample_world: soundness, independence from hidden truth,
// and legality of sampled worlds over random games of 2-6 players, in both
// tracker modes; the exact mode's posterior on worked examples.

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
#include <utility>
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

void test_tracker_ignores_hidden_differences(TrackerMode mode) {
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
            HandTracker truth_tracker(truth, observer, mode);
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
    std::cout << "tracker: " << compared << " dual-trajectory steps compared ("
              << (mode == TrackerMode::exact ? "exact" : "pool") << ")\n";
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
    std::vector<HandTracker> trackers;  // every seat in pool mode, then seat 0 in exact mode
    for (int seat = 0; seat < n; ++seat) {
        trackers.emplace_back(game, seat);
    }
    trackers.emplace_back(game, 0, TrackerMode::exact);
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
    const HandTracker late_exact(game, 0, TrackerMode::exact);
    for (const HandTracker* tracker :
         std::vector<const HandTracker*>{&trackers[0], &late, &trackers[n], &late_exact}) {
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
 * Friendly robber (OpenHex's rule, public points): the observer's robber
 * options reveal nothing. Seat 1 shows 2 points but holds a VP card; it is
 * shielded like seat 2, which holds none, and no world the tracker deals
 * moves the robber options.
 */
void test_friendly_robber_options_reveal_no_victory_point_cards() {
    GameRules rules;
    rules.friendly_robber = true;
    Game game = positioned(4, 31, rules, [](std::vector<PlayerSetup>& players, GameSetup&) {
        players[0].development_cards = {1, 0, 0, 0, 0};
        players[1].development_cards = {0, 0, 0, 0, 1};
        players[2].development_cards = {1, 0, 0, 0, 0};
    });
    HandTracker tracker(game, 0);
    const Game before = game;
    const Action knight{Color::red, ActionType::play_knight_card, std::monostate{}};
    game.execute(knight);
    require(game.current_prompt() == ActionPrompt::move_robber, "the knight moves the robber");
    require(HandTracker::option_events(game, game.playable_actions(), 0).empty(), "the options reveal nothing");
    tracker.observe(before, knight, game);
    require(tracker.consistent_with(game), "the tracker holds the truth");
    for (int seat = 1; seat < 4; ++seat) {
        require(tracker.victory_point_card_floor(seat) == 0, "no floor from the robber options");
    }
    require(tracker.victory_point_card_cap(1) == tracker.victory_point_card_cap(2),
            "seats 1 and 2 look alike");
    std::mt19937_64 rng(5);
    int with_card = 0;
    for (int i = 0; i < 100; ++i) {
        const Game world = sample_world(game, tracker, rng);
        with_card += world.players()[1].development_cards[4];
        require(world.playable_actions() == game.playable_actions(), "every world offers the real robber moves");
    }
    require(with_card < 100, "worlds need not give seat 1 its VP card");
}


/** Seat 0 observes; seats 1 and 2 hold `b` and `c`; `edit` adjusts the rest. */
Game three_hands(const std::array<int, 5>& b, const std::array<int, 5>& c,
                 const std::function<void(std::vector<PlayerSetup>&, GameSetup&)>& edit = {}) {
    return positioned(3, 9, GameRules{}, [&](std::vector<PlayerSetup>& players, GameSetup& setup) {
        players[1].resources = b;
        players[2].resources = c;
        if (edit) {
            edit(players, setup);
        }
    });
}

/**
 * A tracker for seat 0 that watched seats 1 and 2 receive `b` and `c` (a
 * tracker started on a dealt position would not know the hands).
 */
HandTracker knowing(TrackerMode mode, const std::array<int, 5>& b, const std::array<int, 5>& c,
                    int max_hypotheses = kDefaultMaxHypotheses) {
    HandTracker tracker(three_hands({}, {}), 0, mode, max_hypotheses);
    tracker.gain(1, b);
    tracker.gain(2, c);
    return tracker;
}

/** Sampled hands of seats 1 and 2 over `count` worlds, as (hand, hand) frequencies. */
std::map<std::pair<std::array<int, 5>, std::array<int, 5>>, int> sampled_hands(
    const Game& game, const HandTracker& tracker, int count, std::uint64_t seed) {
    std::map<std::pair<std::array<int, 5>, std::array<int, 5>>, int> counts;
    std::mt19937_64 rng(seed);
    for (int i = 0; i < count; ++i) {
        const Game world = sample_world(game, tracker, rng);
        ++counts[{world.players()[1].resources, world.players()[2].resources}];
    }
    return counts;
}

/**
 * The report's examples (section 3). A observes B {wood, sheep} and C
 * {brick, ore, wheat}. (a) B robs C unseen, then pays wood and brick for a
 * road: B holds the sheep, C the ore and wheat. (b) C robs B unseen, then
 * pays ore, wheat and sheep for a development card: C holds the brick, B the
 * wood. Both modes resolve them; the exact mode down to one hypothesis.
 */
void test_report_robbery_examples() {
    const std::array<int, 5> wood_sheep{1, 0, 1, 0, 0};
    const std::array<int, 5> brick_ore_wheat{0, 1, 0, 1, 1};
    const Game start = three_hands(wood_sheep, brick_ore_wheat);
    for (TrackerMode mode : {TrackerMode::pool, TrackerMode::exact}) {
        // (a)
        HandTracker tracker = knowing(mode, wood_sheep, brick_ore_wheat);
        require(tracker.consistent_with(start), "the start is consistent");
        require(tracker.known_resources(1) == wood_sheep && tracker.known_resources(2) == brick_ore_wheat,
                "A knows both hands");
        tracker.steal(1, 2, std::nullopt);
        tracker.lose(1, {1, 1, 0, 0, 0});
        const std::array<int, 5> sheep{0, 0, 1, 0, 0};
        const std::array<int, 5> ore_wheat{0, 0, 0, 1, 1};
        const Game after_a = three_hands(sheep, ore_wheat);
        require(tracker.consistent_with(after_a), "(a): the truth stays consistent");
        require(tracker.known_resources(1) == sheep && tracker.unknown_count(1) == 0,
                "(a): B is known to hold the sheep");
        require(tracker.unknown_pool() == ore_wheat, "(a): the ore and wheat are the only unknown cards");
        require(tracker.expected_resources(2) == std::array<double, 5>{0, 0, 0, 1, 1},
                "(a): C is expected to hold the ore and wheat");
        if (mode == TrackerMode::exact) {
            require(tracker.exact() && tracker.hypothesis_count() == 1, "(a): one world is left");
            require(tracker.hypothesis_hands(0)[2] == ore_wheat && tracker.hypothesis_weight(0) == 1.0,
                    "(a): the one world gives C the ore and wheat");
        }
        const auto worlds_a = sampled_hands(after_a, tracker, 300, 1);
        require(worlds_a.size() == 1 && worlds_a.begin()->first == std::make_pair(sheep, ore_wheat),
                "(a): every sampled world gives B the sheep and C the ore and wheat");

        // (b)
        HandTracker other = knowing(mode, wood_sheep, brick_ore_wheat);
        other.steal(2, 1, std::nullopt);
        other.lose(2, {0, 0, 1, 1, 1});
        other.buy_development_card(2, std::nullopt);
        const std::array<int, 5> wood{1, 0, 0, 0, 0};
        const std::array<int, 5> brick{0, 1, 0, 0, 0};
        const Game after_b = three_hands(wood, brick, [](std::vector<PlayerSetup>& players, GameSetup&) {
            players[2].development_cards = {1, 0, 0, 0, 0};
        });
        require(other.consistent_with(after_b), "(b): the truth stays consistent");
        require(other.known_resources(2) == brick && other.unknown_count(2) == 0,
                "(b): C is known to hold the brick");
        require(other.unknown_pool() == wood && other.unknown_count(1) == 1,
                "(b): the wood is the only unknown card, and B holds it");
        if (mode == TrackerMode::exact) {
            require(other.exact() && other.hypothesis_count() == 1 && other.hypothesis_hands(0)[1] == wood,
                    "(b): the one world gives B the wood");
        }
        const auto worlds_b = sampled_hands(after_b, other, 300, 2);
        require(worlds_b.size() == 1 && worlds_b.begin()->first == std::make_pair(wood, brick),
                "(b): every sampled world gives B the wood and C the brick");
    }
}

/**
 * The report's knight example: A has played 6 knights, B 6 and C 1. When C
 * plays another, the 14th, no knight is left anywhere: not in B's hand, not
 * in the deck, and no sampled world holds one.
 */
void test_report_knight_example() {
    Game game = positioned(3, 9, GameRules{}, [](std::vector<PlayerSetup>& players, GameSetup& setup) {
        players[0].played_development_cards = {6, 0, 0, 0, 0};
        players[1].played_development_cards = {6, 0, 0, 0, 0};
        players[1].development_cards = {0, 1, 0, 0, 1};  // hidden from A: a progress and a VP card
        players[2].played_development_cards = {1, 0, 0, 0, 0};
        players[2].development_cards = {1, 0, 0, 0, 0};
        setup.current_player = 2;
        setup.largest_army = 0;
    });
    for (TrackerMode mode : {TrackerMode::pool, TrackerMode::exact}) {
        HandTracker tracker(game, 0, mode);
        require(tracker.unseen_development_cards()[0] == 1, "one knight is unseen before C plays it");
        Game after = game;
        const Action knight{Color::white, ActionType::play_knight_card, std::monostate{}};
        after.execute(knight);
        tracker.observe(game, knight, after);
        require(tracker.consistent_with(after), "the truth stays consistent");
        require(tracker.unseen_development_cards()[0] == 0, "the 14th knight was the last");
        require(tracker.played_development_cards(2)[0] == 2, "C has played two knights");
        std::mt19937_64 rng(3);
        for (int i = 0; i < 200; ++i) {
            const Game world = sample_world(after, tracker, rng);
            for (const PlayerState& player : world.players()) {
                require(player.development_cards[0] == 0, "no sampled hand holds a knight");
            }
            for (DevelopmentCard card : world.development_deck()) {
                require(card != DevelopmentCard::knight, "no sampled deck holds a knight");
            }
            require(world.players()[1].development_cards == std::array<int, 5>{0, 1, 0, 0, 1} ||
                        world.players()[1].development_cards[0] == 0,
                    "B's two cards are dealt from the non-knights");
        }
    }
}

/** The exact belief's probabilities of (B's hand, C's hand), by hypothesis. */
std::map<std::pair<std::array<int, 5>, std::array<int, 5>>, double> posterior(const HandTracker& tracker) {
    std::map<std::pair<std::array<int, 5>, std::array<int, 5>>, double> out;
    for (std::size_t h = 0; h < tracker.hypothesis_count(); ++h) {
        const auto hands = tracker.hypothesis_hands(h);
        out[{hands[1], hands[2]}] += tracker.hypothesis_weight(h);
    }
    return out;
}

/**
 * Where the modes differ. A steal from {brick, brick, ore} took a brick with
 * probability 2/3: the exact mode weights the kinds by their counts, the
 * pool deals them evenly. A seen steal is evidence too, and a payment from
 * cards the pool calls unknown is weighed against which steals could have
 * provided them, which the pool cannot do.
 */
void test_exact_belief_weights_steals_by_count() {
    const std::array<int, 5> none{};
    const std::array<int, 5> brick{0, 1, 0, 0, 0};
    const std::array<int, 5> ore{0, 0, 0, 0, 1};
    const std::array<int, 5> brick_ore{0, 1, 0, 0, 1};
    const std::array<int, 5> two_bricks{0, 2, 0, 0, 0};
    HandTracker exact = knowing(TrackerMode::exact, {0, 2, 0, 0, 1}, none);
    HandTracker pool = knowing(TrackerMode::pool, {0, 2, 0, 0, 1}, none);
    for (HandTracker* tracker : {&exact, &pool}) {
        tracker->steal(2, 1, std::nullopt);
    }
    const auto after = posterior(exact);
    require(after.size() == 2 && std::abs(after.at({brick_ore, brick}) - 2.0 / 3.0) < 1e-12 &&
                std::abs(after.at({two_bricks, ore}) - 1.0 / 3.0) < 1e-12,
            "the stolen card is a brick with probability 2/3");
    require(exact.expected_resources(2)[1] > 0.66 && exact.expected_resources(2)[1] < 0.67,
            "the expected hand follows the posterior");
    const Game truth = three_hands(brick_ore, brick);
    const int samples = 3000;
    const auto exact_worlds = sampled_hands(truth, exact, samples, 4);
    const auto pool_worlds = sampled_hands(truth, pool, samples, 4);
    const double exact_brick = static_cast<double>(exact_worlds.at({brick_ore, brick})) / samples;
    const double pool_brick = static_cast<double>(pool_worlds.at({brick_ore, brick})) / samples;
    require(std::abs(exact_brick - 2.0 / 3.0) < 0.03, "exact samples take the brick 2/3 of the time");
    require(std::abs(pool_brick - 0.5) < 0.03, "the pool deals the two kinds evenly");
    // An exact tracker can still deal the pool, with the same draws as a pool tracker.
    std::mt19937_64 first(11);
    std::mt19937_64 second(11);
    for (int i = 0; i < 20; ++i) {
        require(sample_world(truth, exact, first, TrackerMode::pool)
                    .search_equivalent(sample_world(truth, pool, second)),
                "deal=pool on an exact tracker is the pool tracker's deal");
    }
    bool refused = false;
    try {
        (void)sample_world(truth, pool, first, TrackerMode::exact);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "a pool tracker cannot deal exactly");

    // A sees its own steal: a brick from B. Under "C took a brick" B held one
    // brick of two cards, under "C took the ore" two of two: even odds now.
    exact.steal(0, 1, Resource::brick);
    const auto seen = posterior(exact);
    require(seen.size() == 2 && std::abs(seen.at({ore, brick}) - 0.5) < 1e-12 &&
                std::abs(seen.at({brick, ore}) - 0.5) < 1e-12,
            "a seen steal weights the hypotheses by the victim's count of the kind");

    // B {brick, brick, wood}; C robs B twice unseen, then pays a brick. C's
    // steals were {brick, brick} with probability 1/3 and {brick, wood} with
    // 2/3, both allow the payment, so C's other card is wood with probability
    // 2/3. The pool, left with one unknown card each from {brick, wood}, says 1/2.
    HandTracker exact_twice = knowing(TrackerMode::exact, {1, 2, 0, 0, 0}, none);
    HandTracker pool_twice = knowing(TrackerMode::pool, {1, 2, 0, 0, 0}, none);
    for (HandTracker* tracker : {&exact_twice, &pool_twice}) {
        tracker->steal(2, 1, std::nullopt);
        tracker->steal(2, 1, std::nullopt);
        tracker->lose(2, brick);
    }
    const std::array<int, 5> wood{1, 0, 0, 0, 0};
    const auto paid = posterior(exact_twice);
    require(paid.size() == 2 && std::abs(paid.at({brick, wood}) - 2.0 / 3.0) < 1e-12 &&
                std::abs(paid.at({wood, brick}) - 1.0 / 3.0) < 1e-12,
            "a payment from unknown cards is weighed against the steals that provided them");
    require(pool_twice.expected_resources(2)[0] == 0.5, "the pool splits the last unknown card evenly");
    require(pool_twice.consistent_with(three_hands(brick, wood)) && exact_twice.consistent_with(three_hands(brick, wood)) &&
                exact_twice.consistent_with(three_hands(wood, brick)),
            "both possible truths stay consistent");
}

/**
 * Above max_hypotheses the exact belief falls back to the pool, and restarts
 * from the one known world once every card is known again.
 */
void test_exact_belief_overflow_falls_back() {
    HandTracker tracker = knowing(TrackerMode::exact, {0, 1, 0, 1, 1}, {}, 2);
    require(tracker.max_hypotheses() == 2 && tracker.exact() && tracker.hypothesis_count() == 1,
            "a fresh game is one world");
    tracker.steal(2, 1, std::nullopt);  // three kinds, two hypotheses allowed
    require(!tracker.exact() && tracker.hypothesis_count() == 0 && tracker.belief_overflows() == 1,
            "three hypotheses overflow a cap of two");
    const Game truth = three_hands({0, 1, 0, 1, 0}, {0, 0, 0, 0, 1});
    require(tracker.consistent_with(truth), "the fallback is sound");
    require(sampled_hands(truth, tracker, 200, 5).size() == 3, "the fallback deals the pool");
    require(tracker.unknown_count(2) == 1 && tracker.unknown_count(1) == 2 &&
                tracker.unknown_pool() == std::array<int, 5>{0, 1, 0, 1, 1},
            "the fallback keeps the pool bounds");
    tracker.lose(1, {0, 1, 0, 1, 0});  // B pays the brick and wheat: C took the ore
    require(tracker.exact() && tracker.hypothesis_count() == 1 && tracker.belief_overflows() == 1,
            "once every card is known the belief restarts");
    require(tracker.hypothesis_hands(0)[2] == std::array<int, 5>{0, 0, 0, 0, 1},
            "the restarted world gives C the ore");
    bool threw = false;
    try {
        (void)HandTracker(truth, 0, TrackerMode::exact, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "the cap must be positive");

    // Impossible events still throw and change nothing, in exact mode too.
    HandTracker unchanged = tracker;
    threw = false;
    try {
        tracker.lose(2, {0, 0, 0, 0, 2});
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw && tracker == unchanged, "impossible events leave an exact tracker unchanged");
}

/**
 * An exact tracker started mid-game enumerates every deal of the pool,
 * weighted as a uniform deal: B holds 3 and C 2 of {wood, wood, brick,
 * sheep, wheat}, so B's hands are the 7 compositions and P(B = {wood, wood,
 * brick}) = 1 / C(5, 3). Its expected hands are the pool's.
 */
void test_exact_tracker_started_mid_game_enumerates_deals() {
    const Game game = three_hands({2, 1, 0, 0, 0}, {0, 0, 1, 1, 0});
    const HandTracker exact(game, 0, TrackerMode::exact);
    const HandTracker pool(game, 0);
    require(exact.exact() && exact.hypothesis_count() == 7, "7 deals of the pool");
    require(exact.consistent_with(game), "the truth is one of them");
    const auto deals = posterior(exact);
    require(std::abs(deals.at({{2, 1, 0, 0, 0}, {0, 0, 1, 1, 0}}) - 0.1) < 1e-12 &&
                std::abs(deals.at({{1, 1, 1, 0, 0}, {1, 0, 0, 1, 0}}) - 0.2) < 1e-12,
            "deals are weighted as a uniform deal of the pool");
    for (int seat = 1; seat < 3; ++seat) {
        const auto expected = exact.expected_resources(seat);
        const auto from_pool = pool.expected_resources(seat);
        for (std::size_t r = 0; r < 5; ++r) {
            require(std::abs(expected[r] - from_pool[r]) < 1e-9, "the deals average to the pool's expectation");
        }
    }
    const HandTracker capped(game, 0, TrackerMode::exact, 6);
    require(!capped.exact() && capped.belief_overflows() == 1, "too many deals: the pool until hands are known");
    require(capped.consistent_with(game), "a capped late tracker is sound");
}

/**
 * Exact trackers over random games of 2-6 players: never exclude the truth,
 * keep the pool's public bounds, sample consistent worlds, and reproduce the
 * pool tracker's deals when asked for them.
 */
void test_exact_tracker_follows_random_games() {
    int games = 0;
    int steps = 0;
    int overflows = 0;
    int restarts = 0;
    int exact_steps = 0;
    std::size_t most_hypotheses = 0;
    double hypotheses_sum = 0.0;
    int samples = 0;
    for (int n = 2; n <= 6; ++n) {
        const std::uint64_t seeds = (n == 3 || n == 4) ? 200 : 8;
        for (std::uint64_t seed = 1; seed <= seeds; ++seed) {
            Game game(seats(n), MapType::base, seed * 7 + static_cast<std::uint64_t>(n), rules_for(n));
            std::vector<HandTracker> exact;
            std::vector<HandTracker> pool;
            for (int seat = 0; seat < n; ++seat) {
                exact.emplace_back(game, seat, TrackerMode::exact);
                pool.emplace_back(game, seat);
                require(exact.back().exact() && exact.back().hypothesis_count() == 1,
                        "a fresh game is the one known world");
            }
            std::mt19937_64 rng(seed);
            ++games;
            for (int step = 0; step < 1500 && !game.winning_color().has_value(); ++step) {
                const Game before = game;
                const Action action = random_action(game, rng);
                game.execute(action);
                ++steps;
                for (int seat = 0; seat < n; ++seat) {
                    const auto index = static_cast<std::size_t>(seat);
                    const bool was_exact = exact[index].exact();
                    const int was_overflows = exact[index].belief_overflows();
                    exact[index].observe(before, action, game);
                    pool[index].observe(before, action, game);
                    const HandTracker& tracker = exact[index];
                    require(
                        tracker.consistent_with(game),
                        "the exact tracker excluded the true state (n=" + std::to_string(n) +
                            ", seed=" + std::to_string(seed) + ", step=" + std::to_string(step) + ")");
                    overflows += tracker.belief_overflows() - was_overflows;
                    restarts += !was_exact && tracker.exact() && tracker.belief_overflows() == was_overflows;
                    if (tracker.exact()) {
                        ++exact_steps;
                        most_hypotheses = std::max(most_hypotheses, tracker.hypothesis_count());
                        hypotheses_sum += static_cast<double>(tracker.hypothesis_count());
                        double total = 0.0;
                        for (double weight : tracker.hypothesis_weights()) {
                            total += weight;
                        }
                        require(std::abs(total - 1.0) < 1e-9, "the weights sum to one");
                    }
                    for (int s = 0; s < n; ++s) {
                        require(tracker.hand_size(s) == pool[index].hand_size(s) &&
                                    tracker.known_resources(s) == pool[index].known_resources(s) &&
                                    tracker.unknown_count(s) == pool[index].unknown_count(s),
                                "the exact tracker keeps the pool's bounds");
                    }
                }
                if (step % 23 == 0 && !game.winning_color().has_value()) {
                    const auto observer = static_cast<std::size_t>(rng() % static_cast<std::uint64_t>(n));
                    const std::uint64_t sample_seed = seed * 1000003 + static_cast<std::uint64_t>(step);
                    std::mt19937_64 sample_rng(sample_seed);
                    const Game world = sample_world(game, exact[observer], sample_rng);
                    require(exact[observer].consistent_with(world) && pool[observer].consistent_with(world),
                            "an exact sample satisfies both trackers");
                    require_conserved(world);
                    std::mt19937_64 first(sample_seed);
                    std::mt19937_64 second(sample_seed);
                    require(sample_world(game, exact[observer], first, TrackerMode::pool)
                                .search_equivalent(sample_world(game, pool[observer], second)),
                            "deal=pool on an exact tracker is the pool tracker's deal");
                    ++samples;
                }
            }
        }
    }
    require(overflows > 0 || most_hypotheses > 100, "random games must stress the belief");
    std::cout << "tracker: exact mode over " << games << " games, " << steps << " steps: " << overflows
              << " overflows, " << restarts << " restarts, " << most_hypotheses
              << " hypotheses at most, " << hypotheses_sum / std::max(1, exact_steps)
              << " on average over " << exact_steps << " exact seat-steps, " << samples << " samples\n";
}

int main() {
    try {
        test_tracker_follows_random_games();
        test_tracker_ignores_hidden_differences(TrackerMode::pool);
        test_tracker_ignores_hidden_differences(TrackerMode::exact);
        test_sample_world_refuses_finished_games();
        test_victory_point_cards_respect_the_target();
        test_samples_match_expected_hands();
        test_steal_and_monopoly_events();
        test_self_confirmed_trade();
        test_victory_point_cap_history();
        test_friendly_robber_options_reveal_no_victory_point_cards();
        test_report_robbery_examples();
        test_report_knight_example();
        test_exact_belief_weights_steals_by_count();
        test_exact_belief_overflow_falls_back();
        test_exact_tracker_started_mid_game_enumerates_deals();
        test_exact_tracker_follows_random_games();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "tracker tests passed\n";
    return 0;
}
