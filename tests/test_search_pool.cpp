// SearchPool: equivalence with MCTSSearch in 2p, N-player max^n backup,
// finding wins, determinized searches that never read the hidden truth,
// over sampled worlds or the most likely ones.

#include <algorithm>
#include <atomic>
#include <bit>
#include <limits>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cppanatron/flat_codec.hpp"
#include "cppanatron/game.hpp"
#include "cppanatron/mcts.hpp"
#include "cppanatron/search_pool.hpp"
#include "cppanatron/tracker.hpp"

namespace {

using namespace cppanatron;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

/** Test-only encoder: a few public numbers; the evaluators read pending games. */
class ToyEncoder final : public ObservationEncoder {
public:
    [[nodiscard]] std::size_t size() const noexcept override { return 4; }
    void encode(const Game& game, int observer, std::span<float> out) const override {
        out[0] = static_cast<float>(observer);
        out[1] = static_cast<float>(game.num_turns());
        out[2] = static_cast<float>(static_cast<int>(game.current_prompt()));
        out[3] = static_cast<float>(game.playable_actions().size());
    }
};

std::vector<Color> seats(int n) {
    return {kColors.begin(), kColors.begin() + n};
}

std::uint64_t mix(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
}

/** A hash of everything in a position, hidden cards included. */
std::uint64_t signature(const Game& game) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto& [node, building] : game.board().buildings()) {
        hash = mix(hash, static_cast<std::uint64_t>(node) * 16 +
                             static_cast<std::uint64_t>(building.color) * 2 +
                             static_cast<std::uint64_t>(building.building));
    }
    for (const auto& [edge, color] : game.board().roads()) {
        hash = mix(hash, static_cast<std::uint64_t>(edge.a) * 1000 + static_cast<std::uint64_t>(edge.b) * 8 +
                             static_cast<std::uint64_t>(color));
    }
    for (const PlayerState& player : game.players()) {
        for (std::size_t i = 0; i < 5; ++i) {
            hash = mix(hash, static_cast<std::uint64_t>(player.resources[i]) * 64 +
                                 static_cast<std::uint64_t>(player.development_cards[i]));
        }
    }
    hash = mix(hash, static_cast<std::uint64_t>(game.current_player_index()));
    hash = mix(hash, static_cast<std::uint64_t>(game.current_prompt()));
    hash = mix(hash, static_cast<std::uint64_t>(game.num_turns()));
    return hash;
}

double unit(std::uint64_t hash) {
    return static_cast<double>(mix(hash, 12345) % 1000003) / 1000003.0;
}

/** Fills one row: logits over the codec and seat-relative values. */
using Evaluator = std::function<void(const Game&, std::span<float>, std::span<float>)>;

void run(SearchPool& pool, const Evaluator& evaluate, std::size_t batch = 64) {
    std::vector<float> observations(batch * pool.observation_size());
    std::vector<std::int64_t> ids(batch);
    const std::size_t width = pool.action_size();
    const auto players = static_cast<std::size_t>(pool.max_players());
    while (true) {
        const std::size_t rows = pool.collect(observations, ids);
        if (rows == 0) {
            return;
        }
        std::vector<float> logits(rows * width);
        std::vector<float> values(rows * players);
        for (std::size_t row = 0; row < rows; ++row) {
            evaluate(
                pool.pending_game(row), std::span<float>(logits).subspan(row * width, width),
                std::span<float>(values).subspan(row * players, players));
        }
        pool.apply(logits, values);
    }
}

void hashed_logits(const Game& game, std::span<float> logits) {
    const std::uint64_t hash = signature(game);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        logits[i] = static_cast<float>(2.0 * unit(mix(hash, i)) - 1.0);
    }
}

double hashed_value(const Game& game) { return 0.1 + 0.8 * unit(signature(game)); }

/**
 * 2 players, PUCT, the same priors and values (v and 1 - v; MCTSSearch gets
 * 2v - 1), no chance on the way: both searches must visit identically.
 */
void test_two_player_matches_mcts_search() {
    for (std::uint64_t seed = 1; seed <= 4; ++seed) {
        Game game(seats(2), MapType::base, seed);
        std::mt19937_64 rng(seed);
        const std::uint64_t steps = (seed - 1) % 2;
        for (std::uint64_t step = 0; step < steps; ++step) {
            const auto& actions = game.playable_actions();
            game.execute(actions[rng() % actions.size()]);
        }
        const int simulations = 300;
        const auto codec = std::make_shared<FlatActionCodec>(2, MapType::base);
        std::vector<float> logits(codec->size());

        MCTSSearch reference(game, FlatActionSpace(2, MapType::base), 1.5, seed);
        hashed_logits(game, logits);
        reference.initialize_root(logits);
        for (int i = 0; i < simulations; ++i) {
            const Game* leaf = reference.select_leaf();
            require(leaf != nullptr, "no terminal leaves in the opening");
            hashed_logits(*leaf, logits);
            reference.evaluate_leaf(logits, 2.0 * hashed_value(*leaf) - 1.0);
        }
        // The two searches draw chance outcomes differently (MCTSSearch samples
        // a one-outcome draw at every step), so no path may reach the first
        // roll: the 2p initial placement is 8 plies, `steps` already played.
        require(reference.metrics().maximum_depth + steps <= 8,
                "the comparison must stay clear of dice");

        SearchConfig config;
        config.c_puct = 0.75;
        config.fpu_value = 0.5;
        config.skip_forced = false;
        config.simulations = simulations;
        config.max_players = 2;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        const auto id = pool.add(game, seed);
        run(pool, [](const Game& leaf, std::span<float> row, std::span<float> values) {
            hashed_logits(leaf, row);
            values[0] = static_cast<float>(hashed_value(leaf));
            values[1] = 1.0F - values[0];
        });
        require(pool.done(id), "the search must finish");
        const auto expected = reference.root_visits();
        const auto actual = pool.visits(id);
        require(std::equal(expected.begin(), expected.end(), actual.begin()),
                "2p visits must equal MCTSSearch's");
        const auto expected_q = reference.root_action_values();
        const auto actual_q = pool.q(id);
        for (std::size_t i = 0; i < expected_q.size(); ++i) {
            require(std::isnan(expected_q[i]) == std::isnan(actual_q[i]), "same visited actions");
            if (!std::isnan(expected_q[i])) {
                // Values pass through float in the pool's value rows.
                require(std::abs((expected_q[i] + 1.0) / 2.0 - actual_q[i]) < 1e-6,
                        "2p action values must equal MCTSSearch's");
            }
        }
        require(pool.select(id) == static_cast<std::size_t>(std::max_element(expected.begin(), expected.end()) -
                                                           expected.begin()),
                "the same move");
    }
}

/** Placements of a random initial build phase, as a setup to edit. */
std::vector<PlayerSetup> random_placements(int n, std::uint64_t seed) {
    Game game(seats(n), MapType::base, seed);
    std::mt19937_64 rng(seed);
    while (game.is_initial_build_phase()) {
        const auto& actions = game.playable_actions();
        game.execute(actions[rng() % actions.size()]);
    }
    std::vector<PlayerSetup> players;
    for (const PlayerState& state : game.players()) {
        PlayerSetup setup;
        setup.settlements = state.settlements;
        setup.roads = state.roads;
        players.push_back(setup);
    }
    return players;
}

void uniform_logits(std::span<float> logits) { std::fill(logits.begin(), logits.end(), 0.0F); }

/**
 * Three players, three points win. Blue can build the winning city next
 * turn; Red holds a monopoly. Only a monopoly on ore or wheat stops Blue.
 */
void test_finds_the_saving_move() {
    for (RootSelection root : {RootSelection::puct, RootSelection::gumbel}) {
        GameRules rules;
        rules.victory_points_to_win = 3;
        auto players = random_placements(3, 21);
        players[0].development_cards = {0, 0, 1, 0, 0};
        players[1].resources = {0, 0, 0, 2, 3};
        Game game(seats(3), MapType::base, 21, rules);
        GameSetup setup;
        setup.players = players;
        setup.has_rolled = true;
        game.apply_setup(setup);

        const auto codec = std::make_shared<FlatActionCodec>(3, MapType::base);
        SearchConfig config;
        config.simulations = 600;
        config.max_players = 3;
        config.root = root;
        config.gumbel_scale = 0.0;
        config.max_pending = 4;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        const auto id = pool.add(game, 5);
        run(pool, [](const Game&, std::span<float> logits, std::span<float> values) {
            uniform_logits(logits);
            std::fill(values.begin(), values.end(), 1.0F / 3.0F);
        });
        const Action chosen = codec->decode(game, pool.select(id));
        require(chosen.type == ActionType::play_monopoly &&
                    (std::get<Resource>(chosen.value) == Resource::ore ||
                     std::get<Resource>(chosen.value) == Resource::wheat),
                "the search must stop Blue's winning city");
        const auto q = pool.q(id);
        const std::size_t end_turn = codec->index(game, {Color::red, ActionType::end_turn, std::monostate{}});
        // Leaves are worth 1/3 until the search finds Blue's win below them.
        require(q[end_turn] < 0.25 && q[end_turn] < q[pool.select(id)] - 0.1, "ending the turn loses");
    }
}

/**
 * 3 players. After Red's first road, Blue places a settlement; every
 * position below that choice is worth t(s) with Blue's component
 * anti-correlated with Red's. Blue must pick by its own component, and Red's
 * value of the road must be Red's component of Blue's choice.
 */
void test_max_n_backup() {
    const auto table = [](const Game& game) {
        std::array<double, 3> t{1.0 / 3, 1.0 / 3, 1.0 / 3};
        const auto& blue = game.players()[1].settlements;
        if (!blue.empty()) {
            const int s = blue.front();
            t[1] = 0.2 + 0.6 * static_cast<double>((s * 37) % 54) / 54.0;
            t[0] = 1.0 - t[1];
            t[2] = 0.5;
        }
        return t;
    };
    Game game(seats(3), MapType::base, 3);
    game.execute(game.playable_actions().front());  // Red's settlement; Red places a road next.
    require(game.current_prompt() == ActionPrompt::build_initial_road, "Red's road comes next");

    const auto codec = std::make_shared<FlatActionCodec>(3, MapType::base);
    SearchConfig config;
    config.simulations = static_cast<int>(game.playable_actions().size());
    config.max_players = 3;
    config.fpu_value = 1.0;
    SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
    const auto evaluate = [&](const Game& leaf, std::span<float> logits, std::span<float> values) {
        uniform_logits(logits);
        const auto t = table(leaf);
        for (int slot = 0; slot < 3; ++slot) {
            values[static_cast<std::size_t>(slot)] =
                static_cast<float>(t[static_cast<std::size_t>((leaf.current_player_index() + slot) % 3)]);
        }
    };

    // One visit per road: each road's Q is its child's own evaluation.
    const auto id = pool.add(game, 1);
    run(pool, evaluate);
    const auto visits = pool.visits(id);
    const auto q = pool.q(id);
    for (const Action& action : game.playable_actions()) {
        const std::size_t index = codec->index(game, action);
        require(visits[index] == 1, "optimistic first-play urgency visits every road once");
        require(std::abs(q[index] - 1.0 / 3.0) < 1e-6, "a road's Q is its child's value for Red");
    }
    const auto value = pool.value(id);
    require(std::abs(value[0] + value[1] + value[2] - 1.0) < 1e-5, "root values stay normalized");

    // Deep search: Blue concentrates on its best settlement.
    config.simulations = 6000;
    config.max_pending = 1;
    SearchPool deep(codec, std::make_shared<ToyEncoder>(), config);
    const auto deep_id = deep.add(game, 2);
    run(deep, evaluate);
    const std::size_t road = deep.select(deep_id);
    double best_blue = 0.0;
    std::size_t best_settlement = 0;
    Game after = game;
    after.execute(codec->decode(game, road));
    for (const Action& action : after.playable_actions()) {
        Game next = after;
        next.execute(action);
        const auto t = table(next);
        if (t[1] > best_blue) {
            best_blue = t[1];
            best_settlement = codec->index(after, action);
        }
    }
    const double red_value = deep.q(deep_id)[road];
    require(red_value < 1.0 - best_blue + 0.08,
            "Red's value must follow Blue's choice, not Red's preference");
    require(deep.advance(deep_id, road), "the road child is deterministic");
    const auto blue_visits = deep.visits(deep_id);
    require(static_cast<std::size_t>(std::max_element(blue_visits.begin(), blue_visits.end()) -
                                     blue_visits.begin()) == best_settlement,
            "Blue must choose by its own value");
    require(std::abs(deep.q(deep_id)[best_settlement] - best_blue) < 1e-6,
            "after re-rooting Q is for Blue, the new root player");
    require(!deep.done(deep_id), "an advanced search has a fresh budget");
    run(deep, evaluate);
    require(deep.done(deep_id), "the advanced search finishes");
}

/** A position some way into a random game and the trackers that followed it. */
struct Played {
    Game game;
    std::vector<HandTracker> trackers;
};

Played play_random(int n, std::uint64_t seed, int steps) {
    Played played{Game(seats(n), MapType::base, seed), {}};
    for (int seat = 0; seat < n; ++seat) {
        played.trackers.emplace_back(played.game, seat);
    }
    std::mt19937_64 rng(seed);
    for (int step = 0; step < steps || played.game.playable_actions().size() < 2; ++step) {
        const auto& actions = played.game.playable_actions();
        const Action action = actions[rng() % actions.size()];
        const Game before = played.game;
        played.game.execute(action);
        for (auto& tracker : played.trackers) {
            tracker.observe(before, action, played.game);
        }
        require(!played.game.winning_color().has_value(), "the random game must still be on");
    }
    return played;
}

/** Determinized searches of two games the root player cannot tell apart agree exactly. */
void test_determinized_search_ignores_hidden_truth() {
    for (RootSelection root : {RootSelection::puct, RootSelection::gumbel}) {
        for (int n : {2, 4}) {
            const Played played = play_random(n, 40 + static_cast<std::uint64_t>(n), 220);
            const Game& truth = played.game;
            const HandTracker& tracker =
                played.trackers[static_cast<std::size_t>(truth.current_player_index())];
            std::mt19937_64 rng(9);
            const Game alternative = sample_world(truth, tracker, rng);
            require(!alternative.search_equivalent(truth), "the alternative must hide other cards");

            const auto codec = std::make_shared<FlatActionCodec>(n, MapType::base);
            SearchConfig config;
            config.simulations = 160;
            config.determinizations = 4;
            config.max_players = n;
            config.root = root;
            config.max_pending = 3;
            config.dirichlet_alpha = 0.3;
            SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
            const auto a = pool.add(truth, 77, &tracker);
            const auto b = pool.add(alternative, 77, &tracker);
            // Values depend on hidden cards too, so a leak would show.
            run(pool, [n](const Game& leaf, std::span<float> logits, std::span<float> values) {
                hashed_logits(leaf, logits);
                const double v = hashed_value(leaf);
                for (int slot = 0; slot < n; ++slot) {
                    values[static_cast<std::size_t>(slot)] = static_cast<float>(slot == 0 ? v : (1.0 - v) / (n - 1));
                }
            }, 7);
            require(pool.visits(a) == pool.visits(b), "determinized visits must not depend on the truth");
            const auto policy_a = pool.policy_target(a);
            const auto policy_b = pool.policy_target(b);
            require(policy_a == policy_b, "policy targets must not depend on the truth");
            require(pool.select(a) == pool.select(b), "moves must not depend on the truth");
            const auto stats = pool.stats(a);
            require(stats.simulations == 160, "every simulation is counted");
            require(stats.worlds >= 1 && stats.worlds <= 4, "up to four distinct worlds");
            const auto visits = pool.visits(a);
            require(std::accumulate(visits.begin(), visits.end(), 0U) == 160, "root visits sum to the budget");

            bool threw = false;
            try {
                (void)pool.add(truth, 1, nullptr);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            require(threw, "determinized search needs a tracker");
        }
    }
}

/** Threads, batch sizes and pending limits change speed, not results. */
void test_parallel_and_batched_runs_agree() {
    const Played played = play_random(4, 8, 180);
    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    std::vector<std::vector<std::uint32_t>> results;
    struct Setting {
        int threads;
        std::size_t batch;
        int pending;
        bool release;
    };
    for (const Setting& setting : std::vector<Setting>{
             {1, 64, 2, true}, {4, 64, 2, true}, {1, 64, 1, true}, {3, 5, 1, true}, {1, 64, 2, false}}) {
        SearchConfig config;
        config.simulations = 120;
        config.max_players = 4;
        config.max_pending = setting.pending;
        config.threads = setting.threads;
        config.release_games = setting.release;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        std::vector<std::int64_t> ids;
        for (std::uint64_t seed = 0; seed < 12; ++seed) {
            ids.push_back(pool.add(played.game, seed));
        }
        run(pool, [](const Game& leaf, std::span<float> logits, std::span<float> values) {
            hashed_logits(leaf, logits);
            std::fill(values.begin(), values.end(), static_cast<float>(hashed_value(leaf) / 2));
        }, setting.batch);
        std::vector<std::uint32_t> all;
        for (auto id : ids) {
            require(pool.done(id), "every search finishes");
            const auto visits = pool.visits(id);
            all.insert(all.end(), visits.begin(), visits.end());
        }
        results.push_back(all);
    }
    require(results[0] == results[1], "threads must not change results");
    // With one leaf in flight per search, a small batch only delays searches.
    require(results[2] == results[3], "batch size must not change results");
    require(results[0] == results[4], "releasing games must not change results");
}

void test_edge_cases() {
    const auto codec = std::make_shared<FlatActionCodec>(2, MapType::base);
    SearchConfig config;
    config.simulations = 16;
    config.max_players = 2;
    SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
    // A finished game is done at once.
    GameRules rules;
    rules.victory_points_to_win = 2;
    Game finished(seats(2), MapType::base, 1, rules);
    std::mt19937_64 rng(1);
    while (!finished.winning_color().has_value()) {
        const auto& actions = finished.playable_actions();
        finished.execute(actions[rng() % actions.size()]);
    }
    const auto id = pool.add(finished, 1);
    require(pool.done(id), "a finished game needs no search");
    std::vector<float> observations(4 * pool.observation_size());
    std::vector<std::int64_t> ids(4);
    require(pool.collect(observations, ids) == 0, "nothing to evaluate");
    pool.remove(id);
    require(pool.size() == 0, "removed");

    // Roots come first; collect without apply is refused.
    const auto live = pool.add(Game(seats(2), MapType::base, 2), 3);
    require(pool.collect(observations, ids) == 1 && ids[0] == live, "the root is the first row");
    bool threw = false;
    try {
        (void)pool.collect(observations, ids);
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw, "collect twice without apply");
    std::vector<float> logits(pool.action_size(), 0.0F);
    std::vector<float> values{0.5F, 0.5F};
    pool.apply(logits, values);
    run(pool, [](const Game&, std::span<float> l, std::span<float> v) {
        uniform_logits(l);
        std::fill(v.begin(), v.end(), 0.5F);
    });
    require(pool.done(live), "done");
    const auto policy = pool.policy_target(live);
    require(std::abs(std::accumulate(policy.begin(), policy.end(), 0.0) - 1.0) < 1e-5, "a distribution");
    const auto stats = pool.stats(live);
    require(stats.simulations == 16 && stats.evaluations == 17, "16 simulations and the root");

    // Two players right after placement: dice totals that pay nobody reach
    // the same position and share a child.
    GameSetup setup;
    setup.players = random_placements(2, 4);
    Game placed(seats(2), MapType::base, 4);
    placed.apply_setup(setup);
    config.simulations = 400;
    SearchPool chance(codec, std::make_shared<ToyEncoder>(), config);
    const auto rolled = chance.add(placed, 1);
    run(chance, [](const Game& leaf, std::span<float> l, std::span<float> v) {
        hashed_logits(leaf, l);
        std::fill(v.begin(), v.end(), 0.5F);
    });
    require(chance.stats(rolled).chance_nodes > 0, "rolls are chance nodes");
    require(chance.stats(rolled).coalesced_outcomes > 0, "identical chance outcomes must share children");
}

}  // namespace

/**
 * Friendly robber (OpenHex's rule, public points): the root player's robber
 * options do not depend on others' VP cards, so every determinized root
 * offers exactly the real options, also when the tracker is misled.
 */
void test_determinized_roots_offer_the_real_options() {
    GameRules rules;
    rules.friendly_robber = true;
    auto players = random_placements(4, 31);
    players[0].development_cards = {1, 0, 0, 0, 0};
    players[1].development_cards = {0, 0, 0, 0, 1};  // 2 public points, 3 actual: shielded
    players[2].development_cards = {1, 0, 0, 0, 0};
    Game game(seats(4), MapType::base, 31, rules);
    GameSetup setup;
    setup.players = players;
    game.apply_setup(setup);
    const Game before = game;
    const Action knight{Color::red, ActionType::play_knight_card, std::monostate{}};
    game.execute(knight);

    HandTracker informed(before, 0);
    informed.observe(before, knight, game);
    // Misled: it believes seat 1 holds no VP card.
    HandTracker misled(before, 0);
    for (const auto& event : HandTracker::events(before, knight, game, 0)) {
        misled.apply(event);
    }
    misled.apply(TrackerEvent::victory_point_cards(1, 0, 0));

    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    std::vector<std::uint8_t> legal(codec->size());
    codec->mask(game, legal);
    std::mt19937_64 rng(3);
    for (const HandTracker* tracker : {&informed, &misled}) {
        for (int i = 0; i < 20; ++i) {
            std::vector<std::uint8_t> world_legal(codec->size());
            codec->mask(sample_world(game, *tracker, rng), world_legal);
            require(world_legal == legal, "every world offers exactly the real options");
        }
        SearchConfig config;
        config.simulations = 64;
        config.max_players = 4;
        config.determinizations = 4;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        const auto id = pool.add(game, 3, tracker);
        run(pool, [](const Game& leaf, std::span<float> logits, std::span<float> values) {
            hashed_logits(leaf, logits);
            std::fill(values.begin(), values.end(), 0.25F);
        });
        const auto visits = pool.visits(id);
        for (std::size_t i = 0; i < visits.size(); ++i) {
            require(visits[i] == 0 || legal[i] != 0, "determinized roots only search real options");
        }
        require(legal[pool.select(id)] != 0, "the chosen move is legal in the real game");
        require(std::accumulate(visits.begin(), visits.end(), 0U) == 64, "every simulation ran");
    }
}

/** Red holds a knight and a monopoly; Blue holds one wheat, White two cards. */
Game knight_and_monopoly_position() {
    auto players = random_placements(4, 12);
    players[0].development_cards = {1, 0, 1, 0, 0};
    players[1].resources = {0, 0, 0, 1, 0};
    players[2].resources = {1, 0, 0, 1, 0};
    players[3].development_cards = {0, 1, 0, 0, 0};
    Game game(seats(4), MapType::base, 12);
    GameSetup setup;
    setup.players = players;
    game.apply_setup(setup);
    return game;
}

/** Search `game` with logits favouring `favoured` at the root; returns the pool's id. */
std::int64_t search_favouring(SearchPool& pool, const Game& game, std::size_t favoured, const HandTracker* tracker) {
    const auto id = pool.add(game, 4, tracker);
    const int turn = game.num_turns();
    run(pool, [&](const Game& leaf, std::span<float> logits, std::span<float> values) {
        uniform_logits(logits);
        if (leaf.num_turns() == turn && leaf.current_prompt() == game.current_prompt()) {
            logits[favoured] = 4.0F;
        }
        std::fill(values.begin(), values.end(), 0.25F);
    });
    return id;
}

/**
 * Determinized searches must not re-root through a drawn outcome: stealing
 * Blue's only card yields the world's card, not the real one.
 */
void test_determinized_advance_refuses_drawn_outcomes() {
    Game game = knight_and_monopoly_position();
    HandTracker tracker(game, 0);
    const Game before = game;
    const Action knight{Color::red, ActionType::play_knight_card, std::monostate{}};
    game.execute(knight);
    tracker.observe(before, knight, game);
    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    const auto steal = std::find_if(game.playable_actions().begin(), game.playable_actions().end(), [](const Action& a) {
        const auto& move = std::get<RobberMove>(a.value);
        return move.victim == Color::blue;
    });
    require(steal != game.playable_actions().end(), "Red can rob Blue");
    const std::size_t index = codec->index(game, *steal);
    for (const int determinizations : {0, 3}) {
        SearchConfig config;
        config.simulations = 120;
        config.max_players = 4;
        config.determinizations = determinizations;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        const auto id = search_favouring(pool, game, index, determinizations > 0 ? &tracker : nullptr);
        require(pool.visits(id)[index] > 0, "the steal was searched");
        // Blue's one card is exactly known in the real game, so the oracle may re-root.
        require(pool.advance(id, index) == (determinizations == 0),
                "only the oracle re-roots through a steal");
    }
}

/** A monopoly reveals every seat's count; determinized worlds would contradict it. */
void test_determinized_advance_refuses_monopoly() {
    const Game game = knight_and_monopoly_position();
    const HandTracker tracker(game, 0);
    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    const std::size_t index = codec->index(game, {Color::red, ActionType::play_monopoly, Resource::wheat});
    for (const int determinizations : {0, 3}) {
        SearchConfig config;
        config.simulations = 120;
        config.max_players = 4;
        config.determinizations = determinizations;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        const auto id = search_favouring(pool, game, index, determinizations > 0 ? &tracker : nullptr);
        require(pool.visits(id)[index] > 0, "the monopoly was searched");
        require(pool.advance(id, index) == (determinizations == 0),
                "only the oracle re-roots through a monopoly");
    }
}

/**
 * Friendly robber (public points): Red's robber options after its knight are
 * the same in every world, so a determinized search re-roots through the
 * knight to Red's robber decision like through any own deterministic move.
 */
void test_determinized_advance_keeps_the_friendly_robber_prompt() {
    GameRules rules;
    rules.friendly_robber = true;
    auto players = random_placements(4, 12);
    players[0].development_cards = {1, 0, 0, 0, 0};
    players[1].development_cards = {0, 0, 0, 0, 1};  // 2 public points, 3 actual
    players[1].resources = {0, 0, 0, 1, 0};
    Game game(seats(4), MapType::base, 12, rules);
    GameSetup setup;
    setup.players = players;
    game.apply_setup(setup);
    const HandTracker tracker(game, 0);
    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    const std::size_t index = codec->index(game, {Color::red, ActionType::play_knight_card, std::monostate{}});
    SearchConfig config;
    config.simulations = 240;
    config.max_players = 4;
    config.determinizations = 3;
    SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
    const auto id = search_favouring(pool, game, index, &tracker);
    require(pool.visits(id)[index] > 0, "the knight was searched");
    require(pool.advance(id, index), "the friendly robber prompt re-roots like any own decision");
    game.execute({Color::red, ActionType::play_knight_card, std::monostate{}});
    require(game.current_prompt() == ActionPrompt::move_robber, "Red moves the robber next");
    std::vector<std::uint8_t> legal(codec->size());
    codec->mask(game, legal);
    run(pool, [](const Game& leaf, std::span<float> logits, std::span<float> values) {
        hashed_logits(leaf, logits);
        std::fill(values.begin(), values.end(), 0.25F);
    });
    const auto visits = pool.visits(id);
    for (std::size_t i = 0; i < visits.size(); ++i) {
        require(visits[i] == 0 || legal[i] != 0, "the re-rooted search visits only real robber moves");
    }
}

/** ToyEncoder that throws on its `fail_at`-th call. */
class FailingEncoder final : public ObservationEncoder {
public:
    explicit FailingEncoder(int fail_at) : fail_at_(fail_at) {}
    [[nodiscard]] std::size_t size() const noexcept override { return 4; }
    void encode(const Game& game, int observer, std::span<float> out) const override {
        if (++calls_ == fail_at_) {
            throw std::runtime_error("encoder failed");
        }
        ToyEncoder().encode(game, observer, out);
    }

private:
    int fail_at_;
    mutable std::atomic<int> calls_{0};
};

/** FlatActionCodec whose decode throws on its `fail_at`-th call. */
class FailingCodec final : public ActionCodec {
public:
    FailingCodec(int players, int fail_at) : inner_(players, MapType::base), fail_at_(fail_at) {}
    [[nodiscard]] std::size_t size() const noexcept override { return inner_.size(); }
    [[nodiscard]] std::size_t index(const Game& game, const Action& action) const override {
        return inner_.index(game, action);
    }
    [[nodiscard]] Action decode(const Game& game, std::size_t index) const override {
        if (++calls_ == fail_at_) {
            throw std::runtime_error("codec failed");
        }
        return inner_.decode(game, index);
    }
    void mask(const Game& game, std::span<std::uint8_t> out) const override { inner_.mask(game, out); }

private:
    FlatActionCodec inner_;
    int fail_at_;
    mutable std::atomic<int> calls_{0};
};

/** collect/apply failures withdraw what they started; the pool stays usable. */
void test_failures_leave_the_pool_usable() {
    const Played played = play_random(3, 15, 150);
    const auto check = [](SearchPool& pool, const std::vector<std::int64_t>& ids, int simulations) {
        for (const auto id : ids) {
            require(pool.done(id), "every search finishes");
            const auto visits = pool.visits(id);
            require(std::accumulate(visits.begin(), visits.end(), 0U) == static_cast<unsigned>(simulations),
                    "no simulation is lost or stranded");
            require(pool.stats(id).simulations == static_cast<std::uint64_t>(simulations), "all counted");
            pool.remove(id);
        }
    };
    const auto evaluate = [](const Game& leaf, std::span<float> logits, std::span<float> values) {
        hashed_logits(leaf, logits);
        std::fill(values.begin(), values.end(), static_cast<float>(hashed_value(leaf) / 3));
    };
    for (const RootSelection root : {RootSelection::puct, RootSelection::gumbel}) {
        SearchConfig config;
        config.simulations = 60;
        config.max_players = 3;
        config.max_pending = 4;
        config.root = root;
        const auto codec = std::make_shared<FlatActionCodec>(3, MapType::base);

        // A non-finite value is rejected before anything changes; a retry works.
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        std::vector<std::int64_t> ids{pool.add(played.game, 1), pool.add(played.game, 2)};
        std::vector<float> observations(16 * pool.observation_size());
        std::vector<std::int64_t> rows(16);
        for (int round = 0; round < 3; ++round) {
            const std::size_t n = pool.collect(observations, rows);
            std::vector<float> logits(n * pool.action_size(), 0.0F);
            std::vector<float> values(n * 3, 0.3F);
            if (round == 2) {
                values[1] = std::numeric_limits<float>::quiet_NaN();
                bool threw = false;
                try {
                    pool.apply(logits, values);
                } catch (const std::invalid_argument&) {
                    threw = true;
                }
                require(threw && pool.pending() == n, "a NaN row is rejected and the rows stay pending");
                values[1] = 0.3F;
            }
            pool.apply(logits, values);
        }
        // Rows can be withdrawn unevaluated.
        require(pool.collect(observations, rows) > 0, "more leaves");
        pool.cancel();
        require(pool.pending() == 0, "cancel withdraws every row");
        run(pool, evaluate, 16);
        check(pool, ids, 60);

        // An encoder or codec failure inside collect withdraws the issued leaves.
        for (int failing = 0; failing < 2; ++failing) {
            std::shared_ptr<const ActionCodec> failing_codec =
                failing == 0 ? std::shared_ptr<const ActionCodec>(codec)
                             : std::make_shared<FailingCodec>(3, 40);
            std::shared_ptr<const ObservationEncoder> encoder =
                failing == 0 ? std::shared_ptr<const ObservationEncoder>(std::make_shared<FailingEncoder>(25))
                             : std::make_shared<ToyEncoder>();
            SearchPool fragile(failing_codec, encoder, config);
            std::vector<std::int64_t> fragile_ids{fragile.add(played.game, 3), fragile.add(played.game, 4)};
            int failures = 0;
            while (true) {
                std::size_t n = 0;
                try {
                    n = fragile.collect(observations, rows);
                } catch (const std::runtime_error&) {
                    ++failures;
                    require(fragile.pending() == 0, "a failed collect leaves nothing pending");
                    continue;
                }
                if (n == 0) {
                    break;
                }
                std::vector<float> logits(n * fragile.action_size());
                std::vector<float> values(n * 3);
                for (std::size_t row = 0; row < n; ++row) {
                    evaluate(
                        fragile.pending_game(row),
                        std::span<float>(logits).subspan(row * fragile.action_size(), fragile.action_size()),
                        std::span<float>(values).subspan(row * 3, 3));
                }
                fragile.apply(logits, values);
            }
            require(failures == 1, "the failure happened once");
            check(fragile, fragile_ids, 60);
        }
    }
}

/** New searches go to the least loaded threads, whatever their ids. */
/** A tracker with determinizations=0 would silently make a fair search an oracle. */
void test_tracker_needs_determinizations() {
    const Played played = play_random(3, 6, 100);
    const auto codec = std::make_shared<FlatActionCodec>(3, MapType::base);
    SearchConfig config;
    config.max_players = 3;
    SearchPool oracle(codec, std::make_shared<ToyEncoder>(), config);
    const HandTracker& tracker = played.trackers[static_cast<std::size_t>(played.game.current_player_index())];
    bool threw = false;
    try {
        (void)oracle.add(played.game, 1, &tracker);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw && oracle.size() == 0, "a tracker without determinizations is refused");
    (void)oracle.add(played.game, 1, &tracker, true);
    require(oracle.size() == 1, "unless the caller says it is intended");
}

void test_searches_spread_over_threads() {
    const auto codec = std::make_shared<FlatActionCodec>(2, MapType::base);
    SearchConfig config;
    config.max_players = 2;
    config.threads = 4;
    SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
    const Game game(seats(2), MapType::base, 1);
    std::vector<std::int64_t> ids;
    for (std::uint64_t seed = 0; seed < 8; ++seed) {
        ids.push_back(pool.add(game, seed));
    }
    for (std::size_t i = 1; i < ids.size(); i += 2) {
        pool.remove(ids[i]);  // empties every other thread
    }
    for (std::uint64_t seed = 8; seed < 12; ++seed) {
        (void)pool.add(game, seed);
    }
    std::array<int, 4> load{};
    for (const auto id : pool.ids()) {
        ++load[pool.stats(id).lane];
    }
    require(load == std::array<int, 4>{2, 2, 2, 2}, "every thread runs two searches");
}

/**
 * Gumbel at temperature 0 plays the Sequential Halving winner: the finalist
 * with the best noise + logits + sigma, even when the last phase is uneven
 * (halving_phases(5, 16) ends (2, 3), so one finalist gets an extra visit).
 * With gumbel_scale 0 the score is log policy_target + const.
 */
void test_gumbel_plays_the_best_scoring_finalist() {
    const auto codec = std::make_shared<FlatActionCodec>(3, MapType::base);
    SearchConfig config;
    config.root = RootSelection::gumbel;
    config.gumbel_scale = 0.0;
    config.gumbel_considered = 5;
    config.simulations = 16;
    config.max_players = 3;
    SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
    std::vector<std::int64_t> ids;
    for (std::uint64_t seed = 0; seed < 400 && ids.size() < 60; ++seed) {
        Game game(seats(3), MapType::base, seed);
        std::mt19937_64 rng(seed);
        // Initial placements and turns of play.
        for (std::uint64_t step = 0; step < seed * 7 % 60; ++step) {
            const auto& actions = game.playable_actions();
            game.execute(actions[rng() % actions.size()]);
        }
        if (game.playable_actions().size() >= 5) {
            ids.push_back(pool.add(game, seed));
        }
    }
    run(pool, [](const Game& leaf, std::span<float> logits, std::span<float> values) {
        hashed_logits(leaf, logits);
        const double v = hashed_value(leaf);
        values[0] = static_cast<float>(v);
        values[1] = static_cast<float>((1.0 - v) / 2.0);
        values[2] = values[1];
    });
    int uneven = 0;
    for (const std::int64_t id : ids) {
        const auto visits = pool.visits(id);
        const auto target = pool.policy_target(id);
        std::vector<std::size_t> order(visits.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return visits[a] > visits[b]; });
        const std::size_t first = order[0];
        const std::size_t second = order[1];
        require(visits[first] + visits[second] >= 12, "two finalists share the last phases");
        uneven += visits[first] != visits[second];
        if (target[first] == target[second]) {
            continue;
        }
        const std::size_t best = target[first] > target[second] ? first : second;
        require(pool.select(id) == best, "temperature 0 plays the best-scoring finalist");
    }
    require(ids.size() > 40 && uneven > 20, "the test must cover uneven last phases");
}

/** A play-turn position some way in: Red holds `hand` after a random placement. */
Game hand_position(const std::array<int, 5>& hand) {
    auto players = random_placements(4, 12);
    players[0].resources = hand;
    Game game(seats(4), MapType::base, 12);
    GameSetup setup;
    setup.players = players;
    setup.has_rolled = true;
    game.apply_setup(setup);
    return game;
}

Evaluator hashed_evaluator(int n) {
    return [n](const Game& leaf, std::span<float> logits, std::span<float> values) {
        hashed_logits(leaf, logits);
        const double v = hashed_value(leaf);
        for (int slot = 0; slot < n; ++slot) {
            values[static_cast<std::size_t>(slot)] = static_cast<float>(slot == 0 ? v : (1.0 - v) / (n - 1));
        }
    };
}

/**
 * Two roads (or two bank trades) in either order reach one position: with
 * transpositions the second order links to the first's node. The search
 * stays a complete, legal search either way.
 */
void test_transpositions_merge_reordered_actions() {
    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    for (const auto& hand : {std::array<int, 5>{4, 4, 0, 0, 0}, std::array<int, 5>{0, 0, 4, 4, 0}}) {
        const Game game = hand_position(hand);
        require(game.playable_actions().size() >= 3, "several builds or trades are affordable");
        std::vector<std::uint8_t> legal(codec->size());
        codec->mask(game, legal);
        for (const bool transpositions : {false, true}) {
            SearchConfig config;
            config.simulations = 256;
            config.max_players = 4;
            config.transpositions = transpositions;
            SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
            const auto id = pool.add(game, 1);
            run(pool, hashed_evaluator(4));
            const auto stats = pool.stats(id);
            require((stats.transpositions > 0) == transpositions, "reordered actions link only with the option");
            const auto visits = pool.visits(id);
            require(std::accumulate(visits.begin(), visits.end(), 0U) == 256, "root visits sum to the budget");
            require(stats.simulations == 256 && pool.done(id), "every simulation ran");
            require(legal[pool.select(id)] != 0, "the chosen move is legal");
            const auto value = pool.value(id);
            for (int seat = 0; seat < 4; ++seat) {
                const float v = value[static_cast<std::size_t>(seat)];
                require(v >= 0.0F && v <= 1.0F, "root values are win probabilities");
            }
        }
    }
}

/** Threads, batch sizes and pool sizes do not change a search over a DAG either. */
void test_transpositions_are_deterministic() {
    struct Setting {
        int threads;
        std::size_t batch;
        int copies;
    };
    for (const auto [n, seed] : std::vector<std::pair<int, std::uint64_t>>{{4, 21}, {2, 5}}) {
        const Played played = play_random(n, seed, 180);
        const HandTracker& tracker =
            played.trackers[static_cast<std::size_t>(played.game.current_player_index())];
        const auto codec = std::make_shared<FlatActionCodec>(n, MapType::base);
        for (const int determinizations : {0, 3}) {
            std::vector<std::vector<std::uint32_t>> results;
            for (const Setting& setting :
                 {Setting{1, 64, 1}, Setting{4, 8, 1}, Setting{3, 5, 4}, Setting{4, 64, 4}}) {
                SearchConfig config;
                config.simulations = 96;
                config.max_players = n;
                config.transpositions = true;
                config.threads = setting.threads;
                config.determinizations = determinizations;
                SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
                std::vector<std::int64_t> ids;
                for (int copy = 0; copy < setting.copies; ++copy) {
                    ids.push_back(pool.add(played.game, 7 + static_cast<std::uint64_t>(copy),
                                           determinizations > 0 ? &tracker : nullptr));
                }
                run(pool, hashed_evaluator(n), setting.batch);
                require(pool.stats(ids.front()).transpositions > 0, "the position has transpositions");
                results.push_back(pool.visits(ids.front()));
            }
            for (std::size_t i = 1; i < results.size(); ++i) {
                require(results[i] == results[0], "threads, batches and pool size must not change a DAG search");
            }
        }
    }
}

/**
 * Off, the search is the one from before the table existed: visit vectors
 * pinned from that build (hashed evaluator, 64 simulations, seed 1).
 */
void test_transpositions_off_is_bitwise_today() {
    struct Pin {
        int players;
        std::uint64_t seed;
        int steps;
        std::uint64_t nodes;
        std::vector<std::pair<std::size_t, std::uint32_t>> visits;
    };
    const std::vector<Pin> pins{
        {4, 8, 180, 139, {{186, 12}, {223, 8}, {224, 16}, {225, 18}, {226, 10}}},
        {2, 5, 120, 223,
         {{248, 2}, {250, 2}, {252, 4}, {254, 16}, {255, 1}, {257, 1}, {260, 2}, {262, 2}, {264, 5},
          {265, 1}, {270, 3}, {272, 3}, {274, 5}, {275, 1}, {278, 3}, {280, 5}, {282, 5}, {284, 3}}},
    };
    for (const Pin& pin : pins) {
        const Played played = play_random(pin.players, pin.seed, pin.steps);
        const auto codec = std::make_shared<FlatActionCodec>(pin.players, MapType::base);
        SearchConfig config;
        config.simulations = 64;
        config.max_players = pin.players;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        const auto id = pool.add(played.game, 1);
        run(pool, hashed_evaluator(pin.players));
        std::vector<std::uint32_t> expected(codec->size(), 0);
        for (const auto& [index, count] : pin.visits) {
            expected[index] = count;
        }
        require(pool.visits(id) == expected, "visits with transpositions off must match the pinned ones");
        require(pool.stats(id).nodes == pin.nodes, "the tree size must match the pinned one");
        require(pool.stats(id).transpositions == 0, "nothing is linked with the option off");
    }
}

/**
 * Re-rooting keeps a subtree whose nodes may have several parents: the
 * retained graph is re-parented and its table rebuilt, so later
 * simulations (which rebuild released games along parents) run through
 * it, and they do so the same on any thread count.
 */
void test_advance_on_a_dag() {
    const Game game = hand_position({4, 4, 0, 0, 0});
    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    for (const bool release : {true, false}) {
        std::vector<std::vector<std::uint32_t>> results;
        for (const int threads : {1, 4}) {
            SearchConfig config;
            config.simulations = 256;
            config.max_players = 4;
            config.transpositions = true;
            config.release_games = release;
            config.threads = threads;
            SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
            const auto id = pool.add(game, 1);
            Game position = game;
            for (int step = 0; step < 2; ++step) {
                run(pool, hashed_evaluator(4));
                const auto before = pool.stats(id);
                require(before.transpositions > 0, "the subtree holds linked nodes");
                // The most visited deterministic root action.
                const auto visits = pool.visits(id);
                std::size_t chosen = codec->size();
                for (std::size_t i = 0; i < visits.size(); ++i) {
                    if (visits[i] == 0 || (chosen < codec->size() && visits[i] <= visits[chosen])) {
                        continue;
                    }
                    const ActionType type = codec->decode(position, i).type;
                    if (type != ActionType::roll && type != ActionType::buy_development_card &&
                        type != ActionType::move_robber) {
                        chosen = i;
                    }
                }
                require(chosen < codec->size(), "a deterministic root action was visited");
                require(pool.advance(id, chosen), "re-rooting at a visited deterministic child");
                position.execute(codec->decode(position, chosen));
                const auto after = pool.stats(id);
                require(after.nodes < before.nodes && after.transpositions == before.transpositions,
                        "the subtree is kept with its counters");
                require(!pool.done(id), "a fresh budget");
            }
            run(pool, hashed_evaluator(4));
            const auto stats = pool.stats(id);
            const auto visits = pool.visits(id);
            require(pool.done(id) && stats.simulations == 256 &&
                        std::accumulate(visits.begin(), visits.end(), 0U) >= 256,
                    "the re-rooted search completes its budget on top of the retained visits");
            require(release == (stats.rebuilt_games > 0), "released games are rebuilt through re-parented nodes");
            results.push_back(visits);
        }
        require(results[0] == results[1], "threads must not change a re-rooted DAG search");
    }
}

/** FlatActionCodec with no legal action at one prompt, to fence a subtree. */
class ClosedPromptCodec final : public ActionCodec {
public:
    ClosedPromptCodec(int players, ActionPrompt closed)
        : inner_(players, MapType::base), closed_(closed) {}
    [[nodiscard]] std::size_t size() const noexcept override { return inner_.size(); }
    [[nodiscard]] std::size_t index(const Game& game, const Action& action) const override {
        return inner_.index(game, action);
    }
    [[nodiscard]] Action decode(const Game& game, std::size_t index) const override {
        return inner_.decode(game, index);
    }
    void mask(const Game& game, std::span<std::uint8_t> out) const override {
        if (game.current_prompt() == closed_) {
            std::fill(out.begin(), out.end(), 0);
            return;
        }
        inner_.mask(game, out);
    }

private:
    FlatActionCodec inner_;
    ActionPrompt closed_;
};

/**
 * Red discards four of two wood, two brick, two sheep and two wheat. Every
 * order of one multiset of kinds reaches the same hand, so with
 * transpositions the discard subtree is the lattice of partial multisets
 * (1 + 4 + 10 + 16 + 19 nodes) however long the search runs; without it,
 * every order has its own path. The robber prompt after the discards is
 * closed, so nothing beyond the chain is searched.
 */
void test_discard_chain_is_a_lattice() {
    auto players = random_placements(4, 12);
    players[0].resources = {2, 2, 2, 2, 0};
    players[0].discards = 4;
    Game game(seats(4), MapType::base, 12);
    GameSetup setup;
    setup.players = players;
    setup.has_rolled = true;
    setup.prompt = ActionPrompt::discard;
    game.apply_setup(setup);
    require(game.current_prompt() == ActionPrompt::discard && game.playable_actions().size() == 4,
            "Red chooses among four kinds");
    const auto codec = std::make_shared<ClosedPromptCodec>(4, ActionPrompt::move_robber);
    std::uint64_t with = 0;
    std::uint64_t without = 0;
    for (const bool transpositions : {false, true}) {
        SearchConfig config;
        config.simulations = 400;
        config.max_players = 4;
        config.transpositions = transpositions;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        const auto id = pool.add(game, 3);
        run(pool, hashed_evaluator(4));
        const auto stats = pool.stats(id);
        require(stats.max_depth <= 4 && pool.done(id), "the search stays within the discard chain");
        (transpositions ? with : without) = stats.nodes;
        require((stats.transpositions > 0) == transpositions, "orders of one multiset link");
    }
    require(with <= 50, "the discard subtree is at most the multiset lattice: " + std::to_string(with));
    require(without > 50,
            "without the table the chain outgrows the lattice: " + std::to_string(without));
}

/**
 * The most visited root action that keeps the turn and leads to no chance
 * node (a re-rootable one whose child is a decision of several options);
 * codec.size() when there is none.
 */
std::size_t most_visited_build(const SearchPool& pool, std::int64_t id, const ActionCodec& codec, const Game& position) {
    const auto visits = pool.visits(id);
    std::size_t chosen = codec.size();
    for (std::size_t i = 0; i < visits.size(); ++i) {
        if (visits[i] == 0 || (chosen < codec.size() && visits[i] <= visits[chosen])) {
            continue;
        }
        const ActionType type = codec.decode(position, i).type;
        if (type != ActionType::roll && type != ActionType::buy_development_card &&
            type != ActionType::move_robber && type != ActionType::end_turn) {
            chosen = i;
        }
    }
    return chosen;
}

std::uint32_t total(const std::vector<std::uint32_t>& visits) {
    return std::accumulate(visits.begin(), visits.end(), 0U);
}

/**
 * add's options are the search's own settings: a search given a budget,
 * leaves per round and noise runs exactly as in a pool configured with
 * them, beside a search that keeps the pool's.
 */
void test_search_options_are_per_search() {
    const Game game = hand_position({4, 4, 0, 0, 0});
    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    const auto alone = [&](int simulations, int pending, double alpha) {
        SearchConfig config;
        config.simulations = simulations;
        config.max_pending = pending;
        config.dirichlet_alpha = alpha;
        config.max_players = 4;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        const auto id = pool.add(game, 7);
        run(pool, hashed_evaluator(4));
        return std::pair{pool.visits(id), pool.q(id)};
    };
    const auto same = [](const std::pair<std::vector<std::uint32_t>, std::vector<double>>& a,
                         const std::pair<std::vector<std::uint32_t>, std::vector<double>>& b) {
        bool equal = a.first == b.first;
        for (std::size_t i = 0; equal && i < a.second.size(); ++i) {
            equal = (std::isnan(a.second[i]) && std::isnan(b.second[i])) || a.second[i] == b.second[i];
        }
        return equal;
    };
    const auto full = alone(256, 8, 0.3);
    const auto fast = alone(48, 2, 0.0);
    require(!same(alone(256, 8, 0.0), full), "the noise must matter for the test to say anything");
    for (const bool noisy_pool : {true, false}) {
        SearchConfig config;
        config.simulations = noisy_pool ? 256 : 48;
        config.max_pending = noisy_pool ? 8 : 2;
        config.dirichlet_alpha = noisy_pool ? 0.3 : 0.0;
        config.max_players = 4;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        // One search with the pool's settings, one with the other pool's.
        const auto a = pool.add(game, 7, noisy_pool ? SearchOptions{} : SearchOptions{256, 8, 0.3});
        const auto b = pool.add(game, 7, noisy_pool ? SearchOptions{48, 2, 0.0} : SearchOptions{});
        // Leaves per round are each search's own.
        std::vector<float> observations(64 * pool.observation_size());
        std::vector<std::int64_t> ids(64);
        const std::size_t width = pool.action_size();
        const Evaluator evaluate = hashed_evaluator(4);
        bool reached = false;
        while (true) {
            const std::size_t rows = pool.collect(observations, ids);
            if (rows == 0) {
                break;
            }
            const auto of_a = std::count(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(rows), a);
            const auto of_b = std::count(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(rows), b);
            require(of_a <= 8 && of_b <= 2, "a search issues at most its own max_pending leaves a round");
            reached = reached || (of_a == 8 && of_b == 2);
            std::vector<float> logits(rows * width);
            std::vector<float> values(rows * 4);
            for (std::size_t row = 0; row < rows; ++row) {
                evaluate(
                    pool.pending_game(row), std::span<float>(logits).subspan(row * width, width),
                    std::span<float>(values).subspan(row * 4, 4));
            }
            pool.apply(logits, values);
        }
        require(reached, "both searches fill their own rounds");
        require(same({pool.visits(a), pool.q(a)}, full), "options must equal a pool configured with them (full)");
        require(same({pool.visits(b), pool.q(b)}, fast), "options must equal a pool configured with them (fast)");
        require(pool.stats(a).simulations == 256 && pool.stats(b).simulations == 48, "each search runs its own budget");
    }
    SearchConfig config;
    config.max_players = 4;
    SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
    for (const SearchOptions& bad : {SearchOptions{0, {}, {}}, SearchOptions{{}, 0, {}},
                                     SearchOptions{{}, {}, std::numeric_limits<double>::quiet_NaN()}}) {
        bool threw = false;
        try {
            pool.add(game, 1, bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        require(threw && pool.size() == 0, "invalid options are refused before a search exists");
    }
}

/**
 * reroot is advance under a visit cap: the new root's visits count towards
 * the budget, the search tops the root up to exactly the cap, and a root
 * that already has the cap runs no simulation at all.
 */
void test_reroot_counts_the_carried_visits() {
    const Game game = hand_position({4, 4, 0, 0, 0});
    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    for (const bool transpositions : {false, true}) {
        SearchConfig config;
        config.simulations = 256;
        config.max_pending = 8;
        config.max_players = 4;
        config.transpositions = transpositions;
        const auto searched = [&](SearchPool& pool) {
            const auto id = pool.add(game, 3);
            run(pool, hashed_evaluator(4));
            require(total(pool.visits(id)) == 256, "a fresh search runs its simulations");
            return id;
        };
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        const auto id = searched(pool);
        const std::size_t chosen = most_visited_build(pool, id, *codec, game);
        require(chosen < codec->size(), "a building action was visited");
        const std::uint32_t child = pool.visits(id)[chosen];
        const int cap = static_cast<int>(child) + 40;
        Game next = game;
        next.execute(codec->decode(game, chosen));

        // A cap above what the root carries: topped up to exactly the cap,
        // with a per-search max_pending.
        require(pool.reroot(id, chosen, SearchOptions{cap, 4, {}}), "re-rooting at a visited deterministic child");
        const std::uint32_t carried = total(pool.visits(id));
        require(carried > 0 && carried < child,
                "the new root carries the child's visits (less its own evaluation, and with "
                "transpositions those that came through other parents)");
        require(!pool.done(id) && pool.stats(id).simulations == 0, "the top-up is still to run");
        std::vector<float> observations(64 * pool.observation_size());
        std::vector<std::int64_t> ids(64);
        const Evaluator evaluate = hashed_evaluator(4);
        while (const std::size_t rows = pool.collect(observations, ids)) {
            require(rows <= 4, "the re-rooted search issues its own max_pending");
            std::vector<float> logits(rows * codec->size());
            std::vector<float> values(rows * 4);
            for (std::size_t row = 0; row < rows; ++row) {
                evaluate(
                    pool.pending_game(row), std::span<float>(logits).subspan(row * codec->size(), codec->size()),
                    std::span<float>(values).subspan(row * 4, 4));
            }
            pool.apply(logits, values);
        }
        require(pool.done(id) && total(pool.visits(id)) == static_cast<std::uint32_t>(cap), "the root is topped up to the cap");
        require(pool.stats(id).simulations == static_cast<std::uint32_t>(cap) - carried, "only the missing visits are simulated");

        // advance is reroot with a cap of the carried visits plus the budget.
        SearchPool twin(codec, std::make_shared<ToyEncoder>(), config);
        const auto other = searched(twin);
        require(twin.advance(other, chosen), "advance re-roots where reroot does");
        require(total(twin.visits(other)) == carried, "both keep the same subtree");
        run(twin, hashed_evaluator(4));
        SearchPool same(codec, std::make_shared<ToyEncoder>(), config);
        const auto second = searched(same);
        require(same.reroot(second, chosen, SearchOptions{static_cast<int>(carried) + 256, {}, {}}),
                "re-rooting with the budget advance gives");
        run(same, hashed_evaluator(4));
        require(same.visits(second) == twin.visits(other) && same.stats(second).nodes == twin.stats(other).nodes,
                "reroot with advance's budget must search as advance does");

        // The settings are the search's: the pool's own do not matter.
        SearchConfig rest_config = config;
        rest_config.simulations = 7;
        rest_config.max_pending = 3;
        SearchPool rest(codec, std::make_shared<ToyEncoder>(), rest_config);
        const auto third = rest.add(game, 3, SearchOptions{256, 8, {}});
        run(rest, hashed_evaluator(4));
        require(rest.reroot(third, chosen, SearchOptions{cap, 4, {}}), "re-rooting in a pool of other settings");
        run(rest, hashed_evaluator(4));
        require(rest.visits(third) == pool.visits(id), "reroot must not depend on the pool's config");

        // A cap the root already has: no simulation, results from the tree.
        SearchPool capped(codec, std::make_shared<ToyEncoder>(), config);
        const auto fourth = searched(capped);
        require(capped.reroot(fourth, chosen, SearchOptions{static_cast<int>(carried), {}, {}}),
                "re-rooting at the cap");
        require(capped.done(fourth), "a root with its visits is done at once");
        require(capped.collect(observations, ids) == 0, "no leaf is issued");
        require(total(capped.visits(fourth)) == carried && capped.stats(fourth).simulations == 0,
                "no simulation runs");
        const auto target = capped.policy_target(fourth);
        require(std::abs(std::accumulate(target.begin(), target.end(), 0.0) - 1.0) < 1e-5,
                "the carried visits are the policy target");
        std::vector<std::uint8_t> legal(codec->size());
        codec->mask(next, legal);
        require(legal[capped.select(fourth, 0.0)] != 0, "the move is chosen from the carried visits");
        SearchPool below(codec, std::make_shared<ToyEncoder>(), config);
        const auto fifth = searched(below);
        require(below.reroot(fifth, chosen, SearchOptions{1, {}, {}}) && below.done(fifth) &&
                    total(below.visits(fifth)) == carried,
                "a cap below the carried visits keeps them all");
        // The next reroot gives the search a budget again.
        const std::size_t again = most_visited_build(below, fifth, *codec, next);
        if (again < codec->size() && below.reroot(fifth, again, SearchOptions{64, {}, {}})) {
            const std::uint32_t kept = total(below.visits(fifth));
            run(below, hashed_evaluator(4));
            require(below.done(fifth) && total(below.visits(fifth)) == std::max<std::uint32_t>(kept, 64),
                    "a later search of the tree has its own cap");
        }
    }
}

/** The hashed network's priors at a position: softmax of its logits over the legal actions. */
std::vector<double> network_priors(const ActionCodec& codec, const Game& position) {
    std::vector<float> logits(codec.size());
    hashed_logits(position, logits);
    std::vector<std::uint8_t> legal(codec.size());
    codec.mask(position, legal);
    double sum = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        sum += legal[i] != 0 ? std::exp(static_cast<double>(logits[i])) : 0.0;
    }
    std::vector<double> priors(codec.size(), 0.0);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        priors[i] = legal[i] != 0 ? std::exp(static_cast<double>(logits[i])) / sum : 0.0;
    }
    return priors;
}

bool close(const std::vector<float>& actual, const std::vector<double>& expected) {
    bool equal = actual.size() == expected.size();
    for (std::size_t i = 0; equal && i < actual.size(); ++i) {
        equal = std::abs(actual[i] - expected[i]) < 1e-6;
    }
    return equal;
}

/**
 * Every search with noise draws it anew at its own root, also a re-rooted
 * one; the tree keeps the network's priors (root_priors), which the next
 * root starts from, so noise reaches a later search only through visits.
 */
void test_reroot_noise_is_per_search() {
    const Game game = hand_position({4, 4, 0, 0, 0});
    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    const auto clean = [&](const Game& position) { return network_priors(*codec, position); };
    SearchConfig config;
    config.simulations = 256;
    config.max_players = 4;
    config.dirichlet_alpha = 0.3;
    config.dirichlet_fraction = 0.25;
    SearchConfig quiet_config = config;
    quiet_config.dirichlet_alpha = 0.0;
    SearchPool quiet(codec, std::make_shared<ToyEncoder>(), quiet_config);
    const auto reference = quiet.add(game, 5);
    require(close(quiet.root_priors(reference), std::vector<double>(codec->size(), 0.0)),
            "no priors before the root is evaluated");
    run(quiet, hashed_evaluator(4));
    require(close(quiet.root_priors(reference), clean(game)), "an un-noised root reports its priors");

    // Three trees searched alike with noise, then re-rooted with new noise,
    // with other noise (another seed would be another tree: the alpha), and
    // with none.
    std::vector<std::vector<std::uint32_t>> results;
    std::size_t chosen = 0;
    Game next = game;
    for (const double alpha : {0.3, 1.0, 0.0}) {
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        const auto id = pool.add(game, 5);
        run(pool, hashed_evaluator(4));
        require(pool.visits(id) != quiet.visits(reference), "the fresh search ran with noise");
        require(close(pool.root_priors(id), clean(game)), "root_priors are the priors before the noise");
        if (results.empty()) {
            chosen = most_visited_build(pool, id, *codec, game);
            require(chosen < codec->size(), "a building action was visited");
            next.execute(codec->decode(game, chosen));
        }
        require(pool.reroot(id, chosen, SearchOptions{256, {}, alpha}), "re-rooting a noised search");
        require(close(pool.root_priors(id), clean(next)),
                "the new root starts from the network's priors, whatever noise the old root had");
        run(pool, hashed_evaluator(4));
        require(total(pool.visits(id)) == 256, "the re-rooted search fills its cap");
        require(close(pool.root_priors(id), clean(next)), "the noise never reaches root_priors");
        results.push_back(pool.visits(id));
    }
    require(results[0] != results[2] && results[1] != results[2] && results[0] != results[1],
            "a re-rooted root is searched under the noise it is given, and under none without");
}

/** reroot refuses what advance refuses, and what a visit cap cannot mean. */
void test_reroot_refusals() {
    const Game game = hand_position({1, 1, 1, 1, 1});
    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    SearchConfig config;
    config.simulations = 128;
    config.max_players = 4;
    SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
    const auto id = pool.add(game, 2);
    require(!pool.reroot(id, 0), "no re-rooting before the root is evaluated");
    run(pool, hashed_evaluator(4));
    const auto visits = pool.visits(id);
    std::vector<std::uint8_t> legal(codec->size());
    codec->mask(game, legal);
    bool chance = false;
    for (std::size_t i = 0; i < codec->size(); ++i) {
        if (legal[i] == 0) {
            continue;
        }
        if (codec->decode(game, i).type == ActionType::buy_development_card) {
            chance = true;
            require(!pool.reroot(id, i, SearchOptions{16, {}, {}}), "a chance action is never re-rooted through");
        } else if (visits[i] == 0) {
            require(!pool.reroot(id, i, SearchOptions{16, {}, {}}), "an unvisited child is refused");
        }
    }
    require(chance, "the position must offer a chance action");
    for (std::size_t i = 0; i < codec->size(); ++i) {
        if (legal[i] == 0) {
            require(!pool.reroot(id, i), "an action the root does not have is refused");
            break;
        }
    }
    require(pool.done(id) && pool.visits(id) == visits && pool.stats(id).simulations == 128,
            "a refused reroot leaves the search as it was");
    const std::size_t chosen = most_visited_build(pool, id, *codec, game);
    require(chosen < codec->size(), "a building action was visited");
    bool threw = false;
    try {
        pool.reroot(id, chosen, SearchOptions{0, {}, {}});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw && pool.visits(id) == visits, "invalid options are refused before anything changes");
    // The refused settings were not taken: advance still starts the pool's budget.
    require(pool.advance(id, chosen), "the search is still re-rootable");
    std::vector<float> observations(64 * pool.observation_size());
    std::vector<std::int64_t> ids(64);
    require(pool.collect(observations, ids) > 0, "leaves are issued");
    threw = false;
    try {
        pool.reroot(id, 0);
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw, "no re-rooting while leaves await apply");
    pool.cancel();
    run(pool, hashed_evaluator(4));
    require(pool.stats(id).simulations == 128, "the search still runs the pool's budget");

    SearchConfig gumbel_config = config;
    gumbel_config.root = RootSelection::gumbel;
    SearchPool gumbel(codec, std::make_shared<ToyEncoder>(), gumbel_config);
    const auto other = gumbel.add(game, 2);
    run(gumbel, hashed_evaluator(4));
    const std::size_t played = most_visited_build(gumbel, other, *codec, game);
    require(played < codec->size(), "a building action was visited (Gumbel)");
    threw = false;
    try {
        gumbel.reroot(other, played);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "a Gumbel root has no visit cap");
    require(gumbel.advance(other, played), "advance still re-roots a Gumbel search");
}

/** With determinizations the cap counts the root's visits over all worlds. */
void test_reroot_determinized() {
    GameRules rules;
    rules.friendly_robber = true;
    auto players = random_placements(4, 12);
    players[0].development_cards = {1, 0, 0, 0, 0};
    players[1].resources = {0, 0, 0, 1, 0};
    Game game(seats(4), MapType::base, 12, rules);
    GameSetup setup;
    setup.players = players;
    game.apply_setup(setup);
    const HandTracker tracker(game, 0);
    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    const std::size_t index = codec->index(game, {Color::red, ActionType::play_knight_card, std::monostate{}});
    SearchConfig config;
    config.simulations = 240;
    config.max_players = 4;
    config.determinizations = 3;
    SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
    const auto id = search_favouring(pool, game, index, &tracker);
    require(pool.reroot(id, index, SearchOptions{300, {}, {}}), "an own decision re-roots under determinization");
    const std::uint32_t carried = total(pool.visits(id));
    require(carried > 0 && carried < 300, "the worlds' visits are carried");
    run(pool, [](const Game& leaf, std::span<float> logits, std::span<float> values) {
        hashed_logits(leaf, logits);
        std::fill(values.begin(), values.end(), 0.25F);
    });
    require(pool.done(id) && total(pool.visits(id)) == 300 && pool.stats(id).simulations == 300 - carried,
            "the worlds together are topped up to the cap");
    require(pool.root_priors(id).size() == pool.stats(id).worlds * codec->size(), "priors per world");
}

/**
 * With the root temperature at 1 and no forced playouts (the defaults) a
 * search is the one it was before either existed: the visits, the tree
 * size, the action chosen and its value match the ones pinned then, for a
 * noised search with leaves in flight, fresh and re-rooted under a cap.
 */
void test_root_options_off_are_bitwise_today() {
    struct Root {
        std::uint64_t nodes;
        std::size_t selected;
        double q;
        std::vector<std::pair<std::size_t, std::uint32_t>> visits;
    };
    struct Pin {
        Game game;
        int players;
        Root fresh;
        std::size_t moved;
        Root rerooted;
    };
    const std::vector<Pin> pins{
        {hand_position({4, 4, 0, 0, 0}), 4,
         {480, 87, 0x1.f36308c2aaaabp-2,
          {{64, 1}, {67, 1}, {68, 1}, {87, 24}, {88, 12}, {103, 1}, {122, 1}, {124, 9}, {186, 10},
           {187, 2}, {188, 1}, {189, 3}, {190, 5}, {235, 16}, {236, 1}, {237, 4}, {238, 4}}},
         87,
         {1085, 103, 0x1.bdd2f984bda13p-2,
          {{64, 20}, {67, 21}, {68, 19}, {86, 17}, {88, 18}, {103, 27}, {122, 4}, {124, 16}, {186, 18}}}},
        {play_random(2, 5, 200).game, 2,
         {284, 195, 0x1.1795d1c1642c8p-1, {{186, 20}, {195, 46}, {196, 18}, {197, 6}, {198, 6}}},
         195,
         {443, 180, 0x1.eb252a1109209p-2, {{180, 81}, {186, 79}}}},
    };
    for (const Pin& pin : pins) {
        const auto codec = std::make_shared<FlatActionCodec>(pin.players, MapType::base);
        SearchConfig config;
        config.simulations = 96;
        config.max_pending = 4;
        config.dirichlet_alpha = 0.3;
        config.max_players = pin.players;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        const auto id = pool.add(pin.game, 3);
        const auto matches = [&](const Root& root) {
            std::vector<std::uint32_t> expected(codec->size(), 0);
            for (const auto& [index, count] : root.visits) {
                expected[index] = count;
            }
            const std::size_t selected = pool.select(id);
            return pool.visits(id) == expected && pool.stats(id).nodes == root.nodes &&
                selected == root.selected && pool.q(id)[selected] == root.q;
        };
        run(pool, hashed_evaluator(pin.players));
        require(matches(pin.fresh), "a search with the root options off must match the pinned one");
        require(most_visited_build(pool, id, *codec, pin.game) == pin.moved &&
                    pool.reroot(id, pin.moved, SearchOptions{160, 3, 0.3}),
                "the pinned action re-roots");
        run(pool, hashed_evaluator(pin.players));
        require(matches(pin.rerooted), "a re-rooted search with the root options off must match the pinned one");
    }
}

/** Priors at a root temperature: each to the power 1 / temperature, renormalized. */
std::vector<double> tempered(const std::vector<float>& priors, double temperature) {
    std::vector<double> result(priors.size(), 0.0);
    double sum = 0.0;
    for (std::size_t i = 0; i < priors.size(); ++i) {
        result[i] = priors[i] > 0.0F ? std::pow(static_cast<double>(priors[i]), 1.0 / temperature) : 0.0;
        sum += result[i];
    }
    for (double& value : result) {
        value /= sum;
    }
    return result;
}

/**
 * The root temperature changes the priors the root selects by, not the
 * tree's: root_search_priors are root_priors to the power 1 / temperature,
 * renormalized, with the noise mixed in after, and every root starts from
 * the network's priors, so the temperature applies once however often a
 * search is re-rooted.
 */
void test_root_temperature_reshapes_the_priors_in_use() {
    const Game game = hand_position({4, 4, 0, 0, 0});
    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    SearchConfig config;
    config.simulations = 256;
    config.max_players = 4;
    SearchPool plain(codec, std::make_shared<ToyEncoder>(), config);
    const auto reference = plain.add(game, 5);
    run(plain, hashed_evaluator(4));
    require(plain.root_search_priors(reference) == plain.root_priors(reference),
            "without temperature and noise the root selects by the network's priors");

    SearchConfig warm_config = config;
    warm_config.root_policy_temperature = 1.5;
    SearchPool warm(codec, std::make_shared<ToyEncoder>(), warm_config);
    const auto id = warm.add(game, 5);
    require(close(warm.root_search_priors(id), std::vector<double>(codec->size(), 0.0)),
            "no priors before the root is evaluated");
    run(warm, hashed_evaluator(4));
    require(warm.root_priors(id) == plain.root_priors(reference), "root_priors are the priors before the temperature");
    require(close(warm.root_search_priors(id), tempered(warm.root_priors(id), 1.5)),
            "the root selects by its priors at the temperature");
    const auto flat = warm.root_search_priors(id);
    const auto sharp = warm.root_priors(id);
    require(*std::max_element(flat.begin(), flat.end()) < *std::max_element(sharp.begin(), sharp.end()) - 1e-3F &&
                warm.visits(id) != plain.visits(reference),
            "a temperature above one flattens the priors, and the search follows them");

    // The noise is mixed into the priors at the temperature: one seed draws
    // one noise, which is what the edges hold beyond their share of priors.
    SearchConfig noisy_config = config;
    noisy_config.dirichlet_alpha = 0.3;
    SearchPool noisy(codec, std::make_shared<ToyEncoder>(), noisy_config);
    noisy_config.root_policy_temperature = 1.5;
    SearchPool both(codec, std::make_shared<ToyEncoder>(), noisy_config);
    const auto noisy_id = noisy.add(game, 5);
    const auto both_id = both.add(game, 5);
    run(noisy, hashed_evaluator(4));
    run(both, hashed_evaluator(4));
    require(both.root_priors(both_id) == plain.root_priors(reference), "root_priors have neither temperature nor noise");
    const auto network = noisy.root_priors(noisy_id);
    const auto warmed = tempered(network, 1.5);
    const auto noisy_priors = noisy.root_search_priors(noisy_id);
    const auto both_priors = both.root_search_priors(both_id);
    double most_noise = 0.0;
    for (std::size_t i = 0; i < network.size(); ++i) {
        const double noise = noisy_priors[i] - 0.75 * network[i];
        most_noise = std::max(most_noise, noise);
        require(std::abs(both_priors[i] - 0.75 * warmed[i] - noise) < 1e-6,
                "the noise is mixed into the priors at the temperature");
    }
    require(most_noise > 1e-2, "the noise must matter for the test to say anything");

    // Re-rooted, the temperature meets the network's priors again.
    Game position = game;
    for (int move = 0; move < 2; ++move) {
        const std::size_t chosen = most_visited_build(warm, id, *codec, position);
        require(chosen < codec->size(), "a building action was visited");
        position.execute(codec->decode(position, chosen));
        require(warm.reroot(id, chosen, SearchOptions{256, {}, {}, {}, {}}), "re-rooting a search with a temperature");
        for (const bool searched : {false, true}) {
            if (searched) {
                run(warm, hashed_evaluator(4));
            }
            require(close(warm.root_priors(id), network_priors(*codec, position)),
                    "the new root starts from the network's priors, whatever the old root selected by");
            require(close(warm.root_search_priors(id), tempered(warm.root_priors(id), 1.5)),
                    "the temperature applies once to every root");
        }
    }
    const std::size_t chosen = most_visited_build(warm, id, *codec, position);
    require(chosen < codec->size(), "a building action was visited");
    require(warm.reroot(id, chosen, SearchOptions{256, {}, {}, 1.0, {}}), "re-rooting to a search without a temperature");
    require(warm.root_search_priors(id) == warm.root_priors(id), "a root at temperature one selects by the network's priors");
}

/**
 * Forced playouts select, whenever a visited root action is below f(t) =
 * sqrt(k * prior * t) visits (t the root's visits so far, both with the
 * leaves in flight), the visited action furthest below; an action without
 * a visit is never forced (KataGo), so PUCT opens the actions it would
 * open anyway. The test replays the root's selections from its visits:
 * they change only in collect and count a leaf from the moment it is
 * issued, so the visits before a round are the state its first selection
 * saw, and each selection the rule decides gives the state the next one
 * saw. A round is replayed until its first selection the rule leaves to
 * PUCT (the visits do not tell what PUCT chose); with one leaf per round
 * that is every selection.
 */
void test_forced_playouts_select_the_visited_action_furthest_below() {
    const Game game = hand_position({4, 4, 0, 0, 0});
    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    const int simulations = 300;
    const double k = 2.0;
    // Priors under which PUCT, with an unvisited action's value 0, leaves root actions unvisited.
    const Evaluator peaked = [](const Game& leaf, std::span<float> logits, std::span<float> values) {
        hashed_evaluator(4)(leaf, logits, values);
        for (float& logit : logits) {
            logit *= 4.0F;
        }
    };
    // What a visited action lacks of its count (none: below zero).
    const auto missing = [&](const std::vector<float>& priors, const std::vector<std::uint32_t>& visits,
                             std::size_t i) {
        return visits[i] == 0 ? -1.0
                              : std::sqrt(k * priors[i] * static_cast<double>(total(visits))) -
                                    static_cast<double>(visits[i]);
    };
    struct Replay {
        int decided{}; // selections the rule decided, which the search made
        int missed{};  // rounds in which the rule decided one the search did not make
    };
    const auto replay = [&](SearchPool& pool, std::int64_t id) {
        Replay result;
        std::vector<float> observations(64 * pool.observation_size());
        std::vector<std::int64_t> ids(64);
        std::vector<std::uint32_t> state = pool.visits(id);
        while (true) {
            const std::size_t rows = pool.collect(observations, ids);
            if (rows == 0) {
                return result;
            }
            const auto after = pool.visits(id);
            const auto priors = pool.root_search_priors(id);
            while (state != after) {
                std::size_t pick = state.size();
                double shortfall = 0.0;
                for (std::size_t i = 0; i < state.size(); ++i) {
                    if (const double lacks = missing(priors, state, i); lacks > shortfall) {
                        shortfall = lacks;
                        pick = i;
                    }
                }
                if (pick == state.size()) {
                    break; // PUCT's selection
                }
                if (state[pick] == after[pick]) {
                    ++result.missed;
                    break;
                }
                ++state[pick];
                ++result.decided;
            }
            state = after;
            const std::size_t width = pool.action_size();
            std::vector<float> logits(rows * width);
            std::vector<float> values(rows * 4);
            for (std::size_t row = 0; row < rows; ++row) {
                peaked(
                    pool.pending_game(row), std::span<float>(logits).subspan(row * width, width),
                    std::span<float>(values).subspan(row * 4, 4));
            }
            pool.apply(logits, values);
        }
    };
    // The visits the visited actions lack of ceil(f(N)) in all.
    const auto lacking = [&](const SearchPool& pool, std::int64_t id) {
        const auto priors = pool.root_search_priors(id);
        const auto visits = pool.visits(id);
        double sum = 0.0;
        for (std::size_t i = 0; i < visits.size(); ++i) {
            sum += std::max(0.0, std::ceil(missing(priors, visits, i)));
        }
        return sum;
    };
    for (const bool shaped : {false, true}) {
        SearchConfig config;
        config.simulations = simulations;
        config.max_players = 4;
        config.fpu_value = 0.0; // an unvisited action counts as lost
        config.dirichlet_alpha = shaped ? 0.3 : 0.0;
        config.root_policy_temperature = shaped ? 1.25 : 1.0;
        for (const int pending : {1, 4, 16}) {
            const std::string at = " (max_pending " + std::to_string(pending) + ")";
            config.max_pending = pending;
            config.forced_playouts = 0.0;
            SearchPool plain(codec, std::make_shared<ToyEncoder>(), config);
            const auto reference = plain.add(game, 5);
            require(replay(plain, reference).missed > 0,
                    "PUCT alone must select otherwise for the test to say anything" + at);

            config.forced_playouts = k;
            SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
            const auto id = pool.add(game, 5);
            const Replay seen = replay(pool, id);
            require(pool.done(id) && total(pool.visits(id)) == static_cast<std::uint32_t>(simulations),
                    "forced playouts are simulations of the budget");
            require(pool.root_search_priors(id) == plain.root_search_priors(reference),
                    "forced playouts leave the priors alone");
            require(seen.decided > 0 && seen.missed == 0,
                    "a visited action below its count is selected, the furthest below first" + at + ": " +
                        std::to_string(seen.decided) + " made, " + std::to_string(seen.missed) + " rounds without");
            const auto priors = pool.root_search_priors(id);
            const auto visits = pool.visits(id);
            bool unopened = false;
            for (std::size_t i = 0; i < visits.size(); ++i) {
                unopened = unopened || (priors[i] > 0.0F && visits[i] == 0);
            }
            require(unopened, "an action without a visit is not forced: PUCT leaves some unvisited" + at);
            require(lacking(pool, id) < lacking(plain, reference),
                    "here the visited actions end nearer their counts than under PUCT alone" + at);
        }
    }
}

/**
 * The root temperature and the forced playouts are a search's own settings
 * like the others: given to add they run as in a pool configured with
 * them, and they are refused where no root, or no Gumbel root, can run
 * with them.
 */
void test_root_options_are_per_search() {
    const Game game = hand_position({4, 4, 0, 0, 0});
    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    SearchConfig base;
    base.simulations = 200;
    base.max_pending = 4;
    base.dirichlet_alpha = 0.3;
    base.max_players = 4;
    const auto result = [](SearchPool& pool, std::int64_t id) {
        return std::tuple{pool.visits(id), pool.root_search_priors(id), pool.q(id)[pool.select(id)]};
    };
    const auto alone = [&](double temperature, double forced) {
        SearchConfig config = base;
        config.root_policy_temperature = temperature;
        config.forced_playouts = forced;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        const auto id = pool.add(game, 7);
        run(pool, hashed_evaluator(4));
        return result(pool, id);
    };
    const auto plain = alone(1.0, 0.0);
    const auto shaped = alone(1.5, 2.0);
    require(alone(1.5, 0.0) != plain && alone(1.0, 2.0) != plain && alone(1.5, 0.0) != shaped &&
                alone(1.0, 2.0) != shaped,
            "each option must matter for the test to say anything");
    for (const bool shaped_pool : {true, false}) {
        SearchConfig config = base;
        config.root_policy_temperature = shaped_pool ? 1.5 : 1.0;
        config.forced_playouts = shaped_pool ? 2.0 : 0.0;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        // One search with the pool's settings, one with the other pool's.
        const auto a = pool.add(game, 7, shaped_pool ? SearchOptions{} : SearchOptions{{}, {}, {}, 1.5, 2.0});
        const auto b = pool.add(game, 7, shaped_pool ? SearchOptions{{}, {}, {}, 1.0, 0.0} : SearchOptions{});
        run(pool, hashed_evaluator(4));
        require(result(pool, a) == shaped, "options must equal a pool configured with them (on)");
        require(result(pool, b) == plain, "options must equal a pool configured with them (off)");
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    const auto refused = [](const auto& attempt) {
        try {
            attempt();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    SearchPool pool(codec, std::make_shared<ToyEncoder>(), base);
    const auto id = pool.add(game, 7);
    run(pool, hashed_evaluator(4));
    const std::size_t chosen = most_visited_build(pool, id, *codec, game);
    require(chosen < codec->size(), "a building action was visited");
    const auto before = result(pool, id);
    for (const auto& [temperature, forced] : std::vector<std::pair<double, double>>{
             {0.0, 0.0}, {-1.0, 0.0}, {nan, 0.0}, {infinity, 0.0}, {1.0, -0.5}, {1.0, nan}, {1.0, infinity}}) {
        SearchConfig config = base;
        config.root_policy_temperature = temperature;
        config.forced_playouts = forced;
        require(refused([&] { SearchPool(codec, std::make_shared<ToyEncoder>(), config); }),
                "a pool refuses a root temperature that is not positive and forced playouts that are negative");
        const SearchOptions bad{{}, {}, {}, temperature, forced};
        require(refused([&] { pool.add(game, 1, bad); }) && pool.size() == 1,
                "invalid options are refused before a search exists");
        require(refused([&] { pool.reroot(id, chosen, bad); }) && result(pool, id) == before && pool.done(id),
                "invalid options leave the search as it was");
    }
    // Sequential Halving plans a Gumbel root's visits: neither option applies.
    SearchConfig gumbel = base;
    gumbel.root = RootSelection::gumbel;
    for (const auto& [temperature, forced] : std::vector<std::pair<double, double>>{{1.25, 0.0}, {1.0, 2.0}}) {
        SearchConfig config = gumbel;
        config.root_policy_temperature = temperature;
        config.forced_playouts = forced;
        require(refused([&] { SearchPool(codec, std::make_shared<ToyEncoder>(), config); }),
                "a Gumbel pool refuses a root temperature and forced playouts");
        SearchPool halving(codec, std::make_shared<ToyEncoder>(), gumbel);
        require(refused([&] { halving.add(game, 1, SearchOptions{{}, {}, {}, temperature, forced}); }) &&
                    halving.size() == 0,
                "a Gumbel search refuses a root temperature and forced playouts");
    }
    SearchPool halving(codec, std::make_shared<ToyEncoder>(), gumbel);
    const auto off = halving.add(game, 1, SearchOptions{{}, {}, {}, 1.0, 0.0});
    run(halving, hashed_evaluator(4));
    require(halving.done(off) && halving.root_search_priors(off) == halving.root_priors(off),
            "a Gumbel search takes both options switched off, and selects by the network's priors");
}

/** Everything a finished search reports, bit for bit (q is NaN where unvisited). */
std::vector<std::uint64_t> report(SearchPool& pool, std::int64_t id) {
    std::vector<std::uint64_t> out{static_cast<std::uint64_t>(id), pool.done(id) ? 1U : 0U};
    for (const std::uint32_t visits : pool.visits(id)) {
        out.push_back(visits);
    }
    for (const double q : pool.q(id)) {
        out.push_back(std::bit_cast<std::uint64_t>(q));
    }
    for (const float p : pool.policy_target(id)) {
        out.push_back(std::bit_cast<std::uint32_t>(p));
    }
    for (const float v : pool.value(id)) {
        out.push_back(std::bit_cast<std::uint32_t>(v));
    }
    const SearchStats stats = pool.stats(id);
    out.insert(out.end(), {stats.nodes, stats.games, stats.simulations, stats.evaluations, pool.select(id)});
    return out;
}

/**
 * SearchConfig::free_backlog only moves where a removed search's trees are
 * freed. A pool that adds, re-roots and removes searches between and during
 * rounds reports the same, bit for bit and under the same ids, whether
 * remove frees them (0, or one thread), the pool's threads do at the next
 * collect, or a small bound has them freed from inside remove as well.
 */
void test_free_backlog_changes_no_result() {
    const Played played = play_random(2, 21, 120);
    const auto codec = std::make_shared<FlatActionCodec>(2, MapType::base);
    const Evaluator evaluate = hashed_evaluator(2);
    struct Outcome {
        std::vector<std::uint64_t> log;
        std::size_t most_unfreed{};
        std::size_t largest{};
        std::size_t flushed_in_remove{};
    };
    const auto churn = [&](int free_backlog, int threads) {
        SearchConfig config;
        config.simulations = 80;
        config.max_players = 2;
        config.max_pending = 2;
        config.threads = threads;
        config.free_backlog = free_backlog;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        std::vector<float> observations(64 * pool.observation_size());
        std::vector<std::int64_t> rows(64);
        // One collect and apply; false once every search is done.
        const auto step = [&] {
            const std::size_t n = pool.collect(observations, rows);
            std::vector<float> logits(n * pool.action_size());
            std::vector<float> values(n * 2);
            for (std::size_t row = 0; row < n; ++row) {
                const std::size_t width = pool.action_size();
                evaluate(
                    pool.pending_game(row), std::span<float>(logits).subspan(row * width, width),
                    std::span<float>(values).subspan(row * 2, 2));
            }
            if (n > 0) {
                pool.apply(logits, values);
            }
            return n > 0;
        };
        Outcome outcome;
        const auto remove = [&](std::int64_t id) {
            outcome.largest = std::max<std::size_t>(outcome.largest, pool.stats(id).nodes);
            const std::size_t before = pool.unfreed_nodes();
            pool.remove(id);
            outcome.flushed_in_remove += pool.unfreed_nodes() < before;
            outcome.most_unfreed = std::max(outcome.most_unfreed, pool.unfreed_nodes());
            require(pool.unfreed_nodes() <= static_cast<std::size_t>(free_backlog), "the backlog keeps to its bound");
        };
        std::vector<std::int64_t> live;
        std::uint64_t seed = 0;
        for (int round = 0; round < 6; ++round) {
            while (live.size() < 10) {
                live.push_back(pool.add(played.game, seed++));
            }
            // A search goes while it and the others are still running.
            for (int i = 0; i < 6; ++i) {
                require(step(), "the searches are still running");
            }
            require(!pool.done(live[3]), "the search removed halfway is unfinished");
            remove(live[3]);
            live.erase(live.begin() + 3);
            while (step()) {
            }
            require(pool.unfreed_nodes() == 0, "collect frees what remove left");
            std::vector<std::int64_t> kept;
            for (std::size_t i = 0; i < live.size(); ++i) {
                const std::int64_t id = live[i];
                const auto reported = report(pool, id);
                outcome.log.insert(outcome.log.end(), reported.begin(), reported.end());
                const std::size_t chosen = pool.select(id);
                const bool keep = i % 3 == 1   ? pool.advance(id, chosen)
                                  : i % 3 == 2 ? pool.reroot(id, chosen, SearchOptions{40, {}, {}, {}, {}})
                                               : false;
                outcome.most_unfreed = std::max(outcome.most_unfreed, pool.unfreed_nodes());
                require(pool.unfreed_nodes() <= static_cast<std::size_t>(free_backlog), "a new root keeps to it too");
                outcome.log.push_back(keep ? 1U : 0U);
                if (keep) {
                    kept.push_back(id);
                } else {
                    remove(id);
                }
            }
            live = kept;
            for (const std::int64_t id : pool.ids()) {
                outcome.log.push_back(static_cast<std::uint64_t>(id));
            }
        }
        return outcome;
    };
    const Outcome inside = churn(0, 4);
    require(inside.most_unfreed == 0 && inside.largest > 0, "at 0 every search is freed inside remove");
    const Outcome threaded = churn(SearchConfig{}.free_backlog, 4);
    require(threaded.most_unfreed > inside.largest, "by default the pool's threads free the trees, several at once");
    require(threaded.log == inside.log, "who frees the trees must not change a result");
    // Room for one tree and a half: the next one has remove free what waits.
    const Outcome tight = churn(static_cast<int>(inside.largest + inside.largest / 2), 4);
    require(tight.most_unfreed > 0 && tight.flushed_in_remove > 0, "a full backlog is freed from inside remove");
    require(tight.log == inside.log, "nor must a backlog that keeps filling");
    const Outcome single = churn(SearchConfig{}.free_backlog, 1);
    require(single.most_unfreed == 0, "a pool of one thread has no thread to leave the trees to");
    require(single.log == inside.log, "one thread reports the same");
}

/**
 * advance and reroot cut most of a tree away, and SearchConfig::free_backlog
 * leaves that to the pool's threads as well. Searches re-rooted round after
 * round (of one world or several, the budget started again or the root's
 * visits carried) report the same, bit for bit, whoever frees what is cut
 * away; and what waits is every node the new roots left behind, counted
 * node for node, within the bound, and freed by the next collect.
 */
void test_free_backlog_covers_what_a_new_root_cuts_away() {
    const Played played = play_random(2, 21, 120);
    const HandTracker& tracker = played.trackers[static_cast<std::size_t>(played.game.current_player_index())];
    const auto codec = std::make_shared<FlatActionCodec>(2, MapType::base);
    struct Outcome {
        std::vector<std::uint64_t> log;
        std::size_t rerooted{};
        /** The most nodes one new root cut away. */
        std::size_t largest{};
        std::size_t most_unfreed{};
        /** How often what waited was freed to make room. */
        std::size_t flushes{};
    };
    const auto reuse = [&](int determinizations, std::size_t bound, int threads) {
        SearchConfig config;
        config.simulations = 96;
        config.max_players = 2;
        config.max_pending = 2;
        config.determinizations = determinizations;
        config.threads = threads;
        config.free_backlog = static_cast<int>(bound);
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        Outcome outcome;
        std::size_t waiting = 0;
        // What the pool does with `nodes` discarded nodes.
        const auto discarded = [&](std::size_t nodes) {
            if (nodes > 0 && nodes <= bound && threads > 1) {
                if (nodes > bound - waiting) {
                    waiting = 0;
                    ++outcome.flushes;
                }
                waiting += nodes;
            }
            require(pool.unfreed_nodes() == waiting && waiting <= bound,
                    "what waits is counted node for node, within the bound");
            outcome.most_unfreed = std::max(outcome.most_unfreed, waiting);
        };
        std::uint64_t seed = 0;
        const auto add = [&] { return pool.add(played.game, seed++, determinizations > 0 ? &tracker : nullptr); };
        std::vector<std::int64_t> ids;
        for (int i = 0; i < 8; ++i) {
            ids.push_back(add());
        }
        for (int round = 0; round < 5; ++round) {
            run(pool, hashed_evaluator(2));
            require(pool.unfreed_nodes() == 0, "a collect frees what waits");
            waiting = 0;
            for (std::size_t i = 0; i < ids.size(); ++i) {
                std::int64_t& id = ids[i];
                const auto reported = report(pool, id);
                outcome.log.insert(outcome.log.end(), reported.begin(), reported.end());
                // The most visited root action the search can be re-rooted at.
                const auto visits = pool.visits(id);
                std::vector<std::size_t> order;
                for (std::size_t action = 0; action < visits.size(); ++action) {
                    if (visits[action] > 0) {
                        order.push_back(action);
                    }
                }
                std::stable_sort(
                    order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return visits[a] > visits[b]; });
                const std::size_t before = pool.stats(id).nodes;
                bool moved = false;
                for (const std::size_t action : order) {
                    moved = (i + static_cast<std::size_t>(round)) % 2 == 0
                        ? pool.advance(id, action)
                        : pool.reroot(id, action, SearchOptions{96, {}, {}, {}, {}});
                    if (moved) {
                        outcome.log.push_back(action);
                        break;
                    }
                    require(pool.unfreed_nodes() == waiting && pool.stats(id).nodes == before,
                            "a refused new root discards nothing");
                }
                outcome.log.push_back(moved ? 1U : 0U);
                if (moved) {
                    const std::size_t kept = pool.stats(id).nodes;
                    require(kept < before, "a new root leaves nodes behind");
                    outcome.largest = std::max(outcome.largest, before - kept);
                    ++outcome.rerooted;
                    discarded(before - kept);
                } else {
                    pool.remove(id);
                    discarded(before);
                    id = add();
                }
            }
        }
        return outcome;
    };
    const auto standard = static_cast<std::size_t>(SearchConfig{}.free_backlog);
    for (const int determinizations : {0, 3}) {
        const Outcome inside = reuse(determinizations, 0, 4);
        require(inside.most_unfreed == 0 && inside.rerooted >= 16, "at 0 advance and reroot free what they cut away");
        const Outcome threaded = reuse(determinizations, standard, 4);
        require(threaded.most_unfreed > inside.largest && threaded.flushes == 0,
                "by default the pool's threads free it, a round's worth at once");
        require(threaded.log == inside.log, "who frees what a new root cuts away must not change a result");
        // Room for one new root's nodes and a half: the next has what waits freed first.
        const Outcome tight = reuse(determinizations, inside.largest + inside.largest / 2, 4);
        require(tight.most_unfreed > 0 && tight.flushes > 0, "a full backlog is freed from inside advance and reroot");
        require(tight.log == inside.log, "nor must a backlog that keeps filling");
        const Outcome single = reuse(determinizations, standard, 1);
        require(single.most_unfreed == 0, "a pool of one thread has no thread to leave it to");
        require(single.log == inside.log, "one thread reports the same");
    }
}

/**
 * What waits to be freed keeps to SearchConfig::free_backlog, tree node for
 * tree node: a search that does not fit beside what waits has it freed
 * first, one larger than the bound is freed inside remove, and the next
 * collect frees the rest. remove refuses what it always refused.
 */
void test_free_backlog_bounds_what_waits() {
    const Played played = play_random(2, 21, 120);
    const auto codec = std::make_shared<FlatActionCodec>(2, MapType::base);
    SearchConfig config;
    config.simulations = 64;
    config.max_players = 2;
    config.threads = 3;
    const std::size_t searches = 12;
    const auto searched = [&](SearchPool& pool) {
        std::vector<std::int64_t> ids;
        for (std::uint64_t seed = 0; seed < searches; ++seed) {
            // One search in four is several times the size of the others.
            ids.push_back(pool.add(played.game, seed, SearchOptions{seed % 4 == 3 ? 640 : 64, {}, {}, {}, {}}));
        }
        run(pool, hashed_evaluator(2));
        return ids;
    };
    std::vector<std::size_t> nodes;
    std::size_t bound = 0;
    {
        config.free_backlog = 0;
        SearchPool sizes(codec, std::make_shared<ToyEncoder>(), config);
        for (const std::int64_t id : searched(sizes)) {
            nodes.push_back(sizes.stats(id).nodes);
            if (id % 4 != 3) {
                bound = std::max(bound, 2 * nodes.back() + nodes.back() / 2);
            }
        }
    }
    config.free_backlog = static_cast<int>(bound);
    SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
    const std::vector<std::int64_t> ids = searched(pool);
    require(pool.unfreed_nodes() == 0, "nothing waits before a search is removed");
    std::size_t expected = 0;
    int inside = 0;
    int flushes = 0;
    int left = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        require(pool.stats(ids[i]).nodes == nodes[i], "the searches are those measured");
        if (nodes[i] > bound) {
            ++inside;  // too large to wait: freed inside remove
        } else {
            if (nodes[i] > bound - expected) {
                ++flushes;  // no room beside what waits: that is freed first
                expected = 0;
            }
            expected += nodes[i];
            ++left;
        }
        pool.remove(ids[i]);
        require(pool.unfreed_nodes() == expected && expected <= bound,
                "what waits is counted node for node, within the bound");
        require(!pool.contains(ids[i]) && pool.size() == ids.size() - i - 1, "the search is gone at once either way");
    }
    require(inside == 3 && flushes >= 2 && left == 9 && expected > 0, "the searches exercise every case");
    std::vector<float> observations(4 * pool.observation_size());
    std::vector<std::int64_t> rows(4);
    require(pool.collect(observations, rows) == 0 && pool.unfreed_nodes() == 0,
            "a collect frees what waits, even with no search left");

    // remove refuses an unknown id, and a search with leaves awaiting apply, which stays whole.
    const auto id = pool.add(played.game, 99);
    const auto other = pool.add(played.game, 100);
    run(pool, hashed_evaluator(2));
    pool.remove(other);
    const std::size_t waiting = pool.unfreed_nodes();
    require(waiting > 0, "a removed search waits");
    bool unknown = false;
    try {
        pool.remove(other);
    } catch (const std::out_of_range&) {
        unknown = true;
    }
    require(unknown && pool.unfreed_nodes() == waiting, "an unknown id is refused as before");
    const auto busy = pool.add(played.game, 101);
    require(pool.collect(observations, rows) == 1 && pool.unfreed_nodes() == 0,
            "the new search's root awaits its evaluation");
    bool pending = false;
    try {
        pool.remove(busy);
    } catch (const std::logic_error&) {
        pending = true;
    }
    require(pending && pool.contains(busy) && pool.unfreed_nodes() == 0,
            "a search with leaves awaiting apply is refused as before");
    pool.cancel();
    run(pool, hashed_evaluator(2));
    require(pool.done(busy) && pool.stats(busy).simulations == 64 && pool.done(id), "and searches on, whole");

    config.free_backlog = -1;
    bool refused = false;
    try {
        SearchPool(codec, std::make_shared<ToyEncoder>(), config);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "a negative backlog is refused");
}

/**
 * Searches come and go many times over, and pools are destroyed (and moved)
 * with trees still waiting for their threads and searches still alive:
 * nothing crashes, hangs or leaks (run it under a sanitizer to see the last).
 */
void test_pools_die_with_trees_still_waiting() {
    const Played played = play_random(2, 21, 120);
    const auto codec = std::make_shared<FlatActionCodec>(2, MapType::base);
    for (const int threads : {1, 4}) {
        for (const int free_backlog : {0, 150, SearchConfig{}.free_backlog}) {
            SearchConfig config;
            config.simulations = 24;
            config.max_players = 2;
            config.threads = threads;
            config.free_backlog = free_backlog;
            std::optional<SearchPool> pool;
            std::size_t waited = 0;
            for (int i = 0; i < 240; ++i) {
                if (i % 40 == 0) {
                    waited += pool.has_value() ? pool->unfreed_nodes() : 0;
                    pool.emplace(codec, std::make_shared<ToyEncoder>(), config);  // the last pool dies as it is
                }
                std::vector<std::int64_t> ids;
                for (std::uint64_t seed = 0; seed < 6; ++seed) {
                    ids.push_back(pool->add(played.game, static_cast<std::uint64_t>(i) * 6 + seed));
                }
                run(*pool, hashed_evaluator(2));
                // Some searches stay alive, for the pool's end or a later round.
                for (std::size_t k = static_cast<std::size_t>(i % 3); k < ids.size(); ++k) {
                    pool->remove(ids[k]);
                }
                if (i % 7 == 0) {
                    for (const std::int64_t id : pool->ids()) {
                        pool->remove(id);
                    }
                }
            }
            const std::size_t waiting = pool->unfreed_nodes();
            SearchPool moved(std::move(*pool));
            require(moved.unfreed_nodes() == waiting, "a moved pool keeps what waits");
            pool.reset();
            const bool leaves = threads > 1 && free_backlog == SearchConfig{}.free_backlog;
            require((waited + waiting > 0) == (threads > 1 && free_backlog > 0) && (!leaves || waiting > 0),
                    "pools were destroyed with trees waiting where trees wait at all");
        }
    }
}

/**
 * A position of a random two-player game for likely worlds: the seat to
 * move has a choice and the other seat holds `hidden` unplayed development
 * cards (`or_more`: at least); `building`: a road is among the choices, a
 * decision of its own turn that a determinized search can re-root through.
 */
Played hidden_cards_position(int hidden, bool or_more = false, bool building = false) {
    for (std::uint64_t seed = 1; seed <= 60; ++seed) {
        Played played{Game(seats(2), MapType::base, seed), {}};
        played.trackers.emplace_back(played.game, 0);
        played.trackers.emplace_back(played.game, 1);
        std::mt19937_64 rng(seed);
        for (int step = 0; step < 1200; ++step) {
            const auto& actions = played.game.playable_actions();
            const Action action = actions[rng() % actions.size()];
            const Game before = played.game;
            played.game.execute(action);
            for (auto& tracker : played.trackers) {
                tracker.observe(before, action, played.game);
            }
            if (played.game.winning_color().has_value()) {
                break;
            }
            const int mover = played.game.current_player_index();
            const int held = played.trackers[static_cast<std::size_t>(mover)].development_card_count(1 - mover);
            const auto& options = played.game.playable_actions();
            const bool builds = std::any_of(options.begin(), options.end(), [](const Action& option) {
                return option.type == ActionType::build_road;
            });
            if (step % 20 == 0 && options.size() >= 2 && (held == hidden || (or_more && held > hidden)) &&
                (!building || (builds && played.game.current_prompt() == ActionPrompt::play_turn))) {
                return played;
            }
        }
    }
    throw std::runtime_error("no random game reached the position asked for");
}

/** The tracker of the seat to move. */
const HandTracker& mover_tracker(const Played& played) {
    return played.trackers[static_cast<std::size_t>(played.game.current_player_index())];
}

bool same_hidden_cards(const Game& world, const std::vector<HiddenHand>& hands) {
    for (std::size_t seat = 0; seat < hands.size(); ++seat) {
        const PlayerState& player = world.players()[seat];
        if (player.resources != hands[seat].resources ||
            player.development_cards != hands[seat].development_cards ||
            player.development_card_owned_at_start != hands[seat].development_card_owned_at_start) {
            return false;
        }
    }
    return true;
}

/** The simulations each root world has had: its root visits. */
std::vector<double> world_totals(const SearchPool& pool, std::int64_t id) {
    const auto visits = pool.world_visits(id);
    std::vector<double> totals(visits.size() / pool.action_size(), 0.0);
    for (std::size_t i = 0; i < visits.size(); ++i) {
        totals[i / pool.action_size()] += visits[i];
    }
    return totals;
}

/** The pool's visits are the worlds' visits, summed. */
void require_pooled(const SearchPool& pool, std::int64_t id) {
    const auto visits = pool.visits(id);
    const auto by_world = pool.world_visits(id);
    std::vector<std::uint32_t> summed(visits.size(), 0);
    for (std::size_t i = 0; i < by_world.size(); ++i) {
        summed[i % visits.size()] += by_world[i];
    }
    require(summed == visits, "the root visits are the worlds' visits summed");
}

SearchConfig likely_config(RootSelection root, int determinizations, int simulations) {
    SearchConfig config;
    config.root = root;
    config.determinizations = determinizations;
    config.world_choice = WorldChoice::likely;
    config.simulations = simulations;
    config.max_players = 2;
    return config;
}

/**
 * Likely worlds are the most likely deals, in order, whatever the seed:
 * the eight most likely of more with their probabilities among the eight,
 * every deal with its own probability when there are no more than asked
 * for, and one world with the whole budget when nothing is hidden. Only the
 * rest of a world (its deck's order) follows the seed.
 */
void test_likely_worlds_are_the_most_likely_deals() {
    const Played played = hidden_cards_position(4, true);
    const Game& game = played.game;
    const HandTracker& tracker = mover_tracker(played);
    const std::vector<LikelyWorld> deals = likely_worlds(game, tracker);
    require(deals.size() > 8, "four hidden cards have more than eight deals");
    double top = 0.0;
    for (std::size_t i = 0; i < 8; ++i) {
        top += deals[i].probability;
    }
    const auto codec = std::make_shared<FlatActionCodec>(2, MapType::base);
    std::vector<float> observations(64 * ToyEncoder().size());
    std::vector<std::int64_t> ids(64);
    for (const RootSelection root : {RootSelection::puct, RootSelection::gumbel}) {
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), likely_config(root, 8, 64));
        std::vector<std::vector<DevelopmentCard>> decks;
        for (const std::uint64_t seed : {5, 6}) {
            const auto id = pool.add(game, seed, &tracker);
            require(pool.stats(id).worlds == 8, "eight worlds of more deals");
            const auto weights = pool.world_weights(id);
            // Every world's root is evaluated before any simulation.
            require(pool.collect(observations, ids) == 8, "the eight roots come first");
            for (std::size_t w = 0; w < 8; ++w) {
                require(same_hidden_cards(pool.pending_game(w), deals[w].hands),
                        "world w holds the w-th most likely deal, whatever the seed");
                require(std::abs(weights[w] - deals[w].probability / top) < 1e-12,
                        "a world weighs its probability among the worlds kept");
                require(!pool.pending_game(w).search_equivalent(game) || same_hidden_cards(game, deals[w].hands),
                        "only the true deal's world is the true game");
            }
            decks.push_back(pool.pending_game(0).development_deck());
            pool.cancel();
            pool.remove(id);
        }
        require(decks[0] != decks[1] && std::is_permutation(decks[0].begin(), decks[0].end(), decks[1].begin(), decks[1].end()),
                "the seed shuffles a world's deck and nothing else");

        // Asked for more worlds than there are deals: all of them, as likely as they are.
        SearchPool wide(codec, std::make_shared<ToyEncoder>(), likely_config(root, 1000, 64));
        const auto all = wide.add(game, 5, &tracker);
        const auto weights = wide.world_weights(all);
        require(weights.size() == deals.size() && wide.stats(all).worlds == deals.size(), "every deal is a world");
        for (std::size_t w = 0; w < weights.size(); ++w) {
            require(std::abs(weights[w] - deals[w].probability) < 1e-12, "all deals keep their exact probabilities");
        }
        run(wide, hashed_evaluator(2));
        require(wide.done(all) && total(wide.visits(all)) == 64, "more worlds than simulations still search");
        require_pooled(wide, all);

        // Nothing hidden: one world, the whole budget.
        const Played open = hidden_cards_position(0);
        require(likely_worlds(open.game, mover_tracker(open)).size() == 1, "nothing hidden is one deal");
        const auto one = pool.add(open.game, 5, &mover_tracker(open));
        run(pool, hashed_evaluator(2));
        require(pool.stats(one).worlds == 1 && pool.world_weights(one) == std::vector<double>{1.0},
                "one world of weight 1");
        require(world_totals(pool, one) == std::vector<double>{64.0}, "the one world has the whole budget");
        require(pool.world_visits(one) == pool.visits(one), "its visits are the root's");
    }

    // Sampled worlds report their share of the draws; an oracle its one world.
    SearchConfig sampled;
    sampled.determinizations = 8;
    sampled.max_players = 2;
    sampled.simulations = 64;
    SearchPool pool(codec, std::make_shared<ToyEncoder>(), sampled);
    const auto id = pool.add(game, 5, &tracker);
    const auto shares = pool.world_weights(id);
    require(shares.size() == pool.stats(id).worlds, "a share per sampled world");
    for (const double share : shares) {
        require(share > 0.0 && std::abs(share * 8.0 - std::round(share * 8.0)) < 1e-12, "shares are eighths");
    }
    require(std::abs(std::accumulate(shares.begin(), shares.end(), 0.0) - 1.0) < 1e-12, "the shares sum to 1");
    run(pool, hashed_evaluator(2));
    const auto sampled_totals = world_totals(pool, id);
    for (std::size_t w = 0; w < shares.size(); ++w) {
        require(sampled_totals[w] == shares[w] * 64.0, "a sampled world gets its draws' simulations");
    }
    require_pooled(pool, id);
}

/**
 * A PUCT search gives every likely world its probability's share of the
 * simulations: to within one, after any number of them, so a search cut
 * short is as well shared as a finished one.
 */
void test_likely_worlds_share_the_simulations() {
    const Played played = hidden_cards_position(4, true);
    const Game& game = played.game;
    const HandTracker& tracker = mover_tracker(played);
    const auto codec = std::make_shared<FlatActionCodec>(2, MapType::base);
    const auto evaluate = hashed_evaluator(2);
    const auto within_one = [](const SearchPool& pool, std::int64_t id, double simulations) {
        const auto weights = pool.world_weights(id);
        const auto totals = world_totals(pool, id);
        for (std::size_t w = 0; w < weights.size(); ++w) {
            if (std::abs(totals[w] - weights[w] * simulations) > 1.0 + 1e-9) {
                return false;
            }
        }
        return true;
    };
    {
        // One leaf at a time, checked after every simulation.
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), likely_config(RootSelection::puct, 8, 400));
        const auto id = pool.add(game, 3, &tracker);
        std::vector<float> observations(pool.observation_size());
        std::vector<std::int64_t> ids(1);
        std::vector<float> logits(pool.action_size());
        std::vector<float> values(2);
        int checked = 0;
        while (pool.collect(observations, ids) == 1) {
            evaluate(pool.pending_game(0), logits, values);
            pool.apply(logits, values);
            const double simulations = static_cast<double>(pool.stats(id).simulations);
            require(total(pool.visits(id)) == simulations, "every simulation is a root visit");
            require(within_one(pool, id, simulations),
                    "after " + std::to_string(pool.stats(id).simulations) + " simulations every world is within one of its share");
            ++checked;
        }
        require(pool.done(id) && pool.stats(id).simulations == 400 && checked >= 400, "the search ran out its budget");
        require_pooled(pool, id);
        const auto weights = pool.world_weights(id);
        const auto totals = world_totals(pool, id);
        require(totals.front() > totals.back() && weights.front() > weights.back(),
                "the most likely world is searched more than the least likely one");
    }
    // Leaves in flight, several threads, and budgets of the search's own.
    SearchConfig config = likely_config(RootSelection::puct, 8, 1000);
    config.max_pending = 8;
    config.threads = 2;
    SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
    std::vector<std::pair<std::int64_t, int>> searches{{pool.add(game, 3, &tracker), 1000}};
    for (const int budget : {1, 5, 37, 250}) {
        searches.emplace_back(pool.add(game, 3, SearchOptions{budget, {}, {}, {}, {}}, &tracker), budget);
    }
    run(pool, evaluate, 32);
    for (const auto& [id, budget] : searches) {
        require(pool.done(id) && total(pool.visits(id)) == static_cast<std::uint32_t>(budget), "the budget is spent");
        require(within_one(pool, id, budget), "a budget of " + std::to_string(budget) + " is shared by weight");
        require_pooled(pool, id);
        const auto q = pool.q(id);
        const auto visits = pool.visits(id);
        require(visits[pool.select(id)] > 0 && !std::isnan(q[pool.select(id)]), "the move chosen was searched");
    }
    // Withdrawn leaves leave the shares as they were.
    const auto id = pool.add(game, 4, &tracker);
    std::vector<float> observations(32 * pool.observation_size());
    std::vector<std::int64_t> ids(32);
    std::vector<float> logits(32 * pool.action_size());
    std::vector<float> values(32 * 2);
    for (int round = 0; round < 40; ++round) {
        const std::size_t rows = pool.collect(observations, ids);
        if (round % 3 == 2) {
            pool.cancel();
            continue;
        }
        for (std::size_t row = 0; row < rows; ++row) {
            evaluate(pool.pending_game(row), std::span<float>(logits).subspan(row * pool.action_size(), pool.action_size()),
                     std::span<float>(values).subspan(row * 2, 2));
        }
        pool.apply(std::span<const float>(logits).first(rows * pool.action_size()),
                   std::span<const float>(values).first(rows * 2));
        require(within_one(pool, id, static_cast<double>(pool.stats(id).simulations)),
                "cancelled leaves do not shift the worlds' shares");
    }
}

/**
 * A Gumbel root over likely worlds: its logits are the worlds' log-priors
 * averaged by their weights, and every root action's simulations are shared
 * between the worlds by weight, so the actions Sequential Halving compares
 * were searched in the same worlds.
 */
void test_likely_worlds_at_a_gumbel_root() {
    const Played played = hidden_cards_position(4, true);
    const Game& game = played.game;
    const HandTracker& tracker = mover_tracker(played);
    const auto codec = std::make_shared<FlatActionCodec>(2, MapType::base);
    const std::size_t actions = codec->size();
    {
        // Equal values everywhere: the improved policy is the root's logits.
        SearchConfig config = likely_config(RootSelection::gumbel, 8, 8);
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        const auto id = pool.add(game, 3, &tracker);
        // Logits that differ widely between the actions and between the worlds.
        run(pool, [](const Game& leaf, std::span<float> logits, std::span<float> values) {
            const std::uint64_t hash = signature(leaf);
            for (std::size_t i = 0; i < logits.size(); ++i) {
                std::mt19937_64 scramble(hash + 0x9e3779b97f4a7c15ULL * (i + 1));
                logits[i] = static_cast<float>(scramble() % 2001) / 500.0F - 2.0F;
            }
            std::fill(values.begin(), values.end(), 0.5F);
        });
        const auto priors = pool.root_priors(id);
        const auto weights = pool.world_weights(id);
        require(!std::equal(priors.begin(), priors.begin() + static_cast<std::ptrdiff_t>(actions),
                            priors.begin() + static_cast<std::ptrdiff_t>(actions)),
                "the worlds' priors differ");
        std::vector<double> by_weight(actions, 0.0);
        std::vector<double> evenly(actions, 0.0);
        double weight_sum = 0.0;
        double even_sum = 0.0;
        for (std::size_t a = 0; a < actions; ++a) {
            if (priors[a] <= 0.0F) {
                continue;
            }
            double weighted = 0.0;
            double even = 0.0;
            for (std::size_t w = 0; w < weights.size(); ++w) {
                weighted += weights[w] * std::log(static_cast<double>(priors[w * actions + a]));
                even += std::log(static_cast<double>(priors[w * actions + a])) / static_cast<double>(weights.size());
            }
            weight_sum += by_weight[a] = std::exp(weighted);
            even_sum += evenly[a] = std::exp(even);
        }
        const auto policy = pool.policy_target(id);
        double off_weights = 0.0;
        double off_even = 0.0;
        for (std::size_t a = 0; a < actions; ++a) {
            off_weights = std::max(off_weights, std::abs(policy[a] - by_weight[a] / weight_sum));
            off_even = std::max(off_even, std::abs(policy[a] - evenly[a] / even_sum));
        }
        require(off_weights < 1e-6, "the root's logits are the worlds' log-priors averaged by weight");
        require(off_even > 1e-2, "which is not their plain average");
    }
    for (const int considered : {4, 16}) {
        SearchConfig config = likely_config(RootSelection::gumbel, 8, 600);
        config.gumbel_considered = considered;
        config.max_pending = 4;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);
        const auto id = pool.add(game, 3, &tracker);
        run(pool, hashed_evaluator(2), 16);
        require(pool.done(id) && total(pool.visits(id)) == 600, "the Gumbel root spends its budget");
        require_pooled(pool, id);
        const auto weights = pool.world_weights(id);
        const auto visits = pool.visits(id);
        const auto by_world = pool.world_visits(id);
        int searched = 0;
        for (std::size_t a = 0; a < actions; ++a) {
            searched += visits[a] > 0;
            for (std::size_t w = 0; w < weights.size(); ++w) {
                require(std::abs(by_world[w * actions + a] - weights[w] * visits[a]) <= 1.0 + 1e-9,
                        "every action's simulations are shared between the worlds by weight");
            }
        }
        require(searched >= std::min(considered, 2), "Sequential Halving considered several actions");
        require(visits[pool.select(id)] > 0, "the action chosen was searched");
        const auto policy = pool.policy_target(id);
        require(std::abs(std::accumulate(policy.begin(), policy.end(), 0.0) - 1.0) < 1e-5, "the improved policy sums to 1");
    }
}

/**
 * Re-rooting keeps the likely worlds and their weights, which are the ones
 * a new search of the position reached would take: what advance accepts
 * reveals nothing. Every world must have searched the move, the least
 * likely one too, whose share of a small budget may be no simulation of it.
 * A refused advance is followed by a new search, whose worlds are the most
 * likely deals of the new position.
 */
void test_likely_worlds_and_reuse() {
    Played played = hidden_cards_position(2, true, true);
    Game& game = played.game;
    const int mover = game.current_player_index();
    const auto codec = std::make_shared<FlatActionCodec>(2, MapType::base);
    const auto& options = game.playable_actions();
    const Action road = *std::find_if(options.begin(), options.end(), [](const Action& option) {
        return option.type == ActionType::build_road;
    });
    const Action pass = *std::find_if(options.begin(), options.end(), [](const Action& option) {
        return option.type == ActionType::end_turn;
    });
    const std::size_t road_index = codec->index(game, road);
    const std::size_t pass_index = codec->index(game, pass);
    Game built = game;
    built.execute(road);
    std::vector<HandTracker> after = played.trackers;
    for (HandTracker& tracker : after) {
        tracker.observe(game, road, built);
    }
    // At the root the road and the pass are favoured, so both are searched.
    const Evaluator favouring = [&](const Game& leaf, std::span<float> logits, std::span<float> values) {
        std::fill(logits.begin(), logits.end(), 0.0F);
        if (leaf.num_turns() == game.num_turns() && leaf.playable_actions() == game.playable_actions()) {
            logits[road_index] = 6.0F;
            logits[pass_index] = 3.0F;
        }
        std::fill(values.begin(), values.end(), 0.5F);
    };
    const int budget = 1200;
    for (const RootSelection root : {RootSelection::puct, RootSelection::gumbel}) {
        const HandTracker& tracker = played.trackers[static_cast<std::size_t>(mover)];
        {
            // A world that has not searched the move cannot be re-rooted on it.
            SearchPool brief(codec, std::make_shared<ToyEncoder>(), likely_config(root, 8, 40));
            const auto id = brief.add(game, 4, &tracker);
            run(brief, favouring);
            const auto by_world = brief.world_visits(id);
            bool everywhere = true;
            for (std::size_t w = 0; w < 8; ++w) {
                everywhere = everywhere && by_world[w * codec->size() + road_index] > 0;
            }
            require(brief.visits(id)[road_index] > 0 && !everywhere, "the least likely world's share is too small for the road");
            require(!brief.advance(id, road_index), "an advance needs the move searched in every world");
        }
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), likely_config(root, 8, budget));
        const auto id = pool.add(game, 4, &tracker);
        run(pool, favouring);
        require(pool.visits(id)[road_index] > 0 && pool.visits(id)[pass_index] > 0, "both moves were searched");
        const auto weights = pool.world_weights(id);
        require(weights.size() == 8, "two hidden cards have more than eight deals");

        // Ending the turn hands the move over: refused, the search unchanged.
        require(!pool.advance(id, pass_index), "another seat's decision is no root for these worlds");
        require(pool.world_weights(id) == weights && pool.done(id), "a refused advance changes nothing");

        require(pool.advance(id, road_index), "an own decision re-roots the likely worlds");
        require(pool.world_weights(id) == weights && pool.stats(id).worlds == 8, "the worlds and weights are kept");
        const std::vector<LikelyWorld> fresh = likely_worlds(built, after[static_cast<std::size_t>(mover)]);
        double top = 0.0;
        for (std::size_t w = 0; w < 8; ++w) {
            top += fresh[w].probability;
        }
        for (std::size_t w = 0; w < 8; ++w) {
            require(std::abs(weights[w] - fresh[w].probability / top) < 1e-12,
                    "a new search of the new position would weigh its worlds the same");
        }
        const auto carried = world_totals(pool, id);
        run(pool, hashed_evaluator(2));
        require(pool.done(id) && pool.stats(id).simulations == budget, "the re-rooted search runs a full budget");
        require_pooled(pool, id);
        const auto totals = world_totals(pool, id);
        std::vector<std::uint8_t> legal(codec->size());
        codec->mask(built, legal);
        const auto visits = pool.visits(id);
        for (std::size_t a = 0; a < visits.size(); ++a) {
            require(visits[a] == 0 || legal[a] != 0, "the re-rooted search visits the new position's moves");
        }
        if (root == RootSelection::puct) {
            for (std::size_t w = 0; w < 8; ++w) {
                require(std::abs(totals[w] - carried[w] - weights[w] * budget) <= 1.0 + 1e-9,
                        "the new budget is shared by weight again");
            }
            // Under a visit cap the carried visits count, and the worlds end by weight.
            SearchPool capped(codec, std::make_shared<ToyEncoder>(), likely_config(root, 8, budget));
            const auto again = capped.add(game, 4, &tracker);
            run(capped, favouring);
            const std::uint32_t cap = 1500;
            require(capped.reroot(again, road_index, SearchOptions{static_cast<int>(cap), {}, {}, {}, {}}),
                    "likely worlds re-root under a cap");
            const std::uint32_t kept = total(capped.visits(again));
            require(kept > 0 && kept < cap, "the worlds' visits are carried");
            run(capped, hashed_evaluator(2));
            require(capped.done(again) && total(capped.visits(again)) == cap && capped.stats(again).simulations == cap - kept,
                    "the worlds together are topped up to the cap");
            const auto capped_totals = world_totals(capped, again);
            for (std::size_t w = 0; w < 8; ++w) {
                require(std::abs(capped_totals[w] - weights[w] * cap) <= 1.0 + 1e-9,
                        "the worlds behind their share are topped up first");
            }
        }

        // The refused move, played for real: a new search lists the deals anew.
        pool.remove(id);
        Played next = played;
        std::mt19937_64 rng(11);
        Action action = pass;
        while (true) {
            const Game before = next.game;
            next.game.execute(action);
            for (HandTracker& follower : next.trackers) {
                follower.observe(before, action, next.game);
            }
            require(!next.game.winning_color().has_value(), "the game goes on");
            const auto& choices = next.game.playable_actions();
            if (next.game.current_player_index() == mover && choices.size() >= 2 &&
                next.trackers[static_cast<std::size_t>(mover)].development_card_count(1 - mover) != 2) {
                break;
            }
            action = choices[rng() % choices.size()];
        }
        const HandTracker& updated = next.trackers[static_cast<std::size_t>(mover)];
        const std::vector<LikelyWorld> deals = likely_worlds(next.game, updated);
        const auto renewed = pool.add(next.game, 4, &updated);
        const std::size_t kept = std::min<std::size_t>(deals.size(), 8);
        require(pool.stats(renewed).worlds == kept, "the new search takes the new position's deals");
        std::vector<float> observations(64 * pool.observation_size());
        std::vector<std::int64_t> ids(64);
        require(pool.collect(observations, ids) == kept, "its roots come first");
        for (std::size_t w = 0; w < kept; ++w) {
            require(same_hidden_cards(pool.pending_game(w), deals[w].hands), "its worlds are the new most likely deals");
        }
        pool.cancel();
        run(pool, hashed_evaluator(2));
        require(pool.done(renewed) && total(pool.visits(renewed)) == static_cast<std::uint32_t>(budget), "the new search runs");
    }
}

/**
 * What likely worlds cannot do says so: without determinizations there is
 * nothing to take, and a position with too many deals to list is refused by
 * add, which leaves the pool as it was. Positions of more seats whose deals
 * can be listed are searched like any other.
 */
void test_likely_worlds_refusals() {
    const auto two = std::make_shared<FlatActionCodec>(2, MapType::base);
    bool threw = false;
    try {
        SearchPool pool(two, std::make_shared<ToyEncoder>(), likely_config(RootSelection::puct, 0, 64));
    } catch (const std::invalid_argument& error) {
        threw = std::string(error.what()).find("likely worlds need determinizations") != std::string::npos;
    }
    require(threw, "likely worlds without determinizations are refused");

    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    for (const RootSelection root : {RootSelection::puct, RootSelection::gumbel}) {
        SearchConfig config = likely_config(root, 4, 120);
        config.max_players = 4;
        SearchPool pool(codec, std::make_shared<ToyEncoder>(), config);

        // A tracker started on dealt hands knows none of thirty cards.
        auto players = random_placements(4, 12);
        for (std::size_t seat = 1; seat < 4; ++seat) {
            players[seat].resources = {2, 2, 2, 2, 2};
        }
        Game wide(seats(4), MapType::base, 12);
        GameSetup setup;
        setup.players = players;
        wide.apply_setup(setup);
        const HandTracker wide_tracker(wide, 0);
        threw = false;
        try {
            (void)pool.add(wide, 1, &wide_tracker);
        } catch (const std::invalid_argument& error) {
            threw = std::string(error.what()).find("likely worlds need") != std::string::npos;
        }
        require(threw && pool.size() == 0, "too many deals are refused and no search is added");

        // Three unknown cards and a development card over three seats: a few deals.
        const Game game = knight_and_monopoly_position();
        const HandTracker tracker(game, 0);
        const std::vector<LikelyWorld> deals = likely_worlds(game, tracker);
        require(deals.size() > 4, "the four-player position has more deals than worlds asked for");
        const auto id = pool.add(game, 1, &tracker);
        run(pool, [](const Game& leaf, std::span<float> logits, std::span<float> values) {
            hashed_logits(leaf, logits);
            std::fill(values.begin(), values.end(), 0.25F);
        });
        require(pool.done(id) && pool.stats(id).worlds == 4 && total(pool.visits(id)) == 120,
                "a listable position of four seats is searched");
        require_pooled(pool, id);
    }
}

int main() {
    try {
        test_free_backlog_changes_no_result();
        test_free_backlog_covers_what_a_new_root_cuts_away();
        test_free_backlog_bounds_what_waits();
        test_pools_die_with_trees_still_waiting();
        test_root_options_off_are_bitwise_today();
        test_root_temperature_reshapes_the_priors_in_use();
        test_forced_playouts_select_the_visited_action_furthest_below();
        test_root_options_are_per_search();
        test_search_options_are_per_search();
        test_reroot_counts_the_carried_visits();
        test_reroot_noise_is_per_search();
        test_reroot_refusals();
        test_reroot_determinized();
        test_transpositions_merge_reordered_actions();
        test_transpositions_are_deterministic();
        test_transpositions_off_is_bitwise_today();
        test_advance_on_a_dag();
        test_discard_chain_is_a_lattice();
        test_gumbel_plays_the_best_scoring_finalist();
        test_two_player_matches_mcts_search();
        test_finds_the_saving_move();
        test_max_n_backup();
        test_determinized_search_ignores_hidden_truth();
        test_parallel_and_batched_runs_agree();
        test_edge_cases();
        test_determinized_roots_offer_the_real_options();
        test_determinized_advance_refuses_drawn_outcomes();
        test_determinized_advance_refuses_monopoly();
        test_determinized_advance_keeps_the_friendly_robber_prompt();
        test_failures_leave_the_pool_usable();
        test_searches_spread_over_threads();
        test_tracker_needs_determinizations();
        test_likely_worlds_are_the_most_likely_deals();
        test_likely_worlds_share_the_simulations();
        test_likely_worlds_at_a_gumbel_root();
        test_likely_worlds_and_reuse();
        test_likely_worlds_refusals();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "search pool tests passed\n";
    return 0;
}
