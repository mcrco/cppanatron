// SearchPool: equivalence with MCTSSearch in 2p, N-player max^n backup,
// finding wins, determinized searches that never read the hidden truth.

#include <algorithm>
#include <atomic>
#include <limits>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
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
 * Friendly robber: the root player's robber options depend on others' VP
 * cards. Determinized roots must offer exactly the real options, also when
 * the tracker is misled and no world can.
 */
void test_determinized_roots_offer_the_real_options() {
    GameRules rules;
    rules.friendly_robber = true;
    auto players = random_placements(4, 31);
    players[0].development_cards = {1, 0, 0, 0, 0};
    players[1].development_cards = {0, 0, 0, 0, 1};  // 2 public points, 3 actual: robbable
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
    // Misled: it believes seat 1 holds no VP card, so every world spares seat 1.
    HandTracker misled(before, 0);
    for (const auto& event : HandTracker::events(before, knight, game, 0)) {
        if (event.type != TrackerEventType::victory_point_cards) {
            misled.apply(event);
        }
    }
    misled.apply(TrackerEvent::victory_point_cards(1, 0, 0));

    const auto codec = std::make_shared<FlatActionCodec>(4, MapType::base);
    std::vector<std::uint8_t> legal(codec->size());
    codec->mask(game, legal);
    for (const HandTracker* tracker : {&informed, &misled}) {
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

int main() {
    try {
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
        test_failures_leave_the_pool_usable();
        test_searches_spread_over_threads();
        test_tracker_needs_determinizations();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "search pool tests passed\n";
    return 0;
}
