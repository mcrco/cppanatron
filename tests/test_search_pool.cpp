// SearchPool: equivalence with MCTSSearch in 2p, N-player max^n backup,
// finding wins, determinized searches that never read the hidden truth.

#include <algorithm>
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
        for (std::uint64_t step = 0; step < seed - 1; ++step) {
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
        require(reference.metrics().maximum_depth < 8, "the comparison must stay clear of dice");

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

int main() {
    try {
        test_two_player_matches_mcts_search();
        test_finds_the_saving_move();
        test_max_n_backup();
        test_determinized_search_ignores_hidden_truth();
        test_parallel_and_batched_runs_agree();
        test_edge_cases();
        test_determinized_roots_offer_the_real_options();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "search pool tests passed\n";
    return 0;
}
