// Throughput and memory of SearchPool (and MCTSSearch for reference) with a
// trivial evaluator: uniform logits, every value 0.5.
//
//   cppanatron_search_bench MODE PLAYERS [SEARCHES] [SIMULATIONS] [THREADS]
//
// The pool runs its searches twice and reports the second round's speed,
// when the allocator reuses the first round's memory as in steady use.
// MODE is pool, mcts or game (the memory of one game copy). Memory is the
// growth of resident memory while every tree is alive, so run one mode per
// process.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include "cppanatron/flat_codec.hpp"
#include "cppanatron/mcts.hpp"
#include "cppanatron/search_pool.hpp"

namespace {

using namespace cppanatron;

class NullEncoder final : public ObservationEncoder {
public:
    [[nodiscard]] std::size_t size() const noexcept override { return 1; }
    void encode(const Game& game, int observer, std::span<float> out) const override {
        out[0] = static_cast<float>(observer + game.num_turns());
    }
};

double resident_bytes() {
    std::ifstream statm("/proc/self/statm");
    long pages = 0;
    long resident = 0;
    statm >> pages >> resident;
    return static_cast<double>(resident) * static_cast<double>(sysconf(_SC_PAGESIZE));
}

Game position(int players) {
    Game game({kColors.begin(), kColors.begin() + players}, MapType::base, 2024);
    std::mt19937_64 rng(7);
    int steps = 0;
    while (game.is_initial_build_phase() || steps < 60 || game.playable_actions().size() < 3) {
        const auto& actions = game.playable_actions();
        game.execute(actions[rng() % actions.size()]);
        ++steps;
        if (game.winning_color().has_value()) {
            throw std::runtime_error("the random game ended");
        }
    }
    return game;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " pool|mcts|game PLAYERS [SEARCHES] [SIMS] [THREADS]\n";
        return 2;
    }
    const std::string mode = argv[1];
    const int players = std::atoi(argv[2]);
    const int searches = argc > 3 ? std::atoi(argv[3]) : 64;
    const int simulations = argc > 4 ? std::atoi(argv[4]) : 256;
    const int threads = argc > 5 ? std::atoi(argv[5]) : 1;
    const Game game = position(players);
    const auto codec = std::make_shared<FlatActionCodec>(players, MapType::base);
    const double before = resident_bytes();
    const auto start = std::chrono::steady_clock::now();
    std::uint64_t total_simulations = 0;
    std::uint64_t nodes = 0;
    std::uint64_t games = 0;
    std::uint64_t tree_bytes = 0;

    if (mode == "game") {
        std::vector<Game> copies(static_cast<std::size_t>(searches), game);
        std::cout << "game copy: " << (resident_bytes() - before) / searches << " bytes (sizeof "
                  << sizeof(Game) << ")\n";
        return 0;
    }

    std::vector<float> logits(codec->size(), 0.0F);
    if (mode == "pool") {
        SearchConfig config;
        config.simulations = simulations;
        config.max_players = players;
        config.threads = threads;
        SearchPool pool(codec, std::make_shared<NullEncoder>(), config);
        const std::size_t batch = static_cast<std::size_t>(searches);
        std::vector<float> observations(batch * pool.observation_size());
        std::vector<std::int64_t> rows(batch);
        std::vector<float> all_logits(batch * codec->size(), 0.0F);
        std::vector<float> values(batch * static_cast<std::size_t>(players), 0.5F);
        double grown = 0.0;
        double seconds = 0.0;
        for (int round = 0; round < 2; ++round) {
            const auto round_start = std::chrono::steady_clock::now();
            std::vector<std::int64_t> ids;
            for (int i = 0; i < searches; ++i) {
                ids.push_back(pool.add(game, static_cast<std::uint64_t>(i)));
            }
            while (const std::size_t n = pool.collect(observations, rows)) {
                pool.apply(
                    std::span<const float>(all_logits).first(n * codec->size()),
                    std::span<const float>(values).first(n * static_cast<std::size_t>(players)));
            }
            seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - round_start).count();
            if (round == 0) {
                grown = resident_bytes() - before;
            }
            total_simulations = nodes = games = tree_bytes = 0;
            for (const auto id : ids) {
                const auto stats = pool.stats(id);
                total_simulations += stats.simulations;
                nodes += stats.nodes;
                games += stats.games;
                tree_bytes += stats.tree_bytes;
                pool.remove(id);
            }
        }
        std::cout << "pool " << players << "p threads=" << threads << ": "
                  << total_simulations / seconds << " sims/s, " << grown / searches / 1e6
                  << " MB/tree, " << grown / static_cast<double>(nodes) << " B/node (" << nodes / searches
                  << " nodes, " << games / searches << " games per tree; tree structure "
                  << tree_bytes / static_cast<double>(nodes) << " B/node)\n";
        return 0;
    }

    if (mode == "mcts") {
        std::vector<std::unique_ptr<MCTSSearch>> trees;
        for (int i = 0; i < searches; ++i) {
            auto search = std::make_unique<MCTSSearch>(
                game, FlatActionSpace(players, MapType::base), 1.5, static_cast<std::uint64_t>(i));
            search->initialize_root(logits);
            for (int s = 0; s < simulations; ++s) {
                if (search->select_leaf() != nullptr) {
                    search->evaluate_leaf(logits, 0.0);
                }
            }
            total_simulations += search->metrics().simulations;
            trees.push_back(std::move(search));
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const double grown = resident_bytes() - before;
        std::cout << "MCTSSearch " << players << "p: " << total_simulations / seconds << " sims/s, "
                  << grown / searches / 1e6 << " MB/tree\n";
        return 0;
    }
    std::cerr << "unknown mode " << mode << '\n';
    return 2;
}
