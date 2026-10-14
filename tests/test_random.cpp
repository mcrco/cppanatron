// Golden values of the portable samplers and of a seeded game. They must not
// depend on the compiler or the standard library; the same game is pinned in
// tests/python/test_golden_random.py against the Python extension's build.
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cppanatron/game.hpp"
#include "cppanatron/random.hpp"

namespace {

using cppanatron::Color;
using cppanatron::Game;
using cppanatron::GameRules;
using cppanatron::MapType;
using cppanatron::NumberPlacement;
namespace random = cppanatron::random;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T, typename Draw>
std::vector<T> draws(std::uint64_t seed, int count, Draw draw) {
    random::Engine rng(seed);
    std::vector<T> result;
    for (int i = 0; i < count; ++i) {
        result.push_back(draw(rng));
    }
    return result;
}

void require_close(const std::vector<double>& actual, const std::vector<double>& expected,
                   const std::string& what) {
    // Exact but for std::log/std::pow, which may differ in the last bit.
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(std::abs(actual[i] - expected[i]) <= 1e-12 * std::abs(expected[i]), what);
    }
}

void test_integer_samplers() {
    require(draws<std::uint64_t>(42, 12, [](auto& rng) { return random::below(rng, 6); }) ==
                std::vector<std::uint64_t>{4, 3, 4, 0, 5, 0, 3, 2, 1, 2, 0, 3},
            "below(6)");
    require(draws<std::uint64_t>(42, 4, [](auto& rng) { return random::below(rng, 1000000007); }) ==
                std::vector<std::uint64_t>{755155538, 639031398, 752145206, 136272684},
            "below(1e9 + 7)");
    random::Engine rng(7);
    std::vector<int> values{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    random::shuffle(values.begin(), values.end(), rng);
    require(values == std::vector<int>{1, 4, 2, 3, 5, 9, 6, 0, 8, 7}, "shuffle");
    const std::vector<double> weights{1, 2, 3, 0, 4};
    require(draws<std::size_t>(3, 12, [&](auto& r) { return random::discrete(r, weights); }) ==
                std::vector<std::size_t>{2, 1, 2, 2, 2, 2, 4, 2, 4, 1, 1, 2},
            "discrete");
    require(draws<int>(9, 2000, [](auto& r) { return random::between(r, 1, 6); }) ==
                draws<int>(9, 2000, [](auto& r) { return 1 + static_cast<int>(random::below(r, 6)); }),
            "between is below shifted");
}

void test_real_samplers() {
    require(draws<double>(5, 3, [](auto& rng) { return random::uniform(rng); }) ==
                std::vector<double>{0x1.589bf68d490b3p-1, 0x1.3b590ccfb827p-5, 0x1.cd6416420861cp-3},
            "uniform");
    require_close(draws<double>(11, 3, [](auto& rng) { return random::normal(rng); }),
                  {-0.59251904807246847, -0.91430439485300219, -0.2376461700585483}, "normal");
    require_close(draws<double>(13, 3, [](auto& rng) { return random::gamma(rng, 0.3); }),
                  {0.17652341706333102, 0.01701698941084779, 0.31284914287632526}, "gamma(0.3)");
    require_close(draws<double>(13, 3, [](auto& rng) { return random::gamma(rng, 2.5); }),
                  {5.6823013892092993, 1.0982414394410571, 2.3430123380747467}, "gamma(2.5)");
    require_close(draws<double>(17, 3, [](auto& rng) { return random::gumbel(rng); }),
                  {1.0192676004451744, -1.295421926138604, 1.037747219316181}, "gumbel");
}

void test_seeded_game() {
    Game game({Color::red, Color::blue, Color::white, Color::orange}, MapType::base, 2024,
              GameRules{}, NumberPlacement::random);
    std::string terrain;
    for (int tile_index : game.board().map().land_tile_indices()) {
        const auto& tile = game.board().map().tiles()[static_cast<std::size_t>(tile_index)];
        terrain += tile.resource.has_value() ? std::string(cppanatron::to_string(*tile.resource)).substr(0, 2)
                                             : std::string("--");
    }
    require(terrain == "WHWHBRBRWOWOWHBRWOORSHOR--SHORSHWHSHWO", "seeded terrain");
    int steps = 0;
    while (!game.winning_color().has_value()) {
        const auto& actions = game.playable_actions();
        game.execute(actions[static_cast<std::size_t>(steps) * 7919 % actions.size()]);
        ++steps;
    }
    require(steps == 710 && game.num_turns() == 245 && game.winning_color() == Color::orange,
            "seeded game length and winner");
    require(game.resource_bank() == std::array<int, 5>{14, 18, 11, 13, 17}, "seeded bank");
    require(game.development_cards_remaining() == 13, "seeded deck");
    require(game.players()[3].actual_victory_points == 10 &&
                game.players()[0].resources == std::array<int, 5>{1, 0, 1, 4, 0},
            "seeded players");
}

}  // namespace

int main() {
    try {
        test_integer_samplers();
        test_real_samplers();
        test_seeded_game();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "cppanatron random golden tests passed\n";
    return 0;
}
