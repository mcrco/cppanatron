#include "cppanatron/dice.hpp"

#include <algorithm>
#include <bit>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>

namespace cppanatron {
namespace {

constexpr int kSevenIndex = 5;

/** The deck bits of each total 2..12. */
constexpr std::array<std::uint64_t, 11> kTotalMasks = [] {
    std::array<std::uint64_t, 11> masks{};
    for (int first = 1; first <= 6; ++first) {
        for (int second = 1; second <= 6; ++second) {
            masks[static_cast<std::size_t>(first + second - 2)] |=
                std::uint64_t{1} << static_cast<unsigned>((first - 1) * 6 + (second - 1));
        }
    }
    return masks;
}();

/** 1 - 0.34 a = (50 - 17 a) / 50, at least 0: the numerator. */
std::int64_t recency_numerator(int appearances) {
    return std::max<std::int64_t>(0, 50 - 17 * static_cast<std::int64_t>(appearances));
}

std::string card_name(Dice dice) {
    return "(" + std::to_string(dice.first) + ", " + std::to_string(dice.second) + ")";
}

}  // namespace

BalancedDice BalancedDice::fresh(int seats) {
    BalancedDice dice;
    dice.seats = seats;
    dice.validate();
    return dice;
}

bool BalancedDice::valid_faces(Dice dice) noexcept {
    return dice.first >= 1 && dice.first <= 6 && dice.second >= 1 && dice.second <= 6;
}

int BalancedDice::card_bit(Dice dice) {
    if (!valid_faces(dice)) {
        throw std::invalid_argument("dice faces must be in [1, 6], got " + card_name(dice));
    }
    return (dice.first - 1) * 6 + (dice.second - 1);
}

Dice BalancedDice::card_of_bit(int bit) noexcept {
    return Dice{bit / 6 + 1, bit % 6 + 1};
}

void BalancedDice::validate() const {
    if (seats < 1 || seats > kMaxSeats) {
        throw std::invalid_argument("balanced dice need one to six seats");
    }
    if ((deck & ~kFullDeck) != 0) {
        throw std::invalid_argument("balanced dice deck holds bits beyond its 36 cards");
    }
    if (recent_count < 0 || recent_count > kWindow) {
        throw std::invalid_argument("balanced dice remember at most five totals");
    }
    for (int i = 0; i < kWindow; ++i) {
        const int total = recent[static_cast<std::size_t>(i)];
        if (i < recent_count ? (total < 2 || total > 12) : total != 0) {
            throw std::invalid_argument("balanced dice recent totals must be in [2, 12]");
        }
    }
    for (int seat = 0; seat < kMaxSeats; ++seat) {
        const int count = sevens[static_cast<std::size_t>(seat)];
        if (count < 0 || (seat >= seats && count != 0)) {
            throw std::invalid_argument("balanced dice need a non-negative sevens count per seat");
        }
    }
    if ((streak_player < 0) != (streak_count == 0) || streak_count < 0 || streak_player < -1 ||
        streak_player >= seats) {
        throw std::invalid_argument(
            "a balanced dice streak has a seat for an owner exactly when its count is positive");
    }
}

int BalancedDice::remaining() const noexcept {
    return std::popcount(deck);
}

std::uint64_t BalancedDice::drawable() const noexcept {
    return remaining() <= kReshuffleAt ? kFullDeck : deck;
}

std::array<int, 11> BalancedDice::counts() const noexcept {
    const std::uint64_t cards = drawable();
    std::array<int, 11> result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = std::popcount(cards & kTotalMasks[i]);
    }
    return result;
}

bool BalancedDice::can_draw(Dice dice) const noexcept {
    return valid_faces(dice) &&
           (drawable() >> static_cast<unsigned>((dice.first - 1) * 6 + (dice.second - 1)) & 1U) != 0;
}

std::pair<std::int64_t, std::int64_t> BalancedDice::seven_multiplier(int roller) const {
    if (roller < -1 || roller >= seats) {
        throw std::invalid_argument("the roller is not a seat");
    }
    if (roller < 0 || seats < 2) {
        return {1, 1};
    }
    // streak = 0.4 * count * direction = 2 * signed_streak / 5.
    std::int64_t signed_streak = 0;
    if (streak_count > 0 && streak_player >= 0) {
        signed_streak = streak_player == roller ? -streak_count : streak_count;
    }
    const std::int64_t total = std::accumulate(sevens.begin(), sevens.end(), std::int64_t{0});
    std::int64_t numerator = 0;
    std::int64_t denominator = 0;
    if (total >= seats) {
        // 1 + (1/n - s/S) / (1/n) = (2S - n s) / S, plus 2 streak / 5: over 5S.
        const std::int64_t own = sevens[static_cast<std::size_t>(roller)];
        denominator = 5 * total;
        numerator = 5 * (2 * total - seats * own) + 2 * signed_streak * total;
    } else {
        denominator = 5;
        numerator = 5 + 2 * signed_streak;
    }
    return {std::clamp<std::int64_t>(numerator, 0, 2 * denominator), denominator};
}

std::array<std::int64_t, 11> BalancedDice::sum_weights(int roller) const {
    const auto [multiplier, denominator] = seven_multiplier(roller);
    const std::array<int, 11> cards = counts();
    std::array<int, 11> appearances{};
    for (int i = 0; i < recent_count; ++i) {
        ++appearances[static_cast<std::size_t>(recent[static_cast<std::size_t>(i)] - 2)];
    }
    // w_k * 50 * T * denominator: c_k * (50 - 17 a_k)+ * (denominator, or the 7's multiplier).
    std::array<std::int64_t, 11> weights{};
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] = cards[i] * recency_numerator(appearances[i]) *
                     (i == kSevenIndex ? multiplier : denominator);
    }
    if (std::all_of(weights.begin(), weights.end(), [](std::int64_t w) { return w == 0; })) {
        std::copy(cards.begin(), cards.end(), weights.begin());
    }
    return weights;
}

std::array<double, 11> BalancedDice::sum_probabilities(int roller) const {
    const std::array<std::int64_t, 11> weights = sum_weights(roller);
    const std::int64_t total = std::accumulate(weights.begin(), weights.end(), std::int64_t{0});
    std::array<double, 11> result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<double>(weights[i]) / static_cast<double>(total);
    }
    return result;
}

Dice BalancedDice::card_for_total(int total) const {
    if (total < 2 || total > 12) {
        throw std::invalid_argument("dice totals are in [2, 12]");
    }
    const std::uint64_t cards = drawable() & kTotalMasks[static_cast<std::size_t>(total - 2)];
    if (cards == 0) {
        throw std::invalid_argument(
            "the balanced deck has no card of total " + std::to_string(total) + " left");
    }
    return card_of_bit(std::countr_zero(cards));
}

Dice BalancedDice::draw(random::Engine& rng, int roller) const {
    const std::array<std::int64_t, 11> weights = sum_weights(roller);
    std::array<double, 11> real{};
    std::transform(weights.begin(), weights.end(), real.begin(),
                   [](std::int64_t w) { return static_cast<double>(w); });
    const std::size_t index = random::discrete(rng, std::span<const double>(real));
    std::uint64_t cards = drawable() & kTotalMasks[index];
    for (std::size_t skip = random::index(rng, static_cast<std::size_t>(std::popcount(cards)));
         skip > 0; --skip) {
        cards &= cards - 1;
    }
    return card_of_bit(std::countr_zero(cards));
}

void BalancedDice::apply(Dice dice, int roller) {
    const int bit = card_bit(dice);
    if (roller < -1 || roller >= seats) {
        throw std::invalid_argument("the roller is not a seat");
    }
    const std::uint64_t cards = drawable();
    if ((cards >> static_cast<unsigned>(bit) & 1U) == 0) {
        throw std::invalid_argument(
            "dice " + card_name(dice) + " are not in the balanced deck: that card was drawn since "
            "the last reshuffle (" + std::to_string(std::popcount(cards)) + " cards left)");
    }
    deck = cards & ~(std::uint64_t{1} << static_cast<unsigned>(bit));
    const int total = dice.first + dice.second;
    if (recent_count == kWindow) {
        std::rotate(recent.begin(), recent.begin() + 1, recent.end());
        recent[kWindow - 1] = static_cast<std::int8_t>(total);
    } else {
        recent[static_cast<std::size_t>(recent_count++)] = static_cast<std::int8_t>(total);
    }
    if (roller < 0) {
        return;
    }
    if (total == 7) {
        ++sevens[static_cast<std::size_t>(roller)];
        if (streak_player == roller) {
            ++streak_count;
        } else {
            streak_player = roller;
            streak_count = 1;
        }
    } else {
        streak_player = -1;
        streak_count = 0;
    }
}

}  // namespace cppanatron
