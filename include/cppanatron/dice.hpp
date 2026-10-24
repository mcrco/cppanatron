#pragma once

#include <array>
#include <cstdint>
#include <utility>

#include "cppanatron/action.hpp"
#include "cppanatron/random.hpp"

namespace cppanatron {

/**
 * OpenHex's balanced dice (DiceMode::balanced, its 1v1 rules): the state
 * between two rolls. Everything in it is public, since every roll is.
 *
 * A deck of 36 cards, one per ordered pair of faces. A roll by `roller`
 * (the turn owner):
 *
 *  1. If at most 13 cards remain, the deck is first refilled to all 36. Only
 *     the deck resets: the window, the sevens and the streak carry over. A
 *     game starts with a full deck and no history (OpenHex refills when the
 *     main phase starts; no roll comes before).
 *  2. Total k in 2..12 weighs w_k = c_k / T * max(0, 1 - 0.34 a_k), with c_k
 *     of the T cards left of total k and a_k the times k is among the last
 *     five totals.
 *  3. With two or more seats, the 7's weight is multiplied by
 *     clamp(imbalance + streak, 0, 2): streak = 0.4 * streak_count, negated
 *     when the roller owns the streak; imbalance = 1 + (1/n - s/S) / (1/n)
 *     once the 7s rolled so far S >= n seats (s the roller's), else 1.
 *  4. The total is k with probability w_k / sum(w), and the card is uniform
 *     among that total's cards left. (All-zero weights would draw a card
 *     uniformly; at least 14 cards span three totals, which rules it out.)
 *  5. The card leaves the deck, the total joins the window (dropping the
 *     oldest of six), a 7 counts for the roller and extends its streak or
 *     starts a new one of 1, and any other total ends the streak.
 *
 * A card's chance depends only on its total, so the cards per total, the
 * window, the sevens and the streak decide every future total; the faces
 * only name the card. The weights are kept as integers (0.34 = 17/50,
 * 0.4 = 2/5), so the probabilities are exact up to one final rounding.
 */
struct BalancedDice {
    static constexpr int kMaxSeats = 6;
    static constexpr int kWindow = 5;
    /** A draw first refills the deck when at most this many cards remain. */
    static constexpr int kReshuffleAt = 13;
    static constexpr std::uint64_t kFullDeck = (std::uint64_t{1} << 36U) - 1U;

    /** The cards left: bit (first - 1) * 6 + (second - 1) per card. */
    std::uint64_t deck{kFullDeck};
    /** The last totals, oldest first, in the first `recent_count` entries (the rest 0). */
    std::array<std::int8_t, kWindow> recent{};
    int recent_count{};
    /** The seats (n of the imbalance term); `sevens` beyond them stay 0. */
    int seats{2};
    /** 7s rolled per seat. */
    std::array<int, kMaxSeats> sevens{};
    /** The seat owning the current run of 7s, -1 for none, and its length (0 for none). */
    int streak_player{-1};
    int streak_count{};

    auto operator<=>(const BalancedDice&) const = default;

    /** The state at the start of the main phase: a full deck and no history. */
    [[nodiscard]] static BalancedDice fresh(int seats);
    /** Whether both faces are in [1, 6]. */
    [[nodiscard]] static bool valid_faces(Dice dice) noexcept;
    /** The deck bit of a card; throws std::invalid_argument for faces outside [1, 6]. */
    [[nodiscard]] static int card_bit(Dice dice);
    [[nodiscard]] static Dice card_of_bit(int bit) noexcept;

    /** Throws std::invalid_argument for a state no game reaches (window, sevens, streak, seats). */
    void validate() const;

    /** Cards left in the deck (before any refill). */
    [[nodiscard]] int remaining() const noexcept;
    /** The deck the next draw takes from: the full deck when it is refilled first. */
    [[nodiscard]] std::uint64_t drawable() const noexcept;
    /** Cards per total 2..12 in drawable(). */
    [[nodiscard]] std::array<int, 11> counts() const noexcept;
    /** Whether the next draw can be this card. */
    [[nodiscard]] bool can_draw(Dice dice) const noexcept;
    /**
     * The factor on the 7's weight when `roller` rolls, as an exact fraction
     * (numerator, denominator); 1 when no seven balancing applies (roller -1
     * or a single seat). Throws std::invalid_argument for a roller that is
     * not a seat.
     */
    [[nodiscard]] std::pair<std::int64_t, std::int64_t> seven_multiplier(int roller) const;
    /** Integer weights of totals 2..12, proportional to w_k (steps 2 and 3). */
    [[nodiscard]] std::array<std::int64_t, 11> sum_weights(int roller) const;
    /** P(next total = 2..12) when `roller` rolls (-1: no seven balancing). */
    [[nodiscard]] std::array<double, 11> sum_probabilities(int roller) const;
    /**
     * The first card of `total` left to draw (faces ascending): the card a
     * chance outcome coded by its total stands for. Any card of the total
     * gives the same future, since only totals matter. Throws
     * std::invalid_argument when no card of `total` is left.
     */
    [[nodiscard]] Dice card_for_total(int total) const;
    /**
     * A card for `roller` from `rng` without changing the state: the total by
     * weight (one uniform draw), then one of its cards uniformly.
     */
    [[nodiscard]] Dice draw(random::Engine& rng, int roller) const;
    /**
     * The state after `roller` drew `dice` (steps 1 and 5). Throws
     * std::invalid_argument, changing nothing, for a card the deck no longer
     * holds or a roller that is not a seat.
     */
    void apply(Dice dice, int roller);
};

}  // namespace cppanatron
