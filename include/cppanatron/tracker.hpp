#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include "cppanatron/game.hpp"

namespace cppanatron {

enum class TrackerEventType : std::uint8_t {
    /** `player` takes `cards` from the bank (production, year of plenty, trades). */
    gain,
    /** `player` returns `cards` to the bank (building, discarding, trades). */
    lose,
    /** `cards` move from `other` to `player` in the open (domestic trade). */
    transfer,
    /**
     * The robber: one card moves from `other` (the victim) to `player` (the
     * thief). `resource` is set only when the observer is thief or victim.
     */
    steal,
    /** `player` took every `resource`; `amounts[seat]` cards from each seat. */
    monopoly,
    /** `player` bought a card: `development_card` is set only for the observer. */
    buy_development_card,
    /** `player` played `development_card`, which everyone sees. */
    play_development_card,
    /** `player` ended its turn: cards it bought become playable. */
    end_turn,
    /**
     * The game goes on with public victory points `amounts[seat]`, so no seat
     * holds enough victory point cards to have won yet.
     */
    public_victory_points,
    /**
     * `player` holds at least `amounts[0]` and at most `amounts[1]` victory
     * point cards (for example, learned from friendly-robber options).
     */
    victory_point_cards,
};

/**
 * One public step of a game, as far as one observer sees it.
 *
 * Seats are indices into Game::colors(). Events carry no payment: buying a
 * card is `lose` (the cost) plus `buy_development_card`, as
 * HandTracker::events derives them.
 */
struct TrackerEvent {
    TrackerEventType type{};
    int player{-1};
    int other{-1};
    std::array<int, 5> cards{};
    std::optional<Resource> resource;
    std::optional<DevelopmentCard> development_card;
    std::vector<int> amounts;

    [[nodiscard]] static TrackerEvent gain(int player, const std::array<int, 5>& cards);
    [[nodiscard]] static TrackerEvent lose(int player, const std::array<int, 5>& cards);
    [[nodiscard]] static TrackerEvent transfer(int from, int to, const std::array<int, 5>& cards);
    [[nodiscard]] static TrackerEvent steal(int thief, int victim, std::optional<Resource> resource);
    [[nodiscard]] static TrackerEvent monopoly(int player, Resource resource, std::vector<int> amounts);
    [[nodiscard]] static TrackerEvent buy_development_card(
        int player, std::optional<DevelopmentCard> card);
    [[nodiscard]] static TrackerEvent play_development_card(int player, DevelopmentCard card);
    [[nodiscard]] static TrackerEvent end_turn(int player);
    [[nodiscard]] static TrackerEvent public_victory_points(std::vector<int> points);
    [[nodiscard]] static TrackerEvent victory_point_cards(int player, int at_least, int at_most);

    auto operator<=>(const TrackerEvent&) const = default;
};

/** How a HandTracker represents the resource cards its observer cannot see. */
enum class TrackerMode : std::uint8_t {
    /**
     * Per seat a known lower bound per resource and a count of unknown cards;
     * one pool of unknown cards, dealt uniformly by sample_world.
     */
    pool,
    /**
     * The pool, plus the exact posterior over every seat's hand: a weighted
     * set of hypotheses (one hypothesis = the exact hand of every seat),
     * branched at hidden steals and pruned by every later event. Falls back
     * to the pool when the hypotheses outgrow a cap (see HandTracker).
     */
    exact,
};

/** The exact mode's hypotheses; an implementation detail of tracker.cpp. */
struct HandBelief;

/** Hypotheses the exact mode keeps at most before it falls back to the pool. */
inline constexpr int kDefaultMaxHypotheses = 4096;

/**
 * Follows a game from one seat, using only what that seat sees.
 *
 * Per seat it keeps the resource hand size, a known lower bound per resource
 * (exact for the observer) and the number of unknown cards beyond those
 * bounds. The cards in all hands are known per resource from the public bank,
 * so the unknown cards of all seats together form a known multiset, the
 * unknown pool. Cards become unknown only when the observer cannot see them:
 * when it starts tracking a game in progress, and when the robber moves a card
 * between two other seats. Revealed cards (discards, payments, monopolies,
 * trades) are taken from the known bounds first and then from the unknown
 * cards, which narrows the pool for everyone.
 *
 * Development cards: the count each seat holds and every played card are
 * public; the unseen cards (the deck plus the other seats' hands) are the
 * starting deck minus the observer's hand and every played card. Victory
 * point cards are never played, so a seat's victory point cards are capped by
 * the fewest it could have held at any earlier point without having won,
 * plus the cards it bought since.
 *
 * The tracker is sound: the true hands always satisfy its constraints
 * (consistent_with). It does not use the weaker evidence of what seats chose
 * not to do. In TrackerMode::pool it also forgets, after a hidden steal, which
 * known card might have moved (see steal) and which card a steal most likely
 * took, so it keeps a superset of the worlds a perfect counter would keep,
 * and sample_world's uniform deal of the pool is not the exact posterior.
 *
 * TrackerMode::exact keeps that posterior as well: a weighted set of
 * hypotheses, each the exact hand of every seat. A fresh game starts from the
 * one known world; a game in progress from every deal of the pool, weighted
 * as a uniform deal, when there are at most max_hypotheses of them. A hidden
 * steal branches every hypothesis over the kinds the victim holds under it,
 * each weighted by its count (the robber takes a uniformly random card), and
 * a steal the observer saw weights hypotheses the same way. Every later event
 * prunes the hypotheses it contradicts (payments, discards, trades, the
 * amounts a monopoly took) and identical hands merge. The posterior is exact
 * under the rules' chance (uniform steals) with no model of what seats choose
 * to do, which is what a perfect counter can know. When the hypotheses
 * outgrow max_hypotheses the belief is dropped (belief_overflows counts it):
 * sample_world deals the pool as in pool mode until every hand is known
 * again (the pool holds one kind at most), when the belief restarts from
 * that one world. The pool bounds are kept in both modes, so an exact
 * tracker can always deal the pool too.
 */
class HandTracker {
public:
    /**
     * Start tracking `game` for seat `observer`. Reads public state (hand
     * sizes, bank, development-card counts, played cards, deck size) and the
     * observer's own hand only: other seats' cards start unknown, which is
     * exact for a fresh game.
     */
    HandTracker(
        const Game& game, int observer, TrackerMode mode = TrackerMode::pool,
        int max_hypotheses = kDefaultMaxHypotheses);

    [[nodiscard]] int observer() const noexcept { return observer_; }
    [[nodiscard]] int num_players() const noexcept { return static_cast<int>(sizes_.size()); }
    [[nodiscard]] TrackerMode mode() const noexcept { return mode_; }
    [[nodiscard]] int max_hypotheses() const noexcept { return max_hypotheses_; }

    void apply(const TrackerEvent& event);
    /** Derive the events of one transition and apply them. */
    void observe(const Game& before, const Action& action, const Game& after);
    /**
     * The events `observer` sees when `action` takes `before` to `after`.
     * Reads the true states but reveals only public changes, plus the
     * observer's own stolen or bought card.
     */
    [[nodiscard]] static std::vector<TrackerEvent> events(
        const Game& before, const Action& action, const Game& after, int observer);
    /**
     * What the observer's own legal actions in `game` (where it is to move)
     * reveal. With the friendly robber, Catanatron protects tiles of seats
     * under 3 actual points, so the robber options bound other seats'
     * victory point cards. `legal` is the observer's legal action list (a
     * view's game may not reproduce it); `game` supplies the public board.
     */
    [[nodiscard]] static std::vector<TrackerEvent> option_events(
        const Game& game, const std::vector<Action>& legal, int observer);

    void gain(int player, const std::array<int, 5>& cards);
    void lose(int player, const std::array<int, 5>& cards);
    void transfer(int from, int to, const std::array<int, 5>& cards);
    /**
     * A robber steal. Unseen steals (`resource` unset; only possible between
     * two other seats) move an unknown card: the thief gains an unknown card
     * and every known lower bound of the victim drops by one, since any of
     * those kinds may have been taken. When only one kind is possible the
     * steal is resolved as a known transfer instead. In exact mode the
     * hypotheses branch over the kinds instead (see the class).
     */
    void steal(int thief, int victim, std::optional<Resource> resource);
    void monopoly(int player, Resource resource, const std::vector<int>& amounts);
    void buy_development_card(int player, std::optional<DevelopmentCard> card);
    void play_development_card(int player, DevelopmentCard card);
    void end_turn(int player);
    /** The game goes on at these public victory points (see public_victory_points). */
    void public_victory_points(const std::vector<int>& points);
    /** Apply option_events(game, legal, observer()). */
    void observe_options(const Game& game, const std::vector<Action>& legal);

    [[nodiscard]] int hand_size(int seat) const;
    /** Lower bound per resource; the exact hand for the observer. */
    [[nodiscard]] const std::array<int, 5>& known_resources(int seat) const;
    [[nodiscard]] int unknown_count(int seat) const;
    /** Cards per resource in all hands together (from the bank). */
    [[nodiscard]] const std::array<int, 5>& resource_totals() const noexcept { return totals_; }
    /** Cards per resource in the unknown parts of all hands. */
    [[nodiscard]] std::array<int, 5> unknown_pool() const;

    /**
     * Whether the exact belief currently enumerates every possible world:
     * false in pool mode, and in exact mode while the belief has fallen back
     * to the pool (it outgrew max_hypotheses, or started from a game in
     * progress with too many deals).
     */
    [[nodiscard]] bool exact() const noexcept { return belief_ != nullptr; }
    [[nodiscard]] std::size_t hypothesis_count() const noexcept;
    /** Hypothesis `index`: every seat's resource hand, in seat order. */
    [[nodiscard]] std::span<const std::array<int, 5>> hypothesis_hands(std::size_t index) const;
    /** Hypothesis `index`'s posterior probability. */
    [[nodiscard]] double hypothesis_weight(std::size_t index) const;
    /** Every hypothesis's posterior probability; they sum to 1. */
    [[nodiscard]] std::span<const double> hypothesis_weights() const noexcept;
    /** How often the exact belief outgrew max_hypotheses and fell back to the pool. */
    [[nodiscard]] int belief_overflows() const noexcept { return overflows_; }

    [[nodiscard]] int development_card_count(int seat) const;
    [[nodiscard]] const std::array<int, 5>& own_development_cards() const noexcept {
        return own_development_cards_;
    }
    [[nodiscard]] const std::array<int, 5>& played_development_cards(int seat) const;
    /** The deck plus every other seat's unplayed cards, in DevelopmentCard order. */
    [[nodiscard]] std::array<int, 5> unseen_development_cards() const;
    [[nodiscard]] int development_deck_size() const noexcept { return deck_size_; }
    /** Cards `seat` bought this turn (not playable before its next turn). */
    [[nodiscard]] int bought_this_turn(int seat) const;
    /** The most victory point cards `seat` can hold, from the game's history. */
    [[nodiscard]] int victory_point_card_cap(int seat) const;
    /** The fewest victory point cards `seat` can hold (they are never played). */
    [[nodiscard]] int victory_point_card_floor(int seat) const;

    /**
     * Expected hand: the mean of the deal sample_world draws from. The known
     * cards plus the unknown cards' share of the unknown pool, or, while the
     * exact belief holds, the weighted mean of its hypotheses.
     */
    [[nodiscard]] std::array<double, 5> expected_resources(int seat) const;
    /**
     * Expected unplayed development cards: the seat's share of the unseen
     * cards. Ignores the cap on victory point cards that sample_world applies.
     */
    [[nodiscard]] std::array<double, 5> expected_development_cards(int seat) const;

    /**
     * Whether `truth` satisfies every constraint the tracker keeps (while the
     * exact belief holds, whether it is one of the hypotheses). Reads the
     * hidden state on purpose; for tests and debugging.
     */
    [[nodiscard]] bool consistent_with(const Game& truth) const;
    /**
     * Throws std::invalid_argument when the public state of `game` (hand
     * sizes, bank, development-card counts, deck, played cards, the
     * observer's own cards) disagrees with the tracker.
     */
    void check_public(const Game& game) const;

    bool operator==(const HandTracker& other) const;

private:
    void check_seat(int seat) const;
    void take(int seat, const std::array<int, 5>& cards);
    void update_belief(const TrackerEvent& event);
    void validate() const;

    int observer_{};
    GameRules rules_;
    std::vector<int> sizes_;
    std::vector<std::array<int, 5>> known_;
    std::array<int, 5> totals_{};
    std::vector<int> development_counts_;
    std::vector<std::array<int, 5>> played_;
    std::array<int, 5> own_development_cards_{};
    int deck_size_{};
    std::vector<int> bought_this_turn_;
    /** Victory point cards each seat can hold at most (before capping by its card count). */
    std::vector<int> victory_point_caps_;
    std::vector<int> victory_point_floors_;
    TrackerMode mode_{TrackerMode::pool};
    int max_hypotheses_{kDefaultMaxHypotheses};
    /**
     * The exact mode's hypotheses; null in pool mode and while the belief has
     * fallen back to the pool. Never mutated: an event builds a new one, so
     * copies of a tracker (apply works on a copy) share it for free.
     */
    std::shared_ptr<const HandBelief> belief_;
    int overflows_{};
};

/**
 * A full game consistent with everything `tracker`'s observer knows.
 *
 * `game` may be the true game or any game with the same public state (for
 * example one rebuilt from the observer's view); its hidden cards are never
 * read. Every other seat's resource hand is drawn as `deal` says, by default
 * the tracker's own mode: TrackerMode::pool deals the unknown pool uniformly
 * to the unknown slots on top of the known cards; TrackerMode::exact (which
 * needs a tracker in exact mode) draws one hypothesis of the exact belief
 * with its posterior probability, and deals the pool only while the belief
 * has fallen back (HandTracker::exact is false). Their development cards are
 * a uniformly random deal of the unseen cards, conditioned on no seat holding
 * more victory point cards than its cap (so nobody could have won already,
 * now or earlier) or fewer than its floor; the rest is the deck, reshuffled,
 * and the game's random stream is reseeded from `rng`.
 *
 * The pool deal is the exact posterior if every unknown card were equally
 * likely to be anywhere, which is what the pool records. It ignores evidence
 * the pool drops (which card a hidden steal likely took, what later payments
 * proved about it), so it is close to, not exactly, the posterior over
 * consistent worlds; the exact deal is that posterior (see HandTracker). When
 * the victory point condition rejects 1000 deals in a row, the victory point
 * cards go to the deck first instead (still consistent, no longer uniform).
 *
 * Throws std::invalid_argument for a finished game: nothing is left to
 * decide, and its worlds would depend on whether the given game counts as
 * won, which it only records through hidden victory point cards.
 */
[[nodiscard]] Game sample_world(
    const Game& game, const HandTracker& tracker, std::mt19937_64& rng,
    std::optional<TrackerMode> deal = std::nullopt);

}  // namespace cppanatron
