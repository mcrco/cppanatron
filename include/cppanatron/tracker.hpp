#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <random>
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

    auto operator<=>(const TrackerEvent&) const = default;
};

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
 * starting deck minus the observer's hand and every played card.
 *
 * The tracker is sound: the true hands always satisfy its constraints
 * (consistent_with). It does not use the weaker evidence of what seats chose
 * not to do, and after a hidden steal it forgets which known card might have
 * moved (see steal), so it keeps a superset of the worlds a perfect counter
 * would keep.
 */
class HandTracker {
public:
    /**
     * Start tracking `game` for seat `observer`. Reads public state (hand
     * sizes, bank, development-card counts, played cards, deck size) and the
     * observer's own hand only: other seats' cards start unknown, which is
     * exact for a fresh game.
     */
    HandTracker(const Game& game, int observer);

    [[nodiscard]] int observer() const noexcept { return observer_; }
    [[nodiscard]] int num_players() const noexcept { return static_cast<int>(sizes_.size()); }

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

    void gain(int player, const std::array<int, 5>& cards);
    void lose(int player, const std::array<int, 5>& cards);
    void transfer(int from, int to, const std::array<int, 5>& cards);
    /**
     * A robber steal. Unseen steals (`resource` unset; only possible between
     * two other seats) move an unknown card: the thief gains an unknown card
     * and every known lower bound of the victim drops by one, since any of
     * those kinds may have been taken. When only one kind is possible the
     * steal is resolved as a known transfer instead.
     */
    void steal(int thief, int victim, std::optional<Resource> resource);
    void monopoly(int player, Resource resource, const std::vector<int>& amounts);
    void buy_development_card(int player, std::optional<DevelopmentCard> card);
    void play_development_card(int player, DevelopmentCard card);
    void end_turn(int player);

    [[nodiscard]] int hand_size(int seat) const;
    /** Lower bound per resource; the exact hand for the observer. */
    [[nodiscard]] const std::array<int, 5>& known_resources(int seat) const;
    [[nodiscard]] int unknown_count(int seat) const;
    /** Cards per resource in all hands together (from the bank). */
    [[nodiscard]] const std::array<int, 5>& resource_totals() const noexcept { return totals_; }
    /** Cards per resource in the unknown parts of all hands. */
    [[nodiscard]] std::array<int, 5> unknown_pool() const;

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

    /**
     * Expected hand: known cards plus the unknown cards' share of the
     * unknown pool (the mean of the deal sample_world draws from).
     */
    [[nodiscard]] std::array<double, 5> expected_resources(int seat) const;
    /**
     * Expected unplayed development cards: the seat's share of the unseen
     * cards. Ignores the cap on victory point cards that sample_world applies.
     */
    [[nodiscard]] std::array<double, 5> expected_development_cards(int seat) const;

    /**
     * Whether `truth` satisfies every constraint the tracker keeps. Reads the
     * hidden state on purpose; for tests and debugging.
     */
    [[nodiscard]] bool consistent_with(const Game& truth) const;
    /**
     * Throws std::invalid_argument when the public state of `game` (hand
     * sizes, bank, development-card counts, deck, played cards, the
     * observer's own cards) disagrees with the tracker.
     */
    void check_public(const Game& game) const;

    bool operator==(const HandTracker&) const = default;

private:
    void check_seat(int seat) const;
    void take(int seat, const std::array<int, 5>& cards);
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
};

/**
 * A full game consistent with everything `tracker`'s observer knows.
 *
 * `game` may be the true game or any game with the same public state (for
 * example one rebuilt from the observer's view); its hidden cards are never
 * read. Every other seat's resource hand is its known cards plus a uniformly
 * random deal of the unknown pool to the unknown slots; their development
 * cards are a uniformly random deal of the unseen cards, conditioned on no
 * seat holding enough victory point cards to have won already; the rest is
 * the deck, reshuffled, and the game's random stream is reseeded from `rng`.
 *
 * The deal is the exact posterior if every unknown card were equally likely
 * to be anywhere, which is what the tracker records. It ignores evidence the
 * tracker drops (which card a hidden steal likely took, what seats did not
 * do), so it is close to, not exactly, uniform over consistent worlds. When
 * the victory point condition rejects 1000 deals in a row, the victory point
 * cards go to the deck first instead (still consistent, no longer uniform).
 */
[[nodiscard]] Game sample_world(const Game& game, const HandTracker& tracker, std::mt19937_64& rng);

}  // namespace cppanatron
