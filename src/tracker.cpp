#include "cppanatron/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>

#include "cppanatron/random.hpp"

namespace cppanatron {
namespace {

constexpr auto kVictoryPoint = static_cast<std::size_t>(DevelopmentCard::victory_point);
constexpr int kVictoryPointDealAttempts = 1000;

int sum(const std::array<int, 5>& counts) {
    return std::accumulate(counts.begin(), counts.end(), 0);
}

bool any_negative(const std::array<int, 5>& counts) {
    return std::any_of(counts.begin(), counts.end(), [](int count) { return count < 0; });
}

std::array<int, 5> single(Resource resource, int count = 1) {
    std::array<int, 5> cards{};
    cards[static_cast<std::size_t>(resource)] = count;
    return cards;
}

int seat_of(const Game& game, Color color) {
    const auto& colors = game.colors();
    const auto it = std::find(colors.begin(), colors.end(), color);
    if (it == colors.end()) {
        throw std::invalid_argument("color is not in this game");
    }
    return static_cast<int>(std::distance(colors.begin(), it));
}

[[noreturn]] void inconsistent(const char* what) {
    throw std::logic_error(std::string("hand tracker is inconsistent: ") + what);
}

}  // namespace

/**
 * The exact mode's hypotheses: `hands[h * seats + seat]` is hypothesis h's
 * hand of `seat` (the observer's included, the same in every hypothesis),
 * `weights[h]` its posterior probability. The weights sum to 1.
 */
struct HandBelief {
    int seats{};
    std::vector<std::array<int, 5>> hands;
    std::vector<double> weights;

    [[nodiscard]] std::size_t size() const noexcept { return weights.size(); }
    [[nodiscard]] std::span<const std::array<int, 5>> hypothesis(std::size_t h) const {
        const auto n = static_cast<std::size_t>(seats);
        return {hands.data() + h * n, n};
    }
    bool operator==(const HandBelief&) const = default;
};

namespace {

using Hands = std::span<std::array<int, 5>>;

/** Scale the weights to sum to 1. No hypothesis left means the events contradict the belief. */
void normalize(HandBelief& belief) {
    if (belief.weights.empty()) {
        inconsistent("no hypothesis of the exact belief explains the event");
    }
    const double total = std::accumulate(belief.weights.begin(), belief.weights.end(), 0.0);
    if (!(total > 0.0) || !std::isfinite(total)) {
        inconsistent("the exact belief's weights are not positive and finite");
    }
    for (double& weight : belief.weights) {
        weight /= total;
    }
}

/**
 * The belief after `update(hands, weight)` on every hypothesis: it edits the
 * hands and the weight in place and returns whether the hypothesis survives.
 * Events other than hidden steals map distinct hands to distinct hands, so
 * nothing needs merging.
 */
template <typename Update>
std::shared_ptr<const HandBelief> transformed(const HandBelief& belief, Update update) {
    const auto n = static_cast<std::size_t>(belief.seats);
    auto next = std::make_shared<HandBelief>();
    next->seats = belief.seats;
    next->hands.reserve(belief.hands.size());
    next->weights.reserve(belief.weights.size());
    for (std::size_t h = 0; h < belief.size(); ++h) {
        const std::size_t start = next->hands.size();
        const auto hands = belief.hypothesis(h);
        next->hands.insert(next->hands.end(), hands.begin(), hands.end());
        double weight = belief.weights[h];
        if (update(Hands(next->hands.data() + start, n), weight)) {
            next->weights.push_back(weight);
        } else {
            next->hands.resize(start);
        }
    }
    normalize(*next);
    return next;
}

/** `candidates` with identical hands merged, their weights added. */
std::shared_ptr<const HandBelief> merged(const HandBelief& candidates) {
    const auto hands_of = [&](std::size_t h) { return candidates.hypothesis(h); };
    const auto less = [&](std::size_t a, std::size_t b) {
        const auto x = hands_of(a);
        const auto y = hands_of(b);
        return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
    };
    std::vector<std::size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), less);
    auto next = std::make_shared<HandBelief>();
    next->seats = candidates.seats;
    for (std::size_t i = 0; i < order.size(); ++i) {
        if (i > 0 && !less(order[i - 1], order[i])) {  // sorted, so not less either way: equal
            next->weights.back() += candidates.weights[order[i]];
            continue;
        }
        const auto hands = hands_of(order[i]);
        next->hands.insert(next->hands.end(), hands.begin(), hands.end());
        next->weights.push_back(candidates.weights[order[i]]);
    }
    normalize(*next);
    return next;
}

/**
 * A hidden steal: every hypothesis branches over the kinds the victim holds
 * under it, each weighted by its count, since the robber takes a uniformly
 * random card (the victim's hand size is public, so it divides out).
 */
std::shared_ptr<const HandBelief> branched(const HandBelief& belief, int thief, int victim) {
    const auto thief_index = static_cast<std::size_t>(thief);
    const auto victim_index = static_cast<std::size_t>(victim);
    HandBelief candidates;
    candidates.seats = belief.seats;
    for (std::size_t h = 0; h < belief.size(); ++h) {
        const auto hands = belief.hypothesis(h);
        const std::array<int, 5> hand = hands[victim_index];
        for (std::size_t r = 0; r < 5; ++r) {
            if (hand[r] <= 0) {
                continue;
            }
            const std::size_t start = candidates.hands.size();
            candidates.hands.insert(candidates.hands.end(), hands.begin(), hands.end());
            --candidates.hands[start + victim_index][r];
            ++candidates.hands[start + thief_index][r];
            candidates.weights.push_back(belief.weights[h] * hand[r]);
        }
    }
    return merged(candidates);
}

/**
 * Every deal of the unknown pool to the seats' unknown slots, weighted as a
 * uniform deal of the pool's cards (hand sizes fixed, a deal's probability is
 * proportional to 1 / prod c! over the counts c it gives each seat of each
 * kind), or null when there are more than `cap` deals. An empty pool gives the
 * one known world.
 */
std::shared_ptr<const HandBelief> enumerated(
    const std::vector<std::array<int, 5>>& known, const std::vector<int>& unknown,
    std::array<int, 5> pool, int cap) {
    // 1 / c! for every count a seat could get of one kind.
    const auto most_of_a_kind = static_cast<std::size_t>(*std::max_element(pool.begin(), pool.end()));
    std::vector<double> inverse_factorial(most_of_a_kind + 1, 1.0);
    for (std::size_t k = 1; k < inverse_factorial.size(); ++k) {
        inverse_factorial[k] = inverse_factorial[k - 1] / static_cast<double>(k);
    }
    const std::size_t n = known.size();
    auto belief = std::make_shared<HandBelief>();
    belief->seats = static_cast<int>(n);
    std::vector<std::array<int, 5>> hands = known;
    bool over = false;
    // Seat by seat, kind by kind: how many unknown cards of kind r seat s got.
    std::function<void(std::size_t, std::size_t, int, double)> deal =
        [&](std::size_t s, std::size_t r, int left, double weight) {
            if (over) {
                return;
            }
            if (s == n) {
                if (static_cast<int>(belief->size()) >= cap) {
                    over = true;
                    return;
                }
                belief->hands.insert(belief->hands.end(), hands.begin(), hands.end());
                belief->weights.push_back(weight);
                return;
            }
            if (r == 5) {
                if (left == 0) {
                    deal(s + 1, 0, s + 1 < n ? unknown[s + 1] : 0, weight);
                }
                return;
            }
            const int most = std::min(left, pool[r]);
            for (int c = r == 4 ? left : 0; c <= most; ++c) {
                hands[s][r] = known[s][r] + c;
                pool[r] -= c;
                deal(s, r + 1, left - c, weight * inverse_factorial[static_cast<std::size_t>(c)]);
                pool[r] += c;
            }
            hands[s][r] = known[s][r];
        };
    deal(0, 0, n > 0 ? unknown[0] : 0, 1.0);
    if (over) {
        return nullptr;
    }
    normalize(*belief);
    return belief;
}

}  // namespace

TrackerEvent TrackerEvent::gain(int player, const std::array<int, 5>& cards) {
    TrackerEvent event;
    event.type = TrackerEventType::gain;
    event.player = player;
    event.cards = cards;
    return event;
}

TrackerEvent TrackerEvent::lose(int player, const std::array<int, 5>& cards) {
    TrackerEvent event = gain(player, cards);
    event.type = TrackerEventType::lose;
    return event;
}

TrackerEvent TrackerEvent::transfer(int from, int to, const std::array<int, 5>& cards) {
    TrackerEvent event = gain(to, cards);
    event.type = TrackerEventType::transfer;
    event.other = from;
    return event;
}

TrackerEvent TrackerEvent::steal(int thief, int victim, std::optional<Resource> resource) {
    TrackerEvent event;
    event.type = TrackerEventType::steal;
    event.player = thief;
    event.other = victim;
    event.resource = resource;
    return event;
}

TrackerEvent TrackerEvent::monopoly(int player, Resource resource, std::vector<int> amounts) {
    TrackerEvent event;
    event.type = TrackerEventType::monopoly;
    event.player = player;
    event.resource = resource;
    event.amounts = std::move(amounts);
    return event;
}

TrackerEvent TrackerEvent::buy_development_card(int player, std::optional<DevelopmentCard> card) {
    TrackerEvent event;
    event.type = TrackerEventType::buy_development_card;
    event.player = player;
    event.development_card = card;
    return event;
}

TrackerEvent TrackerEvent::play_development_card(int player, DevelopmentCard card) {
    TrackerEvent event = buy_development_card(player, card);
    event.type = TrackerEventType::play_development_card;
    return event;
}

TrackerEvent TrackerEvent::end_turn(int player) {
    TrackerEvent event;
    event.type = TrackerEventType::end_turn;
    event.player = player;
    return event;
}

TrackerEvent TrackerEvent::victory_point_cards(int player, int at_least, int at_most) {
    TrackerEvent event;
    event.type = TrackerEventType::victory_point_cards;
    event.player = player;
    event.amounts = {at_least, at_most};
    return event;
}

TrackerEvent TrackerEvent::public_victory_points(std::vector<int> points) {
    TrackerEvent event;
    event.type = TrackerEventType::public_victory_points;
    event.amounts = std::move(points);
    return event;
}

HandTracker::HandTracker(const Game& game, int observer, TrackerMode mode, int max_hypotheses)
    : observer_(observer),
      rules_(game.rules()),
      sizes_(game.colors().size()),
      known_(game.colors().size()),
      development_counts_(game.colors().size()),
      played_(game.colors().size()),
      deck_size_(static_cast<int>(game.development_cards_remaining())),
      bought_this_turn_(game.colors().size()),
      victory_point_caps_(game.colors().size()),
      victory_point_floors_(game.colors().size()),
      mode_(mode),
      max_hypotheses_(max_hypotheses) {
    check_seat(observer);
    if (max_hypotheses < 1) {
        throw std::invalid_argument("max_hypotheses must be positive");
    }
    const bool over = game.winning_color().has_value();
    for (std::size_t seat = 0; seat < sizes_.size(); ++seat) {
        const PlayerState& state = game.players()[seat];
        sizes_[seat] = sum(state.resources);
        development_counts_[seat] = sum(state.development_cards);
        played_[seat] = state.played_development_cards;
        victory_point_caps_[seat] = over ? development_counts_[seat]
                                         : std::min(development_counts_[seat],
                                                    rules_.victory_points_to_win - 1 - state.victory_points);
    }
    const PlayerState& own = game.players()[static_cast<std::size_t>(observer)];
    known_[static_cast<std::size_t>(observer)] = own.resources;
    own_development_cards_ = own.development_cards;
    for (std::size_t r = 0; r < 5; ++r) {
        totals_[r] = rules_.resource_bank[r] - game.resource_bank()[r];
    }
    // Only when the last purchase was this turn; how many is not public state.
    const int turn = game.current_turn_index();
    if (game.players()[static_cast<std::size_t>(turn)].last_development_card_bought_completed_turn ==
        game.completed_turns()) {
        bought_this_turn_[static_cast<std::size_t>(turn)] = 1;
    }
    validate();
    if (mode_ == TrackerMode::exact) {
        // A fresh game is the one known world; a game in progress every deal
        // of its pool, when they fit (a tracker started mid-game by a view
        // rarely does, and then deals the pool until every hand is known).
        std::vector<int> unknown;
        for (int seat = 0; seat < num_players(); ++seat) {
            unknown.push_back(unknown_count(seat));
        }
        belief_ = enumerated(known_, unknown, unknown_pool(), max_hypotheses_);
        if (belief_ == nullptr) {
            ++overflows_;
        }
        validate();
    }
    if (game.current_player_index() == observer) {
        observe_options(game, game.playable_actions());
    }
}

std::size_t HandTracker::hypothesis_count() const noexcept {
    return belief_ == nullptr ? 0 : belief_->size();
}

std::span<const std::array<int, 5>> HandTracker::hypothesis_hands(std::size_t index) const {
    if (index >= hypothesis_count()) {
        throw std::out_of_range("no such hypothesis");
    }
    return belief_->hypothesis(index);
}

double HandTracker::hypothesis_weight(std::size_t index) const {
    if (index >= hypothesis_count()) {
        throw std::out_of_range("no such hypothesis");
    }
    return belief_->weights[index];
}

std::span<const double> HandTracker::hypothesis_weights() const noexcept {
    if (belief_ == nullptr) {
        return {};
    }
    return belief_->weights;
}

bool HandTracker::operator==(const HandTracker& other) const {
    const bool same_belief =
        (belief_ == nullptr) == (other.belief_ == nullptr) &&
        (belief_ == nullptr || belief_ == other.belief_ || *belief_ == *other.belief_);
    return observer_ == other.observer_ && rules_ == other.rules_ && sizes_ == other.sizes_ &&
           known_ == other.known_ && totals_ == other.totals_ &&
           development_counts_ == other.development_counts_ && played_ == other.played_ &&
           own_development_cards_ == other.own_development_cards_ && deck_size_ == other.deck_size_ &&
           bought_this_turn_ == other.bought_this_turn_ &&
           victory_point_caps_ == other.victory_point_caps_ &&
           victory_point_floors_ == other.victory_point_floors_ && mode_ == other.mode_ &&
           max_hypotheses_ == other.max_hypotheses_ && overflows_ == other.overflows_ && same_belief;
}

void HandTracker::check_seat(int seat) const {
    if (seat < 0 || seat >= num_players()) {
        throw std::out_of_range("seat is not in this game");
    }
}

int HandTracker::hand_size(int seat) const {
    check_seat(seat);
    return sizes_[static_cast<std::size_t>(seat)];
}

const std::array<int, 5>& HandTracker::known_resources(int seat) const {
    check_seat(seat);
    return known_[static_cast<std::size_t>(seat)];
}

int HandTracker::unknown_count(int seat) const {
    check_seat(seat);
    return sizes_[static_cast<std::size_t>(seat)] - sum(known_[static_cast<std::size_t>(seat)]);
}

std::array<int, 5> HandTracker::unknown_pool() const {
    std::array<int, 5> pool = totals_;
    for (const auto& known : known_) {
        for (std::size_t r = 0; r < 5; ++r) {
            pool[r] -= known[r];
        }
    }
    return pool;
}

int HandTracker::development_card_count(int seat) const {
    check_seat(seat);
    return development_counts_[static_cast<std::size_t>(seat)];
}

const std::array<int, 5>& HandTracker::played_development_cards(int seat) const {
    check_seat(seat);
    return played_[static_cast<std::size_t>(seat)];
}

std::array<int, 5> HandTracker::unseen_development_cards() const {
    std::array<int, 5> unseen = rules_.development_deck;
    for (std::size_t d = 0; d < 5; ++d) {
        unseen[d] -= own_development_cards_[d];
        for (const auto& played : played_) {
            unseen[d] -= played[d];
        }
    }
    return unseen;
}

int HandTracker::bought_this_turn(int seat) const {
    check_seat(seat);
    return bought_this_turn_[static_cast<std::size_t>(seat)];
}

int HandTracker::victory_point_card_floor(int seat) const {
    check_seat(seat);
    return victory_point_floors_[static_cast<std::size_t>(seat)];
}

int HandTracker::victory_point_card_cap(int seat) const {
    check_seat(seat);
    const auto index = static_cast<std::size_t>(seat);
    return std::min(victory_point_caps_[index], development_counts_[index]);
}

void HandTracker::validate() const {
    const auto pool = unknown_pool();
    for (std::size_t r = 0; r < 5; ++r) {
        if (totals_[r] < 0 || totals_[r] > rules_.resource_bank[r]) {
            inconsistent("more cards in hands than the bank started with");
        }
        if (pool[r] < 0) {
            inconsistent("known cards exceed the cards in all hands");
        }
    }
    for (int seat = 0; seat < num_players(); ++seat) {
        const auto index = static_cast<std::size_t>(seat);
        if (any_negative(known_[index]) || unknown_count(seat) < 0) {
            inconsistent("a hand holds fewer cards than are known");
        }
        if (development_counts_[index] < 0 || any_negative(played_[index])) {
            inconsistent("negative development-card count");
        }
        if (victory_point_caps_[index] < 0) {
            inconsistent("a seat would already have won");
        }
        if (seat != observer_ && victory_point_floors_[index] > victory_point_card_cap(seat)) {
            inconsistent("a seat must hold more victory point cards than it can");
        }
    }
    if (unknown_count(observer_) != 0) {
        inconsistent("the observer's own hand must be fully known");
    }
    if (development_counts_[static_cast<std::size_t>(observer_)] != sum(own_development_cards_) ||
        any_negative(own_development_cards_)) {
        inconsistent("the observer's development cards disagree with its count");
    }
    const auto unseen = unseen_development_cards();
    int hidden = deck_size_;
    for (int seat = 0; seat < num_players(); ++seat) {
        if (seat != observer_) {
            hidden += development_counts_[static_cast<std::size_t>(seat)];
        }
    }
    if (deck_size_ < 0 || any_negative(unseen) || sum(unseen) != hidden) {
        inconsistent("unseen development cards disagree with the deck and hands");
    }
    int floors = 0;
    for (int seat = 0; seat < num_players(); ++seat) {
        floors += seat == observer_ ? 0 : victory_point_floors_[static_cast<std::size_t>(seat)];
    }
    if (floors > unseen[kVictoryPoint]) {
        inconsistent("more victory point cards are held than remain unseen");
    }
    if (belief_ != nullptr) {
        // Every hypothesis is a world the pool bounds allow: the belief is
        // exact and the bounds are sound, so a violation is a bug in one.
        const HandBelief& belief = *belief_;
        const auto n = static_cast<std::size_t>(num_players());
        if (mode_ != TrackerMode::exact || belief.seats != num_players() || belief.size() == 0 ||
            belief.hands.size() != belief.size() * n ||
            static_cast<int>(belief.size()) > max_hypotheses_) {
            inconsistent("the exact belief is malformed");
        }
        for (std::size_t h = 0; h < belief.size(); ++h) {
            const auto hands = belief.hypothesis(h);
            std::array<int, 5> total{};
            for (std::size_t seat = 0; seat < n; ++seat) {
                if (sum(hands[seat]) != sizes_[seat]) {
                    inconsistent("a hypothesis disagrees with a hand size");
                }
                for (std::size_t r = 0; r < 5; ++r) {
                    if (hands[seat][r] < known_[seat][r]) {
                        inconsistent("a hypothesis holds fewer cards than are known");
                    }
                    total[r] += hands[seat][r];
                }
            }
            const auto observer = static_cast<std::size_t>(observer_);
            if (hands[observer] != known_[observer]) {
                inconsistent("a hypothesis disagrees with the observer's own hand");
            }
            if (total != totals_) {
                inconsistent("a hypothesis disagrees with the bank");
            }
            if (!(belief.weights[h] > 0.0) || !std::isfinite(belief.weights[h])) {
                inconsistent("a hypothesis has no weight");
            }
        }
    }
}

void HandTracker::take(int seat, const std::array<int, 5>& cards) {
    if (any_negative(cards)) {
        throw std::invalid_argument("card counts must be non-negative");
    }
    const auto index = static_cast<std::size_t>(seat);
    const int unknown = unknown_count(seat);
    int from_unknown = 0;
    auto& known = known_[index];
    for (std::size_t r = 0; r < 5; ++r) {
        const int from_known = std::min(known[r], cards[r]);
        known[r] -= from_known;
        from_unknown += cards[r] - from_known;
    }
    if (from_unknown > unknown) {
        inconsistent("a seat gave away cards it cannot hold");
    }
    sizes_[index] -= sum(cards);
}

void HandTracker::apply(const TrackerEvent& event) {
    HandTracker next = *this;
    switch (event.type) {
        case TrackerEventType::gain: {
            next.check_seat(event.player);
            if (any_negative(event.cards)) {
                throw std::invalid_argument("card counts must be non-negative");
            }
            const auto index = static_cast<std::size_t>(event.player);
            for (std::size_t r = 0; r < 5; ++r) {
                next.known_[index][r] += event.cards[r];
                next.totals_[r] += event.cards[r];
            }
            next.sizes_[index] += sum(event.cards);
            break;
        }
        case TrackerEventType::lose:
            next.check_seat(event.player);
            next.take(event.player, event.cards);
            for (std::size_t r = 0; r < 5; ++r) {
                next.totals_[r] -= event.cards[r];
            }
            break;
        case TrackerEventType::transfer: {
            next.check_seat(event.player);
            next.check_seat(event.other);
            if (event.player == event.other) {
                throw std::invalid_argument("a seat cannot trade with itself");
            }
            next.take(event.other, event.cards);
            const auto index = static_cast<std::size_t>(event.player);
            for (std::size_t r = 0; r < 5; ++r) {
                next.known_[index][r] += event.cards[r];
            }
            next.sizes_[index] += sum(event.cards);
            break;
        }
        case TrackerEventType::steal: {
            next.check_seat(event.player);
            next.check_seat(event.other);
            if (event.player == event.other) {
                throw std::invalid_argument("a seat cannot rob itself");
            }
            const auto thief = static_cast<std::size_t>(event.player);
            const auto victim = static_cast<std::size_t>(event.other);
            std::optional<Resource> resource = event.resource;
            if (!resource.has_value()) {
                if (event.player == observer_ || event.other == observer_) {
                    throw std::invalid_argument("the observer sees what it steals or loses");
                }
                if (next.sizes_[victim] <= 0) {
                    inconsistent("robbed a seat without cards");
                }
                const auto pool = next.unknown_pool();
                const bool unknown = next.unknown_count(event.other) > 0;
                int kinds = 0;
                for (std::size_t r = 0; r < 5; ++r) {
                    if (next.known_[victim][r] > 0 || (unknown && pool[r] > 0)) {
                        ++kinds;
                        resource = kResources[r];
                    }
                }
                if (kinds != 1) {
                    resource.reset();
                }
            }
            if (resource.has_value()) {
                next.take(event.other, single(*resource));
                ++next.known_[thief][static_cast<std::size_t>(*resource)];
            } else {
                // Any known kind may have gone: keep only what is certain.
                for (int& known : next.known_[victim]) {
                    known = std::max(0, known - 1);
                }
                --next.sizes_[victim];
            }
            ++next.sizes_[thief];
            break;
        }
        case TrackerEventType::monopoly: {
            next.check_seat(event.player);
            if (!event.resource.has_value()) {
                throw std::invalid_argument("a monopoly names its resource");
            }
            if (static_cast<int>(event.amounts.size()) != num_players()) {
                throw std::invalid_argument("a monopoly lists the cards taken from every seat");
            }
            const auto r = static_cast<std::size_t>(*event.resource);
            int taken = 0;
            for (int seat = 0; seat < num_players(); ++seat) {
                if (seat == event.player) {
                    continue;
                }
                next.take(seat, single(*event.resource, event.amounts[static_cast<std::size_t>(seat)]));
                taken += event.amounts[static_cast<std::size_t>(seat)];
                if (next.known_[static_cast<std::size_t>(seat)][r] != 0) {
                    inconsistent("a monopoly left known cards behind");
                }
            }
            // Everyone else now holds none, so the player holds them all.
            const auto index = static_cast<std::size_t>(event.player);
            next.sizes_[index] += taken;
            next.known_[index][r] = next.totals_[r];
            break;
        }
        case TrackerEventType::buy_development_card: {
            next.check_seat(event.player);
            const auto index = static_cast<std::size_t>(event.player);
            if (event.player == observer_) {
                if (!event.development_card.has_value()) {
                    throw std::invalid_argument("the observer knows the card it bought");
                }
                ++next.own_development_cards_[static_cast<std::size_t>(*event.development_card)];
            } else if (event.development_card.has_value()) {
                throw std::invalid_argument("only the buyer sees a bought card");
            }
            ++next.development_counts_[index];
            ++next.bought_this_turn_[index];
            ++next.victory_point_caps_[index];
            --next.deck_size_;
            break;
        }
        case TrackerEventType::play_development_card: {
            next.check_seat(event.player);
            if (!event.development_card.has_value() ||
                *event.development_card == DevelopmentCard::victory_point) {
                throw std::invalid_argument("a played card is a knight or progress card");
            }
            const auto index = static_cast<std::size_t>(event.player);
            const auto card = static_cast<std::size_t>(*event.development_card);
            if (event.player == observer_) {
                --next.own_development_cards_[card];
            }
            --next.development_counts_[index];
            ++next.played_[index][card];
            break;
        }
        case TrackerEventType::end_turn:
            next.check_seat(event.player);
            next.bought_this_turn_[static_cast<std::size_t>(event.player)] = 0;
            break;
        case TrackerEventType::victory_point_cards: {
            next.check_seat(event.player);
            if (event.amounts.size() != 2 || event.amounts[0] > event.amounts[1]) {
                throw std::invalid_argument("victory point card bounds are [at least, at most]");
            }
            const auto index = static_cast<std::size_t>(event.player);
            next.victory_point_floors_[index] = std::max(next.victory_point_floors_[index], event.amounts[0]);
            next.victory_point_caps_[index] = std::min(next.victory_point_caps_[index], event.amounts[1]);
            break;
        }
        case TrackerEventType::public_victory_points:
            if (static_cast<int>(event.amounts.size()) != num_players()) {
                throw std::invalid_argument("public victory points list every seat");
            }
            for (std::size_t seat = 0; seat < event.amounts.size(); ++seat) {
                next.victory_point_caps_[seat] = std::min(
                    next.victory_point_caps_[seat],
                    rules_.victory_points_to_win - 1 - event.amounts[seat]);
            }
            break;
    }
    next.update_belief(event);
    next.validate();
    *this = std::move(next);
}

void HandTracker::update_belief(const TrackerEvent& event) {
    if (mode_ != TrackerMode::exact) {
        return;
    }
    if (belief_ == nullptr) {
        // Fallen back to the pool. Once the pool holds cards of one kind at
        // most, every hand is known again and that one world is the exact
        // posterior, whatever was forgotten.
        const auto pool = unknown_pool();
        if (std::count_if(pool.begin(), pool.end(), [](int count) { return count > 0; }) <= 1) {
            std::vector<int> unknown;
            for (int seat = 0; seat < num_players(); ++seat) {
                unknown.push_back(unknown_count(seat));
            }
            belief_ = enumerated(known_, unknown, pool, max_hypotheses_);
        }
        return;
    }
    const HandBelief& belief = *belief_;
    const auto player = static_cast<std::size_t>(std::max(event.player, 0));
    const auto other = static_cast<std::size_t>(std::max(event.other, 0));
    const auto add = [](std::array<int, 5>& hand, const std::array<int, 5>& cards) {
        for (std::size_t r = 0; r < 5; ++r) {
            hand[r] += cards[r];
        }
    };
    const auto remove = [](std::array<int, 5>& hand, const std::array<int, 5>& cards) {
        for (std::size_t r = 0; r < 5; ++r) {
            hand[r] -= cards[r];
        }
        return !any_negative(hand);
    };
    switch (event.type) {
        case TrackerEventType::gain:
            belief_ = transformed(belief, [&](Hands hands, double&) {
                add(hands[player], event.cards);
                return true;
            });
            break;
        case TrackerEventType::lose:
            belief_ = transformed(belief, [&](Hands hands, double&) { return remove(hands[player], event.cards); });
            break;
        case TrackerEventType::transfer:
            belief_ = transformed(belief, [&](Hands hands, double&) {
                if (!remove(hands[other], event.cards)) {
                    return false;
                }
                add(hands[player], event.cards);
                return true;
            });
            break;
        case TrackerEventType::steal:
            if (event.resource.has_value()) {
                // Seen by the observer: still evidence, since a hand with more
                // of that kind was likelier to give it up.
                const auto r = static_cast<std::size_t>(*event.resource);
                belief_ = transformed(belief, [&](Hands hands, double& weight) {
                    const int count = hands[other][r];
                    if (count <= 0) {
                        return false;
                    }
                    weight *= count;
                    --hands[other][r];
                    ++hands[player][r];
                    return true;
                });
            } else {
                belief_ = branched(belief, event.player, event.other);
                if (static_cast<int>(belief_->size()) > max_hypotheses_) {
                    belief_ = nullptr;
                    ++overflows_;
                }
            }
            break;
        case TrackerEventType::monopoly: {
            // The amounts are public and the monopoly takes every card of the
            // kind, so each seat's count of it under a hypothesis must match.
            const auto r = static_cast<std::size_t>(*event.resource);
            belief_ = transformed(belief, [&](Hands hands, double&) {
                int taken = 0;
                for (std::size_t seat = 0; seat < hands.size(); ++seat) {
                    if (seat == player) {
                        continue;
                    }
                    if (hands[seat][r] != event.amounts[seat]) {
                        return false;
                    }
                    taken += hands[seat][r];
                    hands[seat][r] = 0;
                }
                hands[player][r] += taken;
                return true;
            });
            break;
        }
        case TrackerEventType::buy_development_card:
        case TrackerEventType::play_development_card:
        case TrackerEventType::end_turn:
        case TrackerEventType::victory_point_cards:
        case TrackerEventType::public_victory_points:
            break;  // nothing about resource hands
    }
}

void HandTracker::gain(int player, const std::array<int, 5>& cards) {
    apply(TrackerEvent::gain(player, cards));
}

void HandTracker::lose(int player, const std::array<int, 5>& cards) {
    apply(TrackerEvent::lose(player, cards));
}

void HandTracker::transfer(int from, int to, const std::array<int, 5>& cards) {
    apply(TrackerEvent::transfer(from, to, cards));
}

void HandTracker::steal(int thief, int victim, std::optional<Resource> resource) {
    apply(TrackerEvent::steal(thief, victim, resource));
}

void HandTracker::monopoly(int player, Resource resource, const std::vector<int>& amounts) {
    apply(TrackerEvent::monopoly(player, resource, amounts));
}

void HandTracker::buy_development_card(int player, std::optional<DevelopmentCard> card) {
    apply(TrackerEvent::buy_development_card(player, card));
}

void HandTracker::play_development_card(int player, DevelopmentCard card) {
    apply(TrackerEvent::play_development_card(player, card));
}

void HandTracker::end_turn(int player) {
    apply(TrackerEvent::end_turn(player));
}

void HandTracker::public_victory_points(const std::vector<int>& points) {
    apply(TrackerEvent::public_victory_points(points));
}

void HandTracker::observe_options(const Game& game, const std::vector<Action>& legal) {
    for (const TrackerEvent& event : option_events(game, legal, observer_)) {
        apply(event);
    }
}

std::vector<TrackerEvent> HandTracker::option_events(
    const Game& game, const std::vector<Action>& legal, int observer) {
    std::vector<TrackerEvent> events;
    if (!game.friendly_robber() || game.current_prompt() != ActionPrompt::move_robber ||
        game.current_player_index() != observer) {
        return events;
    }
    // The engine (like Catanatron) drops tiles with a building of another
    // seat under 3 actual points, unless that would drop every tile.
    const Color own = game.colors()[static_cast<std::size_t>(observer)];
    std::set<Coordinate> allowed;
    for (const Action& action : legal) {
        if (action.type == ActionType::move_robber) {
            allowed.insert(std::get<RobberMove>(action.value).coordinate);
        }
    }
    std::vector<std::vector<int>> tiles;  // other seats' buildings per candidate tile
    std::vector<bool> tile_allowed;
    for (const auto& [coordinate, tile_index] : game.board().map().coordinate_to_tile()) {
        const Tile& tile = game.board().map().tiles().at(static_cast<std::size_t>(tile_index));
        if (tile.kind != TileKind::land || coordinate == game.board().robber_coordinate()) {
            continue;
        }
        std::vector<int> owners;
        for (int node : tile.nodes) {
            const auto building = game.board().buildings().find(node);
            if (building != game.board().buildings().end() && building->second.color != own) {
                const int seat = seat_of(game, building->second.color);
                if (std::find(owners.begin(), owners.end(), seat) == owners.end()) {
                    owners.push_back(seat);
                }
            }
        }
        tiles.push_back(std::move(owners));
        tile_allowed.push_back(allowed.contains(coordinate));
    }
    const bool all_allowed = std::all_of(tile_allowed.begin(), tile_allowed.end(), [](bool a) { return a; });
    const bool some_unowned = std::any_of(
        tiles.begin(), tiles.end(), [](const std::vector<int>& owners) { return owners.empty(); });
    if (all_allowed && !some_unowned) {
        return events;  // nothing dropped, or everything dropped and restored
    }
    const int n = static_cast<int>(game.colors().size());
    std::vector<bool> at_least_three(static_cast<std::size_t>(n));
    for (std::size_t t = 0; t < tiles.size(); ++t) {
        if (tile_allowed[t]) {
            for (int seat : tiles[t]) {
                at_least_three[static_cast<std::size_t>(seat)] = true;
            }
        }
    }
    std::vector<bool> under_three(static_cast<std::size_t>(n));
    for (std::size_t t = 0; t < tiles.size(); ++t) {
        if (tile_allowed[t]) {
            continue;
        }
        std::vector<int> candidates;
        for (int seat : tiles[t]) {
            if (!at_least_three[static_cast<std::size_t>(seat)]) {
                candidates.push_back(seat);
            }
        }
        if (candidates.size() == 1) {
            under_three[static_cast<std::size_t>(candidates.front())] = true;
        }
    }
    for (int seat = 0; seat < n; ++seat) {
        const int points = game.players()[static_cast<std::size_t>(seat)].victory_points;
        if (seat == observer || points >= 3) {
            continue;
        }
        if (at_least_three[static_cast<std::size_t>(seat)]) {
            events.push_back(
                TrackerEvent::victory_point_cards(seat, 3 - points, std::numeric_limits<int>::max()));
        } else if (under_three[static_cast<std::size_t>(seat)]) {
            events.push_back(TrackerEvent::victory_point_cards(seat, 0, 2 - points));
        }
    }
    return events;
}

namespace {

std::vector<TrackerEvent> transition_events(
    const Game& before, const Action& action, const Game& after, int observer);

}  // namespace

std::vector<TrackerEvent> HandTracker::events(
    const Game& before, const Action& action, const Game& after, int observer) {
    auto events = transition_events(before, action, after, observer);
    if (!after.winning_color().has_value()) {
        std::vector<int> points;
        for (const PlayerState& state : after.players()) {
            points.push_back(state.victory_points);
        }
        events.push_back(TrackerEvent::public_victory_points(std::move(points)));
        // The observer sees its own options when it is to move.
        if (after.current_player_index() == observer) {
            for (TrackerEvent& event : option_events(after, after.playable_actions(), observer)) {
                events.push_back(std::move(event));
            }
        }
    }
    return events;
}

namespace {

std::vector<TrackerEvent> transition_events(
    const Game& before, const Action& action, const Game& after, int observer) {
    const int n = static_cast<int>(before.colors().size());
    if (before.colors() != after.colors()) {
        throw std::invalid_argument("before and after must be the same game");
    }
    if (observer < 0 || observer >= n) {
        throw std::out_of_range("observer is not in this game");
    }
    const int player = seat_of(before, action.color);
    std::vector<std::array<int, 5>> deltas(static_cast<std::size_t>(n));
    for (std::size_t seat = 0; seat < deltas.size(); ++seat) {
        for (std::size_t r = 0; r < 5; ++r) {
            deltas[seat][r] = after.players()[seat].resources[r] - before.players()[seat].resources[r];
        }
    }

    std::vector<TrackerEvent> events;
    const auto play = [&](DevelopmentCard card) {
        events.push_back(TrackerEvent::play_development_card(player, card));
    };
    switch (action.type) {
        case ActionType::move_robber: {
            const auto& move = std::get<RobberMove>(action.value);
            if (!move.victim.has_value()) {
                return events;
            }
            const int victim = seat_of(before, *move.victim);
            std::optional<Resource> resource;
            if (observer == player || observer == victim) {
                for (std::size_t r = 0; r < 5; ++r) {
                    if (deltas[static_cast<std::size_t>(player)][r] == 1) {
                        resource = kResources[r];
                    }
                }
            }
            events.push_back(TrackerEvent::steal(player, victim, resource));
            return events;
        }
        case ActionType::play_monopoly: {
            const Resource resource = std::get<Resource>(action.value);
            play(DevelopmentCard::monopoly);
            std::vector<int> amounts(static_cast<std::size_t>(n));
            for (int seat = 0; seat < n; ++seat) {
                if (seat != player) {
                    amounts[static_cast<std::size_t>(seat)] =
                        -deltas[static_cast<std::size_t>(seat)][static_cast<std::size_t>(resource)];
                }
            }
            events.push_back(TrackerEvent::monopoly(player, resource, std::move(amounts)));
            return events;
        }
        case ActionType::confirm_trade: {
            const auto& confirmed = std::get<ConfirmedTrade>(action.value);
            const int partner = seat_of(before, confirmed.partner);
            // Like Catanatron, the engine lets the offerer accept and confirm
            // its own offer; nothing moves.
            if (partner == player) {
                return events;
            }
            if (sum(confirmed.trade.offering) > 0) {
                events.push_back(TrackerEvent::transfer(player, partner, confirmed.trade.offering));
            }
            if (sum(confirmed.trade.asking) > 0) {
                events.push_back(TrackerEvent::transfer(partner, player, confirmed.trade.asking));
            }
            return events;
        }
        case ActionType::play_knight_card:
            play(DevelopmentCard::knight);
            break;
        case ActionType::play_year_of_plenty:
            play(DevelopmentCard::year_of_plenty);
            break;
        case ActionType::play_road_building:
            play(DevelopmentCard::road_building);
            break;
        default:
            break;
    }

    // Every other resource movement is between hands and the bank, in the open.
    for (int seat = 0; seat < n; ++seat) {
        std::array<int, 5> lost{};
        for (std::size_t r = 0; r < 5; ++r) {
            lost[r] = std::max(0, -deltas[static_cast<std::size_t>(seat)][r]);
        }
        if (sum(lost) > 0) {
            events.push_back(TrackerEvent::lose(seat, lost));
        }
    }
    for (int seat = 0; seat < n; ++seat) {
        std::array<int, 5> gained{};
        for (std::size_t r = 0; r < 5; ++r) {
            gained[r] = std::max(0, deltas[static_cast<std::size_t>(seat)][r]);
        }
        if (sum(gained) > 0) {
            events.push_back(TrackerEvent::gain(seat, gained));
        }
    }

    if (action.type == ActionType::buy_development_card) {
        std::optional<DevelopmentCard> card;
        if (observer == player) {
            const auto& old_cards = before.players()[static_cast<std::size_t>(player)].development_cards;
            const auto& new_cards = after.players()[static_cast<std::size_t>(player)].development_cards;
            for (std::size_t d = 0; d < 5; ++d) {
                if (new_cards[d] == old_cards[d] + 1) {
                    card = static_cast<DevelopmentCard>(d);
                }
            }
        }
        events.push_back(TrackerEvent::buy_development_card(player, card));
    } else if (action.type == ActionType::end_turn) {
        events.push_back(TrackerEvent::end_turn(player));
    }
    return events;
}

}  // namespace

void HandTracker::observe(const Game& before, const Action& action, const Game& after) {
    for (const TrackerEvent& event : events(before, action, after, observer_)) {
        apply(event);
    }
}

std::array<double, 5> HandTracker::expected_resources(int seat) const {
    check_seat(seat);
    const auto& known = known_[static_cast<std::size_t>(seat)];
    std::array<double, 5> expected{};
    if (belief_ != nullptr) {
        for (std::size_t h = 0; h < belief_->size(); ++h) {
            const auto& hand = belief_->hypothesis(h)[static_cast<std::size_t>(seat)];
            for (std::size_t r = 0; r < 5; ++r) {
                expected[r] += belief_->weights[h] * hand[r];
            }
        }
        return expected;
    }
    const auto pool = unknown_pool();
    const int pool_size = sum(pool);
    const int unknown = unknown_count(seat);
    for (std::size_t r = 0; r < 5; ++r) {
        expected[r] = known[r];
        if (pool_size > 0) {
            expected[r] += static_cast<double>(unknown) * pool[r] / pool_size;
        }
    }
    return expected;
}

std::array<double, 5> HandTracker::expected_development_cards(int seat) const {
    check_seat(seat);
    std::array<double, 5> expected{};
    if (seat == observer_) {
        std::copy(own_development_cards_.begin(), own_development_cards_.end(), expected.begin());
        return expected;
    }
    const auto unseen = unseen_development_cards();
    const int unseen_size = sum(unseen);
    const int count = development_counts_[static_cast<std::size_t>(seat)];
    for (std::size_t d = 0; d < 5; ++d) {
        if (unseen_size > 0) {
            expected[d] = static_cast<double>(count) * unseen[d] / unseen_size;
        }
    }
    return expected;
}

void HandTracker::check_public(const Game& game) const {
    if (static_cast<int>(game.colors().size()) != num_players() || game.rules() != rules_) {
        throw std::invalid_argument("tracker belongs to a different game");
    }
    for (int seat = 0; seat < num_players(); ++seat) {
        const auto index = static_cast<std::size_t>(seat);
        const PlayerState& state = game.players()[index];
        if (sum(state.resources) != sizes_[index] ||
            sum(state.development_cards) != development_counts_[index] ||
            state.played_development_cards != played_[index]) {
            throw std::invalid_argument("tracker disagrees with the game's public hand sizes");
        }
    }
    const PlayerState& own = game.players()[static_cast<std::size_t>(observer_)];
    if (own.resources != known_[static_cast<std::size_t>(observer_)] ||
        own.development_cards != own_development_cards_) {
        throw std::invalid_argument("tracker disagrees with the observer's own hand");
    }
    for (std::size_t r = 0; r < 5; ++r) {
        if (rules_.resource_bank[r] - game.resource_bank()[r] != totals_[r]) {
            throw std::invalid_argument("tracker disagrees with the bank");
        }
    }
    if (static_cast<int>(game.development_cards_remaining()) != deck_size_) {
        throw std::invalid_argument("tracker disagrees with the development deck size");
    }
}

bool HandTracker::consistent_with(const Game& truth) const {
    try {
        check_public(truth);
    } catch (const std::invalid_argument&) {
        return false;
    }
    if (belief_ != nullptr) {
        bool found = false;
        for (std::size_t h = 0; h < belief_->size() && !found; ++h) {
            const auto hands = belief_->hypothesis(h);
            found = true;
            for (std::size_t seat = 0; seat < hands.size(); ++seat) {
                found = found && truth.players()[seat].resources == hands[seat];
            }
        }
        if (!found) {
            return false;
        }
    }
    std::array<int, 5> hidden{};
    for (DevelopmentCard card : truth.development_deck()) {
        ++hidden[static_cast<std::size_t>(card)];
    }
    const bool over = truth.winning_color().has_value();
    for (int seat = 0; seat < num_players(); ++seat) {
        if (seat == observer_) {
            continue;
        }
        const auto index = static_cast<std::size_t>(seat);
        const PlayerState& state = truth.players()[index];
        for (std::size_t r = 0; r < 5; ++r) {
            if (state.resources[r] < known_[index][r]) {
                return false;
            }
            hidden[r] += state.development_cards[r];
        }
        if (!over && state.victory_points + state.development_cards[kVictoryPoint] >=
                         truth.victory_points_to_win()) {
            return false;
        }
        if (state.development_cards[kVictoryPoint] > victory_point_card_cap(seat) ||
            state.development_cards[kVictoryPoint] < victory_point_floors_[index]) {
            return false;
        }
    }
    return hidden == unseen_development_cards();
}

Game sample_world(
    const Game& game, const HandTracker& tracker, std::mt19937_64& rng, std::optional<TrackerMode> deal) {
    // In a game in progress everyone knows nobody has won, which caps every
    // seat's victory point cards. Whether a game is over is public too, but
    // a finished game's worlds would have to keep its winner, and the game
    // object only says so through actual (hidden) points: refuse it.
    if (game.winning_color().has_value()) {
        throw std::invalid_argument("sample_world needs a game in progress, not a finished one");
    }
    const TrackerMode dealing = deal.value_or(tracker.mode());
    if (dealing == TrackerMode::exact && tracker.mode() != TrackerMode::exact) {
        throw std::invalid_argument("an exact deal needs a tracker in exact mode");
    }
    tracker.check_public(game);
    const int n = tracker.num_players();
    const int observer = tracker.observer();
    std::vector<HiddenHand> hands(static_cast<std::size_t>(n));
    std::vector<int> others;
    for (int seat = 0; seat < n; ++seat) {
        if (seat != observer) {
            others.push_back(seat);
        }
    }

    std::size_t cursor = 0;
    if (dealing == TrackerMode::exact && tracker.exact()) {
        // Resources: one hypothesis of the exact belief, by its posterior.
        const auto hypothesis = tracker.hypothesis_hands(random::discrete(rng, tracker.hypothesis_weights()));
        for (int seat : others) {
            hands[static_cast<std::size_t>(seat)].resources = hypothesis[static_cast<std::size_t>(seat)];
        }
    } else {
        // Resources: known cards plus a uniform deal of the unknown pool.
        std::vector<int> cards;
        const auto pool = tracker.unknown_pool();
        for (std::size_t r = 0; r < 5; ++r) {
            cards.insert(cards.end(), static_cast<std::size_t>(pool[r]), static_cast<int>(r));
        }
        random::shuffle(cards.begin(), cards.end(), rng);
        for (int seat : others) {
            auto& resources = hands[static_cast<std::size_t>(seat)].resources;
            resources = tracker.known_resources(seat);
            for (int k = tracker.unknown_count(seat); k > 0; --k) {
                ++resources[static_cast<std::size_t>(cards.at(cursor++))];
            }
        }
    }

    // Development cards: a uniform deal of the unseen cards, conditioned on
    // nobody holding enough victory points to have won, now or earlier.
    std::vector<int> unseen;
    const auto unseen_counts = tracker.unseen_development_cards();
    for (std::size_t d = 0; d < 5; ++d) {
        unseen.insert(unseen.end(), static_cast<std::size_t>(unseen_counts[d]), static_cast<int>(d));
    }
    std::vector<int> caps;
    for (int seat : others) {
        caps.push_back(std::min(
            tracker.victory_point_card_cap(seat),
            game.victory_points_to_win() - 1 -
                game.players()[static_cast<std::size_t>(seat)].victory_points));
    }
    std::vector<int> floors;
    for (int seat : others) {
        floors.push_back(tracker.victory_point_card_floor(seat));
    }
    const auto deal_fits = [&] {
        std::size_t position = 0;
        for (std::size_t i = 0; i < others.size(); ++i) {
            int victory_points = 0;
            for (int k = tracker.development_card_count(others[i]); k > 0; --k) {
                victory_points += unseen[position++] == static_cast<int>(kVictoryPoint);
            }
            if (victory_points > caps[i] || victory_points < floors[i]) {
                return false;
            }
        }
        return true;
    };
    bool fits = false;
    for (int attempt = 0; attempt < kVictoryPointDealAttempts && !fits; ++attempt) {
        random::shuffle(unseen.begin(), unseen.end(), rng);
        fits = deal_fits();
    }
    if (!fits) {
        // Every seat's floor first, then the deck, then seats under their caps.
        std::vector<int> rest;
        int victory_points = 0;
        for (int card : unseen) {
            if (card == static_cast<int>(kVictoryPoint)) {
                ++victory_points;
            } else {
                rest.push_back(card);
            }
        }
        random::shuffle(rest.begin(), rest.end(), rng);
        victory_points -= std::accumulate(floors.begin(), floors.end(), 0);
        const int in_deck = std::clamp(victory_points, 0, tracker.development_deck_size());
        victory_points -= in_deck;
        std::vector<int> dealt;
        for (std::size_t i = 0; i < others.size(); ++i) {
            const int count = tracker.development_card_count(others[i]);
            const int extra = std::min({victory_points, std::max(0, caps[i] - floors[i]), count - floors[i]});
            const int given = floors[i] + std::max(0, extra);
            std::vector<int> hand(static_cast<std::size_t>(given), static_cast<int>(kVictoryPoint));
            victory_points -= std::max(0, extra);
            while (static_cast<int>(hand.size()) < count) {
                if (rest.empty()) {
                    inconsistent("no deal of development cards fits the victory point caps");
                }
                hand.push_back(rest.back());
                rest.pop_back();
            }
            random::shuffle(hand.begin(), hand.end(), rng);
            dealt.insert(dealt.end(), hand.begin(), hand.end());
        }
        if (victory_points != 0) {
            inconsistent("no deal of development cards fits the victory point caps");
        }
        dealt.insert(dealt.end(), rest.begin(), rest.end());
        dealt.insert(dealt.end(), static_cast<std::size_t>(in_deck), static_cast<int>(kVictoryPoint));
        unseen = std::move(dealt);
        if (!deal_fits()) {
            inconsistent("no deal of development cards fits the victory point caps");
        }
    }
    cursor = 0;
    for (int seat : others) {
        auto& hand = hands[static_cast<std::size_t>(seat)];
        const int count = tracker.development_card_count(seat);
        // The first cards of the (random) slice are the ones bought this turn.
        const int bought = std::min(tracker.bought_this_turn(seat), count);
        std::array<int, 5> held_at_start{};
        for (int k = 0; k < count; ++k) {
            const auto card = static_cast<std::size_t>(unseen.at(cursor++));
            ++hand.development_cards[card];
            if (k >= bought) {
                ++held_at_start[card];
            }
        }
        for (std::size_t d = 0; d < hand.development_card_owned_at_start.size(); ++d) {
            hand.development_card_owned_at_start[d] = held_at_start[d] > 0;
        }
    }

    const PlayerState& own = game.players()[static_cast<std::size_t>(observer)];
    hands[static_cast<std::size_t>(observer)] = {
        own.resources, own.development_cards, own.development_card_owned_at_start};
    Game world = game;
    world.redeal_hidden_cards(hands, rng());
    return world;
}

}  // namespace cppanatron
