#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

#include "cppanatron/game.hpp"

namespace cppanatron {

void Game::redeal_hidden_cards(const std::vector<HiddenHand>& hands, std::uint64_t seed) {
    if (hands.size() != players_.size()) {
        throw std::invalid_argument("redeal needs one hand per seat");
    }
    const auto negative = [](const std::array<int, 5>& counts) {
        return std::any_of(counts.begin(), counts.end(), [](int count) { return count < 0; });
    };
    const auto total = [](const std::array<int, 5>& counts) {
        return std::accumulate(counts.begin(), counts.end(), 0);
    };

    std::array<int, 5> old_resources{};
    std::array<int, 5> new_resources{};
    std::array<int, 5> pool{};
    std::array<int, 5> dealt{};
    for (DevelopmentCard card : development_deck_) {
        ++pool[static_cast<std::size_t>(card)];
    }
    for (std::size_t seat = 0; seat < players_.size(); ++seat) {
        const HiddenHand& hand = hands[seat];
        const PlayerState& state = players_[seat];
        if (negative(hand.resources) || negative(hand.development_cards)) {
            throw std::invalid_argument("redealt card counts must be non-negative");
        }
        if (total(hand.resources) != total(state.resources) ||
            total(hand.development_cards) != total(state.development_cards)) {
            throw std::invalid_argument("redeal must keep every hand size");
        }
        for (std::size_t i = 0; i < 5; ++i) {
            old_resources[i] += state.resources[i];
            new_resources[i] += hand.resources[i];
            pool[i] += state.development_cards[i];
            dealt[i] += hand.development_cards[i];
        }
    }
    if (old_resources != new_resources) {
        throw std::invalid_argument("redeal must keep the resources the hands hold in total");
    }
    for (std::size_t i = 0; i < 5; ++i) {
        if (dealt[i] > pool[i]) {
            throw std::invalid_argument(
                "redealt development cards are not drawable from the deck and hands");
        }
    }

    constexpr auto victory_point = static_cast<std::size_t>(DevelopmentCard::victory_point);
    for (std::size_t seat = 0; seat < players_.size(); ++seat) {
        const HiddenHand& hand = hands[seat];
        PlayerState& state = players_[seat];
        state.actual_victory_points +=
            hand.development_cards[victory_point] - state.development_cards[victory_point];
        state.resources = hand.resources;
        state.development_cards = hand.development_cards;
        state.development_card_owned_at_start = hand.development_card_owned_at_start;
    }
    development_deck_.clear();
    for (std::size_t i = 0; i < pool.size(); ++i) {
        development_deck_.insert(
            development_deck_.end(), static_cast<std::size_t>(pool[i] - dealt[i]),
            static_cast<DevelopmentCard>(i));
    }
    std::mt19937_64 shuffle_random(seed);
    std::shuffle(development_deck_.begin(), development_deck_.end(), shuffle_random);
    // Future dice and steals must not follow the old (hidden) random stream.
    random_.seed(shuffle_random());
    playable_actions_ = generate_playable_actions();
}

}  // namespace cppanatron
