#include "cppanatron/flat_codec.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace cppanatron {
namespace {

/** The table's form of a game action: placeholder color, relative victim. */
Action table_key(const Game& game, const Action& action) {
    Action key = action;
    key.color = Color::red;
    if (auto* edge = std::get_if<Edge>(&key.value)) {
        *edge = edge->normalized();
    }
    if (const auto* move = std::get_if<RobberMove>(&action.value)) {
        std::optional<int> slot;
        if (move->victim.has_value()) {
            const auto& colors = game.colors();
            const auto actor = std::find(colors.begin(), colors.end(), action.color);
            const auto victim = std::find(colors.begin(), colors.end(), *move->victim);
            if (actor == colors.end() || victim == colors.end()) {
                throw std::out_of_range("robber actor or victim is not in game");
            }
            const int count = static_cast<int>(colors.size());
            slot = (static_cast<int>(victim - colors.begin()) -
                    static_cast<int>(actor - colors.begin()) + count) %
                   count;
        }
        key.value = FlatRobberMove{move->coordinate, slot};
    }
    return key;
}

}  // namespace

FlatActionCodec::FlatActionCodec(int num_players, MapType map_type)
    : space_(num_players, map_type) {
    for (std::size_t i = 0; i < space_.size(); ++i) {
        indices_.emplace(space_.at(i), i);
    }
}

std::size_t FlatActionCodec::index(const Game& game, const Action& action) const {
    const auto it = indices_.find(table_key(game, action));
    if (it == indices_.end()) {
        throw std::out_of_range("action is not in the flat action space");
    }
    return it->second;
}

Action FlatActionCodec::decode(const Game& game, std::size_t index) const {
    return space_.decode(index, game.current_color(), game.colors());
}

void FlatActionCodec::mask(const Game& game, std::span<std::uint8_t> out) const {
    if (out.size() != size()) {
        throw std::invalid_argument("mask buffer must cover the action space");
    }
    std::fill(out.begin(), out.end(), std::uint8_t{0});
    for (const Action& action : game.playable_actions()) {
        out[index(game, action)] = 1;
    }
}

FlatObservationEncoder::FlatObservationEncoder(int num_players, ObservationLayout layout)
    : num_players_(num_players),
      layout_(std::move(layout)),
      size_(full_observation_size(num_players, layout_)) {}

void FlatObservationEncoder::encode(const Game& game, int observer, std::span<float> out) const {
    if (static_cast<int>(game.colors().size()) != num_players_) {
        throw std::invalid_argument("the encoder was built for another player count");
    }
    if (out.size() != size_) {
        throw std::invalid_argument("observation buffer has the wrong size");
    }
    write_full_observation(game, observer, layout_, out.data(), out.size());
}

}  // namespace cppanatron
