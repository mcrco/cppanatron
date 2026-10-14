#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "cppanatron/game.hpp"

namespace cppanatron {

/**
 * Maps a game's legal actions to and from a fixed-size index space.
 *
 * Learners, search and environments only see indices; the codec is the one
 * place that knows how an index becomes an engine Action. Indices are
 * relative to the acting seat, so implementations may use seat-relative
 * victim slots and the like.
 */
class ActionCodec {
public:
    virtual ~ActionCodec() = default;

    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    /** Index of a legal action of `game`; throws std::out_of_range if it has none. */
    [[nodiscard]] virtual std::size_t index(const Game& game, const Action& action) const = 0;
    /** The engine action for `index` in `game` (for the player to move). */
    [[nodiscard]] virtual Action decode(const Game& game, std::size_t index) const = 0;
    /** Writes 1 at the index of every playable action and 0 elsewhere. */
    virtual void mask(const Game& game, std::span<std::uint8_t> out) const = 0;
};

/**
 * Writes a network input for one game, as seen by one seat.
 *
 * `observer` is a seat index into game.colors(). Implementations must only
 * read what the information level they were configured with allows.
 */
class ObservationEncoder {
public:
    virtual ~ObservationEncoder() = default;

    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    virtual void encode(const Game& game, int observer, std::span<float> out) const = 0;
};

}  // namespace cppanatron
