#pragma once

#include <memory>
#include <vector>

#include "cppanatron/game.hpp"

namespace cppanatron {

/**
 * Maxima a graph encoder or action space is sized for. Every block of a row
 * (players, tiles, nodes, edges) is padded to these, so one row shape and one
 * action space cover mixed maps and player counts. Nodes, edges and tiles
 * count land only (see CatanMap's slots).
 */
struct EncoderDims {
    int max_players{6};
    int max_tiles{19};
    int max_nodes{54};
    int max_edges{72};

    auto operator<=>(const EncoderDims&) const = default;

    /** Throws std::invalid_argument for non-positive maxima or more than six seats. */
    void validate() const;
    /** Whether a game on `map` with `num_players` seats fits. */
    [[nodiscard]] bool fits(const CatanMap& map, int num_players) const noexcept;
    /** Throws std::invalid_argument naming the first dimension `game` exceeds. */
    void require_fits(const Game& game) const;

    /** The smallest dims covering every map for up to `max_players` seats. */
    [[nodiscard]] static EncoderDims covering(
        const std::vector<std::shared_ptr<const CatanMap>>& maps, int max_players);
};

}  // namespace cppanatron
