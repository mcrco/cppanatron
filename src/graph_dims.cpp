#include "cppanatron/graph_dims.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace cppanatron {

void EncoderDims::validate() const {
    if (max_players < 1 || max_players > static_cast<int>(kColors.size())) {
        throw std::invalid_argument("max_players must be between one and six");
    }
    if (max_tiles < 1 || max_nodes < 1 || max_edges < 1) {
        throw std::invalid_argument("max_tiles, max_nodes and max_edges must be positive");
    }
}

bool EncoderDims::fits(const CatanMap& map, int num_players) const noexcept {
    return num_players <= max_players &&
           static_cast<int>(map.land_tile_indices().size()) <= max_tiles &&
           map.num_nodes() <= max_nodes &&
           static_cast<int>(map.land_edges().size()) <= max_edges;
}

void EncoderDims::require_fits(const Game& game) const {
    const CatanMap& map = game.board().map();
    const auto check = [](int value, int maximum, const char* what) {
        if (value > maximum) {
            throw std::invalid_argument(
                "game has " + std::to_string(value) + " " + what + ", more than the " +
                std::to_string(maximum) + " these dims allow");
        }
    };
    check(static_cast<int>(game.colors().size()), max_players, "players");
    check(static_cast<int>(map.land_tile_indices().size()), max_tiles, "land tiles");
    check(map.num_nodes(), max_nodes, "land nodes");
    check(static_cast<int>(map.land_edges().size()), max_edges, "land edges");
}

EncoderDims EncoderDims::covering(
    const std::vector<std::shared_ptr<const CatanMap>>& maps, int max_players) {
    EncoderDims result{max_players, 1, 1, 1};
    for (const auto& map : maps) {
        if (map == nullptr) {
            throw std::invalid_argument("maps must not be null");
        }
        result.max_tiles =
            std::max(result.max_tiles, static_cast<int>(map->land_tile_indices().size()));
        result.max_nodes = std::max(result.max_nodes, map->num_nodes());
        result.max_edges = std::max(result.max_edges, static_cast<int>(map->land_edges().size()));
    }
    result.validate();
    return result;
}

}  // namespace cppanatron
