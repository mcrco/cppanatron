#pragma once

#include <cstddef>
#include <map>
#include <utility>
#include <vector>

#include "cppanatron/game.hpp"

namespace cppanatron {

struct NodePosition {
    int node{};
    int x{};
    int y{};
};

struct EdgePosition {
    Edge edge{};
    int x{};
    int y{};
};

struct TilePosition {
    Coordinate coordinate{};
    int x{};
    int y{};
};

class ObservationLayout {
public:
    ObservationLayout(
        int width,
        int height,
        std::vector<NodePosition> node_positions,
        std::vector<EdgePosition> edge_positions,
        std::vector<TilePosition> tile_positions);

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] const std::pair<int, int>& node_position(int node) const;
    [[nodiscard]] const std::pair<int, int>& edge_position(Edge edge) const;
    [[nodiscard]] const std::pair<int, int>& tile_position(
        Coordinate coordinate) const;

private:
    int width_{};
    int height_{};
    std::map<int, std::pair<int, int>> node_positions_;
    std::map<Edge, std::pair<int, int>> edge_positions_;
    std::map<Coordinate, std::pair<int, int>> tile_positions_;
};

[[nodiscard]] std::size_t full_numeric_observation_size(int num_players);
[[nodiscard]] std::size_t board_observation_size(
    int num_players,
    const ObservationLayout& layout);
[[nodiscard]] std::size_t full_observation_size(
    int num_players,
    const ObservationLayout& layout);

void write_full_observation(
    const Game& game,
    int base_player,
    const ObservationLayout& layout,
    float* output,
    std::size_t output_size);

[[nodiscard]] double production_sum(const Game& game, Color color);

}  // namespace cppanatron
