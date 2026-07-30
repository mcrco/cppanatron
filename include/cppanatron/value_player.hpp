#pragma once

#include "cppanatron/action.hpp"
#include "cppanatron/game.hpp"

namespace cppanatron {

struct ValueWeights {
    double public_vps{3e14};
    double production{1e8};
    double enemy_production{-1e8};
    double num_tiles{1.0};
    double reachable_production_0{0.0};
    double reachable_production_1{1e4};
    double buildable_nodes{1e3};
    double longest_road{10.0};
    double hand_synergy{1e2};
    double hand_resources{1.0};
    double discard_penalty{-5.0};
    double hand_devs{10.0};
    double army_size{10.1};
};

[[nodiscard]] double value_score(
    const Game& game,
    Color perspective,
    const ValueWeights& weights = {});
[[nodiscard]] Action value_action(
    const Game& game,
    const ValueWeights& weights = {});

}  // namespace cppanatron
