#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "cppanatron/action.hpp"
#include "cppanatron/map.hpp"

namespace cppanatron {

[[nodiscard]] std::string flat_action_key(const Action& action);

class FlatActionSpace {
public:
    FlatActionSpace(int num_players, MapType map_type);

    [[nodiscard]] std::size_t size() const noexcept { return actions_.size(); }
    [[nodiscard]] const std::vector<Action>& actions() const noexcept { return actions_; }
    [[nodiscard]] const Action& at(std::size_t index) const { return actions_.at(index); }
    [[nodiscard]] std::size_t index(const Action& action) const;
    [[nodiscard]] std::size_t index(
        const Action& action,
        const std::vector<Color>& game_colors) const;
    [[nodiscard]] Action decode(
        std::size_t index,
        Color actor,
        const std::vector<Color>& game_colors) const;

private:
    std::vector<Action> actions_;
};

}  // namespace cppanatron
