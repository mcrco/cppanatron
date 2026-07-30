#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

#include "cppanatron/action_space.hpp"
#include "cppanatron/game.hpp"

int main() {
    using namespace cppanatron;
    Game game({Color::red, Color::blue}, MapType::tournament, 42);
    const FlatActionSpace action_space(2, MapType::tournament);
    std::cout << "actions";
    while (game.is_initial_build_phase()) {
        const Action* selected = nullptr;
        std::size_t selected_index = std::numeric_limits<std::size_t>::max();
        for (const Action& action : game.playable_actions()) {
            const std::size_t index = action_space.index(action);
            if (index < selected_index) {
                selected = &action;
                selected_index = index;
            }
        }
        std::cout << ' ' << selected_index;
        game.execute(*selected);
    }
    std::cout << '\n';
    std::cout << "state " << game.current_player_index() << ' '
              << static_cast<int>(game.current_prompt()) << ' ' << game.num_turns()
              << '\n';
    for (std::size_t i = 0; i < game.colors().size(); ++i) {
        const PlayerState& player = game.players()[i];
        std::cout << "player " << i << ' ' << player.victory_points << ' '
                  << player.actual_victory_points << ' ' << player.roads_available << ' '
                  << player.settlements_available << " resources";
        for (int count : player.resources) {
            std::cout << ' ' << count;
        }
        std::cout << " settlements";
        for (int node : player.settlements) {
            std::cout << ' ' << node;
        }
        std::cout << " roads";
        for (Edge edge : player.roads) {
            std::cout << ' ' << edge.a << '-' << edge.b;
        }
        std::cout << '\n';
    }
    return 0;
}
