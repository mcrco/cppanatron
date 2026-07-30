#include <iostream>
#include <string>

#include "cppanatron/action_space.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: cppanatron_action_dump <MINI|BASE|TOURNAMENT> <players>\n";
        return 2;
    }
    const std::string map_name = argv[1];
    const cppanatron::MapType map_type =
        map_name == "MINI"
            ? cppanatron::MapType::mini
            : (map_name == "TOURNAMENT" ? cppanatron::MapType::tournament
                                        : cppanatron::MapType::base);
    const cppanatron::FlatActionSpace action_space(std::stoi(argv[2]), map_type);
    for (std::size_t i = 0; i < action_space.size(); ++i) {
        std::cout << i << '\t' << cppanatron::flat_action_key(action_space.at(i)) << '\n';
    }
    return 0;
}
