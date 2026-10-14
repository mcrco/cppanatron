#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>

namespace cppanatron {

enum class Color : std::uint8_t { red, blue, white, orange, green, brown };
enum class Resource : std::uint8_t { wood, brick, sheep, wheat, ore };
enum class DevelopmentCard : std::uint8_t {
    knight,
    year_of_plenty,
    monopoly,
    road_building,
    victory_point,
};
enum class Building : std::uint8_t { settlement, city };
enum class MapType : std::uint8_t { base, mini, tournament };
enum class NumberPlacement : std::uint8_t { official_spiral, random };

inline constexpr std::array<Resource, 5> kResources{
    Resource::wood,
    Resource::brick,
    Resource::sheep,
    Resource::wheat,
    Resource::ore,
};

/** Seating order for games of N players: the first N colors. */
inline constexpr std::array<Color, 6> kColors{
    Color::red,
    Color::blue,
    Color::white,
    Color::orange,
    Color::green,
    Color::brown,
};

struct Coordinate {
    int x{};
    int y{};
    int z{};

    auto operator<=>(const Coordinate&) const = default;
};

struct Edge {
    int a{};
    int b{};

    constexpr Edge normalized() const noexcept {
        return a <= b ? *this : Edge{b, a};
    }

    auto operator<=>(const Edge&) const = default;
};

inline constexpr Coordinate operator+(Coordinate lhs, Coordinate rhs) noexcept {
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

constexpr std::string_view to_string(Resource value) noexcept {
    switch (value) {
        case Resource::wood:
            return "WOOD";
        case Resource::brick:
            return "BRICK";
        case Resource::sheep:
            return "SHEEP";
        case Resource::wheat:
            return "WHEAT";
        case Resource::ore:
            return "ORE";
    }
    return "";
}

}  // namespace cppanatron
