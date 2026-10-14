#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>

#include "cppanatron/action_space.hpp"
#include "cppanatron/codec.hpp"
#include "cppanatron/observation.hpp"

namespace cppanatron {

/**
 * The v1 flat action table (FlatActionSpace) behind the ActionCodec
 * interface. A stand-in for search and tests until the map-agnostic graph
 * action space lands; it covers one map type and up to six players.
 */
class FlatActionCodec final : public ActionCodec {
public:
    FlatActionCodec(int num_players, MapType map_type);

    [[nodiscard]] std::size_t size() const noexcept override { return space_.size(); }
    [[nodiscard]] std::size_t index(const Game& game, const Action& action) const override;
    [[nodiscard]] Action decode(const Game& game, std::size_t index) const override;
    void mask(const Game& game, std::span<std::uint8_t> out) const override;

    [[nodiscard]] const FlatActionSpace& space() const noexcept { return space_; }

private:
    FlatActionSpace space_;
    std::map<Action, std::size_t> indices_;
};

/**
 * The v1 full-information observation (write_full_observation) behind the
 * ObservationEncoder interface: perfect information, no action mask. A
 * stand-in until the graph encoder lands.
 */
class FlatObservationEncoder final : public ObservationEncoder {
public:
    FlatObservationEncoder(int num_players, ObservationLayout layout);

    [[nodiscard]] std::size_t size() const noexcept override { return size_; }
    void encode(const Game& game, int observer, std::span<float> out) const override;

private:
    int num_players_;
    ObservationLayout layout_;
    std::size_t size_;
};

}  // namespace cppanatron
