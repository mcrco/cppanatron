#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "cppanatron/codec.hpp"
#include "cppanatron/graph_dims.hpp"

namespace cppanatron {

/**
 * The map-agnostic action space of graph networks (DESIGN 1.3):
 *
 *     [ globals G | settlement@node N | city@node N | road@edge E |
 *       robber@tile x victim-slot T*P | no-op ]
 *
 * with N, E, T, P the dims' maxima and node, edge and tile slots as in
 * CatanMap (land nodes ascending, land edges ascending, land tiles in tile
 * order). The globals are, in order:
 *
 *     0       roll
 *     1       end turn
 *     2       buy development card
 *     3       play knight
 *     4       play road building
 *     5-9     discard WOOD..ORE
 *     10-14   monopoly WOOD..ORE
 *     15-29   year of plenty, unordered pairs (a <= b) in lexicographic order
 *     30-34   year of plenty of a single card (the bank lacks a pair)
 *     35-94   maritime trade (give, get, ratio): 20 ordered pairs
 *             give != get, each at ratios 4, 3, 2
 *
 * Robber victims are seats relative to the actor: slot 0 steals from nobody,
 * slot k from the player k seats after the actor. The last index is a no-op
 * no game action maps to; environments give it to waiting seats.
 *
 * Domestic trades are outside the space: mask() ignores them and index()
 * throws for them. Games built by the environments never offer trades.
 */
class GraphActionSpace final : public ActionCodec {
public:
    static constexpr std::size_t kRoll = 0;
    static constexpr std::size_t kEndTurn = 1;
    static constexpr std::size_t kBuyDevelopmentCard = 2;
    static constexpr std::size_t kPlayKnight = 3;
    static constexpr std::size_t kPlayRoadBuilding = 4;
    static constexpr std::size_t kDiscard = 5;
    static constexpr std::size_t kMonopoly = 10;
    static constexpr std::size_t kYearOfPlenty = 15;
    static constexpr std::size_t kYearOfPlentyCount = 20;
    static constexpr std::size_t kMaritimeTrade = 35;
    static constexpr std::size_t kMaritimeTradeCount = 60;
    static constexpr std::size_t kNumGlobals = 95;

    explicit GraphActionSpace(EncoderDims dims);

    [[nodiscard]] std::size_t size() const noexcept override { return size_; }
    [[nodiscard]] std::size_t index(const Game& game, const Action& action) const override;
    [[nodiscard]] Action decode(const Game& game, std::size_t index) const override;
    void mask(const Game& game, std::span<std::uint8_t> out) const override;

    [[nodiscard]] const EncoderDims& dims() const noexcept { return dims_; }
    [[nodiscard]] std::size_t settlement_offset() const noexcept { return kNumGlobals; }
    [[nodiscard]] std::size_t city_offset() const noexcept {
        return settlement_offset() + static_cast<std::size_t>(dims_.max_nodes);
    }
    [[nodiscard]] std::size_t road_offset() const noexcept {
        return city_offset() + static_cast<std::size_t>(dims_.max_nodes);
    }
    [[nodiscard]] std::size_t robber_offset() const noexcept {
        return road_offset() + static_cast<std::size_t>(dims_.max_edges);
    }
    [[nodiscard]] std::size_t noop() const noexcept { return size_ - 1; }

    /** A readable name for an index, e.g. "ROAD@12" or "MARITIME_WOOD_ORE_4". */
    [[nodiscard]] std::string name(std::size_t index) const;

private:
    EncoderDims dims_;
    std::size_t size_{};
};

}  // namespace cppanatron
