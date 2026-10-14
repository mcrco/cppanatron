#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "cppanatron/codec.hpp"
#include "cppanatron/graph_action_space.hpp"
#include "cppanatron/graph_dims.hpp"

namespace cppanatron {

/**
 * What an encoded seat may see of the others (DESIGN 1.2).
 *
 * private_: what the seat sees at the table: its own hand and cards, and the
 *           others' public state (hand and development-card counts, played
 *           cards, buildings, visible victory points).
 * public_:  plus the others' exact resource hands (what a perfect card
 *           counter knows in two-player games).
 * full:     plus the others' development cards and actual victory points.
 */
enum class InfoLevel : std::uint8_t { private_, public_, full };

/**
 * Seat-relative graph observations: one float32 row per (game, seat).
 *
 * Seats are relative to the observer: player slot 0 is the observer, slot k
 * the player k seats after it. Every block is padded to the EncoderDims
 * maxima, with a validity feature per slot, so one row shape covers mixed
 * maps and player counts. Node, edge and tile slots are CatanMap's (land
 * nodes ascending, land edges ascending, land tiles in tile order). Rows are
 *
 *     [ global G | players P x Fp | tiles T x Ft | nodes N x Fn | edges E x Fe |
 *       edge_nodes E x 2 | tile_nodes T x 6 | action_mask A ]
 *
 * and layout() gives each block's offset and shape. Features are scaled to
 * roughly [0, 1] (counts by their starting supply, pips by a typical maximum)
 * in the spirit of Canopy's Nexus-v3 encoder as ported in catanrl v1. Pips
 * are a number's dice weight, 6 - |7 - number| (5 for 6 and 8).
 *
 * global (G = 23 + P):
 *   prompt_*[7]           one-hot ActionPrompt of the seat to act
 *   road_building         free roads from a road-building card are pending
 *   bank_<R>[5]           bank count / the rules' starting bank of R
 *   deck                  development cards left / the rules' deck size
 *   turn                  min(num_turns / 500, 1)
 *   vps_to_win            victory points to win / 20
 *   discard_limit         discard limit / 20
 *   friendly_robber       the friendly-robber rule is on
 *   players_<n>[P]        one-hot number of seats (n = 1..P)
 *   board_pips_<R>[5]     pips of all R tiles on the map / 30
 *
 * players (Fp = 46), for the slot's seat; all zero for absent slots:
 *   present               the slot has a seat
 *   acting                the seat must act now (also during discards)
 *   turn_owner            it is the seat's turn
 *   public_vps            visible victory points / vps_to_win
 *   actual_vps            victory points incl. VP cards / vps_to_win (own seat,
 *                         or full; else 0)
 *   resource_<R>[5]       cards of R / the starting bank of R (own seat, or
 *                         public and full; else 0)
 *   hand_size             min(resource cards / 20, 1)
 *   over_discard_limit    hand larger than the discard limit
 *   discards_left         min(cards still to discard / 10, 1)
 *   dev_hand_size         min(unplayed development cards / 10, 1)
 *   dev_<D>[5]            unplayed cards of D / the deck's count of D (own
 *                         seat, or full; else 0)
 *   dev_playable_<D>[4]   holds a D owned since the turn started (own seat,
 *                         or full; else 0), for knight, year of plenty,
 *                         monopoly, road building
 *   played_dev_this_turn  a development card was played this turn
 *   played_<D>[4]         played cards of D / the deck's count of D
 *   longest_road          holds Longest Road
 *   largest_army          holds Largest Army
 *   road_length           min(longest road length / 15, 1)
 *   roads_left            roads left / the rules' roads per player
 *   settlements_left      settlements left / settlements per player
 *   cities_left           cities left / cities per player
 *   has_rolled            the seat rolled this turn
 *   production_<R>[5]     pips of R its buildings collect (cities twice, the
 *                         robber's tile excluded) / 15
 *   port_<R>[5], port_3to1 trades at a 2:1 port of R / a 3:1 port
 *
 * tiles (Ft = 19 + P), per land tile:
 *   valid                 the slot is a tile of this map
 *   resource_<R>[5], desert  one-hot terrain
 *   number_<n>[10]        one-hot number, n in 2-6, 8-12
 *   pips                  pips / 5
 *   robber                the robber is here
 *   buildings_<k>[P]      seat k's settlements + 2 x cities on the tile / 6
 *
 * nodes (Fn = 16 + 3P), per land node:
 *   valid                 the slot is a node of this map
 *   settlement_<k>[P]     a settlement of relative seat k is here
 *   city_<k>[P]           a city of relative seat k is here
 *   port_<R>[5], port_3to1 the node trades at a port
 *   pips_<R>[5]           pips of adjacent R tiles not under the robber / 13
 *   robber                an adjacent tile holds the robber
 *   tiles                 adjacent land tiles / 3
 *   open                  the distance rule allows a settlement here
 *   buildable             the observer may build a settlement here (open
 *                         and on its road network; any open node during
 *                         the initial build phase), cost aside
 *   distance_<k>[P]       min(road hops from relative seat k's buildings and
 *                         roads / 6, 1); 1 without a network
 *
 * edges (Fe = 2 + P), per land edge:
 *   valid                 the slot is an edge of this map
 *   road_<k>[P]           a road of relative seat k is here
 *   buildable             the observer may build a road here, cost aside (while
 *                         it places an initial road: only at the settlement
 *                         it just placed)
 *
 * edge_nodes (E x 2) and tile_nodes (T x 6): node slots of each edge's ends
 * and each tile's corners (NORTH clockwise), as floats; -1 pads empty slots.
 *
 * action_mask (A = the GraphActionSpace's size): 1.0 for each playable
 * action when the observer must act; otherwise only the no-op is 1.0 (also
 * once the game is won).
 *
 * Information levels only change the seat-private features marked above,
 * which read 0 when hidden. The mask is the observer's own.
 */
class GraphEncoder final : public ObservationEncoder {
public:
    struct Block {
        std::string name;
        std::size_t offset{};
        std::vector<std::size_t> shape;
    };

    GraphEncoder(EncoderDims dims, InfoLevel level);

    [[nodiscard]] std::size_t size() const noexcept override { return size_; }
    /** Throws std::invalid_argument if the game exceeds the dims or `out` is not size(). */
    void encode(const Game& game, int observer, std::span<float> out) const override;

    [[nodiscard]] const EncoderDims& dims() const noexcept { return dims_; }
    [[nodiscard]] InfoLevel level() const noexcept { return level_; }
    [[nodiscard]] const GraphActionSpace& action_space() const noexcept { return action_space_; }
    /** Blocks in row order: global, players, tiles, nodes, edges, edge_nodes, tile_nodes,
     * action_mask. */
    [[nodiscard]] const std::vector<Block>& layout() const noexcept { return layout_; }
    /** Per-slot feature names of the global, players, tiles, nodes and edges blocks. */
    [[nodiscard]] const std::vector<std::string>& feature_names(const std::string& block) const;

    [[nodiscard]] std::size_t global_size() const noexcept { return global_names_.size(); }
    [[nodiscard]] std::size_t player_size() const noexcept { return player_names_.size(); }
    [[nodiscard]] std::size_t tile_size() const noexcept { return tile_names_.size(); }
    [[nodiscard]] std::size_t node_size() const noexcept { return node_names_.size(); }
    [[nodiscard]] std::size_t edge_size() const noexcept { return edge_names_.size(); }
    [[nodiscard]] std::size_t mask_offset() const noexcept { return layout_.back().offset; }

    /** Writes the waiting row: zeros with only the no-op set in the mask. */
    void encode_waiting(std::span<float> out) const;

    /** Size of the players block: max_players x player features. */
    [[nodiscard]] std::size_t players_size() const noexcept {
        return static_cast<std::size_t>(dims_.max_players) * player_names_.size();
    }
    /**
     * Writes only the players block (as encode would) for `observer`. The
     * information levels differ in this block alone, so with the actor's row
     * it is what a critic at another level needs.
     */
    void encode_players(const Game& game, int observer, std::span<float> out) const;

private:
    /** Everything about a map the rows need, and the row's map-only features. */
    struct MapCache;
    /** Positions of the features written per call, within their slot. */
    struct Offsets {
        std::size_t tile_robber{};
        std::size_t tile_buildings{};
        std::size_t node_settlement{};
        std::size_t node_city{};
        std::size_t node_pips{};
        std::size_t node_robber{};
        std::size_t node_open{};
        std::size_t node_buildable{};
        std::size_t node_distance{};
        std::size_t edge_road{};
        std::size_t edge_buildable{};
    };

    /** The cache for a map, built once per thread for each live map. */
    [[nodiscard]] const MapCache& map_cache(const std::shared_ptr<const CatanMap>& map) const;
    void write_players(
        const Game& game, int observer, const MapCache& cache, int robber_slot, float* out) const;

    EncoderDims dims_;
    InfoLevel level_;
    GraphActionSpace action_space_;
    std::vector<std::string> global_names_;
    std::vector<std::string> player_names_;
    std::vector<std::string> tile_names_;
    std::vector<std::string> node_names_;
    std::vector<std::string> edge_names_;
    std::vector<Block> layout_;
    Offsets offsets_;
    std::size_t size_{};
};

/**
 * Rows for asymmetric actor/critic models, laid out exactly like VecEnv rows
 * with a critic_level: the actor's GraphEncoder row (action mask included)
 * followed by the critic block, the players block encoded at critic_level
 * (`critic_players`, max_players x player features). The information levels
 * differ in the players block alone, so the actor row with that block
 * swapped in is the critic's full input.
 *
 * Used as SearchPool's encoder, leaves match training rows. With
 * determinizations, a leaf is a sampled world, so a FULL critic block shows
 * that world's sampled hidden cards: the intended behaviour of perfect-
 * information Monte Carlo, which evaluates every world as if it were true.
 */
class ActorCriticEncoder final : public ObservationEncoder {
public:
    ActorCriticEncoder(EncoderDims dims, InfoLevel level, InfoLevel critic_level);

    [[nodiscard]] std::size_t size() const noexcept override {
        return actor_.size() + critic_.players_size();
    }
    void encode(const Game& game, int observer, std::span<float> out) const override;

    [[nodiscard]] const GraphEncoder& actor() const noexcept { return actor_; }
    [[nodiscard]] const GraphEncoder& critic() const noexcept { return critic_; }
    /** The actor's blocks, then critic_players. */
    [[nodiscard]] const std::vector<GraphEncoder::Block>& layout() const noexcept {
        return layout_;
    }
    /** Zeros with only the actor's no-op set (a waiting row). */
    void encode_waiting(std::span<float> out) const;

private:
    GraphEncoder actor_;
    GraphEncoder critic_;
    std::vector<GraphEncoder::Block> layout_;
};

}  // namespace cppanatron
