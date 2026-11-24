#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace cppanatron {

struct EvalCacheStats {
    /** Leaves the pools applied with the cache attached, roots included. */
    std::uint64_t leaves{};
    /** Of them, answered from the table. */
    std::uint64_t hits{};
    /** Of them, answered by another leaf's row of the same round. */
    std::uint64_t repeats{};
    /** Rows whose outputs were written to the table. */
    std::uint64_t stored{};
    /** Of them, written over a live entry (the table was full there). */
    std::uint64_t replaced{};
    std::uint64_t clears{};
    /** Live entries now. */
    std::uint64_t entries{};
};

/**
 * A network's outputs by observation row, for the SearchPools that share
 * the network (SearchPool::set_cache): a leaf whose row the table holds, or
 * whose row another leaf of the same round has, is not written to the batch
 * and gets the outputs of that row at apply. A row's outputs are a function
 * of its bytes, so with an evaluator whose outputs do not depend on the
 * batch the searches are the same with and without a cache.
 *
 * The key is a 128-bit hash of the row's bytes (key); two different rows
 * with one key would share outputs, which at 2^-128 per pair does not
 * happen. The table is a fixed number of entries in sets of four: a new
 * row replaces the entry of its set that was used longest ago. Memory is
 * taken as entries are written, up to bytes().
 *
 * "Used" is a stamp the caller hands in (stamps): a pool takes a run of
 * them per round, one per leaf, so what the table holds after a round is a
 * function of the rounds played and not of the order in which the worker
 * threads reached it.
 *
 * The outputs belong to one set of weights: clear() must be called whenever
 * the network changes. It starts a new generation in O(1); entries of an
 * older one are never served, and a round collected before a clear stores
 * nothing after it.
 *
 * Any number of pools may use one cache from different threads.
 */
class EvalCache {
public:
    struct Key {
        std::uint64_t hi{};
        std::uint64_t lo{};
        friend bool operator==(const Key&, const Key&) = default;
    };

    /**
     * A table of `entries` rows of outputs (rounded down to whole sets of
     * four, at least one set): action_size logits and max_players values
     * each.
     */
    EvalCache(std::size_t action_size, int max_players, std::size_t entries);
    ~EvalCache();
    EvalCache(const EvalCache&) = delete;
    EvalCache& operator=(const EvalCache&) = delete;

    [[nodiscard]] std::size_t action_size() const noexcept;
    [[nodiscard]] int max_players() const noexcept;
    /** Entries the table holds when full. */
    [[nodiscard]] std::size_t capacity() const noexcept;
    /** Memory of the full table. */
    [[nodiscard]] std::size_t bytes() const noexcept;
    /** Memory one entry takes for these output sizes. */
    [[nodiscard]] static std::size_t entry_bytes(std::size_t action_size, int max_players) noexcept;

    /** Forget every entry: the network's weights changed. */
    void clear() noexcept;
    /** Counts clears; lookups and stores name the generation they belong to. */
    [[nodiscard]] std::uint32_t generation() const noexcept;
    [[nodiscard]] EvalCacheStats stats() const noexcept;

    /** The key of a row: a 128-bit hash of its bytes. */
    [[nodiscard]] static Key key(std::span<const float> row) noexcept;
    /**
     * Take the next `count` use stamps, later than every one handed out
     * before; returns the first.
     */
    std::uint64_t stamps(std::size_t count) noexcept;
    /**
     * Copy the outputs stored for `key` into `logits` (action_size) and
     * `values` (max_players); false, with nothing written, if there are
     * none, or if `generation` is no longer the current one. The entry was
     * used at `stamp` (0: now).
     */
    bool find(
        const Key& key, std::uint32_t generation, std::span<float> logits, std::span<float> values,
        std::uint64_t stamp = 0);
    /**
     * Store a row's outputs as used at `stamp` (0: now), unless `generation`
     * is no longer the current one (the table was cleared since the row was
     * collected), or every entry of the row's set was used after `stamp`
     * (rows stored with one round's stamps: the latest four stay, in
     * whatever order they come).
     */
    void store(
        const Key& key, std::uint32_t generation, std::span<const float> logits, std::span<const float> values,
        std::uint64_t stamp = 0);
    /** Add a round's leaves to stats(). */
    void count(std::uint64_t leaves, std::uint64_t hits, std::uint64_t repeats) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cppanatron
