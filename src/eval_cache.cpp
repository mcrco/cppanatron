#include "cppanatron/eval_cache.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace cppanatron {
namespace {

constexpr std::size_t kWays = 4;
/** Locks over the sets: far more than threads, so two rarely meet. */
constexpr std::size_t kStripes = 1024;

std::uint64_t fmix(std::uint64_t k) noexcept {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

}  // namespace

struct EvalCache::Impl {
    struct Slot {
        Key key;
        /** The stamp at which the entry was last stored or served. */
        std::uint64_t used{};
        /** 0: never written. */
        std::uint32_t generation{};
    };

    std::size_t actions{};
    std::size_t players{};
    std::size_t stride{};
    std::size_t sets{};
    std::vector<Slot> slots;
    /** stride floats per slot; pages are first touched when their slot is written. */
    std::unique_ptr<float[]> outputs;
    std::unique_ptr<std::mutex[]> locks;
    std::atomic<std::uint32_t> generation{1};
    std::atomic<std::uint64_t> clock{0};
    std::atomic<std::uint64_t> leaves{0};
    std::atomic<std::uint64_t> hits{0};
    std::atomic<std::uint64_t> repeats{0};
    std::atomic<std::uint64_t> stored{0};
    std::atomic<std::uint64_t> replaced{0};
    std::atomic<std::uint64_t> clears{0};
    std::atomic<std::uint64_t> entries{0};

    [[nodiscard]] std::size_t set_of(const Key& key) const noexcept {
        // The high half picks the set; the low half is checked too on a match.
        return static_cast<std::size_t>(key.hi % sets);
    }
};

EvalCache::EvalCache(std::size_t action_size, int max_players, std::size_t entries)
    : impl_(std::make_unique<Impl>()) {
    if (action_size == 0 || max_players < 1) {
        throw std::invalid_argument("an evaluation cache needs an action size and a value width");
    }
    impl_->actions = action_size;
    impl_->players = static_cast<std::size_t>(max_players);
    impl_->stride = impl_->actions + impl_->players;
    impl_->sets = std::max<std::size_t>(1, entries / kWays);
    if (impl_->sets > std::numeric_limits<std::size_t>::max() / (kWays * impl_->stride * sizeof(float))) {
        throw std::invalid_argument("the evaluation cache is too large");
    }
    impl_->slots.resize(impl_->sets * kWays);
    impl_->outputs = std::make_unique_for_overwrite<float[]>(impl_->sets * kWays * impl_->stride);
    impl_->locks = std::make_unique<std::mutex[]>(kStripes);
}

EvalCache::~EvalCache() = default;

std::size_t EvalCache::action_size() const noexcept { return impl_->actions; }
int EvalCache::max_players() const noexcept { return static_cast<int>(impl_->players); }
std::size_t EvalCache::capacity() const noexcept { return impl_->sets * kWays; }
std::size_t EvalCache::bytes() const noexcept {
    return capacity() * entry_bytes(impl_->actions, static_cast<int>(impl_->players));
}
std::size_t EvalCache::entry_bytes(std::size_t action_size, int max_players) noexcept {
    return sizeof(Impl::Slot) + (action_size + static_cast<std::size_t>(std::max(max_players, 0))) * sizeof(float);
}

void EvalCache::clear() noexcept {
    if (++impl_->generation == 0) {
        // 2^32 clears: the generations start over, so no slot may keep one.
        for (std::size_t stripe = 0; stripe < kStripes; ++stripe) {
            impl_->locks[stripe].lock();
        }
        std::fill(impl_->slots.begin(), impl_->slots.end(), Impl::Slot{});
        impl_->generation = 1;
        for (std::size_t stripe = kStripes; stripe-- > 0;) {
            impl_->locks[stripe].unlock();
        }
    }
    impl_->entries = 0;
    ++impl_->clears;
}

std::uint32_t EvalCache::generation() const noexcept { return impl_->generation; }

EvalCacheStats EvalCache::stats() const noexcept {
    return {impl_->leaves, impl_->hits, impl_->repeats, impl_->stored,
            impl_->replaced, impl_->clears, impl_->entries};
}

/** MurmurHash3 x64 128 (Appleby, public domain) of the row's bytes, its last block zero-padded. */
EvalCache::Key EvalCache::key(std::span<const float> row) noexcept {
    constexpr std::uint64_t c1 = 0x87c37b91114253d5ULL;
    constexpr std::uint64_t c2 = 0x4cf5ad432745937fULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(row.data());
    const std::size_t length = row.size_bytes();
    std::uint64_t h1 = 0x9e3779b97f4a7c15ULL;
    std::uint64_t h2 = 0xc2b2ae3d27d4eb4fULL;
    const auto block = [&](std::uint64_t k1, std::uint64_t k2) {
        k1 *= c1;
        k1 = std::rotl(k1, 31);
        k1 *= c2;
        h1 ^= k1;
        h1 = std::rotl(h1, 27);
        h1 += h2;
        h1 = h1 * 5 + 0x52dce729;
        k2 *= c2;
        k2 = std::rotl(k2, 33);
        k2 *= c1;
        h2 ^= k2;
        h2 = std::rotl(h2, 31);
        h2 += h1;
        h2 = h2 * 5 + 0x38495ab5;
    };
    std::size_t at = 0;
    for (; at + 16 <= length; at += 16) {
        std::uint64_t k[2];
        std::memcpy(k, bytes + at, 16);
        block(k[0], k[1]);
    }
    if (at < length) {
        std::uint64_t k[2] = {0, 0};
        std::memcpy(k, bytes + at, length - at);
        block(k[0], k[1]);
    }
    h1 ^= length;
    h2 ^= length;
    h1 += h2;
    h2 += h1;
    h1 = fmix(h1);
    h2 = fmix(h2);
    h1 += h2;
    h2 += h1;
    return {h1, h2};
}

std::uint64_t EvalCache::stamps(std::size_t count) noexcept { return impl_->clock.fetch_add(count) + 1; }

bool EvalCache::find(
    const Key& key, std::uint32_t generation, std::span<float> logits, std::span<float> values,
    std::uint64_t stamp) {
    Impl& impl = *impl_;
    if (stamp == 0) {
        stamp = stamps(1);
    }
    if (logits.size() != impl.actions || values.size() != impl.players) {
        throw std::invalid_argument("find needs action_size() logits and max_players() values");
    }
    const std::size_t set = impl.set_of(key);
    const std::scoped_lock lock(impl.locks[set % kStripes]);
    // A round collected before a clear finds nothing after it.
    if (generation != impl.generation) {
        return false;
    }
    for (std::size_t slot = set * kWays; slot < (set + 1) * kWays; ++slot) {
        Impl::Slot& entry = impl.slots[slot];
        if (entry.generation == generation && entry.key == key) {
            entry.used = std::max(entry.used, stamp);
            const float* outputs = impl.outputs.get() + slot * impl.stride;
            std::copy_n(outputs, impl.actions, logits.data());
            std::copy_n(outputs + impl.actions, impl.players, values.data());
            return true;
        }
    }
    return false;
}

void EvalCache::store(
    const Key& key, std::uint32_t generation, std::span<const float> logits, std::span<const float> values,
    std::uint64_t stamp) {
    Impl& impl = *impl_;
    if (stamp == 0) {
        stamp = stamps(1);
    }
    if (logits.size() != impl.actions || values.size() != impl.players) {
        throw std::invalid_argument("store needs action_size() logits and max_players() values");
    }
    const std::size_t set = impl.set_of(key);
    const std::scoped_lock lock(impl.locks[set % kStripes]);
    // Under the lock: a clear after this leaves an entry of a past generation, never served.
    if (generation != impl.generation) {
        return;
    }
    std::size_t victim = set * kWays;
    bool live = true;
    for (std::size_t slot = set * kWays; slot < (set + 1) * kWays; ++slot) {
        const Impl::Slot& entry = impl.slots[slot];
        if (entry.generation != generation) {
            if (live) {
                victim = slot;
                live = false;
            }
        } else if (entry.key == key) {
            // Another pool stored the row since this round looked: the same outputs.
            return;
        } else if (live && entry.used < impl.slots[victim].used) {
            victim = slot;
        }
    }
    if (live && impl.slots[victim].used > stamp) {
        // A full set of rows used after this one (stored by other threads in this round).
        return;
    }
    Impl::Slot& entry = impl.slots[victim];
    entry = {key, stamp, generation};
    float* outputs = impl.outputs.get() + victim * impl.stride;
    std::copy_n(logits.data(), impl.actions, outputs);
    std::copy_n(values.data(), impl.players, outputs + impl.actions);
    ++impl.stored;
    if (live) {
        ++impl.replaced;
    } else {
        ++impl.entries;
    }
}

void EvalCache::count(std::uint64_t leaves, std::uint64_t hits, std::uint64_t repeats) noexcept {
    impl_->leaves += leaves;
    impl_->hits += hits;
    impl_->repeats += repeats;
}

}  // namespace cppanatron
