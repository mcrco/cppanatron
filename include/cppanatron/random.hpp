#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>

namespace cppanatron::random {

/**
 * Portable sampling over std::mt19937_64.
 *
 * The standard fixes mt19937_64's output exactly, but not the algorithms of
 * its distributions (uniform_int_distribution, shuffle, discrete, gamma,
 * uniform_real ...), which differ between libstdc++, libc++ and MSVC. Every
 * draw in the engine goes through these instead, so a seed gives the same
 * games, maps and searches with any standard library or compiler. The
 * floating-point samplers (gamma, gumbel) also call std::log, std::sqrt and
 * std::pow: sqrt is exact under IEEE 754, but log and pow may differ in the
 * last bit between math libraries, which rarely changes a result.
 */
using Engine = std::mt19937_64;

/**
 * Uniform integer in [0, bound), bound > 0: Lemire's nearly divisionless
 * method ("Fast Random Integer Generation in an Interval", 2019), exact (no
 * modulo bias) and usually one draw.
 */
[[nodiscard]] inline std::uint64_t below(Engine& rng, std::uint64_t bound) {
    if (bound == 0) {
        throw std::invalid_argument("random::below needs a positive bound");
    }
    __extension__ typedef unsigned __int128 Wide;
    Wide product = static_cast<Wide>(rng()) * bound;
    auto low = static_cast<std::uint64_t>(product);
    if (low < bound) {
        const std::uint64_t threshold = (0 - bound) % bound;
        while (low < threshold) {
            product = static_cast<Wide>(rng()) * bound;
            low = static_cast<std::uint64_t>(product);
        }
    }
    return static_cast<std::uint64_t>(product >> 64U);
}

/** Uniform index into a container of `count` elements (count > 0). */
[[nodiscard]] inline std::size_t index(Engine& rng, std::size_t count) {
    return static_cast<std::size_t>(below(rng, count));
}

/** Uniform integer in [low, high]. */
[[nodiscard]] inline int between(Engine& rng, int low, int high) {
    return low + static_cast<int>(below(rng, static_cast<std::uint64_t>(high - low) + 1));
}

/** Fisher-Yates from the back: element i swaps with a uniform j in [0, i]. */
template <typename RandomIt>
void shuffle(RandomIt first, RandomIt last, Engine& rng) {
    const auto count = static_cast<std::uint64_t>(last - first);
    for (std::uint64_t i = count; i > 1; --i) {
        const std::uint64_t j = below(rng, i);
        using std::swap;
        swap(first[static_cast<std::ptrdiff_t>(i - 1)], first[static_cast<std::ptrdiff_t>(j)]);
    }
}

/** Uniform double in [0, 1): the top 53 bits of one draw. */
[[nodiscard]] inline double uniform(Engine& rng) {
    return static_cast<double>(rng() >> 11U) * 0x1.0p-53;
}

/** Uniform double in (0, 1): never 0, so its logarithm is finite. */
[[nodiscard]] inline double uniform_open(Engine& rng) {
    return (static_cast<double>(rng() >> 11U) + 0.5) * 0x1.0p-53;
}

/**
 * An index drawn with probability proportional to its non-negative weight:
 * one uniform draw scaled by the total, then the first index whose running
 * sum exceeds it. Zero weights are never drawn; all-zero weights draw
 * uniformly.
 */
[[nodiscard]] inline std::size_t discrete(Engine& rng, std::span<const double> weights) {
    if (weights.empty()) {
        throw std::invalid_argument("random::discrete needs weights");
    }
    double total = 0.0;
    for (double weight : weights) {
        if (!(weight >= 0.0) || !std::isfinite(weight)) {
            throw std::invalid_argument("random::discrete weights must be finite and >= 0");
        }
        total += weight;
    }
    if (total <= 0.0) {
        return index(rng, weights.size());
    }
    const double target = uniform(rng) * total;
    double running = 0.0;
    std::size_t last_positive = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        if (weights[i] <= 0.0) {
            continue;
        }
        running += weights[i];
        last_positive = i;
        if (target < running) {
            return i;
        }
    }
    return last_positive;  // rounding left the target at the very end
}

/** Standard normal: Marsaglia's polar method (no trigonometry). */
[[nodiscard]] inline double normal(Engine& rng) {
    for (;;) {
        const double u = 2.0 * uniform(rng) - 1.0;
        const double v = 2.0 * uniform(rng) - 1.0;
        const double s = u * u + v * v;
        if (s > 0.0 && s < 1.0) {
            return u * std::sqrt(-2.0 * std::log(s) / s);
        }
    }
}

/**
 * Gamma(shape, 1): Marsaglia and Tsang's method ("A Simple Method for
 * Generating Gamma Variables", 2000), with Gamma(shape + 1) * U^(1/shape)
 * for shapes below one. Dirichlet noise is gamma draws, normalized.
 */
[[nodiscard]] inline double gamma(Engine& rng, double shape) {
    if (!(shape > 0.0) || !std::isfinite(shape)) {
        throw std::invalid_argument("random::gamma needs a positive shape");
    }
    if (shape < 1.0) {
        const double boost = std::pow(uniform_open(rng), 1.0 / shape);
        return gamma(rng, shape + 1.0) * boost;
    }
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        double x = 0.0;
        double v = 0.0;
        do {
            x = normal(rng);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        const double u = uniform_open(rng);
        if (u < 1.0 - 0.0331 * (x * x) * (x * x) ||
            std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) {
            return d * v;
        }
    }
}

/** Standard Gumbel: -log(-log(U)) with U in (0, 1). */
[[nodiscard]] inline double gumbel(Engine& rng) {
    return -std::log(-std::log(uniform_open(rng)));
}

}  // namespace cppanatron::random
