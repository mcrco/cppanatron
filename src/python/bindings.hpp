#pragma once

// Python headers must precede the standard library ones.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "cppanatron/action.hpp"
#include "cppanatron/game.hpp"

namespace cppanatron::python {

namespace nb = nanobind;

void bind_types(nb::module_& module);
void bind_maps(nb::module_& module);
void bind_graph(nb::module_& module);
void bind_game(nb::module_& module);
void bind_observation(nb::module_& module);
void bind_search(nb::module_& module);

/** Python view of an action value; see the Action docstring for the shapes. */
[[nodiscard]] nb::object action_value_to_python(ActionType type, const ActionValue& value);
[[nodiscard]] ActionValue action_value_from_python(ActionType type, nb::handle value);

/**
 * The playable action a caller means by `action`.
 *
 * Confirmations do not name the offering seat, which the game derives, so they
 * match on everything else.
 */
[[nodiscard]] const Action& resolve_playable(const Game& game, const Action& action);

/**
 * Rules for a game or batch: ``rules`` when given, else the base game with any
 * of the loose ``discard_limit``/``friendly_robber``/``vps_to_win`` arguments.
 * Passing both is an error, since one would silently override the other.
 */
[[nodiscard]] GameRules resolve_rules(
    const std::optional<GameRules>& rules,
    std::optional<int> discard_limit,
    std::optional<bool> friendly_robber,
    std::optional<int> vps_to_win);

[[nodiscard]] nb::tuple coordinate_to_python(Coordinate coordinate);
[[nodiscard]] Coordinate coordinate_from_python(nb::handle value);

template <typename T, std::size_t N>
[[nodiscard]] nb::tuple to_tuple(const std::array<T, N>& values) {
    nb::list items;
    for (const T& value : values) {
        items.append(nb::cast(value));
    }
    return nb::tuple(items);
}

template <typename T>
using Vector = nb::ndarray<nb::numpy, T, nb::ndim<1>>;

/** Hand a freshly computed buffer to NumPy without copying it again. */
template <typename T>
[[nodiscard]] Vector<T> to_numpy(std::vector<T> values) {
    auto* owned = new std::vector<T>(std::move(values));
    nb::capsule deleter(owned, [](void* pointer) noexcept {
        delete static_cast<std::vector<T>*>(pointer);
    });
    return Vector<T>(owned->data(), {owned->size()}, deleter);
}

template <typename T>
using InputVector = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/**
 * A caller buffer the engine writes into. Bind these arguments with
 * ``.noconvert()``: a converted temporary would silently swallow the writes.
 */
template <typename T>
using OutputVector = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

}  // namespace cppanatron::python
