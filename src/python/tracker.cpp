#include "bindings.hpp"

#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "cppanatron/tracker.hpp"

namespace cppanatron::python {

void bind_tracker(nb::module_& module) {
    nb::enum_<TrackerEventType>(module, "TrackerEventType", "What a TrackerEvent records.")
        .value("GAIN", TrackerEventType::gain, "``player`` takes ``cards`` from the bank.")
        .value("LOSE", TrackerEventType::lose, "``player`` returns ``cards`` to the bank.")
        .value("TRANSFER", TrackerEventType::transfer, "``cards`` move from ``other`` to ``player``.")
        .value(
            "STEAL", TrackerEventType::steal,
            "The robber moves one card from ``other`` (victim) to ``player`` (thief);\n"
            "``resource`` only when the observer is one of them.")
        .value(
            "MONOPOLY", TrackerEventType::monopoly,
            "``player`` takes every ``resource``: ``amounts[seat]`` from each seat.")
        .value(
            "BUY_DEVELOPMENT_CARD", TrackerEventType::buy_development_card,
            "``player`` buys a card; ``development_card`` only for the observer.")
        .value(
            "PLAY_DEVELOPMENT_CARD", TrackerEventType::play_development_card,
            "``player`` plays ``development_card``.")
        .value("END_TURN", TrackerEventType::end_turn, "``player`` ends its turn.")
        .value(
            "PUBLIC_VICTORY_POINTS", TrackerEventType::public_victory_points,
            "The game goes on with public victory points ``amounts[seat]``: nobody\n"
            "holds enough victory point cards to have won.");

    nb::class_<TrackerEvent>(
        module, "TrackerEvent",
        "One public step of a game as one observer sees it. Seats are indices into\n"
        "``game.colors``; build events with the static constructors. Events carry\n"
        "no payments: a purchase is LOSE (the cost) plus BUY_DEVELOPMENT_CARD.")
        .def_ro("type", &TrackerEvent::type)
        .def_ro("player", &TrackerEvent::player)
        .def_ro("other", &TrackerEvent::other)
        .def_prop_ro("cards", [](const TrackerEvent& event) { return to_tuple(event.cards); })
        .def_ro("resource", &TrackerEvent::resource)
        .def_ro("development_card", &TrackerEvent::development_card)
        .def_ro("amounts", &TrackerEvent::amounts)
        .def_static("gain", &TrackerEvent::gain, nb::arg("player"), nb::arg("cards"))
        .def_static("lose", &TrackerEvent::lose, nb::arg("player"), nb::arg("cards"))
        .def_static(
            "transfer", &TrackerEvent::transfer, nb::arg("source"), nb::arg("target"), nb::arg("cards"))
        .def_static(
            "steal", &TrackerEvent::steal, nb::arg("thief"), nb::arg("victim"),
            nb::arg("resource") = nb::none())
        .def_static(
            "monopoly", &TrackerEvent::monopoly, nb::arg("player"), nb::arg("resource"),
            nb::arg("amounts"))
        .def_static(
            "buy_development_card", &TrackerEvent::buy_development_card, nb::arg("player"),
            nb::arg("card") = nb::none())
        .def_static(
            "play_development_card", &TrackerEvent::play_development_card, nb::arg("player"),
            nb::arg("card"))
        .def_static("end_turn", &TrackerEvent::end_turn, nb::arg("player"))
        .def_static("public_victory_points", &TrackerEvent::public_victory_points, nb::arg("points"))
        .def(nb::self == nb::self)
        .def("__repr__", [](const TrackerEvent& event) {
            return nb::str("TrackerEvent({!r}, player={}, other={}, cards={!r}, resource={!r}, "
                           "development_card={!r}, amounts={!r})")
                .format(
                    event.type, event.player, event.other, to_tuple(event.cards), event.resource,
                    event.development_card, event.amounts);
        });

    nb::class_<HandTracker>(
        module, "HandTracker",
        "Follows a game from one seat using only what that seat sees.\n\n"
        "Per seat: the resource hand size, a known lower bound per resource (exact\n"
        "for the observer) and the unknown cards beyond it; the cards in all hands\n"
        "are known per resource from the bank, so the unknown cards together form\n"
        "a known pool. Development cards: counts and played cards are public; the\n"
        "unseen cards are the starting deck minus the observer's hand and every\n"
        "played card. Cards become unknown only when tracking starts mid-game and\n"
        "when the robber moves a card between two other seats.\n\n"
        "Drive it with ``observe(before, action, after)`` when you have true states,\n"
        "or with the event methods when you only see views. It is sound: the true\n"
        "hands always satisfy its constraints (``consistent_with``).")
        .def(
            nb::init<const Game&, int>(), nb::arg("game"), nb::arg("observer"),
            "Start tracking for seat ``observer``. Reads public state and the\n"
            "observer's own hand only; other seats' cards start unknown, which is\n"
            "exact for a fresh game.")
        .def("copy", [](const HandTracker& tracker) { return HandTracker(tracker); })
        .def("__copy__", [](const HandTracker& tracker) { return HandTracker(tracker); })
        .def(
            "__deepcopy__", [](const HandTracker& tracker, nb::handle) { return HandTracker(tracker); },
            nb::arg("memo"))
        .def(nb::self == nb::self)
        .def_prop_ro("observer", &HandTracker::observer)
        .def_prop_ro("num_players", &HandTracker::num_players)
        .def(
            "observe", &HandTracker::observe, nb::arg("before"), nb::arg("action"), nb::arg("after"),
            "Apply the events the observer sees when ``action`` takes ``before`` to\n"
            "``after`` (true states; hidden changes are not revealed).")
        .def_static(
            "events", &HandTracker::events, nb::arg("before"), nb::arg("action"), nb::arg("after"),
            nb::arg("observer"), "The events ``observe`` would apply for ``observer``.")
        .def(
            "apply", &HandTracker::apply, nb::arg("event"),
            "Apply one event. Impossible events raise and change nothing.")
        .def("gain", &HandTracker::gain, nb::arg("player"), nb::arg("cards"),
             "``player`` takes ``cards`` (WOOD, BRICK, SHEEP, WHEAT, ORE counts) from the bank.")
        .def("lose", &HandTracker::lose, nb::arg("player"), nb::arg("cards"),
             "``player`` returns ``cards`` to the bank (builds, discards, trades).")
        .def(
            "transfer",
            [](HandTracker& tracker, int source, int target, const std::array<int, 5>& cards) {
                tracker.transfer(source, target, cards);
            },
            nb::arg("source"), nb::arg("target"), nb::arg("cards"),
            "``cards`` move openly from ``source`` to ``target`` (domestic trades).")
        .def(
            "steal", &HandTracker::steal, nb::arg("thief"), nb::arg("victim"),
            nb::arg("resource") = nb::none(),
            "A robber steal. Leave ``resource`` unset when the observer did not see\n"
            "it (neither thief nor victim): the thief gains an unknown card and the\n"
            "victim's known bounds drop by one each.")
        .def(
            "monopoly", &HandTracker::monopoly, nb::arg("player"), nb::arg("resource"),
            nb::arg("amounts"),
            "``player`` took every ``resource``; ``amounts[seat]`` came from each seat.")
        .def(
            "buy_development_card", &HandTracker::buy_development_card, nb::arg("player"),
            nb::arg("card") = nb::none(),
            "``player`` bought a card (pay with ``lose``); the card only for the observer.")
        .def(
            "play_development_card", &HandTracker::play_development_card, nb::arg("player"),
            nb::arg("card"))
        .def("end_turn", &HandTracker::end_turn, nb::arg("player"))
        .def(
            "public_victory_points", &HandTracker::public_victory_points, nb::arg("points"),
            "The game goes on with these public victory points per seat. Send it\n"
            "after every step (``observe`` does): victory point cards are capped by\n"
            "the fewest points a seat could have held without having won.")
        .def("hand_size", &HandTracker::hand_size, nb::arg("seat"))
        .def(
            "known_resources",
            [](const HandTracker& tracker, int seat) { return to_tuple(tracker.known_resources(seat)); },
            nb::arg("seat"), "Lower bound per resource; the exact hand for the observer.")
        .def("unknown_count", &HandTracker::unknown_count, nb::arg("seat"))
        .def_prop_ro(
            "resource_totals",
            [](const HandTracker& tracker) { return to_tuple(tracker.resource_totals()); },
            "Cards per resource in all hands together.")
        .def_prop_ro(
            "unknown_pool", [](const HandTracker& tracker) { return to_tuple(tracker.unknown_pool()); },
            "Cards per resource in the unknown parts of all hands.")
        .def("development_card_count", &HandTracker::development_card_count, nb::arg("seat"))
        .def_prop_ro(
            "own_development_cards",
            [](const HandTracker& tracker) { return to_tuple(tracker.own_development_cards()); })
        .def(
            "played_development_cards",
            [](const HandTracker& tracker, int seat) {
                return to_tuple(tracker.played_development_cards(seat));
            },
            nb::arg("seat"))
        .def_prop_ro(
            "unseen_development_cards",
            [](const HandTracker& tracker) { return to_tuple(tracker.unseen_development_cards()); },
            "The deck plus the other seats' unplayed cards, in DevelopmentCard order.")
        .def_prop_ro("development_deck_size", &HandTracker::development_deck_size)
        .def("bought_this_turn", &HandTracker::bought_this_turn, nb::arg("seat"))
        .def(
            "victory_point_card_cap", &HandTracker::victory_point_card_cap, nb::arg("seat"),
            "The most victory point cards ``seat`` can hold, given the game's history.")
        .def(
            "expected_resources",
            [](const HandTracker& tracker, int seat) { return to_tuple(tracker.expected_resources(seat)); },
            nb::arg("seat"),
            "Known cards plus the seat's share of the unknown pool (the mean of\n"
            "``sample_world``'s deal).")
        .def(
            "expected_development_cards",
            [](const HandTracker& tracker, int seat) {
                return to_tuple(tracker.expected_development_cards(seat));
            },
            nb::arg("seat"),
            "The seat's share of the unseen development cards (ignores the cap on\n"
            "victory point cards).")
        .def(
            "consistent_with", &HandTracker::consistent_with, nb::arg("truth"),
            "Whether ``truth`` satisfies every constraint. Reads hidden state on\n"
            "purpose; for tests and debugging.")
        .def(
            "check_public", &HandTracker::check_public, nb::arg("game"),
            "Raise ValueError if ``game``'s public state disagrees with the tracker.");

    module.def(
        "sample_world",
        [](const Game& game, const HandTracker& tracker, std::uint64_t seed) {
            std::mt19937_64 rng(seed);
            return sample_world(game, tracker, rng);
        },
        nb::arg("game"), nb::arg("tracker"), nb::arg("seed") = 0,
        nb::call_guard<nb::gil_scoped_release>(),
        "A full game consistent with everything the tracker's observer knows.\n\n"
        "``game`` is the true game or any game with the same public state; its\n"
        "hidden cards are never read. Other seats' resource hands are their known\n"
        "cards plus a uniformly random deal of the unknown pool; their development\n"
        "cards are a uniform deal of the unseen cards, conditioned on nobody holding\n"
        "enough victory points to have already won; the rest is the deck, shuffled,\n"
        "and the game's random stream is reseeded. This is the exact posterior if\n"
        "unknown cards were equally likely to be anywhere; it ignores evidence the\n"
        "tracker drops (which card a hidden steal likely took, what seats chose not\n"
        "to do).");
}

}  // namespace cppanatron::python
