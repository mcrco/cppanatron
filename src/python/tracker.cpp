#include "bindings.hpp"

#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "cppanatron/tracker.hpp"

namespace cppanatron::python {

void bind_tracker(nb::module_& module) {
    nb::enum_<TrackerMode>(
        module, "TrackerMode",
        "How a HandTracker represents the resource cards its observer cannot see.")
        .value(
            "POOL", TrackerMode::pool,
            "Known lower bounds per seat and one pool of unknown cards, dealt uniformly.")
        .value(
            "EXACT", TrackerMode::exact,
            "Plus the exact posterior over every seat's hand: weighted hypotheses\n"
            "branched at hidden steals and pruned by every later event, up to\n"
            "``max_hypotheses``, above which it falls back to the pool.");

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
            "holds enough victory point cards to have won.")
        .value(
            "VICTORY_POINT_CARDS", TrackerEventType::victory_point_cards,
            "``player`` holds between ``amounts[0]`` and ``amounts[1]`` victory point\n"
            "cards (a bound learned elsewhere: no engine transition or option implies one).");

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
        .def_static(
            "victory_point_cards", &TrackerEvent::victory_point_cards, nb::arg("player"),
            nb::arg("at_least"), nb::arg("at_most"))
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
        "hands always satisfy its constraints (``consistent_with``).\n\n"
        "``TrackerMode.POOL`` (the default) forgets which card a hidden steal took.\n"
        "``TrackerMode.EXACT`` also keeps the exact posterior over the hands: a\n"
        "weighted set of hypotheses (every seat's exact hand), branched at hidden\n"
        "steals by the victim's counts of each kind and pruned by every later\n"
        "event, merged when identical. Above ``max_hypotheses`` the belief is\n"
        "dropped (``belief_overflows``) and ``sample_world`` deals the pool until\n"
        "every hand is known again.")
        .def(
            nb::init<const Game&, int, TrackerMode, int>(), nb::arg("game"), nb::arg("observer"),
            nb::arg("mode") = TrackerMode::pool, nb::kw_only(),
            nb::arg("max_hypotheses") = kDefaultMaxHypotheses,
            "Start tracking for seat ``observer``. Reads public state and the\n"
            "observer's own hand only; other seats' cards start unknown, which is\n"
            "exact for a fresh game (an exact tracker started mid-game enumerates\n"
            "the pool's deals when at most ``max_hypotheses`` fit).")
        .def("copy", [](const HandTracker& tracker) { return HandTracker(tracker); })
        .def(
            "__eq__", [](const HandTracker& lhs, const HandTracker& rhs) { return lhs == rhs; },
            nb::is_operator())
        .def("__copy__", [](const HandTracker& tracker) { return HandTracker(tracker); })
        .def(
            "__deepcopy__", [](const HandTracker& tracker, nb::handle) { return HandTracker(tracker); },
            nb::arg("memo"))
        .def(nb::self == nb::self)
        .def_prop_ro("observer", &HandTracker::observer)
        .def_prop_ro("num_players", &HandTracker::num_players)
        .def_prop_ro("mode", &HandTracker::mode)
        .def_prop_ro("max_hypotheses", &HandTracker::max_hypotheses)
        .def_prop_ro(
            "exact", &HandTracker::exact,
            "Whether the exact belief currently enumerates every possible world\n"
            "(false in POOL mode and while the belief has fallen back to the pool).")
        .def_prop_ro("num_hypotheses", &HandTracker::hypothesis_count)
        .def_prop_ro(
            "belief_overflows", &HandTracker::belief_overflows,
            "How often the exact belief outgrew ``max_hypotheses`` and fell back to the pool.")
        .def_prop_ro(
            "hypotheses",
            [](const HandTracker& tracker) {
                std::vector<std::pair<double, std::vector<std::array<int, 5>>>> out;
                for (std::size_t h = 0; h < tracker.hypothesis_count(); ++h) {
                    const auto hands = tracker.hypothesis_hands(h);
                    out.emplace_back(tracker.hypothesis_weight(h),
                                     std::vector<std::array<int, 5>>(hands.begin(), hands.end()));
                }
                return out;
            },
            "The exact belief's ``(probability, hands per seat)`` pairs; empty unless ``exact``.")
        .def(
            "observe", &HandTracker::observe, nb::arg("before"), nb::arg("action"), nb::arg("after"),
            "Apply the events the observer sees when ``action`` takes ``before`` to\n"
            "``after`` (true states; hidden changes are not revealed).")
        .def_static(
            "events", &HandTracker::events, nb::arg("before"), nb::arg("action"), nb::arg("after"),
            nb::arg("observer"), "The events ``observe`` would apply for ``observer``.")
        .def_static(
            "option_events", &HandTracker::option_events, nb::arg("game"), nb::arg("legal"),
            nb::arg("observer"),
            "What the observer's legal actions ``legal`` reveal about hidden cards where\n"
            "it is to move in ``game``: nothing, since legal actions follow public state\n"
            "and the mover's own cards (the friendly robber reads public points, OpenHex's\n"
            "rule). Kept as the hook for a rule whose options would.")
        .def(
            "observe_options", &HandTracker::observe_options, nb::arg("game"), nb::arg("legal"),
            "Apply ``option_events(game, legal, observer)``; ``observe`` does this itself.\n"
            "Pass the legal actions the server offered when ``game`` is a rebuilt view.")
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
            "victory_point_card_floor", &HandTracker::victory_point_card_floor, nb::arg("seat"),
            "The fewest victory point cards ``seat`` can hold (they are never played).")
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
        [](const Game& game, const HandTracker& tracker, std::uint64_t seed,
           std::optional<TrackerMode> deal) {
            std::mt19937_64 rng(seed);
            return sample_world(game, tracker, rng, deal);
        },
        nb::arg("game"), nb::arg("tracker"), nb::arg("seed") = 0, nb::arg("deal") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        "A full game consistent with everything the tracker's observer knows.\n\n"
        "``game`` is the true game or any game with the same public state; its\n"
        "hidden cards are never read. Other seats' resource hands are drawn as\n"
        "``deal`` says (the tracker's own mode by default): ``TrackerMode.POOL``\n"
        "deals the unknown pool uniformly on top of the known cards, which is the\n"
        "exact posterior if unknown cards were equally likely to be anywhere and\n"
        "ignores which card a hidden steal likely took; ``TrackerMode.EXACT`` (an\n"
        "exact tracker only) draws one hypothesis of the exact belief by its\n"
        "posterior, dealing the pool only while the belief has fallen back. Their\n"
        "development cards are a uniform deal of the unseen cards, conditioned on\n"
        "nobody holding enough victory points to have already won; the rest is the\n"
        "deck, shuffled, and the game's random stream is reseeded. Neither deal\n"
        "uses what seats chose not to do. Raises ValueError for a finished game.");
}

}  // namespace cppanatron::python
