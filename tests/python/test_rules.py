"""Catan rules, written once and asserted identically on both engines.

Every test takes the ``engine`` fixture, so it runs on Catanatron and on
cppanatron. Resource tuples are ordered WOOD, BRICK, SHEEP, WHEAT, ORE and
development-card tuples KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING,
VICTORY_POINT.
"""

from __future__ import annotations

import pytest
from catanatron.models.enums import Action, ActionPrompt, ActionType

from conftest import BLUE, RED, THREE_PLAYER_SETUP, TWO_PLAYER_SETUP, WHITE, next_turn, setup_game
from cppanatron.parity import new_game

TILE_WHEAT_5 = (1, 0, -1)  # Touches RED's nodes 1 and 20.
TILE_WOOD_3 = (0, -1, 1)  # Touches BLUE's node 3.


def hand_size(game, color) -> int:
    return sum(game.player(color).resources)


def robber_targets(game) -> dict[tuple, set]:
    targets: dict[tuple, set] = {}
    for action in game.playable_actions:
        if action.action_type == ActionType.MOVE_ROBBER:
            coordinate, victim = action.value
            targets.setdefault(coordinate, set()).add(victim)
    return targets


def move_robber_without_victim(game) -> None:
    game.execute(
        next(
            action
            for action in game.playable_actions
            if action.action_type == ActionType.MOVE_ROBBER and action.value[1] is None
        )
    )


# --- Setup ---------------------------------------------------------------------


def test_setup_follows_snake_order_and_pays_the_second_settlement(engine):
    game = new_game(engine, [RED, BLUE], map_type="TOURNAMENT")
    order = []
    for node, road in TWO_PLAYER_SETUP:
        order.append(game.current_color)
        assert game.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT
        game.play(ActionType.BUILD_SETTLEMENT, node)
        assert game.current_prompt == ActionPrompt.BUILD_INITIAL_ROAD
        game.play(ActionType.BUILD_ROAD, road)

    assert order == [RED, BLUE, BLUE, RED]
    assert game.current_color == RED
    assert [action.action_type for action in game.playable_actions] == [ActionType.ROLL]
    # One card per tile touching each second settlement.
    assert game.player(RED).resources == (0, 0, 1, 1, 1)
    assert game.player(BLUE).resources == (0, 1, 1, 0, 1)
    assert game.player(RED).victory_points == game.player(BLUE).victory_points == 2


def test_initial_road_must_touch_the_new_settlement(engine):
    game = new_game(engine, [RED, BLUE], map_type="TOURNAMENT")
    game.play(ActionType.BUILD_SETTLEMENT, 1)
    roads = {action.value for action in game.playable_actions}
    assert roads == {(0, 1), (1, 2), (1, 6)}


def test_settlements_keep_their_distance(engine):
    game = new_game(engine, [RED, BLUE], map_type="TOURNAMENT")
    game.play(ActionType.BUILD_SETTLEMENT, 1)
    game.play(ActionType.BUILD_ROAD, (1, 6))
    nodes = {action.value for action in game.playable_actions}
    assert {0, 1, 2, 6}.isdisjoint(nodes)
    assert 3 in nodes


# --- Production and the robber ---------------------------------------------------


def test_rolls_pay_every_touching_building(engine):
    game = setup_game(engine)
    bank = game.snapshot().resource_bank
    game.play(ActionType.ROLL, outcome=(2, 3))  # WHEAT 5 touches both RED settlements.
    assert game.player(RED).resources == (0, 0, 1, 3, 1)
    assert game.player(BLUE).resources == (0, 1, 1, 0, 1)
    assert game.snapshot().resource_bank == (bank[0], bank[1], bank[2], bank[3] - 2, bank[4])


def test_the_robber_blocks_its_tile(engine):
    game = setup_game(engine)
    game.play(ActionType.ROLL, outcome=(3, 4))
    assert game.current_prompt == ActionPrompt.MOVE_ROBBER
    game.play(ActionType.MOVE_ROBBER, (TILE_WHEAT_5, None))
    before = game.player(RED).resources
    next_turn(game, dice=(2, 3))
    assert game.player(RED).resources == before


def test_a_seven_makes_large_hands_discard_half(engine):
    game = setup_game(engine)
    game.set_resources(BLUE, (2, 2, 2, 2, 1))
    game.play(ActionType.ROLL, outcome=(3, 4))
    assert game.current_prompt == ActionPrompt.DISCARD
    assert game.current_color == BLUE
    for _ in range(4):
        game.execute(game.playable_actions[0])
    assert hand_size(game, BLUE) == 5
    assert game.current_prompt == ActionPrompt.MOVE_ROBBER
    assert game.current_color == RED


def test_the_discard_limit_is_configurable(engine):
    game = setup_game(engine, discard_limit=9)
    game.set_resources(BLUE, (2, 2, 2, 2, 1))
    game.play(ActionType.ROLL, outcome=(3, 4))
    assert game.current_prompt == ActionPrompt.MOVE_ROBBER
    assert hand_size(game, BLUE) == 9


def test_the_robber_steals_one_card_from_its_victim(engine):
    game = setup_game(engine)
    game.play(ActionType.ROLL, outcome=(3, 4))
    assert BLUE in robber_targets(game)[TILE_WOOD_3]
    game.play(ActionType.MOVE_ROBBER, (TILE_WOOD_3, BLUE), outcome="BRICK")
    assert game.player(RED).resources == (0, 1, 1, 1, 1)
    assert game.player(BLUE).resources == (0, 0, 1, 0, 1)
    assert game.snapshot().robber_coordinate == TILE_WOOD_3


def test_the_friendly_robber_spares_players_below_three_points(engine):
    game = setup_game(engine, friendly_robber=True)
    game.play(ActionType.ROLL, outcome=(3, 4))
    assert all(victims == {None} for victims in robber_targets(game).values())


# --- Building --------------------------------------------------------------------


def test_building_pays_the_bank_and_scores(engine):
    game = setup_game(engine)
    game.play(ActionType.ROLL, outcome=(1, 1))
    bank = game.snapshot().resource_bank

    game.set_resources(RED, (2, 2, 1, 1, 0))
    game.play(ActionType.BUILD_ROAD, (6, 7))
    assert game.player(RED).roads_available == 12
    game.play(ActionType.BUILD_SETTLEMENT, 7)
    red = game.player(RED)
    assert red.resources == (0, 0, 0, 0, 0)
    assert red.settlements_available == 2
    assert red.victory_points == 3

    game.set_resources(RED, (0, 0, 0, 2, 3))
    game.play(ActionType.BUILD_CITY, 7)
    red = game.player(RED)
    assert red.resources == (0, 0, 0, 0, 0)
    assert (red.settlements_available, red.cities_available) == (3, 3)
    assert red.victory_points == 4
    # Costs reach the bank: road, settlement and city.
    paid = (2, 2, 1, 3, 3)
    assert game.snapshot().resource_bank == tuple(b + p for b, p in zip(bank, paid))


def test_nothing_can_be_bought_without_resources(engine):
    game = setup_game(engine)
    game.play(ActionType.ROLL, outcome=(1, 1))
    game.set_resources(RED, (0, 0, 0, 0, 0))
    types = {action.action_type for action in game.playable_actions}
    assert types == {ActionType.END_TURN}


def test_roads_must_connect_to_the_network(engine):
    game = setup_game(engine)
    game.play(ActionType.ROLL, outcome=(1, 1))
    game.set_resources(RED, (5, 5, 0, 0, 0))
    roads = {a.value for a in game.playable_actions if a.action_type == ActionType.BUILD_ROAD}
    assert (6, 7) in roads and (20, 19) not in roads and (19, 20) in roads
    assert (9, 10) not in roads


# --- Development cards -----------------------------------------------------------


def test_a_bought_card_waits_until_the_next_turn(engine):
    game = setup_game(engine)
    game.play(ActionType.ROLL, outcome=(1, 1))
    game.play(ActionType.BUY_DEVELOPMENT_CARD, outcome="KNIGHT")
    assert game.player(RED).development_cards == (1, 0, 0, 0, 0)
    assert game.snapshot().development_cards_remaining == 24
    playable = {action.action_type for action in game.playable_actions}
    assert ActionType.PLAY_KNIGHT_CARD not in playable

    next_turn(game)  # BLUE
    game.play(ActionType.END_TURN)
    playable = {action.action_type for action in game.playable_actions}
    assert ActionType.PLAY_KNIGHT_CARD in playable


def test_only_one_development_card_per_turn(engine):
    game = setup_game(engine)
    game.set_development_cards(RED, (1, 0, 1, 0, 0))
    game.play(ActionType.ROLL, outcome=(1, 1))
    game.play(ActionType.PLAY_MONOPOLY, "SHEEP")
    playable = {action.action_type for action in game.playable_actions}
    assert ActionType.PLAY_KNIGHT_CARD not in playable
    assert game.player(RED).has_played_development_card_in_turn


def test_victory_point_cards_score_privately(engine):
    game = setup_game(engine)
    game.play(ActionType.ROLL, outcome=(1, 1))
    game.play(ActionType.BUY_DEVELOPMENT_CARD, outcome="VICTORY_POINT")
    red = game.player(RED)
    assert (red.victory_points, red.actual_victory_points) == (2, 3)


def test_three_knights_take_the_largest_army(engine):
    game = setup_game(engine)
    game.play(ActionType.ROLL, outcome=(1, 1))
    for turn in range(3):
        game.set_development_cards(RED, (1, 0, 0, 0, 0))
        game.play(ActionType.PLAY_KNIGHT_CARD)
        assert game.current_prompt == ActionPrompt.MOVE_ROBBER
        move_robber_without_victim(game)
        assert game.player(RED).played_development_cards[0] == turn + 1
        if turn < 2:
            assert not game.player(RED).has_army
            next_turn(game)  # BLUE
            next_turn(game)  # RED
    red = game.player(RED)
    assert red.has_army
    assert red.victory_points == 4


def test_road_building_places_two_free_roads(engine):
    game = setup_game(engine)
    game.set_development_cards(RED, (0, 0, 0, 1, 0))
    game.play(ActionType.ROLL, outcome=(1, 1))
    hand = game.player(RED).resources
    game.play(ActionType.PLAY_ROAD_BUILDING)
    assert game.snapshot().is_road_building
    game.play(ActionType.BUILD_ROAD, (6, 7))
    game.play(ActionType.BUILD_ROAD, (7, 8))
    red = game.player(RED)
    assert not game.snapshot().is_road_building
    assert red.resources == hand
    assert red.roads_available == 11


def test_year_of_plenty_draws_two_cards_from_the_bank(engine):
    game = setup_game(engine)
    game.set_development_cards(RED, (0, 1, 0, 0, 0))
    game.play(ActionType.ROLL, outcome=(1, 1))
    bank = game.snapshot().resource_bank
    game.play(ActionType.PLAY_YEAR_OF_PLENTY, ("WOOD", "BRICK"))
    assert game.player(RED).resources == (1, 1, 1, 1, 1)
    assert game.snapshot().resource_bank == (bank[0] - 1, bank[1] - 1, *bank[2:])


def test_monopoly_collects_every_card_of_a_resource(engine):
    game = setup_game(engine)
    game.set_development_cards(RED, (0, 0, 1, 0, 0))
    game.play(ActionType.ROLL, outcome=(1, 1))
    game.play(ActionType.PLAY_MONOPOLY, "ORE")
    assert game.player(RED).resources == (0, 0, 1, 1, 2)
    assert game.player(BLUE).resources == (0, 1, 1, 0, 0)


# --- Trading ---------------------------------------------------------------------


def maritime_trades(game) -> set[tuple]:
    return {
        action.value
        for action in game.playable_actions
        if action.action_type == ActionType.MARITIME_TRADE
    }


def test_the_bank_trades_four_to_one(engine):
    game = setup_game(engine)
    game.play(ActionType.ROLL, outcome=(1, 1))
    game.set_resources(RED, (3, 0, 0, 0, 0))
    assert not maritime_trades(game)
    game.set_resources(RED, (4, 0, 0, 0, 0))
    assert ("WOOD", "WOOD", "WOOD", "WOOD", "ORE") in maritime_trades(game)
    game.play(ActionType.MARITIME_TRADE, ("WOOD", "WOOD", "WOOD", "WOOD", "ORE"))
    assert game.player(RED).resources == (0, 0, 0, 0, 1)


@pytest.mark.parametrize(
    ("port_node", "hand", "trade"),
    [
        (52, (3, 0, 0, 0, 0), ("WOOD", "WOOD", "WOOD", None, "ORE")),  # 3:1 port.
        (28, (0, 2, 0, 0, 0), ("BRICK", "BRICK", None, None, "ORE")),  # 2:1 brick port.
    ],
)
def test_ports_improve_the_rate(engine, port_node, hand, trade):
    placements = [(port_node, None), *TWO_PLAYER_SETUP[1:]]
    game = new_game(engine, [RED, BLUE], map_type="TOURNAMENT")
    for node, road in placements:
        game.play(ActionType.BUILD_SETTLEMENT, node)
        if road is None:
            game.execute(game.playable_actions[0])
        else:
            game.play(ActionType.BUILD_ROAD, road)
    game.play(ActionType.ROLL, outcome=(1, 1))
    game.set_resources(RED, hand)
    assert trade in maritime_trades(game)


OFFER_WHEAT_FOR_BRICK = (0, 0, 0, 1, 0, 0, 1, 0, 0, 0)
RESPONSE = (*OFFER_WHEAT_FOR_BRICK, 0)  # Responses also name the offering seat.


def test_a_domestic_trade_is_offered_accepted_and_confirmed(engine):
    game = setup_game(engine)
    game.play(ActionType.ROLL, outcome=(1, 1))
    game.execute(Action(RED, ActionType.OFFER_TRADE, OFFER_WHEAT_FOR_BRICK))
    assert game.current_prompt == ActionPrompt.DECIDE_TRADE
    assert game.current_color == BLUE
    game.play(ActionType.ACCEPT_TRADE, RESPONSE)
    assert game.current_prompt == ActionPrompt.DECIDE_ACCEPTEES
    assert game.current_color == RED
    game.play(ActionType.CONFIRM_TRADE, (*OFFER_WHEAT_FOR_BRICK, BLUE))
    assert game.player(RED).resources == (0, 1, 1, 0, 1)
    assert game.player(BLUE).resources == (0, 0, 1, 1, 1)
    assert game.current_prompt == ActionPrompt.PLAY_TURN


def test_a_trade_nobody_accepts_returns_to_the_turn(engine):
    game = setup_game(engine)
    game.play(ActionType.ROLL, outcome=(1, 1))
    game.execute(Action(RED, ActionType.OFFER_TRADE, OFFER_WHEAT_FOR_BRICK))
    game.play(ActionType.REJECT_TRADE, RESPONSE)
    assert game.current_prompt == ActionPrompt.PLAY_TURN
    assert game.current_color == RED
    assert game.player(RED).resources == (0, 0, 1, 1, 1)


def test_only_players_who_can_pay_may_accept(engine):
    game = setup_game(engine, THREE_PLAYER_SETUP, players=(RED, BLUE, WHITE))
    game.play(ActionType.ROLL, outcome=(1, 1))
    game.execute(Action(RED, ActionType.OFFER_TRADE, OFFER_WHEAT_FOR_BRICK))
    game.play(ActionType.ACCEPT_TRADE, RESPONSE)  # BLUE holds a brick.
    assert game.current_color == WHITE
    assert [a.action_type for a in game.playable_actions] == [ActionType.REJECT_TRADE]
    game.play(ActionType.REJECT_TRADE, RESPONSE)
    assert game.current_prompt == ActionPrompt.DECIDE_ACCEPTEES
    assert {a.value for a in game.playable_actions if a.action_type == ActionType.CONFIRM_TRADE} == {
        (*OFFER_WHEAT_FOR_BRICK, BLUE)
    }
    hands = [game.player(color).resources for color in (RED, BLUE, WHITE)]
    game.play(ActionType.CANCEL_TRADE)
    assert game.current_prompt == ActionPrompt.PLAY_TURN
    assert [game.player(color).resources for color in (RED, BLUE, WHITE)] == hands


# --- Longest road and victory ----------------------------------------------------

RED_LONG_ROAD = [(6, 7), (7, 8), (8, 9), (9, 10)]  # With (1, 6): five in a row.


def build_red_long_road(game) -> None:
    game.set_resources(RED, (4, 4, 0, 0, 0))
    for road in RED_LONG_ROAD[:-1]:
        game.play(ActionType.BUILD_ROAD, road)
    assert game.player(RED).longest_road_length == 4
    assert not game.player(RED).has_road
    game.play(ActionType.BUILD_ROAD, RED_LONG_ROAD[-1])


def test_longest_road_needs_five_roads(engine):
    game = setup_game(engine)
    game.play(ActionType.ROLL, outcome=(1, 1))
    build_red_long_road(game)
    red = game.player(RED)
    assert red.longest_road_length == 5
    assert red.has_road
    assert red.victory_points == 4


def cut_red_road_at_node_9(game) -> None:
    game.play(ActionType.ROLL, outcome=(1, 1))
    build_red_long_road(game)
    next_turn(game)  # BLUE builds into the middle of RED's road.
    game.set_resources(BLUE, (3, 3, 1, 1, 0))
    game.play(ActionType.BUILD_ROAD, (2, 3))
    game.play(ActionType.BUILD_ROAD, (2, 9))
    game.play(ActionType.BUILD_SETTLEMENT, 9)


def test_an_enemy_settlement_cuts_the_longest_road(engine):
    game = setup_game(engine)
    cut_red_road_at_node_9(game)
    assert game.player(RED).longest_road_length == 4


@pytest.mark.xfail(
    strict=True,
    reason="Catanatron keeps Longest Road after a cut leaves no road of five; "
    "the official rules set the card aside. cppanatron matches the reference.",
)
def test_a_cut_below_five_roads_sets_the_award_aside(engine):
    game = setup_game(engine)
    cut_red_road_at_node_9(game)
    assert not game.player(RED).has_road


def test_reaching_the_target_score_wins(engine):
    game = setup_game(engine, vps_to_win=3)
    game.play(ActionType.ROLL, outcome=(1, 1))
    assert game.winning_color is None
    game.set_resources(RED, (0, 0, 0, 2, 3))
    game.play(ActionType.BUILD_CITY, 1)
    assert game.winning_color == RED
