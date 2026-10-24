"""Random dice stay what they were before balanced dice existed.

The digests below were recorded with the engine at e9f12af, before GameRules
had a dice mode: seeded games under the default (random) dice play out step
for step as they did (every roll, steal and card draw shows in the per-step
state digests), and searches over them give the same visits, values and
tree statistics (chance nodes use the fixed 2d6 odds, and chance outcomes
coalesce as before).

The games leave the friendly robber off: wt/friendly changes that rule, so a
friendly-robber game no longer plays as it did at e9f12af, whatever the dice.
Its own tests pin it.

A change here means seeded results changed: only accept it on purpose.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

import cppanatron as cpp
from cppanatron.catanatron import observation_layout


def state_digest(game: cpp.Game) -> str:
    players = tuple(
        (p.resources, p.development_cards, p.actual_victory_points, p.has_rolled)
        for p in game.players)
    return repr((
        game.current_player_index, game.current_prompt.name, game.num_turns,
        game.resource_bank, game.development_cards_remaining, game.board.robber_coordinate,
        players))


def play_digest(game: cpp.Game, salt: int, limit: int = 3000) -> tuple[str, int, int, str | None]:
    """Play deterministic moves; sha256 over every step's choice and resulting state."""
    digest = hashlib.sha256()
    steps = 0
    while game.winning_color is None and steps < limit:
        actions = game.playable_actions
        choice = (steps * 7919 + salt) % len(actions)
        game.execute(actions[choice])
        digest.update(f"{choice}:{actions[choice].action_type.name}|{state_digest(game)}\n".encode())
        steps += 1
    winner = None if game.winning_color is None else game.winning_color.name
    return digest.hexdigest(), steps, game.num_turns, winner


GAMES = {
    "2p base": lambda: cpp.Game(2, seed=5),
    "4p random numbers": lambda: cpp.Game(4, seed=2024, number_placement=cpp.NumberPlacement.RANDOM),
    "3p, 12 vp": lambda: cpp.Game(3, seed=77, vps_to_win=12),
    "15 vp, discard above 9, random dice": lambda: cpp.Game(
        2, seed=31, rules=cpp.GameRules(vps_to_win=15, discard_limit=9)),
}

GOLDEN_GAMES = {'2p base': ('80a2e422328274a0c2ccf3b6f070ff9d8cfffc0a58b65e94c378558cc08b3088', 1177, 438, 'RED'),
 '4p random numbers': ('1dad86728e9c657924140b7c093e86fdde260ab67f2fb598fba93b8c44762122',
                       1141,
                       321,
                       'ORANGE'),
 '3p, 12 vp': ('4ffadf4fcee1ae4502e4c9302a901fa38eccd7d319c6da05687d2113d6dc9623',
               872,
               306,
               'WHITE'),
 '15 vp, discard above 9, random dice': ('8c4e3794e1ff3ff7165eb37fd70cc810c2c519123a9fe99ab6b00cb4cb95efa3',
                                         2575,
                                         1001,
                                         'BLUE')}


def random_position(players: int, seed: int, steps: int, **options):
    """A game ``steps`` random moves in (stopped before a win) with a tracker per seat."""
    game = cpp.Game(players, seed=seed, **options)
    trackers = [cpp.HandTracker(game, seat) for seat in range(players)]
    rng = random.Random(seed)
    for _ in range(steps):
        action = rng.choice(game.playable_actions)
        before = game.copy()
        game.execute(action)
        for tracker in trackers:
            tracker.observe(before, action, game)
        if game.winning_color is not None:
            raise AssertionError("position reached a win")
    return game, trackers


def before_a_roll(game: cpp.Game, trackers) -> cpp.Game:
    """Play the first action until the player to move has not rolled yet (at play_turn)."""
    while not (game.current_prompt == cpp.ActionPrompt.PLAY_TURN
               and not game.player(game.current_player_index).has_rolled
               and not game.is_initial_build_phase):
        action = game.playable_actions[-1]
        before = game.copy()
        game.execute(action)
        for tracker in trackers:
            tracker.observe(before, action, game)
    return game


def evaluate(observations: np.ndarray, action_size: int, max_players: int):
    """A deterministic stand-in network: logits and values from the observation."""
    total = observations.sum(axis=1, keepdims=True, dtype=np.float64)
    logits = np.sin(np.arange(action_size)[None, :] * 0.37 + total * 1e-3).astype(np.float32)
    values = np.empty((len(observations), max_players), dtype=np.float32)
    values[:, 0] = (0.4 + 0.2 * np.tanh(total[:, 0] * 1e-3)).astype(np.float32)
    values[:, 1:] = ((1.0 - values[:, :1]) / (max_players - 1)).astype(np.float32)
    return logits, values


def search_digest(game, trackers, players, **config) -> dict:
    codec = cpp.FlatActionCodec(players, cpp.MapType.BASE)
    encoder = cpp.FlatObservationEncoder(players, observation_layout())
    pool = cpp.SearchPool(codec, encoder, cpp.SearchConfig(max_players=players, **config))
    tracker = trackers[game.current_player_index] if config.get("determinizations") else None
    search = pool.add(game, 7, tracker)
    observations = np.zeros((16, pool.observation_size), dtype=np.float32)
    ids = np.zeros(16, dtype=np.int64)
    while n := pool.collect(observations, ids):
        pool.apply(*evaluate(observations[:n], pool.action_size, pool.max_players))
    visits = pool.visits(search)
    stats = pool.stats(search)
    return {
        "visits": {int(i): int(visits[i]) for i in np.flatnonzero(visits)},
        "value": [float(v) for v in pool.value(search)],
        "stats": (stats.nodes, stats.chance_nodes, stats.edges, stats.coalesced_outcomes,
                  stats.transpositions, stats.max_depth),
    }


def mcts_digest(game, players, canonical_pruning) -> dict:
    space = cpp.FlatActionSpace(players, cpp.MapType.BASE)
    search = cpp.MCTSSearch(game, space, observation_layout(), seed=3,
                            canonical_pruning=canonical_pruning)
    observation = np.zeros(search.observation_size, dtype=np.float32)
    search.root_observation(observation)
    logits, _ = evaluate(observation[None, :], search.action_space_size, 2)
    search.initialize_root(logits[0])
    for _ in range(200):
        player = search.select_leaf(observation)
        if player is None:
            continue
        logits, values = evaluate(observation[None, :], search.action_space_size, 2)
        search.evaluate_leaf(logits[0], float(values[0, 0]) * 2.0 - 1.0)
    visits = search.root_visits()
    metrics = search.metrics()
    return {
        "visits": {int(i): int(visits[i]) for i in np.flatnonzero(visits)},
        "root_value": metrics.root_value,
        "coalesced": metrics.coalesced_outcomes,
        "max_depth": metrics.maximum_depth,
    }


def after_a_roll(game: cpp.Game, trackers) -> cpp.Game:
    """Play the first action until the player to move has rolled and has three options or more."""
    while not (game.current_prompt == cpp.ActionPrompt.PLAY_TURN
               and game.player(game.current_player_index).has_rolled
               and len(game.playable_actions) >= 3):
        action = game.playable_actions[0]
        before = game.copy()
        game.execute(action)
        for tracker in trackers:
            tracker.observe(before, action, game)
    return game


def positions():
    two, two_trackers = random_position(2, 3, 120)
    two = after_a_roll(two, two_trackers)
    four, four_trackers = random_position(4, 21, 200)
    roll, roll_trackers = random_position(2, 8, 90)
    roll = before_a_roll(roll, roll_trackers)
    return {"2p": (two, two_trackers, 2), "4p": (four, four_trackers, 4),
            "2p before a roll": (roll, roll_trackers, 2)}


SEARCHES = {
    "puct": {"simulations": 256},
    "puct transpositions": {"simulations": 256, "transpositions": True, "max_pending": 4},
    "gumbel determinized": {"simulations": 96, "root": cpp.RootSelection.GUMBEL,
                            "gumbel_considered": 8, "determinizations": 2, "max_pending": 2},
}

GOLDEN_SEARCHES = {'2p / puct': {'visits': {186: 27, 219: 32, 220: 50, 221: 57, 222: 90},
               'value': [0.5056337118148804, 0.494366317987442],
               'stats': (1343, 369, 7014, 36, 0, 33)},
 '2p / puct transpositions': {'visits': {186: 31, 219: 29, 220: 54, 221: 63, 222: 79},
                              'value': [0.5034657120704651, 0.4965343177318573],
                              'stats': (1318, 364, 7030, 32, 49, 38)},
 '2p / gumbel determinized': {'visits': {186: 6, 219: 6, 220: 6, 221: 39, 222: 39},
                              'value': [0.5292991399765015, 0.4707008898258209],
                              'stats': (313, 85, 1793, 8, 0, 26)},
 '4p / puct': {'visits': {186: 184, 323: 72},
               'value': [0.3322800397872925,
                         0.21155285835266113,
                         0.2096196860074997,
                         0.2465474158525467],
               'stats': (1196, 315, 6058, 12, 0, 28)},
 '4p / puct transpositions': {'visits': {186: 177, 323: 79},
                              'value': [0.32435497641563416,
                                        0.22025901079177856,
                                        0.20616407692432404,
                                        0.24922195076942444],
                              'stats': (1216, 341, 6268, 16, 27, 28)},
 '4p / gumbel determinized': {'visits': {186: 48, 323: 48},
                              'value': [0.28877249360084534,
                                        0.2387070655822754,
                                        0.21625681221485138,
                                        0.2562636435031891],
                              'stats': (568, 161, 2917, 4, 0, 21)},
 '2p before a roll / puct': {'visits': {312: 256},
                             'value': [0.5118927955627441, 0.48810717463493347],
                             'stats': (1131, 307, 6265, 25, 0, 23)},
 '2p before a roll / puct transpositions': {'visits': {312: 256},
                                            'value': [0.5129756331443787, 0.4870243966579437],
                                            'stats': (1191, 331, 6400, 38, 53, 28)},
 '2p before a roll / gumbel determinized': {'visits': {312: 96},
                                            'value': [0.5124408006668091, 0.4875591993331909],
                                            'stats': (461, 128, 2493, 13, 0, 21)}}
GOLDEN_MCTS = {'2p / pruning=False': {'visits': {186: 29, 219: 25, 220: 37, 221: 47, 222: 62},
                        'root_value': -0.0010951909422874451,
                        'coalesced': 0,
                        'max_depth': 7},
 '2p / pruning=True': {'visits': {186: 36, 219: 22, 220: 31, 221: 51, 222: 60},
                       'root_value': -0.0028053972125053404,
                       'coalesced': 138,
                       'max_depth': 9},
 '2p before a roll / pruning=False': {'visits': {312: 200},
                                      'root_value': 0.015337744653224945,
                                      'coalesced': 0,
                                      'max_depth': 6},
 '2p before a roll / pruning=True': {'visits': {312: 200},
                                     'root_value': 0.005082061290740967,
                                     'coalesced': 145,
                                     'max_depth': 8}}


@pytest.mark.parametrize("name", list(GAMES))
def test_seeded_games_play_out_as_before(name):
    assert play_digest(GAMES[name](), salt=len(name)) == GOLDEN_GAMES[name]


def test_searches_find_what_they_found_before():
    results = {}
    for position, (game, trackers, players) in positions().items():
        for name, config in SEARCHES.items():
            results[f"{position} / {name}"] = search_digest(game, trackers, players, **config)
    assert results == GOLDEN_SEARCHES


def test_mcts_searches_find_what_they_found_before():
    results = {}
    for position, (game, _, players) in positions().items():
        if players != 2:
            continue
        for pruning in (False, True):
            results[f"{position} / pruning={pruning}"] = mcts_digest(game, players, pruning)
    assert results == GOLDEN_MCTS


if __name__ == "__main__":
    import pprint

    games = {name: play_digest(make(), salt=len(name)) for name, make in GAMES.items()}
    searches = {}
    mcts = {}
    for position, (game, trackers, players) in positions().items():
        for name, config in SEARCHES.items():
            searches[f"{position} / {name}"] = search_digest(game, trackers, players, **config)
        if players == 2:
            for pruning in (False, True):
                mcts[f"{position} / pruning={pruning}"] = mcts_digest(game, players, pruning)
    pprint.pprint({"games": games, "searches": searches, "mcts": mcts}, width=100, sort_dicts=False)
