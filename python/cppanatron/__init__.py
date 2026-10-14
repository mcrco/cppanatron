"""A C++20 Catan rules engine with Catanatron-compatible semantics.

The engine types live in the compiled ``cppanatron._core`` extension and are
re-exported here. ``cppanatron.catanatron`` converts to and from Catanatron's
types, and ``cppanatron.parity`` runs the same scenarios on both engines.
"""

from importlib.metadata import PackageNotFoundError, version

from ._core import (
    Action,
    ActionPrompt,
    ActionType,
    Board,
    Building,
    CatanMap,
    Color,
    DevelopmentCard,
    Direction,
    FlatActionSpace,
    Game,
    GameBatch,
    GameRules,
    MapSpec,
    MapType,
    MCTSSearch,
    MCTSSearchMetrics,
    NumberPlacement,
    ObservationLayout,
    PlayerSetup,
    PlayerState,
    Resource,
    RewardFunction,
    Tile,
    TileKind,
    TileSpec,
    ValueEnemies,
    ValueWeights,
    board_observation_size,
    full_numeric_observation_size,
    full_observation,
    full_observation_size,
    production_sum,
    value_action,
    value_score,
    write_full_observation,
)

try:
    __version__ = version("cppanatron")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"

__all__ = [
    "Action",
    "ActionPrompt",
    "ActionType",
    "Board",
    "Building",
    "CatanMap",
    "Color",
    "DevelopmentCard",
    "Direction",
    "FlatActionSpace",
    "Game",
    "GameBatch",
    "GameRules",
    "MCTSSearch",
    "MCTSSearchMetrics",
    "MapSpec",
    "MapType",
    "NumberPlacement",
    "ObservationLayout",
    "PlayerSetup",
    "PlayerState",
    "Resource",
    "RewardFunction",
    "Tile",
    "TileKind",
    "TileSpec",
    "ValueEnemies",
    "ValueWeights",
    "board_observation_size",
    "full_numeric_observation_size",
    "full_observation",
    "full_observation_size",
    "production_sum",
    "value_action",
    "value_score",
    "write_full_observation",
]
