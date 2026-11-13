import importlib.metadata

from .api import create_api
from .const import Bound, inf
from .dict import IntervalDict
from .func import closed, closedopen, empty, iterate, open, openclosed, singleton
from .histogram import (
    BucketMigration,
    GroupRange,
    HistogramMigrator,
    MigrationStatus,
)
from .interval import AbstractDiscreteInterval, Interval
from .io import from_data, from_string, to_data, to_string

__all__ = [
    "CLOSED",
    "OPEN",
    "AbstractDiscreteInterval",
    "BucketMigration",
    "GroupRange",
    "HistogramMigrator",
    "Interval",
    "IntervalDict",
    "MigrationStatus",
    "closed",
    "closedopen",
    "create_api",
    "empty",
    "from_data",
    "from_string",
    "inf",
    "iterate",
    "open",
    "openclosed",
    "singleton",
    "to_data",
    "to_string",
]

CLOSED = Bound.CLOSED
OPEN = Bound.OPEN

__version__ = importlib.metadata.version("portion")
