import importlib.metadata

from .api import create_api
from .const import Bound, inf
from .dict import IntervalDict
from .func import closed, closedopen, empty, iterate, open, openclosed, singleton
from .interval import AbstractDiscreteInterval, Interval
from .io import from_data, from_string, to_data, to_string
from .temporal import (
    InconsistentRecords,
    OrderResult,
    TemporalConstraint,
    TemporalModel,
    make_domain,
    max_wait,
    min_gap,
    before,
)

__all__ = [
    "CLOSED",
    "OPEN",
    "AbstractDiscreteInterval",
    "InconsistentRecords",
    "Interval",
    "IntervalDict",
    "OrderResult",
    "TemporalConstraint",
    "TemporalModel",
    "before",
    "closed",
    "closedopen",
    "create_api",
    "empty",
    "from_data",
    "from_string",
    "inf",
    "iterate",
    "make_domain",
    "max_wait",
    "min_gap",
    "open",
    "openclosed",
    "singleton",
    "to_data",
    "to_string",
]

CLOSED = Bound.CLOSED
OPEN = Bound.OPEN

__version__ = importlib.metadata.version("portion")
