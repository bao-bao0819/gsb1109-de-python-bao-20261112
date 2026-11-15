import importlib.metadata

from .api import create_api
from .const import Bound, inf
from .dict import IntervalDict
from .func import closed, closedopen, empty, iterate, open, openclosed, singleton
from .interval import AbstractDiscreteInterval, Interval
from .io import from_data, from_string, to_data, to_string
from .publish import (
    Bucket,
    Plan,
    PublishedRange,
    Reason,
    SuppressedRange,
    plan_publication,
)

__all__ = [
    "CLOSED",
    "OPEN",
    "AbstractDiscreteInterval",
    "Bucket",
    "Interval",
    "IntervalDict",
    "Plan",
    "PublishedRange",
    "Reason",
    "SuppressedRange",
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
    "plan_publication",
    "singleton",
    "to_data",
    "to_string",
]

CLOSED = Bound.CLOSED
OPEN = Bound.OPEN

__version__ = importlib.metadata.version("portion")
