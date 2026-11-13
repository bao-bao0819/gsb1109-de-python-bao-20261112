import importlib.metadata

from .api import create_api
from .const import Bound, inf
from .dict import IntervalDict
from .func import closed, closedopen, empty, iterate, open, openclosed, singleton
from .interval import AbstractDiscreteInterval, Interval
from .io import from_data, from_string, to_data, to_string
from .stats import KthSmallestCheck, check_kth_smallest, kth_smallest_values

__all__ = [
    "CLOSED",
    "OPEN",
    "AbstractDiscreteInterval",
    "Interval",
    "IntervalDict",
    "KthSmallestCheck",
    "check_kth_smallest",
    "closed",
    "closedopen",
    "create_api",
    "empty",
    "from_data",
    "from_string",
    "inf",
    "iterate",
    "kth_smallest_values",
    "open",
    "openclosed",
    "singleton",
    "to_data",
    "to_string",
]

CLOSED = Bound.CLOSED
OPEN = Bound.OPEN

__version__ = importlib.metadata.version("portion")
