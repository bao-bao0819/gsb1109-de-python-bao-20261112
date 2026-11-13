import importlib.metadata

from .api import create_api
from .const import Bound, inf
from .dict import IntervalDict
from .func import closed, closedopen, empty, iterate, open, openclosed, singleton
from .grading import (
    CONFLICT,
    GAP,
    UNIQUE,
    GradeRule,
    GradingConflict,
    GradingMap,
    Segment,
    compile_grading,
)
from .interval import AbstractDiscreteInterval, Interval
from .io import from_data, from_string, to_data, to_string

__all__ = [
    "CLOSED",
    "CONFLICT",
    "GAP",
    "OPEN",
    "UNIQUE",
    "AbstractDiscreteInterval",
    "GradeRule",
    "GradingConflict",
    "GradingMap",
    "Interval",
    "IntervalDict",
    "Segment",
    "closed",
    "closedopen",
    "compile_grading",
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
