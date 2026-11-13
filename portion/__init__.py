import importlib.metadata

from .api import create_api
from .const import Bound, inf
from .dict import IntervalDict
from .evidence import (
    CONFLICT_EMERGED,
    CONFLICT_RESOLVED,
    COVERAGE_LOST,
    EVIDENCE_GAINED,
    EVIDENCE_LOST,
    NEW_COVERAGE,
    VALUE_CHANGED,
    Change,
    EffectiveValue,
    Reconciliation,
    ScaleTable,
)
from .func import closed, closedopen, empty, iterate, open, openclosed, singleton
from .interval import AbstractDiscreteInterval, Interval
from .io import from_data, from_string, to_data, to_string

__all__ = [
    "CLOSED",
    "CONFLICT_EMERGED",
    "CONFLICT_RESOLVED",
    "COVERAGE_LOST",
    "EVIDENCE_GAINED",
    "EVIDENCE_LOST",
    "NEW_COVERAGE",
    "OPEN",
    "AbstractDiscreteInterval",
    "Change",
    "EffectiveValue",
    "Interval",
    "IntervalDict",
    "Reconciliation",
    "ScaleTable",
    "VALUE_CHANGED",
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
