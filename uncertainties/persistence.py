# coding=utf-8

"""
Partitioned persistence of measurements with uncertainties.

The :mod:`uncertainties` package represents each independent source of
uncertainty through the object identity of a :class:`~uncertainties.core.Variable`.
Ordinary pickling only preserves this identity *inside* a single pickle blob:
two blobs written separately always unpickle into unrelated variables, even
when they represent the same physical error source (a common calibration
bias, for instance).

This module adds partitioned read/write support: a measurement can be split
into separately stored partitions (one per day, per instrument, ...), while
the original error sources keep a persistent identity across partitions.
Partitions that share an error source stay correlated after being read back,
so that the shared calibration bias cancels out when two measurements are
subtracted.  Error sources that merely use the same :attr:`Variable.tag`
remain independent, because the identity is carried by an internally
generated key, never by the tag alone.

A single partition can always be read back on its own: its marginal
uncertainties are then fully reconstructed.
"""

import os
import pickle
import uuid

from uncertainties.core import Variable

# Qualified name under which the variable reconstruction callable is
# referenced inside a partition.  The callable itself is only a pickle-time
# marker: _PartitionUnpickler.find_class() intercepts its global lookup and
# binds it to the registry used for reading:
_RECONSTRUCT_MODULE = "uncertainties.persistence"
_RECONSTRUCT_NAME = "_partition_variable"


def _partition_variable(key, nominal_value, std_dev, tag):
    """
    Pickle-time marker for a registry-identified Variable.

    This function is never meant to be called: partitions are read back by
    :func:`read_partition`, whose unpickler redirects this global to the
    reading :class:`VariableRegistry`.
    """
    raise RuntimeError(
        "Variables stored in an uncertainties partition must be read back "
        "with uncertainties.persistence.read_partition()."
    )


class VariableRegistry:
    """
    Registry of the persistent identities of independent variables.

    A registry is shared between the writes and the reads of the partitions
    of one experiment:

    - While writing, it gives each :class:`Variable` a stable key that is
      embedded in every partition referencing this variable.
    - While reading, it maps each key back to one canonical
      :class:`Variable`, so that variables representing the same error
      source in different partitions are reconstructed only once and keep
      sharing their object identity.

    Two distinct :class:`Variable` objects always receive distinct keys,
    even if they carry the same tag.  A registry used for reading does not
    need to be the same object as the one used for writing: the keys are
    stored inside the partitions themselves, so a fresh registry merges
    partitions written by another process.
    """

    def __init__(self):
        # Writing side: id() of a Variable -> (Variable, persistent key).
        # The Variable itself is kept referenced so that its id() cannot be
        # recycled by another variable while the registry is in use:
        self._keys_by_id = {}

        # Reading side: persistent key -> canonical reconstructed Variable.
        self._variables_by_key = {}

    def key_for(self, variable):
        """
        Return the persistent key of a Variable, creating it on first use.
        """
        entry = self._keys_by_id.get(id(variable))

        if entry is not None and entry[0] is variable:
            return entry[1]

        key = uuid.uuid4().hex
        self._keys_by_id[id(variable)] = (variable, key)

        return key

    def load_variable(self, key, nominal_value, std_dev, tag):
        """
        Return the canonical Variable for a persistent key.

        The variable is created from the given data the first time the key
        is encountered.  Further partitions referencing the same key reuse
        the same variable; their recorded nominal value, standard deviation
        and tag are then ignored, as the key identifies one single error
        source.
        """
        variable = self._variables_by_key.get(key)

        if variable is None:
            variable = Variable(nominal_value, std_dev, tag)
            self._variables_by_key[key] = variable

        return variable


class _PartitionPickler(pickle.Pickler):
    """
    Pickler that stores Variables as registry-keyed reconstruction records.

    The reduction uses the regular pickle machinery, so repeated references
    to the same variable inside one partition are shared through the pickle
    memo.  The reconstruction record carries the variable key together with
    its nominal value, standard deviation and tag, so that each partition
    file remains self-contained and can be read on its own.
    """

    def __init__(self, file, registry, protocol):
        super().__init__(file, protocol=protocol)
        self._registry = registry

    def reducer_override(self, obj):
        # Only exact Variable instances take part in the persistent
        # identity: subclassed variables keep their regular pickle
        # representation (and their extra state):
        if type(obj) is not Variable:
            return NotImplemented

        key = self._registry.key_for(obj)

        return (
            _partition_variable,
            (key, obj.nominal_value, obj.std_dev, obj.tag),
        )


class _PartitionUnpickler(pickle.Unpickler):
    """
    Unpickler that resolves variable reconstruction records via a registry.
    """

    def __init__(self, file, registry):
        super().__init__(file)
        self._registry = registry

    def find_class(self, module, name):
        if module == _RECONSTRUCT_MODULE and name == _RECONSTRUCT_NAME:
            return self._reconstruct_variable

        return super().find_class(module, name)

    def _reconstruct_variable(self, key, nominal_value, std_dev, tag):
        return self._registry.load_variable(
            key, nominal_value, std_dev, tag
        )


def _as_binary_file(file, mode):
    """
    Return (binary file, should_be_closed) for a path or an open file object.
    """
    if isinstance(file, (str, os.PathLike)):
        return open(file, mode), True

    return file, False


def write_partition(measurement, file, registry=None, protocol=None):
    """
    Write a measurement partition to a binary file or file object.

    measurement -- object containing numbers with uncertainties
    (``AffineScalarFunc`` objects, including inside containers or regular
    picklable objects).

    file -- path to the partition file, or a binary file object opened for
    writing.

    registry -- :class:`VariableRegistry` shared by all the partitions of the
    same experiment.  A temporary registry is created when it is omitted,
    which is only useful for writing a single standalone partition.

    protocol -- pickle protocol used internally (the default protocol is used
    if omitted).

    The resulting partition can only be read back with
    :func:`read_partition`, but it is self-contained: reading it alone
    restores the nominal values, correlations internal to the partition and
    marginal uncertainties of the measurement.
    """
    if registry is None:
        registry = VariableRegistry()

    if protocol is None:
        protocol = pickle.DEFAULT_PROTOCOL

    # reducer_override() is only consulted for pickle protocol 2 and newer:
    protocol = max(protocol, 2)

    binary_file, should_be_closed = _as_binary_file(file, "wb")

    try:
        pickler = _PartitionPickler(binary_file, registry, protocol)
        pickler.dump(measurement)
    finally:
        if should_be_closed:
            binary_file.close()


def read_partition(file, registry=None):
    """
    Read a measurement partition written with :func:`write_partition`.

    file -- path to the partition file, or a binary file object opened for
    reading.

    registry -- :class:`VariableRegistry` used to merge error sources across
    partitions.  A fresh registry is created when it is omitted, so reading
    several partitions with the same registry reconstructs their shared
    variables while reading one partition alone restores its marginal
    uncertainties.  The read order does not change the result.
    """
    if registry is None:
        registry = VariableRegistry()

    binary_file, should_be_closed = _as_binary_file(file, "rb")

    try:
        unpickler = _PartitionUnpickler(binary_file, registry)
        return unpickler.load()
    finally:
        if should_be_closed:
            binary_file.close()
