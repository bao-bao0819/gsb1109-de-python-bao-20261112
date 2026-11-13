"""Cost-based selection of cached column groups.

A partition is repeatedly read for different sets of columns.  Each
cached entry is a *column group*: the columns of one source revision
that were unpacked together, each with a checksum.  A read request can
be served in three ways:

* unpack one or more cached groups that jointly cover the requested
  columns, paying each group's fixed unpack cost;
* unpack some cached groups and supplement the columns they miss with
  a remote read;
* skip the cache and read the whole request back from the source.

The planner in this module enumerates the cached group combinations
and returns the one with the lowest total unpack plus supplemental
read cost, comparing it against a full remote read.

Groups may overlap, but a column can only be served by a group of the
same source revision as the request.  Two groups of the same revision
that carry different checksum values for the same column conflict and
cannot be combined into a successful cache hit.
"""

import itertools

__all__ = (
    "CachedGroup",
    "CachePlan",
    "ColumnCache",
    "ColumnSource",
    "ConflictingColumnsError",
    "plan_column_read",
)


class ConflictingColumnsError(ValueError):
    """Cached groups disagree on a column checksum.

    Raised when two cached groups of the same source revision carry
    different checksum values for one of the requested columns.  Such
    groups cannot be combined into a successful cache hit.
    """

    def __init__(self, column, checksums, group_ids):
        self.column = column
        self.checksums = tuple(checksums)
        self.group_ids = tuple(group_ids)
        super().__init__(
            "conflicting checksums for column %r: %r (groups %r)"
            % (column, self.checksums, self.group_ids)
        )


class CachedGroup:
    """A cached group of columns belonging to one source revision.

    `group_id` identifies the group, `revision` is the source revision
    its data was read from, and `checksums` maps every covered column
    to its checksum value; the mapping's keys are the columns the
    group can serve.  `unpack_cost` is the fixed cost charged for
    unpacking the group, at most once per read even when several of
    its columns are requested.  `payload` holds the caller's encoded
    or decoded column data.
    """

    __slots__ = (
        "group_id",
        "revision",
        "columns",
        "checksums",
        "unpack_cost",
        "payload",
    )

    def __init__(self, group_id, revision, checksums, unpack_cost=1.0, payload=None):
        checksums = dict(checksums)
        self.group_id = group_id
        self.revision = revision
        self.columns = tuple(checksums)
        self.checksums = checksums
        self.unpack_cost = unpack_cost
        self.payload = payload

    def __repr__(self):
        return "CachedGroup(%r, revision=%r, columns=%r, unpack_cost=%r)" % (
            self.group_id,
            self.revision,
            self.columns,
            self.unpack_cost,
        )

    def __eq__(self, other):
        if not isinstance(other, CachedGroup):
            return NotImplemented
        return (
            self.group_id == other.group_id
            and self.revision == other.revision
            and self.checksums == other.checksums
            and self.unpack_cost == other.unpack_cost
            and self.payload == other.payload
        )


class CachePlan:
    """The cheapest way found to serve one column read.

    `groups` are the cached groups to unpack, in cache order.
    `remote_columns` are the columns still to read from the source, in
    the order in which they were first requested.  When `full_remote`
    is true, the whole request is read back from the source and no
    cached group is unpacked.
    """

    __slots__ = (
        "groups",
        "remote_columns",
        "revision",
        "unpack_cost",
        "remote_cost",
        "full_remote",
    )

    def __init__(
        self, groups, remote_columns, revision, unpack_cost, remote_cost, full_remote
    ):
        self.groups = tuple(groups)
        self.remote_columns = tuple(remote_columns)
        self.revision = revision
        self.unpack_cost = unpack_cost
        self.remote_cost = remote_cost
        self.full_remote = full_remote

    @classmethod
    def full_remote_read(cls, columns, revision, remote_cost):
        """Build a plan that reads every requested column remotely."""
        return cls((), tuple(columns), revision, 0.0, remote_cost, True)

    @property
    def total_cost(self):
        """Total unpack plus remote read cost of this plan."""
        return self.unpack_cost + self.remote_cost

    @property
    def hit(self):
        """True when every requested column comes from the cache."""
        return not self.full_remote and not self.remote_columns

    def __repr__(self):
        return (
            "CachePlan(groups=%r, remote_columns=%r, full_remote=%r, "
            "total_cost=%r)"
            % (
                [group.group_id for group in self.groups],
                list(self.remote_columns),
                self.full_remote,
                self.total_cost,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, CachePlan):
            return NotImplemented
        return (
            self.groups == other.groups
            and self.remote_columns == other.remote_columns
            and self.revision == other.revision
            and self.unpack_cost == other.unpack_cost
            and self.remote_cost == other.remote_cost
            and self.full_remote == other.full_remote
        )


def _unique_in_order(columns):
    seen = set()
    unique = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            unique.append(column)
    return unique


def _cost(cost, columns):
    if callable(cost):
        return cost(tuple(columns))
    return cost * len(columns)


def plan_column_read(
    groups, columns, revision, *, remote_column_cost=1.0, full_remote_cost=None
):
    """Return the cheapest plan for reading `columns` of `revision`.

    `groups` is an iterable of :class:`CachedGroup` instances in cache
    order.  `columns` is the caller's request; its order and any
    duplicate entries are preserved in the returned plan's remote
    columns.  `remote_column_cost` is either the price per column read
    from the source or a callable taking the missing columns and
    returning the supplemental read cost.  `full_remote_cost` is the
    cost of reading the whole request from the source, as an absolute
    number or a callable taking the requested columns; when omitted, a
    full remote read is not considered.

    Groups from other revisions are ignored for coverage.  Overlapping
    groups of the requested revision must agree on each shared
    column's checksum, otherwise
    :exc:`ConflictingColumnsError` is raised.
    """
    requested = tuple(columns)
    wanted = _unique_in_order(requested)

    eligible = []
    checksums = {}
    owners = {}
    for group in groups:
        if group.revision != revision:
            continue
        covered = tuple(column for column in wanted if column in group.checksums)
        if not covered:
            continue
        for column in covered:
            checksum = group.checksums[column]
            if column in checksums and checksums[column] != checksum:
                raise ConflictingColumnsError(
                    column,
                    (checksums[column], checksum),
                    (owners[column].group_id, group.group_id),
                )
            checksums[column] = checksum
            owners[column] = group
        eligible.append((group, covered))

    best_key = None
    best_mask = 0
    best_unpack = 0.0
    best_remote = 0.0
    best_missing = tuple(wanted)

    for mask in range(1 << len(eligible)):
        unpack_cost = 0.0
        covered = set()
        for index, (group, group_columns) in enumerate(eligible):
            if mask & (1 << index):
                unpack_cost += group.unpack_cost
                covered.update(group_columns)
        missing = tuple(column for column in wanted if column not in covered)
        remote_cost = _cost(remote_column_cost, missing)
        # On equal cost, prefer fewer remote columns (more cache
        # coverage), then fewer unpacked groups, then cache order.
        key = (
            unpack_cost + remote_cost,
            len(missing),
            mask.bit_count(),
            mask,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_mask = mask
            best_unpack = unpack_cost
            best_remote = remote_cost
            best_missing = missing

    chosen_groups = tuple(
        group for index, (group, _) in enumerate(eligible) if best_mask & (1 << index)
    )
    plan = CachePlan(
        chosen_groups,
        best_missing,
        revision,
        best_unpack,
        best_remote,
        False,
    )

    if full_remote_cost is not None:
        if callable(full_remote_cost):
            remote_cost = full_remote_cost(tuple(wanted))
        else:
            remote_cost = full_remote_cost
        if remote_cost < plan.total_cost:
            return CachePlan.full_remote_read(wanted, revision, remote_cost)

    return plan


class ColumnSource:
    """Source backing a :class:`ColumnCache`.

    Subclasses must implement :meth:`read`.  The cost methods feed the
    planner and may be overridden with source-specific numbers.
    Override :meth:`checksums` to let supplemental and full remote
    reads be stored back into the cache as new column groups.
    """

    def read(self, partition, columns, revision):
        """Return a mapping of the requested columns to their values."""
        raise NotImplementedError

    def unpack(self, group):
        """Return a mapping of column values for a cached group."""
        if group.payload is None:
            raise ValueError(
                "group %r has no payload; override unpack()" % group.group_id
            )
        return group.payload

    def supplemental_cost(self, partition, columns):
        """Cost of reading `columns` to supplement cached groups."""
        return float(len(columns))

    def full_remote_cost(self, partition, columns, revision):
        """Cost of reading the whole request from the source."""
        return float(len(columns))

    def checksums(self, partition, columns, revision, values):
        """Return per-column checksums, or None to skip caching."""
        return None


class ColumnCache:
    """Cache of :class:`CachedGroup` instances keyed by partition.

    The cache plans each read with :func:`plan_column_read`, unpacks
    the selected groups, supplements missing columns through its
    :class:`ColumnSource`, and falls back to a full remote read when
    no cache combination is cheaper or when cached groups conflict.
    """

    def __init__(self, source, *, default_unpack_cost=1.0):
        self.source = source
        self.default_unpack_cost = default_unpack_cost
        self._groups = {}
        self._group_ids = itertools.count()

    def add(
        self,
        partition,
        revision,
        checksums,
        unpack_cost=None,
        payload=None,
        group_id=None,
    ):
        """Store a column group and return it."""
        if group_id is None:
            group_id = next(self._group_ids)
        if unpack_cost is None:
            unpack_cost = self.default_unpack_cost
        group = CachedGroup(
            group_id,
            revision,
            checksums,
            unpack_cost=unpack_cost,
            payload=payload,
        )
        self._groups.setdefault(partition, []).append(group)
        return group

    def groups(self, partition):
        """Return the cached groups of `partition` in cache order."""
        return tuple(self._groups.get(partition, ()))

    def plan(self, partition, columns, revision):
        """Return the cheapest :class:`CachePlan` for a read."""
        source = self.source
        requested = tuple(columns)
        return plan_column_read(
            self.groups(partition),
            requested,
            revision,
            remote_column_cost=lambda missing: source.supplemental_cost(
                partition, missing
            ),
            full_remote_cost=lambda requested_columns: source.full_remote_cost(
                partition, requested_columns, revision
            ),
        )

    def read(self, partition, columns, revision):
        """Read columns and return values in the requested order."""
        values, _ = self.read_with_plan(partition, columns, revision)
        return values

    def read_with_plan(self, partition, columns, revision):
        """Like :meth:`read`, also returning the executed plan."""
        requested = tuple(columns)
        try:
            plan = self.plan(partition, requested, revision)
        except ConflictingColumnsError:
            wanted = _unique_in_order(requested)
            remote_cost = self.source.full_remote_cost(partition, wanted, revision)
            plan = CachePlan.full_remote_read(wanted, revision, remote_cost)

        if plan.full_remote:
            values = self._read_remote(partition, plan.remote_columns, revision)
        else:
            values = {}
            for group in plan.groups:
                values.update(self.source.unpack(group))
            if plan.remote_columns:
                values.update(
                    self._read_remote(partition, plan.remote_columns, revision)
                )
        ordered = [values[column] for column in requested]
        return ordered, plan

    def _read_remote(self, partition, columns, revision):
        columns = tuple(columns)
        values = dict(self.source.read(partition, columns, revision))
        missing = [column for column in columns if column not in values]
        if missing:
            raise KeyError("source did not return columns %r" % missing)
        checksums = self.source.checksums(partition, columns, revision, values)
        if checksums is not None:
            self.add(
                partition,
                revision,
                {column: checksums[column] for column in columns},
                payload={column: values[column] for column in columns},
            )
        return values
