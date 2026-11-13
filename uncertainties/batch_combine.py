"""
Optimal combination of batch measurements that share error sources.

Several experimental batches each report a mean of the same measurand.
Some batches share calibration artefacts (common systematic errors), others
are only repeated submissions of the same sample (which share *every*
upstream error).  A plain inverse-variance average would silently treat
those correlated results as independent and would overstate the achieved
precision.

This module builds each measurement from *tagged* error sources so that the
full covariance matrix is constructed automatically, and then solves

        minimize   w' C w                       (combined variance)
        subject to sum(w) = 1,  w_i >= 0
        tie-break  minimize sum(w_i**2)         (among equal-variance optima)

The tie-break picks the most diffuse weight vector when several weights
achieve the same minimal variance (which happens, for example, with
duplicate submissions that carry identical error information).

Only NumPy is required.  Small batch counts use exact support enumeration
on the weight simplex; larger problems use a ridge-regularized pairwise
Frank-Wolfe iteration, with a primal active-set routine for the
equality-constrained minimum-norm step.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from uncertainties import UFloat, covariance_matrix, ufloat


@dataclass
class ErrorSource:
    """An error source shared by zero, one or several batches.

    tag: unique identity of the source (e.g. a calibration artefact id).
    std_dev: standard deviation of the zero-mean error contributed by it.
    batches: ids of the batches affected by this source.
    """

    tag: object
    std_dev: float
    batches: tuple

    def __post_init__(self):
        if self.std_dev < 0:
            raise ValueError("error source standard deviations must be >= 0")
        self.batches = tuple(self.batches)


@dataclass
class BatchResult:
    """A single batch mean together with its private (unshared) error."""

    batch_id: object
    value: float
    private_std_dev: float = 0.0

    def __post_init__(self):
        if self.private_std_dev < 0:
            raise ValueError("private standard deviations must be >= 0")


def build_measurements(batches, error_sources):
    """Build tagged UFloat measurements from batches and error sources.

    Each returned measurement is

        value_i + private_i + sum(source_j for sources affecting batch i)

    where every source is a tagged zero-mean variable.  Sharing the same
    source object between batches is what makes their errors correlated;
    duplicate submissions that share every source therefore end up perfectly
    correlated and cannot gain precision from one another.
    """

    batch_ids = tuple(batch.batch_id for batch in batches)
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("batch ids must be unique")

    source_vars = {}
    for source in error_sources:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            source_vars[source.tag] = ufloat(
                0.0, source.std_dev, str(source.tag)
            )

    measurements = []
    for batch in batches:
        if batch.private_std_dev > 0.0:
            measurement = ufloat(
                batch.value,
                batch.private_std_dev,
                f"private:{batch.batch_id}",
            )
        else:
            # A resubmitted sample carries no independent repeat noise:
            # start from the plain nominal value and let the shared sources
            # explain all of its variance.
            measurement = float(batch.value)
        for source in error_sources:
            if batch.batch_id in source.batches:
                measurement = measurement + source_vars[source.tag]
        if not isinstance(measurement, UFloat):
            # Deterministic batch with neither private nor shared error;
            # a zero-variance variable is intentional here, so silence the
            # library's zero-standard-deviation warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                measurement = ufloat(
                    batch.value, 0.0, f"private:{batch.batch_id}"
                )
        measurements.append(measurement)

    return measurements


def covariance_from_sources(batches, error_sources):
    """Covariance matrix of the batch means implied by the error sources."""

    measurements = build_measurements(batches, error_sources)
    return np.array(covariance_matrix(measurements), dtype=float)


def _independent_rows(equalities, rhs, tol):
    """Drop linearly dependent equality rows and check consistency.

    Rows are selected greedily by row-space independence.  A feasible full
    system makes every selected subset feasible too, so consistency of the
    reduced system is verified directly with a least-squares residual rather
    than by reconstructing the dropped right-hand sides.
    """

    keep = []
    for index, row in enumerate(equalities):
        if not keep:
            keep.append(index)
            continue
        basis = equalities[keep].T
        coeff, *_ = np.linalg.lstsq(basis, row, rcond=None)
        residual = np.max(np.abs(basis @ coeff - row))
        if residual > tol * max(1.0, np.max(np.abs(row))):
            keep.append(index)

    keep = np.array(keep)
    reduced_a = equalities[keep]
    reduced_b = rhs[keep]
    solution, *_ = np.linalg.lstsq(reduced_a, reduced_b, rcond=None)
    mismatch = np.max(np.abs(reduced_a @ solution - reduced_b))
    if mismatch > 1e-8 * max(1.0, np.max(np.abs(rhs))):
        raise ValueError("inconsistent equality constraints")
    return reduced_a, reduced_b


def _kkt_solve(quadratic, equalities, rhs, free):
    """Equality-constrained QP step on a set of free variables.

    Minimizes 1/2 x' Q x subject to A x = b with every variable outside
    `free` fixed at zero.  Q restricted to the free set must be positive
    definite and A must have full row rank, so the KKT system is
    nonsingular.
    """

    free = np.asarray(free)
    sub_q = quadratic[np.ix_(free, free)]
    sub_a = equalities[:, free]
    size = sub_q.shape[0]
    n_constraints = sub_a.shape[0]

    kkt = np.zeros((size + n_constraints, size + n_constraints))
    kkt[:size, :size] = sub_q
    kkt[:size, size:] = sub_a.T
    kkt[size:, :size] = sub_a
    target = np.concatenate((np.zeros(size), rhs))
    try:
        solution = np.linalg.solve(kkt, target)
    except np.linalg.LinAlgError:
        # Only reached from the enumeration fallback when a candidate
        # support makes the restricted constraints rank deficient.
        solution, *_ = np.linalg.lstsq(kkt, target, rcond=None)

    target_x = np.zeros(quadratic.shape[0])
    target_x[free] = solution[:size]
    multipliers = solution[size:]
    return target_x, multipliers


def _pairwise_frank_wolfe(quadratic, x, tol, max_iter=100000):
    """Pairwise Frank-Wolfe on the simplex (away-step variant).

    Each step moves mass from the active vertex with the largest gradient
    to the vertex with the smallest one.  For a positive-definite Q this
    converges linearly, so it reaches a tight KKT residual even in high
    dimensions, unlike vanilla Frank-Wolfe whose O(1/t) rate stalls.
    """

    x = np.array(x, dtype=float)
    for _ in range(max_iter):
        gradient = quadratic @ x
        forward = int(np.argmin(gradient))
        active = np.flatnonzero(x > 1e-14)
        away = int(active[np.argmax(gradient[active])])
        gap = gradient[away] - gradient[forward]
        if gap <= tol:
            return x
        direction = np.zeros_like(x)
        direction[forward] = 1.0
        direction[away] -= 1.0
        curvature = direction @ quadratic @ direction
        step = min(gap / curvature, x[away])
        x = x + step * direction
        x[away] = max(0.0, x[away])
    return x


def _enumeration_qp(quadratic, equalities, rhs, tol):
    """Exact fallback: evaluate every support, keep the feasible minimizer.

    Batch counts are small, so the exponential cost is only paid if the
    active-set method ever cycles, which redundant-row removal should make
    impossible in practice.
    """

    n = quadratic.shape[0]
    best_x = None
    best_objective = np.inf
    for mask in range(1, 1 << n):
        free = np.array([i for i in range(n) if mask & (1 << i)])
        candidate, _ = _kkt_solve(quadratic, equalities, rhs, free)
        if np.any(candidate < -tol):
            continue
        if np.max(np.abs(equalities @ candidate - rhs)) > tol * max(
            1.0, np.max(np.abs(rhs))
        ):
            continue
        objective = 0.5 * candidate @ quadratic @ candidate
        if objective < best_objective:
            best_objective = objective
            best_x = np.maximum(candidate, 0.0)
    if best_x is None:  # pragma: no cover - the start point is feasible
        raise RuntimeError("no feasible point found for the constraints")
    return best_x


def constrained_qp(quadratic, equalities, rhs, start, tol=None):
    """Minimize a positive-definite quadratic under A x = b, x >= 0.

    A primal active-set method started from the feasible point `start`.
    Variables that reach zero enter the working set and are only released
    when their gradient shows that doing so decreases the objective.  With
    a positive-definite Q the method terminates at the unique optimizer;
    redundant equality rows are removed up front (duplicate covariance rows
    would otherwise make A rank deficient), and a working-set cycle falls
    back to Frank-Wolfe (single equality) or support enumeration.
    """

    n = quadratic.shape[0]
    if tol is None:
        tol = 1e-10 * max(1.0, float(np.max(np.abs(quadratic))))
    equalities, rhs = _independent_rows(equalities, rhs, tol)

    x = np.array(start, dtype=float)
    working = {i for i in range(n) if x[i] <= 1e-12}
    seen_states = set()

    for _ in range(100 * n + 50):
        state = frozenset(working)
        if state in seen_states:
            if equalities.shape[0] == 1:
                return _pairwise_frank_wolfe(quadratic, x, tol)
            return _enumeration_qp(quadratic, equalities, rhs, tol)
        seen_states.add(state)

        free = np.array([i for i in range(n) if i not in working])
        target_x, equality_nu = _kkt_solve(
            quadratic, equalities, rhs, free
        )
        gap = target_x - x

        if np.max(np.abs(gap)) <= tol:
            # Stationary on the free set: an active variable may be
            # released when its reduced gradient (objective gradient after
            # accounting for the equality multipliers) is negative, i.e.
            # giving it mass would decrease the objective.
            reduced_gradient = (
                quadratic @ x + equalities.T @ equality_nu
            )
            candidates = [
                i for i in working if reduced_gradient[i] < -tol
            ]
            if not candidates:
                return x
            working.remove(
                min(candidates, key=lambda i: reduced_gradient[i])
            )
            continue

        step = 1.0
        blocking = None
        for i in free:
            if gap[i] < -1e-12:
                boundary = -x[i] / gap[i]
                if boundary < step:
                    step = boundary
                    blocking = i

        x = x + step * gap
        if blocking is not None and step < 1.0:
            x[blocking] = 0.0
            working.add(blocking)

    if equalities.shape[0] == 1:
        return _pairwise_frank_wolfe(quadratic, x, tol)
    return _enumeration_qp(quadratic, equalities, rhs, tol)


def _variance_vertex_on_support(cov, free):
    """Min w' C w on one support with 1'w = 1; None if a weight is negative."""

    sub_cov = cov[np.ix_(free, free)]
    size = free.size
    kkt = np.zeros((size + 1, size + 1))
    kkt[:size, :size] = 2.0 * sub_cov
    kkt[:size, size] = 1.0
    kkt[size, :size] = 1.0
    target = np.concatenate((np.zeros(size), np.ones(1)))
    try:
        solution = np.linalg.solve(kkt, target)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(kkt, target, rcond=None)
    sub_weights = solution[:size]
    if np.any(sub_weights < -1e-9):
        return None
    weights = np.zeros(cov.shape[0])
    weights[free] = np.maximum(sub_weights, 0.0)
    return weights


def _min_norm_face_by_enumeration(cov, tol):
    """Exact minimum-variance, minimum-norm solution via enumeration.

    Phase A walks every nonempty support S and solves the variance problem
    restricted to S; feasible supports whose variance reaches the global
    minimum give the vertices of the optimal face.  At an optimum the KKT
    multiplier equals the optimal variance itself (w' C w = lambda * 1' w
    = lambda), so phase B solves  C_F w = v* 1,  1'w = 1  on that face and
    minimizes the squared norm, starting from a feasible relative-interior
    point (the mean of the optimal vertices).  Exponential cost, used for
    small batch counts where rank-deficient covariance rows make iterative
    face identification ill conditioned.
    """

    n = cov.shape[0]
    vertices = []
    variances = []
    for mask in range(1, 1 << n):
        free = np.array([i for i in range(n) if mask & (1 << i)])
        weights = _variance_vertex_on_support(cov, free)
        if weights is None:
            continue
        vertices.append(weights)
        variances.append(float(weights @ cov @ weights))

    if not vertices:  # pragma: no cover - the full support always works
        raise RuntimeError("could not identify the optimal face")

    optimal_variance = min(variances)
    optimal_vertices = [
        weights
        for weights, variance in zip(vertices, variances)
        if abs(variance - optimal_variance)
        <= tol * max(1.0, abs(optimal_variance))
    ]
    face = np.array(
        sorted(
            {
                i
                for weights in optimal_vertices
                for i in np.flatnonzero(weights)
            }
        )
    )

    # Mean of optimal vertices: a relative-interior point with positive
    # weight on every face coordinate, hence an exactly feasible start for
    # the stationarity equalities C_F w = v* * 1.
    interior = np.mean(optimal_vertices, axis=0)[face]
    face_cov = cov[np.ix_(face, face)]
    equalities = np.vstack((np.ones((1, face.size)), face_cov))
    rhs = np.concatenate(
        (np.ones(1), optimal_variance * np.ones(face.size))
    )
    face_weights = constrained_qp(
        2.0 * np.eye(face.size), equalities, rhs, interior, tol=tol
    )

    weights = np.zeros(n)
    weights[face] = face_weights
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()
    return weights, float(weights @ cov @ weights)


def _symmetrize_identical_rows(cov, weights, tol):
    """Average weights over numerically identical covariance rows.

    If two rows of C are identical, swapping their weights leaves the
    variance unchanged; the minimum squared-norm optimum therefore assigns
    them equal weight.  Averaging inside each identical-row group enforces
    that tie-break exactly and preserves the unit sum automatically.
    """

    weights = np.array(weights, dtype=float)
    n = weights.size
    assigned = np.zeros(n, dtype=bool)
    for i in range(n):
        if assigned[i]:
            continue
        same = [
            j
            for j in range(i, n)
            if np.allclose(cov[i], cov[j], rtol=0.0, atol=tol)
        ]
        group = np.array(same)
        weights[group] = weights[group].mean()
        assigned[group] = True
    return weights


def optimal_simplex_weights(cov):
    """Non-negative, unit-sum weights minimizing w' C w.

    Ties (several weight vectors reaching the same minimal variance) are
    resolved in favor of the vector with the smallest squared norm.

    For the typical small batch count (up to 12) the answer is obtained by
    exact support enumeration, which remains correct when the covariance is
    rank deficient because duplicate submissions produce identical rows.
    A Tikhonov-ridge pairwise Frank-Wolfe path covers larger problems;
    exact duplicate rows still split their weight equally there by symmetry.
    """

    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if cov.shape != (n, n):
        raise ValueError("the covariance matrix must be square")
    if n == 0:
        raise ValueError("at least one batch is required")

    cov = 0.5 * (cov + cov.T)  # Guard against tiny asymmetries.
    scale = max(1.0, float(np.max(np.diag(cov))))
    tol = 1e-9 * scale

    if n <= 12:
        weights, _ = _min_norm_face_by_enumeration(cov, tol)
    else:
        # Large batch count: ridge regularization restores strict convexity
        # and pairwise Frank-Wolfe solves the strictly convex simplex QP
        # with linear convergence; its minimum-norm bias as the ridge tends
        # to zero is the requested tie-break.
        ridge = 1e-10 * scale
        regularized = 2.0 * (cov + ridge * np.eye(n))
        weights = _pairwise_frank_wolfe(
            regularized, np.full(n, 1.0 / n), tol
        )

    weights = _symmetrize_identical_rows(cov, weights, tol)
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()

    # Clamp away round-off noise on rank-deficient matrices, where the true
    # optimal variance is exactly zero but w' C w can come out at ~1e-16.
    optimal_variance = max(0.0, float(weights @ cov @ weights))
    return weights, optimal_variance


def combine_batches(batches, error_sources):
    """Return the optimal weighted combination of the batch means.

    batches: sequence of BatchResult.
    error_sources: sequence of ErrorSource describing shared errors.

    Returns a dict with the per-batch weights, the combined nominal value,
    the combined standard deviation, the covariance matrix and the tagged
    measurements.
    """

    batches = tuple(batches)
    measurements = build_measurements(batches, error_sources)
    cov = np.array(covariance_matrix(measurements), dtype=float)
    weights, variance = optimal_simplex_weights(cov)

    combined_value = float(
        sum(w * b.value for w, b in zip(weights, batches))
    )
    return {
        "batch_ids": tuple(b.batch_id for b in batches),
        "weights": weights,
        "combined_value": combined_value,
        "combined_variance": variance,
        "combined_std_dev": float(np.sqrt(variance)),
        "covariance": cov,
        "measurements": measurements,
    }


def format_report(result, *, precision=4):
    """Human-readable report showing each batch's effective weight."""

    lines = [
        "batch weights (non-negative, sum to 1, minimum combined variance):",
    ]
    for batch_id, weight in zip(result["batch_ids"], result["weights"]):
        lines.append(f"  {str(batch_id):>12s}  w = {weight:.{precision}f}")
    lines.append(
        f"  combined estimate = {result['combined_value']:.{precision}f}"
        f" +/- {result['combined_std_dev']:.{precision}f}"
    )
    lines.append(
        f"  combined variance = {result['combined_variance']:.{precision}f}"
    )
    return "\n".join(lines)
