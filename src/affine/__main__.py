"""Command line interface for grid alignment."""

from __future__ import annotations

import argparse
import json
import sys

from affine import Affine, GridAlignmentError, align_grid


def _affine(value: str) -> Affine:
    try:
        coefficients = [float(item) for item in value.replace(",", " ").split()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid affine: {value!r}") from None
    if len(coefficients) != 6:
        raise argparse.ArgumentTypeError(
            "an affine needs 6 comma-separated coefficients: a,b,c,d,e,f"
        )
    return Affine(*coefficients)


def _shape(value: str) -> tuple[int, int]:
    try:
        parts = [int(item) for item in value.replace(",", " ").split()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid shape: {value!r}") from None
    if len(parts) != 2 or parts[0] <= 0 or parts[1] <= 0:
        raise argparse.ArgumentTypeError("a shape needs positive rows,cols")
    return (parts[0], parts[1])


def _window(window) -> dict[str, int]:
    return {
        "row_start": window.row_start,
        "row_stop": window.row_stop,
        "col_start": window.col_start,
        "col_stop": window.col_stop,
        "shape": list(window.shape),
        "pixels": window.count,
    }


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the command line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m affine",
        description="Align two raster pixel grids by whole pixels only.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    grid = subparsers.add_parser(
        "grid",
        help="report whether two Affine/shape pairs share a pixel grid",
    )
    grid.add_argument("--a", type=_affine, required=True, metavar="AFFINE")
    grid.add_argument("--shape-a", type=_shape, required=True, metavar="ROWS,COLS")
    grid.add_argument("--b", type=_affine, required=True, metavar="AFFINE")
    grid.add_argument("--shape-b", type=_shape, required=True, metavar="ROWS,COLS")
    grid.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        metavar="PIXELS",
        help="maximum allowed grid-node deviation in pixels (default: 1e-6)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface and return an exit status."""
    args = build_parser().parse_args(argv)
    if args.command != "grid":
        return 2

    try:
        alignment = align_grid(
            args.a, args.shape_a, args.b, args.shape_b, tolerance=args.tolerance
        )
    except GridAlignmentError as error:
        print(json.dumps({"aligned": False, "reason": str(error)}, indent=2))
        return 1

    print(
        json.dumps(
            {
                "aligned": True,
                "column_mapping": list(alignment.column_mapping),
                "row_mapping": list(alignment.row_mapping),
                "permutation": list(alignment.permutation),
                "offset": list(alignment.offset),
                "worst_case_deviation": alignment.worst_case_deviation,
                "worst_case_deviation_pixels": alignment.worst_case_deviation_pixels,
                "tolerance_pixels": alignment.tolerance,
                "overlap_a": _window(alignment.overlap_a),
                "overlap_b": _window(alignment.overlap_b),
                "overlap_pixels": alignment.overlap_pixels,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
