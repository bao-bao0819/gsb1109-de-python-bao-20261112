"""Command line interface tests."""

import json

from affine.__main__ import main


def test_grid_command_reports_alignment(capsys):
    status = main(
        [
            "grid",
            "--a",
            "10,0,1000,0,-10,2000",
            "--shape-a",
            "10,10",
            "--b",
            "10,0,1030,0,-10,1980",
            "--shape-b",
            "10,10",
        ]
    )
    assert status == 0
    report = json.loads(capsys.readouterr().out)
    assert report["aligned"] is True
    assert report["column_mapping"] == [1, 0, -3]
    assert report["row_mapping"] == [0, 1, -2]
    assert report["overlap_a"]["shape"] == [8, 7]


def test_grid_command_reports_misalignment(capsys):
    status = main(
        [
            "grid",
            "--a",
            "10,0,0,0,-10,0",
            "--shape-a",
            "1000,1000",
            "--b",
            "10.0000001,0,0,0,-10,0",
            "--shape-b",
            "1000,1000",
        ]
    )
    assert status == 1
    report = json.loads(capsys.readouterr().out)
    assert report["aligned"] is False
    assert "tolerance" in report["reason"]


def test_grid_command_reports_edge_contact(capsys):
    status = main(
        [
            "grid",
            "--a",
            "10,0,0,0,-10,100",
            "--shape-a",
            "10,10",
            "--b",
            "10,0,100,0,-10,100",
            "--shape-b",
            "10,10",
        ]
    )
    assert status == 0
    report = json.loads(capsys.readouterr().out)
    assert report["overlap_pixels"] == 0
