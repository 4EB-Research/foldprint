from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from protein_sae_pipeline.scoring import usalign
from protein_sae_pipeline.scoring.usalign import (
    USAlignError,
    classify_tm_score,
    parse_usalign_output,
    run_usalign,
)

USALIGN_OUTPUT = """\
********************************************************************
 * US-align (Version 20260819)                                      *
********************************************************************
Name of Structure_1: candidate_ca.pdb:A (to be superimposed onto Structure_2)
Name of Structure_2: wt_ca.pdb:A
Length of Structure_1: 2 residues
Length of Structure_2: 3 residues
Aligned length= 2, RMSD= 1.25, Seq_ID=n_identical/n_aligned= 0.500
TM-score= 0.71234 (normalized by length of Structure_1: L=2, d0=0.50)
TM-score= 0.82345 (normalized by length of Structure_2: L=3, d0=0.50)
"""


def atom_line(
    serial: int,
    atom: str,
    residue: str,
    chain: str,
    residue_number: int,
    xyz: tuple[float, float, float],
) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} {chain:1s}{residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          C  "
    )


def write_ca_structure(path: Path, count: int, *, residue: str = "ALA") -> None:
    lines = [
        atom_line(
            position,
            "CA",
            residue,
            "Q",
            10 + position,
            (float(position), float(position % 2), 0.0),
        )
        for position in range(1, count + 1)
    ]
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_labeled_parser_selects_structure_2_as_primary() -> None:
    # The labeled parser must remain correct even if the two score lines are
    # presented in the opposite order.
    lines = USALIGN_OUTPUT.splitlines()
    lines[-2], lines[-1] = lines[-1], lines[-2]

    result = parse_usalign_output("\n".join(lines))

    assert result.candidate_coordinate_length == 2
    assert result.wt_coordinate_length == 3
    assert result.tm_score_candidate_normalized == pytest.approx(0.71234)
    assert result.tm_score_wt_normalized == pytest.approx(0.82345)
    assert result.primary_tm_score == pytest.approx(0.82345)
    assert result.aligned_length == 2
    assert result.aligned_fraction_candidate == pytest.approx(1.0)
    assert result.aligned_fraction_wt == pytest.approx(2 / 3)
    assert result.rmsd_angstrom == pytest.approx(1.25)
    assert result.aligned_sequence_identity == pytest.approx(0.5)
    assert result.version == "20260819"


def test_parser_refuses_to_guess_from_unlabeled_tm_scores() -> None:
    unlabeled = USALIGN_OUTPUT.replace(
        " (normalized by length of Structure_2: L=3, d0=0.50)", ""
    )

    with pytest.raises(USAlignError, match="both structures"):
        parse_usalign_output(unlabeled)


def test_tm_classification_is_inclusive_at_threshold() -> None:
    assert classify_tm_score(0.499999) == "negative"
    assert classify_tm_score(0.5) == "analogue"
    assert classify_tm_score(0.9) == "analogue"
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        classify_tm_score(float("nan"))


def test_run_usalign_writes_candidate_as_1_and_wt_as_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = tmp_path / "candidate.pdb"
    wt_path = tmp_path / "wt.pdb"
    write_ca_structure(candidate_path, 2)
    write_ca_structure(wt_path, 3)
    observed: dict[str, object] = {}

    def fake_run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        candidate_lines = Path(command[1]).read_text(encoding="ascii").splitlines()
        wt_lines = Path(command[2]).read_text(encoding="ascii").splitlines()
        observed["candidate_atom_count"] = sum(line.startswith("ATOM") for line in candidate_lines)
        observed["wt_atom_count"] = sum(line.startswith("ATOM") for line in wt_lines)
        observed["candidate_residue_numbers"] = [
            int(line[22:26]) for line in candidate_lines if line.startswith("ATOM")
        ]
        observed["wt_residue_numbers"] = [
            int(line[22:26]) for line in wt_lines if line.startswith("ATOM")
        ]
        return subprocess.CompletedProcess(command, 0, stdout=USALIGN_OUTPUT, stderr="")

    monkeypatch.setattr(usalign.subprocess, "run", fake_run)

    result = run_usalign(
        candidate_path,
        wt_path,
        executable="/opt/usalign/USalign",
        temp_root=tmp_path / "temporary",
    )

    command = observed["command"]
    assert isinstance(command, tuple)
    assert command[0] == "/opt/usalign/USalign"
    assert Path(command[1]).name == "candidate_ca.pdb"
    assert Path(command[2]).name == "wt_ca.pdb"
    assert command[3:] == ("-mol", "prot")
    assert "-TMscore" not in command
    assert observed["candidate_atom_count"] == 2
    assert observed["wt_atom_count"] == 3
    assert observed["candidate_residue_numbers"] == [1, 2]
    assert observed["wt_residue_numbers"] == [1, 2, 3]
    assert result.primary_tm_score == pytest.approx(0.82345)
    assert result.command == command


def test_run_usalign_checks_reported_coordinate_lengths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = tmp_path / "candidate.pdb"
    wt_path = tmp_path / "wt.pdb"
    write_ca_structure(candidate_path, 3)
    write_ca_structure(wt_path, 3)

    monkeypatch.setattr(
        usalign.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=USALIGN_OUTPUT, stderr=""
        ),
    )

    with pytest.raises(USAlignError, match="Structure_1 length disagrees"):
        run_usalign(candidate_path, wt_path, temp_root=tmp_path / "temporary")
