from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from protein_sae_pipeline.structures.pdb import (
    PDBValidationError,
    parse_pdb,
    write_sequential_ca_pdb,
)


def atom_line(
    serial: int,
    atom: str,
    residue: str,
    chain: str,
    residue_number: int,
    xyz: tuple[float, float, float],
    *,
    record: str = "ATOM",
    altloc: str = "",
    occupancy: float = 1.0,
    insertion_code: str = "",
) -> str:
    x, y, z = xyz
    return (
        f"{record:<6s}{serial:5d} {atom:^4s}{altloc:1s}{residue:>3s} {chain:1s}"
        f"{residue_number:4d}{insertion_code:1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{0.0:6.2f}          C  "
    )


def test_parser_uses_first_model_and_normalizes_mse(tmp_path: Path) -> None:
    path = tmp_path / "models.pdb"
    lines = [
        "MODEL        7",
        atom_line(1, "N", "ALA", "A", 5, (0, 0, 0)),
        atom_line(2, "CA", "ALA", "A", 5, (1, 2, 3)),
        atom_line(3, "C", "ALA", "A", 5, (2, 2, 3)),
        atom_line(4, "N", "MSE", "A", 9, (3, 4, 5), record="HETATM"),
        atom_line(5, "CA", "MSE", "A", 9, (4, 5, 6), record="HETATM"),
        atom_line(6, "C", "MSE", "A", 9, (5, 5, 6), record="HETATM"),
        # Non-MSE HETATM records are not protein residues.
        atom_line(7, "CA", "LIG", "A", 10, (99, 99, 99), record="HETATM"),
        "ENDMDL",
        "MODEL        8",
        atom_line(8, "CA", "VAL", "B", 1, (8, 8, 8)),
        "ENDMDL",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")

    parsed = parse_pdb(path, expected_sequence="AM")

    assert parsed.model_id == 7
    assert parsed.chain_id == "A"
    assert parsed.sequence == "AM"
    assert parsed.residues[1].residue_name == "MSE"
    assert parsed.residues[1].canonical_residue_name == "MET"
    np.testing.assert_allclose(parsed.ca_coordinates, [[1, 2, 3], [4, 5, 6]])
    assert not parsed.ca_coordinates.flags.writeable


def test_parser_normalizes_location_specific_modres_hetatm(tmp_path: Path) -> None:
    path = tmp_path / "modified.pdb"
    # PDB columns: source residue 13--15, chain 17, residue 19--22,
    # insertion code 23, canonical parent 25--27.
    modres = "MODRES TEST SNN A   28  ASN  UNIT TEST"
    path.write_text(
        "\n".join(
            [
                modres,
                atom_line(1, "N", "SNN", "A", 28, (0, 0, 0), record="HETATM"),
                atom_line(2, "CA", "SNN", "A", 28, (1, 0, 0), record="HETATM"),
                atom_line(3, "C", "SNN", "A", 28, (2, 0, 0), record="HETATM"),
                # An undeclared HETATM remains ignored.
                atom_line(4, "CA", "LIG", "A", 29, (9, 9, 9), record="HETATM"),
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    parsed = parse_pdb(path, expected_sequence="N")

    assert len(parsed.residues) == 1
    assert parsed.residues[0].residue_name == "SNN"
    assert parsed.residues[0].canonical_residue_name == "ASN"
    assert parsed.sequence == "N"


def test_parser_altloc_rule_prefers_occupancy_then_blank(tmp_path: Path) -> None:
    path = tmp_path / "altloc.pdb"
    path.write_text(
        "\n".join(
            [
                atom_line(1, "N", "GLY", "A", 1, (0, 0, 0)),
                atom_line(2, "CA", "GLY", "A", 1, (4, 4, 4), altloc="A", occupancy=0.4),
                atom_line(3, "CA", "GLY", "A", 1, (6, 6, 6), altloc="B", occupancy=0.6),
                atom_line(4, "C", "GLY", "A", 1, (1, 1, 1), altloc="A", occupancy=0.5),
                atom_line(5, "C", "GLY", "A", 1, (2, 2, 2), altloc="", occupancy=0.5),
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    residue = parse_pdb(path).residues[0]

    assert residue.atoms["CA"].altloc == "B"
    np.testing.assert_allclose(residue.coordinate("CA"), [6, 6, 6])
    assert residue.atoms["C"].altloc == ""
    np.testing.assert_allclose(residue.coordinate("C"), [2, 2, 2])


def test_parser_rejects_ambiguous_multichain_input(tmp_path: Path) -> None:
    path = tmp_path / "two_chains.pdb"
    path.write_text(
        "\n".join(
            [
                atom_line(1, "CA", "ALA", "A", 1, (0, 0, 0)),
                atom_line(2, "CA", "GLY", "B", 1, (1, 0, 0)),
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    with pytest.raises(PDBValidationError, match="multiple protein chains"):
        parse_pdb(path)
    assert parse_pdb(path, chain_id="B").sequence == "G"


def test_sequential_ca_writer_is_contiguous_and_canonical(tmp_path: Path) -> None:
    source = tmp_path / "source.pdb"
    source.write_text(
        "\n".join(
            [
                atom_line(1, "CA", "ALA", "Q", 18, (1, 2, 3)),
                atom_line(2, "CA", "MSE", "Q", 21, (4, 5, 6), record="HETATM"),
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    parsed = parse_pdb(source)

    output = write_sequential_ca_pdb(parsed, tmp_path / "sequential.pdb")
    reparsed = parse_pdb(output)

    assert reparsed.chain_id == "A"
    assert [residue.author_id.residue_number for residue in reparsed.residues] == [1, 2]
    assert [residue.residue_name for residue in reparsed.residues] == ["ALA", "MET"]
    np.testing.assert_allclose(reparsed.ca_coordinates, [[1, 2, 3], [4, 5, 6]])
    assert all(line[12:16].strip() == "CA" for line in output.read_text().splitlines()[:2])
