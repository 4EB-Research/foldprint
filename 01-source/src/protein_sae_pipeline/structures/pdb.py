"""Dependency-light, deterministic parsing for PDB inputs.

The parser deliberately supports a small and auditable subset of the PDB format:

* only the first model is read;
* a structure must resolve to one protein chain (or an explicitly selected chain);
* canonical ``ATOM`` residues, ``MSE``, and location-specific ``MODRES`` records
  with canonical parents are protein;
* undeclared non-MSE ``HETATM`` records are ignored;
* alternate locations are selected by occupancy, then blank, then ``A``, then
  lexical altloc value; and
* every parsed protein residue must have a C-alpha atom.

This is intentionally stricter than a general structural-biology parser.  It makes
the atoms passed to US-align and the local-RMSD calculation deterministic and keeps
unexpected input records from silently changing a result.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Iterable, Mapping, TypeAlias

import numpy as np


THREE_TO_ONE: Mapping[str, str] = MappingProxyType(
    {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "MSE": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
)
ONE_TO_THREE: Mapping[str, str] = MappingProxyType(
    {
        one_letter: three_letter
        for three_letter, one_letter in THREE_TO_ONE.items()
        if three_letter != "MSE"
    }
)
CANONICAL_AMINO_ACIDS = frozenset(ONE_TO_THREE)


class PDBValidationError(ValueError):
    """Raised when a PDB is outside the pipeline's strict input contract."""


@dataclass(frozen=True, order=True)
class ResidueID:
    """An author residue identifier from a PDB coordinate record."""

    chain_id: str
    residue_number: int
    insertion_code: str = ""

    def __post_init__(self) -> None:
        chain = str(self.chain_id).strip()
        insertion = str(self.insertion_code).strip().upper()
        if len(chain) > 1:
            raise ValueError("A PDB chain ID must be zero or one character.")
        if isinstance(self.residue_number, bool) or not isinstance(self.residue_number, int):
            raise TypeError("A PDB residue number must be an integer.")
        if len(insertion) > 1:
            raise ValueError("A PDB insertion code must be zero or one character.")
        object.__setattr__(self, "chain_id", chain)
        object.__setattr__(self, "insertion_code", insertion)

    @property
    def label(self) -> str:
        chain = self.chain_id if self.chain_id else "<blank>"
        return f"{chain}:{self.residue_number}{self.insertion_code}"


ResidueIDLike: TypeAlias = ResidueID | tuple[str, int] | tuple[str, int, str] | str

_RESIDUE_ID_RE = re.compile(r"^(?P<chain>.?):(?P<number>-?\d+)(?P<icode>[A-Za-z]?)$")


def coerce_residue_id(value: ResidueIDLike) -> ResidueID:
    """Normalize a public-schema residue locator to :class:`ResidueID`.

    Strings use ``CHAIN:NUMBER`` or ``CHAIN:NUMBERICODE`` (for example ``A:34``
    or ``A:34B``).  A blank-chain record should be represented with a tuple.
    """

    if isinstance(value, ResidueID):
        return value
    if isinstance(value, str):
        match = _RESIDUE_ID_RE.fullmatch(value.strip())
        if match is None:
            raise ValueError(
                f"Invalid residue locator {value!r}; expected a value such as 'A:34'."
            )
        return ResidueID(
            match.group("chain"),
            int(match.group("number")),
            match.group("icode"),
        )
    if isinstance(value, tuple) and len(value) in {2, 3}:
        chain_id = value[0]
        residue_number = value[1]
        insertion_code = value[2] if len(value) == 3 else ""
        return ResidueID(str(chain_id), residue_number, str(insertion_code))
    raise TypeError(f"Unsupported residue locator: {value!r}.")


def normalize_sequence(sequence: str) -> str:
    """Return an uppercase, whitespace-free canonical protein sequence."""

    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError("A protein sequence cannot be empty.")
    invalid = sorted(set(normalized) - CANONICAL_AMINO_ACIDS)
    if invalid:
        raise ValueError(
            "Protein sequence contains unsupported residue code(s): " + ", ".join(invalid)
        )
    return normalized


@dataclass(frozen=True)
class PDBAtom:
    """The deterministically selected coordinate for one named atom."""

    name: str
    coordinate: np.ndarray
    occupancy: float
    altloc: str

    def __post_init__(self) -> None:
        name = str(self.name).strip().upper()
        altloc = str(self.altloc).strip().upper()
        coordinate = np.asarray(self.coordinate, dtype=np.float64)
        occupancy = float(self.occupancy)
        if not name:
            raise PDBValidationError("A parsed PDB atom must have a name.")
        if len(altloc) > 1:
            raise PDBValidationError("A PDB alternate-location code must be one character.")
        if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
            raise PDBValidationError("A PDB atom coordinate must be a finite length-three vector.")
        if not math.isfinite(occupancy):
            raise PDBValidationError("A PDB atom occupancy must be finite.")
        coordinate = np.array(coordinate, dtype=np.float64, copy=True)
        coordinate.flags.writeable = False
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "altloc", altloc)
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "coordinate", coordinate)


@dataclass(frozen=True)
class PDBResidue:
    """One protein residue and its deterministically selected atoms."""

    author_id: ResidueID
    residue_name: str
    canonical_residue_name: str
    atoms: Mapping[str, PDBAtom]

    def __post_init__(self) -> None:
        source_name = str(self.residue_name).strip().upper()
        canonical_name = str(self.canonical_residue_name).strip().upper()
        if not source_name or len(source_name) > 3:
            raise PDBValidationError(f"Invalid deposited residue name: {source_name!r}.")
        if canonical_name == "MSE" or canonical_name not in set(ONE_TO_THREE.values()):
            raise PDBValidationError(
                f"Residue parent must be canonical, not {canonical_name!r}."
            )
        normalized_atoms = {str(name).upper(): atom for name, atom in self.atoms.items()}
        if "CA" not in normalized_atoms:
            raise PDBValidationError(f"Residue {self.author_id.label} has no C-alpha atom.")
        if any(name != atom.name for name, atom in normalized_atoms.items()):
            raise PDBValidationError("PDB atom mapping keys must agree with atom names.")
        object.__setattr__(self, "residue_name", source_name)
        object.__setattr__(self, "canonical_residue_name", canonical_name)
        object.__setattr__(self, "atoms", MappingProxyType(normalized_atoms))

    @property
    def one_letter(self) -> str:
        return THREE_TO_ONE[self.canonical_residue_name]

    def coordinate(self, atom_name: str) -> np.ndarray:
        normalized = str(atom_name).strip().upper()
        try:
            return self.atoms[normalized].coordinate
        except KeyError as exc:
            raise PDBValidationError(
                f"Residue {self.author_id.label} lacks required atom {normalized}."
            ) from exc


@dataclass(frozen=True)
class ParsedStructure:
    """A strict first-model, single-chain protein structure."""

    model_id: str | int
    chain_id: str
    residues: tuple[PDBResidue, ...]
    source_path: Path | None = None

    def __post_init__(self) -> None:
        residues = tuple(self.residues)
        if not residues:
            raise PDBValidationError("A parsed protein structure cannot be empty.")
        chain = str(self.chain_id).strip()
        if any(residue.author_id.chain_id != chain for residue in residues):
            raise PDBValidationError("All parsed residues must belong to the selected chain.")
        ids = tuple(residue.author_id for residue in residues)
        if len(ids) != len(set(ids)):
            raise PDBValidationError("A parsed structure contains duplicate author residue IDs.")
        object.__setattr__(self, "chain_id", chain)
        object.__setattr__(self, "residues", residues)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def sequence(self) -> str:
        """Canonical sequence of coordinate-observed residues, in PDB order."""

        return "".join(residue.one_letter for residue in self.residues)

    @property
    def ca_coordinates(self) -> np.ndarray:
        coordinates = np.asarray(
            [residue.coordinate("CA") for residue in self.residues], dtype=np.float64
        )
        coordinates.flags.writeable = False
        return coordinates

    def residue_by_author_id(self, value: ResidueIDLike) -> PDBResidue:
        author_id = coerce_residue_id(value)
        for residue in self.residues:
            if residue.author_id == author_id:
                return residue
        raise PDBValidationError(f"Structure lacks author residue {author_id.label}.")

    def residue_at_position(self, position_1_based: int) -> PDBResidue:
        if isinstance(position_1_based, bool) or not isinstance(position_1_based, int):
            raise TypeError("A biological sequence position must be an integer.")
        if not 1 <= position_1_based <= len(self.residues):
            raise PDBValidationError(
                f"Sequence position {position_1_based} is outside 1..{len(self.residues)}."
            )
        return self.residues[position_1_based - 1]


@dataclass
class _ResidueBuilder:
    segment_id: int
    author_id: ResidueID
    residue_name: str
    canonical_residue_name: str
    # atom name -> altloc -> (occupancy, coordinate)
    atom_candidates: dict[str, dict[str, tuple[float, np.ndarray]]]


@dataclass(frozen=True)
class _Modres:
    author_id: ResidueID
    source_residue_name: str
    canonical_residue_name: str


def _model_id(line: str) -> str | int:
    raw = line[10:14].strip() if len(line) >= 14 else line[6:].strip()
    if not raw:
        return 1
    try:
        return int(raw)
    except ValueError:
        return raw


def _first_model(lines: list[str]) -> tuple[str | int, list[tuple[int, str]]]:
    first_model_index = next(
        (index for index, line in enumerate(lines) if line[:6].strip().upper() == "MODEL"),
        None,
    )
    if first_model_index is None:
        selected: list[tuple[int, str]] = []
        for line_number, line in enumerate(lines, start=1):
            if line[:6].strip().upper() in {"END", "ENDMDL"}:
                break
            selected.append((line_number, line))
        return 1, selected

    selected = []
    for index in range(first_model_index + 1, len(lines)):
        line = lines[index]
        if line[:6].strip().upper() in {"END", "ENDMDL", "MODEL"}:
            break
        selected.append((index + 1, line))
    return _model_id(lines[first_model_index]), selected


def _parse_float(line: str, start: int, end: int, label: str, line_number: int) -> float:
    raw = line[start:end].strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise PDBValidationError(
            f"Invalid PDB {label} field {raw!r} on line {line_number}."
        ) from exc
    if not math.isfinite(value):
        raise PDBValidationError(
            f"Non-finite PDB {label} field {raw!r} on line {line_number}."
        )
    return value


def _parse_modres(lines: list[str], *, pdb_path: Path) -> dict[ResidueID, _Modres]:
    """Read location-specific modified-residue declarations from the PDB header."""

    declarations: dict[ResidueID, _Modres] = {}
    for line_number, line in enumerate(lines, start=1):
        if line[:6].strip().upper() != "MODRES":
            continue
        if len(line) < 27:
            raise PDBValidationError(
                f"Malformed MODRES record on line {line_number} of {pdb_path}."
            )
        source_name = line[12:15].strip().upper()
        chain = line[16].strip()
        raw_residue_number = line[18:22].strip()
        insertion_code = line[22].strip().upper()
        canonical_name = line[24:27].strip().upper()
        try:
            residue_number = int(raw_residue_number)
        except ValueError as exc:
            raise PDBValidationError(
                f"Invalid MODRES residue number {raw_residue_number!r} on line "
                f"{line_number} of {pdb_path}."
            ) from exc
        if not source_name or not canonical_name:
            raise PDBValidationError(
                f"Malformed MODRES residue names on line {line_number} of {pdb_path}."
            )
        author_id = ResidueID(chain, residue_number, insertion_code)
        declaration = _Modres(author_id, source_name, canonical_name)
        previous = declarations.get(author_id)
        if previous is not None and previous != declaration:
            raise PDBValidationError(
                f"Conflicting MODRES declarations at {author_id.label} in {pdb_path}."
            )
        declarations[author_id] = declaration
    return declarations


def _canonical_parent(
    residue_name: str,
    declaration: _Modres | None,
    *,
    author_id: ResidueID,
) -> str | None:
    """Return a canonical parent name, or ``None`` for an undeclared ligand."""

    canonical_names = frozenset(ONE_TO_THREE.values())
    if declaration is not None and declaration.source_residue_name != residue_name:
        raise PDBValidationError(
            f"MODRES declaration at {author_id.label} names "
            f"{declaration.source_residue_name!r}, not coordinate residue "
            f"{residue_name!r}."
        )
    if declaration is not None:
        parent = declaration.canonical_residue_name
        if parent not in canonical_names:
            raise PDBValidationError(
                f"Modified residue {residue_name!r} at {author_id.label} maps to "
                f"unsupported non-canonical parent {parent!r}."
            )
        return parent
    if residue_name == "MSE":
        return "MET"
    if residue_name in canonical_names:
        return residue_name
    return None


def _select_atom(
    atom_name: str,
    candidates: Mapping[str, tuple[float, np.ndarray]],
) -> PDBAtom:
    # Prefer occupancy, then blank altloc, then A, then lexical altloc.
    def rank(item: tuple[str, tuple[float, np.ndarray]]) -> tuple[float, int, int, str]:
        altloc, (occupancy, _coordinate) = item
        return (occupancy, int(altloc == ""), int(altloc == "A"), altloc)

    altloc, (occupancy, coordinate) = max(candidates.items(), key=rank)
    return PDBAtom(atom_name, coordinate, occupancy, altloc)


def parse_pdb(
    path: str | os.PathLike[str],
    *,
    chain_id: str | None = None,
    expected_sequence: str | None = None,
) -> ParsedStructure:
    """Parse one deterministic protein chain from the first PDB model.

    With no ``chain_id``, exactly one protein chain must be present.  Passing a
    chain explicitly selects only that chain, which is useful for unprepared
    reference PDBs.  Multiple TER-delimited protein segments within the selected
    chain are rejected in either mode.

    ``expected_sequence``, when supplied, is compared to the complete observed
    coordinate sequence.  This direct comparison is the default mapping used by
    the local-RMSD implementation.
    """

    pdb_path = Path(path).expanduser()
    if not pdb_path.is_file():
        raise PDBValidationError(f"PDB path is not a regular file: {pdb_path}.")
    requested_chain = None if chain_id is None else str(chain_id).strip()
    if requested_chain is not None and len(requested_chain) > 1:
        raise ValueError("A requested PDB chain ID must be zero or one character.")
    try:
        lines = pdb_path.read_text(encoding="ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise PDBValidationError(f"PDB is not ASCII text: {pdb_path}.") from exc
    except OSError as exc:
        raise PDBValidationError(f"Could not read PDB {pdb_path}: {exc}.") from exc

    modres = _parse_modres(lines, pdb_path=pdb_path)
    model_id, numbered_lines = _first_model(lines)
    builders: OrderedDict[tuple[int, ResidueID], _ResidueBuilder] = OrderedDict()
    segment_id = 0
    segment_active = False

    for line_number, line in numbered_lines:
        record = line[:6].strip().upper()
        if record == "TER":
            segment_active = False
            continue
        if record not in {"ATOM", "HETATM"}:
            continue
        if len(line) < 27:
            if record == "HETATM":
                continue
            raise PDBValidationError(
                f"Malformed ATOM residue identifier on line {line_number} of {pdb_path}."
            )
        record_chain = line[21].strip()
        if requested_chain is not None and record_chain != requested_chain:
            continue
        residue_name = line[17:20].strip().upper()
        raw_residue_number = line[22:26].strip()
        try:
            residue_number = int(raw_residue_number)
        except ValueError as exc:
            raise PDBValidationError(
                f"Invalid residue number {raw_residue_number!r} on line {line_number} "
                f"of {pdb_path}."
            ) from exc
        insertion_code = line[26].strip().upper()
        author_id = ResidueID(record_chain, residue_number, insertion_code)
        declaration = modres.get(author_id)
        if record == "HETATM" and declaration is None and residue_name != "MSE":
            continue
        canonical_name = _canonical_parent(
            residue_name,
            declaration,
            author_id=author_id,
        )
        if canonical_name is None:
            raise PDBValidationError(
                f"Unsupported protein residue {residue_name!r} on line {line_number} "
                f"of {pdb_path}."
            )
        if not segment_active:
            segment_id += 1
            segment_active = True
        key = (segment_id, author_id)
        builder = builders.get(key)
        if builder is None:
            builder = _ResidueBuilder(
                segment_id=segment_id,
                author_id=author_id,
                residue_name=residue_name,
                canonical_residue_name=canonical_name,
                atom_candidates={},
            )
            builders[key] = builder
        elif (
            builder.residue_name != residue_name
            or builder.canonical_residue_name != canonical_name
        ):
            raise PDBValidationError(
                f"Conflicting residue names at {author_id.label} in {pdb_path}."
            )

        atom_name = line[12:16].strip().upper() if len(line) >= 16 else ""
        if not atom_name:
            raise PDBValidationError(
                f"Missing atom name on line {line_number} of {pdb_path}."
            )
        if len(line) < 54:
            raise PDBValidationError(
                f"Malformed coordinate record on line {line_number} of {pdb_path}."
            )
        altloc = line[16].strip().upper()
        atom_candidates = builder.atom_candidates.setdefault(atom_name, {})
        if altloc in atom_candidates:
            raise PDBValidationError(
                f"Duplicate {atom_name} altloc {altloc or '<blank>'} at "
                f"{author_id.label} in {pdb_path}."
            )
        coordinate = np.asarray(
            [
                _parse_float(line, 30, 38, "x", line_number),
                _parse_float(line, 38, 46, "y", line_number),
                _parse_float(line, 46, 54, "z", line_number),
            ],
            dtype=np.float64,
        )
        occupancy_raw = line[54:60].strip() if len(line) >= 60 else ""
        if occupancy_raw:
            try:
                occupancy = float(occupancy_raw)
            except ValueError as exc:
                raise PDBValidationError(
                    f"Invalid occupancy {occupancy_raw!r} on line {line_number}."
                ) from exc
        else:
            occupancy = 0.0
        if not math.isfinite(occupancy):
            raise PDBValidationError(
                f"Non-finite occupancy {occupancy_raw!r} on line {line_number}."
            )
        atom_candidates[altloc] = (occupancy, coordinate)

    if not builders:
        selected = requested_chain if requested_chain is not None else "any"
        raise PDBValidationError(
            f"First model contains no supported protein residues for chain {selected!r}: "
            f"{pdb_path}."
        )
    chains = tuple(dict.fromkeys(builder.author_id.chain_id for builder in builders.values()))
    if len(chains) != 1:
        displayed = ", ".join(chain if chain else "<blank>" for chain in chains)
        raise PDBValidationError(
            f"First model contains multiple protein chains ({displayed}): {pdb_path}."
        )
    selected_chain = chains[0]
    segment_ids = tuple(dict.fromkeys(builder.segment_id for builder in builders.values()))
    if len(segment_ids) != 1:
        raise PDBValidationError(
            "Selected chain contains multiple TER-delimited protein segments: "
            f"{pdb_path}."
        )

    residues: list[PDBResidue] = []
    for builder in builders.values():
        if "CA" not in builder.atom_candidates:
            raise PDBValidationError(
                f"Protein residue {builder.author_id.label} lacks a C-alpha atom in "
                f"{pdb_path}."
            )
        atoms = {
            atom_name: _select_atom(atom_name, candidates)
            for atom_name, candidates in builder.atom_candidates.items()
        }
        residues.append(
            PDBResidue(
                author_id=builder.author_id,
                residue_name=builder.residue_name,
                canonical_residue_name=builder.canonical_residue_name,
                atoms=atoms,
            )
        )
    structure = ParsedStructure(
        model_id=model_id,
        chain_id=selected_chain,
        residues=tuple(residues),
        source_path=pdb_path,
    )
    if expected_sequence is not None:
        expected = normalize_sequence(expected_sequence)
        if structure.sequence != expected:
            mismatches = [
                index
                for index, (observed, wanted) in enumerate(
                    zip(structure.sequence, expected), start=1
                )
                if observed != wanted
            ]
            length_detail = (
                ""
                if len(structure.sequence) == len(expected)
                else f" (lengths {len(structure.sequence)} and {len(expected)})"
            )
            raise PDBValidationError(
                f"Coordinate sequence disagrees with expected sequence{length_detail}; "
                f"first mismatch positions: {mismatches[:10]}."
            )
    return structure


def write_sequential_ca_pdb(
    structure: ParsedStructure,
    path: str | os.PathLike[str],
    *,
    chain_id: str = "A",
) -> Path:
    """Write selected C-alpha atoms as a contiguous, canonical one-chain PDB.

    Residue and atom serials are both one-based and sequential.  MSE is emitted
    as its canonical parent MET.  These are the exact inputs used for both sides
    of the pipeline's conventional, alignment-optimized US-align invocation.
    """

    output_chain = str(chain_id).strip()
    if len(output_chain) != 1:
        raise ValueError("Temporary PDB chain_id must be one non-whitespace character.")
    if len(structure.residues) > 9_999:
        raise ValueError("Sequential PDB output supports at most 9,999 residues.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for position, residue in enumerate(structure.residues, start=1):
        coordinate = residue.coordinate("CA")
        x, y, z = (float(value) for value in coordinate)
        residue_name = ONE_TO_THREE[residue.one_letter]
        line = (
            f"ATOM  {position:5d} {'CA':^4} {residue_name:>3s} "
            f"{output_chain}{position:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          C  "
        )
        if len(line) < 78:
            raise AssertionError("Internal error while formatting a temporary PDB record.")
        lines.append(line)
    lines.extend(("TER", "END"))
    destination.write_text("\n".join(lines) + "\n", encoding="ascii")
    return destination


def coordinates_for_atoms(
    residues: Iterable[PDBResidue], atom_names: Iterable[str]
) -> np.ndarray:
    """Return residue-major coordinates for an explicit atom order."""

    normalized_atoms = tuple(str(atom).strip().upper() for atom in atom_names)
    if not normalized_atoms or any(not atom for atom in normalized_atoms):
        raise ValueError("At least one non-empty atom name is required.")
    coordinates = np.asarray(
        [
            residue.coordinate(atom_name)
            for residue in residues
            for atom_name in normalized_atoms
        ],
        dtype=np.float64,
    )
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise PDBValidationError(
            f"Selected atom coordinates have invalid shape {coordinates.shape}."
        )
    coordinates.flags.writeable = False
    return coordinates
