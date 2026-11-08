"""Deterministic PyMOL rendering for the structural panels in Figure 2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image


COLOURING_COLUMNS = {
    "feature_id",
    "sequence_position",
    "author_chain",
    "author_residue_number",
    "author_insertion_code",
    "pymol_color_value_0_1",
}
FEATURE_WIDTH = 3600
FEATURE_HEIGHT = 3600
FOCUS_WIDTH = 3600
FOCUS_HEIGHT = 3000
OUTPUT_DPI = 600
CROP_PADDING = 84


@dataclass(frozen=True)
class PyMOLStructureRenders:
    feature_images: tuple[Path, ...]
    focal_structure_image: Path
    manifest: Path


def _quoted_path(path: Path) -> str:
    return '"' + str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _png_commands(path: Path, *, width: int, height: int) -> list[str]:
    """Use PyMOL's Python API so absolute paths are not reparsed as PML tokens."""

    return [
        "python",
        "from pymol import cmd",
        (
            f"cmd.png({json.dumps(str(path.resolve()))}, width={width}, height={height}, "
            f"dpi={OUTPUT_DPI}, ray=1, quiet=1)"
        ),
        "python end",
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _insertion_code(value: object) -> str:
    insertion = str(value)
    return "" if insertion.lower() == "nan" else insertion


def _author_selection(row: object, *, object_name: str = "wt") -> str:
    residue = f"{int(row.author_residue_number)}{_insertion_code(row.author_insertion_code)}"
    return f"({object_name} and chain {row.author_chain} and resi {residue})"


def _publication_settings() -> list[str]:
    return [
        "bg_color white",
        "set ray_opaque_background, 0",
        "set opaque_background, 0",
        "set orthoscopic, 1",
        "set depth_cue, 0",
        "set ray_trace_fog, 0",
        "set fog, 0",
        "set ray_shadows, 1",
        "set antialias, 2",
        "set ambient, 0.42",
        "set direct, 0.68",
        "set specular, 0.14",
        "set shininess, 16",
        "set two_sided_lighting, 1",
        "set cartoon_fancy_helices, 1",
        "set cartoon_flat_sheets, 1",
        "set cartoon_smooth_loops, 1",
        "set cartoon_sampling, 14",
        "set cartoon_loop_radius, 0.24",
        "set cartoon_tube_radius, 0.34",
        "set stick_quality, 18",
        "set sphere_quality, 3",
        "hide labels, all",
    ]


def build_pymol_script(
    pdb_path: Path,
    feature_rows: pd.DataFrame,
    output_path: Path,
    *,
    width: int = FEATURE_WIDTH,
    height: int = FEATURE_HEIGHT,
) -> str:
    """Build a publication-quality feature-coloured WT cartoon script."""

    missing = COLOURING_COLUMNS - set(feature_rows.columns)
    if missing:
        raise ValueError(f"WT colouring table lacks columns: {sorted(missing)}")
    feature_ids = feature_rows["feature_id"].astype(int).unique()
    if len(feature_ids) != 1 or width <= 0 or height <= 0:
        raise ValueError(
            "A ribbon script requires one feature and positive image dimensions"
        )
    commands = [
        "reinitialize",
        f"load {_quoted_path(pdb_path)}, wt",
        "remove wt and not polymer.protein",
        "remove wt and alt B",
        "alter wt and alt A, alt=''",
        "sort wt",
        "hide everything, all",
        "show cartoon, wt",
        "color gray80, wt",
        *_publication_settings(),
    ]
    # Common zero-to-one scale: #F5EDED -> #A85C5C.
    neutral = np.asarray([0.96, 0.93, 0.93], dtype=float)
    active = np.asarray([0.66, 0.36, 0.36], dtype=float)
    ordered = feature_rows.sort_values(
        ["author_chain", "author_residue_number", "author_insertion_code"],
        kind="stable",
    )
    for ordinal, row in enumerate(ordered.itertuples(index=False)):
        value = float(np.clip(float(row.pymol_color_value_0_1), 0.0, 1.0))
        colour = neutral + value * (active - neutral)
        name = f"feature_colour_{ordinal}"
        commands.extend(
            [
                f"set_color {name}, [{colour[0]:.6f}, {colour[1]:.6f}, {colour[2]:.6f}]",
                f"color {name}, {_author_selection(row)}",
            ]
        )
    commands.extend(
        [
            "orient wt",
            "zoom wt, 5.0",
            "turn x, 75",
            "turn y, -12",
            *_png_commands(output_path, width=width, height=height),
            "quit",
        ]
    )
    return "\n".join(commands) + "\n"


def build_focal_structure_script(
    pdb_path: Path,
    structure_rows: pd.DataFrame,
    output_path: Path,
    *,
    focal_position: int,
    window: tuple[int, int] = (14, 26),
    adjacent_strand: tuple[int, int] = (22, 25),
    packing_positions: tuple[int, ...] = (32, 35, 66),
    width: int = FOCUS_WIDTH,
    height: int = FOCUS_HEIGHT,
) -> str:
    """Build the high-resolution focal-residue/adjacent-strand PyMOL view."""

    missing = COLOURING_COLUMNS - set(structure_rows.columns)
    if missing:
        raise ValueError(f"WT colouring table lacks columns: {sorted(missing)}")
    if width <= 0 or height <= 0:
        raise ValueError("Focal structure image dimensions must be positive")
    positions = (
        structure_rows.sort_values("sequence_position", kind="stable")
        .drop_duplicates("sequence_position")
        .set_index("sequence_position")
    )
    requested = {
        *range(window[0], window[1] + 1),
        *range(adjacent_strand[0], adjacent_strand[1] + 1),
        focal_position - 1,
        focal_position,
        focal_position + 1,
        *packing_positions,
    }
    missing_positions = sorted(requested - set(positions.index.astype(int)))
    if missing_positions:
        raise ValueError(f"WT structure mapping lacks positions: {missing_positions}")

    def selection(selected: list[int] | tuple[int, ...] | range) -> str:
        return " or ".join(
            _author_selection(positions.loc[int(position)]) for position in selected
        )

    focal_tripeptide = selection(
        [focal_position - 1, focal_position, focal_position + 1]
    )
    focal = selection([focal_position])
    local_window = selection(range(window[0], window[1] + 1))
    strand = selection(range(adjacent_strand[0], adjacent_strand[1] + 1))
    packing = selection(packing_positions)
    commands = [
        "reinitialize",
        f"load {_quoted_path(pdb_path)}, wt",
        "remove wt and not polymer.protein",
        "remove wt and alt B",
        "alter wt and alt A, alt=''",
        "sort wt",
        *_publication_settings(),
        "set_color protein_grey, [0.5725, 0.6000, 0.6314]",
        "set_color dark_rose, [0.6588, 0.3608, 0.3608]",
        "set_color focal_blue, [0.1569, 0.4902, 0.6863]",
        "set_color packing_neutral, [0.8200, 0.8100, 0.7900]",
        f"select focal_tripeptide, {focal_tripeptide}",
        f"select focal_residue, {focal}",
        f"select local_window, {local_window}",
        f"select adjacent_strand, {strand}",
        f"select packing_sidechains, ({packing}) and not name N+C+O",
        "hide everything, all",
        "show cartoon, wt",
        "color protein_grey, wt",
        "color dark_rose, adjacent_strand",
        "set cartoon_transparency, 0.52, wt",
        "set cartoon_transparency, 0.00, adjacent_strand",
        "show sticks, focal_tripeptide",
        "show spheres, focal_residue",
        "set stick_radius, 0.09, focal_tripeptide",
        "set stick_radius, 0.14, focal_residue",
        "set sphere_scale, 0.16, focal_residue",
        "color protein_grey, focal_tripeptide",
        "color focal_blue, focal_residue",
        "show sticks, adjacent_strand",
        "set stick_radius, 0.18, adjacent_strand",
        "color dark_rose, adjacent_strand",
        "show sticks, packing_sidechains",
        "set stick_radius, 0.15, packing_sidechains",
        "color packing_neutral, packing_sidechains",
        "orient wt",
        "turn x, -8",
        "turn y, -24",
        "turn z, 5",
        "select closeup_extent, focal_residue or adjacent_strand or packing_sidechains",
        "zoom closeup_extent, 3.0",
        *_png_commands(output_path, width=width, height=height),
        "quit",
    ]
    return "\n".join(commands) + "\n"


def _run_pymol(script: str, executable: str, expected_output: Path, label: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".pml",
        prefix="protein_sae_figure2_",
        delete=False,
    ) as handle:
        script_path = Path(handle.name)
        handle.write(script)
    try:
        completed = subprocess.run(
            [executable, "-cq", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        script_path.unlink(missing_ok=True)
    if (
        completed.returncode
        or not expected_output.is_file()
        or not expected_output.stat().st_size
    ):
        detail = (completed.stderr or completed.stdout)[-2000:]
        raise RuntimeError(f"PyMOL rendering failed for {label}: {detail}")


def _crop_transparent(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        rgba = image.convert("RGBA")
        bounds = rgba.getchannel("A").getbbox()
        if bounds is None:
            raise RuntimeError(f"PyMOL produced a transparent image: {source}")
        left, top, right, bottom = bounds
        padded = (
            max(0, left - CROP_PADDING),
            max(0, top - CROP_PADDING),
            min(rgba.width, right + CROP_PADDING),
            min(rgba.height, bottom + CROP_PADDING),
        )
        rgba.crop(padded).save(destination, dpi=(OUTPUT_DPI, OUTPUT_DPI))


def render_wt_structure_panels(
    pdb_path: Path,
    colouring: Path | pd.DataFrame,
    output_dir: Path,
    *,
    focal_position: int,
    pymol_executable: str = "pymol",
    window: tuple[int, int] = (14, 26),
    adjacent_strand: tuple[int, int] = (22, 25),
    packing_positions: tuple[int, ...] = (32, 35, 66),
) -> PyMOLStructureRenders:
    """Ray-trace all Figure 2 structural panels directly from the WT PDB."""

    executable = shutil.which(pymol_executable)
    if executable is None:
        raise FileNotFoundError(f"PyMOL executable was not found: {pymol_executable}")
    pdb_path = pdb_path.resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(f"WT PDB was not found: {pdb_path}")
    table = pd.read_csv(colouring) if isinstance(colouring, Path) else colouring.copy()
    missing = COLOURING_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"WT colouring table lacks columns: {sorted(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_outputs: list[Path] = []
    for feature_id, rows in table.groupby("feature_id", sort=True, observed=True):
        output = output_dir / f"feature_{int(feature_id)}.png"
        raw = output_dir / f"feature_{int(feature_id)}.raw.png"
        _run_pymol(
            build_pymol_script(pdb_path, rows, raw),
            executable,
            raw,
            f"feature {int(feature_id)}",
        )
        _crop_transparent(raw, output)
        raw.unlink(missing_ok=True)
        feature_outputs.append(output)
    if not feature_outputs:
        raise ValueError("WT colouring table contains no feature rows")

    focal_output = output_dir / "focal_structure.png"
    _run_pymol(
        build_focal_structure_script(
            pdb_path,
            table,
            focal_output,
            focal_position=focal_position,
            window=window,
            adjacent_strand=adjacent_strand,
            packing_positions=packing_positions,
        ),
        executable,
        focal_output,
        f"focal position {focal_position}",
    )
    manifest = output_dir / "pymol_render_manifest.json"
    payload = {
        "schema_version": 1,
        "pdb_sha256": _sha256(pdb_path),
        "pymol_executable": Path(executable).name,
        "focal_position": int(focal_position),
        "feature_images": [
            {
                "feature_id": int(path.stem.split("_")[-1]),
                "file": path.name,
                "sha256": _sha256(path),
            }
            for path in feature_outputs
        ],
        "focal_structure_image": {
            "file": focal_output.name,
            "sha256": _sha256(focal_output),
        },
        "render": {
            "projection": "orthoscopic",
            "background": "transparent",
            "dpi": OUTPUT_DPI,
            "ray_traced": True,
        },
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return PyMOLStructureRenders(tuple(feature_outputs), focal_output, manifest)
