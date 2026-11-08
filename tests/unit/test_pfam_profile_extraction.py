from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import yaml


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "extract_pfam_profiles.py"
SPEC = importlib.util.spec_from_file_location("extract_pfam_profiles", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_extracts_byte_exact_versioned_profile(tmp_path: Path) -> None:
    first = b"HMMER3/f [3.4]\nNAME  first\nACC   PF00001.7\n//\n"
    wanted = b"HMMER3/f [3.4]\nNAME  wanted\nACC   PF12345.9\nDATA  1  2  3\n//\n"
    library = tmp_path / "Pfam-A.hmm"
    library.write_bytes(first + wanted)
    expected_sha256 = hashlib.sha256(wanted).hexdigest()
    manifest_path = tmp_path / "profiles.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profiles": {
                    "reference_x": {
                        "accession": "PF12345.9",
                        "target_pfam": "PF12345",
                        "sha256": expected_sha256,
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "profiles"

    profiles = MODULE.load_manifest(manifest_path)
    records = MODULE.extract_profiles(library, profiles, output)

    assert (output / "reference_x.hmm").read_bytes() == wanted
    assert records == [
        {
            "reference_id": "reference_x",
            "filename": "reference_x.hmm",
            "accession": "PF12345.9",
            "target_pfam": "PF12345",
            "sha256": expected_sha256,
        }
    ]
    assert MODULE.verify_outputs(profiles, output) == records
