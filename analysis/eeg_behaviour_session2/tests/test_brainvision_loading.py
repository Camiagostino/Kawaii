from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.loading import (  # noqa: E402
    BrainVisionLoadingError,
    BrainVisionValidationError,
    discover_recording_parts,
)


CHANNELS = ["Cz", "Fz", "LO1", "ECG1"]


def base_config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "raw_eeg_dir": "raw_eeg",
            "brainvision_working_dir": "derivatives/brainvision_working_copies",
            "derivatives_dir": "derivatives",
            "output_dir": "out",
            "qc_dir": "out/qc",
            "session2_logs_dir": "logs",
            "session2_schedules_dir": "schedules",
        },
        "recordings": {
            "filename_template": "kawaii{subject_number:04d}",
            "expected_sampling_rate_hz": 500.0,
            "sampling_rate_tolerance_hz": 0.001,
            "participant_recordings": {},
        },
        "channels": {
            "eeg": {
                "expected_count": len(CHANNELS),
                "expected_names": ["Cz", "Fz"],
                "eog": ["LO1"],
                "unused": ["ECG1"],
                "misc": [],
            }
        },
    }


def write_triplet(root: Path, file_base: str, *, datafile: str | None = None, markerfile: str | None = None, sampling_interval: float = 2000.0, channels: list[str] | None = None) -> None:
    raw = root / "raw_eeg"
    raw.mkdir(parents=True, exist_ok=True)
    datafile = datafile or f"{file_base}.eeg"
    markerfile = markerfile or f"{file_base}.vmrk"
    channels = channels or CHANNELS
    channel_lines = "\n".join(f"Ch{i}={name},,0.1,uV" for i, name in enumerate(channels, start=1))
    (raw / f"{file_base}.vhdr").write_text(
        "\n".join(
            [
                "Brain Vision Data Exchange Header File Version 1.0",
                "[Common Infos]",
                f"DataFile={datafile}",
                f"MarkerFile={markerfile}",
                "DataFormat=BINARY",
                f"NumberOfChannels={len(channels)}",
                f"SamplingInterval={sampling_interval}",
                "[Channel Infos]",
                channel_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    (raw / f"{file_base}.vmrk").write_text(
        "\n".join(
            [
                "Brain Vision Data Exchange Marker File, Version 1.0",
                "[Common Infos]",
                f"DataFile={datafile}",
                "[Marker Infos]",
                "Mk1=New Segment,,1,1,0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (raw / f"{file_base}.eeg").write_bytes(b"tiny test eeg")


class TestBrainVisionLoading(unittest.TestCase):
    def test_missing_triplet_files_raise_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = base_config(root)
            (root / "raw_eeg").mkdir()
            (root / "raw_eeg" / "kawaii0005.vhdr").write_text("", encoding="utf-8")

            with self.assertRaises(BrainVisionLoadingError):
                discover_recording_parts("sub005", cfg, root=root)

    def test_filename_mismatch_creates_working_copies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = base_config(root)
            write_triplet(root, "kawaii0005", datafile="wrong.eeg", markerfile="wrong.vmrk")
            source_vhdr = root / "raw_eeg" / "kawaii0005.vhdr"
            source_before = source_vhdr.read_text(encoding="utf-8")

            parts = discover_recording_parts("sub005", cfg, root=root)

            self.assertEqual(len(parts), 1)
            self.assertTrue(parts[0].filename_validation.working_copy_created)
            self.assertIn("derivatives", str(parts[0].vhdr_path))
            self.assertTrue(parts[0].vhdr_path.exists())
            self.assertTrue(parts[0].vmrk_path.exists())
            self.assertTrue(parts[0].eeg_path.exists())
            self.assertIn("DataFile=kawaii0005.eeg", parts[0].vhdr_path.read_text(encoding="utf-8"))
            self.assertIn("MarkerFile=kawaii0005.vmrk", parts[0].vhdr_path.read_text(encoding="utf-8"))
            self.assertEqual(source_vhdr.read_text(encoding="utf-8"), source_before)

    def test_multiple_recording_parts_are_discovered_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = base_config(root)
            write_triplet(root, "kawaii0004_2")
            write_triplet(root, "kawaii0004")

            parts = discover_recording_parts("pilot04", cfg, root=root)

            self.assertEqual([part.recording_part for part in parts], ["part1", "part2"])
            self.assertEqual([part.file_base for part in parts], ["kawaii0004", "kawaii0004_2"])

    def test_invalid_sampling_rate_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = base_config(root)
            write_triplet(root, "kawaii0005", sampling_interval=1000.0)

            with self.assertRaises(BrainVisionValidationError):
                discover_recording_parts("sub005", cfg, root=root)

    def test_invalid_channel_list_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = base_config(root)
            write_triplet(root, "kawaii0005", channels=["Cz", "Fz", "LO1", "Unexpected"])

            with self.assertRaises(BrainVisionValidationError):
                discover_recording_parts("sub005", cfg, root=root)

    def test_channel_aliases_are_accepted_during_header_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = base_config(root)
            cfg["channels"]["channel_name_aliases"] = {"FPz": "Fpz", "FP1": "Fp1", "FP2": "Fp2"}
            cfg["channels"]["eeg"]["expected_count"] = 4
            cfg["channels"]["eeg"]["expected_names"] = ["Fpz", "Fp1", "Fp2", "Cz"]
            cfg["channels"]["eeg"]["eog"] = []
            cfg["channels"]["eeg"]["unused"] = []
            write_triplet(root, "kawaii0005", channels=["FPz", "FP1", "FP2", "Cz"])

            parts = discover_recording_parts("sub005", cfg, root=root)

            self.assertEqual(parts[0].channel_names, ("FPz", "FP1", "FP2", "Cz"))


if __name__ == "__main__":
    unittest.main()
