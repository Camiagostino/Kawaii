from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd


def synthetic_session2_config(base: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(base)
    config["recordings"]["participant_recordings"] = {}
    return config


def write_complete_brainvision_trial(
    root: Path,
    config: dict[str, Any],
    *,
    file_base: str = "kawaii0001",
) -> None:
    raw = root / "workspace" / "kawaii" / "session2" / "source" / "eeg"
    raw.mkdir(parents=True, exist_ok=True)
    channel_config = config["channels"]["eeg"]
    channels = [
        *channel_config["expected_names"],
        *channel_config.get("eog", []),
        *channel_config.get("unused", []),
        *channel_config.get("misc", []),
    ]
    channel_lines = "\n".join(
        f"Ch{index}={name},,0.1,uV" for index, name in enumerate(channels, start=1)
    )
    (raw / f"{file_base}.vhdr").write_text(
        "\n".join(
            [
                "Brain Vision Data Exchange Header File Version 1.0",
                "[Common Infos]",
                f"DataFile={file_base}.eeg",
                f"MarkerFile={file_base}.vmrk",
                "DataFormat=BINARY",
                f"NumberOfChannels={len(channels)}",
                "SamplingInterval=2000",
                "[Channel Infos]",
                channel_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    marker_codes = [120, 150, 100, 11, 200, 21]
    marker_lines = [
        f"Mk{index}=Stimulus,S {code}, {100 * index},1,0"
        for index, code in enumerate(marker_codes, start=1)
    ]
    (raw / f"{file_base}.vmrk").write_text(
        "\n".join(
            [
                "Brain Vision Data Exchange Marker File, Version 1.0",
                "[Common Infos]",
                f"DataFile={file_base}.eeg",
                "[Marker Infos]",
                *marker_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    n_samples = 2000
    (raw / f"{file_base}.eeg").write_bytes(b"\0" * (n_samples * len(channels) * 2))


def write_trial_log(root: Path, *, trial: int = 1, participant: str = "sub001") -> Path:
    path = (
        root
        / "workspace"
        / "kawaii"
        / "session2"
        / "source"
        / "behavioral_logs"
        / f"{participant}_session2"
        / f"{participant}_session2_trial_log_analysis.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "subject_id": participant,
                "session": 2,
                "block": 1,
                "trial": trial,
                "image_filename": "synthetic.png",
                "old_new_status": "old",
                "recognition_response": "yes",
                "recognition_correct": True,
                "kawaii_rating": 1,
                "arousal_rating": 1,
            }
        ]
    ).to_csv(path, index=False)
    return path
