from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from eeg_software.workflows.kawaii_session2.ica_identity import (
    ICAFitIdentityError,
    build_ica_fit_artifact_identity,
    build_ica_fit_context_identity,
    build_ica_fit_selection_identity,
    validate_ica_fit_artifact_identity,
    write_ica_fit_artifact_identity,
)
from eeg_software.workflows.kawaii_session2.identity import (
    SourceIdentityError,
    build_brainvision_working_lineage,
    build_source_manifest,
    identify_behavioral_inputs,
    identify_brainvision_dataset,
    identify_source_file,
    validate_brainvision_working_lineage,
)
from eeg_software.workflows.kawaii_session2.resolved_config import canonical_sha256
from eeg_software.workflows.kawaii_session2.stage_identity import (
    StageIdentityError,
    bind_epoch_metadata_rows,
    build_alignment_artifact_identity,
    build_dataset_context_identity,
    build_participant_erp_artifact_identity,
    build_timing_qc_input_identity,
    epoch_metadata_binding,
    participant_erp_identity_path,
    read_validated_participant_erp_artifact,
    validate_alignment_artifact_identity,
    validate_epoch_metadata_binding,
    validate_timing_qc_input_identity,
    write_participant_erp_artifact_identity,
)


SHA_CONFIG = "sha256:" + "1" * 64
SHA_SCIENCE = "sha256:" + "2" * 64


def _write_brainvision(directory: Path, *, stem: str, eeg: bytes) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    vhdr = directory / f"{stem}.vhdr"
    vmrk = directory / f"{stem}.vmrk"
    binary = directory / f"{stem}.eeg"
    vhdr.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\n"
        f"DataFile={binary.name}\n"
        f"MarkerFile={vmrk.name}\n",
        encoding="utf-8",
    )
    vmrk.write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n"
        "[Common Infos]\n"
        f"DataFile={binary.name}\n",
        encoding="utf-8",
    )
    binary.write_bytes(eeg)
    return vhdr, vmrk, binary


def _fake_ica():
    return SimpleNamespace(
        method="infomax",
        ch_names=["Cz", "Fz"],
        n_components_=2,
        unmixing_matrix_=np.eye(2),
        mixing_matrix_=np.eye(2),
        pca_components_=np.eye(2),
        pre_whitener_=np.ones((2, 1)),
    )


def _chain(tmp_path: Path) -> dict[str, object]:
    source_paths = _write_brainvision(tmp_path / "source", stem="recording", eeg=b"source-eeg")
    working_paths = _write_brainvision(tmp_path / "working", stem="recording", eeg=b"source-eeg")
    source = identify_brainvision_dataset(*source_paths)
    working = identify_brainvision_dataset(*working_paths)
    lineage = build_brainvision_working_lineage(
        participant="sub005",
        session="session2",
        recording_part="part1",
        source_dataset=source,
        working_dataset=working,
    )

    behavioral_path = tmp_path / "behavioral.csv"
    schedule_path = tmp_path / "schedule.csv"
    behavioral = pd.DataFrame(
        [{"subject_id": "sub005", "session": 2, "block": 1, "trial": 1, "recognition_rt_s": 0.8, "recognition_timeout": 0}]
    )
    behavioral.to_csv(behavioral_path, index=False)
    schedule_path.write_text("trial,image_filename\n1,item.png\n", encoding="utf-8")
    behavioral_inputs = identify_behavioral_inputs(
        behavioral_log_path=behavioral_path,
        schedule_path=schedule_path,
    )
    manifest = build_source_manifest(
        participant="sub005",
        session="session2",
        recording_part="part1",
        brainvision=source,
        behavioral_inputs=behavioral_inputs,
    )
    context = build_dataset_context_identity(
        participant="sub005",
        session="session2",
        recording_part="part1",
        brainvision=source,
        behavioral_inputs=behavioral_inputs,
        resolved_config_fingerprint=SHA_CONFIG,
        scientific_fingerprint=SHA_SCIENCE,
    )

    alignment = pd.DataFrame(
        [{"subject": "sub005", "recording_part": "part1", "block": 1, "trial": 1, "image_onset_sample": 100, "recognition_response_sample": 500}]
    )
    alignment_path = tmp_path / "trial_alignment.csv"
    alignment.to_csv(alignment_path, index=False)
    alignment_artifact = build_alignment_artifact_identity(
        contexts=[context], alignment=alignment, output_path=alignment_path
    )

    metadata = bind_epoch_metadata_rows(
        pd.DataFrame(
            [{"subject": "sub005", "recording_part": "part1", "trial": 1, "event_sample": 100, "event_code": 120, "condition": "old"}]
        ),
        participant="sub005",
        session="session2",
        recording_part="part1",
    )
    epoch_binding = epoch_metadata_binding(metadata)

    filtered_path = tmp_path / "filtered_raw.fif"
    filtered_path.write_bytes(b"filtered-eeg")
    filtered_identity = identify_source_file(
        filtered_path, "ica_filtered_raw_input", logical_path=filtered_path.name
    )
    selection = build_ica_fit_selection_identity(
        mode="task_crop",
        source_first_sample=0,
        source_last_sample=99,
        windows=[{"start_sample": 0, "end_sample": 99, "start_trial": 1, "end_trial": 1, "block": "1"}],
        fit_first_sample=0,
        fit_n_times=100,
        annotations_fingerprint=canonical_sha256([]),
    )
    fit_context = build_ica_fit_context_identity(
        participant="sub005",
        recording_part="part1",
        input_eeg=filtered_identity,
        fit_data_content_sha256="sha256:" + "3" * 64,
        fit_data_shape=(2, 100),
        fit_data_dtype="<f8",
        fit_sampling_frequency_hz=500.0,
        selection=selection,
        input_channel_names=("Cz", "Fz"),
        input_channel_types=("eeg", "eeg"),
        bad_channel_names=(),
        fit_channel_names=("Cz", "Fz"),
        filter_parameters={
            "operation": "mne.io.BaseRaw.filter",
            "requested_highpass_hz": 1.0,
            "requested_lowpass_hz": 30.0,
            "effective_highpass_hz": 1.0,
            "effective_lowpass_hz": 30.0,
            "picks": ["Cz", "Fz"],
        },
        estimated_effective_rank=2,
        algorithm_parameters={
            "method": "infomax",
            "n_components_strategy": "rank_aware",
            "n_components_requested": 2,
            "n_components_effective": 2,
            "decimation": 5,
            "random_seed": 7,
            "max_iter": "auto",
            "fit_params": {"extended": True},
            "reject_by_annotation": True,
            "mne_version": "test",
            "numpy_version": "test",
        },
        reference_state={"custom_ref_applied": 0, "projectors": []},
        resolved_config_fingerprint=SHA_CONFIG,
        scientific_fingerprint=SHA_SCIENCE,
    )
    ica = _fake_ica()
    solution_path = tmp_path / "sub005_part1-ica.fif"
    solution_path.write_bytes(b"ica-solution")
    ica_artifact = build_ica_fit_artifact_identity(
        fit_context=fit_context, ica=ica, solution_path=solution_path
    )
    ica_identity_path = tmp_path / "ica_fit_identity.json"
    write_ica_fit_artifact_identity(ica_artifact, ica_identity_path)

    timing_settings = {"sampling_rate_hz": 500.0, "response_types": {}}
    timing_identity = build_timing_qc_input_identity(
        context=context,
        alignment_artifact=alignment_artifact,
        behavioral_inputs=behavioral_inputs,
        behavioral=behavioral,
        alignment=alignment,
        settings=timing_settings,
    )

    branch_settings = {"branch": "recognition", "conditions": ["hit", "correct_rejection"]}
    participant_outputs = {}
    for label in (
        "condition_counts",
        "recognition_measurements",
        "recognition_measurements_wide",
        "recognition_qc_summary",
    ):
        path = tmp_path / "erp" / f"{label}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("subject,recording_part,branch\nsub005,part1,recognition\n", encoding="utf-8")
        participant_outputs[label] = path
    participant_identity = build_participant_erp_artifact_identity(
        participant="sub005",
        session="session2",
        recording_part="part1",
        branch="recognition",
        epoch_metadata_binding_fingerprint=epoch_binding.fingerprint,
        configuration_fingerprint=SHA_CONFIG,
        branch_settings=branch_settings,
        output_paths=participant_outputs,
    )
    participant_identity_file = participant_erp_identity_path(tmp_path / "erp")
    write_participant_erp_artifact_identity(participant_identity, participant_identity_file)

    return locals()


def test_coherent_cross_stage_identity_chain_validates(tmp_path: Path) -> None:
    chain = _chain(tmp_path)
    validate_brainvision_working_lineage(
        chain["lineage"],
        participant="sub005",
        session="session2",
        recording_part="part1",
        source_dataset=chain["source"],
        working_dataset=chain["working"],
    )
    assert chain["manifest"].fingerprint.startswith("sha256:")
    validate_alignment_artifact_identity(
        chain["alignment_artifact"],
        expected_contexts=[chain["context"]],
        alignment_path=chain["alignment_path"],
        alignment=chain["alignment"],
    )
    assert validate_epoch_metadata_binding(
        chain["metadata"],
        chain["metadata"],
        participant="sub005",
        session="session2",
        recording_part="part1",
    ).fingerprint == chain["epoch_binding"].fingerprint
    validate_ica_fit_artifact_identity(
        chain["ica_identity_path"],
        ica=chain["ica"],
        solution_path=chain["solution_path"],
        expected_participant="sub005",
        expected_recording_part="part1",
        expected_input_eeg=chain["filtered_identity"],
        expected_resolved_config_fingerprint=SHA_CONFIG,
        expected_scientific_fingerprint=SHA_SCIENCE,
    )
    validate_timing_qc_input_identity(
        chain["timing_identity"],
        behavioral=chain["behavioral"],
        alignment=chain["alignment"],
        subject="sub005",
        recording_part="part1",
        settings=chain["timing_settings"],
    )
    assert read_validated_participant_erp_artifact(
        chain["participant_identity_file"],
        expected_participant="sub005",
        expected_recording_part="part1",
        expected_branch="recognition",
        expected_configuration_fingerprint=SHA_CONFIG,
        expected_branch_settings=chain["branch_settings"],
        current_output_paths=chain["participant_outputs"],
    ).epoch_metadata_binding_fingerprint == chain["epoch_binding"].fingerprint


@pytest.mark.parametrize(
    "mismatch",
    [
        "changed_source_eeg",
        "changed_behavioral_log",
        "changed_schedule",
        "wrong_participant",
        "wrong_recording_part",
        "wrong_resolved_config",
        "stale_working_lineage",
        "foreign_alignment",
        "foreign_epoch_metadata",
        "foreign_ica_fit_context",
        "foreign_timing_scope",
    ],
)
def test_cross_stage_high_consequence_mismatches_fail_closed(
    tmp_path: Path, mismatch: str
) -> None:
    chain = _chain(tmp_path)
    if mismatch == "changed_source_eeg":
        chain["source_paths"][2].write_bytes(b"changed-source")
        with pytest.raises(SourceIdentityError, match="original BrainVision"):
            validate_brainvision_working_lineage(
                chain["lineage"], participant="sub005", session="session2", recording_part="part1",
                source_dataset=identify_brainvision_dataset(*chain["source_paths"]), working_dataset=chain["working"],
            )
    elif mismatch == "changed_behavioral_log":
        chain["behavioral_path"].write_text("subject_id,session,trial\nsub005,2,99\n", encoding="utf-8")
        current = identify_behavioral_inputs(behavioral_log_path=chain["behavioral_path"], schedule_path=chain["schedule_path"])
        with pytest.raises(StageIdentityError, match="behavioral source"):
            build_timing_qc_input_identity(context=chain["context"], alignment_artifact=chain["alignment_artifact"], behavioral_inputs=current, behavioral=chain["behavioral"], alignment=chain["alignment"], settings=chain["timing_settings"])
    elif mismatch == "changed_schedule":
        chain["schedule_path"].write_text("trial,image_filename\n1,other.png\n", encoding="utf-8")
        current = identify_behavioral_inputs(behavioral_log_path=chain["behavioral_path"], schedule_path=chain["schedule_path"])
        with pytest.raises(StageIdentityError, match="behavioral source"):
            build_timing_qc_input_identity(context=chain["context"], alignment_artifact=chain["alignment_artifact"], behavioral_inputs=current, behavioral=chain["behavioral"], alignment=chain["alignment"], settings=chain["timing_settings"])
    elif mismatch in {"wrong_participant", "wrong_recording_part", "foreign_timing_scope"}:
        with pytest.raises(StageIdentityError, match="participant/part"):
            validate_timing_qc_input_identity(
                chain["timing_identity"], behavioral=chain["behavioral"], alignment=chain["alignment"],
                subject="sub006" if mismatch != "wrong_recording_part" else "sub005",
                recording_part="part2" if mismatch != "wrong_participant" else "part1",
                settings=chain["timing_settings"],
            )
    elif mismatch == "wrong_resolved_config":
        foreign_context = build_dataset_context_identity(
            participant="sub005", session="session2", recording_part="part1",
            brainvision=chain["source"], behavioral_inputs=chain["behavioral_inputs"],
            resolved_config_fingerprint="sha256:" + "9" * 64, scientific_fingerprint=SHA_SCIENCE,
        )
        with pytest.raises(StageIdentityError, match="dataset context"):
            validate_alignment_artifact_identity(chain["alignment_artifact"], expected_contexts=[foreign_context])
    elif mismatch == "stale_working_lineage":
        chain["working_paths"][2].write_bytes(b"stale-working")
        with pytest.raises(SourceIdentityError, match="working lineage"):
            validate_brainvision_working_lineage(
                chain["lineage"], participant="sub005", session="session2", recording_part="part1",
                source_dataset=chain["source"], working_dataset=identify_brainvision_dataset(*chain["working_paths"]),
            )
    elif mismatch == "foreign_alignment":
        foreign_context = build_dataset_context_identity(
            participant="sub005", session="session2", recording_part="part2",
            brainvision=chain["source"], behavioral_inputs=chain["behavioral_inputs"],
            resolved_config_fingerprint=SHA_CONFIG, scientific_fingerprint=SHA_SCIENCE,
        )
        foreign = build_alignment_artifact_identity(contexts=[foreign_context], alignment=chain["alignment"], output_path=chain["alignment_path"])
        with pytest.raises(StageIdentityError, match="expected dataset context"):
            build_timing_qc_input_identity(context=chain["context"], alignment_artifact=foreign, behavioral_inputs=chain["behavioral_inputs"], behavioral=chain["behavioral"], alignment=chain["alignment"], settings=chain["timing_settings"])
    elif mismatch == "foreign_epoch_metadata":
        foreign = bind_epoch_metadata_rows(
            pd.DataFrame([{"subject": "sub006", "recording_part": "part2", "trial": 1, "event_sample": 100, "event_code": 120}]),
            participant="sub006", session="session2", recording_part="part2",
        )
        with pytest.raises(StageIdentityError, match="scope"):
            validate_epoch_metadata_binding(chain["metadata"], foreign, participant="sub005", session="session2", recording_part="part1")
    elif mismatch == "foreign_ica_fit_context":
        foreign_path = tmp_path / "foreign_filtered_raw.fif"
        foreign_path.write_bytes(b"foreign-filtered-eeg")
        foreign_input = identify_source_file(foreign_path, "ica_filtered_raw_input", logical_path="filtered_raw.fif")
        with pytest.raises(ICAFitIdentityError, match="input EEG content"):
            validate_ica_fit_artifact_identity(
                chain["ica_identity_path"], ica=chain["ica"], solution_path=chain["solution_path"],
                expected_participant="sub005", expected_recording_part="part1", expected_input_eeg=foreign_input,
            )
