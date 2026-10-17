# Original Kawaii cohort — recorded input identities and open reference selections

**Readiness record: 16 October 2026.** These hashes were exported by a **metadata-only read-only inventory** of the internal research repository at commit `78e2624a7e8559aecbc0d54330ebc70bf8ef9475`. They identify files present at inventory time; they are **not** approval to redistribute those files, a verified complete pipeline lineage, or evidence of an end-to-end rerun. See [Reproduce the existing study](REPRODUCE_EXISTING_STUDY.md).

## Source and review file identities

| Authority | Inventory observation | SHA-256 |
|---|---|---|
| Core run configuration | 8 included, 5 excluded; group includes 8 and excludes 3 | `bf82f92815614fb1957d7fb2edfd784486e356285fdec0c7b7278f2fda6bf3d3` |
| Source-exposure manifest | 8 participant entries; configured artifact-ID field matches manifest | `adffbc46c879b2907951b31686da6d2bb7e007abc4b71946b72a01be09950262` |
| Source-exposure disposition CSV | Present (746,164 bytes) | `9a21e5071ea345ec1c2829190f2b41cd5727f12a6fdfff6927d04149ba63b357` |
| Reviewed bad-channel decisions | YAML present | `9ed552d0a915efd5615fd428be82e366d7e678d9d008eaacd1edeff22a575d27` |
| Reviewed ICA-component decisions | YAML present | `06b80213446ece7d58419f14818a303dcb6b1cdffbfe20cf5f43c8a101e89719` |
| Original secondary-analysis configuration | 8 configured participants; 6 input entries | `ee56ff6bfdb8275f5146bb79e3083cf32fc06b65f0e413e948d359af7b5ac8f9` |

The study lead should verify the files held in approved lab storage against these recorded identities. The manifest's matching `artifact_id` field does **not** by itself demonstrate that every disposition row, participant and manual review matches. Source data and manual decisions must be obtained through authorised lab storage; they are not bundled in this GitHub repository.

## Corrected publication evidence and outstanding acceptance

The inventory found **27 `current.json` and 27 `generation.json` files** under `workspace/kawaii/session2/corrections/r11_qc_v3`. It also found older/different publication manifests under the general analysis directory. The original correction's **R11 QC V2 is quarantined and must not be used**. Directory presence and counts alone do not select or confirm the exact authoritative Gate 3–5 graph.

Before an exact reproduction is declared, fill in (in an **approved reference inventory stored under lab authority**, not by guessing):

| Required reference | Status from inventory |
|---|---|
| Exact accepted EEG Gate 3C epoch generation IDs by participant | **Awaiting authoritative selection** |
| Accepted participant recognition/cuteness (Gate 4) generation IDs | **Awaiting authoritative selection** |
| Accepted group aggregation/statistics/figures (Gate 5) generation IDs and output hashes | **Awaiting authoritative selection** |
| Per-participant source-disposition/EEG epoch/behavioural cohort correspondence | **Not verified by metadata inventory** |
| Original behavioural d′/bias, rating-memory output tables and source provenance | **Awaiting behavioural verification** |
| Approved reference tables/figures, expected values, and reviewer sign-off | **Awaiting scientific-owner confirmation** |
| Controlled fresh reproduction using an isolated output namespace | **Not performed** |

Keep scientific distinctions intact: EEG-retained epochs need not equal the trial set eligible for old/new behavioural inference. Preserve statistical-model convergence/boundary warnings in any comparison.

## Checklist for an authorised lab handover

- [ ] Lab custodian confirms access path and permissions for raw EEG, Session 1/2 behavioural sources and schedules.
- [ ] Hashes of the six authority files above are verified against the approved originals (or changes explicitly reconciled).
- [ ] Corrected Gate 3–5 generation IDs are verified and saved as an accepted reference inventory.
- [ ] Behavioural eligibility, selected tables and figures are approved for comparison.
- [ ] New output paths are reviewed to prevent any overwrite of accepted published results.
- [ ] Reproduction differences, model warnings and scientific-owner decision are recorded.
