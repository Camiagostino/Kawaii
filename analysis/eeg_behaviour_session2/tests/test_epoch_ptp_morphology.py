from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.epoch_ptp_morphology import (  # noqa: E402
    FORBIDDEN_REVIEW_FIELDS,
    blinded_review_table,
    ptp_stratum,
    select_review_epochs,
)


class TestEpochPTPMorphology(unittest.TestCase):
    def test_stratum_boundaries_are_exact_and_nonoverlapping(self) -> None:
        self.assertEqual(ptp_stratum(600), "A_above_505_895")
        self.assertEqual(ptp_stratum(505.89503606950973), "B_330_277_to_505_895")
        self.assertEqual(ptp_stratum(330.2767354136942), "C_286_865_to_330_277")
        self.assertEqual(ptp_stratum(286.86500419323795), "D_at_or_below_286_865_reference")

    def test_selection_keeps_all_upper_epochs_and_balances_reference(self) -> None:
        rows = []
        for participant in ("sub005", "sub006"):
            for index, value in enumerate([10, 20, 30, 40, 50, 60, 290, 340, 510]):
                rows.append(
                    {
                        "participant": participant,
                        "source_epoch_index": index,
                        "ptp_max_uv": value,
                        "max_ptp_channel": "Fz",
                    }
                )
        metrics = pd.DataFrame(rows)

        selected = select_review_epochs(metrics, reference_per_participant=6)

        self.assertEqual(int((selected["ptp_max_uv"] > 286.86500419323795).sum()), 6)
        references = selected[selected["ptp_stratum"].str.startswith("D_")]
        self.assertEqual(references.groupby("participant").size().to_dict(), {"sub005": 6, "sub006": 6})

    def test_review_table_contains_no_condition_or_outcome_fields(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "participant": "sub005",
                    "source_epoch_index": 3,
                    "ptp_max_uv": 600.0,
                    "max_ptp_channel": "Fp1",
                    "condition": "forbidden",
                }
            ]
        )

        table = blinded_review_table(select_review_epochs(metrics, reference_per_participant=1))

        self.assertFalse(FORBIDDEN_REVIEW_FIELDS.intersection(table.columns))
        self.assertEqual(table.iloc[0]["morphology_assessment"], "")


if __name__ == "__main__":
    unittest.main()
