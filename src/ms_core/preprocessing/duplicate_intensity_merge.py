"""Duplicate-row intensity merge policy for Step 3 duplicate removal."""

from typing import Any, Literal, cast

import pandas as pd

MergeMode = Literal["fill_gaps", "per_sample_max"]


class DuplicateIntensityMerger:
    """Merge donor-row intensities into selected duplicate representatives."""

    def merge(
        self,
        df: pd.DataFrame,
        merge_groups: list[list[int]],
        intensity_cols: list[str],
        merge_mode: MergeMode | str,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """
        Merge intensity data from donor rows into the representative row.

        ``fill_gaps`` keeps legacy behavior: only fill zero/NaN cells on the
        representative row. ``per_sample_max`` keeps the highest positive sample
        value across the representative and donor rows.
        """
        normalized_merge_mode = self.normalize_merge_mode(merge_mode)
        merge_stats = {
            "groups_merged": 0,
            "data_points_recovered": 0,
            "data_points_upgraded": 0,
        }

        if not merge_groups or not intensity_cols:
            return df, merge_stats

        for group in merge_groups:
            best_idx = group[0]
            donor_indices = group[1:]
            recovered_in_group = 0
            upgraded_in_group = 0

            for col in intensity_cols:
                best_raw = df.at[best_idx, col]
                best_numeric = self.coerce_positive_number(best_raw)

                if normalized_merge_mode == "fill_gaps":
                    if best_numeric is not None:
                        continue
                    for donor_idx in donor_indices:
                        donor_raw = df.at[donor_idx, col]
                        donor_numeric = self.coerce_positive_number(donor_raw)
                        if donor_numeric is None:
                            continue
                        df.at[best_idx, col] = donor_raw
                        recovered_in_group += 1
                        break
                    continue

                selected_raw = best_raw
                selected_numeric = best_numeric
                selected_from_donor = False
                for donor_idx in donor_indices:
                    donor_raw = df.at[donor_idx, col]
                    donor_numeric = self.coerce_positive_number(donor_raw)
                    if donor_numeric is None:
                        continue
                    if selected_numeric is None or donor_numeric > selected_numeric:
                        selected_raw = donor_raw
                        selected_numeric = donor_numeric
                        selected_from_donor = True

                if not selected_from_donor:
                    continue

                df.at[best_idx, col] = selected_raw
                if best_numeric is None:
                    recovered_in_group += 1
                else:
                    upgraded_in_group += 1

            if recovered_in_group > 0 or upgraded_in_group > 0:
                merge_stats["groups_merged"] += 1
                merge_stats["data_points_recovered"] += recovered_in_group
                merge_stats["data_points_upgraded"] += upgraded_in_group

        return df, merge_stats

    @staticmethod
    def normalize_merge_mode(merge_mode: MergeMode | str) -> MergeMode:
        """Validate and normalize duplicate-row merge policy."""
        normalized = str(merge_mode).strip().lower()
        valid_modes: set[str] = {"fill_gaps", "per_sample_max"}
        if normalized not in valid_modes:
            valid_list = ", ".join(sorted(valid_modes))
            raise ValueError(
                f"Unsupported merge_mode '{merge_mode}'. Expected one of: {valid_list}"
            )
        return cast(MergeMode, normalized)

    @staticmethod
    def coerce_positive_number(value: Any) -> float | None:
        """Return a positive numeric value, or None for zero/NaN/non-numeric cells."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if pd.isna(numeric_value) or numeric_value <= 0:
            return None
        return numeric_value
