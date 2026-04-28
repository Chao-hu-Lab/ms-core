"""Degeneracy annotation policy for Step 3 duplicate removal."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DegeneracyAnnotator:
    """Annotate adduct-like degeneracy relationships on a deduplicated matrix."""

    def annotate(
        self,
        df: pd.DataFrame,
        *,
        col_info: dict[str, object],
        sample_type_row: pd.Series,
        ppm_tolerance: float,
        rt_tolerance: float,
        correlation_threshold: float,
        min_correlation_points: int,
        adduct_table_file: str | None,
    ) -> tuple[pd.DataFrame, dict[str, object], str]:
        """Annotate adduct-like degeneracy relationships on the deduplicated matrix."""
        annotated = df.copy()
        annotation_cols = {
            "Degeneracy_Type": "None",
            "Degeneracy_Description": "",
            "Degeneracy_Base_mz": "",
            "Degeneracy_PPM_Error": "",
            "Degeneracy_Group_Role": "singleton",
            "Degeneracy_Group_ID": "",
            "Degeneracy_Pearson_R": "",
        }
        for col, default in annotation_cols.items():
            annotated[col] = default

        if len(annotated) == 0:
            return annotated, self._empty_stats(), "empty"

        adduct_table, source = self._load_adduct_table(adduct_table_file)
        if adduct_table.empty:
            return annotated, self._empty_stats(), source

        valid = annotated[
            annotated["_mz"].notna() & annotated["_rt"].notna() & (annotated["_mz"] > 0)
        ].copy()
        if valid.empty:
            return annotated, self._empty_stats(), source

        correlation_cols = self._select_correlation_columns(sample_type_row, col_info)
        valid = valid.sort_values(["_rt", "_mz", "_total_intensity"], ascending=[True, True, False])
        pair_matches: dict[int, list[dict[str, Any]]] = {}
        base_matches: dict[int, list[dict[str, Any]]] = {}
        corr_rejected = 0

        rows = list(valid.iterrows())
        for i, current in enumerate(rows):
            j = i + 1
            while j < len(rows):
                other = rows[j]
                current_idx, current_row = current
                other_idx, other_row = other
                rt_diff = float(other_row["_rt"] - current_row["_rt"])
                if rt_diff > rt_tolerance:
                    break

                base, pair = (
                    (current, other) if current_row["_mz"] <= other_row["_mz"] else (other, current)
                )
                base_idx, base_row = base
                pair_idx, pair_row = pair
                mz_diff = float(pair_row["_mz"] - base_row["_mz"])
                match = self._find_best_adduct_match(
                    mz_diff,
                    float(max(base_row["_mz"], pair_row["_mz"])),
                    adduct_table,
                    ppm_tolerance,
                )
                if match is not None:
                    corr_value = self._compute_feature_correlation(
                        annotated,
                        int(base_idx),
                        int(pair_idx),
                        correlation_cols,
                        min_correlation_points,
                    )
                    if corr_value is None or corr_value < correlation_threshold:
                        corr_rejected += 1
                        j += 1
                        continue
                    payload = {
                        "base_idx": int(base_idx),
                        "base_mz": float(base_row["_mz"]),
                        "adduct_type": match["To"],
                        "ppm_error": float(match["ppm_error"]),
                        "corr_value": float(corr_value),
                    }
                    pair_matches.setdefault(int(pair_idx), []).append(payload)
                    base_matches.setdefault(int(base_idx), []).append(payload)
                j += 1

        self._write_annotations(annotated, pair_matches)

        stats = {
            "degeneracy_annotation_enabled": True,
            "degeneracy_matches": int(sum(len(v) for v in pair_matches.values())),
            "degeneracy_groups": int(len(base_matches)),
            "degeneracy_base_count": int(len(base_matches)),
            "degeneracy_adduct_count": int(len(pair_matches)),
            "degeneracy_corr_rejected": int(corr_rejected),
        }
        return annotated, stats, source

    @staticmethod
    def _empty_stats() -> dict[str, object]:
        return {
            "degeneracy_annotation_enabled": True,
            "degeneracy_matches": 0,
            "degeneracy_groups": 0,
            "degeneracy_base_count": 0,
            "degeneracy_adduct_count": 0,
            "degeneracy_corr_rejected": 0,
        }

    @staticmethod
    def _select_correlation_columns(
        sample_type_row: pd.Series,
        col_info: dict[str, object],
    ) -> list[str]:
        """Choose columns used for Pearson correlation in degeneracy annotation."""
        intensity_cols = [
            str(col) for col in col_info.get("intensity_cols", []) if col in sample_type_row.index
        ]
        if not intensity_cols:
            return intensity_cols

        preferred_cols: list[str] = []
        fallback_cols: list[str] = []
        for col in intensity_cols:
            sample_type = str(sample_type_row.get(col, "")).strip().lower()
            if sample_type in {"", "nan", "na", "none"}:
                continue
            fallback_cols.append(col)
            if sample_type not in {"qc", "blank", "standard", "sdolek"}:
                preferred_cols.append(col)

        if len(preferred_cols) >= 3:
            return preferred_cols
        if len(fallback_cols) >= 2:
            return fallback_cols
        return preferred_cols or fallback_cols

    @staticmethod
    def _compute_feature_correlation(
        df: pd.DataFrame,
        base_idx: int,
        pair_idx: int,
        correlation_cols: list[str],
        min_correlation_points: int,
    ) -> float | None:
        """Compute Pearson correlation on shared positive intensities."""
        if len(correlation_cols) < 2:
            return None

        base_series = pd.to_numeric(df.loc[base_idx, correlation_cols], errors="coerce")
        pair_series = pd.to_numeric(df.loc[pair_idx, correlation_cols], errors="coerce")
        valid_mask = (
            base_series.notna() & pair_series.notna() & (base_series > 0) & (pair_series > 0)
        )
        shared = int(valid_mask.sum())
        if shared < max(2, min_correlation_points):
            return None

        base_vals = np.log1p(base_series[valid_mask].astype(float).to_numpy())
        pair_vals = np.log1p(pair_series[valid_mask].astype(float).to_numpy())
        if np.std(base_vals) == 0 or np.std(pair_vals) == 0:
            return None

        corr = np.corrcoef(base_vals, pair_vals)[0, 1]
        if np.isnan(corr):
            return None
        return float(corr)

    @staticmethod
    def _find_best_adduct_match(
        mz_diff: float,
        reference_mz: float,
        adduct_table: pd.DataFrame,
        ppm_tolerance: float,
    ) -> dict[str, Any] | None:
        """Return the closest adduct-table match within ppm tolerance."""
        if reference_mz <= 0:
            return None

        best_match: dict[str, Any] | None = None
        tolerance_da = ppm_tolerance * reference_mz / 1_000_000

        for row in adduct_table.itertuples(index=False):
            delta = float(row.Delta_Da)
            abs_error = abs(mz_diff - delta)
            if abs_error > tolerance_da:
                continue
            ppm_error = abs_error / reference_mz * 1_000_000
            candidate = {
                "To": row.To,
                "Delta_Da": delta,
                "ppm_error": ppm_error,
            }
            if best_match is None or candidate["ppm_error"] < best_match["ppm_error"]:
                best_match = candidate

        return best_match

    def _load_adduct_table(self, custom_file: str | None) -> tuple[pd.DataFrame, str]:
        """Load a custom adduct table or fall back to built-in defaults."""
        if custom_file:
            path = Path(custom_file)
            if path.exists():
                try:
                    if path.suffix.lower() in {".csv"}:
                        custom_df = pd.read_csv(path)
                    elif path.suffix.lower() in {".tsv", ".txt"}:
                        custom_df = pd.read_csv(path, sep="\t")
                    else:
                        custom_df = pd.read_excel(path)
                    required_cols = {"To", "Delta_Da"}
                    ordered_cols = ["To", "Delta_Da"]
                    if required_cols.issubset(custom_df.columns) and not custom_df.empty:
                        return custom_df[ordered_cols].copy(), str(path)
                except (OSError, ValueError, ImportError):
                    pass
        return self._create_default_adduct_table(), "built-in"

    @staticmethod
    def _create_default_adduct_table() -> pd.DataFrame:
        """Provide a compact default adduct table for v1 degeneracy annotation."""
        return pd.DataFrame(
            [
                {"To": "[M+Na]+", "Delta_Da": 21.981943},
                {"To": "[M+K]+", "Delta_Da": 37.955882},
                {"To": "[M+NH4]+", "Delta_Da": 17.026549},
                {"To": "[M+ACN+H]+", "Delta_Da": 41.026549},
                {"To": "[M+H]+ isotope", "Delta_Da": 1.003355},
                {"To": "[M+H]+ isotope +2", "Delta_Da": 2.006710},
            ]
        )

    @staticmethod
    def _write_annotations(
        annotated: pd.DataFrame,
        pair_matches: dict[int, list[dict[str, Any]]],
    ) -> None:
        group_counter = 1
        assigned_bases: set[int] = set()
        for pair_idx in sorted(pair_matches):
            matches = sorted(
                pair_matches[pair_idx], key=lambda item: (item["ppm_error"], item["base_mz"])
            )
            adduct_types = "; ".join(match["adduct_type"] for match in matches)
            base_mz_values = "; ".join(f"{match['base_mz']:.4f}" for match in matches)
            ppm_values = "; ".join(f"{match['ppm_error']:.2f}" for match in matches)
            corr_values = "; ".join(f"{match['corr_value']:.3f}" for match in matches)
            descriptions = "; ".join(
                (
                    f"{match['adduct_type']} of base m/z {match['base_mz']:.4f} "
                    f"(r={match['corr_value']:.3f})"
                )
                for match in matches
            )
            group_id = f"DG{group_counter:04d}"
            annotated.at[pair_idx, "Degeneracy_Type"] = adduct_types
            annotated.at[pair_idx, "Degeneracy_Description"] = descriptions
            annotated.at[pair_idx, "Degeneracy_Base_mz"] = base_mz_values
            annotated.at[pair_idx, "Degeneracy_PPM_Error"] = ppm_values
            annotated.at[pair_idx, "Degeneracy_Group_Role"] = "adduct"
            annotated.at[pair_idx, "Degeneracy_Group_ID"] = group_id
            annotated.at[pair_idx, "Degeneracy_Pearson_R"] = corr_values

            for match in matches:
                base_idx = match["base_idx"]
                if base_idx in assigned_bases:
                    continue
                annotated.at[base_idx, "Degeneracy_Type"] = "[M+H]+"
                annotated.at[base_idx, "Degeneracy_Description"] = (
                    f"Base peak for degeneracy group {group_id} (best r={match['corr_value']:.3f})"
                )
                annotated.at[base_idx, "Degeneracy_Base_mz"] = f"{match['base_mz']:.4f}"
                annotated.at[base_idx, "Degeneracy_PPM_Error"] = ""
                annotated.at[base_idx, "Degeneracy_Group_Role"] = "base"
                annotated.at[base_idx, "Degeneracy_Group_ID"] = group_id
                annotated.at[base_idx, "Degeneracy_Pearson_R"] = f"{match['corr_value']:.3f}"
                assigned_bases.add(base_idx)
            group_counter += 1
