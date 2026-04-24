import math

import pandas as pd

from ms_core.preprocessing.data_organizer import DataOrganizer


def test_false_positive_fix_treats_zero_as_missing_and_clears_zero_outputs() -> None:
    organizer = DataOrganizer()
    df = pd.DataFrame(
        [
            [245.1332, 12.21, 0, 5, "row_1", 245.133243, 12.279866, 398734.3, 14731746.6],
            [260.0000, 13.50, "", 0, "row_2", 260.000100, 13.600000, 10.0, 0.0],
        ],
        columns=[
            "Mz",
            "RT",
            "SampleA",
            "SampleB",
            "MZmine ID",
            "MZmine m/z",
            "MZmine RT (min)",
            "SampleA",
            "SampleB",
        ],
    )

    result = organizer.false_positive_fix(df)

    assert list(result.columns) == ["Mz", "RT", "SampleA", "SampleB"]
    assert result.shape == (2, 4)

    assert math.isclose(result.iloc[0, 0], 245.133243)
    assert math.isclose(result.iloc[0, 1], 12.279866)

    assert pd.isna(result.iloc[0, 2])
    assert math.isclose(result.iloc[0, 3], 14731746.6 * organizer.MZMINE_AREA_UNIT_FACTOR)

    assert pd.isna(result.iloc[1, 2])
    assert pd.isna(result.iloc[1, 3])

    sample_data = result.iloc[:, 2:]
    assert not sample_data.eq(0).any().any()
