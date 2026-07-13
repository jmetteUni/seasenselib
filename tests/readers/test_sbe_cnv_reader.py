from types import SimpleNamespace

import numpy as np

from seasenselib.readers.sbe_cnv_reader import SbeCnvReader


CNV_HEADER = "\n".join(
    [
        "# interval = seconds: 15",
        "# start_time = May 04 2026 08:00:01 [Instrument's time stamp, first data scan]",
    ]
)


def _calculate_time_coordinates(xarray_data, max_count=None):
    reader = object.__new__(SbeCnvReader)
    cnv = SimpleNamespace(header=CNV_HEADER, date=None)
    if max_count is None:
        max_count = len(next(iter(xarray_data.values())))
    coords = reader._SbeCnvReader__calculate_time_coordinates(  # noqa: SLF001
        xarray_data,
        cnv,
        max_count,
    )
    return reader, coords


def test_timek_is_used_for_time_coordinate_before_pipeline_mapping():
    data = {
        "temperature": np.arange(4),
        "timeK": np.array(
            [
                831196801.0,
                831196816.0,
                831196832.0,
                831196846.0,
            ]
        ),
    }

    reader, coords = _calculate_time_coordinates(data)

    expected = np.array(
        [
            "2026-05-04T08:00:01",
            "2026-05-04T08:00:16",
            "2026-05-04T08:00:32",
            "2026-05-04T08:00:46",
        ],
        dtype="datetime64[ns]",
    )
    np.testing.assert_array_equal(coords, expected)
    np.testing.assert_array_equal(
        np.diff(coords).astype("timedelta64[s]").astype(int),
        np.array([15, 16, 14]),
    )
    assert reader._time_coordinate_source_name == "timeK"
    assert reader._time_coordinate_source_type == "seconds_since_2000"


def test_interval_fallback_is_used_only_without_time_source_channel():
    data = {"temperature": np.arange(4)}

    reader, coords = _calculate_time_coordinates(data, max_count=4)

    expected = np.array(
        [
            "2026-05-04T08:00:01",
            "2026-05-04T08:00:16",
            "2026-05-04T08:00:31",
            "2026-05-04T08:00:46",
        ],
        dtype="datetime64[ns]",
    )
    np.testing.assert_array_equal(coords, expected)
    assert reader._time_coordinate_source_name == "start_time + interval"
    assert reader._time_coordinate_source_type == "start_time_plus_interval"
