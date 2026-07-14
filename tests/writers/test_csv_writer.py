import numpy as np
import xarray as xr

from seasenselib.writers.csv_writer import CsvWriter


def test_csv_writer_allows_duplicate_time_coordinates(tmp_path):
    ds = xr.Dataset(
        data_vars={
            "temperature": ("time", [1.0, 2.0, 3.0]),
        },
        coords={
            "time": np.array(
                [
                    "2026-05-04T08:00:01",
                    "2026-05-04T08:00:01",
                    "2026-05-04T08:00:02",
                ],
                dtype="datetime64[ns]",
            )
        },
    )
    output = tmp_path / "duplicate_time.csv"

    CsvWriter(ds).write(str(output))

    text = output.read_text(encoding="utf-8")
    assert "temperature" in text
    assert text.count("2026-05-04 08:00:01") == 2
