from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seasenselib.readers.rbr_matlab_legacy_reader import (
    RbrMatlabLegacyReader,
    _parse_start_end,
    _parse_time_any,
)


def test_rbr_legacy_time_parser_prefers_iso_dates():
    assert _parse_time_any("2026-04-03 03:47:46.000") == pd.Timestamp(
        "2026-04-03 03:47:46"
    )
    assert _parse_time_any("03/04/2026 03:47:46.000 AM") == pd.Timestamp(
        "2026-04-03 03:47:46"
    )
    assert _parse_start_end("2026-04-02 16:37:41") == np.datetime64(
        "2026-04-02T16:37:41"
    )
    assert _parse_start_end("02/04/2026 04:37:41 PM") == np.datetime64(
        "2026-04-02T16:37:41"
    )


def test_rbr_legacy_reader_preserves_iso_sampletimes(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    mat_file = tmp_path / "rbr_legacy_iso.mat"
    scipy_io.savemat(
        mat_file,
        {
            "RBR": {
                "name": "RBR TR-1050 6.51 15581",
                "starttime": "2026-04-02 16:37:41",
                "endtime": "2026-04-03 03:47:46",
                "sampletimes": np.array(
                    [
                        "2026-04-02 16:37:46.000",
                        "2026-04-03 03:47:46.000",
                    ],
                    dtype=object,
                ),
                "data": np.array([22.1, 10.8], dtype=float),
                "channelnames": "Temperature",
                "channelunits": "degC",
                "events": np.array([], dtype=object),
                "coefficients": np.array([], dtype=float),
            }
        },
    )

    ds = RbrMatlabLegacyReader(
        str(mat_file),
        perform_default_postprocessing=False,
    ).data

    assert ds.time.values[0] == np.datetime64("2026-04-02T16:37:46")
    assert ds.time.values[-1] == np.datetime64("2026-04-03T03:47:46")
    assert ds.attrs["rbr_start_date"] == np.datetime64("2026-04-02T16:37:41")
