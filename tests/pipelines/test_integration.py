"""
Integration tests for the complete stage pipeline with real data.
"""

import pytest
import seasenselib as ssl
from pathlib import Path


# Path to example files
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / 'examples'


@pytest.mark.skipif(not (EXAMPLES_DIR / 'sea-practical-2023.cnv').exists(),
                    reason="Example file not available")
def test_read_with_default_pipeline():
    """Test reading a CNV file with default stage pipeline."""
    file_path = EXAMPLES_DIR / 'sea-practical-2023.cnv'
    
    # Read with default settings (stages enabled)
    ds = ssl.read(str(file_path))
    
    # Verify basic structure
    assert ds is not None
    assert len(ds.data_vars) > 0
    
    # Verify CF conventions were applied
    assert 'Conventions' in ds.attrs
    assert 'CF-' in ds.attrs['Conventions']
    
    # Verify history was added
    assert 'history' in ds.attrs
    assert 'SeaSenseLib' in ds.attrs['history']
    
    # Verify variables have standard names
    if 'temperature' in ds:
        assert 'standard_name' in ds['temperature'].attrs
        assert ds['temperature'].attrs['standard_name'] == 'sea_water_temperature'


@pytest.mark.skipif(not (EXAMPLES_DIR / 'sea-practical-2023.cnv').exists(),
                    reason="Example file not available")
def test_read_with_custom_stages():
    """Test reading with custom stage selection."""
    file_path = EXAMPLES_DIR / 'sea-practical-2023.cnv'
    
    # Read with only mapping and metadata enrichment
    ds = ssl.read(
        str(file_path),
        pipeline_apply_stages=['mapping', 'metadata_enrichment']
    )
    
    assert ds is not None
    assert 'Conventions' in ds.attrs


@pytest.mark.skipif(not (EXAMPLES_DIR / 'sea-practical-2023.cnv').exists(),
                    reason="Example file not available")
def test_read_with_raw_mode():
    """Test that raw mode still works."""
    file_path = EXAMPLES_DIR / 'sea-practical-2023.cnv'
    
    # Read with stages disabled (raw mode)
    ds = ssl.read(str(file_path), use_steps=False)
    
    assert ds is not None
    assert len(ds.data_vars) > 0


@pytest.mark.skipif(not (EXAMPLES_DIR / 'sea-practical-2023.cnv').exists(),
                    reason="Example file not available")
def test_raw_mode_comparison():
    """Test that raw and processed modes produce valid results."""
    file_path = EXAMPLES_DIR / 'sea-practical-2023.cnv'
    
    # Read with new system (default)
    ds_new = ssl.read(str(file_path))
    
    # Read with raw mode
    ds_raw = ssl.read(str(file_path), use_steps=False)
    
    # Processed data should include canonical variables
    core_vars = ['temperature', 'pressure', 'salinity']
    for var in core_vars:
        has_var_new = any(v.startswith(var) for v in ds_new.data_vars)
        assert has_var_new, f"New system missing {var}"

    # Raw mode should still load data, but names may be instrument-specific
    assert len(ds_raw.data_vars) > 0
    
    # Both should have coordinates
    assert 'time' in ds_new.coords or 'time' in ds_new.dims
    assert 'time' in ds_raw.coords or 'time' in ds_raw.dims


@pytest.mark.skipif(not (EXAMPLES_DIR / 'denmark-strait-ds-m1-17.cnv').exists(),
                    reason="Example file not available")
def test_mooring_data_with_stages():
    """Test mooring time series data with stage pipeline."""
    file_path = EXAMPLES_DIR / 'denmark-strait-ds-m1-17.cnv'
    
    ds = ssl.read(str(file_path))
    
    assert ds is not None
    
    # Mooring data should have time dimension
    assert 'time' in ds.dims
    
    # Should have CF conventions
    assert 'Conventions' in ds.attrs
    assert 'CF-' in ds.attrs['Conventions']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
