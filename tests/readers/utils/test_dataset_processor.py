"""
Unit tests for DatasetProcessor utility class.

Tests static methods for Dataset transformations:
- Sorting variables alphabetically
- Renaming parameters with standard names
- Deriving oceanographic parameters (density, potential temperature)
- Assigning default global attributes
"""

import unittest
import numpy as np
import xarray as xr
from datetime import datetime
from seasenselib.readers.utils import DatasetBuilder, DatasetProcessor


class TestDatasetProcessor(unittest.TestCase):
    """Test suite for DatasetProcessor utility class."""

    def setUp(self):
        """Set up test fixtures with a sample Dataset."""
        time_array = np.array([datetime(2023, 1, 1, 12, i, 0) for i in range(5)], dtype='datetime64[ns]')
        depth_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        self.ds = DatasetBuilder.create_template(
            time_array=time_array,
            depth_array=depth_array,
            latitude=54.0,
            longitude=10.0
        )
        
        # Add some variables in non-alphabetical order
        DatasetBuilder.assign_data(self.ds, 'temperature', np.array([15.5, 15.4, 15.3, 15.2, 15.1]))
        DatasetBuilder.assign_data(self.ds, 'salinity', np.array([35.0, 35.1, 35.2, 35.3, 35.4]))
        DatasetBuilder.assign_data(self.ds, 'pressure', np.array([0.0, 1.0, 2.0, 3.0, 4.0]))

    def test_sort_variables_basic(self):
        """Test basic variable sorting."""
        # Add variables in reverse alphabetical order
        ds = xr.Dataset()
        ds['zebra'] = xr.DataArray([1, 2, 3])
        ds['yankee'] = xr.DataArray([4, 5, 6])
        ds['alpha'] = xr.DataArray([7, 8, 9])
        
        sorted_ds = DatasetProcessor.sort_variables(ds)
        
        var_names = list(sorted_ds.data_vars.keys())
        self.assertEqual(var_names, ['alpha', 'yankee', 'zebra'])

    def test_sort_variables_preserves_data(self):
        """Test that sorting preserves data values."""
        original_temp = self.ds['temperature'].values.copy()
        original_sal = self.ds['salinity'].values.copy()
        
        sorted_ds = DatasetProcessor.sort_variables(self.ds)
        
        np.testing.assert_array_equal(sorted_ds['temperature'].values, original_temp)
        np.testing.assert_array_equal(sorted_ds['salinity'].values, original_sal)

    def test_sort_variables_preserves_coordinates(self):
        """Test that sorting preserves coordinates."""
        sorted_ds = DatasetProcessor.sort_variables(self.ds)
        
        self.assertIn('time', sorted_ds.coords)
        self.assertIn('latitude', sorted_ds.coords)
        self.assertIn('longitude', sorted_ds.coords)

    def test_rename_parameters_basic(self):
        """Test basic parameter renaming."""
        # Create dataset with raw names
        ds = xr.Dataset()
        ds['temp'] = xr.DataArray([15.0], attrs={'standard_name': 'sea_water_temperature'})
        ds['sal'] = xr.DataArray([35.0], attrs={'standard_name': 'sea_water_salinity'})
        
        renamed_ds = DatasetProcessor.rename_parameters(ds)
        
        self.assertIn('temperature', renamed_ds.data_vars)
        self.assertIn('salinity', renamed_ds.data_vars)

    def test_rename_parameters_preserves_data(self):
        """Test that renaming preserves data values."""
        ds = xr.Dataset()
        test_data = np.array([15.0, 15.5, 16.0])
        ds['temp'] = xr.DataArray(test_data, attrs={'standard_name': 'sea_water_temperature'})
        
        renamed_ds = DatasetProcessor.rename_parameters(ds)
        
        np.testing.assert_array_equal(renamed_ds['temperature'].values, test_data)

    def test_rename_parameters_no_standard_name(self):
        """Test renaming with variables that have no standard_name."""
        ds = xr.Dataset()
        ds['raw_sensor_1'] = xr.DataArray([100.0])
        ds['temp'] = xr.DataArray([15.0], attrs={'standard_name': 'sea_water_temperature'})
        
        renamed_ds = DatasetProcessor.rename_parameters(ds)
        
        # Variable without standard_name should keep original name
        self.assertIn('raw_sensor_1', renamed_ds.data_vars)
        self.assertIn('temperature', renamed_ds.data_vars)

    def test_assign_default_global_attributes_basic(self):
        """Test assigning default global attributes."""
        ds_with_attrs = DatasetProcessor.assign_default_global_attributes(self.ds, "test.csv", "Test Format", "TestReader")
        
        # Check for CF convention
        self.assertIn('Conventions', ds_with_attrs.attrs)
        self.assertEqual(ds_with_attrs.attrs['Conventions'], 'CF-1.13')

    def test_assign_default_global_attributes_history(self):
        """Test that history attribute is added."""
        ds_with_attrs = DatasetProcessor.assign_default_global_attributes(self.ds, "test.csv", "Test Format", "TestReader")
        
        self.assertIn('history', ds_with_attrs.attrs)
        # History should contain timestamp
        self.assertIn('Z', ds_with_attrs.attrs['history'])

    def test_assign_default_global_attributes_source(self):
        """Test that source/creator attributes are added."""
        ds_with_attrs = DatasetProcessor.assign_default_global_attributes(self.ds, "test.csv", "Test Format", "TestReader")
        
        # Should have creator or source information
        self.assertIn('processor_name', ds_with_attrs.attrs)
        self.assertIn('seasenselib', ds_with_attrs.attrs['processor_name'])

    def test_assign_default_global_attributes_preserves_existing(self):
        """Test that existing attributes are preserved."""
        self.ds.attrs['custom_attribute'] = 'custom_value'
        self.ds.attrs['institution'] = 'Test University'
        
        ds_with_attrs = DatasetProcessor.assign_default_global_attributes(self.ds, "test.csv", "Test Format", "TestReader")
        
        # Custom attributes should be preserved
        self.assertIn('custom_attribute', ds_with_attrs.attrs)
        self.assertEqual(ds_with_attrs.attrs['custom_attribute'], 'custom_value')
        self.assertIn('institution', ds_with_attrs.attrs)
        self.assertEqual(ds_with_attrs.attrs['institution'], 'Test University')

    def test_assign_default_global_attributes_preserves_data(self):
        """Test that assigning attributes doesn't modify data."""
        original_temp = self.ds['temperature'].values.copy()
        
        ds_with_attrs = DatasetProcessor.assign_default_global_attributes(self.ds, "test.csv", "Test Format", "TestReader")
        
        np.testing.assert_array_equal(ds_with_attrs['temperature'].values, original_temp)

    def test_full_processing_workflow(self):
        """Test a complete processing workflow combining methods."""
        # Start with unsorted dataset
        ds = self.ds.copy(deep=True)
        
        # Apply transformations
        ds = DatasetProcessor.sort_variables(ds)
        ds = DatasetProcessor.assign_default_global_attributes(
            ds,
            input_file='test_file.csv',
            format_name='Test Format',
            reader_class_name='TestReader'
        )
        
        # Verify results
        var_names = list(ds.data_vars.keys())
        self.assertEqual(var_names, sorted(var_names))  # Sorted
        self.assertIn('Conventions', ds.attrs)  # Global attributes

    def test_dataset_processor_methods_are_static(self):
        """Verify all DatasetProcessor methods are static methods."""
        import inspect
        methods = [
            'sort_variables',
            'rename_parameters', 
            'derive_oceanographic_parameters',
            'assign_default_global_attributes'
        ]
        
        for method_name in methods:
            method = getattr(DatasetProcessor, method_name)
            self.assertTrue(isinstance(inspect.getattr_static(DatasetProcessor, method_name), staticmethod),
                          f"{method_name} should be a static method")


if __name__ == '__main__':
    unittest.main()


    def test_sort_variables_basic(self):
        """Test basic variable sorting."""
        # Add variables in reverse alphabetical order
        ds = xr.Dataset()
        ds['zebra'] = xr.DataArray([1, 2, 3])
        ds['yankee'] = xr.DataArray([4, 5, 6])
        ds['alpha'] = xr.DataArray([7, 8, 9])
        
        sorted_ds = DatasetProcessor.sort_variables(ds)
        
        var_names = list(sorted_ds.data_vars.keys())
        self.assertEqual(var_names, ['alpha', 'yankee', 'zebra'])

    def test_sort_variables_preserves_data(self):
        """Test that sorting preserves data values."""
        original_temp = self.ds['temperature'].values.copy()
        original_sal = self.ds['salinity'].values.copy()
        
        sorted_ds = DatasetProcessor.sort_variables(self.ds)
        
        np.testing.assert_array_equal(sorted_ds['temperature'].values, original_temp)
        np.testing.assert_array_equal(sorted_ds['salinity'].values, original_sal)

    def test_sort_variables_preserves_attributes(self):
        """Test that sorting preserves variable attributes."""
        # Add custom attribute
        self.ds['temperature'].attrs['custom_attr'] = 'test_value'
        
        sorted_ds = DatasetProcessor.sort_variables(self.ds)
        
        self.assertIn('custom_attr', sorted_ds['temperature'].attrs)
        self.assertEqual(sorted_ds['temperature'].attrs['custom_attr'], 'test_value')

    def test_sort_variables_preserves_coordinates(self):
        """Test that sorting preserves coordinates."""
        sorted_ds = DatasetProcessor.sort_variables(self.ds)
        
        self.assertIn('time', sorted_ds.coords)
        self.assertIn('latitude', sorted_ds.coords)
        self.assertIn('longitude', sorted_ds.coords)

    def test_rename_parameters_basic(self):
        """Test basic parameter renaming."""
        # Create dataset with raw names
        ds = xr.Dataset()
        ds['temp'] = xr.DataArray([15.0], attrs={'standard_name': 'sea_water_temperature'})
        ds['sal'] = xr.DataArray([35.0], attrs={'standard_name': 'sea_water_salinity'})
        
        renamed_ds = DatasetProcessor.rename_parameters(ds)
        
        self.assertIn('temperature', renamed_ds.data_vars)
        self.assertIn('salinity', renamed_ds.data_vars)

    def test_rename_parameters_with_duplicates(self):
        """Test parameter renaming with duplicate standard names (sensor numbering)."""
        ds = xr.Dataset()
        ds['temp1'] = xr.DataArray([15.0], attrs={'standard_name': 'sea_water_temperature'})
        ds['temp2'] = xr.DataArray([15.1], attrs={'standard_name': 'sea_water_temperature'})
        
        renamed_ds = DatasetProcessor.rename_parameters(ds)
        
        # Should have numbered variants
        self.assertIn('temperature', renamed_ds.data_vars)
        self.assertIn('temperature_2', renamed_ds.data_vars)

    def test_rename_parameters_preserves_data(self):
        """Test that renaming preserves data values."""
        ds = xr.Dataset()
        test_data = np.array([15.0, 15.5, 16.0])
        ds['temp'] = xr.DataArray(test_data, attrs={'standard_name': 'sea_water_temperature'})
        
        renamed_ds = DatasetProcessor.rename_parameters(ds)
        
        np.testing.assert_array_equal(renamed_ds['temperature'].values, test_data)

    def test_rename_parameters_no_standard_name(self):
        """Test renaming with variables that have no standard_name."""
        ds = xr.Dataset()
        ds['raw_sensor_1'] = xr.DataArray([100.0])
        ds['temp'] = xr.DataArray([15.0], attrs={'standard_name': 'sea_water_temperature'})
        
        renamed_ds = DatasetProcessor.rename_parameters(ds)
        
        # Variable without standard_name should keep original name
        self.assertIn('raw_sensor_1', renamed_ds.data_vars)
        self.assertIn('temperature', renamed_ds.data_vars)

    def test_assign_default_global_attributes_basic(self):
        """Test assigning default global attributes."""
        ds_with_attrs = DatasetProcessor.assign_default_global_attributes(self.ds, "test.csv", "Test Format", "TestReader")
        
        # Check for CF convention
        self.assertIn('Conventions', ds_with_attrs.attrs)
        self.assertEqual(ds_with_attrs.attrs['Conventions'], 'CF-1.13')

    def test_assign_default_global_attributes_history(self):
        """Test that history attribute is added."""
        ds_with_attrs = DatasetProcessor.assign_default_global_attributes(self.ds, "test.csv", "Test Format", "TestReader")
        
        self.assertIn('history', ds_with_attrs.attrs)
        # History should contain timestamp
        self.assertIn('Z', ds_with_attrs.attrs['history'])

    def test_assign_default_global_attributes_source(self):
        """Test that source/creator attributes are added."""
        ds_with_attrs = DatasetProcessor.assign_default_global_attributes(self.ds, "test.csv", "Test Format", "TestReader")
        
        # Should have creator or source information
        self.assertIn('processor_name', ds_with_attrs.attrs)
        self.assertIn('seasenselib', ds_with_attrs.attrs['processor_name'])

    def test_assign_default_global_attributes_preserves_existing(self):
        """Test that existing attributes are preserved."""
        self.ds.attrs['custom_attribute'] = 'custom_value'
        self.ds.attrs['institution'] = 'Test University'
        
        ds_with_attrs = DatasetProcessor.assign_default_global_attributes(self.ds, "test.csv", "Test Format", "TestReader")
        
        # Custom attributes should be preserved
        self.assertIn('custom_attribute', ds_with_attrs.attrs)
        self.assertEqual(ds_with_attrs.attrs['custom_attribute'], 'custom_value')
        self.assertIn('institution', ds_with_attrs.attrs)
        self.assertEqual(ds_with_attrs.attrs['institution'], 'Test University')

    def test_assign_default_global_attributes_preserves_data(self):
        """Test that assigning attributes doesn't modify data."""
        original_temp = self.ds['temperature'].values.copy()
        
        ds_with_attrs = DatasetProcessor.assign_default_global_attributes(self.ds, "test.csv", "Test Format", "TestReader")
        
        np.testing.assert_array_equal(ds_with_attrs['temperature'].values, original_temp)

    def test_full_processing_workflow(self):
        """Test a complete processing workflow combining all methods."""
        # Start with unsorted dataset
        ds = self.ds.copy(deep=True)
        
        # Apply all transformations
        ds = DatasetProcessor.sort_variables(ds)
        ds = DatasetProcessor.derive_oceanographic_parameters(ds)
        ds = DatasetProcessor.assign_default_global_attributes(ds)
        
        # Verify results
        var_names = list(ds.data_vars.keys())
        self.assertEqual(var_names, sorted(var_names))  # Sorted
        self.assertIn('density', ds.data_vars)  # Derived parameters
        self.assertIn('potential_temperature', ds.data_vars)
        self.assertIn('Conventions', ds.attrs)  # Global attributes

    def test_dataset_processor_methods_are_static(self):
        """Verify all DatasetProcessor methods are static methods."""
        import inspect
        methods = [
            'sort_variables',
            'rename_parameters', 
            'derive_oceanographic_parameters',
            'assign_default_global_attributes'
        ]
        
        for method_name in methods:
            method = getattr(DatasetProcessor, method_name)
            self.assertTrue(isinstance(inspect.getattr_static(DatasetProcessor, method_name), staticmethod),
                          f"{method_name} should be a static method")


if __name__ == '__main__':
    unittest.main()
