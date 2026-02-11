"""
Tests for the stage system base classes.
"""

import pytest
import xarray as xr
import numpy as np

from seasenselib.pipeline import Stage, StageContext, Pipeline


class SimpleStage(Stage):
    """Simple test stage that adds an attribute."""
    
    def __init__(self, suffix: str = "_processed"):
        self.suffix = suffix
    
    def name(self) -> str:
        return "simple"
    
    def process(self, context: StageContext) -> StageContext:
        # Add metadata
        context.metadata['simple_stage_applied'] = True
        
        # Add attribute to all variables
        for var_name in context.dataset.data_vars:
            context.dataset[var_name].attrs['processed'] = f"yes{self.suffix}"
        
        return context


class ConditionalStage(Stage):
    """Stage that only runs if a certain variable exists."""
    
    def name(self) -> str:
        return "conditional"
    
    def can_process(self, context: StageContext) -> bool:
        return 'temperature' in context.dataset.data_vars
    
    def process(self, context: StageContext) -> StageContext:
        context.metadata['conditional_stage_applied'] = True
        return context


def test_stage_context_creation():
    """Test StageContext creation."""
    ds = xr.Dataset({'temp': (['time'], [10, 11, 12])})
    context = StageContext(ds, {'source': 'test.cnv'})
    
    assert context.dataset is ds
    assert context.metadata['source'] == 'test.cnv'


def test_stage_context_copy():
    """Test StageContext copy."""
    ds = xr.Dataset({'temp': (['time'], [10, 11, 12])})
    context = StageContext(ds, {'source': 'test.cnv'})
    
    copied = context.copy()
    
    assert copied.dataset is context.dataset  # Dataset not copied
    assert copied.metadata == context.metadata  # Metadata copied
    assert copied.metadata is not context.metadata  # But different dict


def test_simple_stage():
    """Test a simple stage."""
    ds = xr.Dataset({
        'temp': (['time'], [10, 11, 12]),
        'sal': (['time'], [35, 35.1, 35.2]),
    })
    context = StageContext(ds, {})
    
    stage = SimpleStage()
    result = stage.process(context)
    
    assert result.metadata['simple_stage_applied'] is True
    assert result.dataset['temp'].attrs['processed'] == 'yes_processed'
    assert result.dataset['sal'].attrs['processed'] == 'yes_processed'


def test_conditional_stage_runs():
    """Test conditional stage when condition is met."""
    ds = xr.Dataset({'temperature': (['time'], [10, 11, 12])})
    context = StageContext(ds, {})
    
    stage = ConditionalStage()
    
    assert stage.can_process(context) is True
    result = stage.process(context)
    assert result.metadata['conditional_stage_applied'] is True


def test_conditional_stage_skips():
    """Test conditional stage when condition is not met."""
    ds = xr.Dataset({'salinity': (['time'], [35, 35.1, 35.2])})
    context = StageContext(ds, {})
    
    stage = ConditionalStage()
    
    assert stage.can_process(context) is False


def test_pipeline_basic():
    """Test basic pipeline execution."""
    ds = xr.Dataset({
        'temp': (['time'], [10, 11, 12]),
        'sal': (['time'], [35, 35.1, 35.2]),
    })
    
    pipeline = Pipeline([SimpleStage()])
    result = pipeline.execute(ds)
    
    assert 'processed' in result['temp'].attrs
    assert result['temp'].attrs['processed'] == 'yes_processed'


def test_pipeline_ordering():
    """Test pipeline executes stages in given order."""
    class FirstStage(Stage):
        def name(self) -> str:
            return "first"
        def process(self, context: StageContext) -> StageContext:
            context.dataset.attrs.setdefault('order', []).append('first')
            return context
    
    class SecondStage(Stage):
        def name(self) -> str:
            return "second"
        def process(self, context: StageContext) -> StageContext:
            context.dataset.attrs.setdefault('order', []).append('second')
            return context
    
    ds = xr.Dataset({'temp': (['time'], [10, 11, 12])})
    
    # Add in explicit order
    pipeline = Pipeline([FirstStage(), SecondStage()])
    
    # Execute with metadata tracking
    result = pipeline.execute(ds, metadata={})
    
    assert result.attrs.get('order') == ['first', 'second']


def test_pipeline_skips_conditional():
    """Test pipeline skips stages that can't process."""
    ds = xr.Dataset({'salinity': (['time'], [35, 35.1, 35.2])})
    
    pipeline = Pipeline([SimpleStage(), ConditionalStage()])
    result = pipeline.execute(ds, metadata={})
    
    # Simple stage should run
    assert 'processed' in result['salinity'].attrs
    
    # Conditional stage should not run (no temperature variable)
    # We can't directly check metadata from execute(), but conditional layer won't error


def test_pipeline_get_stage_order():
    """Test getting stage execution order."""
    pipeline = Pipeline([SimpleStage(), ConditionalStage()])
    
    order = pipeline.get_stage_order()
    
    assert order == ['simple', 'conditional']


def test_pipeline_describe():
    """Test pipeline description."""
    pipeline = Pipeline([SimpleStage(), ConditionalStage()])
    
    desc = pipeline.describe()
    
    assert 'Pipeline' in desc
    assert 'simple' in desc
    assert 'conditional' in desc


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
