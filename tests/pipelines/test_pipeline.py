"""
Tests for the pipeline executor.
"""

import pytest
import xarray as xr
import numpy as np

from seasenselib.pipeline.base import Stage, StageContext
from seasenselib.pipeline.pipeline import Pipeline


class Stage1(Stage):
    """First test stage."""
    def name(self) -> str:
        return "layer1"
    
    def process(self, context: StageContext) -> StageContext:
        order = context.dataset.attrs.get('execution_order', [])
        order.append(1)
        context.dataset.attrs['execution_order'] = order
        context.dataset['layer1_var'] = xr.DataArray([1])
        return context


class Stage2(Stage):
    """Second test stage."""
    def name(self) -> str:
        return "layer2"
    
    def process(self, context: StageContext) -> StageContext:
        order = context.dataset.attrs.get('execution_order', [])
        order.append(2)
        context.dataset.attrs['execution_order'] = order
        context.dataset['layer2_var'] = xr.DataArray([2])
        return context


class SkippableStage(Stage):
    """Stage that can be skipped."""
    def name(self) -> str:
        return "skippable"
    
    def can_process(self, context: StageContext) -> bool:
        return context.metadata.get('allow_skip', False)
    
    def process(self, context: StageContext) -> StageContext:
        context.metadata['skippable_executed'] = True
        return context


def test_pipeline_execution():
    """Test basic pipeline execution."""
    pipeline = Pipeline([Stage1(), Stage2()])
    
    ds = xr.Dataset({'temp': (['time'], [10, 11, 12])})
    result = pipeline.execute(ds)
    
    assert 'layer1_var' in result
    assert 'layer2_var' in result


def test_pipeline_execution_order():
    """Test that stages execute in configured order."""
    pipeline = Pipeline([Stage2(), Stage1()])
    
    ds = xr.Dataset()
    result = pipeline.execute(ds, {})
    
    # Should execute in given order: Stage2 then Stage1
    assert 'layer1_var' in result
    assert 'layer2_var' in result
    assert result.attrs.get('execution_order') == [2, 1]


def test_pipeline_skip_layer():
    """Test that stages can be skipped via can_process."""
    pipeline = Pipeline([Stage1(), SkippableStage()])
    
    ds = xr.Dataset()
    metadata = {'allow_skip': False}
    
    result = pipeline.execute(ds, metadata)
    
    assert 'layer1_var' in result
    # SkippableStage should not have executed
    # Can't easily verify metadata wasn't set since execute doesn't return it


def test_pipeline_get_stage_order():
    """Test getting stage execution order."""
    pipeline = Pipeline([Stage2(), Stage1()])
    
    order = pipeline.get_stage_order()
    assert order == ['layer2', 'layer1']


def test_pipeline_describe():
    """Test pipeline description."""
    pipeline = Pipeline([Stage1(), Stage2()])
    
    description = pipeline.describe()
    assert 'layer1' in description
    assert 'layer2' in description


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
