"""
SeaSenseLib - Oceanographic Sensor Data Processing Library

Modern Python API for reading, processing, and visualizing oceanographic sensor data.

Basic Usage:
-----------
```python
import seasenselib as ssl
ds = ssl.read('ctd_profile.cnv')
ssl.write(ds, 'output.nc')
ssl.plot('ts-diagram', ds, dot_size=50)
```

API Structure:
-------------
- ssl.read()         : Read sensor data files from various formats
- ssl.write()        : Write datasets to various formats
- ssl.plot()         : Create plots using any registered plotter
- ssl.formats()      : List reader formats (legacy)
- ssl.list_readers() : List available readers (including plugins)
- ssl.list_writers() : List available writers (including plugins)
- ssl.list_plotters(): List available plotters (including plugins)
- ssl.list_parameters(): List canonical parameter names
- ssl.list_all()     : List all available resources
"""

# Core API imports - always available
from .api import read, write, plot, formats, list_readers, list_writers, list_plotters, list_parameters, list_all

# Lazy loading for heavy modules
from importlib import import_module
from typing import Any

# Module cache for lazy loading
_loaded_modules = {}

# Version info
__version__ = "0.3.0"

def __getattr__(name: str) -> Any:
    """
    Lazy loading of package modules.
    
    This allows for fast import times while still providing access to all functionality.
    Heavy modules (plotters, processors) are only imported when first accessed.
    """
    
    # Define module mappings for lazy loading
    _module_map = {
        # Legacy access - for backward compatibility
        'plotters': '.plotters',
        'processors': '.processors', 
        'readers': '.readers',
        'writers': '.writers'
    }
    
    if name in _module_map:
        if name not in _loaded_modules:
            module_name = f'seasenselib{_module_map[name]}'
            _loaded_modules[name] = import_module(module_name)
        return _loaded_modules[name]
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Define what's available at top level
__all__ = [
    'read',
    'write',
    'plot',
    'formats',
    'list_readers',
    'list_writers', 
    'list_plotters',
    'list_parameters',
    'list_all',
    '__version__'
]
