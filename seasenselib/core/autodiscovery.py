"""
Autodiscovery module for readers, writers, and plotters.

This module provides functionality to automatically discover and register
reader, writer, and plotter classes without requiring manual registry maintenance.
It also includes format detection utilities and plugin loading via entry points.
"""

import importlib
import inspect
import os
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Type, Set, Optional
from abc import ABC
from .exceptions import FormatDetectionError

# Import entry_points with fallback for older Python versions
try:
    from importlib.metadata import entry_points
except ImportError:
    from importlib_metadata import entry_points  # Python < 3.8


def _convert_class_name_to_module_name(class_name: str) -> str:
    """Convert PascalCase class name to snake_case module name."""
    # Handle special cases first
    special_cases = {
        'NetCdfWriter': 'netcdf_writer',
        'NetCdfReader': 'netcdf_reader',
    }

    if class_name in special_cases:
        return special_cases[class_name]

    # Convert PascalCase to snake_case
    import re
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    return name


def _get_expected_module_name(class_name: str, base_class_name: str) -> str:
    """Get expected module name based on class name and type."""
    if base_class_name.lower() in class_name.lower():
        # If class name contains base class name, use conversion
        return _convert_class_name_to_module_name(class_name)
    else:
        # Fallback for unusual naming patterns
        return _convert_class_name_to_module_name(class_name)


def _normalize_extensions(extensions: Any) -> tuple[str, ...]:
    """Normalize extension declarations to lowercase dotted suffixes."""
    if extensions is None:
        return ()

    if isinstance(extensions, str):
        raw_extensions = (extensions,)
    else:
        try:
            raw_extensions = tuple(extensions)
        except TypeError:
            return ()

    normalized = []
    for extension in raw_extensions:
        if extension is None:
            continue
        text = str(extension).strip().lower()
        if not text:
            continue
        if not text.startswith('.'):
            text = f'.{text}'
        if text not in normalized:
            normalized.append(text)

    return tuple(normalized)


class BaseDiscovery:
    """Base class for autodiscovery functionality with plugin support."""

    def __init__(self, package_name: str, base_class: Type[ABC], entry_point_group: Optional[str] = None):
        """
        Initialize the discovery system.
        
        Parameters:
        -----------
        package_name : str
            The full package name (e.g., 'seasenselib.readers')
        base_class : Type[ABC]
            The abstract base class that all discovered classes must inherit from
        entry_point_group : str, optional
            Entry point group name for plugin discovery (e.g., 'seasenselib.readers')
        """
        self.package_name = package_name
        self.base_class = base_class
        self.entry_point_group = entry_point_group
        self._discovered_classes: Dict[str, Type] = {}
        self._class_modules: Dict[str, str] = {}
        self._plugin_classes: Dict[str, Type] = {}

    def _discover_builtin_classes(self) -> Dict[str, Type]:
        """
        Discover built-in classes from the package.
        
        Returns:
        --------
        Dict[str, Type]
            Dictionary mapping class names to class objects
        """
        builtin_classes = {}

        try:
            # Import the package
            package = importlib.import_module(self.package_name)
            package_path = package.__path__

            # Walk through all modules in the package
            for _, modname, ispkg in pkgutil.iter_modules(package_path):
                if ispkg:
                    continue

                # Skip certain modules
                if modname in ['__init__', 'base', 'registry', 'api']:
                    continue

                full_module_name = f"{self.package_name}.{modname}"
                try:
                    # Import the module
                    module = importlib.import_module(full_module_name)

                    # Find all classes in the module that inherit from base class
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (obj != self.base_class and 
                            issubclass(obj, self.base_class) and 
                            obj.__module__ == full_module_name):

                            builtin_classes[name] = obj
                            self._class_modules[name] = modname

                except ImportError as e:
                    # Log warning but continue discovery
                    print(f"Warning: Could not import {full_module_name}: {e}")
                    continue

        except ImportError as e:
            print(f"Error: Could not import package {self.package_name}: {e}")

        return builtin_classes

    def _discover_plugin_classes(self) -> Dict[str, Type]:
        """
        Discover plugin classes from entry points.
        
        Returns:
        --------
        Dict[str, Type]
            Dictionary mapping class names to plugin class objects
        """
        if not self.entry_point_group:
            return {}

        plugin_classes = {}

        try:
            # Load entry points for this group
            eps = entry_points()
            
            # Handle different return types (dict vs SelectableGroups)
            if hasattr(eps, 'select'):
                # Python 3.10+ - returns SelectableGroups
                group_eps = eps.select(group=self.entry_point_group)
            elif isinstance(eps, dict):
                # Python 3.9 - returns dict
                group_eps = eps.get(self.entry_point_group, [])
            else:
                # Fallback - try to get group directly
                group_eps = getattr(eps, self.entry_point_group, [])

            for ep in group_eps:
                try:
                    # Load the class from the entry point
                    cls = ep.load()

                    # Validate that it's a class
                    if not isinstance(cls, type):
                        print(f"Warning: Entry point '{ep.name}' in group '{self.entry_point_group}' "
                              f"does not point to a class, skipping")
                        continue

                    # Validate that it's a subclass of the base class
                    if not issubclass(cls, self.base_class):
                        print(f"Warning: Entry point '{ep.name}' class {cls.__name__} "
                              f"is not a subclass of {self.base_class.__name__}, skipping")
                        continue

                    # Validate that it has required methods (different for plotters vs readers/writers)
                    base_class_name = self.base_class.__name__
                    if base_class_name == 'AbstractPlotter':
                        # Plotters use key() and name()
                        required_methods = ['key', 'name']
                    else:
                        # Readers and writers use format_key() and format_name()
                        required_methods = ['format_key', 'format_name']
                    
                    missing_methods = [m for m in required_methods if not hasattr(cls, m)]
                    if missing_methods:
                        print(f"Warning: Plugin class {cls.__name__} from entry point '{ep.name}' "
                              f"does not implement {', '.join(missing_methods)}(), skipping")
                        continue

                    # Add to plugin classes
                    class_name = cls.__name__
                    plugin_classes[class_name] = cls
                    self._class_modules[class_name] = f"plugin:{ep.name}"

                    # Only print in debug mode or when explicitly requested
                    # print(f"Loaded plugin: {class_name} from entry point '{ep.name}'")

                except Exception as e:
                    print(f"Warning: Could not load entry point '{ep.name}' "
                          f"in group '{self.entry_point_group}': {e}")
                    continue

        except Exception as e:
            print(f"Warning: Error during plugin discovery for group '{self.entry_point_group}': {e}")

        return plugin_classes

    def discover_classes(self) -> Dict[str, Type]:
        """
        Discover all classes (built-in + plugins) that inherit from the base class.
        
        Returns:
        --------
        Dict[str, Type]
            Dictionary mapping class names to class objects
        """
        if self._discovered_classes:
            return self._discovered_classes

        # Discover built-in classes
        builtin_classes = self._discover_builtin_classes()

        # Discover plugin classes
        plugin_classes = self._discover_plugin_classes()
        self._plugin_classes = plugin_classes

        # Merge: plugins can override built-ins (with warning)
        self._discovered_classes = dict(builtin_classes)
        for name, cls in plugin_classes.items():
            if name in self._discovered_classes:
                print(f"Warning: Plugin class '{name}' overrides built-in class")
            self._discovered_classes[name] = cls

        return self._discovered_classes

    def get_class_by_name(self, class_name: str) -> Optional[Type]:
        """Get a specific class by name."""
        classes = self.discover_classes()
        return classes.get(class_name)

    def get_all_class_names(self) -> List[str]:
        """Get list of all discovered class names."""
        classes = self.discover_classes()
        return list(classes.keys())

    def get_class_modules(self) -> Dict[str, str]:
        """Get mapping of class names to module names."""
        self.discover_classes()  # Ensure discovery is complete
        return self._class_modules.copy()

    def get_plugin_classes(self) -> Dict[str, Type]:
        """
        Get only the plugin classes (not built-in).
        
        Returns:
        --------
        Dict[str, Type]
            Dictionary of plugin class names to class objects
        """
        self.discover_classes()  # Ensure discovery is complete
        return self._plugin_classes.copy()


class ReaderDiscovery(BaseDiscovery):
    """Autodiscovery for reader classes with plugin support."""

    def __init__(self):
        from ..readers.base import AbstractReader
        super().__init__(
            package_name='seasenselib.readers',
            base_class=AbstractReader,
            entry_point_group='seasenselib.readers'
        )

    def _get_reader_file_extensions(self, reader_class: Type) -> tuple[str, ...]:
        """Return all auto-detect extensions declared by a reader class."""
        if hasattr(reader_class, 'file_extensions'):
            extensions = reader_class.file_extensions()
        elif hasattr(reader_class, 'file_extension'):
            extensions = reader_class.file_extension()
        else:
            extensions = None
        return _normalize_extensions(extensions)

    def get_reader_by_format_key(self, format_key: str) -> Optional[Type]:
        """
        Get reader class by format key.
        
        Parameters:
        -----------
        format_key : str
            The format key to search for
            
        Returns:
        --------
        Optional[Type]
            The reader class if found, None otherwise
        """
        classes = self.discover_classes()

        for _, class_obj in classes.items():
            try:
                # Check if class has format_key method and it matches
                if hasattr(class_obj, 'format_key'):
                    if class_obj.format_key() == format_key:
                        return class_obj
            except (AttributeError, TypeError):
                # Skip classes that don't implement format_key properly
                continue

        return None

    def get_readers_by_extension(self, extension: str) -> List[Type]:
        """
        Get reader classes that can handle a specific file extension.
        
        Parameters:
        -----------
        extension : str
            The file extension to search for
            
        Returns:
        --------
        List[Type]
            List of reader classes that can handle the extension
        """
        classes = self.discover_classes()
        matching_readers = []
        normalized_extension = _normalize_extensions(extension)
        if not normalized_extension:
            return matching_readers
        normalized_extension = normalized_extension[0]

        for _, class_obj in classes.items():
            try:
                if normalized_extension in self._get_reader_file_extensions(class_obj):
                    matching_readers.append(class_obj)
            except (AttributeError, TypeError, NotImplementedError):
                # Skip classes that don't implement extension methods properly
                continue

        return matching_readers

    def get_format_info(self) -> List[Dict[str, Any]]:
        """
        Get format information for all discovered readers using their static methods.
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of format information dictionaries with keys:
            'class_name', 'name', 'key', 'extension', 'extensions', 'is_plugin'
            Note: 'extension' is the primary extension and is always present
            (None if not applicable). 'extensions' contains all advertised
            auto-detect extensions.
        """
        classes = self.discover_classes()
        plugin_classes = self.get_plugin_classes()
        formats = []

        for class_name, class_obj in classes.items():
            try:
                # Use static methods to get format information
                if (hasattr(class_obj, 'format_key') and 
                    hasattr(class_obj, 'format_name')):

                    extensions = self._get_reader_file_extensions(class_obj)
                    primary_extensions = _normalize_extensions(
                        class_obj.file_extension()
                    ) if hasattr(class_obj, 'file_extension') else ()
                    primary_extension = (
                        primary_extensions[0]
                        if primary_extensions
                        else (extensions[0] if extensions else None)
                    )

                    format_info = {
                        'class_name': class_name,
                        'name': class_obj.format_name(),
                        'key': class_obj.format_key(),
                        'is_plugin': class_name in plugin_classes,
                        'extension': primary_extension,
                        'extensions': list(extensions),
                    }

                    formats.append(format_info)
            except (AttributeError, TypeError, NotImplementedError):
                # Skip classes that don't implement required methods properly
                continue

        return formats


class WriterDiscovery(BaseDiscovery):
    """Autodiscovery for writer classes with plugin support."""

    def __init__(self):
        from ..writers.base import AbstractWriter
        super().__init__(
            package_name='seasenselib.writers',
            base_class=AbstractWriter,
            entry_point_group='seasenselib.writers'
        )

    def get_writer_by_extension(self, extension: str) -> Optional[Type]:
        """
        Get writer class by file extension.
        
        Parameters:
        -----------
        extension : str
            The file extension to search for
            
        Returns:
        --------
        Optional[Type]
            The writer class if found, None otherwise
        """
        classes = self.discover_classes()

        for _, class_obj in classes.items():
            try:
                # Check if class has file_extension method and it matches
                if hasattr(class_obj, 'file_extension'):
                    if class_obj.file_extension() == extension:
                        return class_obj
            except (AttributeError, TypeError):
                # Skip classes that don't implement file_extension properly
                continue

        return None

    def get_writer_by_format_key(self, format_key: str) -> Optional[Type]:
        """
        Get writer class by format key.
        
        Parameters:
        -----------
        format_key : str
            The format key to search for (e.g., 'netcdf', 'csv', 'excel')
            
        Returns:
        --------
        Optional[Type]
            The writer class if found, None otherwise
        """
        classes = self.discover_classes()

        for _, class_obj in classes.items():
            try:
                # Check if class has format_key method and it matches
                if hasattr(class_obj, 'format_key'):
                    if class_obj.format_key() == format_key:
                        return class_obj
            except (AttributeError, TypeError):
                # Skip classes that don't implement format_key properly
                continue

        return None

    def get_format_info(self) -> List[Dict[str, str]]:
        """
        Get format information for all discovered writers using their static methods.
        
        Returns:
        --------
        List[Dict[str, str]]
            List of format information dictionaries with keys:
            'class_name', 'name', 'key', 'extension', 'is_plugin'
            Note: 'extension' is always present (None if not applicable)
        """
        classes = self.discover_classes()
        plugin_classes = self.get_plugin_classes()
        formats = []

        for class_name, class_obj in classes.items():
            try:
                # Use static methods to get format information
                if (hasattr(class_obj, 'format_key') and 
                    hasattr(class_obj, 'format_name')):

                    format_info = {
                        'class_name': class_name,
                        'name': class_obj.format_name(),
                        'key': class_obj.format_key(),
                        'is_plugin': class_name in plugin_classes,
                        'extension': None  # Default to None
                    }

                    # Get file extension if available
                    if hasattr(class_obj, 'file_extension'):
                        ext = class_obj.file_extension()
                        if ext:
                            format_info['extension'] = ext

                    formats.append(format_info)
            except (AttributeError, TypeError, NotImplementedError):
                # Skip classes that don't implement required methods properly
                continue

        return formats

    def get_supported_extensions(self) -> Set[str]:
        """
        Get all supported file extensions.
        
        Returns:
        --------
        Set[str]
            Set of supported file extensions
        """
        classes = self.discover_classes()
        extensions = set()

        for _, class_obj in classes.items():
            try:
                if hasattr(class_obj, 'file_extension'):
                    ext = class_obj.file_extension()
                    if ext:
                        extensions.add(ext)
            except (AttributeError, TypeError):
                # Skip classes that don't implement file_extension properly
                continue
 
        return extensions


class PlotterDiscovery(BaseDiscovery):
    """Autodiscovery for plotter classes with plugin support."""

    def __init__(self):
        from ..plotters.base import AbstractPlotter
        super().__init__(
            package_name='seasenselib.plotters',
            base_class=AbstractPlotter,
            entry_point_group='seasenselib.plotters'
        )

    def get_format_info(self) -> List[Dict[str, str]]:
        """
        Get plotter information for all discovered plotters.
        
        Returns:
        --------
        List[Dict[str, str]]
            List of plotter information dictionaries
        """
        classes = self.discover_classes()
        plotters = []
        plugin_classes = self.get_plugin_classes()

        for class_name, class_obj in classes.items():
            try:
                # Plotters use key() and name() static methods
                if hasattr(class_obj, 'key') and hasattr(class_obj, 'name'):
                    plotter_info = {
                        'class_name': class_name,
                        'name': class_obj.name(),
                        'key': class_obj.key(),
                        'is_plugin': class_name in plugin_classes
                    }
                else:
                    # Fall back to class name if methods not available
                    plotter_info = {
                        'class_name': class_name,
                        'name': class_name.replace('Plotter', '').replace('_', ' ').title(),
                        'key': class_name.lower().replace('plotter', ''),
                        'is_plugin': class_name in plugin_classes
                    }

                plotters.append(plotter_info)

            except (AttributeError, TypeError) as e:
                # Skip classes that don't provide the required interface
                import sys
                print(f"Warning: Skipping {class_name}: {e}", file=sys.stderr)
                continue

        return plotters

    def get_class_by_key(self, key: str) -> Optional[Type]:
        """
        Get plotter class by its key.
        
        Parameters:
        -----------
        key : str
            The plotter key (e.g., 'ts-diagram', 'histogram')
            
        Returns:
        --------
        Optional[Type]
            The plotter class if found, None otherwise
        """
        classes = self.discover_classes()
        
        for class_name, class_obj in classes.items():
            try:
                # Check if class has the key() method
                if hasattr(class_obj, 'key'):
                    if class_obj.key() == key:
                        return class_obj
            except (AttributeError, TypeError):
                # Skip classes that don't implement key() properly
                continue
        
        return None


# Public format constants for CLI and other modules
def get_input_formats():
    """Get list of all supported input format keys."""
    discovery = ReaderDiscovery()
    format_info = discovery.get_format_info()
    return [info['key'] for info in format_info]


def get_output_formats():
    """Get list of all supported output format keys."""
    discovery = WriterDiscovery()
    format_info = discovery.get_format_info()
    return [info['key'] for info in format_info]


# Deprecated: kept for backward compatibility
OUTPUT_FORMATS = None  # Will be populated dynamically when accessed


class FormatDetector:
    """File format detection using autodiscovery."""

    @staticmethod
    def detect_format(input_file: str, format_hint: Optional[str] = None) -> str:
        """
        Detect file format without importing readers.
        
        Parameters:
        -----------
        input_file : str
            Path to the input file
        format_hint : str, optional
            Explicit format hint to override detection
            
        Returns:
        --------
        str
            The detected format key
            
        Raises:
        -------
        FormatDetectionError
            If format cannot be determined
        """
        if format_hint:
            input_formats = get_input_formats()
            if format_hint in input_formats:
                return format_hint
            else:
                raise FormatDetectionError(f"Unknown format hint: {format_hint}")

        # Check if file exists
        if not os.path.exists(input_file):
            raise FormatDetectionError(f"Input file does not exist: {input_file}")

        # Get file extension
        file_path = Path(input_file)
        extension = file_path.suffix.lower()

        # Map extension to format using autodiscovery
        discovery = ReaderDiscovery()
        matching_readers = discovery.get_readers_by_extension(extension)

        if len(matching_readers) == 1:
            return matching_readers[0].format_key()

        if len(matching_readers) > 1:
            matching_keys = sorted(reader.format_key() for reader in matching_readers)
            raise FormatDetectionError(
                f"Cannot determine format for file: {input_file}. "
                f"Extension '{extension}' is ambiguous. Use file_format with one of: "
                f"{', '.join(matching_keys)}"
            )

        # If no extension match, raise an error
        raise FormatDetectionError(
            f"Cannot determine format for file: {input_file}. "
            f"Extension '{extension}' not recognized and content detection failed."
        )

    @staticmethod
    def validate_output_format(output_file: str, format_hint: Optional[str] = None) -> str:
        """
        Validate and determine output format.
        
        Parameters:
        -----------
        output_file : str
            Path to the output file
        format_hint : str, optional
            Explicit format hint
            
        Returns:
        --------
        str
            The validated output format key
        """
        if format_hint:
            # Validate format hint against available writers
            available_formats = get_output_formats()
            if format_hint in available_formats:
                return format_hint
            else:
                raise FormatDetectionError(
                    f"Unknown output format: {format_hint}. "
                    f"Available formats: {', '.join(available_formats)}"
                )

        # Detect from file extension using WriterDiscovery
        file_path = Path(output_file)
        extension = file_path.suffix.lower()

        # Find writer by extension
        discovery = WriterDiscovery()
        writer_class = discovery.get_writer_by_extension(extension)

        if writer_class and hasattr(writer_class, 'format_key'):
            return writer_class.format_key()
        else:
            # List available extensions for better error message
            available_extensions = discovery.get_supported_extensions()
            raise FormatDetectionError(
                f"No writer found for extension: {extension}. "
                f"Supported extensions: {', '.join(sorted(available_extensions))}"
            )
