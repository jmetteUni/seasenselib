"""
Stage group interfaces for the transformation pipeline.

This module defines the core interfaces for different stage groups:
- IMappingStrategy: Variable name mapping strategies
- IMetadataExtractor: Metadata extraction from various sources
- IConvention: Standards-compliant metadata enrichment
- IDerivation: Parameter derivation with dependency resolution
- ITransformation: Data/value transformations with provenance
- IValidator: Standards validation

Each interface represents a distinct concern in the data transformation pipeline.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import xarray as xr


# ============================================================================
# Variable Mapping Interface
# ============================================================================

class IMappingStrategy(ABC):
    """
    Interface for variable mapping strategies.
    
    Strategies are tried in the order they are registered.
    Different strategies handle different mapping sources:
    - User custom mappings
    - Format-specific mappings
    - Default mappings
    - Regex patterns
    """
    
    @abstractmethod
    def map(self, variable_name: str) -> Optional[str]:
        """
        Map a variable name to its canonical name.
        
        Parameters
        ----------
        variable_name : str
            The sensor-specific variable name.
        
        Returns
        -------
        Optional[str]
            Canonical name, or None if no mapping found.
        """
        pass
    
    @abstractmethod
    def description(self) -> str:
        """
        Get a human-readable description of this strategy.
        
        Returns
        -------
        str
            Description of what this strategy does.
        """
        pass


# ============================================================================
# Metadata Extraction Interface
# ============================================================================

class MetadataRegistry:
    """
    Registry for extracted metadata from various sources.
    
    Holds metadata extracted from:
    - File headers (instrument info, calibration)
    - Variable attributes
    - Global attributes
    """
    
    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        sources: Optional[Dict[str, str]] = None
    ):
        """
        Initialize metadata registry.
        
        Parameters
        ----------
        data : Dict[str, Any], optional
            Initial metadata dictionary.
        sources : Dict[str, str], optional
            Optional map of key -> source name.
        """
        self._data = data.copy() if data else {}
        self._sources = sources.copy() if sources else {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self._data.get(key, default)
    
    def add(self, key: str, value: Any, source: Optional[str] = None) -> "MetadataRegistry":
        """Add or overwrite a metadata entry (in-place)."""
        self._data[key] = value
        if source:
            self._sources[key] = source
        return self
    
    def set(self, key: str, value: Any, source: Optional[str] = None) -> "MetadataRegistry":
        """Return new registry with additional key-value pair."""
        new_data = self._data.copy()
        new_data[key] = value
        new_sources = self._sources.copy()
        if source:
            new_sources[key] = source
        return MetadataRegistry(new_data, new_sources)
    
    def merge(self, other: "MetadataRegistry") -> "MetadataRegistry":
        """Merge with another registry, returning new registry."""
        new_data = self._data.copy()
        new_data.update(other._data)
        new_sources = self._sources.copy()
        new_sources.update(other._sources)
        return MetadataRegistry(new_data, new_sources)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._data.copy()
    
    def sources(self) -> Dict[str, str]:
        """Get sources for metadata keys."""
        return self._sources.copy()


class IMetadataExtractor(ABC):
    """
    Interface for metadata extractors.
    
    Extractors pull metadata from various sources and populate
    a MetadataRegistry. Multiple extractors can be combined using
    the Composite pattern.
    """
    
    @abstractmethod
    def extract(
        self,
        dataset: xr.Dataset,
        context: Optional[Dict[str, Any]] = None
    ) -> MetadataRegistry:
        """
        Extract metadata from a dataset or context.
        
        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to extract metadata from.
        context : Dict[str, Any]
            Additional context (source file, format, etc.).
        
        Returns
        -------
        MetadataRegistry
            Registry containing extracted metadata.
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of this extractor.
        
        Returns
        -------
        str
            Extractor name.
        """
        pass


# ============================================================================
# Convention Interface
# ============================================================================

class ValidationError:
    """Represents a validation error."""
    
    def __init__(self, message: str, severity: str = "error", path: Optional[str] = None):
        """
        Initialize validation error.
        
        Parameters
        ----------
        message : str
            Error message.
        severity : str
            Severity level: "error", "warning", "info".
        path : str, optional
            Path to the problematic element (variable name, attribute path).
        """
        self.message = message
        self.severity = severity
        self.path = path
    
    def __repr__(self) -> str:
        if self.path:
            return f"{self.severity.upper()}: {self.path}: {self.message}"
        return f"{self.severity.upper()}: {self.message}"


class IConvention(ABC):
    """
    Interface for standards conventions.
    
    Conventions enrich datasets with standards-compliant metadata
    and can validate compliance. Examples: CF, ACDD, IOOS.
    """
    
    @abstractmethod
    def name(self) -> str:
        """
        Get the convention name.
        
        Returns
        -------
        str
            Convention name (e.g., "CF-1.13", "ACDD-1.3").
        """
        pass
    
    @abstractmethod
    def enrich(self, dataset: xr.Dataset, metadata_registry: MetadataRegistry) -> xr.Dataset:
        """
        Enrich dataset with convention-compliant metadata.
        
        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to enrich.
        metadata_registry : MetadataRegistry
            Available metadata from extraction phase.
        
        Returns
        -------
        xr.Dataset
            Enriched dataset.
        """
        pass
    
    @abstractmethod
    def validate(self, dataset: xr.Dataset) -> List[ValidationError]:
        """
        Validate dataset compliance with this convention.
        
        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to validate.
        
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid).
        """
        pass
    

# ============================================================================
# Derivation Interface
# ============================================================================

class IDerivation(ABC):
    """
    Interface for parameter derivations.
    
    Derivations compute new parameters from existing ones with
    automatic dependency resolution. Examples: density, potential
    temperature, chlorophyll from fluorescence.
    """
    
    @abstractmethod
    def output_parameter(self) -> str:
        """
        Get the name of the derived parameter.
        
        Returns
        -------
        str
            Parameter name (e.g., "density", "potential_temperature").
        """
        pass
    
    @abstractmethod
    def required_inputs(self) -> List[str]:
        """
        Get required input parameters.
        
        Returns
        -------
        List[str]
            List of required parameter names.
        """
        pass
    
    @abstractmethod
    def can_derive(self, dataset: xr.Dataset) -> bool:
        """
        Check if derivation is possible with available data.
        
        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to check.
        
        Returns
        -------
        bool
            True if all required inputs are present.
        """
        pass
    
    @abstractmethod
    def derive(self, dataset: xr.Dataset) -> xr.DataArray:
        """
        Derive the parameter.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing required inputs.
        
        Returns
        -------
        xr.DataArray
            The derived parameter.
        """
        pass
    
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata for the derived parameter.
        
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary with units, standard_name, etc.
        """
        pass


# ============================================================================
# Transformation Interface
# ============================================================================

@dataclass
class TransformationRecord:
    """Structured provenance for one applied data transformation."""

    transformation: str
    description: str
    variables: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a JSON-friendly dictionary."""
        data: Dict[str, Any] = {
            "transformation": self.transformation,
            "description": self.description,
        }
        if self.variables:
            data["variables"] = list(self.variables)
        if self.parameters:
            data["parameters"] = dict(self.parameters)
        return data


class ITransformation(ABC):
    """
    Interface for dataset transformations.

    Transformations change data values, coordinates, or data structure after
    units have been normalized and before validation. Implementations must
    return structured transformation records so the processing protocol and
    output metadata remain reproducible.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Get the transformation handler name.

        Returns
        -------
        str
            Stable handler name used in processing metadata.
        """
        pass

    def can_transform(
        self,
        dataset: xr.Dataset,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Return whether this transformation can run for the dataset.

        The default allows the transformation to run. Handlers should override
        this when they require specific variables, coordinate systems, or
        reader metadata.
        """
        return True

    @abstractmethod
    def transform(
        self,
        dataset: xr.Dataset,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[xr.Dataset, List[TransformationRecord | Dict[str, Any]]]:
        """
        Transform the dataset and return structured provenance records.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset to transform.
        context : Dict[str, Any], optional
            Pipeline metadata context.

        Returns
        -------
        tuple[xr.Dataset, list]
            The transformed dataset and one or more transformation records.
            Return an empty record list when no transformation was applied.
        """
        pass


# ============================================================================
# Validator Interface
# ============================================================================

class IValidator(ABC):
    """
    Interface for dataset validators.
    
    Validators check datasets for compliance with standards
    and best practices.
    """
    
    @abstractmethod
    def name(self) -> str:
        """
        Get validator name.
        
        Returns
        -------
        str
            Validator name.
        """
        pass
    
    @abstractmethod
    def validate(self, dataset: xr.Dataset) -> List[ValidationError]:
        """
        Validate the dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Dataset to validate.
        
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid).
        """
        pass
