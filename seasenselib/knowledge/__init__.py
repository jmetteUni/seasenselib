"""
Knowledge base for SeaSenseLib domain metadata.

Provides structured, versioned data (parameters, mappings, unit conventions, etc.)
that can be loaded at runtime.
"""

from .loader import load_json

__all__ = ["load_json"]
