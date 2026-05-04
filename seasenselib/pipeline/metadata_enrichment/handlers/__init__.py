"""
Metadata enrichment handlers.

Includes convention implementations.
"""

from .cf_convention import CFConvention
from .acdd_convention import ACDDConvention
from .acdd_auto_metadata import AcddAutoMetadata
from .user_metadata_handler import UserMetadataHandler

__all__ = [
    "CFConvention",
    "ACDDConvention",
    "AcddAutoMetadata",
    "UserMetadataHandler",
]
