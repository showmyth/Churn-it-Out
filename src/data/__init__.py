"""
Data loading, validation, and processing layer.

Public API:
- load_data: Load raw CSV to pandas DataFrame
- process_data: Clean, encode, and transform data
- validate_data: Validate data against Great Expectations checks
"""

from .load_data import load_data
from .process_data import process_data
from .validate_data import validate_data

__all__ = [
    "load_data",
    "process_data",
    "validate_data",
]
