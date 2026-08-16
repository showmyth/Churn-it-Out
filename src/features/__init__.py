"""
Features module: data loading, validation, and transformation pipeline.

Exports:
    - pipeline_from_file: Load CSV → validate → process → return (X, y)
    - build_features: Process raw dataframe → return (X, y)
    - RAW_SCHEMA: Raw data schema contract
    - MODEL_INPUT_SCHEMA: Processed data schema contract
"""

from .pipeline import build_features, pipeline_from_file
from .schema import RAW_SCHEMA, MODEL_INPUT_SCHEMA

__all__ = [
    "build_features",
    "pipeline_from_file",
    "RAW_SCHEMA",
    "MODEL_INPUT_SCHEMA",
]
