"""Feature pipeline exports.

The feature package intentionally avoids eager imports here because the schema and
pipeline modules import each other through the package boundary. Lazy imports keep
all modules composable without circular-import recursion.
"""

__all__ = [
    "clean_features",
    "pipeline_from_file",
    "RAW_SCHEMA",
    "MODEL_INPUT_SCHEMA",
]

from .schema import MODEL_INPUT_SCHEMA, RAW_SCHEMA


def clean_features(*args, **kwargs):
    from .pipeline import clean_features as _clean_features
    return _clean_features(*args, **kwargs)


def pipeline_from_file(*args, **kwargs):
    from .pipeline import pipeline_from_file as _pipeline_from_file
    return _pipeline_from_file(*args, **kwargs)
