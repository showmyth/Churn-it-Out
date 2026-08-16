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


def clean_features(*args, **kwargs):
    from .pipeline import clean_features as _clean_features
    return _clean_features(*args, **kwargs)


def pipeline_from_file(*args, **kwargs):
    from .pipeline import pipeline_from_file as _pipeline_from_file
    return _pipeline_from_file(*args, **kwargs)


def __getattr__(name):
    if name in {"RAW_SCHEMA", "MODEL_INPUT_SCHEMA"}:
        from .schema import RAW_SCHEMA, MODEL_INPUT_SCHEMA
        return {"RAW_SCHEMA": RAW_SCHEMA, "MODEL_INPUT_SCHEMA": MODEL_INPUT_SCHEMA}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
