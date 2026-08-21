"""
Hugging Face resource configuration for the Carbon enrichment pipeline.

This module provides the project-specific configuration for
`dagster-hf-datasets`.

The actual Hugging Face dataset operations are delegated to
`dagster_hf_datasets.HuggingFaceResource`. This module intentionally does not
implement a second dataset client or duplicate `load_dataset()`.

Responsibilities
----------------
- configure the Hugging Face cache location;
- provide a single project-level HuggingFaceResource;
- keep Hugging Face infrastructure configuration separate from assets.

The CPU ingestion asset consumes the resource through Dagster dependency
injection:

    resources/hf_client.py
            │
            ▼
    HuggingFaceResource
            │
            ▼
    assets/cpu/ingest.py
"""

from pathlib import Path

from dagster_hf_datasets import HuggingFaceResource

# ============================================================================
# Project defaults
# ============================================================================

DEFAULT_HF_CACHE_DIR = ".hf_cache"


# ============================================================================
# Resource factory
# ============================================================================


def create_huggingface_resource(
    cache_dir: str | Path = DEFAULT_HF_CACHE_DIR,
) -> HuggingFaceResource:
    """Create the project-configured HuggingFaceResource.

    Parameters
    ----------
    cache_dir:
        Local directory used for Hugging Face dataset/model cache files.

    Returns
    -------
    HuggingFaceResource
        Configured resource used by Dagster assets.
    """

    cache_path = Path(cache_dir)

    return HuggingFaceResource(
        cache_dir=str(cache_path),
    )
