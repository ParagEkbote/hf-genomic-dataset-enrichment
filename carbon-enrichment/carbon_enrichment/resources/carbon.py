"""
Carbon-3B model/tokenizer resource for the Carbon enrichment pipeline.

This module provides the single, project-level source of the Carbon-3B
model and its hybrid 6-mer tokenizer, shared by every asset that needs
either one -- `tokenize_and_tag` (tokenizer only) and the GPU enrichment
assets (`assets/gpu/*`, model + tokenizer). Centralizing this here means
the tokenizer/model are resolved exactly once per process, at one pinned
revision, rather than each asset independently calling
`AutoTokenizer.from_pretrained(...)` and risking version skew between
them -- required for the determinism/provenance contract (design doc #12).

Responsibilities
----------------
- resolve `HuggingFaceBio/Carbon-3B` (tokenizer + model) at a pinned
  revision;
- load the tokenizer with `trust_remote_code=True` (required by the
  model card for the custom 6-mer tokenizer; the model itself does not
  need it);
- load the model in bfloat16 -- the model's native training precision,
  not a throughput tradeoff (design doc #17);
- wire FlashAttention-2 via the HF Kernels Hub, not pip `flash-attn`
  (design doc #18);
- put the model in `.eval()` mode, ready for `torch.inference_mode()`
  call sites (design doc #4);
- expose `model_checkpoint`, `tokenizer_revision`, and the FA2 kernel
  revision as plain attributes for the provenance manifest (#12);
- lazily load both tokenizer and model -- resources that only need
  config values (e.g. `model_checkpoint` for a manifest) never trigger
  a 3B-parameter weight load.

Instrumentation
---------------
This resource records model-lifecycle telemetry only:
- tokenizer load time;
- model load time;
- model device and dtype;
- parameter count;
- requested/resolved attention implementation;
- resolved FlashAttention-2 kernel revision.

Inference telemetry such as throughput, forward-pass count, OOM retries,
and peak CUDA memory belongs to `assets/gpu/embeddings.py`.

What this module deliberately does NOT own
-------------------------------------------
- `torch.compile` application with bucket-stable static shapes (#19) --
  that depends on the M3.5 bucket/batch lookup table, which is asset-level
  config, not resource-level. `compile_for_buckets()` below is a method
  the GPU asset calls explicitly at startup with that table; the resource
  does not compile eagerly on load.
- YaRN long-context config (#16) -- the corpus fits native 32,768-token
  context entirely; YaRN is a defensive/exception path only, wired at
  the call site that needs it, not baked into every model load here.

    resources/carbon.py
            │
            ├──────────────┐
            ▼              ▼
    tokenize_and_tag   assets/gpu/*
    (tokenizer only)   (tokenizer + model)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Final

import dagster as dg


logger = logging.getLogger(__name__)


# ============================================================================
# Project defaults
# ============================================================================

DEFAULT_MODEL_REPO = "HuggingFaceBio/Carbon-3B"

# Pin an explicit revision once this is decided for the run rather than
# floating on "main" -- shard-level reproducibility (#12) depends on this
# being a fixed commit/tag, not a moving branch head. "main" is the
# placeholder until a specific commit SHA is chosen for the production run.
DEFAULT_MODEL_REVISION = "main"

# HF Kernels Hub FlashAttention-2, pinned trusted namespace (#18) --
# avoids `allow_all_kernels=True` and the CUDA/torch wheel-matching pain
# of building flash-attn from source.
DEFAULT_ATTN_IMPLEMENTATION = "kernels-community/flash-attn2"

MAX_NATIVE_CONTEXT_TOKENS: Final[int] = 32_768


# ============================================================================
# Resource
# ============================================================================


class CarbonModelResource(dg.ConfigurableResource):
    """
    Lazily-loaded Carbon-3B model + tokenizer, shared across assets.

    Config fields are plain strings/bools rather than richly-typed model
    objects because Dagster resources are pydantic models under the hood
    (ConfigurableResource) -- the actual `transformers` objects are built
    lazily in `tokenizer`/`model` and cached on the private state below,
    not stored as pydantic fields.
    """

    model_repo: str = DEFAULT_MODEL_REPO
    model_revision: str = DEFAULT_MODEL_REVISION
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION
    device: str = "cuda"

    # ------------------------------------------------------------------------
    # Private, per-process cache. Populated on first access. Dagster's
    # multiprocess executor already isolates each asset's STEP_WORKER
    # subprocess.
    # ------------------------------------------------------------------------

    _tokenizer_cache: Any = None
    _model_cache: Any = None
    _resolved_tokenizer_revision: str | None = None
    _resolved_kernel_revision: str | None = None

    # ------------------------------------------------------------------------
    # Provenance-facing identifiers (#12)
    # ------------------------------------------------------------------------

    @property
    def model_checkpoint(self) -> str:
        """`repo@revision` string for the provenance manifest."""
        return f"{self.model_repo}@{self.model_revision}"

    @property
    def tokenizer_revision(self) -> str:
        """
        Resolved tokenizer revision, for the provenance manifest.

        Same repo/revision as the model for Carbon-3B (tokenizer and
        model are versioned together in one repo), but exposed
        separately since #12 tracks it as its own manifest field and a
        future model revision could in principle diverge from the
        tokenizer's.
        """
        return self._resolved_tokenizer_revision or self.model_revision

    @property
    def kernel_revision(self) -> str | None:
        """
        Resolved commit/revision of the FA2 kernel build actually used.

        None until the model has been loaded at least once in this
        process (the kernel revision is only known once the Kernels Hub
        has resolved and downloaded the pinned kernel). Design doc #18
        requires this be recorded per shard, not just the kernel repo
        name -- kernel builds change over time.
        """
        return self._resolved_kernel_revision

    # ------------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------------

    @property
    def tokenizer(self) -> Any:
        """
        The Carbon hybrid 6-mer tokenizer, loaded once per process.

        `trust_remote_code=True` is required here -- the model card is
        explicit that the custom DNA tokenizer needs it, while the model
        weights themselves (stock LlamaForCausalLM) do not.
        """

        if self._tokenizer_cache is None:
            from transformers import AutoTokenizer

            load_start = time.perf_counter()

            self._tokenizer_cache = AutoTokenizer.from_pretrained(
                self.model_repo,
                revision=self.model_revision,
                trust_remote_code=True,
            )

            load_seconds = time.perf_counter() - load_start

            self._resolved_tokenizer_revision = self.model_revision

            logger.info(
                "Carbon tokenizer loaded: "
                f"repo={self.model_repo!r}, "
                f"revision={self.model_revision!r}, "
                f"elapsed={load_seconds:.2f}s"
            )

        return self._tokenizer_cache

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    @property
    def model(self) -> Any:
        """
        Carbon-3B, loaded once per process: bf16, FA2 via Kernels Hub,
        eval mode.

        Callers wrap forward passes in `torch.inference_mode()` themselves
        (#4) -- this resource does not force that context manager globally,
        since some call sites (e.g. a future fine-tuning path) may
        legitimately need grads.
        """

        if self._model_cache is None:
            import torch
            from transformers import AutoModelForCausalLM

            load_start = time.perf_counter()

            model = AutoModelForCausalLM.from_pretrained(
                self.model_repo,
                revision=self.model_revision,
                dtype=torch.bfloat16,
                attn_implementation=self.attn_implementation,
            )

            model = model.to(self.device).eval()

            self._model_cache = model

            self._resolved_kernel_revision = _resolve_kernel_revision(
                self.attn_implementation
            )

            load_seconds = time.perf_counter() - load_start

            parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
            )

            parameter_count_billions = parameter_count / 1e9

            parameter_devices = {
                str(parameter.device)
                for parameter in model.parameters()
            }

            parameter_dtypes = {
                str(parameter.dtype)
                for parameter in model.parameters()
            }

            resolved_attention = getattr(
                model.config,
                "_attn_implementation",
                None,
            )

            logger.info(
                "Carbon model loaded: "
                f"repo={self.model_repo!r}, "
                f"revision={self.model_revision!r}, "
                f"elapsed={load_seconds:.2f}s, "
                f"parameters={parameter_count_billions:.3f}B, "
                f"device={sorted(parameter_devices)}, "
                f"dtype={sorted(parameter_dtypes)}, "
                f"attention_requested={self.attn_implementation!r}, "
                f"attention_resolved={resolved_attention!r}, "
                f"kernel_revision={self.kernel_revision!r}"
            )

        return self._model_cache

    # ------------------------------------------------------------------------
    # torch.compile — explicit, bucket-driven, not eager on load (#19)
    # ------------------------------------------------------------------------

    def compile_for_buckets(
        self,
        bucket_batch_config: "BucketBatchConfig",
    ) -> Any:
        """
        Apply `torch.compile` with bucket-stable static shapes.

        Called explicitly by the GPU asset at startup, after the M3.5
        bucket/batch lookup table (design doc #15) has been resolved --
        never called implicitly from `model` above, since compilation
        without a fixed bucket/batch-size axis would defeat the point of
        `dynamic=False` and trigger per-batch recompilation.

        Returns the compiled model; the GPU asset is responsible for
        warming up every bucket with one dummy batch each before real
        throughput measurement begins (#19) -- this method only performs
        the `torch.compile(...)` wrap itself.
        """

        import torch

        return torch.compile(
            self.model,
            mode="reduce-overhead",
            dynamic=False,
        )


# ============================================================================
# M3.5 config artifact shape (design doc #15)
# ============================================================================

# Not populated here -- this is the shape `compile_for_buckets` expects,
# produced by the (separate) M3.5 calibration step and consumed at GPU
# asset startup. Defined here only so `compile_for_buckets`'s signature is
# concrete rather than `Any`.


@dataclass
class BucketBatchConfig:
    """One row of the M3.5 bucket -> batch-size/dtype/backend lookup table."""

    bucket_max_tokens: int
    batch_size: int
    dtype: str
    attn_backend: str
    compile_enabled: bool
    extra: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Kernel revision resolution
# ============================================================================


def _resolve_kernel_revision(
    attn_implementation: str,
) -> str | None:
    """
    Best-effort lookup of the resolved Kernels Hub build for
    `attn_implementation`, for the provenance manifest (#18).

    Does NOT call `kernels.get_kernel(attn_implementation)` itself --
    that function requires an explicit `version` or `revision` argument
    (see kernels API reference), which this resource does not have and
    should not guess at. Instead, this reads back what was *actually*
    resolved when `transformers.AutoModelForCausalLM.from_pretrained(...,
    attn_implementation=...)` loaded the kernel internally, via
    `kernels.get_loaded_kernels()` -- a snapshot of every kernel loaded
    into the current process, each carrying a `repo_info.revision` field
    (kernels.RepoInfo). This must be called *after* `model` has
    triggered the actual load; calling it earlier will simply find
    nothing and return None.

    Falls back to None (rather than raising) if no loaded kernel matches
    `attn_implementation`, or if the installed `kernels` version doesn't
    expose this API -- absence of this field should not block a
    pipeline run, but its presence should be recorded whenever
    available.
    """

    try:
        from kernels import get_loaded_kernels

        for loaded in get_loaded_kernels():
            repo_info = loaded.repo_info

            if (
                repo_info is not None
                and repo_info.repo_id == attn_implementation
            ):
                return repo_info.revision

        return None

    except Exception:
        return None


# ============================================================================
# Resource factory (matches hf_client.py's create_* naming convention)
# ============================================================================


def create_carbon_resource(
    model_repo: str = DEFAULT_MODEL_REPO,
    model_revision: str = DEFAULT_MODEL_REVISION,
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
    device: str = "cuda",
) -> CarbonModelResource:
    """Create the project-configured CarbonModelResource.

    Parameters
    ----------
    model_repo:
        HF Hub repo id for Carbon-3B.
    model_revision:
        Pinned revision/commit/tag. Should be an explicit commit SHA for
        production runs, not a floating branch, per the provenance
        requirement in design doc #12.
    attn_implementation:
        Kernels Hub attention backend identifier (#18).
    device:
        Device the model is moved to on first access.

    Returns
    -------
    CarbonModelResource
        Configured resource used by Dagster assets. Loading is lazy --
        constructing this does not download or load any weights.
    """

    return CarbonModelResource(
        model_repo=model_repo,
        model_revision=model_revision,
        attn_implementation=attn_implementation,
        device=device,
    )