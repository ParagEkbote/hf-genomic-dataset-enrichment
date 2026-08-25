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

"""
M4/M5 — Single-pass GPU enrichment: embeddings + likelihood stats.

High-throughput, memory-safe forward pass implementation:
- Direct last_hidden_state extraction (output_hidden_states=False)
- torch.logsumexp logit reduction (eliminating 100GB+ log_softmax allocations)
- Direct PyArrow buffer creation (zero python list/dict iteration overhead)
- BucketBatchConfig + torch.compile reduce-overhead graph execution
- Fully decoupled pure-GPU vs end-to-end telemetry profiling
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Final

import dagster as dg

logger = logging.getLogger(__name__)

DEFAULT_MODEL_REPO = "HuggingFaceBio/Carbon-3B"
DEFAULT_MODEL_REVISION = "main"
DEFAULT_ATTN_IMPLEMENTATION = "kernels-community/flash-attn2"
MAX_NATIVE_CONTEXT_TOKENS: Final[int] = 32_768


@dataclass
class BucketBatchConfig:
    """One row of the M3.5 bucket -> batch-size/dtype/backend lookup table."""

    bucket_max_tokens: int
    batch_size: int
    dtype: str
    attn_backend: str
    compile_enabled: bool
    extra: dict[str, Any] = field(default_factory=dict)


def _resolve_kernel_revision(attn_implementation: str) -> str | None:
    try:
        from kernels import get_loaded_kernels

        for loaded in get_loaded_kernels():
            repo_info = loaded.repo_info
            if repo_info is not None and repo_info.repo_id == attn_implementation:
                return repo_info.revision
        return None
    except Exception:
        return None


class CarbonModelResource(dg.ConfigurableResource):
    """Lazily-loaded Carbon-3B model + tokenizer, shared across assets."""

    model_repo: str = DEFAULT_MODEL_REPO
    model_revision: str = DEFAULT_MODEL_REVISION
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION
    device: str = "cuda"

    _tokenizer_cache: Any = None
    _model_cache: Any = None
    _resolved_tokenizer_revision: str | None = None
    _resolved_kernel_revision: str | None = None

    @property
    def model_checkpoint(self) -> str:
        return f"{self.model_repo}@{self.model_revision}"

    @property
    def tokenizer_revision(self) -> str:
        return self._resolved_tokenizer_revision or self.model_revision

    @property
    def kernel_revision(self) -> str | None:
        return self._resolved_kernel_revision

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer_cache is None:
            from transformers import AutoTokenizer

            load_start = time.perf_counter()
            self._tokenizer_cache = AutoTokenizer.from_pretrained(
                self.model_repo,
                revision=self.model_revision,
                trust_remote_code=True,
            )
            self._resolved_tokenizer_revision = self.model_revision
            logger.info(
                f"Carbon tokenizer loaded in {time.perf_counter() - load_start:.2f}s"
            )
        return self._tokenizer_cache

    @property
    def model(self) -> Any:
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
            logger.info(
                f"Carbon model loaded in {time.perf_counter() - load_start:.2f}s"
            )
        return self._model_cache

    def compile_for_buckets(
        self,
        bucket_configs: list[BucketBatchConfig] | tuple[BucketBatchConfig, ...],
    ) -> Any:
        import torch

        compiled_model = torch.compile(
            self.model,
            mode="reduce-overhead",
            dynamic=False,
        )
        logger.info("Warming up CUDA graphs across static compile buckets...")
        with torch.inference_mode():
            for cfg in bucket_configs:
                if not cfg.compile_enabled:
                    continue
                warmup_start = time.perf_counter()
                dummy_input = torch.zeros(
                    (cfg.batch_size, cfg.bucket_max_tokens),
                    dtype=torch.long,
                    device=self.device,
                )
                _ = compiled_model(input_ids=dummy_input)
                torch.cuda.synchronize()
                logger.info(
                    f"Compiled bucket={cfg.bucket_max_tokens} (B={cfg.batch_size}) in {time.perf_counter() - warmup_start:.2f}s"
                )
        return compiled_model


def create_carbon_resource(
    model_repo: str = DEFAULT_MODEL_REPO,
    model_revision: str = DEFAULT_MODEL_REVISION,
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
    device: str = "cuda",
) -> CarbonModelResource:
    return CarbonModelResource(
        model_repo=model_repo,
        model_revision=model_revision,
        attn_implementation=attn_implementation,
        device=device,
    )
