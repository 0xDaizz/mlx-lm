# Copyright © 2023-2024 Apple Inc.

import copy
import gc
import glob
import importlib
import inspect
import json
import logging
import os
import resource
import sys
import shutil
import threading
from pathlib import Path
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import mlx.core as mx
import mlx.nn as nn

if os.getenv("MLXLM_USE_MODELSCOPE", "False").lower() == "true":
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise ImportError("Run `pip install modelscope` to use ModelScope.")
else:
    from huggingface_hub import snapshot_download

# For large models with lots of files
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))

from mlx.utils import tree_flatten, tree_map, tree_reduce, tree_unflatten

logger = logging.getLogger(__name__)


def eval_with_timeout(params, timeout_seconds=300.0, on_timeout=None):
    """Evaluate params with a timeout watchdog (exo-style)."""
    completed = threading.Event()

    def watchdog():
        if not completed.wait(timeout=timeout_seconds):
            logger.error(
                f"mx.eval timed out after {timeout_seconds:.0f}s. "
                "This may indicate a FAST_SYNCH or tensor parallel issue. "
                "Terminating process."
            )
            if on_timeout is not None:
                on_timeout()
            # Best-effort Metal cleanup before hard exit.
            # os._exit() skips atexit/finally, so we must clean up here.
            try:
                mx.set_wired_limit(0)
                mx.set_cache_limit(0)
                mx.clear_cache()
            except Exception:
                pass
            os._exit(1)

    t = threading.Thread(target=watchdog, daemon=True)
    t.start()
    try:
        mx.eval(params)
    finally:
        completed.set()


def set_wired_limit_for_model(model):
    """Set MLX wired limit based on model size (exo-style). macOS 15+ only."""
    try:
        if not mx.metal.is_available():
            return
        device_info = mx.device_info()
        max_rec_size = int(device_info.get("max_recommended_working_set_size", 0))
        if max_rec_size <= 0:
            return

        # Calculate model size
        model_bytes = sum(
            p.nbytes for p in tree_flatten(model.parameters())[0]
        )
        model_mb = model_bytes // (1024 * 1024)
        max_rec_mb = max_rec_size // (1024 * 1024)

        if model_bytes > 0.9 * max_rec_size:
            logger.warning(
                f"Model requires {model_mb} MB which is close to the "
                f"maximum recommended size of {max_rec_mb} MB. This can be slow."
            )

        mx.set_wired_limit(max_rec_size)
        logger.info(f"Wired limit set to {max_rec_mb} MB.")
    except (AttributeError, Exception) as e:
        logger.debug(f"Could not set wired limit: {e}")


# Local imports
from .tokenizer_utils import TokenizerWrapper
from .tokenizer_utils import load as _load_tokenizer

# Constants
MODEL_REMAPPING = {
    "mistral": "llama",
    "llava": "mistral3",
    "phi-msft": "phixtral",
    "falcon_mamba": "mamba",
    "kimi_k2": "deepseek_v3",
    "qwen2_5_vl": "qwen2_vl",
    "minimax_m2": "minimax",
    "iquestcoder": "llama",
}

MAX_FILE_SIZE_GB = 5


def _unpack_awq_weights(qweight: mx.array) -> mx.array:
    bits = 4
    pack_factor = 32 // bits
    out_features, packed_in = qweight.shape
    in_features = packed_in * pack_factor
    mask = (1 << bits) - 1  # e.g., 0xF for 4-bit
    shifts = mx.array([0, 4, 1, 5, 2, 6, 3, 7]) * bits
    unpacked = (qweight[..., None] >> shifts) & mask
    return unpacked.reshape(out_features, in_features)


def _transform_awq_weights(
    weights: Dict[str, mx.array],
    quantization_config: Dict[str, Any],
) -> Tuple[Dict[str, mx.array], Dict[str, Any]]:
    bits = quantization_config.get("bits", 4)
    if bits != 4:
        raise ValueError(f"Only {bits=} is supported for AutoAWQ/GPTQ models.")
    group_size = quantization_config.get("group_size", 128)

    new_weights = {}

    for key in list(weights.keys()):
        if key.endswith(".g_idx"):
            raise ValueError(
                f"Found {key} in weights. Models with non-contiguous group indices "
                "(g_idx) are not currently supported. Please use a model without g_idx "
                "or re-quantize the model using mlx_lm.convert."
            )

        if key.endswith(".qweight"):
            prefix = key[:-8]  # Remove ".qweight"

            qweight = weights[f"{prefix}.qweight"]
            scales_key = f"{prefix}.scales"
            qzeros_key = f"{prefix}.qzeros"

            scales = weights[scales_key]

            # AutoAWQ stores qweight as [in_features, out_features // pack_factor]
            # MLX expects [out_features, in_features // pack_factor]
            # We need to unpack, transpose, and repack

            pack_factor = 32 // bits
            in_features, packed_out = qweight.shape
            out_features = packed_out * pack_factor
            n_groups = in_features // group_size

            # Unpack qweight: [in_features, out_features // pack_factor] -> [in_features, out_features]
            unpacked_weight = _unpack_awq_weights(qweight)
            # Transpose to MLX format: [out_features, in_features]
            unpacked_weight = unpacked_weight.T

            # Repack for MLX: [out_features, in_features] -> [out_features, in_features // pack_factor]
            packed_in = in_features // pack_factor
            repacked = unpacked_weight.reshape(out_features, packed_in, pack_factor)
            shifts = mx.arange(pack_factor) * bits
            weight = (
                (repacked.astype(mx.uint32) << shifts).sum(axis=-1).astype(mx.uint32)
            )

            scales = mx.contiguous(scales.T)

            # Handle qzeros if present (asymmetric quantization)
            if qzeros_key in weights:
                qzeros = weights[qzeros_key]
                # qzeros shape: [n_groups, out_features // pack_factor]
                # Unpack to get [n_groups, out_features]
                unpacked_zeros = _unpack_awq_weights(qzeros)
                # Transpose to [out_features, n_groups]
                unpacked_zeros = unpacked_zeros.T

                # Compute biases: MLX dequant = weight * scale + bias
                # AWQ dequant = (weight - zero) * scale
                # So: bias = -zero * scale
                biases = -unpacked_zeros.astype(mx.float32) * scales
            else:
                # Symmetric quantization - zeros are implicitly 2^(bits-1)
                zero_point = 1 << (bits - 1)  # e.g., 8 for 4-bit
                biases = mx.full(scales.shape, -zero_point, dtype=mx.float32) * scales

            new_weights[f"{prefix}.weight"] = weight
            new_weights[f"{prefix}.scales"] = scales
            new_weights[f"{prefix}.biases"] = biases.astype(scales.dtype)
            model_dtype = scales.dtype

        elif not any(
            key.endswith(suffix) for suffix in [".qweight", ".qzeros", ".scales"]
        ):
            new_weights[key] = weights[key]

    for k, w in new_weights.items():
        if mx.issubdtype(w.dtype, mx.floating):
            new_weights[k] = w.astype(model_dtype)

    mlx_quantization = {
        "group_size": group_size,
        "bits": bits,
    }

    return new_weights, mlx_quantization


def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"mlx_lm.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        raise ValueError(msg)

    return arch.Model, arch.ModelArgs


def get_total_parameters(model):
    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )

    def nparams(m):
        if hasattr(m, "bits"):
            n = 0 if not hasattr(m, "bias") else m.bias.size
            return n + m.weight.size * 32 // m.bits
        return sum(v.size for _, v in tree_flatten(m.parameters()))

    return sum(nparams(m) for _, m in leaf_modules)


def compute_bits_per_weight(model):
    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    model_params = get_total_parameters(model)
    return model_bytes * 8 / model_params


def _download(
    path_or_hf_repo: str,
    revision: Optional[str] = None,
    allow_patterns: List[str] = None,
) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The local file path.
    """
    model_path = Path(path_or_hf_repo)

    if not model_path.exists():
        allow_patterns = allow_patterns or [
            "*.json",
            "model*.safetensors",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "tiktoken.model",
            "*.txt",
            "*.jsonl",
            "*.jinja",
        ]
        model_path = Path(
            snapshot_download(
                path_or_hf_repo,
                revision=revision,
                allow_patterns=allow_patterns,
            )
        )

    return model_path


def hf_repo_to_path(hf_repo):
    return Path(snapshot_download(hf_repo, local_files_only=True))


def load_config(model_path: Path) -> dict:
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)

    generation_config_file = model_path / "generation_config.json"
    if generation_config_file.exists():
        generation_config = {}
        try:
            with open(generation_config_file, "r") as f:
                generation_config = json.load(f)
        except json.JSONDecodeError:
            pass

        if eos_token_id := generation_config.get("eos_token_id", False):
            config["eos_token_id"] = eos_token_id

    return config


def load_model(
    model_path: Path,
    lazy: bool = False,
    strict: bool = True,
    model_config: Optional[Dict[str, Any]] = None,
    get_model_classes: Callable[[dict], Tuple[Type[nn.Module], Type]] = _get_classes,
) -> Tuple[nn.Module, dict]:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        strict (bool): Whether or not to raise an exception if weights don't
            match. Default: ``True``
        model_config (dict, optional): Optional configuration parameters for the
            model. Defaults to an empty dictionary.
        get_model_classes (Callable[[dict], Tuple[Type[nn.Module], Type]], optional):
            A function that returns the model class and model args class given a config.
            Defaults to the ``_get_classes`` function.

    Returns:
        Tuple[nn.Module, dict[str, Any]]: The loaded and initialized model and config.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    config = load_config(model_path)
    if model_config is not None:
        config.update(model_config)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))

    if not weight_files and strict:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    if (model_file := config.get("model_file")) is not None:
        spec = importlib.util.spec_from_file_location(
            "custom_model",
            model_path / model_file,
        )
        arch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arch)
        model_class, model_args_class = arch.Model, arch.ModelArgs
    else:
        model_class, model_args_class = get_model_classes(config=config)

    if "quantization_config" not in config:
        text_config = config.get("text_config", {})
        if "quantization_config" in text_config:
            config["quantization_config"] = text_config["quantization_config"]

    model_args = model_args_class.from_dict(config)

    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    def _quantize(quantization):
        def class_predicate(p, m):
            # Handle custom per layer quantizations
            if p in config["quantization"]:
                return config["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    if (quantization := config.get("quantization", None)) is not None:
        _quantize(quantization)

    elif quantization_config := config.get("quantization_config", False):
        # Handle legacy quantization config
        quant_method = quantization_config["quant_method"]
        if quant_method == "bitnet":
            from .models.bitlinear_layers import bitnet_quantize

            model = bitnet_quantize(model, quantization_config)
        elif quant_method == "mxfp4":
            quantization = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)
        elif quant_method == "compressed-tensors":
            quantization = {"group_size": 32, "bits": 4, "mode": "affine"}
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)
        elif quant_method in ("awq", "gptq"):
            # Transform AutoAWQ/GPTQ packed weights to MLX format
            weights, quantization = _transform_awq_weights(weights, quantization_config)
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)

    if config.get("quantize_activations", False):

        def _maybe_qq(m):
            if isinstance(m, nn.QuantizedLinear):
                if m.mode not in ("nvfp4", "mxfp8"):
                    raise ValueError(
                        "Mode ({m.mode}) does not support activation quantization"
                    )
                if m.get("bias", False):
                    raise ValueError(
                        "Linear layer with bias does not support activation quantization"
                    )
                out_dims, in_dims = m.weight.shape
                in_dims *= 32 // m.bits
                return nn.QQLinear(in_dims, out_dims, m.group_size, m.bits, m.mode)
            else:
                return m

        leaves = tree_map(_maybe_qq, model.leaf_modules(), is_leaf=nn.Module.is_module)

        model.update_modules(leaves)

    model.eval()
    model.load_weights(list(weights.items()), strict=strict)

    if not lazy:
        mx.eval(model.parameters())

    return model, config


def load_adapters(model: nn.Module, adapter_path: str) -> nn.Module:
    from .tuner.utils import load_adapters as _load_adapters

    return _load_adapters(model, adapter_path)


def load_tokenizer(model_path, tokenizer_config_extra=None, eos_token_ids=None):
    """Load a huggingface tokenizer and try to infer the type of streaming
    detokenizer to use.
    """
    model_path = _download(
        model_path,
        allow_patterns=[
            "*.json",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "tiktoken.model",
            "*.txt",
            "*.jsonl",
            "*.jinja",
        ],
    )
    return _load_tokenizer(
        model_path,
        tokenizer_config_extra,
        eos_token_ids=eos_token_ids,
    )


def load(
    path_or_hf_repo: str,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    adapter_path: Optional[str] = None,
    lazy: bool = False,
    return_config: bool = False,
    revision: Optional[str] = None,
) -> Union[
    Tuple[nn.Module, TokenizerWrapper],
    Tuple[nn.Module, TokenizerWrapper, Dict[str, Any]],
]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If ``False`` eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        return_config (bool: If ``True`` return the model config as the last item..
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.
    Returns:
        Union[Tuple[nn.Module, TokenizerWrapper], Tuple[nn.Module, TokenizerWrapper, Dict[str, Any]]]:
            A tuple containing the loaded model, tokenizer and, if requested, the model config.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = _download(path_or_hf_repo, revision=revision)

    model, config = load_model(model_path, lazy, model_config=model_config)
    if adapter_path is not None:
        model = load_adapters(model, adapter_path)
        model.eval()
    tokenizer = load_tokenizer(
        model_path, tokenizer_config, eos_token_ids=config.get("eos_token_id", None)
    )

    if return_config:
        return model, tokenizer, config
    else:
        return model, tokenizer


def _get_total_physical_memory() -> int:
    """Return total physical memory in bytes (macOS / Linux)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError):
        return 0


def _check_memory_guard(
    threshold_bytes: int,
    layer_idx: int,
    num_layers: int,
) -> None:
    """Abort loading if remaining memory drops below threshold.

    This prevents macOS kernel panics (watchdog timeout) caused by memory
    compressor saturation when loading 600GB+ models in distributed TP.
    Both nodes will shut down: the exiting node calls ``sys.exit``, and the
    peer's ``mx.distributed.all_sum`` barrier will timeout, triggering its
    own cleanup path in ``__main__.py``.
    """
    try:
        active = mx.get_active_memory()
        peak = mx.get_peak_memory()
    except AttributeError:
        return  # older MLX without metal memory API

    total_ram = _get_total_physical_memory()
    if total_ram == 0:
        return  # cannot determine system memory

    remaining = total_ram - active

    if remaining < threshold_bytes:
        logger.critical(
            "MEMORY GUARD: aborting model load — remaining memory %.1f GB "
            "is below safety threshold %.1f GB "
            "(layer %d/%d, active=%.1f GB, peak=%.1f GB, total_ram=%.1f GB). "
            "Exiting to prevent kernel panic.",
            remaining / (1024**3),
            threshold_bytes / (1024**3),
            layer_idx + 1,
            num_layers,
            active / (1024**3),
            peak / (1024**3),
            total_ram / (1024**3),
        )
        sys.exit(1)


def _extract_layer_index(param_name: str) -> int | None:
    """Extract transformer layer index from a flattened parameter name.

    Handles patterns like:
      - "model.layers.42.self_attn.q_proj.weight"
      - "layers.0.mlp.gate_proj.weight"
      - "transformer.h.10.attn.weight"
      - "blocks.5.norm.weight"
    """
    parts = param_name.split(".")
    for i, part in enumerate(parts):
        if part in ("layers", "h", "blocks") and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return None


def _chunked_eval_params(model) -> None:
    """Evaluate model parameters layer-by-layer to limit peak memory.

    Instead of ``mx.eval(model.parameters())`` which materializes all lazy
    weight tensors simultaneously (problematic for 600GB+ models in distributed
    tensor-parallel setups), this evaluates parameters in chunks: non-layer
    params first, then one transformer layer at a time.

    After each layer is materialized the lazy computation graph for the
    preceding full (pre-shard) tensors becomes unreachable and the MLX
    allocator can reclaim that memory before the next layer is loaded.

    A memory guard (configurable via ``MLX_MEMORY_GUARD_GB``) monitors
    remaining system memory after each layer and aborts with ``sys.exit(1)``
    if it drops below the safety threshold, preventing macOS kernel panics.
    Set to ``0`` to disable.  Default: 10% of physical RAM (min 5 GB).
    """
    all_params = dict(tree_flatten(model.parameters()))

    layer_params: dict[int, list] = {}
    non_layer_params: list = []

    for name, param in all_params.items():
        layer_idx = _extract_layer_index(name)
        if layer_idx is not None:
            layer_params.setdefault(layer_idx, []).append(param)
        else:
            non_layer_params.append(param)

    num_layers = len(layer_params)

    # Fall back to bulk eval if the model has no recognisable layer structure.
    if num_layers == 0:
        logger.warning(
            "_chunked_eval_params: no layer structure detected, "
            "falling back to mx.eval(model.parameters())"
        )
        mx.eval(model.parameters())
        return

    # --- memory guard threshold ---
    env_guard = os.environ.get("MLX_MEMORY_GUARD_GB")
    if env_guard is not None:
        guard_bytes = int(float(env_guard) * (1024**3))
    else:
        total_ram = _get_total_physical_memory()
        guard_bytes = max(int(total_ram * 0.10), 5 * (1024**3)) if total_ram > 0 else 0

    if guard_bytes > 0:
        logger.info("Memory guard active: threshold=%.1f GB", guard_bytes / (1024**3))

    logger.info(
        "Chunked eval: %d layers, %d non-layer param groups",
        num_layers,
        len(non_layer_params),
    )

    # 1. Materialise non-layer params (embeddings, final norm, lm_head …)
    if non_layer_params:
        mx.eval(*non_layer_params)
        gc.collect()
        mx.clear_cache()

    if guard_bytes > 0:
        _check_memory_guard(guard_bytes, -1, num_layers)

    # 2. Materialise one layer at a time.
    for idx in sorted(layer_params.keys()):
        mx.eval(*layer_params[idx])
        gc.collect()
        mx.clear_cache()
        if guard_bytes > 0:
            _check_memory_guard(guard_bytes, idx, num_layers)

    logger.info("Chunked eval complete: %d layers materialised", num_layers)


def _find_layers_container(model) -> tuple:
    """Locate the (parent_module, attr_name) holding the transformer layers list.

    All model ``shard()`` methods iterate ``self.model.layers``.  This helper
    walks the module tree to find the ``layers`` attribute used during sharding
    so that ``_eval_shard_eval`` can intercept iteration.

    Returns:
        ``(parent, "layers")`` if found, otherwise ``(None, None)``.
    """
    # Common patterns (ordered by specificity):
    #   kimi_k25:     model.language_model.model.layers
    #   deepseek_v3:  model.model.layers
    #   llama:        model.model.layers
    candidates = []

    def _walk(obj, depth=0):
        if depth > 4:
            return
        layers = getattr(obj, "layers", None)
        if isinstance(layers, list) and len(layers) > 0 and isinstance(layers[0], nn.Module):
            candidates.append((obj, "layers", depth))
        for attr_name in ("model", "language_model", "transformer"):
            child = getattr(obj, attr_name, None)
            if child is not None and isinstance(child, nn.Module):
                _walk(child, depth + 1)

    _walk(model)
    if not candidates:
        return None, None
    # Pick the deepest match — that is the one the shard() loop references
    candidates.sort(key=lambda c: c[2], reverse=True)
    return candidates[0][0], candidates[0][1]


class _EvalShardEvalIter:
    """A list-like wrapper that interposes mx.eval / gc between shard iterations.

    When ``model.shard(group)`` iterates ``self.model.layers``, each element
    goes through the following sequence:

    1. **Before yield** — ``mx.eval(layer.parameters())`` materialises the
       *original* (full, pre-shard) lazy-mmap weights for this layer.
    2. **Yield** — the shard method applies ``shard_linear`` / ``shard_inplace``
       to the layer, creating new lazy tensors that reference the now-concrete
       original weights.
    3. **On the next iteration** (or after the loop ends) — the *previous*
       layer's sharded parameters are materialised with ``mx.eval``, after
       which its original weights become unreachable and can be freed.

    This keeps peak Metal memory to roughly *one full layer + one sharded
    layer* at a time, instead of holding all layers' lazy shard graphs
    simultaneously.

    The wrapper also delegates ``len()`` and index access so that shard methods
    using ``enumerate(self.model.layers)`` or ``len(self.model.layers)`` keep
    working.
    """

    def __init__(self, layers: list, guard_bytes: int = 0):
        self._layers = layers
        self._guard_bytes = guard_bytes
        self._prev_layer = None
        self._prev_idx = -1

    # --- list protocol (read-only) so shard methods can len() / index ---
    def __len__(self):
        return len(self._layers)

    def __getitem__(self, idx):
        return self._layers[idx]

    def __iter__(self):
        num = len(self._layers)
        for i, layer in enumerate(self._layers):
            # Flush the *previous* layer's sharded result so its original
            # (pre-shard) concrete weights become unreachable.
            if self._prev_layer is not None:
                eval_with_timeout(self._prev_layer.parameters(), timeout_seconds=300.0)
                gc.collect()
                mx.clear_cache()
                if self._guard_bytes > 0:
                    _check_memory_guard(self._guard_bytes, self._prev_idx, num)

            # Materialise this layer's original (lazy-mmap) weights.
            eval_with_timeout(layer.parameters(), timeout_seconds=300.0)
            gc.collect()
            mx.clear_cache()

            self._prev_layer = layer
            self._prev_idx = i
            yield layer

        # Flush the last layer after the shard loop finishes.
        if self._prev_layer is not None:
            eval_with_timeout(self._prev_layer.parameters(), timeout_seconds=300.0)
            gc.collect()
            mx.clear_cache()
            if self._guard_bytes > 0:
                _check_memory_guard(self._guard_bytes, self._prev_idx, num)


def _eval_shard_eval(model, group) -> None:
    """Materialise and shard model weights layer-by-layer (exo-style).

    The naive approach — ``model.shard(group); mx.eval(model.parameters())``
    — creates a single giant lazy computation graph spanning *all* layers.
    MLX cannot free intermediate buffers because every layer's shard ops
    reference the original lazy-mmap arrays simultaneously.

    This function instead interposes ``mx.eval`` calls *inside* the shard
    loop so that only one layer's full-size tensors are live at a time:

    For each transformer layer:
      1. ``mx.eval(layer.parameters())`` — materialise full (pre-shard) weights
      2. ``shard_linear`` / ``shard_inplace`` — create lazy shard ops on the
         now-concrete data (short graph, not referencing mmap)
      3. ``mx.eval(layer.parameters())`` — materialise the sharded result
      4. ``gc.collect()`` + ``mx.clear_cache()`` — free pre-shard tensors

    Peak memory: ~1 full layer + 1 sharded layer at a time.
    """
    container, attr = _find_layers_container(model)
    if container is None:
        logger.warning(
            "_eval_shard_eval: could not locate layers container; "
            "falling back to shard + chunked eval"
        )
        model.shard(group)
        _chunked_eval_params(model)
        return

    original_layers = getattr(container, attr)
    num_layers = len(original_layers)
    logger.info("eval-shard-eval: %d transformer layers", num_layers)

    # --- memory guard threshold ---
    env_guard = os.environ.get("MLX_MEMORY_GUARD_GB")
    if env_guard is not None:
        guard_bytes = int(float(env_guard) * (1024**3))
    else:
        total_ram = _get_total_physical_memory()
        guard_bytes = max(int(total_ram * 0.10), 5 * (1024**3)) if total_ram > 0 else 0

    if guard_bytes > 0:
        logger.info("Memory guard active: threshold=%.1f GB", guard_bytes / (1024**3))

    # 1. Materialise non-layer params (embeddings, final norm, lm_head, …)
    all_params = dict(tree_flatten(model.parameters()))
    non_layer = [p for name, p in all_params.items()
                 if _extract_layer_index(name) is None]
    if non_layer:
        logger.info("Materialising %d non-layer param groups", len(non_layer))
        mx.eval(*non_layer)
        gc.collect()
        mx.clear_cache()
    del all_params, non_layer

    if guard_bytes > 0:
        _check_memory_guard(guard_bytes, -1, num_layers)

    # 2. Replace the layers list with our eval-interposing wrapper, then
    #    call model.shard().  The wrapper's __iter__ will eval each layer's
    #    original weights before yield and the previous layer's sharded
    #    weights at the start of the next iteration.
    wrapper = _EvalShardEvalIter(original_layers, guard_bytes=guard_bytes)
    setattr(container, attr, wrapper)
    try:
        model.shard(group)
    finally:
        # Restore the original list (which has been mutated in-place by shard)
        setattr(container, attr, original_layers)

    logger.info("eval-shard-eval complete: %d layers materialised", num_layers)


def sharded_load(
    repo,
    pipeline_group: Optional[mx.distributed.Group] = None,
    tensor_group: Optional[mx.distributed.Group] = None,
    return_config: bool = False,
):
    # --- Pre-load system memory check ---
    # Warn if wired memory is too high (leftover from previous sessions)
    try:
        import subprocess as _sp

        _vmstat = _sp.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        if _vmstat.returncode == 0:
            _page_size = 16384  # macOS ARM64 page size
            for _line in _vmstat.stdout.splitlines():
                if "Pages free" in _line:
                    _free_pages = int(_line.split(":")[1].strip().rstrip("."))
                    _free_gb = (_free_pages * _page_size) / (1024**3)
                    if _free_gb < 50:
                        logger.warning(
                            "Low free memory: %.1f GB. If loading fails with exit 255, "
                            "reboot to reclaim wired Metal memory.",
                            _free_gb,
                        )
                    else:
                        logger.info("Free memory: %.1f GB", _free_gb)
                    break
            del _vmstat
    except Exception:
        pass

    # Get model path with everything but weight safetensors
    model_path = _download(
        repo,
        allow_patterns=[
            "*.json",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "tiktoken.model",
            "*.txt",
            "*.jsonl",
            "*.jinja",
        ],
    )

    # Lightweight config-only probe to determine sharding capabilities
    # without loading any safetensors (avoids OOM on large models like
    # Kimi K2.5 with 182 shards / 612 GB).
    config = load_config(model_path)
    model_class, model_args_class = _get_classes(config=config)
    has_tensor_parallel = hasattr(model_class, "shard")

    # For pipelining we need a model instance to inspect the inner .model
    # attribute, but we can build the skeleton without touching safetensors.
    model_args = model_args_class.from_dict(config)
    _probe = model_class(model_args)
    has_pipelining = hasattr(getattr(_probe, "model", None), "pipeline")

    if pipeline_group is not None and not has_pipelining:
        raise ValueError(
            "The model does not support pipelining but a pipeline_group was provided"
        )
    if tensor_group is not None and not has_tensor_parallel:
        raise ValueError(
            "The model does not support tensor parallelism but a tensor_group was provided"
        )
    if not has_pipelining and not has_tensor_parallel:
        raise ValueError("The model does not support any sharding")

    if pipeline_group is tensor_group is None:
        if has_tensor_parallel:
            tensor_group = mx.distributed.init()
        elif has_pipelining:
            pipeline_group = mx.distributed.init()

    # If pipelining then figure out which files we need for the local shard
    if pipeline_group is not None:
        _probe.model.pipeline(pipeline_group)

        # Figure out which files we need for the local shard
        with open(model_path / "model.safetensors.index.json", "r") as fid:
            weight_index = json.load(fid)["weight_map"]

        local_files = set()
        for k, _ in tree_flatten(_probe.parameters()):
            if file_name := weight_index.get(k, None) is None:
                raise ValueError(
                    "Pipeline loading is only supported for MLX converted models."
                )
            local_files.add(weight_index[k])

        # Download weights for local shard
        _download(repo, allow_patterns=local_files)
    else:
        _download(repo)

    # Free the lightweight probe model before the real load
    del _probe
    gc.collect()
    mx.clear_cache()

    # Load and shard the model, and load the weights
    tokenizer = load_tokenizer(
        model_path,
        {"trust_remote_code": True},
        eos_token_ids=config.get("eos_token_id", None),
    )
    model, _ = load_model(model_path, lazy=True, strict=False)

    if tensor_group is not None:
        _eval_shard_eval(model, tensor_group)
    elif pipeline_group is not None:
        model.model.pipeline(pipeline_group)
        _chunked_eval_params(model)
    else:
        _chunked_eval_params(model)

    set_wired_limit_for_model(model)

    # Synchronize processes to avoid timeout
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    if return_config:
        return model, tokenizer, config
    else:
        return model, tokenizer


def pipeline_load(repo, return_config=False):
    return sharded_load(repo, mx.distributed.init(), None, return_config)


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def create_model_card(path: Union[str, Path], hf_path: Union[str, Path, None]):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (Union[str, Path]): Local path to the model.
        hf_path (Union[str, Path, None]): Path to the original Hugging Face model.
    """
    from huggingface_hub import ModelCard, ModelCardData

    if hf_path is None:
        card = ModelCard.from_template(ModelCardData(language="en"))
    else:
        card = ModelCard.load(hf_path)
    card.data.library_name = "mlx"
    card.data.pipeline_tag = "text-generation"
    if card.data.tags is None:
        card.data.tags = ["mlx"]
    elif "mlx" not in card.data.tags:
        card.data.tags += ["mlx"]
    if hf_path is not None:
        card.data.base_model = str(hf_path)
    card.text = ""
    card.save(os.path.join(path, "README.md"))


def upload_to_hub(path: str, upload_repo: str):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
    """
    from huggingface_hub import HfApi, ModelCard, logging

    from . import __version__

    logging.set_verbosity_info()
    card_path = Path(path) / "README.md"
    card = ModelCard.load(card_path)

    hf_path = card.data.base_model

    if hf_path is not None:
        provenance = f"""
        This model [{upload_repo}](https://huggingface.co/{upload_repo}) was
        converted to MLX format from [{hf_path}](https://huggingface.co/{hf_path})
        using mlx-lm version **{__version__}**.
        """
    else:
        provenance = ""

    card.text = dedent(
        f"""
        # {upload_repo}
        {provenance}
        ## Use with mlx

        ```bash
        pip install mlx-lm
        ```

        ```python
        from mlx_lm import load, generate

        model, tokenizer = load("{upload_repo}")

        prompt = "hello"

        if tokenizer.chat_template is not None:
            messages = [{{"role": "user", "content": prompt}}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_dict=False,
            )

        response = generate(model, tokenizer, prompt=prompt, verbose=True)
        ```
        """
    )
    card.save(card_path)

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_large_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def save_model(
    save_path: Union[str, Path],
    model: nn.Module,
    *,
    donate_model: bool = False,
) -> None:
    """Save model weights and metadata index into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    weights = dict(tree_flatten(model.parameters()))
    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {
        "metadata": {
            "total_size": total_size,
            "total_parameters": get_total_parameters(model),
        },
        "weight_map": {},
    }
    if donate_model:
        model.update(tree_map(lambda _: mx.array([]), model.parameters()))

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    weights.clear()
    del weights

    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def quantize_model(
    model: nn.Module,
    config: dict,
    group_size: Optional[int],
    bits: Optional[int],
    mode: str = "affine",
    quant_predicate: Optional[Callable[[str, nn.Module], Union[bool, dict]]] = None,
) -> Tuple[nn.Module, dict]:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        group_size (Optional[int]): Group size for quantization.
        bits (Optional[int]): Bits per weight for quantization.
        mode (str): The quantization mode.
        quant_predicate (Callable): A callable that decides how to quantize
          each layer based on the path. Accepts the layer `path` and the
          `module`. Returns either a bool to signify quantize/no quantize or
          a dict of quantization parameters to pass to `to_quantized`.

    Returns:
        Tuple: Tuple containing quantized model and config.
    """

    def defaults_for_mode(mode, group_size, bits):
        mode_defaults = {
            "affine": (64, 4),
            "mxfp4": (32, 4),
            "nvfp4": (16, 4),
            "mxfp8": (32, 8),
        }
        default_group_size, default_bits = mode_defaults[mode]
        return group_size or default_group_size, bits or default_bits

    quantized_config = copy.deepcopy(config)

    quant_predicate = quant_predicate or getattr(model, "quant_predicate", None)
    group_size, bits = defaults_for_mode(mode, group_size, bits)
    quant_params = {"group_size": group_size, "bits": bits, "mode": mode}
    if "quantization" in quantized_config:
        # If the model is already partially quantized, return params so that
        # the config is set on a per-layer basis
        fine_grained_config = True
    else:
        fine_grained_config = False
        quantized_config["quantization"] = quant_params

    def wrapped_predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[-1] % group_size != 0:
            return False
        bool_or_params = True
        if quant_predicate is not None:
            bool_or_params = quant_predicate(path, module)
        if isinstance(bool_or_params, dict):
            quantized_config["quantization"][path] = bool_or_params
        elif fine_grained_config and bool_or_params:
            quantized_config["quantization"][path] = quant_params
        return bool_or_params

    nn.quantize(
        model,
        group_size,
        bits,
        mode=mode,
        class_predicate=wrapped_predicate,
    )
    # support hf model tree #957
    quantized_config["quantization_config"] = quantized_config["quantization"]

    bpw = compute_bits_per_weight(model)
    print(f"[INFO] Quantized model with {bpw:.3f} bits per weight.")

    return model, quantized_config


def dequantize_model(model: nn.Module) -> nn.Module:
    """
    Dequantize the quantized layers in the model.

    Args:
        model (nn.Module): The model with quantized layers.

    Returns:
        nn.Module: The model with dequantized layers.
    """
    from .models.switch_layers import QuantizedSwitchLinear, SwitchLinear

    dequantize_layers = []
    for name, module in model.named_modules():
        bias = "bias" in module
        if isinstance(module, nn.QuantizedLinear):
            cls = nn.Linear
            kwargs = {"bias": bias}
        elif isinstance(module, nn.QuantizedEmbedding):
            kwargs = {}
            cls = nn.Embedding
        elif isinstance(module, QuantizedSwitchLinear):
            kwargs = {"bias": bias}
            cls = SwitchLinear
        else:
            continue
        weight = mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            module.group_size,
            module.bits,
            module.mode,
        )
        args = weight.shape[::-1]
        m = cls(*args, **kwargs)
        if bias:
            m.bias = module.bias
        m.weight = weight
        dequantize_layers.append((name, m))

    if len(dequantize_layers) > 0:
        model.update_modules(tree_unflatten(dequantize_layers))
    return model


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # Clean unused keys
    config.pop("_name_or_path", None)
    config.pop("vision_config", None)
    if "quantization" in config:
        config["quantization_config"] = config["quantization"]

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


def save(
    dst_path: Union[str, Path],
    src_path_or_repo: Union[str, Path],
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    config: Dict[str, Any],
    donate_model: bool = True,
):

    src_path = Path(src_path_or_repo)
    if not src_path.exists():
        hf_repo = src_path_or_repo
        src_path = hf_repo_to_path(hf_repo)
    else:
        hf_repo = None

    dst_path = Path(dst_path)
    save_model(dst_path, model, donate_model=True)
    save_config(config, config_path=dst_path / "config.json")
    tokenizer.save_pretrained(dst_path)

    for p in ["*.py", "generation_config.json"]:
        for file in glob.glob(str(src_path / p)):
            shutil.copy(file, dst_path)

    create_model_card(dst_path, hf_repo)


def common_prefix_len(list1, list2):
    """
    Calculates the length of the common prefix of two lists.

    Args:
        list1: The first list of strings.
        list2: The second list of strings.

    Returns:
        The length of the common prefix. Returns 0 if lists are empty
        or do not match at the first element.
    """
    # Determine the maximum possible length of the common prefix
    min_len = min(len(list1), len(list2))

    # Iterate up to the length of the shorter list
    for i in range(min_len):
        if list1[i] != list2[i]:
            # Mismatch found, the common prefix length is the current index
            return i

    # No mismatch found within the bounds of the shorter list,
    # so the common prefix length is the length of the shorter list.
    return min_len


def does_model_support_input_embeddings(model: nn.Module) -> bool:
    """
    Check if the model supports input_embeddings in its call signature.
    Args:
        model (nn.Module): The model to check.
    Returns:
        bool: True if the model supports input_embeddings, False otherwise.
    """
    try:
        signature = inspect.signature(model.__call__)
        return "input_embeddings" in signature.parameters
    except (ValueError, TypeError):
        return False
