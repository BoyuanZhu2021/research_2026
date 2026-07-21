"""Auditable Qwen3.5 fast-kernel preflight for the H20 attacker."""
from __future__ import annotations

import copy
import importlib.metadata
from typing import Any

from .victim_decision_protocol import canonical_sha256


QWEN35_FAST_KERNEL_PINS = {
    "fla-core": "0.5.1",
    "causal-conv1d": "1.6.2.post1",
    "einops": "0.8.2",
    "tilelang": "0.1.12",
}
QWEN35_FAST_KERNEL_ATTRS = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "recurrent_gated_delta_rule",
)


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in QWEN35_FAST_KERNEL_PINS:
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def qwen35_fast_kernel_status(model: Any | None = None) -> dict:
    versions = _package_versions()
    layers = []
    if model is not None:
        for name, module in model.named_modules():
            if type(module).__name__ == "Qwen3_5GatedDeltaNet":
                availability = {
                    attribute: callable(getattr(module, attribute, None))
                    for attribute in QWEN35_FAST_KERNEL_ATTRS
                }
                layers.append({"name": name, "available": availability})
    result = {
        "schema_version": 1,
        "kind": "h1_qwen35_attacker_fast_kernel_status",
        "required_distribution_pins": copy.deepcopy(QWEN35_FAST_KERNEL_PINS),
        "observed_distribution_versions": versions,
        "distribution_pins_match": versions == QWEN35_FAST_KERNEL_PINS,
        "gated_delta_net_layer_count": len(layers),
        "layers": layers,
        "model_fast_path_ready": bool(layers) and all(
            all(layer["available"].values()) for layer in layers
        ),
    }
    result["payload_sha256"] = canonical_sha256(result)
    return result


def require_qwen35_fast_kernels(model: Any) -> dict:
    status = qwen35_fast_kernel_status(model)
    if not status["distribution_pins_match"]:
        raise RuntimeError(
            "Qwen3.5 fast-kernel distribution pins are absent or drifted: "
            f"{status['observed_distribution_versions']}"
        )
    if not status["model_fast_path_ready"]:
        raise RuntimeError("Qwen3.5 model did not bind all fast-kernel callables")
    return status


__all__ = [
    "QWEN35_FAST_KERNEL_ATTRS",
    "QWEN35_FAST_KERNEL_PINS",
    "qwen35_fast_kernel_status",
    "require_qwen35_fast_kernels",
]
