"""Powered runtime adapter for the formal single-H20 Gate 1 and smoke.

This module is intentionally independent of both training entrypoints.  In
particular, importing it cannot initialize the dual-V100 distributed runtime.
All CUDA/Transformers imports remain lazy until the deployed tree, live FP8
victim, exact environment, and physical H20 identity have been checked.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from .attacker import ATTACKER_OUTPUT_PROTOCOL
from .deployment_identity import verify_deployment
from .h20_serving_identity import (
    ENDPOINT as VICTIM_URL,
    validate_service_manifest,
)
from .model_pins import (
    ATTACKER_MODEL,
    ATTACKER_REVISION,
    REMOTE_HF_HOME,
    VICTIM_H20_SERVED_NAME,
    VICTIM_REVISION,
)
from .runtime_profile import (
    H20_GPU_UUID_ENV,
    H20_RUNTIME_PROFILE_SHA256,
    LEGACY_H20_PROFILE_ID,
)
from .tooluse_gate1_spec import (
    ATTACKER_MAX_NEW,
    ATTACKER_TEMPERATURE,
    RUN_SEED,
    VICTIM_INCLUDE_STOP_STR_IN_OUTPUT,
    VICTIM_MAX_TOKENS,
    VICTIM_QUANTIZATION,
    VICTIM_REACT_STOP,
    VICTIM_TEMPERATURE,
)
from .victim_decision_protocol import VICTIM_DECISION_PROTOCOL


FORMAL_DEPLOYMENT_ROOT = Path("/root/autodl-tmp/h1mt")
H20_GATE_DEPLOYMENT_REQUIRED = (
    "data/InjecAgent/data/test_cases_ds_base.json",
    "data/InjecAgent/data/tools.json",
    "data/InjecAgent/src/prompts/agent_prompts.py",
    "code/configs/injecagent_ds_base_split_v1.json",
    "code/scripts/h1_tooluse_gate1_local.py",
    "code/scripts/h1_victim_fp8_repeatability.py",
    "code/scripts/h1_victim_quant_spotcheck.py",
    "code/src/attacker.py",
    "code/src/deployment_identity.py",
    "code/src/domains/base.py",
    "code/src/domains/injecagent.py",
    "code/src/domains/injecagent_ds_oracle.py",
    "code/src/domains/tooluse_injection.py",
    "code/src/domains/tooluse_oracle.py",
    "code/src/h20_gate_runtime.py",
    "code/src/h20_serving_identity.py",
    "code/src/interactive_episode.py",
    "code/src/model_pins.py",
    "code/src/mt_victim.py",
    "code/src/runtime_profile.py",
    "code/src/tooluse_gate1_spec.py",
    "code/src/victim_decision_protocol.py",
)

ATTACKER_RUNTIME_IDENTITY = {
    "model": ATTACKER_MODEL,
    "revision": ATTACKER_REVISION,
    "quantization": "bitsandbytes-nf4",
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "output_protocol": ATTACKER_OUTPUT_PROTOCOL,
}
MAX_H20_MEMORY_ACCOUNTING_DELTA_MIB = 1024

def attacker_call_seed(goal_id: str, turn: int) -> int:
    """Return the frozen Gate seed for one attacker call."""
    if not isinstance(goal_id, str) or not goal_id:
        raise ValueError("Gate attacker seed requires a non-empty goal_id")
    if not isinstance(turn, int) or isinstance(turn, bool) or not 1 <= turn <= 5:
        raise ValueError("Gate attacker seed turn must be an integer in [1, 5]")
    payload = f"{RUN_SEED}\0{goal_id}\0{turn}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _exact_or_set_environment(service_manifest: dict) -> dict[str, str]:
    """Bind the Gate process to the live victim's exact physical H20 before torch."""
    manifest = validate_service_manifest(
        service_manifest, expected_quantization=VICTIM_QUANTIZATION
    )
    gpu_uuid = manifest["gpu"]["uuid"]
    exact = {
        "ATTACKER_MODEL": ATTACKER_MODEL,
        "ATTACKER_REVISION": ATTACKER_REVISION,
        "VICTIM_MODEL": VICTIM_H20_SERVED_NAME,
        "VICTIM_REVISION": VICTIM_REVISION,
        "VICTIM_QUANT": VICTIM_QUANTIZATION,
        "VICTIM_URL": VICTIM_URL,
        "HF_HOME": REMOTE_HF_HOME,
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "H1_RUNTIME_PROFILE_ID": LEGACY_H20_PROFILE_ID,
        "H1_RUNTIME_PROFILE_SHA256": H20_RUNTIME_PROFILE_SHA256,
    }
    mismatched = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in exact.items()
        if os.environ.get(key) not in (None, expected)
    }
    if mismatched:
        raise RuntimeError(f"formal H20 Gate environment override mismatch: {mismatched}")
    for key, value in exact.items():
        os.environ[key] = value

    # These selectors are never silently inferred: the caller must explicitly
    # launch the Gate on the UUID sealed by the live victim manifest.
    selectors = {
        H20_GPU_UUID_ENV: gpu_uuid,
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
    }
    wrong_selectors = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in selectors.items()
        if os.environ.get(key) != expected
    }
    if wrong_selectors:
        raise RuntimeError(
            f"formal H20 Gate requires explicit exact GPU selectors: {wrong_selectors}"
        )

    allowed_single_process = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
    }
    distributed_drift = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in allowed_single_process.items()
        if os.environ.get(key) not in (None, "", expected)
    }
    if distributed_drift:
        raise RuntimeError(
            "formal H20 Gate rejects distributed/V100 process state: "
            f"{distributed_drift}"
        )
    return {**exact, **selectors}


def validate_single_h20_cuda(torch_module: Any, service_manifest: dict) -> dict:
    """Verify one BF16-capable logical H20 bound to the sealed victim UUID."""
    manifest = validate_service_manifest(
        service_manifest, expected_quantization=VICTIM_QUANTIZATION
    )
    gpu_uuid = manifest["gpu"]["uuid"]
    if os.environ.get(H20_GPU_UUID_ENV) != gpu_uuid \
            or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise RuntimeError("H20 CUDA validation observed GPU selector drift")
    cuda = torch_module.cuda
    if not cuda.is_available() or cuda.device_count() != 1:
        raise RuntimeError("formal H20 Gate must see exactly one CUDA device")
    cuda.set_device(0)
    properties = cuda.get_device_properties(0)
    name = str(properties.name)
    capability = list(cuda.get_device_capability(0))
    memory_total_mib = int(properties.total_memory // (1024 * 1024))
    if "H20" not in name.upper() or capability != [9, 0] or memory_total_mib < 90_000:
        raise RuntimeError(
            "logical cuda:0 is not the registered H20: "
            f"{name!r}/{capability}/{memory_total_mib}MiB"
        )
    if not cuda.is_bf16_supported():
        raise RuntimeError("formal H20 Gate requires native BF16 support")
    sealed_gpu = manifest["gpu"]
    if sealed_gpu["name"] != name:
        raise RuntimeError(
            "attacker CUDA device name differs from the sealed victim H20: "
            f"{name!r} != {sealed_gpu['name']!r}"
        )
    # nvidia-smi reports physical device memory while torch exposes the CUDA
    # runtime's allocatable total after driver-reserved regions.  Preserve both
    # measurements and allow only the documented small, one-way accounting gap.
    nvidia_smi_memory_total_mib = int(sealed_gpu["memory_total_mib"])
    memory_total_delta_mib = nvidia_smi_memory_total_mib - memory_total_mib
    if not 0 <= memory_total_delta_mib <= MAX_H20_MEMORY_ACCOUNTING_DELTA_MIB:
        raise RuntimeError(
            "H20 nvidia-smi/Torch memory accounting delta is outside the "
            f"fail-closed range: physical={nvidia_smi_memory_total_mib}MiB "
            f"torch={memory_total_mib}MiB delta={memory_total_delta_mib}MiB "
            f"allowed=[0,{MAX_H20_MEMORY_ACCOUNTING_DELTA_MIB}]"
        )
    return {
        "logical_device_index": 0,
        "logical_device_count": 1,
        "gpu_uuid": gpu_uuid,
        "name": name,
        "compute_capability": capability,
        "memory_total_mib": memory_total_mib,
        "nvidia_smi_memory_total_mib": nvidia_smi_memory_total_mib,
        "torch_memory_total_mib": memory_total_mib,
        "memory_total_delta_mib": memory_total_delta_mib,
        "bf16_supported": True,
        "torch_version": str(torch_module.__version__),
        "cuda_version": str(torch_module.version.cuda),
    }


def _load_powered_stack():
    """Import the model stack only after all torch-free identity checks pass."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    return torch, snapshot_download, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _trim_response(response, pad_id):
    mask = response != pad_id
    if not bool(mask.any()):
        return response[:0]
    return response[:int(mask.nonzero()[-1]) + 1]


def make_gate_generator(model, tokenizer, torch_module, device):
    """Build the batch callable expected by the frozen Gate orchestration."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("formal H20 attacker tokenizer lacks a pad token")

    def tokenize(messages: list[dict]):
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        return (encoded if isinstance(encoded, torch_module.Tensor)
                else encoded["input_ids"])[0]

    def generate(batch_messages: list[list[dict]]) -> list[dict]:
        prompts = [tokenize(messages) for messages in batch_messages]
        if not prompts:
            return []
        max_length = max(prompt.shape[0] for prompt in prompts)
        input_ids = torch_module.full(
            (len(prompts), max_length), pad_id, dtype=torch_module.long
        )
        attention_mask = torch_module.zeros(
            (len(prompts), max_length), dtype=torch_module.long
        )
        for offset, prompt in enumerate(prompts):
            length = prompt.shape[0]
            input_ids[offset, max_length - length:] = prompt
            attention_mask[offset, max_length - length:] = 1
        with torch_module.no_grad():
            output = model.generate(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=ATTACKER_MAX_NEW,
                do_sample=True,
                temperature=ATTACKER_TEMPERATURE,
                top_p=1.0,
                pad_token_id=pad_id,
            )
        results = []
        for offset, prompt in enumerate(prompts):
            response = _trim_response(
                output[offset, max_length:].detach().cpu(), pad_id
            )
            results.append({
                "text": tokenizer.decode(response, skip_special_tokens=True),
                "prompt_ids": prompt.detach().cpu(),
                "resp_ids": response,
            })
        return results

    return generate


def victim_request_payload(
    messages: list[dict], *, max_tokens: int, temperature: float,
    enable_thinking: bool | None, seed: int | None,
    structured_outputs: dict,
) -> dict:
    """Return the exact OpenAI-compatible request used by Gate victim calls."""
    payload = {
        "model": VICTIM_H20_SERVED_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": list(VICTIM_REACT_STOP),
        "include_stop_str_in_output": VICTIM_INCLUDE_STOP_STR_IN_OUTPUT,
        "structured_outputs": structured_outputs,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    if seed is not None:
        payload["seed"] = int(seed)
    return payload


def victim_chat_h20(
    messages: list[dict], *, max_tokens: int = VICTIM_MAX_TOKENS,
    temperature: float = VICTIM_TEMPERATURE, enable_thinking: bool | None = False,
    seed: int | None = 0, retries: int = 1,
    structured_outputs: dict,
) -> str:
    """Call only the exact local H20 vLLM model; no provider fallback exists."""
    body = json.dumps(
        victim_request_payload(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            seed=seed,
            structured_outputs=structured_outputs,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    last = ""
    for attempt in range(retries):
        try:
            req = request.Request(
                f"{VICTIM_URL}/chat/completions",
                data=body,
                method="POST",
                headers={
                    "Authorization": "Bearer EMPTY",
                    "Content-Type": "application/json",
                },
            )
            with request.urlopen(req, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("model") != VICTIM_H20_SERVED_NAME:
                raise ValueError("H20 victim response model identity mismatch")
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("H20 victim response content is not a string")
            if any(marker in content for marker in VICTIM_REACT_STOP):
                raise ValueError("H20 victim response retained the ReAct Observation stop marker")
            return content.strip()
        except (
            error.URLError,
            error.HTTPError,
            OSError,
            KeyError,
            IndexError,
            TypeError,
            TimeoutError,
            ValueError,
        ) as exc:
            last = f"{type(exc).__name__}: {str(exc)[:160]}"
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"H20 victim service failed after {retries} attempts: {last}")


class H20GateRuntime:
    """Pinned offline 4-bit attacker plus the exact live FP8 H20 victim."""

    def __init__(
        self,
        *,
        live_victim_validator: Callable[[str], dict],
        deployment_root: str | Path = FORMAL_DEPLOYMENT_ROOT,
    ):
        root = Path(deployment_root)
        self.deployment_manifest = verify_deployment(
            root, required_paths=H20_GATE_DEPLOYMENT_REQUIRED
        )
        self.victim_manifest = live_victim_validator(VICTIM_QUANTIZATION)
        self.victim_manifest = validate_service_manifest(
            self.victim_manifest, expected_quantization=VICTIM_QUANTIZATION
        )
        self.environment = _exact_or_set_environment(self.victim_manifest)

        (
            torch_module,
            snapshot_download,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        ) = _load_powered_stack()
        self.gpu_runtime = validate_single_h20_cuda(
            torch_module, self.victim_manifest
        )

        snapshot = Path(snapshot_download(
            ATTACKER_MODEL,
            revision=ATTACKER_REVISION,
            local_files_only=True,
        )).resolve()
        if snapshot.name != ATTACKER_REVISION:
            raise RuntimeError(
                "offline attacker snapshot revision mismatch: "
                f"expected {ATTACKER_REVISION}, got {snapshot}"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("formal H20 attacker tokenizer lacks EOS/pad tokens")
            tokenizer.pad_token = tokenizer.eos_token

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_module.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            quantization_config=quantization,
            torch_dtype=torch_module.bfloat16,
            device_map={"": 0},
        )
        if getattr(model, "is_loaded_in_4bit", False) is not True:
            raise RuntimeError("formal H20 Gate attacker is not actually loaded in 4-bit")
        model.eval()
        device = next(model.parameters()).device
        if device.type != "cuda" or device.index not in (None, 0):
            raise RuntimeError(f"formal H20 attacker loaded on unexpected device: {device}")

        self.attacker_runtime = {
            **ATTACKER_RUNTIME_IDENTITY,
            "snapshot_path": str(snapshot),
            "is_loaded_in_4bit": True,
            "device": str(device),
        }
        self.tokenizer = tokenizer
        self.model = model
        self._torch = torch_module
        self._device = device
        self._generate = make_gate_generator(model, tokenizer, torch_module, device)

    def attacker(self, messages: list[dict], context: dict, _goal) -> str:
        seed = attacker_call_seed(context["goal_id"], int(context["turn"]))
        with self._torch.random.fork_rng(devices=[0]):
            self._torch.manual_seed(seed)
            self._torch.cuda.manual_seed_all(seed)
            result = self._generate([messages])
        if len(result) != 1 or not isinstance(result[0].get("text"), str):
            raise RuntimeError("formal H20 attacker generator violated the Gate batch contract")
        return result[0]["text"]

    @staticmethod
    def victim(
        messages: list[dict], _context: dict, _goal, *, structured_outputs: dict,
    ) -> str:
        return victim_chat_h20(
            messages,
            max_tokens=VICTIM_MAX_TOKENS,
            temperature=VICTIM_TEMPERATURE,
            enable_thinking=False,
            seed=0,
            retries=1,
            structured_outputs=structured_outputs,
        )
