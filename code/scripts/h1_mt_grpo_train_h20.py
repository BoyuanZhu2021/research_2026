"""Formal single-H20 full-trajectory GRPO with 4-bit QLoRA.

This entry point is intentionally independent of ``h1_mt_grpo_train.py`` (the
dual-V100 torchrun implementation).  It accepts only the sealed H20 PASS Gate,
its restored-FP8 runtime bundle, the exact deployed tree, and a passing
full-shape budget benchmark.  There is no remote-API or SiliconFlow fallback.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from urllib import error, request

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "src"))

from src.deployment_identity import verify_deployment  # noqa: E402
from src.domains.tooluse_injection import ToolUseInjectionDomain  # noqa: E402
from src.h20_serving_identity import (  # noqa: E402
    ENDPOINT as VICTIM_URL,
    canonical_provenance_summary,
    validate_h20_formal_runtime_bundle,
    validate_live_runtime_check,
)
from src.h20_training_artifacts import (  # noqa: E402
    build_artifact_manifest,
    build_checkpoint_manifest,
    sha256_file,
    tree_sha256,
    validate_artifact_manifest,
    validate_checkpoint_manifest,
)
from src.generation_runtime import cached_eval_generation  # noqa: E402
from src.h20_training_protocol import (  # noqa: E402
    ATTACKER_BNB_QUANT_TYPE,
    ATTACKER_BNB_USE_DOUBLE_QUANT,
    ATTACKER_COMPUTE_DTYPE,
    ATTACKER_QUANTIZATION,
    DATA_IDENTITY,
    FIXED_TRAINING,
    FORMAL_TRAINING_PROTOCOL_SHA256,
    GRADIENT_CLIPPING,
    LORA_CONFIG,
    MODEL_IDENTITY,
    OPTIMIZER_CONFIG,
    ORACLE_AND_INTERACTION,
    QLORA_CONFIG,
    SINGLE_H20_EXECUTION,
    build_benchmark_manifest,
    build_benchmark_result,
    build_goal_schedule,
    canonical_sha256,
    construction_seed_record,
    formal_training_protocol,
    generation_call_seed,
    load_and_validate_formal_training_inputs,
    seal_payload,
    validate_formal_training_values,
    validate_run_config,
)
from src.model_pins import (  # noqa: E402
    ATTACKER_MODEL,
    ATTACKER_REVISION,
    INJECAGENT_COMMIT,
    VICTIM_H20_SERVED_NAME,
    VICTIM_REVISION,
)
from src.runtime_profile import (  # noqa: E402
    H20_GPU_UUID_ENV,
    H20_RUNTIME_PROFILE_SHA256,
    LEGACY_H20_PROFILE_ID,
)
from src.tooluse_gate1_spec import (  # noqa: E402
    VICTIM_MAX_TOKENS,
    goal_ids_sha256,
    load_frozen_gate1,
)

# Powered imports are deliberately lazy.  This keeps ``--help`` and every
# fail-closed argparse error CPU-only, so deployment gates can validate the CLI
# before a torch/PEFT environment (or a GPU) is available.
_POWERED_STACK_LOADED = False


def _load_powered_stack() -> None:
    global _POWERED_STACK_LOADED
    global torch, F, snapshot_download
    global LoraConfig, get_peft_model, prepare_model_for_kbit_training
    global AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessorList
    global verify_cycle_runtime_live, group_advantages, per_turn_rewards
    global rollout_batch, make_victim_batch_fn
    if _POWERED_STACK_LOADED:
        return

    import torch as torch_module
    import torch.nn.functional as functional_module
    from huggingface_hub import snapshot_download as snapshot_download_function
    from peft import (
        LoraConfig as LoraConfigClass,
        get_peft_model as get_peft_model_function,
        prepare_model_for_kbit_training as prepare_model_for_kbit_training_function,
    )
    from transformers import (
        AutoModelForCausalLM as AutoModelForCausalLMClass,
        AutoTokenizer as AutoTokenizerClass,
        BitsAndBytesConfig as BitsAndBytesConfigClass,
        LogitsProcessorList as LogitsProcessorListClass,
    )

    from h1_victim_fp8_repeatability import (
        verify_fp8_cycle_runtime_live as verify_cycle_runtime_live_function,
    )
    from src.mt_grpo import (
        group_advantages as group_advantages_function,
        per_turn_rewards as per_turn_rewards_function,
    )
    from src.mt_rollout import rollout_batch as rollout_batch_function
    from src.mt_victim import make_victim_batch_fn as make_victim_batch_fn_function

    torch = torch_module
    F = functional_module
    snapshot_download = snapshot_download_function
    LoraConfig = LoraConfigClass
    get_peft_model = get_peft_model_function
    prepare_model_for_kbit_training = prepare_model_for_kbit_training_function
    AutoModelForCausalLM = AutoModelForCausalLMClass
    AutoTokenizer = AutoTokenizerClass
    BitsAndBytesConfig = BitsAndBytesConfigClass
    LogitsProcessorList = LogitsProcessorListClass
    verify_cycle_runtime_live = verify_cycle_runtime_live_function
    group_advantages = group_advantages_function
    per_turn_rewards = per_turn_rewards_function
    rollout_batch = rollout_batch_function
    make_victim_batch_fn = make_victim_batch_fn_function
    _POWERED_STACK_LOADED = True


FORMAL_DEPLOYMENT_REQUIRED = (
    "data/InjecAgent/data/test_cases_ds_base.json",
    "data/InjecAgent/data/tools.json",
    "data/InjecAgent/src/prompts/agent_prompts.py",
    "code/configs/injecagent_ds_base_split_v1.json",
    "code/scripts/h1_deploy_mt.py",
    "code/scripts/h1_mt_grpo_train_h20.py",
    "code/scripts/h1_serve_victim_h20.py",
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
    "code/src/h20_training_artifacts.py",
    "code/src/h20_training_protocol.py",
    "code/src/interactive_episode.py",
    "code/src/model_pins.py",
    "code/src/mt_grpo.py",
    "code/src/mt_rollout.py",
    "code/src/mt_victim.py",
    "code/src/runtime_profile.py",
    "code/src/tooluse_gate1_spec.py",
)

BENCHMARK_RUN_KIND = "h1_h20_full_shape_training_benchmark_run"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: str | Path, label: str) -> dict:
    artifact = Path(path)
    try:
        value = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} unavailable: {artifact}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {artifact}")
    return value


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _identity(value: dict, *, path: str | Path | None = None) -> dict:
    result = {"canonical_sha256": canonical_sha256(value)}
    if path is not None:
        artifact = Path(path).resolve()
        result.update({"path": str(artifact), "file_sha256": sha256_file(artifact)})
    return result


def _software_versions() -> dict[str, str]:
    values = {}
    for distribution in (
        "torch", "transformers", "peft", "bitsandbytes", "accelerate",
        "triton", "fla-core", "causal-conv1d", "tilelang",
    ):
        try:
            values[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            values[distribution] = "not-installed"
    return values


def seed_before_model_construction(seed: int) -> dict:
    """Set every RNG used by model/LoRA construction before loading any weights."""
    record = construction_seed_record(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return record


def validate_single_h20_process(runtime_bundle: dict) -> dict:
    """Reject torchrun, hidden extra GPUs, a wrong UUID, or non-H20 CUDA."""
    distributed = {
        "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"
    }
    for key, expected in distributed.items():
        actual = os.environ.get(key)
        if actual not in (None, "", expected):
            raise RuntimeError(f"single-H20 trainer rejects distributed {key}={actual!r}")
    reference = runtime_bundle["restored_fp8_runtime"]
    gpu_uuid = reference["gpu_uuid"]
    for key in (H20_GPU_UUID_ENV, "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(key) != gpu_uuid:
            raise RuntimeError(f"single-H20 trainer requires {key}={gpu_uuid!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("single-H20 trainer must see exactly one CUDA device")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    name = str(properties.name)
    capability = list(torch.cuda.get_device_capability(0))
    memory_total_mib = int(properties.total_memory // (1024 * 1024))
    if "H20" not in name.upper() or capability != [9, 0] or memory_total_mib < 90_000:
        raise RuntimeError(
            f"logical cuda:0 is not the registered H20: {name!r}/{capability}/{memory_total_mib}MiB"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("registered H20 training requires native BF16 support")
    return {
        "logical_device_index": 0,
        "logical_device_count": 1,
        "name": name,
        "compute_capability": capability,
        "memory_total_mib": memory_total_mib,
        "gpu_uuid": gpu_uuid,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
    }


def _validate_environment_overrides() -> None:
    exact = {
        "ATTACKER_MODEL": ATTACKER_MODEL,
        "ATTACKER_REVISION": ATTACKER_REVISION,
        "VICTIM_MODEL": VICTIM_H20_SERVED_NAME,
        "VICTIM_REVISION": VICTIM_REVISION,
        "VICTIM_URL": VICTIM_URL,
    }
    mismatched = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in exact.items()
        if os.environ.get(key) not in (None, expected)
    }
    if mismatched:
        raise RuntimeError(f"formal H20 model/service override mismatch: {mismatched}")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("formal H20 training requires fully offline model loading")


def _validate_gate_deployment_runtime(
    gate_path: Path, run_root: Path,
) -> tuple[dict, dict, dict, dict, dict, Path]:
    gate = load_frozen_gate1(
        gate_path,
        expected_profile_id=LEGACY_H20_PROFILE_ID,
        code_root=CODE,
        verify_external_artifacts=True,
    )
    if gate.get("verdict") != "PASS" or gate.get("passed") is not True:
        raise RuntimeError("formal H20 training requires a PASS Gate")
    quant = gate.get("quantization_check") or {}
    runtime_bundle = validate_h20_formal_runtime_bundle(
        quant.get("runtime_bundle"), require_gate_checks=True
    )
    deployment = verify_deployment(run_root, required_paths=FORMAL_DEPLOYMENT_REQUIRED)
    gate_deployment = gate.get("deployment") or {}
    if Path(gate_deployment.get("root", "")).resolve() != run_root.resolve():
        raise RuntimeError("Gate deployment root differs from the executing H20 tree")
    if (gate_deployment.get("deployed_tree_sha256") != deployment.get("deployed_tree_sha256")
            or gate_deployment.get("injecagent_commit") != INJECAGENT_COMMIT
            or deployment.get("injecagent_commit") != INJECAGENT_COMMIT
            or (quant.get("deployment") or {}).get("deployed_tree_sha256")
            != deployment.get("deployed_tree_sha256")):
        raise RuntimeError("Gate/FP8-repeatability/current deployment identity mismatch")
    cycle_path = Path(quant.get("cycle_status_path", "")).resolve()
    if not cycle_path.is_file() or sha256_file(cycle_path) != quant.get("cycle_status_file_sha256"):
        raise RuntimeError("Gate FP8 cycle status file binding mismatch")
    gate_identity = _identity(gate, path=gate_path)
    runtime_identity = _identity(runtime_bundle)
    deployment_identity = _identity(
        deployment, path=run_root / "deployment_manifest.json"
    )
    return (
        gate, runtime_bundle, deployment,
        gate_identity, runtime_identity, deployment_identity, cycle_path,
    )


def _probe_victim() -> None:
    try:
        with request.urlopen(f"{VICTIM_URL}/models", timeout=15) as response:
            payload = json.loads(response.read().decode())
    except (error.URLError, error.HTTPError, TimeoutError, ValueError) as exc:
        raise RuntimeError(f"H20 victim model endpoint unavailable: {exc}") from exc
    model_ids = [item.get("id") for item in payload.get("data", [])]
    if model_ids != [VICTIM_H20_SERVED_NAME]:
        raise RuntimeError(f"H20 victim model identity mismatch: {model_ids!r}")


def victim_chat(
    messages: list[dict], *, max_tokens=VICTIM_MAX_TOKENS, temperature=0.0,
    enable_thinking=None, seed=None, retries=1, structured_outputs: dict | None = None,
) -> str:
    del seed
    if not isinstance(structured_outputs, dict) or "json" not in structured_outputs:
        raise ValueError("formal H20 training victim requires structured_outputs.json")
    body = json.dumps({
        "model": VICTIM_H20_SERVED_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": list(ORACLE_AND_INTERACTION["victim_generation_stop"]),
        "include_stop_str_in_output": ORACLE_AND_INTERACTION[
            "victim_include_stop_str_in_output"
        ],
        "structured_outputs": structured_outputs,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }).encode()
    last = ""
    for attempt in range(retries):
        try:
            req = request.Request(
                f"{VICTIM_URL}/chat/completions", data=body, method="POST",
                headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
            )
            with request.urlopen(req, timeout=120) as response:
                payload = json.loads(response.read().decode())
            if payload.get("model") != VICTIM_H20_SERVED_NAME:
                raise ValueError("victim response model identity mismatch")
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("victim response content is not a string")
            if any(marker in content
                   for marker in ORACLE_AND_INTERACTION["victim_generation_stop"]):
                raise ValueError("victim response retained the ReAct Observation stop marker")
            return content.strip()
        except (error.URLError, error.HTTPError, KeyError, IndexError, TypeError,
                TimeoutError, ValueError) as exc:
            last = f"{type(exc).__name__}: {str(exc)[:120]}"
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"H20 victim service failed after {retries} attempts: {last}")


def _trim_response(response, pad_id):
    mask = response != pad_id
    if not bool(mask.any()):
        return response[:0]
    return response[:int(mask.nonzero()[-1]) + 1]


def make_gen_batch_fn(
    model, tokenizer, *, training_seed: int, cached_eval_mode: bool = False,
    max_new_tokens: int | None = None, logits_processor_factory=None,
):
    if max_new_tokens is None:
        max_new_tokens = FIXED_TRAINING["max_new"]
    if (not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool)
            or max_new_tokens < 1 or max_new_tokens > FIXED_TRAINING["max_new"]):
        raise ValueError("in-process attacker max_new_tokens is outside the frozen ceiling")
    if logits_processor_factory is not None and not callable(logits_processor_factory):
        raise ValueError("logits processor factory must be callable")
    pad = tokenizer.pad_token_id
    state = {
        "step": None, "turn": 0, "generation_seconds": 0.0,
        "max_new_tokens": max_new_tokens,
    }

    def set_step(step: int) -> None:
        state["step"] = step
        state["turn"] = 0

    def tokenize(messages):
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False,
        )
        return (encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"])[0]

    def generate(batch_messages):
        if state["step"] is None:
            raise RuntimeError("generation step context was not initialized")
        state["turn"] += 1
        call_seed = generation_call_seed(
            seed=training_seed, step=state["step"], turn=state["turn"]
        )
        prompts = [tokenize(messages) for messages in batch_messages]
        results = [None] * len(prompts)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(call_seed)
            torch.cuda.manual_seed(call_seed)
            for start in range(0, len(prompts), FIXED_TRAINING["gen_chunk"]):
                subset = prompts[start:start + FIXED_TRAINING["gen_chunk"]]
                max_length = max(item.shape[0] for item in subset)
                inputs = torch.full((len(subset), max_length), pad, dtype=torch.long)
                attention = torch.zeros((len(subset), max_length), dtype=torch.long)
                for offset, ids in enumerate(subset):
                    length = ids.shape[0]
                    inputs[offset, max_length - length:] = ids
                    attention[offset, max_length - length:] = 1
                generation_mode = (
                    cached_eval_generation(model)
                    if cached_eval_mode else contextlib.nullcontext()
                )
                with generation_mode, torch.no_grad():
                    generation_started = time.monotonic()
                    generate_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": True,
                        "temperature": FIXED_TRAINING["attacker_temperature"],
                        "top_p": 1.0,
                        "pad_token_id": pad,
                    }
                    if cached_eval_mode:
                        generate_kwargs["use_cache"] = True
                    if logits_processor_factory is not None:
                        generate_kwargs["logits_processor"] = LogitsProcessorList([
                            logits_processor_factory(max_length)
                        ])
                    output = model.generate(
                        inputs.to("cuda:0"), attention_mask=attention.to("cuda:0"),
                        **generate_kwargs,
                    )
                    state["generation_seconds"] += time.monotonic() - generation_started
                for offset, ids in enumerate(subset):
                    response = _trim_response(output[offset, max_length:].detach().cpu(), pad)
                    results[start + offset] = {
                        "text": tokenizer.decode(response, skip_special_tokens=True),
                        "prompt_ids": ids.detach().cpu(),
                        "resp_ids": response,
                    }
        if cached_eval_mode:
            # The co-located FP8 victim already owns its fixed vLLM allocation. Generated
            # tensors are now on CPU, so release only PyTorch's unused reservation before
            # victim calls; weights and live tensors remain allocated.
            torch.cuda.empty_cache()
        return results

    generate.set_step = set_step
    generate.metrics = state
    return generate


def token_logps(model, prompt_ids, response_ids):
    sequence = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to("cuda:0")
    logits = model(sequence).logits[0, :-1]
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    start = prompt_ids.shape[0] - 1
    target = response_ids.to("cuda:0")
    return log_probabilities[start:start + response_ids.shape[0]].gather(
        1, target.unsqueeze(1)
    ).squeeze(1)


def _lora_named_parameters(model):
    trainable = []
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if parameter.requires_grad:
            if "lora_" not in name:
                raise RuntimeError(f"non-LoRA parameter is trainable: {name}")
            trainable.append((name, parameter))
        elif parameter.grad is not None:
            raise RuntimeError(f"frozen base parameter acquired a gradient: {name}")
    if not trainable:
        raise RuntimeError("no trainable LoRA parameters found")
    return trainable


def lora_parameter_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in _lora_named_parameters(model):
        tensor = parameter.detach().contiguous().cpu()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _save_adapter_atomic(model, target: Path) -> str:
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if target.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite adapter: {target}")
    model.save_pretrained(str(temporary))
    os.replace(temporary, target)
    return tree_sha256(target)


def construct_qlora_model():
    """Construct in the required order: NF4 base -> k-bit prepare -> LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(
        ATTACKER_MODEL, revision=ATTACKER_REVISION, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=ATTACKER_BNB_QUANT_TYPE,
        bnb_4bit_use_double_quant=ATTACKER_BNB_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        ATTACKER_MODEL,
        revision=ATTACKER_REVISION,
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    if not bool(getattr(base_model, "is_loaded_in_4bit", False)):
        raise RuntimeError("attacker base model did not load in 4-bit mode")
    loaded_quantization = getattr(
        getattr(base_model, "hf_quantizer", None), "quantization_config", None
    )
    if loaded_quantization is None:
        loaded_quantization = getattr(base_model, "quantization_config", None)
    if (loaded_quantization is None
            or getattr(loaded_quantization, "load_in_4bit", None) is not True
            or getattr(loaded_quantization, "bnb_4bit_quant_type", None)
            != ATTACKER_BNB_QUANT_TYPE
            or getattr(loaded_quantization, "bnb_4bit_use_double_quant", None)
            is not ATTACKER_BNB_USE_DOUBLE_QUANT
            or getattr(loaded_quantization, "bnb_4bit_compute_dtype", None)
            != torch.bfloat16):
        raise RuntimeError(
            "loaded attacker quantization differs from NF4 double-quant/BF16"
        )
    prepared_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True
    )
    if not bool(getattr(prepared_model, "is_gradient_checkpointing", False)):
        raise RuntimeError("k-bit preparation did not enable gradient checkpointing")
    model = get_peft_model(prepared_model, LoraConfig(**LORA_CONFIG))
    model.config.use_cache = False
    model.train()
    _lora_named_parameters(model)
    return tokenizer, model


def _build_run_config(
    *, args, tag: str, gate: dict, gate_identity: dict,
    runtime_bundle: dict, runtime_identity: dict, runtime_open_check: dict,
    deployment: dict, deployment_identity: dict, benchmark: dict,
    benchmark_identity: dict, benchmark_result: dict,
    benchmark_result_identity: dict, budget_authorization: dict | None,
    budget_authorization_identity: dict | None, construction_seeds: dict,
    hardware: dict, goals: list, domain, schedule: list[list[int]], initial_lora_sha256: str,
) -> dict:
    record = {
        "schema_version": 1,
        "kind": "h1_h20_formal_training_run_config",
        "canonical_training_run": True,
        "run_kind": "formal",
        "tag": tag,
        "arm": args.arm,
        "seed": args.seed,
        **FIXED_TRAINING,
        "smoke": False,
        "benchmark": False,
        "run_root": str(Path(args.run_root).resolve()),
        "gate1_spec": str(Path(args.gate1_spec).resolve()),
        "benchmark_manifest_path": benchmark_identity["path"],
        "benchmark_result_path": benchmark_result_identity["path"],
        "budget_authorization_path": (
            None if budget_authorization_identity is None
            else budget_authorization_identity["path"]
        ),
        "runtime_profile": LEGACY_H20_PROFILE_ID,
        "runtime_profile_sha256": H20_RUNTIME_PROFILE_SHA256,
        "execution": deepcopy(SINGLE_H20_EXECUTION),
        "models": deepcopy(MODEL_IDENTITY),
        "qlora": deepcopy(QLORA_CONFIG),
        "lora": deepcopy(LORA_CONFIG),
        "data": deepcopy(DATA_IDENTITY),
        "oracle_and_interaction": deepcopy(ORACLE_AND_INTERACTION),
        "training_protocol": formal_training_protocol(),
        "training_protocol_sha256": FORMAL_TRAINING_PROTOCOL_SHA256,
        "global_goal_schedule": schedule,
        "global_goal_schedule_sha256": canonical_sha256(schedule),
        "goal_ids": [goal.id for goal in goals],
        "goal_ids_sha256": goal_ids_sha256([goal.id for goal in goals]),
        "dataset_sha256": domain.dataset_sha256,
        "split_manifest_id": domain.split_manifest["manifest_id"],
        "gate1": gate,
        "gate1_identity": gate_identity,
        "runtime": runtime_bundle,
        "runtime_identity": runtime_identity,
        "runtime_open_check": runtime_open_check,
        "deployment": deployment,
        "deployment_identity": deployment_identity,
        "benchmark_manifest": benchmark,
        "benchmark_identity": benchmark_identity,
        "benchmark_result": benchmark_result,
        "benchmark_result_identity": benchmark_result_identity,
        "budget_authorization": budget_authorization,
        "budget_authorization_identity": budget_authorization_identity,
        "construction_seeds": construction_seeds,
        "hardware": hardware,
        "software_versions": _software_versions(),
        "initial_lora_sha256": initial_lora_sha256,
    }
    return seal_payload(record)


def _checkpoint(
    model, run_dir: Path, *, step: int, lora_sha256: str, run_config: dict,
) -> None:
    adapter_path = run_dir / f"adapter_step{step}"
    adapter_tree = _save_adapter_atomic(model, adapter_path)
    manifest = build_checkpoint_manifest(
        step=step,
        adapter_tree_sha256=adapter_tree,
        lora_sha256=lora_sha256,
        run_config_file_sha256=sha256_file(run_dir / "run_config.json"),
        run_config=run_config,
    )
    validate_checkpoint_manifest(manifest, run_dir=run_dir, run_config=run_config)
    _atomic_json(run_dir / f"adapter_step{step}.manifest.json", manifest)


def _run_training_loop(
    *, model, tokenizer, domain, goals, schedule, args, run_dir: Path, steps: int,
    save_checkpoints: bool, run_config: dict | None, victim_chat_fn=None,
    victim_provider: str = "local-vllm", victim_model: str = "local-victim",
    victim_max_tokens: int | None = None, checkpoint_fn=None,
    cached_attacker_generation: bool = False,
    victim_decision_protocol_id: str | None = None,
    attacker_generator=None, shape_override: dict | None = None,
) -> tuple[str, float, int]:
    shape = {
        "n_goals": FIXED_TRAINING["n_goals"],
        "G": FIXED_TRAINING["G"],
        "T": FIXED_TRAINING["T"],
    }
    if shape_override is not None:
        if not isinstance(shape_override, dict) or set(shape_override) != set(shape):
            raise ValueError("training shape override must contain exactly n_goals/G/T")
        for key, value in shape_override.items():
            if (
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                or value > FIXED_TRAINING[key]
            ):
                raise ValueError(f"training shape override exceeds frozen ceiling: {key}")
        shape.update(shape_override)
    if len(schedule) != steps or any(len(row) != shape["n_goals"] for row in schedule):
        raise ValueError("training schedule does not match the requested shape")
    if victim_chat_fn is None:
        victim_chat_fn = victim_chat
    if checkpoint_fn is None:
        checkpoint_fn = _checkpoint
    if victim_max_tokens is None:
        victim_max_tokens = FIXED_TRAINING["victim_max_tokens"]
    victim_batch_fn = make_victim_batch_fn(
        domain, FIXED_TRAINING["workers"], victim_chat_fn,
        max_calls=FIXED_TRAINING["max_calls"],
        max_tokens=victim_max_tokens,
        temperature=FIXED_TRAINING["victim_temperature"],
        provider=victim_provider, model=victim_model,
        decision_protocol_id=victim_decision_protocol_id,
    )
    generator = attacker_generator or make_gen_batch_fn(
        model,
        tokenizer,
        training_seed=args.seed,
        cached_eval_mode=cached_attacker_generation,
    )
    optimizer_kwargs = dict(OPTIMIZER_CONFIG["kwargs"])
    optimizer_kwargs["betas"] = tuple(optimizer_kwargs["betas"])
    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in _lora_named_parameters(model)],
        **optimizer_kwargs,
    )
    progress_path = run_dir / "progress.jsonl"
    rollouts_path = run_dir / "rollouts.jsonl"
    progress = progress_path.open("x", encoding="utf-8")
    rollouts = rollouts_path.open("x", encoding="utf-8")
    final_lora_hash = lora_parameter_sha256(model)
    total_elapsed = 0.0
    rollout_rows = 0
    try:
        for step in range(1, steps + 1):
            # Budget evidence must not be understated by an NTP/manual wall-clock step.
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(0)
            prepare_step = getattr(generator, "prepare_step", None)
            if callable(prepare_step):
                prepare_step(
                    step=step,
                    lora_sha256=final_lora_hash,
                )
            else:
                generator.set_step(step)
            generation_seconds_before = generator.metrics["generation_seconds"]
            items, groups = [], []
            for slot in range(shape["n_goals"]):
                goal = goals[schedule[step - 1][slot]]
                indices = list(range(len(items), len(items) + shape["G"]))
                items.extend([goal] * shape["G"])
                groups.append((goal, indices, slot))
            rollout_started = time.monotonic()
            results = rollout_batch(
                domain, items, generator, victim_batch_fn,
                T=shape["T"], tau=FIXED_TRAINING["tau"],
            )
            rollout_seconds = time.monotonic() - rollout_started
            attacker_generation_seconds = (
                generator.metrics["generation_seconds"] - generation_seconds_before
            )
            examples = []
            phi_sum = successes = zero_advantages = advantage_count = 0
            for goal, indices, slot in groups:
                reward_rows = [
                    per_turn_rewards(results[index]["phi_trace"], 1.0, args.arm)
                    for index in indices
                ]
                advantages = group_advantages(reward_rows)
                for trajectory_index, (index, advantage_row) in enumerate(zip(indices, advantages)):
                    result = results[index]
                    phi_sum += result["max_phi"]
                    successes += int(result["success"])
                    zero_advantages += sum(abs(value) < 1e-9 for value in advantage_row)
                    advantage_count += len(advantage_row)
                    rollouts.write(json.dumps({
                        "step": step,
                        "global_group_slot": slot,
                        "trajectory_index": trajectory_index,
                        "arm": args.arm,
                        "seed": args.seed,
                        "goal": goal.id,
                        "phi_trace": result["phi_trace"],
                        "max_phi": result["max_phi"],
                        "success": result["success"],
                        "n_turns": result["n_turns"],
                        "calls": [tool for tool, _arguments in result["calls"]],
                        "turns": [{
                            "turn": turn["turn"],
                            "attacker_output": turn["response"],
                            "raw_attacker_output": turn.get(
                                "raw_model_response", turn["response"]
                            ),
                            "attacker_transport_normalization": turn.get(
                                "attacker_transport_normalization", "none"
                            ),
                            "attacker_transport": turn["attacker_transport"],
                            "injection": turn["injection"],
                            "victim_transcript": turn["victim_reply"],
                            "calls": [
                                {"tool": tool, "args": arguments}
                                for tool, arguments in turn["attempt_calls"]
                            ],
                            "oracle": turn["oracle"],
                            "attempt_phi": turn["attempt_phi"],
                            "phi": turn["phi"],
                        } for turn in result["turns"]],
                    }, sort_keys=True) + "\n")
                    rollout_rows += 1
                    for turn_index, turn in enumerate(result["turns"]):
                        advantage = (
                            advantage_row[turn_index]
                            if turn_index < len(advantage_row) else 0.0
                        )
                        prompt_ids, response_ids = turn.get("prompt_ids"), turn.get("resp_ids")
                        if (abs(advantage) >= 1e-9 and prompt_ids is not None
                                and response_ids is not None and response_ids.shape[0] > 0):
                            examples.append((prompt_ids, response_ids, advantage))
            rollouts.flush()
            backward_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            # Rollouts are CPU-resident at this point.  Release only unused
            # PyTorch reservations left by model construction before a new
            # sequence shape triggers Triton autotuning during backward.
            # Live weights, optimizer state, and the scientific batch remain
            # unchanged.
            torch.cuda.empty_cache()
            pg_value = kl_value = 0.0
            if examples:
                for prompt_ids, response_ids, advantage in examples:
                    policy_logp = token_logps(model, prompt_ids, response_ids)
                    with torch.no_grad(), model.disable_adapter():
                        reference_logp = token_logps(model, prompt_ids, response_ids)
                    ratio = reference_logp - policy_logp
                    kl = (ratio.exp() - ratio - 1.0).sum()
                    pg = -advantage * policy_logp.sum()
                    loss = (pg + FIXED_TRAINING["beta_kl"] * kl) / len(examples)
                    loss.backward()
                    pg_value += float(pg.detach()) / len(examples)
                    kl_value += float((FIXED_TRAINING["beta_kl"] * kl).detach()) / len(examples)
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad and parameter.grad is not None:
                        raise RuntimeError(f"frozen base parameter acquired a gradient: {name}")
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for _name, parameter in _lora_named_parameters(model)],
                    max_norm=GRADIENT_CLIPPING["max_norm"],
                    norm_type=GRADIENT_CLIPPING["norm_type"],
                    error_if_nonfinite=GRADIENT_CLIPPING["error_if_nonfinite"],
                    foreach=GRADIENT_CLIPPING["foreach"],
                )
                if not math.isfinite(float(grad_norm)):
                    raise RuntimeError(f"non-finite LoRA gradient norm: {grad_norm}")
                optimizer.step()
            else:
                grad_norm = torch.tensor(0.0, device="cuda:0")
            backward_seconds = time.monotonic() - backward_started
            final_lora_hash = lora_parameter_sha256(model)
            elapsed = time.monotonic() - started
            total_elapsed += elapsed
            count = len(results)
            record = {
                "step": step,
                "tag": run_dir.name,
                "arm": args.arm,
                "seed": args.seed,
                "pg_loss": round(pg_value, 6),
                "kl_loss": round(kl_value, 6),
                "grad_norm": round(float(grad_norm), 6),
                "mean_max_phi": round(phi_sum / count, 6),
                "success_rate": round(successes / count, 6),
                "global_B": count,
                "training_shape": dict(shape),
                "global_n_examples": len(examples),
                "optimizer_step": bool(examples),
                "frac_zero_grad": round(zero_advantages / max(1, advantage_count), 6),
                "step_time": round(elapsed, 3),
                "rollout_seconds": round(rollout_seconds, 6),
                "attacker_generation_seconds": round(attacker_generation_seconds, 6),
                "victim_api_and_harness_seconds": round(
                    max(0.0, rollout_seconds - attacker_generation_seconds), 6
                ),
                "backward_seconds": round(backward_seconds, 6),
                "lora_sha256": final_lora_hash,
                "goal_schedule_sha256": canonical_sha256(schedule),
                "attacker_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "attacker_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            }
            rollout_identity = getattr(generator, "current_identity", None)
            if isinstance(rollout_identity, dict):
                record["attacker_rollout_identity"] = deepcopy(rollout_identity)
            progress.write(json.dumps(record, sort_keys=True) + "\n")
            progress.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
            if (save_checkpoints and FIXED_TRAINING["ckpt_every"]
                    and step % FIXED_TRAINING["ckpt_every"] == 0 and step < steps):
                if run_config is None:
                    raise RuntimeError("formal checkpoint lacks its run config")
                checkpoint_fn(
                    model, run_dir, step=step,
                    lora_sha256=final_lora_hash, run_config=run_config,
                )
    finally:
        progress.close()
        rollouts.close()
    return final_lora_hash, total_elapsed, rollout_rows


def _checkpoint_registry(run_dir: Path, run_config: dict) -> list[dict]:
    registry = []
    for step in range(
        FIXED_TRAINING["ckpt_every"], FIXED_TRAINING["steps"], FIXED_TRAINING["ckpt_every"]
    ):
        manifest_path = run_dir / f"adapter_step{step}.manifest.json"
        manifest = validate_checkpoint_manifest(
            _read_json(manifest_path, f"checkpoint {step} manifest"),
            run_dir=run_dir, run_config=run_config,
        )
        registry.append({
            "step": step,
            "adapter_path": manifest["adapter_path"],
            "adapter_tree_sha256": manifest["adapter_tree_sha256"],
            "lora_sha256": manifest["lora_sha256"],
            "manifest_path": manifest_path.name,
            "manifest_file_sha256": sha256_file(manifest_path),
            "manifest_payload_sha256": manifest["payload_sha256"],
        })
    return registry


def _execute(args) -> Path:
    _load_powered_stack()
    _validate_environment_overrides()
    run_root = Path(args.run_root).resolve()
    gate_path = Path(args.gate1_spec).resolve()
    (
        gate, runtime_bundle, deployment,
        gate_identity, runtime_identity, deployment_identity, cycle_path,
    ) = _validate_gate_deployment_runtime(gate_path, run_root)
    hardware = validate_single_h20_process(runtime_bundle)
    snapshot_download(ATTACKER_MODEL, revision=ATTACKER_REVISION, local_files_only=True)
    _probe_victim()

    domain = ToolUseInjectionDomain(attack="ds", defense_tier=gate["chosen_tier"])
    goals = domain.load_goals("train", seed=FIXED_TRAINING["goal_seed"], n=322)
    if (len(goals) != DATA_IDENTITY["train_count"]
            or domain.dataset_sha256 != DATA_IDENTITY["dataset_sha256"]
            or domain.split_manifest.get("manifest_id") != DATA_IDENTITY["split_manifest_id"]):
        raise RuntimeError("H20 training dataset/split identity mismatch")
    if max(len(goal.meta.get("target_tools", [])) for goal in goals) > 3:
        raise RuntimeError("frozen max_calls cannot execute the target chain")

    benchmark_mode = bool(args.benchmark)
    if benchmark_mode and (args.arm != "dense" or args.seed != 0):
        raise RuntimeError("canonical full-shape benchmark requires --arm dense --seed 0")
    if not benchmark_mode:
        validate_formal_training_values({
            **FIXED_TRAINING, "arm": args.arm, "seed": args.seed,
            "smoke": False, "benchmark": False,
        })
    schedule = build_goal_schedule(
        seed=args.seed,
        steps=1 if benchmark_mode else FIXED_TRAINING["steps"],
        n_goals=FIXED_TRAINING["n_goals"], n_train=FIXED_TRAINING["n_train"],
    )

    formal_inputs = None
    if not benchmark_mode:
        formal_inputs = load_and_validate_formal_training_inputs(
            benchmark_manifest_path=args.benchmark_manifest,
            benchmark_result_path=args.benchmark_result,
            budget_authorization_path=args.budget_authorization,
            runtime_bundle=runtime_bundle,
            gate1_identity=gate_identity,
            runtime_identity=runtime_identity,
            deployment_identity=deployment_identity,
            arm=args.arm,
            seed=args.seed,
        )
    runtime_open_check = verify_cycle_runtime_live(
        cycle_path, phase="benchmark_open" if benchmark_mode else "train_open"
    )
    validate_live_runtime_check(
        runtime_open_check, runtime_bundle["restored_fp8_runtime"],
        expected_phase="benchmark_open" if benchmark_mode else "train_open",
    )
    construction_seeds = seed_before_model_construction(args.seed)
    tokenizer, model = construct_qlora_model()
    initial_lora_hash = lora_parameter_sha256(model)

    if benchmark_mode:
        tag = args.tag or "h20-fullshape-dense-s0"
        run_dir = run_root / "benchmarks" / tag
        if run_dir.exists():
            raise FileExistsError(f"benchmark directory already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        benchmark_config = seal_payload({
            "schema_version": 1,
            "kind": BENCHMARK_RUN_KIND,
            "decision_bearing": False,
            "profile_id": LEGACY_H20_PROFILE_ID,
            "profile_sha256": H20_RUNTIME_PROFILE_SHA256,
            "training_protocol_sha256": FORMAL_TRAINING_PROTOCOL_SHA256,
            "shape": {"steps": 1, **{
                key: FIXED_TRAINING[key]
                for key in ("n_goals", "G", "T", "max_calls", "gen_chunk", "workers")
            }},
            "arm": "dense",
            "seed": 0,
            "gate1_identity": gate_identity,
            "runtime_identity": runtime_identity,
            "runtime_open_check": runtime_open_check,
            "deployment_identity": deployment_identity,
            "hardware": hardware,
            "construction_seeds": construction_seeds,
            "created_at": _now(),
        })
        _atomic_json(run_dir / "benchmark_config.json", benchmark_config)
        _final_lora, elapsed, _rows = _run_training_loop(
            model=model, tokenizer=tokenizer, domain=domain, goals=goals,
            schedule=schedule, args=args, run_dir=run_dir, steps=1,
            save_checkpoints=False, run_config=None,
        )
        close_check = verify_cycle_runtime_live(cycle_path, phase="benchmark_close")
        validate_live_runtime_check(
            close_check, runtime_bundle["restored_fp8_runtime"],
            expected_phase="benchmark_close",
        )
        serialized_adapter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for _name, parameter in _lora_named_parameters(model)
        )
        manifest = build_benchmark_manifest(
            gate1_identity=gate_identity,
            runtime_identity=runtime_identity,
            deployment_identity=deployment_identity,
            step_seconds=elapsed,
            benchmark_artifact_bytes=_directory_bytes(run_dir),
            serialized_adapter_bytes=serialized_adapter_bytes,
        )
        wrapper = build_benchmark_result(
            manifest,
            runtime_bundle=runtime_bundle,
            runtime_open_check=runtime_open_check,
            runtime_close_check=close_check,
        )
        _atomic_json(run_dir / "benchmark_manifest.json", manifest)
        _atomic_json(run_dir / "benchmark_result.json", wrapper)
        print(json.dumps({
            "benchmark_manifest": str(run_dir / "benchmark_manifest.json"),
            "status": manifest["status"],
            "projected_single_run_gpu_hours": manifest["measurement"][
                "projected_single_run_gpu_hours"
            ],
            "projected_training_campaign_gpu_hours": manifest["measurement"][
                "projected_training_campaign_gpu_hours"
            ],
            "projected_single_run_artifact_bytes": manifest["measurement"][
                "projected_single_run_artifact_bytes"
            ],
            "projected_training_campaign_artifact_bytes": manifest["measurement"][
                "projected_training_campaign_artifact_bytes"
            ],
        }, sort_keys=True))
        return run_dir

    # Re-read both evidence files from the exact bytes whose hashes enter the
    # sealed run config.  This detects any change during model construction.
    checked_inputs = load_and_validate_formal_training_inputs(
        benchmark_manifest_path=args.benchmark_manifest,
        benchmark_result_path=args.benchmark_result,
        budget_authorization_path=args.budget_authorization,
        runtime_bundle=runtime_bundle,
        gate1_identity=gate_identity,
        runtime_identity=runtime_identity,
        deployment_identity=deployment_identity,
        arm=args.arm,
        seed=args.seed,
    )
    if checked_inputs != formal_inputs:
        raise RuntimeError("formal benchmark/budget evidence changed during model construction")
    benchmark = checked_inputs["benchmark_manifest"]
    tag = args.tag or f"h20-tooluse-{args.arm}-s{args.seed}"
    run_dir = run_root / "runs" / tag
    if run_dir.exists():
        raise FileExistsError(f"formal run directory already exists: {run_dir}")
    run_config = _build_run_config(
        args=args, tag=tag, gate=gate, gate_identity=gate_identity,
        runtime_bundle=runtime_bundle, runtime_identity=runtime_identity,
        runtime_open_check=runtime_open_check, deployment=deployment,
        deployment_identity=deployment_identity, benchmark=benchmark,
        benchmark_identity=checked_inputs["benchmark_identity"],
        benchmark_result=checked_inputs["benchmark_result"],
        benchmark_result_identity=checked_inputs["benchmark_result_identity"],
        budget_authorization=checked_inputs["budget_authorization"],
        budget_authorization_identity=checked_inputs["budget_authorization_identity"],
        construction_seeds=construction_seeds,
        hardware=hardware, goals=goals, domain=domain, schedule=schedule,
        initial_lora_sha256=initial_lora_hash,
    )
    validate_run_config(run_config)
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(run_dir / "run_config.json", run_config)
    final_lora_hash, _elapsed, rollout_rows = _run_training_loop(
        model=model, tokenizer=tokenizer, domain=domain, goals=goals,
        schedule=schedule, args=args, run_dir=run_dir, steps=60,
        save_checkpoints=True, run_config=run_config,
    )
    runtime_close_check = verify_cycle_runtime_live(cycle_path, phase="train_close")
    validate_live_runtime_check(
        runtime_close_check, runtime_bundle["restored_fp8_runtime"],
        expected_phase="train_close",
    )
    adapter_tree = _save_adapter_atomic(model, run_dir / "adapter")
    checkpoints = _checkpoint_registry(run_dir, run_config)
    artifact_manifest = build_artifact_manifest(
        run_config=run_config,
        run_config_file_sha256=sha256_file(run_dir / "run_config.json"),
        adapter_tree_sha256=adapter_tree,
        final_lora_sha256=final_lora_hash,
        progress_sha256=sha256_file(run_dir / "progress.jsonl"),
        progress_rows=FIXED_TRAINING["steps"],
        rollouts_sha256=sha256_file(run_dir / "rollouts.jsonl"),
        rollout_rows=rollout_rows,
        checkpoints=checkpoints,
        runtime_close_check=runtime_close_check,
    )
    validate_artifact_manifest(
        artifact_manifest, run_dir=run_dir, run_config=run_config
    )
    _atomic_json(run_dir / "artifact_manifest.json", artifact_manifest)
    print(json.dumps({
        "status": "complete",
        "run_dir": str(run_dir),
        "artifact_manifest_payload_sha256": artifact_manifest["payload_sha256"],
        "runtime": canonical_provenance_summary(runtime_bundle),
    }, sort_keys=True))
    return run_dir


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=(LEGACY_H20_PROFILE_ID,))
    parser.add_argument("--arm", required=True, choices=("dense", "sparse"))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--gate1-spec", required=True)
    parser.add_argument("--benchmark-manifest")
    parser.add_argument("--benchmark-result")
    parser.add_argument("--budget-authorization")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--run-root", default="/root/autodl-tmp/h1mt")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    if args.profile != LEGACY_H20_PROFILE_ID:
        parser.error("formal trainer accepts only the explicit single-H20 profile")
    supplied_evidence = (
        args.benchmark_manifest, args.benchmark_result, args.budget_authorization,
    )
    if args.benchmark and any(value is not None for value in supplied_evidence):
        parser.error("--benchmark rejects formal benchmark/budget evidence arguments")
    if not args.benchmark and (
        args.benchmark_manifest is None or args.benchmark_result is None
    ):
        parser.error(
            "formal training requires both --benchmark-manifest and --benchmark-result"
        )
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    _execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
