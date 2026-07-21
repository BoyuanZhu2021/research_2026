"""Audited local-vLLM boundary for the fast H20 H1 profile.

The local victim is still an external process from the trainer's point of view.  This
client therefore records the credential-free request and the exact HTTP response bytes
before parsing, just like the SiliconFlow boundary.  Malformed HTTP-200 responses fail
closed except for the explicitly registered, semantic-preserving final-answer C0
transport; no response is retried or converted into a zero-reward sample.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .generation_runtime import cached_eval_generation_identity
from .h20_serving_identity import ENDPOINT
from .local_victim_decision_protocol import LOCAL_COMPACT_VICTIM_DECISION_PROTOCOL
from .model_pins import VICTIM_H20_SERVED_NAME, VICTIM_REVISION
from .tooluse_gate1_spec import (
    VICTIM_INCLUDE_STOP_STR_IN_OUTPUT,
    VICTIM_MAX_TOKENS,
    VICTIM_REACT_STOP,
)
from .victim_decision_protocol import canonical_sha256


LOCAL_FAST_PROFILE_ID = "h20-attacker-cached-nf4-local-fp8-victim-final-c0-v1"
LOCAL_VICTIM_PROVIDER = "local-vllm"
LOCAL_VICTIM_MODEL = VICTIM_H20_SERVED_NAME
LOCAL_VICTIM_URL = ENDPOINT + "/chat/completions"
LOCAL_VICTIM_TEMPERATURE = 0.0
LOCAL_VICTIM_TIMEOUT_SECONDS = 120.0
LOCAL_LEDGER_KIND = "h1_local_vllm_victim_ledger_event"
FINAL_C0_TRANSPORT_ID = "h1-victim-final-c0-canonicalization-v1"
FINAL_C0_TRANSPORT_POLICY = {
    "schema_version": 1,
    "transport_id": FINAL_C0_TRANSPORT_ID,
    "activation": "strict JSONDecodeError with Invalid control character only",
    "accepted_shape": {
        "kind": "final",
        "exact_keys": ["answer", "kind"],
        "answer_type": "string",
    },
    "action_decisions": "fail_closed",
    "canonicalization": (
        "json.dumps(parsed_strict_false, ensure_ascii=False, sort_keys=True, "
        "separators=comma_colon)"
    ),
    "raw_response_logged_before_parse": True,
    "retry": False,
    "semantic_invariant": (
        "json.loads(canonical)==json.loads(raw_content,strict=False)"
    ),
}
FINAL_C0_TRANSPORT_POLICY_SHA256 = canonical_sha256(FINAL_C0_TRANSPORT_POLICY)
JSON_CONTROL_ESCAPE_BAD_WORDS = (
    r"\b", r"\f", r"\r", r"\t",
    *(f"\\u{codepoint:04x}" for codepoint in range(32) if codepoint != 10),
    r"\u007f",
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def local_fast_profile() -> dict:
    result = {
        "schema_version": 1,
        "profile_id": LOCAL_FAST_PROFILE_ID,
        "attacker_runtime": {
            "host": "single-h20",
            "quantization": "bitsandbytes-nf4-double-quant-bfloat16",
            "generation": cached_eval_generation_identity(),
        },
        "victim_runtime": {
            "host": "same-single-h20-separate-vllm-process",
            "provider": LOCAL_VICTIM_PROVIDER,
            "model": LOCAL_VICTIM_MODEL,
            "revision": VICTIM_REVISION,
            "endpoint": LOCAL_VICTIM_URL,
            "quantization": "fp8",
            "temperature": LOCAL_VICTIM_TEMPERATURE,
            "max_tokens": VICTIM_MAX_TOKENS,
            "enable_thinking": False,
            "structured_outputs": "xgrammar-json-schema",
            "http_200_malformed_retry": False,
            "prefix_repair": False,
            "raw_response_logged_before_parse": True,
            "final_c0_transport": copy.deepcopy(FINAL_C0_TRANSPORT_POLICY),
            "final_c0_transport_payload_sha256": (
                FINAL_C0_TRANSPORT_POLICY_SHA256
            ),
        },
        "victim_decision_protocol": copy.deepcopy(
            LOCAL_COMPACT_VICTIM_DECISION_PROTOCOL
        ),
    }
    result["payload_sha256"] = canonical_sha256(result)
    return result


LOCAL_FAST_PROFILE = local_fast_profile()


class LocalVllmVictimError(RuntimeError):
    """The exact local-vLLM boundary failed closed."""


def _json_string_control_positions(content: str) -> list[dict[str, Any]]:
    """Locate literal C0 controls inside JSON strings without changing the bytes."""
    result: list[dict[str, Any]] = []
    in_string = False
    escaped = False
    for position, character in enumerate(content):
        if not in_string:
            if character == '"':
                in_string = True
            continue
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            in_string = False
        elif ord(character) < 32:
            result.append({
                "position": position,
                "codepoint": f"U+{ord(character):04X}",
            })
    return result


def _canonicalize_final_c0_content(
    content: str,
    error: json.JSONDecodeError,
    *,
    final_answer_max_length: int | None,
) -> tuple[str, dict[str, Any]]:
    """Canonicalize only the approved final-answer C0 transport failure."""
    if error.msg != "Invalid control character at":
        raise error
    controls = _json_string_control_positions(content)
    if not controls or error.pos not in {item["position"] for item in controls}:
        raise error

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    parsed = json.loads(content, strict=False, parse_constant=reject_constant)
    if (
        not isinstance(parsed, dict)
        or set(parsed) != {"answer", "kind"}
        or parsed.get("kind") != "final"
        or not isinstance(parsed.get("answer"), str)
        or (
            final_answer_max_length is not None
            and len(parsed["answer"]) > final_answer_max_length
        )
    ):
        raise ValueError("control-character fallback is not an exact final decision")
    canonical = json.dumps(
        parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if json.loads(canonical, parse_constant=reject_constant) != parsed:
        raise ValueError("control-character canonicalization changed semantics")
    return canonical, {
        "transport_id": FINAL_C0_TRANSPORT_ID,
        "policy_payload_sha256": FINAL_C0_TRANSPORT_POLICY_SHA256,
        "applied": True,
        "strict_error": error.msg,
        "strict_error_position": error.pos,
        "controls": controls,
        "original_content_sha256": _sha256_bytes(content.encode("utf-8")),
        "canonical_content_sha256": _sha256_bytes(canonical.encode("utf-8")),
        "semantic_object_sha256": canonical_sha256(parsed),
    }


class LocalVllmVictimClient:
    """Thread-safe chat callback with an append-only raw-response ledger."""

    supports_audit_context = True

    def __init__(
        self,
        ledger_path: str | Path,
        *,
        timeout: float = LOCAL_VICTIM_TIMEOUT_SECONDS,
        endpoint: str = LOCAL_VICTIM_URL,
        model: str = LOCAL_VICTIM_MODEL,
        provider: str = LOCAL_VICTIM_PROVIDER,
        max_tokens: int = VICTIM_MAX_TOKENS,
        ledger_kind: str = LOCAL_LEDGER_KIND,
        final_answer_max_length: int | None = None,
        action_string_max_length: int | None = None,
        strict_declared_action_arguments: bool = False,
        final_c0_transport_id: str | None = None,
    ) -> None:
        self._timeout = float(timeout)
        if self._timeout <= 0:
            raise ValueError("local victim timeout must be positive")
        if not endpoint.startswith("http://127.0.0.1:") or not endpoint.endswith(
            "/v1/chat/completions"
        ):
            raise ValueError("local victim endpoint must be a localhost v1 chat endpoint")
        if not all(isinstance(value, str) and value for value in (model, provider, ledger_kind)):
            raise ValueError("local victim identity strings must be non-empty")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("local victim max_tokens must be positive")
        if (
            final_answer_max_length is not None
            and (
                not isinstance(final_answer_max_length, int)
                or isinstance(final_answer_max_length, bool)
                or final_answer_max_length < 1
            )
        ):
            raise ValueError("local victim final answer bound must be positive")
        if (
            action_string_max_length is not None
            and (
                not isinstance(action_string_max_length, int)
                or isinstance(action_string_max_length, bool)
                or action_string_max_length < 1
            )
        ):
            raise ValueError("local victim action string bound must be positive")
        if not isinstance(strict_declared_action_arguments, bool):
            raise ValueError("local victim strict argument policy must be boolean")
        if final_c0_transport_id not in (None, FINAL_C0_TRANSPORT_ID):
            raise ValueError("unknown local victim final C0 transport")
        self._endpoint = endpoint
        self._model = model
        self._provider = provider
        self._max_tokens = max_tokens
        self._ledger_kind = ledger_kind
        self._final_answer_max_length = final_answer_max_length
        self._action_string_max_length = action_string_max_length
        self._strict_declared_action_arguments = strict_declared_action_arguments
        self._final_c0_transport_id = final_c0_transport_id
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("x", encoding="utf-8"):
            pass
        self._lock = threading.Lock()

    def _append(self, event: Mapping[str, Any]) -> None:
        row = {
            "schema_version": 1,
            "kind": self._ledger_kind,
            "recorded_at_unix_ns": time.time_ns(),
            **copy.deepcopy(dict(event)),
        }
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with self.ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())

    def __call__(
        self,
        messages: list[dict],
        *,
        max_tokens: int = VICTIM_MAX_TOKENS,
        temperature: float = LOCAL_VICTIM_TEMPERATURE,
        enable_thinking: bool | None = False,
        seed: int | None = None,
        retries: int = 1,
        structured_outputs: dict | None = None,
        role: str | None = None,
        context: dict | None = None,
    ) -> dict:
        del seed
        if max_tokens != self._max_tokens:
            raise ValueError("local victim max_tokens drift")
        if float(temperature) != LOCAL_VICTIM_TEMPERATURE:
            raise ValueError("local victim temperature drift")
        if enable_thinking not in (None, False):
            raise ValueError("local victim thinking setting drift")
        if retries != 1:
            raise ValueError("local victim does not retry formal HTTP calls")
        if not isinstance(structured_outputs, Mapping):
            raise ValueError("local victim requires structured_outputs")
        schema = structured_outputs.get("json")
        if not isinstance(schema, Mapping):
            raise ValueError("local victim requires structured_outputs.json")
        schema = copy.deepcopy(dict(schema))
        if self._final_answer_max_length is not None:
            branches = schema.get("oneOf")
            if not isinstance(branches, list):
                raise ValueError("local victim decision schema lacks oneOf")
            matches = []
            for branch in branches:
                properties = branch.get("properties") if isinstance(branch, Mapping) else None
                kind = properties.get("kind") if isinstance(properties, Mapping) else None
                answer = properties.get("answer") if isinstance(properties, Mapping) else None
                if isinstance(kind, Mapping) and kind.get("const") == "final":
                    if not isinstance(answer, Mapping) or answer.get("type") != "string":
                        raise ValueError("local victim final answer schema is malformed")
                    matches.append(answer)
            if len(matches) != 1:
                raise ValueError("local victim decision schema has ambiguous final branch")
            matches[0]["maxLength"] = self._final_answer_max_length
        if self._action_string_max_length is not None:
            action_branches = []
            for branch in schema.get("oneOf") or []:
                properties = branch.get("properties") if isinstance(branch, Mapping) else None
                kind = properties.get("kind") if isinstance(properties, Mapping) else None
                arguments = properties.get("arguments") if isinstance(properties, Mapping) else None
                if isinstance(kind, Mapping) and kind.get("const") == "action":
                    if not isinstance(arguments, Mapping):
                        raise ValueError("local victim action arguments schema is malformed")
                    action_branches.append(arguments)
            if not action_branches:
                raise ValueError("local victim decision schema has no action branch")

            def bound_strings(node: Any) -> None:
                if not isinstance(node, dict):
                    return
                if node.get("type") == "string":
                    node["maxLength"] = self._action_string_max_length
                for key in ("oneOf", "anyOf"):
                    for child in node.get(key) or []:
                        bound_strings(child)
                properties = node.get("properties")
                if isinstance(properties, Mapping):
                    for child in properties.values():
                        bound_strings(child)

            for arguments in action_branches:
                bound_strings(arguments)

        if self._strict_declared_action_arguments:
            action_arguments = []
            for branch in schema.get("oneOf") or []:
                properties = branch.get("properties") if isinstance(branch, Mapping) else None
                kind = properties.get("kind") if isinstance(properties, Mapping) else None
                arguments = properties.get("arguments") if isinstance(properties, Mapping) else None
                if isinstance(kind, Mapping) and kind.get("const") == "action":
                    if (
                        not isinstance(arguments, dict)
                        or arguments.get("type") != "object"
                        or not isinstance(arguments.get("properties"), Mapping)
                    ):
                        raise ValueError("local victim action arguments schema is malformed")
                    action_arguments.append(arguments)
            if not action_arguments:
                raise ValueError("local victim decision schema has no action branch")
            for arguments in action_arguments:
                arguments["additionalProperties"] = False

        call_id = uuid.uuid4().hex
        body = {
            "model": self._model,
            "messages": copy.deepcopy(messages),
            "max_tokens": self._max_tokens,
            "temperature": LOCAL_VICTIM_TEMPERATURE,
            "stop": list(VICTIM_REACT_STOP),
            "include_stop_str_in_output": VICTIM_INCLUDE_STOP_STR_IN_OUTPUT,
            "structured_outputs": {"json": copy.deepcopy(schema)},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self._strict_declared_action_arguments:
            body["bad_words"] = list(JSON_CONTROL_ESCAPE_BAD_WORDS)
        body_bytes = _json_bytes(body)
        request_sha256 = _sha256_bytes(body_bytes)
        schema_sha256 = canonical_sha256(schema)
        request_event = {
            "event": "request",
            "call_id": call_id,
            "role": role or "victim",
            "context": copy.deepcopy(context or {}),
            "provider": self._provider,
            "model": self._model,
            "request_sha256": request_sha256,
            "schema_sha256": schema_sha256,
            "request_body": copy.deepcopy(body),
        }
        if self._final_c0_transport_id is not None:
            request_event.update({
                "final_c0_transport_id": self._final_c0_transport_id,
                "final_c0_transport_policy_sha256": (
                    FINAL_C0_TRANSPORT_POLICY_SHA256
                ),
            })
        self._append(request_event)

        started = time.monotonic()
        status: int | None = None
        raw = ""
        try:
            request_value = urllib.request.Request(
                self._endpoint,
                data=body_bytes,
                method="POST",
                headers={
                    "Authorization": "Bearer EMPTY",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(
                request_value, timeout=self._timeout
            ) as response:
                status = int(response.status)
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            raw = exc.read().decode("utf-8", "replace")
            self._append({
                "event": "response_raw",
                "call_id": call_id,
                "http_status": status,
                "latency_seconds": time.monotonic() - started,
                "raw_response": raw,
                "raw_response_sha256": _sha256_bytes(raw.encode("utf-8")),
            })
            raise LocalVllmVictimError(f"local victim HTTP {status}") from exc
        except Exception as exc:  # noqa: BLE001 - preserve the actual transport failure
            self._append({
                "event": "transport_error",
                "call_id": call_id,
                "latency_seconds": time.monotonic() - started,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            })
            raise LocalVllmVictimError(
                f"local victim transport error: {type(exc).__name__}: {exc}"
            ) from exc

        latency = time.monotonic() - started
        self._append({
            "event": "response_raw",
            "call_id": call_id,
            "http_status": status,
            "latency_seconds": latency,
            "raw_response": raw,
            "raw_response_sha256": _sha256_bytes(raw.encode("utf-8")),
        })
        content: Any = None
        returned_model: Any = None
        finish_reason: Any = None
        usage: dict[str, Any] = {}
        transport: dict[str, Any] | None = None
        try:
            if status != 200:
                raise ValueError(f"unexpected HTTP status {status}")
            payload = json.loads(raw)
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content")
            returned_model = payload.get("model")
            finish_reason = choice.get("finish_reason")
            usage = copy.deepcopy(payload.get("usage") or {})
            if returned_model != self._model:
                raise ValueError(f"returned model mismatch: {returned_model!r}")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("response content is empty or non-text")
            if finish_reason == "length":
                raise ValueError("local victim exhausted max_tokens")
            if any(marker in content for marker in VICTIM_REACT_STOP):
                raise ValueError("local victim retained the Observation stop marker")
            try:
                json.loads(content)
            except json.JSONDecodeError as strict_error:
                if self._final_c0_transport_id is None:
                    raise
                original_content = content
                content, transport = _canonicalize_final_c0_content(
                    original_content,
                    strict_error,
                    final_answer_max_length=self._final_answer_max_length,
                )
            if self._final_c0_transport_id is not None and transport is None:
                transport = {
                    "transport_id": FINAL_C0_TRANSPORT_ID,
                    "policy_payload_sha256": FINAL_C0_TRANSPORT_POLICY_SHA256,
                    "applied": False,
                    "original_content_sha256": _sha256_bytes(
                        content.encode("utf-8")
                    ),
                    "canonical_content_sha256": _sha256_bytes(
                        content.encode("utf-8")
                    ),
                }
        except Exception as exc:  # noqa: BLE001 - all unregistered malformed cases fail closed
            self._append({
                "event": "response",
                "call_id": call_id,
                "http_status": status,
                "latency_seconds": latency,
                "returned_model": returned_model,
                "finish_reason": finish_reason,
                "usage": usage,
                "accepted": False,
                "content_sha256": (
                    _sha256_bytes(content.encode("utf-8"))
                    if isinstance(content, str) else None
                ),
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            })
            raise LocalVllmVictimError(
                f"local victim HTTP 200 response is invalid: {type(exc).__name__}: {exc}"
            ) from exc

        accepted_event = {
            "event": "response",
            "call_id": call_id,
            "http_status": status,
            "latency_seconds": latency,
            "returned_model": returned_model,
            "finish_reason": finish_reason,
            "usage": usage,
            "content": content,
            "content_sha256": _sha256_bytes(content.encode("utf-8")),
            "accepted": True,
        }
        if transport is not None:
            accepted_event["final_c0_transport"] = copy.deepcopy(transport)
        self._append(accepted_event)
        return {
            "content": content,
            "reasoning": message.get("reasoning_content") or "",
            "usage": usage,
            "latency": latency,
            "finish_reason": finish_reason,
            "model": returned_model,
            "provider": self._provider,
            "_audit_call_id": call_id,
            "_audit_request_sha256": request_sha256,
            "_audit_schema_sha256": schema_sha256,
        }

    def record_parse_outcome(
        self,
        response: Mapping[str, Any],
        *,
        status: str,
        error: Exception | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        call_id = response.get("_audit_call_id")
        if not isinstance(call_id, str) or not call_id:
            return
        if status not in {"canonical_valid", "canonical_invalid"}:
            raise ValueError("unknown local victim parse status")
        self._append({
            "event": "parse_outcome",
            "call_id": call_id,
            "status": status,
            "context": copy.deepcopy(dict(context or {})),
            "error_type": None if error is None else type(error).__name__,
            "error": None if error is None else str(error)[:500],
        })


def load_local_vllm_ledger(
    path: str | Path,
    *,
    require_complete: bool,
    expected_model: str = LOCAL_VICTIM_MODEL,
    expected_max_tokens: int = VICTIM_MAX_TOKENS,
    expected_ledger_kind: str = LOCAL_LEDGER_KIND,
    expected_final_c0_transport_id: str | None = None,
) -> dict:
    artifact = Path(path)
    rows: list[dict] = []
    for line_number, line in enumerate(
        artifact.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"local victim ledger line {line_number} is invalid") from exc
        if not isinstance(row, dict) or row.get("kind") != expected_ledger_kind:
            raise ValueError(f"local victim ledger line {line_number} kind mismatch")
        rows.append(row)
    calls: dict[str, list[dict]] = {}
    for row in rows:
        call_id = row.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("local victim ledger call_id is missing")
        calls.setdefault(call_id, []).append(row)
    if expected_final_c0_transport_id not in (None, FINAL_C0_TRANSPORT_ID):
        raise ValueError("unknown expected final C0 transport")
    accepted = canonical_valid = canonicalized = 0
    for call_id, events in calls.items():
        requests = [row for row in events if row.get("event") == "request"]
        raw_rows = [row for row in events if row.get("event") == "response_raw"]
        responses = [row for row in events if row.get("event") == "response"]
        outcomes = [row for row in events if row.get("event") == "parse_outcome"]
        if len(requests) != 1:
            raise ValueError(f"local victim call {call_id} request count mismatch")
        request_body = requests[0].get("request_body")
        if (
            not isinstance(request_body, Mapping)
            or _sha256_bytes(_json_bytes(request_body))
            != requests[0].get("request_sha256")
            or request_body.get("model") != expected_model
            or request_body.get("max_tokens") != expected_max_tokens
        ):
            raise ValueError(f"local victim call {call_id} request identity mismatch")
        if requests[0].get("final_c0_transport_id") != expected_final_c0_transport_id:
            raise ValueError(f"local victim call {call_id} transport identity mismatch")
        if expected_final_c0_transport_id is not None and (
            requests[0].get("final_c0_transport_policy_sha256")
            != FINAL_C0_TRANSPORT_POLICY_SHA256
        ):
            raise ValueError(f"local victim call {call_id} transport policy mismatch")
        accepted_rows = [row for row in responses if row.get("accepted") is True]
        if len(accepted_rows) == 1:
            accepted += 1
        if any(row.get("status") == "canonical_valid" for row in outcomes):
            canonical_valid += 1
        for row in raw_rows:
            raw_value = row.get("raw_response")
            if (
                not isinstance(raw_value, str)
                or not _is_sha256(row.get("raw_response_sha256"))
                or _sha256_bytes(raw_value.encode("utf-8"))
                != row.get("raw_response_sha256")
            ):
                raise ValueError(f"local victim call {call_id} raw response mismatch")
        if len(accepted_rows) == 1:
            response = accepted_rows[0]
            content = response.get("content")
            if (
                not isinstance(content, str)
                or _sha256_bytes(content.encode("utf-8"))
                != response.get("content_sha256")
            ):
                raise ValueError(f"local victim call {call_id} content mismatch")
            if expected_final_c0_transport_id is not None:
                transport = response.get("final_c0_transport")
                if (
                    not isinstance(transport, Mapping)
                    or transport.get("transport_id") != FINAL_C0_TRANSPORT_ID
                    or transport.get("policy_payload_sha256")
                    != FINAL_C0_TRANSPORT_POLICY_SHA256
                    or transport.get("canonical_content_sha256")
                    != response.get("content_sha256")
                ):
                    raise ValueError(
                        f"local victim call {call_id} transport evidence mismatch"
                    )
                if len(raw_rows) != 1:
                    raise ValueError(
                        f"local victim call {call_id} transport raw count mismatch"
                    )
                raw_payload = json.loads(raw_rows[0]["raw_response"])
                original_content = raw_payload["choices"][0]["message"]["content"]
                if (
                    not isinstance(original_content, str)
                    or _sha256_bytes(original_content.encode("utf-8"))
                    != transport.get("original_content_sha256")
                ):
                    raise ValueError(
                        f"local victim call {call_id} original content mismatch"
                    )
                if transport.get("applied") is True:
                    try:
                        json.loads(original_content)
                    except json.JSONDecodeError as strict_error:
                        expected_content, expected_transport = (
                            _canonicalize_final_c0_content(
                                original_content,
                                strict_error,
                                final_answer_max_length=None,
                            )
                        )
                    else:
                        raise ValueError(
                            f"local victim call {call_id} unnecessary canonicalization"
                        )
                    if content != expected_content or dict(transport) != expected_transport:
                        raise ValueError(
                            f"local victim call {call_id} canonicalization drift"
                        )
                    canonicalized += 1
                elif transport.get("applied") is False:
                    if content != original_content:
                        raise ValueError(
                            f"local victim call {call_id} clean content drift"
                        )
                else:
                    raise ValueError(
                        f"local victim call {call_id} transport applied flag mismatch"
                    )
        if require_complete and (
            len(raw_rows) != 1
            or len(accepted_rows) != 1
            or len(outcomes) != 1
            or outcomes[0].get("status") != "canonical_valid"
        ):
            raise ValueError(f"local victim call {call_id} is incomplete")
    return {
        "file": artifact.name,
        "file_sha256": _sha256_bytes(artifact.read_bytes()),
        "logical_calls": len(calls),
        "accepted_responses": accepted,
        "canonical_valid": canonical_valid,
        "final_c0_transport_id": expected_final_c0_transport_id,
        "final_c0_transport_policy_sha256": (
            FINAL_C0_TRANSPORT_POLICY_SHA256
            if expected_final_c0_transport_id is not None else None
        ),
        "final_c0_canonicalized": canonicalized,
        "complete": bool(require_complete and canonical_valid == len(calls)),
    }


__all__ = [
    "LOCAL_FAST_PROFILE",
    "LOCAL_FAST_PROFILE_ID",
    "FINAL_C0_TRANSPORT_ID",
    "FINAL_C0_TRANSPORT_POLICY",
    "FINAL_C0_TRANSPORT_POLICY_SHA256",
    "LOCAL_VICTIM_MODEL",
    "LOCAL_VICTIM_PROVIDER",
    "LocalVllmVictimClient",
    "LocalVllmVictimError",
    "load_local_vllm_ledger",
    "local_fast_profile",
]
