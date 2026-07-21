from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from .local_vllm_victim import (
    FINAL_C0_TRANSPORT_ID,
    FINAL_C0_TRANSPORT_POLICY_SHA256,
    JSON_CONTROL_ESCAPE_BAD_WORDS,
    LOCAL_FAST_PROFILE,
    LOCAL_FAST_PROFILE_ID,
    LOCAL_VICTIM_MODEL,
    LocalVllmVictimClient,
    LocalVllmVictimError,
    load_local_vllm_ledger,
)


class LocalVllmVictimTest(unittest.TestCase):
    def _response(self, content: str, *, finish_reason: str = "stop") -> MagicMock:
        response = MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({
            "model": LOCAL_VICTIM_MODEL,
            "choices": [{
                "message": {"content": content},
                "finish_reason": finish_reason,
            }],
            "usage": {"completion_tokens": 4},
        }).encode("utf-8")
        response.__enter__.return_value = response
        return response

    def test_profile_binds_cached_attacker_and_fp8_victim(self):
        self.assertEqual(LOCAL_FAST_PROFILE["profile_id"], LOCAL_FAST_PROFILE_ID)
        self.assertTrue(
            LOCAL_FAST_PROFILE["attacker_runtime"]["generation"]["use_cache"]
        )
        self.assertEqual(
            LOCAL_FAST_PROFILE["victim_runtime"]["quantization"], "fp8"
        )
        self.assertTrue(
            LOCAL_FAST_PROFILE["victim_runtime"]["raw_response_logged_before_parse"]
        )
        self.assertEqual(
            LOCAL_FAST_PROFILE["victim_runtime"]["final_c0_transport"][
                "transport_id"
            ],
            FINAL_C0_TRANSPORT_ID,
        )
        self.assertEqual(
            LOCAL_FAST_PROFILE["victim_runtime"][
                "final_c0_transport_payload_sha256"
            ],
            "6fb06e5896d28fd05f097f50ef98cbd7752773e6c1fe86795090e6e4361e5ebd",
        )
        self.assertEqual(
            FINAL_C0_TRANSPORT_POLICY_SHA256,
            "6fb06e5896d28fd05f097f50ef98cbd7752773e6c1fe86795090e6e4361e5ebd",
        )

    def test_raw_response_precedes_parse_and_ledger_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            client = LocalVllmVictimClient(ledger)
            with patch(
                "urllib.request.urlopen",
                return_value=self._response('{"kind":"final"}'),
            ):
                result = client(
                    [{"role": "user", "content": "x"}],
                    structured_outputs={"json": {"type": "object"}},
                    role="victim",
                    context={"goal_id": "g"},
                )
            client.record_parse_outcome(
                result, status="canonical_valid", context={"goal_id": "g"}
            )
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertEqual(
                [row["event"] for row in rows],
                ["request", "response_raw", "response", "parse_outcome"],
            )
            summary = load_local_vllm_ledger(ledger, require_complete=True)
            self.assertEqual(summary["logical_calls"], 1)
            self.assertTrue(summary["complete"])

    def test_malformed_http_200_is_preserved_and_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            client = LocalVllmVictimClient(ledger)
            urlopen = MagicMock(return_value=self._response('{"kind":'))
            with patch("urllib.request.urlopen", urlopen):
                with self.assertRaises(LocalVllmVictimError):
                    client(
                        [{"role": "user", "content": "x"}],
                        structured_outputs={"json": {"type": "object"}},
                    )
            self.assertEqual(urlopen.call_count, 1)
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertEqual(rows[1]["event"], "response_raw")
            self.assertEqual(rows[1]["raw_response"], self._response('{"kind":').read().decode())
            self.assertFalse(rows[2]["accepted"])

    def test_custom_local_endpoint_and_identity_are_audited(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "dual.jsonl"
            client = LocalVllmVictimClient(
                ledger,
                endpoint="http://127.0.0.1:8001/v1/chat/completions",
                model="qwen3.5-9b-dual",
                provider="local-vllm-dual",
                max_tokens=1024,
                ledger_kind="h1_dual_local_victim_ledger_event",
            )
            response = self._response('{"kind":"final"}')
            response.read.return_value = json.dumps({
                "model": "qwen3.5-9b-dual",
                "choices": [{
                    "message": {"content": '{"kind":"final"}'},
                    "finish_reason": "stop",
                }],
                "usage": {"completion_tokens": 4},
            }).encode("utf-8")
            with patch("urllib.request.urlopen", return_value=response) as urlopen:
                result = client(
                    [{"role": "user", "content": "x"}],
                    structured_outputs={"json": {"type": "object"}},
                )
            client.record_parse_outcome(result, status="canonical_valid")
            request = urlopen.call_args.args[0]
            self.assertEqual(
                request.full_url,
                "http://127.0.0.1:8001/v1/chat/completions",
            )
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertEqual(rows[0]["provider"], "local-vllm-dual")
            self.assertEqual(rows[0]["model"], "qwen3.5-9b-dual")
            summary = load_local_vllm_ledger(
                ledger,
                require_complete=True,
                expected_model="qwen3.5-9b-dual",
                expected_max_tokens=1024,
                expected_ledger_kind="h1_dual_local_victim_ledger_event",
            )
            self.assertTrue(summary["complete"])

    def test_dual_profile_bounds_strings_and_aligns_action_schema_with_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "bounded.jsonl"
            client = LocalVllmVictimClient(
                ledger,
                final_answer_max_length=512,
                action_string_max_length=512,
                strict_declared_action_arguments=True,
            )
            schema = {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"const": "action"},
                            "tool": {"const": "retrieve"},
                            "arguments": {
                                "type": "object",
                                "additionalProperties": True,
                                "properties": {
                                    "query": {"type": "string"},
                                    "body": {
                                        "oneOf": [
                                            {"type": "string"},
                                            {
                                                "type": "object",
                                                "properties": {
                                                    "$observation": {"type": "integer"}
                                                },
                                            },
                                        ]
                                    },
                                },
                            },
                        },
                    },
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"const": "final"},
                            "answer": {"type": "string"},
                        },
                    },
                ]
            }
            with patch(
                "urllib.request.urlopen",
                return_value=self._response('{"kind":"final","answer":"ok"}'),
            ) as urlopen:
                client(
                    [{"role": "user", "content": "x"}],
                    structured_outputs={"json": schema},
                )
            body = json.loads(urlopen.call_args.args[0].data)
            self.assertNotIn(
                "maxLength",
                body["structured_outputs"]["json"]["oneOf"][0]["properties"].get("tool", {}),
            )
            action_properties = body["structured_outputs"]["json"]["oneOf"][0]["properties"]["arguments"]["properties"]
            self.assertFalse(
                body["structured_outputs"]["json"]["oneOf"][0]["properties"]["arguments"]["additionalProperties"]
            )
            self.assertEqual(action_properties["query"]["maxLength"], 512)
            self.assertEqual(action_properties["body"]["oneOf"][0]["maxLength"], 512)
            self.assertNotIn(
                "maxLength",
                action_properties["body"]["oneOf"][1]["properties"]["$observation"],
            )
            self.assertEqual(
                body["structured_outputs"]["json"]["oneOf"][1]["properties"]["answer"]["maxLength"],
                512,
            )
            self.assertEqual(body["bad_words"], list(JSON_CONTROL_ESCAPE_BAD_WORDS))

    def test_control_character_response_still_fails_closed_without_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "control.jsonl"
            client = LocalVllmVictimClient(
                ledger,
                final_answer_max_length=512,
                action_string_max_length=512,
                strict_declared_action_arguments=True,
            )
            response = self._response('{"kind":"final","answer":"bad\bvalue"}')
            with patch("urllib.request.urlopen", return_value=response) as urlopen:
                with self.assertRaises(LocalVllmVictimError):
                    client(
                        [{"role": "user", "content": "x"}],
                        structured_outputs={
                            "json": {
                                "oneOf": [
                                    {
                                        "type": "object",
                                        "properties": {
                                            "kind": {"const": "action"},
                                            "arguments": {
                                                "type": "object",
                                                "properties": {
                                                    "query": {"type": "string"}
                                                },
                                            },
                                        },
                                    },
                                    {
                                        "type": "object",
                                        "properties": {
                                            "kind": {"const": "final"},
                                            "answer": {"type": "string"},
                                        },
                                    },
                                ]
                            }
                        },
                    )
            self.assertEqual(urlopen.call_count, 1)
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertEqual([row["event"] for row in rows], ["request", "response_raw", "response"])
            self.assertFalse(rows[-1]["accepted"])
            self.assertEqual(rows[-1]["error_type"], "JSONDecodeError")

    def test_final_c0_transport_canonicalizes_exact_crash_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "final-c0.jsonl"
            client = LocalVllmVictimClient(
                ledger,
                final_answer_max_length=512,
                action_string_max_length=512,
                strict_declared_action_arguments=True,
                final_c0_transport_id=FINAL_C0_TRANSPORT_ID,
            )
            content = '{"answer": "' + ("x" * 511) + '\b", "kind": "final"}'
            with self.assertRaises(json.JSONDecodeError) as strict_error:
                json.loads(content)
            self.assertEqual(strict_error.exception.pos, 523)
            with patch(
                "urllib.request.urlopen", return_value=self._response(content)
            ) as urlopen:
                result = client(
                    [{"role": "user", "content": "x"}],
                    structured_outputs={
                        "json": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "kind": {"const": "action"},
                                        "arguments": {
                                            "type": "object",
                                            "properties": {
                                                "query": {"type": "string"}
                                            },
                                        },
                                    },
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "answer": {"type": "string"},
                                        "kind": {"const": "final"},
                                    },
                                },
                            ]
                        }
                    },
                )
            client.record_parse_outcome(result, status="canonical_valid")
            self.assertEqual(urlopen.call_count, 1)
            parsed = json.loads(result["content"])
            self.assertEqual(parsed["kind"], "final")
            self.assertEqual(parsed["answer"], ("x" * 511) + "\b")
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertEqual(
                [row["event"] for row in rows],
                ["request", "response_raw", "response", "parse_outcome"],
            )
            self.assertIn("\\b", rows[1]["raw_response"])
            transport = rows[2]["final_c0_transport"]
            self.assertTrue(transport["applied"])
            self.assertEqual(transport["strict_error_position"], 523)
            self.assertEqual(
                transport["controls"], [{"codepoint": "U+0008", "position": 523}]
            )
            summary = load_local_vllm_ledger(
                ledger,
                require_complete=True,
                expected_final_c0_transport_id=FINAL_C0_TRANSPORT_ID,
            )
            self.assertEqual(summary["final_c0_canonicalized"], 1)

    def test_final_c0_transport_keeps_clean_content_byte_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "clean.jsonl"
            client = LocalVllmVictimClient(
                ledger, final_c0_transport_id=FINAL_C0_TRANSPORT_ID
            )
            content = '{"answer":"clean","kind":"final"}'
            with patch(
                "urllib.request.urlopen", return_value=self._response(content)
            ):
                result = client(
                    [{"role": "user", "content": "x"}],
                    structured_outputs={"json": {"type": "object"}},
                )
            client.record_parse_outcome(result, status="canonical_valid")
            self.assertEqual(result["content"], content)
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertFalse(rows[2]["final_c0_transport"]["applied"])
            summary = load_local_vllm_ledger(
                ledger,
                require_complete=True,
                expected_final_c0_transport_id=FINAL_C0_TRANSPORT_ID,
            )
            self.assertEqual(summary["final_c0_canonicalized"], 0)

    def test_final_c0_transport_rejects_action_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "action-control.jsonl"
            client = LocalVllmVictimClient(
                ledger, final_c0_transport_id=FINAL_C0_TRANSPORT_ID
            )
            content = (
                '{"arguments":{"query":"bad\bvalue"},'
                '"kind":"action","tool":"retrieve"}'
            )
            with patch(
                "urllib.request.urlopen", return_value=self._response(content)
            ) as urlopen:
                with self.assertRaises(LocalVllmVictimError):
                    client(
                        [{"role": "user", "content": "x"}],
                        structured_outputs={"json": {"type": "object"}},
                    )
            self.assertEqual(urlopen.call_count, 1)
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertEqual(
                [row["event"] for row in rows],
                ["request", "response_raw", "response"],
            )
            self.assertFalse(rows[-1]["accepted"])


if __name__ == "__main__":
    unittest.main()
