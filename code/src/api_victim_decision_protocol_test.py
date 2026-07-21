from __future__ import annotations

import copy
import unittest

from .api_victim_decision_protocol import (
    API_OUTPUT_CONTRACT_SUFFIX,
    API_PROTOCOL_ID,
    API_VICTIM_DECISION_PROTOCOL,
    HARNESS_TERMINAL_TEXT,
    api_decision_json_schema,
    api_decision_to_react,
    build_api_request_contract,
    parse_api_victim_decision,
)
from .victim_decision_protocol import VictimDecisionError, canonical_sha256, decision_json_schema


TOOLS = {
    "Lookup": {
        "name": "Lookup",
        "parameters": [{"name": "query", "type": "string", "required": True}],
    },
    "GmailSendEmail": {
        "name": "GmailSendEmail",
        "parameters": [
            {"name": "to", "type": "string", "required": True},
            {"name": "body", "type": "string", "required": True},
        ],
    },
}


class ApiVictimDecisionProtocolTest(unittest.TestCase):
    def test_only_terminal_branch_differs_from_shared_schema(self):
        shared = decision_json_schema(TOOLS, current_step=1)
        api = api_decision_json_schema(TOOLS, current_step=1)
        self.assertEqual(api["oneOf"][:-1], shared["oneOf"][:-1])
        self.assertEqual(
            api["oneOf"][-1],
            {
                "type": "object",
                "properties": {"kind": {"const": "final"}},
                "required": ["kind"],
                "additionalProperties": False,
            },
        )
        self.assertNotEqual(canonical_sha256(api), canonical_sha256(shared))

    def test_request_contract_is_api_scoped_and_contains_no_answer_field(self):
        messages = [{"role": "user", "content": "prompt"}]
        contract = build_api_request_contract(messages, TOOLS, current_step=0)
        self.assertEqual(messages[0]["content"], "prompt")
        self.assertEqual(contract["protocol"]["protocol_id"], API_PROTOCOL_ID)
        self.assertEqual(contract["protocol"], API_VICTIM_DECISION_PROTOCOL)
        self.assertTrue(contract["messages"][0]["content"].endswith(API_OUTPUT_CONTRACT_SUFFIX))
        terminal = contract["structured_outputs"]["json"]["oneOf"][-1]
        self.assertNotIn("answer", terminal["properties"])

    def test_compact_terminal_round_trip_and_legacy_terminal_rejection(self):
        decision = parse_api_victim_decision('{"kind":"final"}', TOOLS)
        self.assertEqual(api_decision_to_react(decision), f"Final Answer: {HARNESS_TERMINAL_TEXT}")
        rejected = (
            '{"kind":"final","answer":"no"}',
            '{"answer":"no"}',
            '{"kind":"final","extra":1}',
            '{"kind":"final","kind":"action"}',
        )
        for text in rejected:
            with self.subTest(text=text), self.assertRaises(VictimDecisionError):
                parse_api_victim_decision(text, TOOLS)

    def test_action_and_observation_reference_are_unchanged(self):
        action = parse_api_victim_decision(
            '{"kind":"action","tool":"Lookup","arguments":{"query":"x"}}', TOOLS
        )
        self.assertEqual(
            api_decision_to_react(action),
            'Action: Lookup\nAction Input: {"query":"x"}',
        )
        referenced = parse_api_victim_decision(
            '{"kind":"action","tool":"GmailSendEmail","arguments":'
            '{"to":"outside@example.com","body":{"$observation":0}}}',
            copy.deepcopy(TOOLS),
            observations=["CANARY"],
            current_step=1,
        )
        self.assertEqual(referenced.arguments, '{"body":"CANARY","to":"outside@example.com"}')


if __name__ == "__main__":
    unittest.main()
