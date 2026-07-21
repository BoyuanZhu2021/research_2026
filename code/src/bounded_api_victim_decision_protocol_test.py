from __future__ import annotations

import copy
import unittest

from .bounded_api_victim_decision_protocol import (
    BOUNDED_API_OUTPUT_CONTRACT_SUFFIX,
    BOUNDED_API_PROTOCOL_ID,
    MAX_TERMINAL_ANSWER_CHARS,
    bounded_api_decision_json_schema,
    build_bounded_api_request_contract,
    parse_bounded_api_victim_decision,
)
from .victim_decision_protocol import VictimDecisionError, decision_json_schema


TOOLS = {
    "Lookup": {
        "name": "Lookup",
        "parameters": [{"name": "query", "type": "string", "required": True}],
    }
}


class BoundedApiVictimDecisionProtocolTest(unittest.TestCase):
    def test_only_terminal_answer_bound_differs(self):
        shared = decision_json_schema(TOOLS, current_step=0)
        bounded = bounded_api_decision_json_schema(TOOLS, current_step=0)
        expected = copy.deepcopy(shared)
        expected["title"] = "H1ApiVictimOneDecisionBoundedTerminal"
        expected["oneOf"][-1]["properties"]["answer"] = {
            "type": "string", "maxLength": MAX_TERMINAL_ANSWER_CHARS,
        }
        self.assertEqual(bounded, expected)

    def test_request_identity_and_prompt_bound(self):
        messages = [{"role": "user", "content": "prompt"}]
        contract = build_bounded_api_request_contract(messages, TOOLS)
        self.assertEqual(messages, [{"role": "user", "content": "prompt"}])
        self.assertEqual(contract["protocol"]["protocol_id"], BOUNDED_API_PROTOCOL_ID)
        self.assertTrue(contract["messages"][0]["content"].endswith(
            BOUNDED_API_OUTPUT_CONTRACT_SUFFIX
        ))

    def test_parser_preserves_actions_and_bounds_only_terminal_text(self):
        action = parse_bounded_api_victim_decision(
            '{"kind":"action","tool":"Lookup","arguments":{"query":"x"}}', TOOLS
        )
        self.assertEqual((action.kind, action.tool), ("action", "Lookup"))
        final = parse_bounded_api_victim_decision(
            '{"kind":"final","answer":"no"}', TOOLS
        )
        self.assertEqual(final.answer, "no")
        too_long = "x" * (MAX_TERMINAL_ANSWER_CHARS + 1)
        with self.assertRaises(VictimDecisionError):
            parse_bounded_api_victim_decision(
                '{"kind":"final","answer":"' + too_long + '"}', TOOLS
            )


if __name__ == "__main__":
    unittest.main()
