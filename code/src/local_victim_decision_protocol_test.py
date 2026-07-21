from __future__ import annotations

import json
import unittest

from .local_victim_decision_protocol import (
    LOCAL_COMPACT_PROTOCOL_ID,
    build_local_compact_request_contract,
    local_compact_decision_to_react,
    parse_local_compact_victim_decision,
)
from .victim_decision_protocol import VictimDecisionError


TOOLS = {
    "Retrieve": {"name": "Retrieve", "description": "retrieve", "parameters": []},
}


class LocalVictimDecisionProtocolTest(unittest.TestCase):
    def test_compact_terminal_and_distinct_identity(self):
        contract = build_local_compact_request_contract(
            [{"role": "user", "content": "do the task"}], TOOLS
        )
        self.assertEqual(contract["protocol"]["protocol_id"], LOCAL_COMPACT_PROTOCOL_ID)
        terminal = contract["structured_outputs"]["json"]["oneOf"][-1]
        self.assertEqual(terminal["required"], ["kind"])
        self.assertNotIn("answer", terminal["properties"])
        decision = parse_local_compact_victim_decision(json.dumps({"kind": "final"}), TOOLS)
        self.assertEqual(local_compact_decision_to_react(decision), "Final Answer: [terminal]")

    def test_old_answer_terminal_is_rejected(self):
        with self.assertRaises(VictimDecisionError):
            parse_local_compact_victim_decision(
                json.dumps({"kind": "final", "answer": "refuse"}), TOOLS
            )


if __name__ == "__main__":
    unittest.main()
