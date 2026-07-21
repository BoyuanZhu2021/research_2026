from __future__ import annotations

import unittest

from src.process_identity import ProcessIdentityError, classify_owned_process, parse_proc_stat


def _stat(state: str, start_ticks: int) -> str:
    # Fields after comm start at Linux proc field 3; index 19 is field 22 starttime.
    remainder = [state, *(["0"] * 18), str(start_ticks), "0"]
    return "123 (python worker with spaces) " + " ".join(remainder)


class ProcessIdentityTest(unittest.TestCase):
    def test_parser_handles_spaces_in_comm(self):
        self.assertEqual(parse_proc_stat(_stat("S", 9876)), ("S", 9876))

    def test_matching_live_process(self):
        self.assertTrue(classify_owned_process(
            stat_text=_stat("R", 123), actual_cmdline=["python", "job.py"],
            actual_pgid=55, expected_start_ticks=123,
            expected_cmdline=["python", "job.py"], expected_pgid=55,
        ))

    def test_zombie_is_finished_not_identity_drift(self):
        self.assertFalse(classify_owned_process(
            stat_text=_stat("Z", 123), actual_cmdline=[], actual_pgid=55,
            expected_start_ticks=123, expected_cmdline=["python", "job.py"],
            expected_pgid=55,
        ))

    def test_live_identity_drift_fails_closed(self):
        with self.assertRaises(ProcessIdentityError):
            classify_owned_process(
                stat_text=_stat("S", 124), actual_cmdline=["other"], actual_pgid=99,
                expected_start_ticks=123, expected_cmdline=["python", "job.py"],
                expected_pgid=55,
            )


if __name__ == "__main__":
    unittest.main()
