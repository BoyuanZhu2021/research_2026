from __future__ import annotations

import argparse
import contextlib
import io
import shlex
import unittest

import h1_h20_fp8_controller as fp8_controller
import h1_h20_gate_controller as gate_controller


class H20ControllerBindingTest(unittest.TestCase):
    def test_fp8_cycle_command_uses_explicit_gpu_uuid(self):
        cycle_id = "h1-fp8-cycle-20260718T191747Z"
        gpu_uuid = "GPU-018361b3-f812-a23d-e886-82d38e8501eb"
        tokens = shlex.split(fp8_controller.cycle_command(cycle_id, gpu_uuid))
        self.assertIn(f"H1_H20_GPU_UUID={gpu_uuid}", tokens)
        self.assertIn(f"CUDA_VISIBLE_DEVICES={gpu_uuid}", tokens)
        self.assertIn(f"/fp8_cycles/{cycle_id}", " ".join(tokens))

    def test_gate_command_binds_explicit_cycle_and_gpu(self):
        controller_id = "h1-formal-gate-20260718T200000Z"
        cycle_id = "h1-fp8-cycle-20260718T191747Z"
        gpu_uuid = "GPU-018361b3-f812-a23d-e886-82d38e8501eb"
        command = gate_controller.gate_command(controller_id, cycle_id, gpu_uuid)
        tokens = shlex.split(command)
        self.assertIn(f"H1_H20_GPU_UUID={gpu_uuid}", tokens)
        self.assertIn(f"CUDA_VISIBLE_DEVICES={gpu_uuid}", tokens)
        self.assertIn(
            f"/root/autodl-tmp/h1mt/fp8_cycles/{cycle_id}/repeatability.json", tokens
        )
        self.assertIn(
            f"/root/autodl-tmp/h1mt/fp8_cycles/{cycle_id}/cycle_status.json", tokens
        )
        self.assertNotIn("h1-fp8-cycle-20260718T094922Z", command)
        self.assertNotIn("GPU-5467966a-0fdf-bf65-f715-629e45792480", command)

    def test_identifier_and_gpu_validators_reject_traversal_or_placeholder(self):
        parser = argparse.ArgumentParser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                gate_controller._validate_cycle_id(parser, "../h1-fp8-cycle-bad")
            with self.assertRaises(SystemExit):
                fp8_controller._validate_gpu_uuid(parser, "GPU-<exact-uuid>")


if __name__ == "__main__":
    unittest.main()
