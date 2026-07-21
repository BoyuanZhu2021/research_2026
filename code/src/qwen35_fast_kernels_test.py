from __future__ import annotations

import unittest
from unittest.mock import patch

from .qwen35_fast_kernels import (
    QWEN35_FAST_KERNEL_PINS,
    qwen35_fast_kernel_status,
    require_qwen35_fast_kernels,
)


class Qwen3_5GatedDeltaNet:
    causal_conv1d_fn = staticmethod(lambda: None)
    causal_conv1d_update = staticmethod(lambda: None)
    chunk_gated_delta_rule = staticmethod(lambda: None)
    recurrent_gated_delta_rule = staticmethod(lambda: None)


class _Model:
    def __init__(self, layer):
        self.layer = layer

    def named_modules(self):
        return [("", self), ("model.layers.0.linear_attn", self.layer)]


class FastKernelStatusTest(unittest.TestCase):
    @patch(
        "src.qwen35_fast_kernels._package_versions",
        return_value=QWEN35_FAST_KERNEL_PINS,
    )
    def test_ready_model_is_accepted(self, _versions):
        status = require_qwen35_fast_kernels(_Model(Qwen3_5GatedDeltaNet()))
        self.assertTrue(status["model_fast_path_ready"])
        self.assertEqual(status["gated_delta_net_layer_count"], 1)
        self.assertEqual(QWEN35_FAST_KERNEL_PINS["tilelang"], "0.1.12")

    @patch(
        "src.qwen35_fast_kernels._package_versions",
        return_value={
            "fla-core": None, "causal-conv1d": None, "einops": None,
            "tilelang": None,
        },
    )
    def test_missing_packages_fail_closed(self, _versions):
        status = qwen35_fast_kernel_status(_Model(Qwen3_5GatedDeltaNet()))
        self.assertFalse(status["distribution_pins_match"])
        with self.assertRaisesRegex(RuntimeError, "absent or drifted"):
            require_qwen35_fast_kernels(_Model(Qwen3_5GatedDeltaNet()))


if __name__ == "__main__":
    unittest.main()
