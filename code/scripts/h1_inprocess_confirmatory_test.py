"""CPU contract tests for the multi-seed/final-OOD confirmation profile."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from h1_inprocess_confirmatory_analyze import (
    PI_AUTHORIZATION, _contrast, _holm, build_final_authorization,
)
from h1_inprocess_confirmatory_eval import (
    GRID, PROFILE_ID, _load_profile_config, _tree_sha256, panel_key,
)
from src.local_vllm_victim import (  # noqa: E402
    FINAL_C0_TRANSPORT_ID, FINAL_C0_TRANSPORT_POLICY_SHA256,
)

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"


def test_confirmatory_curriculum_registers_three_seeds() -> None:
    import sys
    sys.path.insert(0, str(CODE))
    from src.inprocess_curriculum_protocol import load_config

    config = load_config(CODE / "configs" / "h1_gate_partial_legacy_confirmatory_v1.json")
    assert config["training"]["seeds"] == [0, 1, 2]
    assert config["interaction"]["victim_decision_protocol_id"] == (
        "h1-victim-one-decision-step-bound-observation-ref-v3"
    )
    assert config["interaction"]["strict_declared_action_arguments"] is True
    assert config["interaction"]["final_answer_max_length"] == 512
    assert config["evaluation"]["final_ood_read"] is False


def test_grid_is_base_k4_plus_three_matched_seeds() -> None:
    assert [panel_key(arm, seed) for arm, seed in GRID] == [
        "base-k4", "dense-s0", "sparse-s0", "dense-s1", "sparse-s1",
        "dense-s2", "sparse-s2",
    ]


def test_final_c0_profile_is_versioned_and_symmetric() -> None:
    config = _load_profile_config()
    assert PROFILE_ID == "h1-gate-partial-confirmatory-final-c0-transport-v1"
    assert config["campaign_policy"]["registered_grid"] == [
        panel_key(arm, seed) for arm, seed in GRID
    ]
    assert config["decoder_guard"]["post_generation_repair"] is False
    assert config["victim_transport"]["transport_id"] == FINAL_C0_TRANSPORT_ID
    assert config["victim_transport"]["policy_payload_sha256"] == (
        FINAL_C0_TRANSPORT_POLICY_SHA256
    )
    assert config["campaign_policy"]["post_exposure_confirmation"] is True


def test_confirmatory_adapter_tree_uses_training_canonical_hash() -> None:
    from src.h20_training_artifacts import tree_sha256

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (root / "nested").mkdir()
        (root / "nested" / "adapter.bin").write_bytes(b"sealed-adapter")
        assert _tree_sha256(root) == tree_sha256(root)


def test_final_authorization_binds_learning_report_and_pi_quote() -> None:
    from src.inprocess_curriculum_protocol import seal_payload, validate_seal

    report = seal_payload({
        "schema_version": 1,
        "kind": "h1_gate_partial_learning_report",
        "profile_id": PROFILE_ID,
        "complete": True,
        "decision_bearing": False,
        "final_ood_read": False,
        "post_exposure_confirmation": True,
        "victim_final_c0_transport_id": FINAL_C0_TRANSPORT_ID,
        "victim_final_c0_transport_policy_sha256": (
            FINAL_C0_TRANSPORT_POLICY_SHA256
        ),
        "policy_registry": {panel_key(arm, seed): {} for arm, seed in GRID},
    })
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "learning.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        authorization = validate_seal(build_final_authorization(
            path, final_campaign_id="h1-confirm-final-20260720T000000Z"
        ))
    assert authorization["pi_authorization"] == PI_AUTHORIZATION
    assert authorization["scope"] == "unlock the exact 153-goal final-OOD campaign once"
    assert authorization["post_exposure_confirmation"] is True
    assert authorization["victim_final_c0_transport_id"] == FINAL_C0_TRANSPORT_ID
    assert authorization["decision_rule"]["holm_family"] == [
        "dense_minus_sparse", "dense_minus_baseK", "sparse_minus_baseK",
    ]


def test_registered_bootstrap_and_holm_are_directional_but_strict() -> None:
    values = {
        f"g{index}": {"dense": 1.0, "sparse": 0.0, "baseK": 0.0}
        for index in range(20)
    }
    statistic, pvalue = _contrast(values, "dense", "sparse", rng_seed=7)
    assert statistic["point"] == 1.0
    assert statistic["ci95"] == [1.0, 1.0]
    adjusted = _holm({"a": pvalue, "b": 0.02, "c": 0.04})
    assert set(adjusted) == {"a", "b", "c"}
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())


def test_final_split_load_is_textually_after_authorization_check() -> None:
    source = (CODE / "scripts" / "h1_inprocess_confirmatory_eval.py").read_text(
        encoding="utf-8"
    )
    assert source.index("_validate_final_inputs(args, provenance)") < source.index(
        "domain.load_goals(split, seed=0, n=expected_count)"
    )
    assert "raw_attacker_ledger.jsonl" in source
    assert "raw_victim_ledger.jsonl" in source
    assert "ReservedTagDecoderGuard" in source


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"OK: {len(tests)} confirmatory protocol tests")
