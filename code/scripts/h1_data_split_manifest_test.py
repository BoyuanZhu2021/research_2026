"""CPU golden test for the frozen InjecAgent ds/base experiment split.

No model, API, or GPU is used.  This test proves that training, calibration, and final held-out
retrieve-tool OOD evaluation are fixed by manifest and cannot overlap silently.
"""
from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from src.domains.injecagent import SPLIT_MANIFEST_PATH  # noqa: E402
from src.domains.tooluse_injection import ToolUseInjectionDomain  # noqa: E402

EXPECTED_DATASET_SHA256 = "7959777711834b23e8cca2f89717a8fd6ca92ac3a0a915e38fa563a2cf9cf8bd"
EXPECTED_COUNTS = {"train": 322, "calibration": 69, "ood": 153}


def _ids(goals) -> set[str]:
    return {goal.id for goal in goals}


def _retrieve_tools(goals) -> set[str]:
    return {goal.meta["target_tools"][0] for goal in goals}


def _contexts(goals) -> set[str]:
    return {goal.meta["User Tool"] for goal in goals}


def main() -> int:
    domain = ToolUseInjectionDomain(attack="ds", defense_tier="none")
    assert domain.split_manifest_path == SPLIT_MANIFEST_PATH
    assert domain.dataset_sha256 == EXPECTED_DATASET_SHA256
    manifest = domain.split_manifest
    assert manifest is not None and manifest["manifest_id"] == "injecagent-ds-base-split-v1"

    goals = {name: domain.load_goals(name, n=None) for name in EXPECTED_COUNTS}
    actual_counts = {name: len(rows) for name, rows in goals.items()}
    assert actual_counts == EXPECTED_COUNTS, actual_counts

    ids = {name: _ids(rows) for name, rows in goals.items()}
    assert ids["train"].isdisjoint(ids["calibration"])
    assert ids["train"].isdisjoint(ids["ood"])
    assert ids["calibration"].isdisjoint(ids["ood"])
    assert len(set().union(*ids.values())) == 544

    manifest_names = {"train": "train", "calibration": "calibration", "ood": "final_ood"}
    for api_name, manifest_name in manifest_names.items():
        entry = manifest["splits"][manifest_name]
        returned_ids = [goal.id for goal in goals[api_name]]
        assert returned_ids == entry["goal_ids"]
        assert all(goal.meta["manifest_split"] == manifest_name for goal in goals[api_name])
        assert all(goal.meta["split_manifest_id"] == manifest["manifest_id"] for goal in goals[api_name])
        assert all(goal.meta["dataset_sha256"] == EXPECTED_DATASET_SHA256 for goal in goals[api_name])

    train_tools = _retrieve_tools(goals["train"])
    calibration_tools = _retrieve_tools(goals["calibration"])
    ood_tools = _retrieve_tools(goals["ood"])
    assert len(train_tools) == len(calibration_tools) == 23
    assert train_tools == calibration_tools
    assert len(ood_tools) == 9 and ood_tools.isdisjoint(train_tools)

    train_contexts = _contexts(goals["train"])
    calibration_contexts = _contexts(goals["calibration"])
    ood_contexts = _contexts(goals["ood"])
    assert len(train_contexts) == 14
    assert calibration_contexts == set(manifest["policy"]["calibration_contexts"])
    assert len(calibration_contexts) == 3 and train_contexts.isdisjoint(calibration_contexts)
    assert len(ood_contexts) == 17

    # The source is a complete 32 x 17 grid.  Every in-domain retrieve tool contributes fourteen
    # training contexts plus the same three held-out calibration contexts; every OOD tool keeps all
    # seventeen contexts exclusively for final evaluation.
    assert set(Counter(g.meta["target_tools"][0] for g in goals["train"]).values()) == {14}
    assert set(Counter(g.meta["target_tools"][0] for g in goals["calibration"]).values()) == {3}
    assert set(Counter(g.meta["target_tools"][0] for g in goals["ood"]).values()) == {17}

    # Sampling is deterministic inside a fixed split and cannot cross into calibration or OOD.
    sample_a = domain.load_goals("train", seed=7, n=48)
    sample_b = domain.load_goals("train", seed=7, n=48)
    assert [goal.id for goal in sample_a] == [goal.id for goal in sample_b]
    assert _ids(sample_a).issubset(ids["train"])

    try:
        domain.load_goals("indomain", n=1)
    except ValueError as exc:
        assert "ambiguous" in str(exc) and "calibration" in str(exc)
    else:
        raise AssertionError("tool-use loader must reject the ambiguous legacy 'indomain' split")

    print(
        "DATA SPLIT MANIFEST: PASS "
        f"train={len(goals['train'])} calibration={len(goals['calibration'])} "
        f"final_ood={len(goals['ood'])}; tools=23/23/9; contexts=14/3/17; "
        f"sha256={domain.dataset_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
