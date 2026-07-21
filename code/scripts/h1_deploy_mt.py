"""Transactionally deploy the profile-separated H1 harness to ``/root/autodl-tmp/h1mt``.

The deployed tree contains only the active single-H20 confirmatory path. Historical implementations
are preserved in the audited local source archive, not shipped as executable fallbacks. It never
copies ``.env`` or any credential-bearing file. GPU-bound entrypoints are
compiled and source-tested on the staging tree; they are never imported or invoked with ``--help``
during the no-card CPU gate because their pre-torch guards must fail without the audited UUIDs.

  python code/scripts/h1_deploy_mt.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from model_pins import INJECAGENT_COMMIT  # noqa: E402


def _load_remote():
    """Import the SSH dependency only after the caller explicitly selects --execute."""
    import remote
    return remote

REMOTE_ROOT = "/root/autodl-tmp/h1mt"
REMOTE_STAGE = "/root/autodl-tmp/h1mt_deploy_stage"
REMOTE_BACKUP = "/root/autodl-tmp/h1mt_deploy_backup"
REMOTE_BUNDLE = "/root/autodl-tmp/h1mt_deploy_bundle.tar.gz"
INJEC = CODE.parent / "data" / "InjecAgent"
DATA_FILES = {
    INJEC / "data" / "test_cases_ds_base.json": "data/InjecAgent/data/test_cases_ds_base.json",
    INJEC / "data" / "tools.json": "data/InjecAgent/data/tools.json",
    INJEC / "src" / "prompts" / "agent_prompts.py": "data/InjecAgent/src/prompts/agent_prompts.py",
}

# Every active script and source module is explicit. This prevents a newly added diagnostic or
# historical implementation from silently expanding the decision-bearing remote tree.
DEPLOYED_SCRIPT_FILES = (
    "h1_data_split_manifest_test.py",
    "h1_deploy_mt.py",
    "h1_ds_oracle_test.py",
    "h1_h20_controller_binding_test.py",
    "h1_h20_controller_common.py",
    "h1_h20_fp8_controller.py",
    "h1_h20_gate_controller.py",
    "h1_h20_smoke_controller.py",
    "h1_inprocess_curriculum_analyze.py",
    "h1_inprocess_curriculum_analyze_test.py",
    "h1_inprocess_curriculum_controller.py",
    "h1_inprocess_curriculum_controller_test.py",
    "h1_inprocess_curriculum_pilot.py",
    "h1_inprocess_curriculum_pilot_test.py",
    "h1_inprocess_confirmatory_analyze.py",
    "h1_inprocess_confirmatory_controller.py",
    "h1_inprocess_confirmatory_eval.py",
    "h1_inprocess_confirmatory_test.py",
    "h1_mt_grpo_train_h20.py",
    "h1_provision.py",
    "h1_remote_status.py",
    "h1_serve_victim_h20.py",
    "h1_tooluse_gate1_local.py",
    "h1_tooluse_gate1_local_test.py",
    "h1_tooluse_mt_pipeline_test.py",
    "h1_tooluse_stage0_test.py",
    "h1_victim_h20_runtime_test.py",
    "h1_victim_fp8_repeatability.py",
    "h1_victim_fp8_repeatability_test.py",
    "h1_victim_quant_spotcheck.py",
    "h1_victim_quant_spotcheck_test.py",
)

DEPLOYED_SRC_FILES = (
    "src/__init__.py",
    "src/api_victim_decision_protocol.py",
    "src/api_victim_decision_protocol_test.py",
    "src/attacker.py",
    "src/bounded_api_victim_decision_protocol.py",
    "src/bounded_api_victim_decision_protocol_test.py",
    "src/deployment_identity.py",
    "src/direct_extraction_episode.py",
    "src/domains/__init__.py",
    "src/domains/base.py",
    "src/domains/extraction_direct.py",
    "src/domains/extraction_multifield.py",
    "src/domains/extraction_oracle.py",
    "src/domains/injecagent.py",
    "src/domains/injecagent_ds_oracle.py",
    "src/domains/injecagent_oracle.py",
    "src/domains/tooluse_injection.py",
    "src/domains/tooluse_oracle.py",
    "src/dual_worker_artifacts.py",
    "src/generation_runtime.py",
    "src/generation_runtime_test.py",
    "src/h20_eval_artifacts.py",
    "src/h20_gate_runtime.py",
    "src/h20_serving_identity.py",
    "src/h20_training_artifacts.py",
    "src/h20_training_protocol.py",
    "src/h20_training_protocol_test.py",
    "src/inprocess_curriculum_protocol.py",
    "src/inprocess_curriculum_protocol_test.py",
    "src/interactive_episode.py",
    "src/llm_client.py",
    "src/local_victim_decision_protocol.py",
    "src/local_victim_decision_protocol_test.py",
    "src/local_vllm_victim.py",
    "src/local_vllm_victim_test.py",
    "src/model_pins.py",
    "src/mt_grpo.py",
    "src/mt_rollout.py",
    "src/mt_victim.py",
    "src/process_identity.py",
    "src/process_identity_test.py",
    "src/providers.py",
    "src/qwen35_fast_kernels.py",
    "src/qwen35_fast_kernels_test.py",
    "src/remote.py",
    "src/runtime_profile.py",
    "src/tooluse_gate1_spec.py",
    "src/trace.py",
    "src/training_protocol.py",
    "src/v100_serving_identity.py",
    "src/victim_decision_protocol.py",
    "src/victim_decision_protocol_test.py",
)

# These entrypoints bind an audited GPU UUID and validate CUDA/formal runtime state.  Static compilation is the
# only safe no-card entrypoint check; protocol/source tests below inspect their contracts.
GPU_GUARDED_ENTRYPOINTS = (
    "code/scripts/h1_inprocess_curriculum_pilot.py",
    "code/scripts/h1_inprocess_confirmatory_eval.py",
    "code/scripts/h1_mt_grpo_train_h20.py",
)

SAFE_HELP_ENTRYPOINTS = (
    "h1_inprocess_curriculum_analyze.py",
    "h1_inprocess_curriculum_controller.py",
    "h1_inprocess_confirmatory_analyze.py",
    "h1_inprocess_confirmatory_controller.py",
    "h1_provision.py",
    "h1_remote_status.py",
    "h1_serve_victim_h20.py",
    "h1_tooluse_gate1_local.py",
    "h1_victim_fp8_repeatability.py",
    "h1_victim_quant_spotcheck.py",
)

REMOTE_CPU_GATES = (
    (
        "partial-reachable curriculum",
        "cd code && python -m unittest src.inprocess_curriculum_protocol_test && "
        "cd scripts && python h1_inprocess_curriculum_analyze_test.py && "
        "python h1_inprocess_curriculum_controller_test.py && "
        "python h1_inprocess_curriculum_pilot_test.py",
    ),
    (
        "in-process confirmatory profile",
        "cd code/scripts && python h1_inprocess_confirmatory_test.py",
    ),
    ("split manifest", "python code/scripts/h1_data_split_manifest_test.py"),
    ("Oracle", "python code/scripts/h1_ds_oracle_test.py"),
    ("tool-use Stage 0", "python code/scripts/h1_tooluse_stage0_test.py"),
    ("tool-use multi-turn", "python code/scripts/h1_tooluse_mt_pipeline_test.py"),
    ("H20 Gate contract", "python code/scripts/h1_tooluse_gate1_local_test.py"),
    ("H20 quantization", "python code/scripts/h1_victim_quant_spotcheck_test.py"),
    ("H20 FP8 repeatability", "python code/scripts/h1_victim_fp8_repeatability_test.py"),
    ("H20 serving runtime", "python code/scripts/h1_victim_h20_runtime_test.py"),
    (
        "active H20 protocols",
        "cd code && python -m unittest src.victim_decision_protocol_test "
        "src.api_victim_decision_protocol_test "
        "src.bounded_api_victim_decision_protocol_test "
        "src.generation_runtime_test "
        "src.local_victim_decision_protocol_test "
        "src.local_vllm_victim_test src.qwen35_fast_kernels_test "
        "src.h20_training_protocol_test src.inprocess_curriculum_protocol_test "
        "src.process_identity_test",
    ),
)


def _assert_fixed_remote_paths(
    root: str = REMOTE_ROOT,
    stage: str = REMOTE_STAGE,
    backup: str = REMOTE_BACKUP,
    bundle: str = REMOTE_BUNDLE,
) -> None:
    """Reject any drift from the reviewed project-only remote paths."""
    expected = (
        "/root/autodl-tmp/h1mt",
        "/root/autodl-tmp/h1mt_deploy_stage",
        "/root/autodl-tmp/h1mt_deploy_backup",
        "/root/autodl-tmp/h1mt_deploy_bundle.tar.gz",
    )
    if (root, stage, backup, bundle) != expected:
        raise RuntimeError("remote deployment paths differ from the reviewed fixed layout")
    paths = tuple(PurePosixPath(value) for value in (root, stage, backup, bundle))
    if len(set(paths)) != 4 or any(path == PurePosixPath("/") for path in paths):
        raise RuntimeError("remote deployment paths must be distinct non-root paths")
    if any(not str(path).startswith("/root/autodl-tmp/h1") for path in paths):
        raise RuntimeError("remote deployment path escapes the project-owned prefix")


def _deployment_plan() -> list[tuple[Path, str]]:
    plan: list[tuple[Path, str]] = []
    for name in DEPLOYED_SRC_FILES:
        plan.append((CODE / name, f"code/{name}"))
    for path in sorted((CODE / "configs").glob("*.json")):
        plan.append((path, path.relative_to(CODE.parent).as_posix()))
    for name in DEPLOYED_SCRIPT_FILES:
        plan.append((CODE / "scripts" / name, f"code/scripts/{name}"))
    plan.extend(DATA_FILES.items())

    relative_paths = [relative for _local, relative in plan]
    duplicates = sorted({item for item in relative_paths if relative_paths.count(item) > 1})
    if duplicates:
        raise RuntimeError(f"deployment plan contains duplicate paths: {duplicates}")
    forbidden = [
        relative for relative in relative_paths
        if PurePosixPath(relative).name == ".env" or "credential" in relative.lower()
    ]
    if forbidden:
        raise RuntimeError(f"deployment plan contains forbidden credential paths: {forbidden}")
    return plan


def _git_output(repo: Path, *args: str) -> str:
    cmd = ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _deployment_manifest(plan: list[tuple[Path, str]]) -> dict:
    files = {}
    tree = hashlib.sha256()
    for path, rel in sorted(plan, key=lambda item: item[1]):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[rel] = digest
        tree.update(f"{rel}\0{digest}\n".encode())
    project = CODE.parent
    return {
        "schema_version": 1,
        "project_commit": _git_output(project, "rev-parse", "HEAD"),
        "project_code_dirty": bool(_git_output(project, "status", "--porcelain", "--", "code")),
        "injecagent_commit": INJECAGENT_COMMIT,
        "injecagent_clean": True,
        "deployed_tree_sha256": tree.hexdigest(),
        "files": files,
    }


def _assert_injecagent_identity() -> None:
    """Refuse to deploy data/tool/prompt bytes from a drifted vendored checkout."""
    actual_commit = _git_output(INJEC, "rev-parse", "HEAD")
    if actual_commit != INJECAGENT_COMMIT:
        raise RuntimeError(
            "InjecAgent commit mismatch: "
            f"expected {INJECAGENT_COMMIT}, got {actual_commit!r}"
        )
    dirty = _git_output(INJEC, "status", "--porcelain")
    if dirty:
        raise RuntimeError(
            "InjecAgent checkout is not clean; refusing to deploy experiment inputs:\n"
            + dirty[:1000]
        )


def _prepare_local_release() -> tuple[list[tuple[Path, str]], dict]:
    _assert_fixed_remote_paths(REMOTE_ROOT, REMOTE_STAGE, REMOTE_BACKUP, REMOTE_BUNDLE)
    plan = _deployment_plan()
    missing = [loc for loc, _ in plan if not loc.is_file()]
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"deployment aborted; required local files are missing:\n{details}")
    _assert_injecagent_identity()
    return plan, _deployment_manifest(plan)


def _write_deployment_bundle(
    destination: Path,
    plan: list[tuple[Path, str]],
    manifest: dict,
) -> None:
    """Write one credential-free archive to avoid a network round trip per file."""
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tarfile.open(destination, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for local, relative in plan:
            archive.add(local, arcname=relative, recursive=False)
        info = tarfile.TarInfo("deployment_manifest.json")
        info.size = len(manifest_bytes)
        info.mode = 0o600
        archive.addfile(info, io.BytesIO(manifest_bytes))


def _execute_deployment(plan: list[tuple[Path, str]], manifest: dict) -> int:
    RM = _load_remote()
    rem_dirs = {
        str(PurePosixPath(relative).parent)
        for _local, relative in plan
    }

    cli = RM.connect()
    print("connected.")

    stage_dirs = " && ".join(f"mkdir -p {REMOTE_STAGE}/{d}" for d in sorted(rem_dirs))
    rc, _out, err = RM.run(
        cli,
        f"rm -rf {REMOTE_STAGE} && rm -f {REMOTE_BUNDLE} && "
        f"mkdir -p {REMOTE_STAGE} && {stage_dirs}",
        timeout=60,
    )
    if rc != 0:
        cli.close()
        raise RuntimeError(f"remote staging setup failed (rc={rc}): {err[:400]}")
    # The SeetaCloud gateway can make one SFTP round trip per file take minutes.  Transfer one
    # credential-free bundle, then retain the same byte-level manifest and staging gates.
    with tempfile.TemporaryDirectory(prefix="h1mt-deploy-") as temp_dir:
        bundle = Path(temp_dir) / "h1mt_deploy_bundle.tar.gz"
        _write_deployment_bundle(bundle, plan, manifest)
        print(f"prepared deployment bundle: {bundle.stat().st_size} bytes", flush=True)
        sftp = cli.open_sftp()
        try:
            sftp.put(str(bundle), REMOTE_BUNDLE)
        finally:
            sftp.close()
    rc, _out, err = RM.run(
        cli,
        f"tar --no-same-owner --no-same-permissions -xzf {REMOTE_BUNDLE} "
        f"-C {REMOTE_STAGE} && rm -f {REMOTE_BUNDLE}",
        timeout=120,
    )
    if rc != 0:
        cli.close()
        raise RuntimeError(f"remote bundle extraction failed (rc={rc}): {err[:400]}")
    print(f"transferred {len(plan)} files -> staging {REMOTE_STAGE}/", flush=True)

    rc, out, err = RM.run(
        cli,
        f"cd {REMOTE_STAGE} && python -c \"import sys; sys.path.insert(0,'code'); "
        f"from src.deployment_identity import verify_deployment; "
        f"m=verify_deployment('.'); print('DEPLOYMENT_IDENTITY_OK',len(m['files']),"
        f"m['deployed_tree_sha256'])\"",
        timeout=120,
    )
    print("remote deployment identity:", out.strip() or err.strip()[:400])
    if rc != 0 or "DEPLOYMENT_IDENTITY_OK" not in out:
        cli.close()
        raise RuntimeError(f"remote deployment identity check failed (rc={rc}): {err[:400]}")

    rc, out, err = RM.run(cli, f"cd {REMOTE_STAGE} && python -c \"import sys; sys.path.insert(0,'code'); "
                               f"sys.path.insert(0,'code/src'); from src.mt_rollout import rollout_trajectory; "
                               f"from src.mt_grpo import group_advantages; "
                               f"from src.mt_victim import make_victim_batch_fn; "
                               f"from src.domains.extraction_multifield import MultiFieldExtractionDomain as M; "
                               f"from src.domains.tooluse_injection import ToolUseInjectionDomain as T; "
                               f"gm=M(K=5,tau=1.0,defense_tier='light').load_goals('indomain',n=2); "
                               f"gt=T(attack='ds',defense_tier='none').load_goals('train',n=2); "
                               f"print('IMPORT_OK multifield',len(gm),'tooluse',len(gt),"
                               f"'m',len(gt[0].meta['target_tools']))\"", timeout=120)
    print("remote import check:", out.strip() or err.strip()[:400])
    if rc != 0 or "IMPORT_OK" not in out:
        cli.close()
        raise RuntimeError(f"remote tooluse import smoke failed (rc={rc}): {err[:400]}")

    compile_targets = " ".join(GPU_GUARDED_ENTRYPOINTS)
    rc, out, err = RM.run(
        cli,
        f"cd {REMOTE_STAGE} && python -m py_compile {compile_targets}",
        timeout=120,
    )
    if rc != 0:
        cli.close()
        raise RuntimeError(f"remote guarded-entrypoint py_compile failed (rc={rc}): {err[:400]}")
    print("remote GPU-guarded entrypoints: PY_COMPILE_OK")

    for label, command in REMOTE_CPU_GATES:
        rc, out, err = RM.run(cli, f"cd {REMOTE_STAGE} && {command}", timeout=180)
        detail = out.strip()[-800:] or err.strip()[-800:]
        print(f"remote {label}:", detail)
        if rc != 0:
            cli.close()
            raise RuntimeError(
                f"remote {label} CPU gate failed (rc={rc}): {(err or out)[-800:]}"
            )

    for entrypoint in SAFE_HELP_ENTRYPOINTS:
        rc, out, err = RM.run(
            cli,
            f"cd {REMOTE_STAGE} && python code/scripts/{entrypoint} --help >/dev/null",
            timeout=120,
        )
        if rc != 0:
            cli.close()
            raise RuntimeError(f"remote {entrypoint} import/help failed (rc={rc}): {err[:400]}")
    print("remote CPU-safe entrypoints: HELP_OK")

    promote = (
        f"set -eu; rm -rf {REMOTE_BACKUP}; mkdir -p {REMOTE_BACKUP}/data "
        f"{REMOTE_ROOT}/runs {REMOTE_ROOT}/data; "
        "had_code=0; had_data=0; had_manifest=0; "
        "rollback() { trap - ERR; "
        f"if test -e {REMOTE_BACKUP}/code; then rm -rf {REMOTE_ROOT}/code; "
        f"mv {REMOTE_BACKUP}/code {REMOTE_ROOT}/code; "
        f"elif test \"$had_code\" = 0; then rm -rf {REMOTE_ROOT}/code; fi; "
        f"if test -e {REMOTE_BACKUP}/data/InjecAgent; then "
        f"rm -rf {REMOTE_ROOT}/data/InjecAgent; mv {REMOTE_BACKUP}/data/InjecAgent "
        f"{REMOTE_ROOT}/data/InjecAgent; elif test \"$had_data\" = 0; then "
        f"rm -rf {REMOTE_ROOT}/data/InjecAgent; fi; "
        f"if test -e {REMOTE_BACKUP}/deployment_manifest.json; then "
        f"rm -f {REMOTE_ROOT}/deployment_manifest.json; mv "
        f"{REMOTE_BACKUP}/deployment_manifest.json {REMOTE_ROOT}/deployment_manifest.json; "
        f"elif test \"$had_manifest\" = 0; then rm -f "
        f"{REMOTE_ROOT}/deployment_manifest.json; fi; "
        "}; trap rollback HUP INT TERM ERR; "
        f"if test -e {REMOTE_ROOT}/code; then had_code=1; "
        f"mv {REMOTE_ROOT}/code {REMOTE_BACKUP}/code; fi; "
        f"if test -e {REMOTE_ROOT}/data/InjecAgent; then had_data=1; "
        f"mv {REMOTE_ROOT}/data/InjecAgent {REMOTE_BACKUP}/data/InjecAgent; fi; "
        f"if test -e {REMOTE_ROOT}/deployment_manifest.json; then had_manifest=1; "
        f"mv {REMOTE_ROOT}/deployment_manifest.json {REMOTE_BACKUP}/deployment_manifest.json; fi; "
        f"mv {REMOTE_STAGE}/code {REMOTE_ROOT}/code; "
        f"mv {REMOTE_STAGE}/data/InjecAgent {REMOTE_ROOT}/data/InjecAgent; "
        f"mv {REMOTE_STAGE}/deployment_manifest.json {REMOTE_ROOT}/deployment_manifest.json; "
        f"cd {REMOTE_ROOT}; python -c \"import sys; sys.path.insert(0,'code'); "
        "from src.deployment_identity import verify_deployment; verify_deployment('.')\"; "
        f"trap - HUP INT TERM ERR; rm -rf {REMOTE_BACKUP} {REMOTE_STAGE}; "
        "echo DEPLOY_PROMOTED"
    )
    rc, out, err = RM.run(cli, promote, timeout=60)
    if rc != 0 or "DEPLOY_PROMOTED" not in out:
        cli.close()
        raise RuntimeError(f"remote staging promotion failed (rc={rc}): {err[:400]}")
    print(f"DEPLOY_PROMOTED -> {REMOTE_ROOT}/")
    print(f"deployed_tree_sha256={manifest['deployed_tree_sha256']}")
    cli.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or explicitly execute the transactional H1 deployment. "
            "The default is local plan-only and never connects to a server."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan", action="store_true",
        help="print the fully hashed local deployment manifest (default)",
    )
    mode.add_argument(
        "--execute", action="store_true",
        help="explicitly authorize remote staging, CPU gates, and promotion",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan, manifest = _prepare_local_release()
    if not args.execute:
        print(json.dumps({
            "mode": "plan",
            "remote_mutation": False,
            "file_count": len(plan),
            "manifest": manifest,
        }, indent=2, sort_keys=True))
        return 0
    return _execute_deployment(plan, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
