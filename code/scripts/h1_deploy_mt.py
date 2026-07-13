"""Ship the full-trajectory multi-turn GRPO trainer + src package + InjecAgent data to the H20 at
/root/autodl-tmp/h1mt/ (package layout so `from src...` + providers' PROJECT_ROOT resolve). Victim is
local vLLM (no API) so no .env is shipped. Re-runnable.

  python code/scripts/h1_deploy_mt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
import remote as RM  # noqa: E402

REMOTE_ROOT = "/root/autodl-tmp/h1mt"
INJEC = CODE.parent / "data" / "InjecAgent"
DATA_FILES = {
    INJEC / "data" / "test_cases_ds_base.json": "data/InjecAgent/data/test_cases_ds_base.json",
    INJEC / "data" / "tools.json": "data/InjecAgent/data/tools.json",
    INJEC / "src" / "prompts" / "agent_prompts.py": "data/InjecAgent/src/prompts/agent_prompts.py",
}


def main():
    cli = RM.connect()
    print("connected.")
    plan: list[tuple[Path, str]] = []
    rem_dirs = set()
    for p in sorted((CODE / "src").rglob("*.py")):
        rel = p.relative_to(CODE.parent).as_posix()
        plan.append((p, rel)); rem_dirs.add(str(Path(rel).parent).replace("\\", "/"))
    plan.append((CODE / "scripts" / "h1_mt_grpo_train.py", "code/scripts/h1_mt_grpo_train.py"))
    rem_dirs.add("code/scripts")
    for loc, rel in DATA_FILES.items():
        plan.append((loc, rel)); rem_dirs.add(str(Path(rel).parent).replace("\\", "/"))

    RM.run(cli, f"mkdir -p {REMOTE_ROOT}/runs && "
                + " && ".join(f"mkdir -p {REMOTE_ROOT}/{d}" for d in sorted(rem_dirs)), timeout=60)
    n = 0
    for loc, rel in plan:
        if loc.exists():
            RM.put_file(cli, str(loc), f"{REMOTE_ROOT}/{rel}"); n += 1
        else:
            print(f"  MISSING: {loc}")
    for pkg in ("code/src", "code/src/domains"):
        RM.run(cli, f"touch {REMOTE_ROOT}/{pkg}/__init__.py", timeout=20)
    print(f"transferred {n} files -> {REMOTE_ROOT}/")

    rc, out, err = RM.run(cli, f"cd {REMOTE_ROOT} && python -c \"import sys; sys.path.insert(0,'code'); "
                               f"sys.path.insert(0,'code/src'); from src.mt_rollout import rollout_trajectory; "
                               f"from src.mt_grpo import group_advantages; "
                               f"from src.domains.extraction_multifield import MultiFieldExtractionDomain as M; "
                               f"g=M(K=5,tau=1.0,defense_tier='light').load_goals('indomain',n=2); "
                               f"print('IMPORT_OK goals',len(g))\"", timeout=120)
    print("remote import check:", out.strip() or err.strip()[:400])
    cli.close()


if __name__ == "__main__":
    main()
