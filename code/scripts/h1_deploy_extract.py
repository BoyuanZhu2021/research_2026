"""Ship the Gate-2 extraction trainer + src package + minimal InjecAgent data to the H20, preserving
the package layout at /root/autodl-tmp/h1x/ so `from src... import ...` and providers' PROJECT_ROOT
(= h1x, since providers.py is at h1x/code/src/providers.py -> parents[2]) resolve. The 27B victim is
NOT shipped (served via API). Re-runnable (overwrites).

  python code/scripts/h1_deploy_extract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
import remote as RM  # noqa: E402

REMOTE_ROOT = "/root/autodl-tmp/h1x"
INJEC = CODE.parent / "data" / "InjecAgent"

# minimal InjecAgent data the domain loads (ds cases + tools + the agent prompts module)
DATA_FILES = {
    INJEC / "data" / "test_cases_ds_base.json": "data/InjecAgent/data/test_cases_ds_base.json",
    INJEC / "data" / "tools.json": "data/InjecAgent/data/tools.json",
    INJEC / "src" / "prompts" / "agent_prompts.py": "data/InjecAgent/src/prompts/agent_prompts.py",
}


def main():
    cli = RM.connect()
    print("connected.")

    # collect every .py under code/src (small, 384K) + the trainer
    src_files = sorted(p for p in (CODE / "src").rglob("*.py"))
    rem_dirs = set()
    plan: list[tuple[Path, str]] = []
    for p in src_files:
        rel = p.relative_to(CODE.parent).as_posix()          # code/src/...
        plan.append((p, rel))
        rem_dirs.add(str(Path(rel).parent).replace("\\", "/"))
    plan.append((CODE / "scripts" / "h1_grpo_train_extract.py", "code/scripts/h1_grpo_train_extract.py"))
    rem_dirs.add("code/scripts")
    for loc, rel in DATA_FILES.items():
        plan.append((loc, rel))
        rem_dirs.add(str(Path(rel).parent).replace("\\", "/"))

    # ensure __init__.py presence for `src` + `src/domains` package imports
    RM.run(cli, f"mkdir -p {REMOTE_ROOT}/runs && "
                + " && ".join(f"mkdir -p {REMOTE_ROOT}/{d}" for d in sorted(rem_dirs)), timeout=60)

    n = 0
    for loc, rel in plan:
        if not loc.exists():
            print(f"  MISSING (skip): {loc}")
            continue
        RM.put_file(cli, str(loc), f"{REMOTE_ROOT}/{rel}")
        n += 1
    # src packages need __init__.py (repo may rely on implicit ns pkgs locally; be explicit on remote)
    for pkg in ("code/src", "code/src/domains"):
        RM.run(cli, f"touch {REMOTE_ROOT}/{pkg}/__init__.py", timeout=20)

    # minimal remote .env (SiliconFlow creds ONLY) via SFTP so the key never hits a command line/log.
    env = RM.load_env() if hasattr(RM, "load_env") else __import__("providers").load_env()
    lines = [f"{k}={env[k]}" for k in ("SILICONFLOW_API_KEY", "SILICONFLOW_BASE_URL") if env.get(k)]
    assert len(lines) == 2, "SILICONFLOW_API_KEY / SILICONFLOW_BASE_URL missing from local .env"
    RM.put_text(cli, "\n".join(lines) + "\n", f"{REMOTE_ROOT}/.env")
    print("wrote minimal remote .env (SiliconFlow creds only)")

    print(f"transferred {n} files -> {REMOTE_ROOT}/ (code/src + trainer + InjecAgent data)")
    rc, out, err = RM.run(cli, f"cd {REMOTE_ROOT} && python -c \"import sys; sys.path.insert(0,'code'); "
                               f"sys.path.insert(0,'code/src'); "
                               f"from src.domains.extraction_multifield import MultiFieldExtractionDomain as M; "
                               f"d=M(K=5,tau=1.0,defense_tier='light'); g=d.load_goals('indomain',n=2); "
                               f"print('IMPORT_OK goals',len(g),'K',g[0].meta['K'])\"", timeout=120)
    print("remote import check:", out.strip() or err.strip()[:400])
    cli.close()


if __name__ == "__main__":
    main()
