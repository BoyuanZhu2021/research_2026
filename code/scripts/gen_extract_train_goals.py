"""Generate self-contained multifield-extraction goals for GRPO on the H20 (CPU, no GPU).

Bakes everything the remote trainer needs into a JSONL so the H20 needs NO InjecAgent repo / domain
code / .env: each row = {id, split, prompt (attacker turn-1 chat messages), victim_system,
canaries, tau, K, labels, defense_tier}. The trainer's reward_fn just needs the victim_system +
canaries to run one victim API turn and score disclosure.

  python code/scripts/gen_extract_train_goals.py --K 5 --tau 1.0 --defense light --n-train 48 --n-ood 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from src.attacker import build_initial_messages  # noqa: E402
from src.domains.extraction_multifield import MultiFieldExtractionDomain  # noqa: E402


def rows_for(dom: MultiFieldExtractionDomain, split: str, n: int) -> list[dict]:
    out = []
    for g in dom.load_goals(split, seed=0, n=n):
        out.append({
            "id": g.id, "split": split,
            "prompt": build_initial_messages(dom, g),           # [system, examples..., task]
            "victim_system": dom.build_victim_system(g),
            "canaries": [c for _, c, _ in g.meta["fields"]],
            "labels": [lab for lab, _, _ in g.meta["fields"]],
            "sensitivities": [s for _, _, s in g.meta["fields"]],
            "tau": g.meta["tau"], "K": g.meta["K"], "defense_tier": g.meta["defense_tier"],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--defense", default="light")
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--n-ood", type=int, default=60)
    ap.add_argument("--out", default=str(CODE / "runs" / "extract_goals"))
    args = ap.parse_args()

    dom = MultiFieldExtractionDomain(K=args.K, tau=args.tau, defense_tier=args.defense)
    train = rows_for(dom, "indomain", args.n_train)   # GRPO trains on in-domain profiles
    ood = rows_for(dom, "ood", args.n_ood)            # held-out data-types for eval (Stage 3)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "goals_train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
    (out / "goals_ood.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ood), encoding="utf-8")
    (out / "gen_meta.json").write_text(json.dumps(
        {"K": args.K, "tau": args.tau, "defense": args.defense,
         "n_train": len(train), "n_ood": len(ood),
         "victim": {"provider": "siliconflow", "model": "Qwen/Qwen3.6-27B"}}, indent=2), encoding="utf-8")
    print(f"[gen] wrote {len(train)} train + {len(ood)} ood goals -> {out}")
    print(f"      sample train id={train[0]['id']} K={train[0]['K']} canaries={train[0]['canaries'][:2]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
