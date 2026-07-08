"""Live H20 status: provisioning + victim server + GRPO training, in one view.

  python code/scripts/h1_remote_status.py            # one-shot snapshot
  python code/scripts/h1_remote_status.py --watch    # refresh every 20s until Ctrl-C
  python code/scripts/h1_remote_status.py --watch --interval 10

Shows disk / GPU / model-download size, and tails every job log (downloads, installs, victim server,
GRPO training) plus the training progress stream (progress.jsonl: step, reward, ASR).
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import remote as RM  # noqa: E402

TMP = "/root/autodl-tmp"
CHECK = rf"""
echo "## disk";  df -h {TMP} | tail -1 | awk '{{print "  free="$4"  used="$3"/"$2}}'
echo "## gpu";   nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | awk '{{print "  "$0}}'
echo "## models"; du -sh {TMP}/hf_home 2>/dev/null | awk '{{print "  hf_home="$1}}'
echo "## downloads (h1_dl.log)";     tail -1 {TMP}/h1_dl.log       2>/dev/null | tr '\r' '\n' | tail -1
echo "## train-stack (h1_pip_train.log)"; tail -1 {TMP}/h1_pip_train.log 2>/dev/null
echo "## vllm-env (h1_vllm.log)";    tail -1 {TMP}/h1_vllm.log     2>/dev/null
echo "## victim server (h1_victim.log)"; tail -2 {TMP}/h1_victim.log  2>/dev/null | tr '\r' '\n' | tail -1
echo "## GRPO training (h1_grpo.log)";   tail -3 {TMP}/h1_grpo.log    2>/dev/null | tr '\r' '\n' | tail -2
echo "## progress (last 3 steps)";   tail -3 {TMP}/h1/progress.jsonl 2>/dev/null
"""


def snapshot(cli):
    rc, out, err = RM.run(cli, CHECK, timeout=90)
    return out.rstrip() + (("\nSTDERR: " + err[:200]) if err.strip() else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=20)
    args = ap.parse_args()

    if not args.watch:
        cli = RM.connect()
        print(snapshot(cli))
        cli.close()
        return

    print("watching H20 (Ctrl-C to stop)...")
    while True:
        try:
            cli = RM.connect()
            stamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\n===== {stamp} =====")
            print(snapshot(cli))
            cli.close()
        except KeyboardInterrupt:
            print("\nstopped.")
            return
        except Exception as e:  # noqa: BLE001
            print(f"[{datetime.datetime.now():%H:%M:%S}] status error: {type(e).__name__}: {e}")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nstopped.")
            return


if __name__ == "__main__":
    main()
