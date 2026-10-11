"""Run evaluation jobs in parallel from a jobs file (evaluation tooling).

Each non-empty, non-comment line is ``<log path> <script> [args...]`` (shell-quoted;
JSON arguments in single quotes). The script runs with this project's Python and
``PYTHONPATH=<cwd>/src`` from ``--cwd`` (a pinned git worktree for validation runs).

* at most ``--jobs`` at a time (each simulation takes 2-2.5 GB; 8 fit in 48 GB);
* on macOS the whole batch runs under ``caffeinate -dims``: rendering needs an awake
  display (12 validation-4 jobs failed when it slept);
* a job whose ``--out`` directory already holds ``result.json`` (or ``summary.json``)
  is skipped, so an interrupted batch can be resumed;
* a failed job is retried up to ``--retries`` times; every failed attempt's log is kept
  as ``<log>.failed-<n>.log``.

    PYTHONPATH=src python scripts/eval/batch.py jobs.txt --jobs 8 --cwd work/wt/r9g-abc1234
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse(path):
    jobs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line)
        if len(parts) < 2:
            raise SystemExit(f"bad job line (need '<log> <script> ...'): {line}")
        jobs.append({"log": Path(parts[0]), "argv": parts[1:]})
    return jobs


def out_dir(argv):
    for flag in ("--out",):
        if flag in argv and argv.index(flag) + 1 < len(argv):
            return Path(argv[argv.index(flag) + 1])
    return None


def done(job):
    out = out_dir(job["argv"])
    return out is not None and any((out / name).exists() for name in ("result.json", "summary.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_file")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--cwd", default=str(ROOT), help="checkout to run in (pinned worktree for validation runs)")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    ap.add_argument("--no-caffeinate", action="store_true")
    args = ap.parse_args()
    if sys.platform == "darwin" and not args.no_caffeinate and not os.environ.get("AMR_CAFFEINATED"):
        env = dict(os.environ, AMR_CAFFEINATED="1")
        os.execvpe("caffeinate", ["caffeinate", "-dims", sys.executable, *sys.argv], env)
    cwd = Path(args.cwd).resolve()
    env = dict(os.environ, PYTHONPATH=str(cwd / "src"), PYTHONDONTWRITEBYTECODE="1")
    queue = [j for j in parse(args.jobs_file) if not done(j)]
    skipped = len(parse(args.jobs_file)) - len(queue)
    print(f"{len(queue)} jobs to run ({skipped} already done), {args.jobs} at a time, in {cwd}", flush=True)
    attempts = {id(j): 0 for j in queue}
    running, failed = [], []
    while queue or running:
        while queue and len(running) < args.jobs:
            job = queue.pop(0)
            attempts[id(job)] += 1
            job["log"].parent.mkdir(parents=True, exist_ok=True)
            fh = open(job["log"], "w")
            proc = subprocess.Popen([args.python, *job["argv"]], cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            running.append((proc, job, fh, time.time()))
            print(f"start {job['log'].name} (attempt {attempts[id(job)]})", flush=True)
        time.sleep(2)
        for item in list(running):
            proc, job, fh, t0 = item
            if proc.poll() is None:
                continue
            running.remove(item)
            fh.close()
            mins = (time.time() - t0) / 60
            if proc.returncode == 0:
                print(f"done  {job['log'].name} ({mins:.1f} min)", flush=True)
                continue
            kept = job["log"].with_name(f"{job['log'].stem}.failed-{attempts[id(job)]}.log")
            job["log"].rename(kept)
            print(f"FAIL  {job['log'].name} exit {proc.returncode} ({mins:.1f} min), log kept as {kept.name}",
                  flush=True)
            if attempts[id(job)] <= args.retries:
                queue.append(job)
            else:
                failed.append(job)
    print(f"finished: {len(failed)} failed" + "".join(f"\n  {j['log']}" for j in failed), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
