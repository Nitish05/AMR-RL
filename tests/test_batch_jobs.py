"""The batch runner's job-file parser (evaluation tooling; no simulator)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _batch():
    spec = importlib.util.spec_from_file_location("batch", ROOT / "scripts" / "eval" / "batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_reads_jobs_and_skips_comments(tmp_path):
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("# a comment\n\n/tmp/a.log scripts/eval/learning.py --seeds 0 --out /tmp/a\n")
    parsed = _batch().parse(jobs)
    assert len(parsed) == 1 and parsed[0]["argv"][0] == "scripts/eval/learning.py" and parsed[0]["log"] == Path("/tmp/a.log")


def test_parse_refuses_a_repeated_option(tmp_path):
    """argparse keeps only the last occurrence of a repeated nargs='*' option; validation 6's
    base navigation arm lost `--runtime imu_zupt_temp_first_c=0` to a second `--runtime`."""
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("/tmp/b.log scripts/eval/navigation.py --runtime imu_zupt_temp_first_c=0 "
                    "--runtime near_field_guard=false --out /tmp/b\n")
    with pytest.raises(SystemExit, match="repeats --runtime"):
        _batch().parse(jobs)
    jobs.write_text("/tmp/b.log scripts/eval/navigation.py --runtime imu_zupt_temp_first_c=0 near_field_guard=false "
                    "--out /tmp/b\n")
    assert len(_batch().parse(jobs)) == 1
