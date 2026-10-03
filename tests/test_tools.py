"""Exercise the Bash helpers without retraining the full dataset."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="Bash is required for shell helper tests")


@pytest.fixture
def tool_repo(tmp_path):
    repo = tmp_path / "group project"
    for folder in ["tools", "experiments", "results/runs"]:
        (repo / folder).mkdir(parents=True, exist_ok=True)
    for name in ["accept_runs.sh", "run_all_configs.sh"]:
        shutil.copy2(ROOT / "tools" / name, repo / "tools" / name)
    shutil.copy2(ROOT / "experiments/aggregate.py", repo / "experiments/aggregate.py")
    (repo / "experiments/protocol.json").write_text('{"include_duration": false}')
    (repo / "experiments/final_runs.json").write_text('["previous_run"]\n')
    return repo


def invoke(repo, name, extra_env=None, args=()):
    env = {**os.environ, "PYTHON_BIN": Path(sys.executable).as_posix(), **(extra_env or {})}
    return subprocess.run(["bash", (repo / "tools" / name).as_posix(), *args], cwd=repo.parent,
                          env=env, capture_output=True, text=True)


def write_run(repo, experiment, stamp, dirty=False, python_version="test", protocol_path=None, run_root=None):
    run_id = f"{experiment}__{stamp}__test"
    folder = (run_root or repo / "results/runs") / run_id
    folder.mkdir(parents=True)
    protocol_path = protocol_path or repo / "experiments/protocol.json"
    (folder / "config.json").write_text(json.dumps({"experiment_id": experiment, "model": "dummy"}))
    (folder / "protocol.json").write_text(protocol_path.read_text())
    metadata = {"dirty_tree": dirty, "code_commit": "synthetic-test-commit", "seed": 42,
                "data_sha256": "data", "folds_sha256": "folds", "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                "versions": {"test": "1"}, "python_version": python_version}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    (folder / "fold_metrics.csv").write_text(
        "fold,mcc,fit_time_seconds\n" + "".join(f"{fold},0.1,1\n" for fold in range(1, 6)))
    return run_id


def test_accept_newest_eligible_run_and_skip_rejected(tool_repo):
    write_run(tool_repo, "sample", "20260101T000000Z")
    newest = write_run(tool_repo, "sample", "20260201T000000Z")
    write_run(tool_repo, "sample", "20260301T000000Z", dirty=True)
    (tool_repo / "results/runs/sample__20260401T000000Z__test").mkdir()
    result = invoke(tool_repo, "accept_runs.sh")
    assert result.returncode == 0, result.stderr
    assert json.loads((tool_repo / "experiments/final_runs.json").read_text()) == [newest]
    assert "clean working tree" in result.stderr
    assert "Skipping" in result.stderr


def test_no_eligible_runs_preserve_manifest(tool_repo):
    write_run(tool_repo, "sample", "20260101T000000Z", dirty=True)
    result = invoke(tool_repo, "accept_runs.sh")
    assert result.returncode != 0
    assert "No eligible runs" in result.stderr
    assert json.loads((tool_repo / "experiments/final_runs.json").read_text()) == ["previous_run"]


def test_incompatible_runs_preserve_manifest(tool_repo):
    write_run(tool_repo, "first", "20260101T000000Z")
    write_run(tool_repo, "second", "20260101T000000Z", python_version="different")
    result = invoke(tool_repo, "accept_runs.sh")
    assert result.returncode != 0
    assert "cannot be compared" in result.stderr
    assert json.loads((tool_repo / "experiments/final_runs.json").read_text()) == ["previous_run"]


def prepare_batch(repo):
    shutil.copytree(ROOT / "experiments/configs", repo / "experiments/configs")
    # Record the real CLI calls while avoiding expensive model fits in this test.
    (repo / "experiments/run.py").write_text(
        "import argparse, json, os\nfrom pathlib import Path\n"
        "parser = argparse.ArgumentParser()\nparser.add_argument('--config', required=True)\n"
        "parser.add_argument('--protocol')\nparser.add_argument('--output')\n"
        "args = parser.parse_args()\n"
        "with Path('calls.jsonl').open('a') as stream:\n"
        "    stream.write(json.dumps(args.config) + '\\n')\n"
        "with Path('options.jsonl').open('a') as stream:\n"
        "    stream.write(json.dumps({'protocol': args.protocol, 'output': args.output}) + '\\n')\n"
        "if Path(args.config).name == os.environ.get('FAIL_CONFIG'):\n"
        "    raise SystemExit(2)\n")


def test_batch_invokes_all_configs_from_other_directory(tool_repo):
    prepare_batch(tool_repo)
    result = invoke(tool_repo, "run_all_configs.sh")
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tool_repo / "calls.jsonl").read_text().splitlines()]
    expected = sorted(str(path.relative_to(tool_repo)) for path in (tool_repo / "experiments/configs").glob("*.json"))
    assert len(expected) == 12
    assert calls == expected
    assert json.loads((tool_repo / "experiments/final_runs.json").read_text()) == ["previous_run"]


def test_batch_stops_at_first_failure(tool_repo):
    prepare_batch(tool_repo)
    filenames = sorted(path.name for path in (tool_repo / "experiments/configs").glob("*.json"))
    result = invoke(tool_repo, "run_all_configs.sh", {"FAIL_CONFIG": filenames[1]})
    assert result.returncode == 2
    calls = [json.loads(line) for line in (tool_repo / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 2
    assert "All configurations completed" not in result.stdout


def test_batch_forwards_protocol_and_output_to_all_configs(tool_repo):
    prepare_batch(tool_repo)
    protocol = tool_repo / "alternate protocol.json"
    protocol.write_text('{"include_duration": true}')
    output = "results/runs/with duration"
    result = invoke(tool_repo, "run_all_configs.sh",
                    args=["--protocol", protocol.as_posix(), "--output", output])
    assert result.returncode == 0, result.stderr
    options = [json.loads(line) for line in (tool_repo / "options.jsonl").read_text().splitlines()]
    assert len(options) == 12
    assert all(item == {"protocol": protocol.as_posix(), "output": output} for item in options)


def test_accept_filters_selected_protocol_before_choosing_latest(tool_repo):
    alternate = tool_repo / "alternate protocol.json"
    alternate.write_text('{"include_duration": true}')
    original = write_run(tool_repo, "sample", "20260101T000000Z")
    duration = write_run(tool_repo, "sample", "20260201T000000Z", protocol_path=alternate)
    result = invoke(tool_repo, "accept_runs.sh")
    assert result.returncode == 0, result.stderr
    assert json.loads((tool_repo / "experiments/final_runs.json").read_text()) == [original]
    result = invoke(tool_repo, "accept_runs.sh", args=["--protocol", alternate.as_posix()])
    assert result.returncode == 0, result.stderr
    assert json.loads((tool_repo / "experiments/final_runs.json").read_text()) == [duration]


def test_accept_custom_runs_and_manifest_preserves_default(tool_repo):
    alternate = tool_repo / "alternate protocol.json"
    alternate.write_text('{"include_duration": true}')
    run_root = tool_repo / "results/runs/with duration"
    expected = write_run(tool_repo, "sample", "20260101T000000Z", protocol_path=alternate, run_root=run_root)
    manifest = tool_repo / "experiments/with duration/final_runs.json"
    result = invoke(tool_repo, "accept_runs.sh", args=["--protocol", alternate.as_posix(),
                    "--runs", run_root.as_posix(), "--manifest", manifest.as_posix()])
    assert result.returncode == 0, result.stderr
    assert json.loads(manifest.read_text()) == [expected]
    assert json.loads((tool_repo / "experiments/final_runs.json").read_text()) == ["previous_run"]


def test_no_matching_protocol_preserves_manifest(tool_repo):
    write_run(tool_repo, "sample", "20260101T000000Z")
    alternate = tool_repo / "alternate protocol.json"
    alternate.write_text('{"include_duration": true}')
    result = invoke(tool_repo, "accept_runs.sh", args=["--protocol", alternate.as_posix()])
    assert result.returncode != 0
    assert "No eligible runs" in result.stderr
    assert json.loads((tool_repo / "experiments/final_runs.json").read_text()) == ["previous_run"]
