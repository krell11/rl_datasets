import json
import sys
from pathlib import Path

import pytest

from rldatasets import File, Format, Message, Task, catalog, convert, load, register
from rldatasets.cli import main

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def write_harbor(root: Path) -> None:
    task = root / "tasks" / "add"
    task.mkdir(parents=True)
    (task / "instruction.md").write_bytes(b"What is 2 + 2?\n")
    (task / "task.toml").write_bytes(
        b'''schema_version = "1.4"

[task]
name = "math/add"
description = "Add two numbers"

[metadata]
answer = "4"
category = "math"

[environment]
cpus = 1
'''
    )
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_bytes(b"FROM python:3.12-slim\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_bytes(b"#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")


def write_verifiers(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
        "answer": "4",
        "task": "math",
        "info": json.dumps({"options": ["4", "5"], "grading_mode": "strict"}),
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def write_nemo(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "id": "add",
        "expected_answer": "4",
        "grading_mode": "strict",
        "options": ["4", "5"],
        "responses_create_params": {
            "input": [
                {"role": "developer", "content": "Be brief."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            "tools": [{"type": "function", "name": "calc"}],
        },
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_detects_each_builtin(tmp_path: Path) -> None:
    harbor = tmp_path / "harbor"
    write_harbor(harbor)
    verifiers = tmp_path / "train.jsonl"
    write_verifiers(verifiers)
    nemo = tmp_path / "nemo.jsonl"
    write_nemo(nemo)
    opened = tmp_path / "opened"
    write_harbor(opened)
    (opened / "openenv.yaml").write_text("spec_version: 1\nname: opened\n", encoding="utf-8")

    assert catalog.detect(harbor).name == "harbor"
    assert catalog.detect(verifiers).name == "verifiers"
    assert catalog.detect(nemo).name == "nemo-gym"
    assert catalog.detect(opened).name == "openenv"


def test_harbor_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "src"
    write_harbor(source)
    destination = tmp_path / "out"
    convert(source, destination, to="harbor")

    copied = destination / "tasks" / "math-add"
    assert (copied / "instruction.md").read_text(encoding="utf-8") == "What is 2 + 2?\n"
    assert (copied / "environment" / "Dockerfile").read_bytes() == b"FROM python:3.12-slim\n"
    assert (copied / "tests" / "test.sh").read_bytes().startswith(b"#!/bin/bash\n")
    document = tomllib.loads((copied / "task.toml").read_text(encoding="utf-8"))
    assert document["task"]["name"] == "math/add"
    assert document["metadata"]["answer"] == "4"
    assert document["metadata"]["category"] == "math"
    assert document["environment"]["cpus"] == 1


def test_verifiers_nemo_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "nemo.jsonl"
    write_nemo(source)
    direct = tmp_path / "direct"
    convert(source, direct, to="nemo-gym")
    kept = json.loads((direct / "data" / "train.jsonl").read_text(encoding="utf-8"))
    assert kept["responses_create_params"]["input"][0]["role"] == "developer"
    assert kept["responses_create_params"]["tools"][0]["name"] == "calc"
    assert kept["grading_mode"] == "strict"

    as_verifiers = tmp_path / "vf"
    convert(source, as_verifiers, to="verifiers")
    prompt = json.loads((as_verifiers / "data" / "train.jsonl").read_text(encoding="utf-8"))
    assert prompt["prompt"][0]["role"] == "system"
    assert prompt["answer"] == "4"

    restored = tmp_path / "back"
    convert(as_verifiers, restored, to="nemo-gym")
    row = json.loads((restored / "data" / "train.jsonl").read_text(encoding="utf-8"))
    assert row["expected_answer"] == "4"
    assert row["grading_mode"] == "strict"
    assert row["options"] == ["4", "5"]
    assert row["responses_create_params"]["input"][1]["content"] == "What is 2 + 2?"


def test_harbor_files_are_copied_not_rewritten(tmp_path: Path) -> None:
    source = tmp_path / "src"
    write_harbor(source)
    destination = tmp_path / "vf"
    convert(source, destination, to="verifiers")
    script = destination / "files" / "train" / "0" / "tests" / "test.sh"
    assert script.read_bytes() == b"#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    assert list(destination.rglob("gold.txt")) == []


def test_rows_do_not_gain_a_verifier(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    write_verifiers(source)
    destination = tmp_path / "harbor"
    convert(source, destination, to="harbor")
    assert list(destination.rglob("test.sh")) == []
    assert list(destination.rglob("Dockerfile")) == []
    document = tomllib.loads(next(destination.rglob("task.toml")).read_text(encoding="utf-8"))
    assert document["metadata"]["answer"] == "4"
    assert "What is 2 + 2?" in next(destination.rglob("instruction.md")).read_text(encoding="utf-8")


def test_openenv_writes_manifest(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    write_verifiers(source)
    destination = tmp_path / "env"
    convert(source, destination, to="openenv")
    manifest = (destination / "openenv.yaml").read_text(encoding="utf-8")
    assert "spec_version: 1" in manifest
    assert (destination / "tasks" / "train-0" / "instruction.md").read_text(encoding="utf-8") == "What is 2 + 2?"


def test_refuses_paths_outside_the_dataset(tmp_path: Path) -> None:
    class Sneaky(Format):
        name = "sneaky"

        def detect(self, path: Path) -> bool:
            return False

        def load(self, path: Path) -> list[Task]:
            return [Task("x", [Message("user", "hi")], files=[File("../outside.txt", text="nope")])]

        def dump(self, tasks: list[Task], path: Path) -> None:
            return None

    if "sneaky" not in catalog.names():
        register(Sneaky())
    source = tmp_path / "train.jsonl"
    write_verifiers(source)
    with pytest.raises(ValueError, match="outside the dataset"):
        convert(source, tmp_path / "dest", to="harbor", source_format="sneaky")
    assert not (tmp_path / "outside.txt").exists()


def test_registered_format_and_cli_import(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = tmp_path / "samplefmt.py"
    module.write_text(
        "\n".join(
            [
                "from rldatasets import Format, register",
                "class Sample(Format):",
                "    name = 'sample'",
                "    def detect(self, path):",
                "        return False",
                "    def load(self, path):",
                "        return []",
                "    def dump(self, tasks, path):",
                "        path.mkdir(parents=True, exist_ok=True)",
                "        (path / 'ids.txt').write_text('\\n'.join(task.id for task in tasks), encoding='utf-8')",
                "register(Sample())",
                "",
            ]
        ),
        encoding="utf-8",
    )
    source = tmp_path / "src"
    write_harbor(source)
    destination = tmp_path / "out"
    sys.path.insert(0, str(tmp_path))
    try:
        code = main(
            [
                "convert",
                str(source),
                "--to",
                "sample",
                "--output",
                str(destination),
                "--import",
                "samplefmt",
            ]
        )
    finally:
        sys.path.remove(str(tmp_path))
    assert code == 0
    assert (destination / "ids.txt").read_text(encoding="utf-8") == "math/add"
    printed = capsys.readouterr().out
    assert "|_| \\_\\_____|____/" in printed
    assert "harbor  -->  sample" in printed


def test_load_uses_question_column(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"question": "What is 3 + 3?", "answer": "6"}) + "\n", encoding="utf-8")
    tasks = load(path)
    assert tasks[0].messages[0].content == "What is 3 + 3?"
    assert tasks[0].answer == "6"
