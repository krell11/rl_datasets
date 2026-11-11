# Add a dataset format

A format is a `Format` subclass with `name`, `detect`, `load`, and `dump`. The four built-ins in `src/dataset_types/` are the pattern. `Catalog` calls `detect` in this order: `openenv`, `harbor`, `nemo-gym`, `verifiers`, then any name registered after those. `--from` skips detection.

## Where the file goes

Put `src/dataset_types/<name>.py` in this repo. `dataset_types.load` imports every module next to its `__init__.py` except names that start with `_`. End the module with `catalog.register(YourFormat())`.

A format that lives outside the repo stays unregistered until the CLI imports it:

```bash
rldata convert INPUT --to your-name --output DIR --import your_module
```

`register` raises if `name` is empty or already taken. Built-in names are `harbor`, `verifiers`, `openenv`, and `nemo-gym`.

## The task you read and write

```python
Task(
    id="math/add",
    messages=[Message("user", "What is 2 + 2?")],
    answer="4",          # None when the source has no answer
    split="train",
    metadata={"category": "math"},
    files=[File("environment/Dockerfile", text="FROM python:3.12-slim\n")],
)
```

`messages` is the prompt. One user message is the Harbor `instruction.md` text. `metadata` holds everything else that should survive a round-trip (`info`, grader fields, the parsed `task.toml`). `files` are paths relative to one task, such as `tests/test.sh`.

`File.write_into(root)` writes that path under `root` and refuses `..` or an absolute path. Use it. Do not open a destination with a raw join.

`Message.parse` accepts a string, one chat object, or a list of `{role, content}`. `task.instruction` is the single text Harbor shows an agent.

Leave `answer` empty when the source has no answer field. Copy a verifier or Dockerfile onto `files`. Do not generate `tests/test.sh` or a container for a row that did not have one.

## Class

```python
from pathlib import Path

from rldatasets import File, Message, Task
from rldatasets.task import catalog


class CsvQA:
    name = "csv-qa"

    def detect(self, path: Path) -> bool:
        return path.is_file() and path.suffix == ".csv"

    def load(self, path: Path) -> list[Task]:
        tasks: list[Task] = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip() or index == 0:
                continue
            question, answer = line.split(",", 1)
            tasks.append(Task(str(index), [Message("user", question)], answer=answer))
        return tasks

    def dump(self, tasks: list[Task], path: Path) -> None:
        rows = ["question,answer"]
        for task in tasks:
            rows.append(f"{task.instruction.strip()},{task.answer or ''}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")


catalog.register(CsvQA())
```

Subclass `Format` when you want the JSONL and file helpers (`rows`, `write_jsonl`, `read_files`, `files_root`). `Verifiers` and `NemoGym` in `src/dataset_types/` show that shape. A duck-typed object with the three methods is enough, as in the example above. `register` stores the instance.

## Check

```bash
rldata convert sample.csv --from csv-qa --to harbor --output ./tasks
```

`detect` should return true only for your files, so a Harbor tree or a NeMo JSONL is not claimed by the new format. Cover `load`, `dump`, and a path that tries to leave the output directory.
