from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rldatasets.task import Format, Message, Task, catalog, plain


class Verifiers(Format):
    name = "verifiers"

    def detect(self, path: Path) -> bool:
        row = self._peek(path)
        if not row or "responses_create_params" in row:
            return False
        return "prompt" in row or "question" in row

    def load(self, path: Path) -> list[Task]:
        root = self.files_root(path)
        counts: dict[str, int] = {}
        tasks: list[Task] = []
        for split, row in self.rows(path):
            index = counts.get(split, 0)
            counts[split] = index + 1
            tasks.append(self._task(split, row, root, index))
        return tasks

    def dump(self, tasks: list[Task], path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        groups: dict[str, list[Task]] = {}
        for task in tasks:
            groups.setdefault(task.split or "train", []).append(task)
        for split, group in groups.items():
            rows = [self._row(task, index, path, split) for index, task in enumerate(group)]
            self.write_jsonl(path / "data" / f"{split}.jsonl", rows)

    def _task(self, split: str, row: dict[str, Any], root: Path, index: int) -> Task:
        prompt = row.get("prompt")
        if prompt is None and "question" in row:
            prompt = row["question"]
        info = self._info(row.get("info"))
        if isinstance(row.get("task"), str):
            info.setdefault("task", row["task"])
        reserved = {"prompt", "question", "answer", "info", "example_id", "task"}
        for key, value in row.items():
            if key not in reserved and key not in info:
                info[key] = value
        if "id" in info and info["id"] is not None:
            task_id = str(info.pop("id"))
        elif row.get("example_id") is not None:
            task_id = str(row["example_id"])
        else:
            task_id = f"{split}-{index}"
        answer = row.get("answer")
        sidecar = root / "files" / split / str(index)
        return Task(
            id=task_id,
            split=split,
            messages=Message.parse(prompt),
            answer=None if answer is None else str(answer),
            metadata=info,
            files=self.read_files(sidecar) if sidecar.is_dir() else [],
        )

    def _row(self, task: Task, index: int, root: Path, split: str) -> dict[str, Any]:
        info = dict(task.metadata)
        info["id"] = task.id
        group = info.get("task") if isinstance(info.get("task"), str) else "default"
        for asset in task.files:
            asset.write_into(root / "files" / split / str(index))
        row: dict[str, Any] = {
            "example_id": index,
            "task": group,
            "prompt": [message.export_verifiers() for message in task.messages],
            "info": json.dumps(plain(info), ensure_ascii=False),
        }
        if task.answer is not None:
            row["answer"] = task.answer
        return row

    def _info(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {"text": value}
            if isinstance(parsed, dict):
                return parsed
        return {}


catalog.register(Verifiers())
