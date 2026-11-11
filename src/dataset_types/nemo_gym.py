from __future__ import annotations

from pathlib import Path
from typing import Any

from rldatasets.task import Format, Message, Task, catalog


class NemoGym(Format):
    name = "nemo-gym"

    def detect(self, path: Path) -> bool:
        row = self._peek(path)
        return bool(row) and "responses_create_params" in row

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
        params = dict(row.get("responses_create_params") or {})
        messages = Message.parse(params.pop("input", None))
        if not messages and row.get("question") is not None:
            messages = [Message("user", str(row["question"]))]
        metadata: dict[str, Any] = {}
        if params:
            metadata["responses_create_params"] = params
        for key, value in row.items():
            if key in {"responses_create_params", "expected_answer", "question", "id"}:
                continue
            metadata[key] = value
        answer = row.get("expected_answer")
        sidecar = root / "files" / split / str(index)
        return Task(
            id=str(row.get("id") if row.get("id") is not None else f"{split}-{index}"),
            split=split,
            messages=messages,
            answer=None if answer is None else str(answer),
            metadata=metadata,
            files=self.read_files(sidecar) if sidecar.is_dir() else [],
        )

    def _row(self, task: Task, index: int, root: Path, split: str) -> dict[str, Any]:
        row = {key: value for key, value in task.metadata.items() if key != "responses_create_params"}
        params = dict(task.metadata.get("responses_create_params") or {})
        params["input"] = [message.export() for message in task.messages]
        row["responses_create_params"] = params
        row["id"] = task.id
        if task.answer is not None:
            row["expected_answer"] = task.answer
        user = next((message.content for message in reversed(task.messages) if message.role in {"user", "human"}), "")
        if not user and task.messages:
            user = task.messages[-1].content
        row["question"] = user
        sidecar = root / "files" / split / str(index)
        for asset in task.files:
            asset.write_into(sidecar)
        return row


catalog.register(NemoGym())
