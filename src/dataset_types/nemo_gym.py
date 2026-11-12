from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset_types.harbor import Harbor, Toml
from rldatasets.task import Format, Message, Task, catalog, slug

_GYM_KEYS = ("type", "license", "allowed_agents", "domain", "verified", "source")
_SPLIT_TO_GYM = {"train": "train", "validation": "validation", "test": "benchmark", "example": "example"}
_GYM_TO_SPLIT = {"train": "train", "validation": "validation", "benchmark": "test", "example": "example"}


class NemoGym(Format):
    name = "nemo-gym"

    def __init__(self) -> None:
        self._harbor = Harbor()

    def detect(self, path: Path) -> bool:
        if self._gym_document(path) and self._harbor.task_dirs(path):
            return True
        row = self._peek(path)
        return bool(row) and "responses_create_params" in row

    def load(self, path: Path) -> list[Task]:
        document = self._gym_document(path)
        if document and self._harbor.task_dirs(path):
            return self._load_folders(path, document)
        return self._load_rows(path)

    def dump(self, tasks: list[Task], path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self._folder_dataset(tasks):
            self._harbor.dump(tasks, path)
            body = Toml.write(self._dataset_document(tasks, path))
            if "[gym]" not in body:
                body = body.rstrip() + "\n\n[gym]\n"
            (path / "dataset.toml").write_bytes(body.encode("utf-8"))
            return
        self._harbor._write_dataset_files(tasks, path)
        groups: dict[str, list[Task]] = {}
        for task in tasks:
            groups.setdefault(task.split or "train", []).append(task)
        for split, group in groups.items():
            rows = [self._row(task, index, path, split) for index, task in enumerate(group)]
            self.write_jsonl(path / "data" / f"{split}.jsonl", rows)

    def _gym_document(self, path: Path) -> dict[str, Any] | None:
        file = path / "dataset.toml"
        if not path.is_dir() or not file.is_file():
            return None
        try:
            document = Toml.read(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            return None
        gym = document.get("gym")
        if not isinstance(gym, dict):
            return None
        return document

    def _load_rows(self, path: Path) -> list[Task]:
        root = self.files_root(path)
        counts: dict[str, int] = {}
        tasks: list[Task] = []
        for split, row in self.rows(path):
            index = counts.get(split, 0)
            counts[split] = index + 1
            tasks.append(self._task(split, row, root, index))
        return tasks

    def _load_folders(self, path: Path, document: dict[str, Any]) -> list[Task]:
        tasks = self._harbor.load(path)
        gym = document.get("gym") if isinstance(document.get("gym"), dict) else {}
        split = _GYM_TO_SPLIT.get(str(gym.get("type") or ""))
        for task in tasks:
            files = None
            current = task.metadata.get("rldatasets_dataset")
            if isinstance(current, dict) and isinstance(current.get("files"), dict):
                files = current["files"]
            payload: dict[str, Any] = {
                "name": document.get("name"),
                "version": document.get("version"),
                "gym": dict(gym),
            }
            if files:
                payload["files"] = files
            if split:
                task.split = split
            task.metadata["rldatasets_dataset"] = payload
        return tasks

    def _folder_dataset(self, tasks: list[Task]) -> bool:
        return bool(tasks) and all(self._has_verifier(task) for task in tasks)

    def _has_verifier(self, task: Task) -> bool:
        paths = {Path(asset.path).as_posix() for asset in task.files}
        if "tests/test.sh" in paths or "tests/verifier.py" in paths:
            return True
        steps = {parts[1] for path in paths if (parts := path.split("/"))[:1] == ["steps"] and len(parts) > 1}
        if not steps:
            return False
        for step in steps:
            instruction = f"steps/{step}/instruction.md" in paths
            own_test = any(path.startswith(f"steps/{step}/tests/") and path.endswith("test.sh") for path in paths)
            if not instruction or not (own_test or "tests/test.sh" in paths):
                return False
        return True

    def _dataset_document(self, tasks: list[Task], path: Path) -> dict[str, Any]:
        stored: dict[str, Any] = {}
        for task in tasks:
            raw = task.metadata.get("rldatasets_dataset")
            if isinstance(raw, dict):
                stored = raw
                break
        gym_raw = stored.get("gym") if isinstance(stored.get("gym"), dict) else {}
        gym = {key: gym_raw[key] for key in _GYM_KEYS if key in gym_raw and gym_raw[key] is not None}
        splits = {task.split or "train" for task in tasks}
        if "type" not in gym and len(splits) == 1:
            mapped = _SPLIT_TO_GYM.get(next(iter(splits)))
            if mapped:
                gym["type"] = mapped
        name = stored.get("name") if isinstance(stored.get("name"), str) and stored.get("name") else f"local/{slug(path.name)}"
        version = stored.get("version") if isinstance(stored.get("version"), str) and stored.get("version") else "0.1.0"
        return {"name": name, "version": version, "gym": gym}

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
        row = {
            key: value
            for key, value in task.metadata.items()
            if key not in {"responses_create_params", "rldatasets_dataset"}
        }
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
