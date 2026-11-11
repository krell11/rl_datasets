from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from rldatasets.task import SKIP_DIRS, File, Format, Message, Task, catalog, plain, slug

import tomllib

_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
_ANSWER_KEYS = ("answer", "expected_answer", "gold_answer", "gold", "target")
_SKIP_FILES = {"instruction.md", "task.toml"}


class Toml:
    @staticmethod
    def read(text: str) -> dict[str, Any]:
        try:
            loaded = tomllib.loads(text)
        except Exception as exc:
            raise ValueError(f"Invalid task.toml: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError("task.toml must be a table")
        return plain(loaded)

    @staticmethod
    def write(data: dict[str, Any]) -> str:
        cleaned = Toml._drop_none(data)
        lines: list[str] = []
        scalars = {key: value for key, value in cleaned.items() if not isinstance(value, dict)}
        tables = {key: value for key, value in cleaned.items() if isinstance(value, dict)}
        for key, value in scalars.items():
            lines.append(f"{Toml._key(key)} = {Toml._value(value)}")
        if scalars and tables:
            lines.append("")
        for index, (key, value) in enumerate(tables.items()):
            if index:
                lines.append("")
            Toml._table(lines, Toml._key(key), value)
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _table(lines: list[str], header: str, data: dict[str, Any]) -> None:
        scalars = {key: value for key, value in data.items() if not isinstance(value, dict)}
        tables = {key: value for key, value in data.items() if isinstance(value, dict)}
        if scalars or not tables:
            lines.append(f"[{header}]")
            for key, value in scalars.items():
                lines.append(f"{Toml._key(key)} = {Toml._value(value)}")
        for key, value in tables.items():
            if scalars:
                lines.append("")
            Toml._table(lines, f"{header}.{Toml._key(key)}", value)
            scalars = True

    @staticmethod
    def _drop_none(value: Any) -> Any:
        if isinstance(value, dict):
            kept: dict[str, Any] = {}
            for key, item in value.items():
                if item is None:
                    continue
                if isinstance(item, dict):
                    nested = Toml._drop_none(item)
                    if nested:
                        kept[str(key)] = nested
                elif isinstance(item, list):
                    kept[str(key)] = [Toml._drop_none(entry) if isinstance(entry, dict) else entry for entry in item]
                else:
                    kept[str(key)] = item
            return kept
        return value

    @staticmethod
    def _key(key: str) -> str:
        if _BARE_KEY.fullmatch(key):
            return key
        return Toml._string(key)

    @staticmethod
    def _value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if value.is_integer() and abs(value) < 1e15:
                return f"{value:.1f}"
            return repr(value)
        if isinstance(value, str):
            return Toml._string(value)
        if isinstance(value, list):
            return "[" + ", ".join(Toml._value(item) for item in value) + "]"
        if isinstance(value, dict):
            inner = ", ".join(f"{Toml._key(str(key))} = {Toml._value(item)}" for key, item in value.items())
            return "{ " + inner + " }"
        return Toml._string(str(value))

    @staticmethod
    def _string(value: str) -> str:
        if "\n" in value or "\r" in value:
            escaped = value.replace("\\", "\\\\").replace('"""', '""\\"')
            return '"""\n' + escaped + '\n"""'
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\t", "\\t")
            .replace("\b", "\\b")
            .replace("\f", "\\f")
        )
        return f'"{escaped}"'


class Harbor(Format):
    name = "harbor"

    def detect(self, path: Path) -> bool:
        return bool(self.task_dirs(path))

    def load(self, path: Path) -> list[Task]:
        tasks = [self._read_dir(task_dir) for task_dir in self.task_dirs(path)]
        seen: dict[str, int] = {}
        for task in tasks:
            count = seen.get(task.id, 0) + 1
            seen[task.id] = count
            if count > 1:
                task.id = f"{task.id}-{count}"
        return tasks

    def dump(self, tasks: list[Task], path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        used: set[str] = set()
        for task in tasks:
            name = slug(task.id)
            folder_name = name
            index = 2
            while folder_name in used:
                folder_name = f"{name[: 80 - len(str(index)) - 1]}-{index}"
                index += 1
            used.add(folder_name)
            self._write_dir(task, path / "tasks" / folder_name)

    def task_dirs(self, path: Path) -> list[Path]:
        if not path.is_dir():
            return []
        if (path / "task.toml").is_file() and (path / "instruction.md").is_file():
            return [path]
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIRS and not name.startswith("."))
            if "task.toml" in filenames and "instruction.md" in filenames:
                found.append(Path(dirpath))
                dirnames[:] = []
        return found

    def _read_dir(self, task_dir: Path) -> Task:
        try:
            document = Toml.read((task_dir / "task.toml").read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"Could not read {task_dir / 'task.toml'}: {exc}") from exc
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
        task_table = document.get("task") if isinstance(document.get("task"), dict) else {}
        meta = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        return Task(
            id=str(task_table.get("name") or task_dir.name),
            messages=self._messages(instruction, meta),
            answer=self._answer(meta),
            split=str(meta.get("split") or "train"),
            metadata=document,
            files=self.read_files(task_dir, _SKIP_FILES),
        )

    def _write_dir(self, task: Task, task_dir: Path) -> None:
        task_dir.mkdir(parents=True, exist_ok=True)
        for asset in task.files:
            if Path(asset.path).name in _SKIP_FILES and "/" not in asset.path:
                continue
            asset.write_into(task_dir)
        (task_dir / "instruction.md").write_bytes(task.instruction.encode("utf-8"))
        (task_dir / "task.toml").write_bytes(Toml.write(self._document(task)).encode("utf-8"))

    def _messages(self, instruction: str, meta: dict[str, Any]) -> list[Message]:
        stored = meta.pop("rldatasets_messages", None) if isinstance(meta, dict) else None
        if isinstance(stored, str):
            try:
                parsed = json.loads(stored)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return Message.parse(parsed)
        return [Message("user", instruction)]

    def _answer(self, meta: dict[str, Any]) -> str | None:
        for key in _ANSWER_KEYS:
            if key in meta and meta[key] is not None:
                value = meta[key]
                return value if isinstance(value, str) else json.dumps(plain(value), ensure_ascii=False)
        return None

    def _document(self, task: Task) -> dict[str, Any]:
        if self._is_document(task.metadata):
            document = json.loads(json.dumps(plain(task.metadata)))
        else:
            document = {
                "schema_version": "1.4",
                "task": {"name": task.id},
                "metadata": dict(task.metadata),
            }
        task_table = document.get("task")
        if not isinstance(task_table, dict):
            task_table = {}
            document["task"] = task_table
        task_table["name"] = task.id
        meta = document.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
            document["metadata"] = meta
        meta["split"] = task.split
        if task.answer is not None:
            meta["answer"] = task.answer
        if len(task.messages) != 1 or task.messages[0].role not in {"user", "human"}:
            meta["rldatasets_messages"] = json.dumps(
                [message.export() for message in task.messages],
                ensure_ascii=False,
            )
        else:
            meta.pop("rldatasets_messages", None)
        document["schema_version"] = str(document.get("schema_version") or "1.4")
        return document

    def _is_document(self, metadata: dict[str, Any]) -> bool:
        if "schema_version" in metadata:
            return True
        return isinstance(metadata.get("verifier"), dict) or isinstance(metadata.get("environment"), dict)


catalog.register(Harbor())
