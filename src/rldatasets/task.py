from __future__ import annotations

import datetime as dt
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


SKIP_DIRS = {".git", ".hg", "__pycache__", "node_modules", ".venv", "venv"}
ROW_SUFFIXES = (".jsonl", ".json")
SPLITS = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
    "eval": "test",
    "evaluation": "test",
    "example": "example",
    "examples": "example",
}


def slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower()).strip("-._")
    return (text or "task")[:80]


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    return str(value)


def safe_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Refusing to write outside the dataset: {relative}")
    destination = (root / rel).resolve()
    base = root.resolve()
    if destination != base and base not in destination.parents:
        raise ValueError(f"Refusing to write outside the dataset: {relative}")
    return destination


@dataclass
class Message:
    role: str
    content: str

    def export(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    def export_verifiers(self) -> dict[str, str]:
        role = "system" if self.role == "developer" else self.role
        return {"role": role, "content": self.content}

    @classmethod
    def parse(cls, value: Any) -> list[Message]:
        if value is None:
            return []
        if isinstance(value, str):
            return [cls("user", value)]
        if isinstance(value, dict):
            return cls.parse([value])
        if isinstance(value, list):
            messages: list[Message] = []
            for item in value:
                if isinstance(item, str):
                    messages.append(cls("user", item))
                elif isinstance(item, dict):
                    messages.append(cls(str(item.get("role") or "user"), cls.text(item.get("content", ""))))
            return messages
        return [cls("user", str(value))]

    @staticmethod
    def text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [Message.text(part) if not isinstance(part, dict) else str(part.get("text") or "") for part in content]
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            if "text" in content:
                return str(content.get("text") or "")
            return Message.text(content.get("content"))
        return str(content)


@dataclass
class File:
    path: str
    text: str | None = None
    data: bytes | None = None

    def write_into(self, root: Path) -> None:
        destination = safe_path(root, self.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.data if self.data is not None else (self.text or "").encode("utf-8")
        destination.write_bytes(payload)


@dataclass
class Task:
    id: str
    messages: list[Message] = field(default_factory=list)
    answer: str | None = None
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)
    files: list[File] = field(default_factory=list)

    @property
    def instruction(self) -> str:
        if not self.messages:
            return ""
        if len(self.messages) == 1:
            return self.messages[0].content
        return "\n\n".join(f"[{message.role}]\n{message.content}" for message in self.messages)


@dataclass
class Result:
    source_format: str
    target_format: str
    output: Path
    tasks: list[Task]

    @property
    def notes(self) -> list[str]:
        notes: list[str] = []
        missing = sum(task.answer is None for task in self.tasks)
        if missing:
            notes.append(f"{missing} tasks have no answer.")
        if self.target_format in {"verifiers", "nemo-gym"} and any(task.files for task in self.tasks):
            notes.append("Sandbox files were copied under files/.")
        return notes

    def render(self) -> str:
        count = len(self.tasks)
        noun = "task" if count == 1 else "tasks"
        lines = [
            f"Converted {count} {noun} from {self.source_format} to {self.target_format}.",
            f"Output: {self.output}",
        ]
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)


class Format(ABC):
    """One dataset layout. Subclasses live in dataset_types."""

    name: str

    @abstractmethod
    def detect(self, path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> list[Task]:
        raise NotImplementedError

    @abstractmethod
    def dump(self, tasks: list[Task], path: Path) -> None:
        raise NotImplementedError

    def rows(self, path: Path) -> list[tuple[str, dict[str, Any]]]:
        files = self._row_files(path)
        if not files:
            raise ValueError(f"No JSON task file in {path}")
        loaded: list[tuple[str, dict[str, Any]]] = []
        for file in files:
            loaded.extend(self._read_file(file))
        return loaded

    def write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(plain(row), ensure_ascii=False) for row in rows]
        text = "\n".join(lines)
        if text:
            text += "\n"
        path.write_bytes(text.encode("utf-8"))

    def read_files(self, root: Path, skip: set[str] | None = None) -> list[File]:
        if not root.is_dir():
            return []
        skipped = skip or set()
        base = root.resolve()
        files: list[File] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if any(part in SKIP_DIRS for part in relative.parts):
                continue
            rel = relative.as_posix()
            if rel in skipped:
                continue
            try:
                path.resolve().relative_to(base)
            except ValueError:
                continue
            blob = path.read_bytes()
            if b"\0" in blob:
                files.append(File(rel, data=blob))
            else:
                files.append(File(rel, text=blob.decode("utf-8")))
        return files

    def files_root(self, path: Path) -> Path:
        if path.is_dir():
            return path
        if path.parent.name == "data":
            return path.parent.parent
        return path.parent

    def _row_files(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        if not path.is_dir():
            return []
        roots = [path]
        data = path / "data"
        if data.is_dir():
            roots.append(data)
        found: list[Path] = []
        for root in roots:
            for file in sorted(root.iterdir()):
                if file.is_file() and file.name.lower().endswith(ROW_SUFFIXES):
                    found.append(file)
        return found

    def _read_file(self, path: Path) -> list[tuple[str, dict[str, Any]]]:
        raw = path.read_text(encoding="utf-8-sig")
        split = SPLITS.get(path.name.split(".")[0].lower(), "train")
        if path.suffix.lower() == ".jsonl":
            return [(split, row) for row in self._lines(path, raw)]
        stripped = raw.strip()
        if not stripped:
            return []
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return [(split, row) for row in self._lines(path, raw)]
        return list(self._payload(payload, split, path))

    def _lines(self, path: Path, raw: str) -> Iterator[dict[str, Any]]:
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number} must be a JSON object")
            yield row

    def _payload(self, payload: Any, split: str, path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
        if isinstance(payload, list):
            for index, item in enumerate(payload, start=1):
                if not isinstance(item, dict):
                    raise ValueError(f"{path} row {index} must be a JSON object")
                yield split, item
            return
        if isinstance(payload, dict):
            if _looks_like_row(payload):
                yield split, payload
                return
            yielded = False
            for key, value in payload.items():
                if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
                    continue
                yielded = True
                if key in {"data", "rows", "examples", "tasks"}:
                    row_split = split
                else:
                    row_split = SPLITS.get(str(key).lower(), str(key))
                for item in value:
                    yield row_split, item
            if yielded:
                return
        raise ValueError(f"{path} must be a JSON object, a list of objects, or JSONL")

    def _peek(self, path: Path) -> dict[str, Any] | None:
        try:
            found = self.rows(path)
        except (OSError, UnicodeError, ValueError):
            return None
        if not found:
            return None
        return found[0][1]


def _looks_like_row(payload: dict[str, Any]) -> bool:
    markers = {"prompt", "question", "answer", "instruction", "messages", "responses_create_params", "task_id", "expected_answer"}
    return any(key in payload for key in markers)


class Catalog:
    """Registered formats and the convert path between them."""

    priority = ("openenv", "harbor", "nemo-gym", "verifiers")

    def __init__(self) -> None:
        self._formats: dict[str, Format] = {}

    def register(self, fmt: Format) -> Format:
        if not getattr(fmt, "name", ""):
            raise ValueError("Format name is required.")
        if fmt.name in self._formats:
            raise ValueError(f"Format {fmt.name!r} is already registered.")
        self._formats[fmt.name] = fmt
        return fmt

    def names(self) -> list[str]:
        ordered = [name for name in self.priority if name in self._formats]
        ordered.extend(name for name in self._formats if name not in ordered)
        return ordered

    def get(self, name: str) -> Format:
        try:
            return self._formats[name]
        except KeyError:
            known = ", ".join(self.names()) or "none"
            raise ValueError(f"Unknown format {name!r}. Known formats: {known}.") from None

    def detect(self, path: Path) -> Format:
        for name in self.names():
            if self._formats[name].detect(path):
                return self._formats[name]
        known = ", ".join(self.names())
        raise ValueError(f"Could not tell which format {path} is. Pass --from. Known formats: {known}.")

    def load(self, path: str | Path, fmt: str | None = None) -> list[Task]:
        source = Path(path)
        if not source.exists():
            raise ValueError(f"Dataset not found: {source}")
        reader = self.get(fmt) if fmt else self.detect(source)
        return reader.load(source)

    def convert(
        self,
        source: str | Path,
        destination: str | Path,
        to: str,
        source_format: str | None = None,
    ) -> Result:
        src = Path(source)
        dest = Path(destination)
        if src.resolve() == dest.resolve():
            raise ValueError("Source and output are the same path.")
        if not src.exists():
            raise ValueError(f"Dataset not found: {src}")
        origin = self.get(source_format) if source_format else self.detect(src)
        tasks = origin.load(src)
        target = self.get(to)
        target.dump(tasks, dest)
        return Result(origin.name, target.name, dest, tasks)


catalog = Catalog()
