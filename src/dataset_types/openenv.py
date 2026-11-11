from __future__ import annotations

import re
from pathlib import Path

from dataset_types.harbor import Harbor
from rldatasets.task import Format, Task, catalog

_NAME = re.compile(r"[A-Za-z0-9._/-]+")


class OpenEnv(Format):
    name = "openenv"

    def __init__(self) -> None:
        self._harbor = Harbor()

    def detect(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        return (path / "openenv.yaml").is_file() or (path / "openenv.yml").is_file()

    def load(self, path: Path) -> list[Task]:
        return self._harbor.load(path)

    def dump(self, tasks: list[Task], path: Path) -> None:
        self._harbor.dump(tasks, path)
        name = path.name
        if not _NAME.fullmatch(name):
            name = '"' + name.replace("\\", "\\\\").replace('"', '\\"') + '"'
        text = (
            "spec_version: 1\n"
            f"name: {name}\n"
            "type: space\n"
            "runtime: fastapi\n"
            "app: server:app\n"
            "port: 8000\n"
        )
        path.mkdir(parents=True, exist_ok=True)
        (path / "openenv.yaml").write_bytes(text.encode("utf-8"))


catalog.register(OpenEnv())
