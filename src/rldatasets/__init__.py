from rldatasets.task import Catalog, File, Format, Message, Result, Task, catalog

__all__ = [
    "Catalog",
    "File",
    "Format",
    "Message",
    "Result",
    "Task",
    "catalog",
    "convert",
    "load",
    "register",
]

__version__ = "0.1.0"


def register(fmt: Format) -> Format:
    return catalog.register(fmt)


def load(path: str, fmt: str | None = None) -> list[Task]:
    return catalog.load(path, fmt)


def convert(source: str, destination: str, to: str, source_format: str | None = None) -> Result:
    return catalog.convert(source, destination, to=to, source_format=source_format)


def _load_dataset_types() -> None:
    import dataset_types

    dataset_types.load()


_load_dataset_types()
