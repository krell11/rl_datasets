# rldatasets

Convert LLM RL environment datasets between Harbor, Verifiers, OpenEnv, and NeMo Gym.

```bash
pip install -e .
rldata convert ./tasks --to verifiers --output ./vf
rldata convert train.jsonl --from verifiers --to nemo-gym --output ./nemo
```

```python
from rldatasets import convert

convert("train.jsonl", "tasks", to="harbor")
```

A task is an id, chat messages, an optional answer, metadata, and files. `tests/test.sh` and Dockerfiles are copied with the task. The converter does not invent a grader or a container when the source did not have one.

| Format | What it reads and writes |
| --- | --- |
| `harbor` | `tasks/<id>/instruction.md`, `task.toml`, and the rest of the task directory |
| `verifiers` | JSONL with `prompt`, `answer`, `info`, `task`, `example_id` |
| `nemo-gym` | JSONL with `responses_create_params` and `expected_answer` |
| `openenv` | A Harbor task tree plus `openenv.yaml` |

OpenEnv is chosen when `openenv.yaml` is present. Otherwise a directory with `instruction.md` and `task.toml` is Harbor. JSONL with `responses_create_params` is NeMo Gym. JSONL with `prompt` or `question` is Verifiers.

## Add a format

The steps for a new layout are in [.agents/add-format.md](.agents/add-format.md). Put the module in `src/dataset_types/` so it loads with the built-ins, or pass it with `--import`.

```python
from rldatasets import Format, register

class CsvQA(Format):
    name = "csv-qa"

    def detect(self, path):
        return path.suffix == ".csv"

    def load(self, path):
        ...

    def dump(self, tasks, path):
        ...

register(CsvQA())
```

```bash
rldata convert tasks/ --to csv-qa --output out.csv --import myformats
```
