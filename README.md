# rldatasets

Convert LLM RL environment datasets between Harbor, Verifiers, OpenEnv, and NeMo Gym.

Harbor task folders are the shared dataset. OpenEnv serves that tree with `openenv harbor serve --dataset <path>`. NeMo Gym runs the same folders on its harbor environment server, and uses JSONL when every row shares one verifier. Verifiers stores one row per example (`prompt`, `answer`, `info`, `task`) for a Python rubric.

```bash
pip install -e .
rldata convert ./tasks --to verifiers --output ./vf
rldata convert train.jsonl --from verifiers --to nemo-gym --output ./nemo
rldata merge ./tasks train.jsonl --to harbor --output ./merged
```

```python
from rldatasets import convert, merge

convert("train.jsonl", "tasks", to="harbor")
merge(["./tasks", "train.jsonl"], "merged", to="harbor")
```

`merge` reads each dataset in the order given and writes one dataset in `--to`. Tasks keep their own files, including a Dockerfile or `tests/test.sh`. A repeated task id gains a numeric suffix. When two sources disagree on `verifier.py` or `task_data.py`, the first copy is the one that is written.

A task is an id, chat messages, an optional answer, metadata, and files. `tests/test.sh` and Dockerfiles are copied with the task. The converter does not invent a grader or a container when the source did not have one.

| Format | What it reads and writes |
| --- | --- |
| `harbor` | `tasks/<id>/` with `task.toml`, `instruction.md` or `steps/`, plus `environment/`, `tests/`, and `solution/` |
| `verifiers` | JSONL with `prompt`, `answer`, `info` (a JSON object), `task`, and `example_id` |
| `nemo-gym` | JSONL with `responses_create_params` and `expected_answer`. When every task already has `tests/test.sh`, `tests/verifier.py`, or per-step tests, a Harbor task tree plus `dataset.toml` with `[gym]` |
| `openenv` | The same Harbor task tree, marked with `openenv.yaml` (`type: harbor-dataset`) so `openenv harbor serve --dataset <path>` can serve it |

OpenEnv is chosen when `openenv.yaml` is present. A directory with `dataset.toml` containing `[gym]` and Harbor task folders is NeMo Gym. A directory with `task.toml` plus `instruction.md` or `steps/` is Harbor. JSONL with `responses_create_params` is NeMo Gym. JSONL with `prompt` or `question` is Verifiers. A dataset-level `verifier.py` or `task_data.py` is copied with the dataset. `[gym]` fields such as `license` and `allowed_agents` are copied from `dataset.toml` when the source has them.

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
