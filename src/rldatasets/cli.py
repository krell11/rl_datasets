from __future__ import annotations

import argparse
import importlib
import sys

from rldatasets.task import Result

_MARK = r"""
 ____  _     ____        _
|  _ \| |   |  _ \  __ _| |_ __ _
| |_) | |   | | | |/ _` | __/ _` |
|  _ <| |___| |_| | (_| | || (_| |
|_| \_\_____|____/ \__,_|\__\__,_|
""".strip("\n")


def sign_off(result: Result) -> str:
    route = f"{result.source_format}  -->  {result.target_format}"
    return f"{result.render()}\n\n{_MARK}\n  {route}"


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--to", required=True, help="harbor, verifiers, openenv, nemo-gym, or a registered name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--import", dest="module", help="import a module that calls register()")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rldata",
        description="Convert and merge RL environment datasets between Harbor, Verifiers, OpenEnv, and NeMo Gym.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    convert_parser = commands.add_parser("convert", help="rewrite a dataset into another format")
    convert_parser.add_argument("source")
    _add_output_arguments(convert_parser)
    convert_parser.add_argument("--from", dest="source_format", help="source format when detection is ambiguous")
    merge_parser = commands.add_parser("merge", help="combine datasets and write one format")
    merge_parser.add_argument("sources", nargs="+", help="two or more datasets, in the order to keep")
    _add_output_arguments(merge_parser)
    merge_parser.add_argument("--from", dest="source_format", help="format of every source when detection is ambiguous")
    args = parser.parse_args(argv)
    from rldatasets import convert, merge

    try:
        if args.module:
            importlib.import_module(args.module)
        if args.command == "merge":
            result = merge(args.sources, args.output, to=args.to, source_format=args.source_format)
        else:
            result = convert(args.source, args.output, to=args.to, source_format=args.source_format)
    except (OSError, ValueError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(sign_off(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
