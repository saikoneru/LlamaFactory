#!/usr/bin/env python
"""Build centralized parquet views whose markers and columns select encoders.

The source datasets are immutable. Rewritten parquet files are streamed one
row group at a time and installed atomically; already-correct native RGB
datasets are validated and exposed through symlinks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


HF_METADATA_KEY = b"huggingface"
MESSAGES_COLUMN = "messages"


class ConfigurationError(ValueError):
    """The manifest or checkpoint encoder contract is inconsistent."""


class RowValidationError(ValueError):
    """A parquet row does not satisfy its configured encoder routes."""


@dataclass(frozen=True)
class RouteBinding:
    name: str
    source_column: str
    target_column: str
    target_marker: str
    fallback_markers: tuple[str, ...]
    required_modalities: tuple[str, ...]
    min_modalities: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_config = Path(__file__).with_name("encoder_views.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset key to process; repeat as needed. The default is all.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        metavar="DATASET:SPLIT",
        help="Limit work to an exact dataset/split pair; repeat as needed.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override output_root from the manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and source paths without scanning rows.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Scan every selected row but do not write outputs or symlinks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated files or non-matching symlinks.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"Cannot read JSON config {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} must contain a JSON object.")
    return value


def validate_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if manifest.get("version") != 1:
        raise ConfigurationError("Only encoder view manifest version 1 is supported.")

    route_configs = manifest.get("routes")
    datasets = manifest.get("datasets")
    if not isinstance(route_configs, dict) or not route_configs:
        raise ConfigurationError("Manifest 'routes' must be a non-empty object.")
    if not isinstance(datasets, dict) or not datasets:
        raise ConfigurationError("Manifest 'datasets' must be a non-empty object.")

    default_checkpoint_path = Path(str(manifest.get("checkpoint_config", "")))
    checkpoint_specs_by_path: dict[Path, dict[str, dict[str, Any]]] = {}

    for route_name, route in route_configs.items():
        if not isinstance(route, dict):
            raise ConfigurationError(f"Route {route_name!r} must be an object.")
        marker = route.get("marker")
        column = route.get("column")
        if not isinstance(marker, str) or not marker:
            raise ConfigurationError(f"Route {route_name!r} has no marker.")
        if not isinstance(column, str) or not column:
            raise ConfigurationError(f"Route {route_name!r} has no column.")
        if route.get("native"):
            continue

        checkpoint_path = Path(
            str(route.get("checkpoint_config") or default_checkpoint_path)
        )
        if checkpoint_path not in checkpoint_specs_by_path:
            checkpoint = load_json(checkpoint_path)
            checkpoint_specs_by_path[checkpoint_path] = {
                str(spec["name"]): spec
                for spec in checkpoint.get("encoders", [])
                if isinstance(spec, dict) and spec.get("name")
            }
        checkpoint_specs = checkpoint_specs_by_path[checkpoint_path]
        encoder_name = str(route.get("encoder") or route_name)
        spec = checkpoint_specs.get(encoder_name)
        if spec is None:
            raise ConfigurationError(
                f"Route {route_name!r} references missing checkpoint encoder "
                f"{encoder_name!r}."
            )
        if marker != spec.get("marker"):
            raise ConfigurationError(
                f"Route {route_name!r} marker {marker!r} does not match "
                f"checkpoint marker {spec.get('marker')!r}."
            )
        if column != encoder_name:
            raise ConfigurationError(
                f"Canonical column {column!r} must match encoder {encoder_name!r}."
            )
        expected_modalities = tuple(
            str(item)
            for item in (
                (spec.get("extra") or {}).get("modalities") or []
                if isinstance(spec.get("extra"), dict)
                else []
            )
        )
        configured_modalities = tuple(
            str(item) for item in route.get("required_modalities", [])
        )
        if configured_modalities != expected_modalities:
            raise ConfigurationError(
                f"Route {route_name!r} modalities {configured_modalities} do "
                f"not match checkpoint modalities {expected_modalities}."
            )
        extra = spec.get("extra") if isinstance(spec.get("extra"), dict) else {}
        expected_minimum = int(
            extra.get("min_modalities", len(expected_modalities))
        )
        configured_minimum = int(
            route.get("min_modalities", len(configured_modalities))
        )
        if configured_minimum != expected_minimum:
            raise ConfigurationError(
                f"Route {route_name!r} min_modalities={configured_minimum} "
                f"does not match checkpoint min_modalities={expected_minimum}."
            )

    for dataset_name, dataset in datasets.items():
        if not isinstance(dataset, dict):
            raise ConfigurationError(f"Dataset {dataset_name!r} must be an object.")
        if dataset.get("mode") not in {"rewrite", "symlink"}:
            raise ConfigurationError(
                f"Dataset {dataset_name!r} mode must be 'rewrite' or 'symlink'."
            )
        if not isinstance(dataset.get("splits"), dict) or not dataset["splits"]:
            raise ConfigurationError(f"Dataset {dataset_name!r} has no splits.")
        build_route_bindings(dataset_name, dataset, route_configs)

    return route_configs


def build_route_bindings(
    dataset_name: str,
    dataset: dict[str, Any],
    route_configs: dict[str, dict[str, Any]],
) -> list[RouteBinding]:
    bindings: list[RouteBinding] = []
    seen_columns: set[str] = set()
    for item in dataset.get("routes", []):
        if not isinstance(item, dict) or not item.get("route"):
            raise ConfigurationError(
                f"Dataset {dataset_name!r} has an invalid route binding."
            )
        route_name = str(item["route"])
        if route_name not in route_configs:
            raise ConfigurationError(
                f"Dataset {dataset_name!r} references unknown route {route_name!r}."
            )
        route = route_configs[route_name]
        source_column = str(item.get("source_column") or route["column"])
        target_column = str(route["column"])
        if source_column in seen_columns:
            raise ConfigurationError(
                f"Dataset {dataset_name!r} maps source column "
                f"{source_column!r} more than once."
            )
        seen_columns.add(source_column)
        bindings.append(
            RouteBinding(
                name=route_name,
                source_column=source_column,
                target_column=target_column,
                target_marker=str(route["marker"]),
                fallback_markers=tuple(
                    str(marker) for marker in item.get("fallback_markers", [])
                ),
                required_modalities=tuple(
                    str(modality)
                    for modality in route.get("required_modalities", [])
                ),
                min_modalities=int(
                    route.get(
                        "min_modalities",
                        len(route.get("required_modalities", [])),
                    )
                ),
            )
        )
    if not bindings:
        raise ConfigurationError(f"Dataset {dataset_name!r} has no encoder routes.")
    return bindings


def _message_marker_count(messages: list[dict[str, Any]], marker: str) -> int:
    return sum(
        message["content"].count(marker)
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), str)
    )


def _replace_marker(
    messages: list[dict[str, Any]], source: str, target: str
) -> None:
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and source in content:
            message["content"] = content.replace(source, target)


def _validate_messages(messages: Any, location: str) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise RowValidationError(f"{location}: messages must be a non-empty list.")
    copied: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RowValidationError(
                f"{location}: message {index} must be an object."
            )
        if not isinstance(message.get("content"), str):
            raise RowValidationError(
                f"{location}: message {index} has no string content."
            )
        copied.append(dict(message))
    return copied


def _unwrap_single_payload(payload: Any) -> Any:
    if isinstance(payload, (list, tuple)) and len(payload) == 1:
        return payload[0]
    return payload


def _validate_payload(
    binding: RouteBinding,
    payload: Any,
    payload_count: int,
    location: str,
) -> None:
    if payload_count != 1:
        raise RowValidationError(
            f"{location}: route {binding.name!r} needs exactly one payload in "
            f"{binding.source_column!r}, found {payload_count}."
        )
    if not binding.required_modalities:
        return

    sample = _unwrap_single_payload(payload)
    if not isinstance(sample, dict):
        raise RowValidationError(
            f"{location}: route {binding.name!r} payload must be an object with "
            f"{list(binding.required_modalities)}."
        )
    missing = [
        modality
        for modality in binding.required_modalities
        if sample.get(modality) is None
    ]
    present_count = len(binding.required_modalities) - len(missing)
    if present_count < binding.min_modalities:
        raise RowValidationError(
            f"{location}: route {binding.name!r} needs at least "
            f"{binding.min_modalities} of {list(binding.required_modalities)}, "
            f"found {present_count}; missing={missing}."
        )


def rewrite_row_messages(
    messages: Any,
    bindings: list[RouteBinding],
    payloads: dict[str, Any],
    payload_counts: dict[str, int],
    remove_markers: Iterable[str],
    location: str,
) -> list[dict[str, Any]]:
    rewritten = _validate_messages(messages, location)

    for binding in bindings:
        _validate_payload(
            binding,
            payloads.get(binding.source_column),
            payload_counts[binding.source_column],
            location,
        )
        target_count = _message_marker_count(rewritten, binding.target_marker)
        fallback_counts = {
            marker: _message_marker_count(rewritten, marker)
            for marker in binding.fallback_markers
            if marker != binding.target_marker
        }
        fallback_count = sum(fallback_counts.values())
        if target_count == 0 and fallback_count == 1:
            source_marker = next(
                marker for marker, count in fallback_counts.items() if count == 1
            )
            _replace_marker(rewritten, source_marker, binding.target_marker)
        elif target_count != 1:
            raise RowValidationError(
                f"{location}: route {binding.name!r} needs exactly one "
                f"{binding.target_marker!r} marker or one fallback marker; "
                f"found target={target_count}, fallbacks={fallback_counts}."
            )

    for marker in remove_markers:
        _replace_marker(rewritten, str(marker), "")

    for binding in bindings:
        count = _message_marker_count(rewritten, binding.target_marker)
        if count != 1:
            raise RowValidationError(
                f"{location}: route {binding.name!r} has {count} canonical "
                f"{binding.target_marker!r} markers after rewriting."
            )
    for marker in remove_markers:
        if _message_marker_count(rewritten, str(marker)):
            raise RowValidationError(
                f"{location}: forbidden marker {marker!r} remains after rewriting."
            )
    return rewritten


def _payload_counts(column: pa.ChunkedArray) -> list[int]:
    values = column.combine_chunks()
    if pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
        return [0 if value is None else int(value) for value in pc.list_value_length(values).to_pylist()]
    if pa.types.is_fixed_size_list(values.type):
        size = values.type.list_size
        return [0 if values[index].as_py() is None else size for index in range(len(values))]
    return [0 if values[index].as_py() is None else 1 for index in range(len(values))]


def _payload_values(
    table: pa.Table, binding: RouteBinding, counts: list[int]
) -> list[Any]:
    if binding.required_modalities:
        return table.column(binding.source_column).to_pylist()
    return [None if count == 0 else True for count in counts]


def validate_source_schema(
    source_schema: pa.Schema,
    bindings: list[RouteBinding],
    dataset_name: str,
    split_name: str,
) -> None:
    required = {MESSAGES_COLUMN, *(binding.source_column for binding in bindings)}
    missing = sorted(required - set(source_schema.names))
    if missing:
        raise ConfigurationError(
            f"{dataset_name}:{split_name} source is missing columns {missing}."
        )


def build_output_schema(
    source_schema: pa.Schema,
    bindings: list[RouteBinding],
    drop_columns: Iterable[str],
) -> pa.Schema:
    drop = {str(name) for name in drop_columns}
    renames = {
        binding.source_column: binding.target_column for binding in bindings
    }
    fields: list[pa.Field] = []
    output_names: set[str] = set()
    for field in source_schema:
        if field.name in drop and field.name not in renames:
            continue
        name = renames.get(field.name, field.name)
        if name in output_names:
            raise ConfigurationError(
                f"Output schema would contain duplicate column {name!r}."
            )
        output_names.add(name)
        fields.append(
            pa.field(
                name,
                field.type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
        )

    metadata = dict(source_schema.metadata or {})
    raw_hf = metadata.get(HF_METADATA_KEY)
    if raw_hf is not None:
        try:
            hf_info = json.loads(raw_hf.decode())
            features = hf_info.get("info", {}).get("features")
            if isinstance(features, dict):
                for name in drop:
                    if name not in renames:
                        features.pop(name, None)
                for source_name, target_name in renames.items():
                    if source_name in features and source_name != target_name:
                        features[target_name] = features.pop(source_name)
                metadata[HF_METADATA_KEY] = json.dumps(hf_info).encode()
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ConfigurationError(
                f"Invalid Hugging Face schema metadata: {error}"
            ) from error
    return pa.schema(fields, metadata=metadata)


def _compression(source: pq.ParquetFile) -> str:
    if source.num_row_groups == 0:
        return "snappy"
    value = source.metadata.row_group(0).column(0).compression.lower()
    return "none" if value == "uncompressed" else value


def process_row_group(
    table: pa.Table,
    bindings: list[RouteBinding],
    remove_markers: Iterable[str],
    source_path: Path,
    row_offset: int,
) -> list[list[dict[str, Any]]]:
    messages = table.column(MESSAGES_COLUMN).to_pylist()
    counts_by_column = {
        binding.source_column: _payload_counts(
            table.column(binding.source_column)
        )
        for binding in bindings
    }
    values_by_column = {
        binding.source_column: _payload_values(
            table,
            binding,
            counts_by_column[binding.source_column],
        )
        for binding in bindings
    }

    rewritten = []
    for row_index, row_messages in enumerate(messages):
        location = f"{source_path}:row {row_offset + row_index}"
        rewritten.append(
            rewrite_row_messages(
                row_messages,
                bindings,
                {
                    column: values[row_index]
                    for column, values in values_by_column.items()
                },
                {
                    column: counts[row_index]
                    for column, counts in counts_by_column.items()
                },
                remove_markers,
                location,
            )
        )
    return rewritten


def scan_parquet(
    source_path: Path,
    bindings: list[RouteBinding],
    remove_markers: Iterable[str],
    *,
    writer: pq.ParquetWriter | None = None,
    output_schema: pa.Schema | None = None,
) -> int:
    source = pq.ParquetFile(source_path)
    row_offset = 0
    renames = {
        binding.source_column: binding.target_column for binding in bindings
    }
    source_by_output = {target: source for source, target in renames.items()}

    for group_index in range(source.num_row_groups):
        table = source.read_row_group(group_index)
        rewritten = process_row_group(
            table,
            bindings,
            remove_markers,
            source_path,
            row_offset,
        )
        if writer is not None:
            if output_schema is None:
                raise AssertionError("Writer requires an output schema.")
            arrays: list[pa.Array | pa.ChunkedArray] = []
            for field in output_schema:
                if field.name == MESSAGES_COLUMN:
                    arrays.append(pa.array(rewritten, type=field.type))
                else:
                    source_name = source_by_output.get(field.name, field.name)
                    arrays.append(table.column(source_name).combine_chunks())
            writer.write_table(pa.Table.from_arrays(arrays, schema=output_schema))
        row_offset += table.num_rows
    return row_offset


def rewrite_parquet(
    source_path: Path,
    output_path: Path,
    bindings: list[RouteBinding],
    remove_markers: Iterable[str],
    drop_columns: Iterable[str],
    *,
    overwrite: bool,
    validate_only: bool,
) -> int:
    source = pq.ParquetFile(source_path)
    output_schema = build_output_schema(
        source.schema_arrow, bindings, drop_columns
    )
    if validate_only:
        return scan_parquet(source_path, bindings, remove_markers)
    if output_path.exists() and not overwrite:
        print(f"skip existing {output_path}")
        return source.metadata.num_rows

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}"
    )
    temporary.unlink(missing_ok=True)
    writer = pq.ParquetWriter(
        temporary,
        output_schema,
        compression=_compression(source),
    )
    try:
        rows = scan_parquet(
            source_path,
            bindings,
            remove_markers,
            writer=writer,
            output_schema=output_schema,
        )
        writer.close()
        writer = None
        os.replace(temporary, output_path)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return rows


def source_parquet_files(split: dict[str, Any]) -> list[Path]:
    source = Path(str(split.get("source", "")))
    pattern = str(split.get("pattern") or "*.parquet")
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise ConfigurationError(f"Source does not exist: {source}")
    files = sorted(path for path in source.glob(pattern) if path.is_file())
    if not files:
        raise ConfigurationError(
            f"No files matching {pattern!r} under {source}."
        )
    return files


def install_symlink(source: Path, target: Path, *, overwrite: bool) -> None:
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(
                f"{target} exists and is not the expected symlink; "
                "pass --overwrite to replace it."
            )
        if target.is_dir() and not target.is_symlink():
            raise IsADirectoryError(
                f"Refusing to replace directory {target} with a symlink."
            )
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(source)
    os.replace(temporary, target)


def process_symlink_split(
    files: list[Path],
    output_dir: Path,
    bindings: list[RouteBinding],
    remove_markers: Iterable[str],
    *,
    overwrite: bool,
    validate_only: bool,
) -> int:
    rows = 0
    for source_path in files:
        rows += scan_parquet(source_path, bindings, remove_markers)
    if not validate_only:
        for source_path in files:
            install_symlink(
                source_path,
                output_dir / source_path.name,
                overwrite=overwrite,
            )
    return rows


def selected_split_pairs(
    manifest: dict[str, Any], args: argparse.Namespace
) -> list[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    datasets = manifest["datasets"]
    requested_datasets = set(args.dataset)
    unknown_datasets = requested_datasets - set(datasets)
    if unknown_datasets:
        raise ConfigurationError(
            f"Unknown datasets requested: {sorted(unknown_datasets)}."
        )
    blocked_requests = {
        name: datasets[name].get("blocked_reason", "disabled in the manifest")
        for name in requested_datasets
        if datasets[name].get("enabled", True) is False
    }
    if blocked_requests:
        details = "; ".join(
            f"{name}: {reason}" for name, reason in blocked_requests.items()
        )
        raise ConfigurationError(f"Requested datasets are disabled: {details}")

    requested_pairs: set[tuple[str, str]] = set()
    for value in args.split:
        if ":" not in value:
            raise ConfigurationError(
                f"Split selector {value!r} must have DATASET:SPLIT form."
            )
        requested_pairs.add(tuple(value.split(":", 1)))

    pairs = []
    for dataset_name, dataset in datasets.items():
        if dataset.get("enabled", True) is False:
            continue
        if requested_datasets and dataset_name not in requested_datasets:
            continue
        for split_name, split in dataset["splits"].items():
            if requested_pairs and (dataset_name, split_name) not in requested_pairs:
                continue
            pairs.append((dataset_name, split_name, dataset, split))

    unknown_pairs = requested_pairs - {
        (dataset_name, split_name)
        for dataset_name, split_name, _, _ in pairs
    }
    if unknown_pairs:
        raise ConfigurationError(
            f"Unknown or filtered split selectors: {sorted(unknown_pairs)}."
        )
    if not pairs:
        raise ConfigurationError("No dataset splits were selected.")
    return pairs


def write_report(output_root: Path, report: dict[str, Any]) -> None:
    path = output_root / "last_build_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_json(args.config)
    route_configs = validate_manifest(manifest)
    output_root = args.output_root or Path(str(manifest["output_root"]))
    pairs = selected_split_pairs(manifest, args)

    report: dict[str, Any] = {
        "manifest": str(args.config.resolve()),
        "output_root": str(output_root),
        "splits": {},
    }
    for dataset_name, split_name, dataset, split in pairs:
        bindings = build_route_bindings(dataset_name, dataset, route_configs)
        files = source_parquet_files(split)
        for source_path in files:
            source = pq.ParquetFile(source_path)
            validate_source_schema(
                source.schema_arrow, bindings, dataset_name, split_name
            )
        output_path = output_root / str(split["output"])
        print(
            f"{dataset_name}:{split_name} mode={dataset['mode']} "
            f"files={len(files)} output={output_path}"
        )
        if args.dry_run:
            rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
        elif dataset["mode"] == "rewrite":
            if len(files) != 1:
                raise ConfigurationError(
                    f"Rewrite split {dataset_name}:{split_name} must have one "
                    f"source parquet, found {len(files)}."
                )
            rows = rewrite_parquet(
                files[0],
                output_path,
                bindings,
                dataset.get("remove_markers", []),
                dataset.get("drop_columns", []),
                overwrite=args.overwrite,
                validate_only=args.validate_only,
            )
        else:
            rows = process_symlink_split(
                files,
                output_path,
                bindings,
                dataset.get("remove_markers", []),
                overwrite=args.overwrite,
                validate_only=args.validate_only,
            )
        report["splits"][f"{dataset_name}:{split_name}"] = {
            "files": len(files),
            "rows": rows,
            "mode": dataset["mode"],
            "output": str(output_path),
        }
        verb = "contains" if args.dry_run else "validated"
        print(f"{dataset_name}:{split_name} {verb} {rows} rows")

    if not args.dry_run and not args.validate_only:
        write_report(output_root, report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigurationError, RowValidationError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
