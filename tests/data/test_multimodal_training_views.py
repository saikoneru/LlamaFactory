import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "dvps"
    / "multimodal_training"
    / "build_views.py"
)
SPEC = importlib.util.spec_from_file_location("multimodal_training_views", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
views = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = views
SPEC.loader.exec_module(views)


def binding(
    name: str,
    source_column: str,
    target_column: str,
    target_marker: str,
    *fallback_markers: str,
    required_modalities: tuple[str, ...] = (),
    min_modalities: int | None = None,
):
    if min_modalities is None:
        min_modalities = len(required_modalities)
    return views.RouteBinding(
        name=name,
        source_column=source_column,
        target_column=target_column,
        target_marker=target_marker,
        fallback_markers=fallback_markers,
        required_modalities=required_modalities,
        min_modalities=min_modalities,
    )


def messages(content: str):
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": "answer"},
    ]


def test_manifest_matches_checkpoint_encoder_contract(tmp_path):
    manifest = views.load_json(SCRIPT_PATH.with_name("encoder_views.json"))
    checkpoint = tmp_path / "config.json"
    checkpoint.write_text(
        json.dumps(
            {
                "encoders": [
                    {
                        "name": "autoavsr_audio",
                        "marker": "<autoavsr_audio>",
                    },
                    {
                        "name": "autoavsr_video",
                        "marker": "<autoavsr_video>",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest["checkpoint_config"] = str(checkpoint)
    routes = views.validate_manifest(manifest)

    assert routes["terramind"]["marker"] == "<terramind>"
    assert routes["terramind"]["min_modalities"] == 1
    assert routes["autoavsr_audio"]["column"] == "autoavsr_audio"
    assert routes["autoavsr_video"]["column"] == "autoavsr_video"

    processor_config = views.load_json(
        Path(routes["terramind"]["checkpoint_config"]).with_name(
            "processor_config.json"
        )
    )
    processor_terramind = next(
        spec
        for spec in processor_config["encoder_specs"]
        if spec["name"] == "terramind"
    )
    assert processor_terramind["extra"]["min_modalities"] == 1


def test_bigearthnet_is_selected_for_materialization():
    manifest = views.load_json(SCRIPT_PATH.with_name("encoder_views.json"))
    pairs = views.selected_split_pairs(
        manifest,
        SimpleNamespace(dataset=[], split=[]),
    )

    ben_splits = {
        split for dataset, split, _, _ in pairs if dataset == "BigEarthNet.txt"
    }
    assert ben_splits == {"train", "val", "test"}


def test_lipcrops_routes_both_streams_to_autoavsr():
    bindings = [
        binding(
            "autoavsr_audio",
            "audios",
            "autoavsr_audio",
            "<autoavsr_audio>",
            "<audio>",
        ),
        binding(
            "autoavsr_video",
            "lipread",
            "autoavsr_video",
            "<autoavsr_video>",
            "<lipread>",
        ),
    ]

    rewritten = views.rewrite_row_messages(
        messages("<audio><lipread>Transcribe this clip."),
        bindings,
        {"audios": True, "lipread": True},
        {"audios": 1, "lipread": 1},
        [],
        "test:row 0",
    )

    content = rewritten[0]["content"]
    assert content.startswith("<autoavsr_audio><autoavsr_video>")
    assert "<audio>" not in content
    assert "<lipread>" not in content


def test_bigearthnet_drops_native_marker_and_accepts_s2_only():
    terramind = binding(
        "terramind",
        "terramind",
        "terramind",
        "<terramind>",
        "<image>",
        required_modalities=("S1GRD", "S2L2A"),
        min_modalities=1,
    )
    payload = {"S2L2A": "s2.npy"}

    rewritten = views.rewrite_row_messages(
        messages("<image><terramind>Describe the patch."),
        [terramind],
        {"terramind": payload},
        {"terramind": 1},
        ["<image>"],
        "test:row 0",
    )

    assert rewritten[0]["content"].startswith("<terramind>")
    assert "<image>" not in rewritten[0]["content"]

    with pytest.raises(views.RowValidationError, match="needs at least 1"):
        views.rewrite_row_messages(
            messages("<terramind>Describe the patch."),
            [terramind],
            {"terramind": {}},
            {"terramind": 1},
            ["<image>"],
            "test:row 1",
        )


def test_output_schema_renames_features_and_preserves_hf_metadata():
    hf_metadata = {
        "info": {
            "features": {
                "messages": {"_type": "List"},
                "audios": {"_type": "Audio"},
                "lipread": {"dtype": "string", "_type": "Value"},
                "unused": {"dtype": "string", "_type": "Value"},
            }
        }
    }
    schema = pa.schema(
        [
            pa.field("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            pa.field("audios", pa.list_(pa.string())),
            pa.field("lipread", pa.list_(pa.string())),
            pa.field("unused", pa.string()),
        ],
        metadata={views.HF_METADATA_KEY: json.dumps(hf_metadata).encode()},
    )
    bindings = [
        binding(
            "autoavsr_audio",
            "audios",
            "autoavsr_audio",
            "<autoavsr_audio>",
            "<audio>",
        ),
        binding(
            "autoavsr_video",
            "lipread",
            "autoavsr_video",
            "<autoavsr_video>",
            "<lipread>",
        ),
    ]

    output = views.build_output_schema(schema, bindings, ["unused"])
    features = json.loads(output.metadata[views.HF_METADATA_KEY])["info"]["features"]

    assert output.names == ["messages", "autoavsr_audio", "autoavsr_video"]
    assert set(features) == {"messages", "autoavsr_audio", "autoavsr_video"}
    assert features["autoavsr_audio"]["_type"] == "Audio"


def write_lipcrops_source(path: Path, user_content: str) -> None:
    table = pa.table(
        {
            "messages": [messages(user_content)],
            "audios": [["clip.flac"]],
            "lipread": [["clip.mp4"]],
            "sample_id": ["sample-1"],
        }
    )
    pq.write_table(table, path, row_group_size=1)


def test_rewrite_parquet_is_atomic_and_preserves_metadata_columns(tmp_path):
    source = tmp_path / "source.parquet"
    output = tmp_path / "output.parquet"
    write_lipcrops_source(source, "<audio><lipread>Transcribe.")
    bindings = [
        binding(
            "autoavsr_audio",
            "audios",
            "autoavsr_audio",
            "<autoavsr_audio>",
            "<audio>",
        ),
        binding(
            "autoavsr_video",
            "lipread",
            "autoavsr_video",
            "<autoavsr_video>",
            "<lipread>",
        ),
    ]

    rows = views.rewrite_parquet(
        source,
        output,
        bindings,
        [],
        [],
        overwrite=False,
        validate_only=False,
    )
    result = pq.read_table(output).to_pylist()

    assert rows == 1
    assert result[0]["sample_id"] == "sample-1"
    assert result[0]["autoavsr_audio"] == ["clip.flac"]
    assert result[0]["autoavsr_video"] == ["clip.mp4"]
    assert result[0]["messages"][0]["content"].startswith(
        "<autoavsr_audio><autoavsr_video>"
    )

    invalid = tmp_path / "invalid.parquet"
    invalid_output = tmp_path / "invalid-output.parquet"
    write_lipcrops_source(invalid, "<audio>Missing video marker.")
    with pytest.raises(views.RowValidationError):
        views.rewrite_parquet(
            invalid,
            invalid_output,
            bindings,
            [],
            [],
            overwrite=False,
            validate_only=False,
        )
    assert not invalid_output.exists()
    assert not list(tmp_path.glob(f".{invalid_output.name}.tmp-*"))


def test_native_rgb_is_validated_then_symlinked(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    source = source_dir / "train-00000.parquet"
    pq.write_table(
        pa.table(
            {
                "messages": [messages("<image>Describe this.")],
                "images": [["rgb.jpg"]],
            }
        ),
        source,
    )
    qwen = binding("qwen_vision", "images", "images", "<image>")

    rows = views.process_symlink_split(
        [source],
        output_dir,
        [qwen],
        [],
        overwrite=False,
        validate_only=False,
    )

    link = output_dir / source.name
    assert rows == 1
    assert link.is_symlink()
    assert link.resolve() == source.resolve()


def test_path_mapper_uses_longest_prefix():
    mapper = views.PathMapper(
        [("/old/datasets/", "/new/datasets/"), ("/old/datasets/mt/", "/new/mt/")]
    )

    assert mapper.path("/old/datasets/mt/a.parquet") == "/new/mt/a.parquet"
    assert mapper.path("/old/datasets/x/b.png") == "/new/datasets/x/b.png"
    assert mapper.path("/elsewhere/c.png") == "/elsewhere/c.png"
    assert mapper.value([{"bytes": None, "path": "/old/datasets/x/d.png"}]) == [
        {"bytes": None, "path": "/new/datasets/x/d.png"}
    ]


def test_rewrite_remaps_media_paths_and_checks_them(tmp_path):
    media = tmp_path / "new" / "img.png"
    media.parent.mkdir()
    media.write_bytes(b"png")
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.table(
            {
                "messages": [messages("<image>Describe."), messages("<image>Again.")],
                "images": [
                    [{"bytes": None, "path": "/helios/img.png"}],
                    [{"bytes": None, "path": "/helios/gone.png"}],
                ],
            }
        ),
        source,
    )
    qwen = binding("qwen_vision", "images", "images", "<image>")
    mapper = views.PathMapper([("/helios/", f"{tmp_path}/new/")])
    check = views.PathCheck({})
    output = tmp_path / "out.parquet"

    views.rewrite_parquet(
        source,
        output,
        [qwen],
        [],
        [],
        overwrite=False,
        validate_only=False,
        path_mapper=mapper,
        path_columns=["images"],
        path_check=check,
    )
    rows = pq.read_table(output).to_pylist()

    assert rows[0]["images"][0]["path"] == str(media)
    assert rows[1]["images"][0]["path"] == f"{tmp_path}/new/gone.png"
    assert check.missing == 1
    assert check.examples == [f"{tmp_path}/new/gone.png"]


def test_embedded_bytes_need_no_file():
    check = views.PathCheck({})
    check.add([[{"bytes": b"x", "path": "/nowhere/a.png"}]])

    assert check.missing == 0


def test_multi_payload_route_needs_one_marker_per_payload():
    ecg = binding("medgemma", "medgemma", "medgemma", "<medgemma>")
    ecg = views.RouteBinding(**{**ecg.__dict__, "multi_payload": True})

    rewritten = views.rewrite_row_messages(
        messages("<medgemma><medgemma>Compare the two ECGs."),
        [ecg],
        {"medgemma": True},
        {"medgemma": 2},
        [],
        "test:row 0",
    )
    assert rewritten[0]["content"].count("<medgemma>") == 2

    with pytest.raises(views.RowValidationError, match="exactly 2"):
        views.rewrite_row_messages(
            messages("<medgemma>Only one marker."),
            [ecg],
            {"medgemma": True},
            {"medgemma": 2},
            [],
            "test:row 1",
        )


def test_symlink_split_can_use_relative_links(tmp_path):
    source_dir = tmp_path / "datasets" / "Aya" / "train"
    source_dir.mkdir(parents=True)
    source = source_dir / "train-00000.parquet"
    pq.write_table(pa.table({"messages": [messages("Translate this.")]}), source)
    output_dir = tmp_path / "views" / "Aya" / "train"

    rows = views.process_symlink_split(
        [source], output_dir, [], [], overwrite=False, validate_only=False, relative=True
    )
    link = output_dir / source.name

    assert rows == 1
    assert not Path(os.readlink(link)).is_absolute()
    assert link.resolve() == source.resolve()


def test_text_only_dataset_needs_explicit_flag():
    with pytest.raises(views.ConfigurationError, match="no encoder routes"):
        views.build_route_bindings("Aya", {"routes": []}, {})
    assert views.build_route_bindings("Aya", {"routes": [], "text_only": True}, {}) == []


def test_main_maps_sources_and_writes_relative_registry(tmp_path):
    helios = tmp_path / "helios" / "datasets" / "GeoChat" / "train"
    helios.mkdir(parents=True)
    (tmp_path / "helios" / "datasets" / "GeoChat" / "img.png").write_bytes(b"png")
    for index in range(2):
        pq.write_table(
            pa.table(
                {
                    "messages": [messages("<image>Describe.")],
                    "images": [["/net/datasets/GeoChat/img.png"]],
                }
            ),
            helios / f"train-0000{index}.parquet",
        )
    template = tmp_path / "dataset_info.json"
    template.write_text(
        json.dumps({"geochat_train": {"file_name": "/net/x", "columns": {"images": "images"}}})
    )
    manifest = tmp_path / "views.json"
    output_root = tmp_path / "views"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "output_root": str(output_root),
                "path_map": [["/net/", f"{tmp_path}/helios/"]],
                "routes": {
                    "qwen_vision": {"native": True, "marker": "<image>", "column": "images"}
                },
                "datasets": {
                    "GeoChat": {
                        "mode": "rewrite",
                        "routes": [{"route": "qwen_vision"}],
                        "splits": {
                            "train": {
                                "source": "/net/datasets/GeoChat/train",
                                "output": "GeoChat/train",
                                "dataset_info": "geochat_train",
                            }
                        },
                    }
                },
            }
        )
    )

    code = views.main(
        ["--config", str(manifest), "--registry-template", str(template), "--check-paths"]
    )
    registry = json.loads((output_root / "dataset_info.json").read_text())
    shard = pq.read_table(output_root / "GeoChat" / "train" / "train-00001.parquet")

    assert code == 0
    assert registry["geochat_train"]["file_name"] == "GeoChat/train"
    assert registry["geochat_train"]["columns"] == {"images": "images"}
    assert shard.column("images").to_pylist() == [
        [f"{tmp_path}/helios/datasets/GeoChat/img.png"]
    ]


def test_existing_mode_registers_in_place_and_refuses_other_paths(tmp_path):
    output_root = tmp_path / "views"
    (output_root / "ACDC").mkdir(parents=True)
    view = output_root / "ACDC" / "train.parquet"
    pq.write_table(
        pa.table(
            {
                "messages": [messages("<medgemma>Segment the heart.")],
                "medgemma": [[{"bytes": b"png", "path": "slice.png"}]],
            }
        ),
        view,
    )
    template = tmp_path / "dataset_info.json"
    template.write_text(json.dumps({"acdc_train": {"file_name": "/net/x", "columns": {"medgemma": "medgemma"}}}))

    def manifest(source):
        path = tmp_path / "views.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "output_root": str(output_root),
                    "routes": {"medgemma": {"native": True, "marker": "<medgemma>", "column": "medgemma"}},
                    "datasets": {
                        "ACDC": {
                            "mode": "existing",
                            "routes": [{"route": "medgemma"}],
                            "splits": {
                                "train": {
                                    "source": str(source),
                                    "output": "ACDC/train.parquet",
                                    "dataset_info": "acdc_train",
                                }
                            },
                        }
                    },
                }
            )
        )
        return path

    before = view.read_bytes()
    code = views.main(
        ["--config", str(manifest(view)), "--registry-template", str(template), "--check-paths"]
    )
    registry = json.loads((output_root / "dataset_info.json").read_text())

    assert code == 0
    assert view.read_bytes() == before
    assert registry["acdc_train"]["file_name"] == "ACDC/train.parquet"

    elsewhere = tmp_path / "other.parquet"
    elsewhere.write_bytes(before)
    with pytest.raises(views.ConfigurationError, match="must be its own output"):
        views.main(["--config", str(manifest(elsewhere)), "--registry-template", str(template)])


def test_remap_mode_translates_paths_without_touching_markers(tmp_path):
    media = tmp_path / "leo" / "ecg.png"
    media.parent.mkdir()
    media.write_bytes(b"png")
    source_dir = tmp_path / "helios"
    source_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                # two markers, one path: a route check would reject this row
                "messages": [messages("<image><image>Compare.")],
                "images": [["/helios/ecg.png"]],
                "question_type": ["single-verify"],
            }
        ),
        source_dir / "test.parquet",
    )
    manifest = tmp_path / "bench.json"
    output_root = tmp_path / "bench_data"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "output_root": str(output_root),
                "path_map": [["/helios/", f"{tmp_path}/leo/"]],
                "routes": {"qwen_vision": {"native": True, "marker": "<image>", "column": "images"}},
                "datasets": {
                    "ECG-QA-test": {
                        "mode": "remap",
                        "path_columns": ["images", "audios"],
                        "splits": {"test": {"source": str(source_dir / "test.parquet"), "output": "ECG-QA/test.parquet"}},
                    }
                },
            }
        )
    )

    assert views.main(["--config", str(manifest), "--check-paths"]) == 0
    row = pq.read_table(output_root / "ECG-QA" / "test.parquet").to_pylist()[0]
    assert row["images"] == [str(media)]
    assert row["messages"][0]["content"] == "<image><image>Compare."
    assert row["question_type"] == "single-verify"


def test_remap_mode_requires_path_columns():
    manifest = {"version": 1, "routes": {"x": {"native": True, "marker": "<image>", "column": "images"}},
                "datasets": {"d": {"mode": "remap", "splits": {"s": {"source": "a", "output": "b"}}}}}
    with pytest.raises(views.ConfigurationError, match="lists no path_columns"):
        views.validate_manifest(manifest)


def _shuffle_manifest(tmp_path, seed=7, mode="rewrite"):
    source = tmp_path / "datasets" / "ECG-QA" / "train"
    source.mkdir(parents=True, exist_ok=True)
    # Two shards written in task order, as the ECG-QA conversion does.
    for shard, task in enumerate(("single", "comparison")):
        pq.write_table(
            pa.table(
                {
                    "messages": [messages(f"<image>{task} question {shard * 50 + i}") for i in range(50)],
                    "images": [[f"{tmp_path}/img.png"]] * 50,
                }
            ),
            source / f"part-0000{shard}.parquet",
        )
    (tmp_path / "img.png").write_bytes(b"png")
    manifest = tmp_path / "views.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "output_root": str(tmp_path / "views"),
                "routes": {"qwen_vision": {"native": True, "marker": "<image>", "column": "images"}},
                "datasets": {
                    "ECG-QA": {
                        "mode": mode,
                        **({"path_columns": ["images"]} if mode == "remap" else {}),
                        "routes": [{"route": "qwen_vision"}] if mode != "remap" else [],
                        "splits": {
                            "train": {"source": str(source), "output": "ECG-QA/train", "shuffle_seed": seed}
                        },
                    }
                },
            }
        )
    )
    return manifest, tmp_path / "views" / "ECG-QA" / "train"


def _questions(output):
    return [
        row[0]["content"]
        for path in sorted(output.glob("*.parquet"))
        for row in pq.read_table(path).column("messages").to_pylist()
    ]


def test_shuffle_seed_permutes_rows_across_shards(tmp_path):
    manifest, output = _shuffle_manifest(tmp_path)
    assert views.main(["--config", str(manifest)]) == 0

    shards = sorted(output.glob("*.parquet"))
    assert [path.name for path in shards] == ["part-00000.parquet", "part-00001.parquet"]
    assert [pq.ParquetFile(path).metadata.num_rows for path in shards] == [50, 50]
    questions = _questions(output)
    assert sorted(questions) == sorted(f"<image>{t} question {i}" for t, r in (("single", range(50)), ("comparison", range(50, 100))) for i in r)
    first = [q.split()[0] for q in questions[:50]]
    assert "<image>single" in first and "<image>comparison" in first  # both tasks in each shard
    report = json.loads((tmp_path / "views" / "last_build_report.json").read_text())
    assert report["splits"]["ECG-QA:train"]["shuffle_seed"] == 7
    assert not list(output.parent.glob(".train.shuffle-*"))


def test_shuffle_seed_is_reproducible_and_seed_dependent(tmp_path):
    manifest, output = _shuffle_manifest(tmp_path / "a", seed=7)
    views.main(["--config", str(manifest)])
    again, output_again = _shuffle_manifest(tmp_path / "b", seed=7)
    views.main(["--config", str(again)])
    other, output_other = _shuffle_manifest(tmp_path / "c", seed=8)
    views.main(["--config", str(other)])

    strip = lambda qs: [q.split(" ", 1)[1] for q in qs]  # noqa: E731
    assert strip(_questions(output)) == strip(_questions(output_again))
    assert strip(_questions(output)) != strip(_questions(output_other))


def test_shuffled_split_is_skipped_when_built_and_rebuilt_with_overwrite(tmp_path):
    manifest, output = _shuffle_manifest(tmp_path)
    views.main(["--config", str(manifest)])
    before = _questions(output)
    views.main(["--config", str(manifest)])
    assert _questions(output) == before
    (output / "part-00001.parquet").unlink()
    with pytest.raises(views.ConfigurationError, match="--overwrite"):
        views.main(["--config", str(manifest)])
    assert views.main(["--config", str(manifest), "--overwrite"]) == 0
    assert _questions(output) == before


def test_shuffle_seed_needs_a_written_view(tmp_path):
    manifest, _ = _shuffle_manifest(tmp_path)
    data = json.loads(manifest.read_text())
    data["datasets"]["ECG-QA"]["mode"] = "symlink"
    manifest.write_text(json.dumps(data))
    with pytest.raises(views.ConfigurationError, match="shuffle_seed"):
        views.main(["--config", str(manifest)])
    data["datasets"]["ECG-QA"]["mode"] = "rewrite"
    data["datasets"]["ECG-QA"]["splits"]["train"]["shuffle_seed"] = "7"
    manifest.write_text(json.dumps(data))
    with pytest.raises(views.ConfigurationError, match="integer"):
        views.main(["--config", str(manifest)])


def _pairs_options(tmp_path):
    times = tmp_path / "record_list.csv"
    times.write_text("path,ecg_time\nfiles/p1/s1/1,2180-01-02 08:00:00\nfiles/p1/s2/2,2181-05-06 09:00:00\nfiles/p1/s3/3,2181-05-06 09:00:00\n")
    return {
        "column": "medgemma",
        "marker": "<medgemma>",
        "drop_ties": True,
        "times": [{"csv": str(times), "path_regex": r"(files/p\d+/s\d+/\d+)\.png", "key_column": "path", "time_column": "ecg_time"}],
        "rules": [
            {"name": "consecutive", "match": "(?i)previous|recent", "labels": ["Previous ECG: ", "Recent ECG: "], "chronological": True},
            {"name": "irrelevant", "labels": ["First ECG: ", "Second ECG: "]},
        ],
    }


def _ecg_table(rows):
    return pa.table(
        {
            "messages": [[{"role": "user", "content": q}, {"role": "assistant", "content": "yes"}] for q, _ in rows],
            "medgemma": [list(p) for _, p in rows],
        }
    )


def test_payload_pairs_order_label_and_drop_ties(tmp_path):
    pairs = views.PayloadPairs(_pairs_options(tmp_path), None)
    out = pairs.apply(
        _ecg_table(
            [
                ("<medgemma><medgemma>Is it gone in the recent tracing vs the previous one?", ["/x/files/p1/s2/2.png", "/x/files/p1/s1/1.png"]),
                ("<medgemma><medgemma>Is it in the recent tracing?", ["/x/files/p1/s2/2.png", "/x/files/p1/s3/3.png"]),
                ("<medgemma><medgemma>Is it in the second ECG?", ["/x/files/p1/s2/2.png", "/x/files/p1/s1/1.png"]),
                ("<medgemma>Is there an AV block?", ["/x/files/p1/s1/1.png"]),
            ]
        )
    ).to_pylist()

    assert len(out) == 3  # the tie (same recording time) is dropped
    assert out[0]["medgemma"] == ["/x/files/p1/s1/1.png", "/x/files/p1/s2/2.png"]  # swapped to chronological
    assert out[0]["messages"][0]["content"] == (
        "Previous ECG: <medgemma>\nRecent ECG: <medgemma>\nIs it gone in the recent tracing vs the previous one?"
    )
    # An unrelated pair keeps its order and gets the first / second labels.
    assert out[1]["medgemma"] == ["/x/files/p1/s2/2.png", "/x/files/p1/s1/1.png"]
    assert out[1]["messages"][0]["content"].startswith("First ECG: <medgemma>\nSecond ECG: <medgemma>\n")
    assert out[2]["messages"][0]["content"] == "<medgemma>Is there an AV block?"  # single payload untouched
    assert pairs.stats == {"pairs[consecutive]": 2, "chronological_swapped": 1, "chronological_ties": 1, "pairs[irrelevant]": 1}


def test_payload_pairs_in_a_remap_build(tmp_path):
    source = tmp_path / "test.parquet"
    pq.write_table(
        _ecg_table([("<medgemma><medgemma>Compared to the previous one?", ["/x/files/p1/s2/2.png", "/x/files/p1/s1/1.png"])]),
        source,
    )
    manifest = tmp_path / "views.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "output_root": str(tmp_path / "out"),
                "routes": {"qwen_vision": {"native": True, "marker": "<image>", "column": "images"}},
                "datasets": {
                    "ECG-QA": {
                        "mode": "remap",
                        "path_columns": ["medgemma"],
                        "payload_pairs": _pairs_options(tmp_path),
                        "splits": {"test": {"source": str(source), "output": "ECG-QA/test.parquet"}},
                    }
                },
            }
        )
    )
    assert views.main(["--config", str(manifest)]) == 0
    row = pq.read_table(tmp_path / "out" / "ECG-QA" / "test.parquet").to_pylist()[0]
    assert row["medgemma"] == ["/x/files/p1/s1/1.png", "/x/files/p1/s2/2.png"]
    assert row["messages"][0]["content"].startswith("Previous ECG: <medgemma>\nRecent ECG: <medgemma>\n")
    report = json.loads((tmp_path / "out" / "last_build_report.json").read_text())
    assert report["splits"]["ECG-QA:test"]["payload_pairs"]["chronological_swapped"] == 1


def test_payload_pairs_need_a_written_view_and_two_labels(tmp_path):
    for mode, rules, message in (
        ("symlink", [], "payload_pairs"),
        ("remap", [{"labels": ["Only one: "]}], "both payloads"),
    ):
        manifest = tmp_path / f"{mode}.json"
        manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "output_root": str(tmp_path / "out"),
                    "routes": {"qwen_vision": {"native": True, "marker": "<image>", "column": "images"}},
                    "datasets": {
                        "ECG-QA": {
                            "mode": mode,
                            "path_columns": ["medgemma"],
                            "text_only": True,
                            "payload_pairs": {"column": "medgemma", "marker": "<medgemma>", "rules": rules},
                            "splits": {"test": {"source": str(tmp_path), "output": "x"}},
                        }
                    },
                }
            )
        )
        with pytest.raises(views.ConfigurationError, match=message):
            views.main(["--config", str(manifest)])
