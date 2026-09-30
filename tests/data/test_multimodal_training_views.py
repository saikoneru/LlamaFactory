import importlib.util
import json
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


def test_manifest_matches_checkpoint_encoder_contract():
    manifest = views.load_json(SCRIPT_PATH.with_name("encoder_views.json"))
    routes = views.validate_manifest(manifest)

    assert routes["terramind"]["marker"] == "<terramind>"
    assert routes["terramind"]["min_modalities"] == 1
    assert routes["omniavsr_audio"]["column"] == "omniavsr_audio"
    assert routes["omniavsr_video"]["column"] == "omniavsr_video"

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


def test_lipcrops_routes_both_streams_to_omniavsr():
    bindings = [
        binding(
            "omniavsr_audio",
            "audios",
            "omniavsr_audio",
            "<omniavsr_audio>",
            "<audio>",
        ),
        binding(
            "omniavsr_video",
            "lipread",
            "omniavsr_video",
            "<omniavsr_video>",
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
    assert content.startswith("<omniavsr_audio><omniavsr_video>")
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
            "omniavsr_audio",
            "audios",
            "omniavsr_audio",
            "<omniavsr_audio>",
            "<audio>",
        ),
        binding(
            "omniavsr_video",
            "lipread",
            "omniavsr_video",
            "<omniavsr_video>",
            "<lipread>",
        ),
    ]

    output = views.build_output_schema(schema, bindings, ["unused"])
    features = json.loads(output.metadata[views.HF_METADATA_KEY])["info"]["features"]

    assert output.names == ["messages", "omniavsr_audio", "omniavsr_video"]
    assert set(features) == {"messages", "omniavsr_audio", "omniavsr_video"}
    assert features["omniavsr_audio"]["_type"] == "Audio"


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
            "omniavsr_audio",
            "audios",
            "omniavsr_audio",
            "<omniavsr_audio>",
            "<audio>",
        ),
        binding(
            "omniavsr_video",
            "lipread",
            "omniavsr_video",
            "<omniavsr_video>",
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
    assert result[0]["omniavsr_audio"] == ["clip.flac"]
    assert result[0]["omniavsr_video"] == ["clip.mp4"]
    assert result[0]["messages"][0]["content"].startswith(
        "<omniavsr_audio><omniavsr_video>"
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
