# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from datasets import Dataset

from llamafactory.data import Role
from llamafactory.data.converter import align_dataset, get_dataset_converter
from llamafactory.data.parser import DatasetAttr
from llamafactory.hparams import DataArguments


@pytest.mark.runs_on(["cpu", "mps"])
def test_alpaca_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "instruction": "Solve the math problem.",
        "input": "3 + 4",
        "output": "The answer is 7.",
    }
    dataset_converter = get_dataset_converter("alpaca", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
        "_encoders": {},
        "_terramind": None,
    }


@pytest.mark.runs_on(["cpu", "mps"])
def test_sharegpt_medgemma_from_hf_image_dict():
    dataset_attr = DatasetAttr("file", "mimic")
    dataset_attr.join(
        {
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "medgemma": "medgemma"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    )
    data_args = DataArguments()
    example = {
        "messages": [
            {"role": "user", "content": "<medgemma>\nGenerate a radiology report."},
            {"role": "assistant", "content": "The lungs are clear."},
        ],
        "medgemma": [{"bytes": b"fake-jpeg", "path": None}],
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    result = dataset_converter(example)
    assert result["_images"] is None
    assert result["_encoders"]["medgemma"] == b"fake-jpeg"
    assert result["_prompt"][0]["content"].startswith("<medgemma>")


@pytest.mark.runs_on(["cpu", "mps"])
def test_sharegpt_medgemma_from_image_path_list():
    dataset_attr = DatasetAttr("file", "mimic")
    dataset_attr.join(
        {
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "medgemma": "medgemma"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    )
    data_args = DataArguments()
    example = {
        "messages": [
            {"role": "user", "content": "<medgemma>\nGenerate a radiology report."},
            {"role": "assistant", "content": "The lungs are clear."},
        ],
        "medgemma": ["/tmp/cxr.png"],
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    result = dataset_converter(example)
    assert result["_images"] is None
    assert result["_encoders"]["medgemma"] == "/tmp/cxr.png"


@pytest.mark.runs_on(["cpu", "mps"])
def test_sharegpt_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": "Solve the math problem.\n3 + 4"},
            {"from": "gpt", "value": "The answer is 7."},
        ]
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "You are a helpful assistant.",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
        "_encoders": {},
        "_terramind": None,
    }


def _sharegpt_attr(**fields):
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset", formatting="sharegpt")
    dataset_attr.messages = "messages"
    dataset_attr.role_tag = "role"
    dataset_attr.content_tag = "content"
    dataset_attr.user_tag = "user"
    dataset_attr.assistant_tag = "assistant"
    dataset_attr.system_tag = "system"
    for key, value in fields.items():
        setattr(dataset_attr, key, value)
    return dataset_attr


def _box_example(user, assistant):
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


@pytest.mark.runs_on(["cpu", "mps"])
def test_sharegpt_converter_normalizes_boxes_in_user_and_assistant_turns():
    converter = get_dataset_converter("sharegpt", _sharegpt_attr(box_format="unit_list"), DataArguments())
    output = converter(_box_example("Box the class at <point>(0.62, 0.33)</point>.", "[0.05 0.0, 1.0 0.89]"))
    assert output["_prompt"][0]["content"] == "Box the class at <point>(62, 33)</point>."
    assert output["_response"][0]["content"] == "{<5><0><100><89>|<0>}"


@pytest.mark.runs_on(["cpu", "mps"])
def test_sharegpt_converter_skips_a_row_with_an_invalid_box():
    converter = get_dataset_converter("sharegpt", _sharegpt_attr(box_format="geochat"), DataArguments())
    output = converter(_box_example("[refer] where is it?", "{<981><21><63><33>|<20>}"))
    assert output["_prompt"] == []
    assert output["_response"] == []


@pytest.mark.runs_on(["cpu", "mps"])
def test_sharegpt_converter_leaves_text_alone_without_box_format():
    converter = get_dataset_converter("sharegpt", _sharegpt_attr(), DataArguments())
    output = converter(_box_example("[0.1 0.2, 0.3 0.4]", "{<981><21><63><33>|<20>}"))
    assert output["_prompt"][0]["content"] == "[0.1 0.2, 0.3 0.4]"
    assert output["_response"][0]["content"] == "{<981><21><63><33>|<20>}"


@pytest.mark.runs_on(["cpu", "mps"])
def test_box_format_requires_a_sharegpt_dataset():
    with pytest.raises(ValueError):
        DatasetAttr("hf_hub", "name").join({"box_format": "geochat"})
    with pytest.raises(ValueError):
        DatasetAttr("hf_hub", "name").join({"formatting": "sharegpt", "box_format": "pixels"})


@pytest.mark.runs_on(["cpu", "mps"])
def test_resuming_past_an_invalid_box_lands_on_the_same_row():
    rows = [
        _box_example("first", "{<1><2><3><4>|<5>}"),
        _box_example("second", "{<981><2><3><4>|<5>}"),
        _box_example("third", "{<6><7><8><9>}"),
        _box_example("fourth", "{<10><11><12><13>|<14>}"),
    ]
    dataset = Dataset.from_list(rows).to_iterable_dataset()
    dataset_attr = _sharegpt_attr(box_format="geochat")
    aligned = align_dataset(dataset, dataset_attr, DataArguments(streaming=True), None)

    full = list(aligned)
    assert len(full) == len(rows)  # the converter keeps rows one to one; the processor drops the empty one
    assert full[1]["_prompt"] == [] and full[1]["_response"] == []
    assert full[2]["_response"][0]["content"] == "{<6><7><8><9>|<0>}"

    for count in range(1, len(rows)):
        resumed = list(aligned.skip_before_align(count))
        assert [row["_response"] for row in resumed] == [row["_response"] for row in full[count:]]
