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

from llamafactory.data import Role
from llamafactory.data.converter import get_dataset_converter
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
