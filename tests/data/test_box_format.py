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

import math

import pytest

from llamafactory.data.box_format import BOX_FORMATS, canonicalize_angle, normalize_boxes


def _corners(x1, y1, x2, y2, angle):
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) / 2, (y2 - y1) / 2
    cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    points = {
        (round(center_x + s * cos - t * sin, 6), round(center_y + s * sin + t * cos, 6))
        for s in (-half_w, half_w)
        for t in (-half_h, half_h)
    }
    return points


@pytest.mark.parametrize(
    "box_format, text, expected",
    [
        ("geochat", "{<51><21><63><33>|<20>}", "{<51><21><63><33>|<20>}"),
        ("geochat", "{<51><21><63><33>}", "{<51><21><63><33>|<0>}"),
        ("percent", "{<0><68><11><83>}", "{<0><68><11><83>|<0>}"),
        ("percent", "{<0><68><11><83>|<90>}", "{<0><68><11><83>|<90>}"),
        ("unit_list", "[0.05 0.0, 1.0 0.89]", "{<5><0><100><89>|<0>}"),
        ("unit_list", "[0.29, 0.57, 0.58, 1.0]", "{<29><57><58><100>|<0>}"),
        ("unit_list", "at <point>(0.62, 0.33)</point> here", "at <point>(62, 33)</point> here"),
    ],
)
def test_converts_each_format(box_format, text, expected):
    result = normalize_boxes(text, box_format)
    assert result.text == expected
    assert result.valid


def test_converts_every_box_in_a_message():
    result = normalize_boxes("a {<1><2><3><4>|<5>} b {<6><7><8><9>}{<10><11><12><13>|<14>}", "geochat")
    assert result.text == "a {<1><2><3><4>|<5>} b {<6><7><8><9>|<0>}{<10><11><12><13>|<14>}"
    assert result.stats["boxes"] == 3
    assert result.stats["defaulted"] == 1


def test_custom_default_angle():
    assert normalize_boxes("{<1><2><3><4>}", "percent", default_angle=90).text == "{<1><2><3><4>|<90>}"


@pytest.mark.parametrize(
    "box_format, text",
    [
        ("geochat", "{<981><21><63><33>|<20>}"),  # out of range
        ("geochat", "{<63><21><51><33>|<20>}"),  # reversed x
        ("geochat", "{<51><33><63><21>|<20>}"),  # reversed y
        ("geochat", "{<51><21><63><33>|<181>}"),  # angle out of range
        ("geochat", "{<51><21><63>"),  # cut off
        ("geochat", "{<-1><21><63><33>}"),  # negative
        ("geochat", "fine {<1><2><3><4>|<5>} but {<981><2><3><4>|<5>}"),  # one bad box in the message
        ("unit_list", "[1.2 0.0, 1.5 0.9]"),  # out of range
        ("unit_list", "[0.9 0.0, 0.1 0.9]"),  # reversed
        ("unit_list", "<point>(1.5, 0.3)</point>"),  # out of range
    ],
)
def test_invalid_boxes_are_flagged_and_not_repaired(box_format, text):
    result = normalize_boxes(text, box_format)
    assert not result.valid
    assert result.stats["invalid"] >= 1


@pytest.mark.parametrize(
    "box_format, text",
    [
        ("percent", "There are 4 planes, 3 ships, 2 cars and 1 truck."),
        ("percent", "A list [1, 2, 3, 4] and a <ref>region</ref>."),
        ("unit_list", "The answers are 0.1, 0.2, 0.3 and 0.4."),
        ("unit_list", "Are there touching boundaries? no"),
        ("geochat", ""),
    ],
)
def test_text_without_boxes_is_unchanged(box_format, text):
    result = normalize_boxes(text, box_format)
    assert result.text == text
    assert result.valid
    assert not result.stats


def test_unit_conversion_does_not_touch_braced_boxes():
    text = "{<1><2><3><4>|<5>}"
    assert normalize_boxes(text, "unit_list").text == text


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError):
        normalize_boxes("text", "pixels")
    assert set(BOX_FORMATS) == {"geochat", "percent", "unit_list"}


def test_invalid_default_angle_is_rejected():
    with pytest.raises(ValueError):
        normalize_boxes("text", "percent", default_angle=181)


def test_canonical_angle_rewrites_ninety_degrees():
    # {<14><75><24><100>|<90>} = {<6.5><82.5><31.5><92.5>|<0>}, rounded half up.
    result = normalize_boxes("{<14><75><24><100>|<90>}", "geochat", canonical_angle=True)
    assert result.text == "{<7><83><32><93>|<0>}"
    assert result.stats["rewritten"] == 1


def test_canonical_angle_is_off_by_default():
    assert normalize_boxes("{<14><75><24><100>|<90>}", "geochat").text == "{<14><75><24><100>|<90>}"


@pytest.mark.parametrize("box", [(10, 20, 30, 60, 90), (40, 40, 60, 50, 130), (20, 30, 50, 40, 179)])
def test_canonical_angle_keeps_the_same_rectangle(box):
    # Even widths and heights have no half-units, so the rewrite is exact.
    rewritten = canonicalize_angle(*box)
    assert rewritten[4] == box[4] - 90
    assert _corners(*rewritten) == _corners(*box)


def test_canonical_angle_half_turn_and_low_angles():
    assert canonicalize_angle(1, 2, 3, 4, 180) == (1, 2, 3, 4, 0)
    assert canonicalize_angle(1, 2, 3, 4, 89) == (1, 2, 3, 4, 89)
    assert canonicalize_angle(1, 2, 3, 4, 0) == (1, 2, 3, 4, 0)


def test_canonical_angle_keeps_a_box_that_would_leave_the_image():
    # A tall box against the left edge: swapping its sides would give a negative x.
    result = normalize_boxes("{<0><10><4><90>|<90>}", "geochat", canonical_angle=True)
    assert result.text == "{<0><10><4><90>|<90>}"
    assert result.valid
    assert result.stats["kept_outside_range"] == 1
