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

from llamafactory.data.box_format import (
    BOX_FORMATS,
    canonicalize_angle,
    enclosing_pixel_box,
    has_qwen_boxes,
    normalize_boxes,
    pixel_point,
    render_qwen_boxes,
    to_qwen_markers,
    validate_box_grid,
)


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


# --- Qwen2.5-VL target ------------------------------------------------------------------------------


def test_qwen_markers_replace_normalized_boxes_only():
    text = "There are <p>two ships</p> {<10><20><30><40>|<0>} and {<50><60><70><80>|<35>}."
    assert to_qwen_markers(text) == (
        "There are <p>two ships</p> <qwen_box>10,20,30,40,0</qwen_box> and <qwen_box>50,60,70,80,35</qwen_box>."
    )
    assert to_qwen_markers("no boxes") == "no boxes"


def test_upright_box_scales_to_pixels_of_the_resized_image():
    # A 504 x 504 model input (512 x 512 VRSBench image after patch rounding).
    assert enclosing_pixel_box(25, 40, 33, 60, 0, 504, 504) == (126, 202, 166, 302)
    # Non-square: x scales with the width, y with the height.
    assert enclosing_pixel_box(0, 0, 50, 50, 0, 700, 350) == (0, 0, 350, 175)
    assert enclosing_pixel_box(0, 0, 100, 100, 180, 280, 140) == (0, 0, 280, 140)


def test_ninety_degrees_swaps_width_and_height_about_the_centre():
    # GeoChat writes an axis-aligned 10 x 2 car as the 2 x 10 box at 90 degrees.
    assert enclosing_pixel_box(49, 45, 51, 55, 90, 100, 100) == (45, 49, 55, 51)
    # Same result as canonicalizing first, then rendering the upright box.
    assert enclosing_pixel_box(*canonicalize_angle(49, 45, 51, 55, 90), 100, 100) == (45, 49, 55, 51)


def test_rotated_box_becomes_its_upright_enclosing_box_in_pixels():
    # A 20 x 4 percent box at 45 degrees on a 200 x 100 image: rotated in pixels, not in percent.
    x1, y1, x2, y2 = enclosing_pixel_box(40, 48, 60, 52, 45, 200, 100)
    corners = _corners(80, 48, 120, 52, 45)
    assert (x1, y1, x2, y2) == (
        round(min(x for x, _ in corners)),
        round(min(y for _, y in corners)),
        round(max(x for x, _ in corners)),
        round(max(y for _, y in corners)),
    )


def test_enclosing_box_is_clipped_to_the_image():
    # The whole image rotated by 30 degrees encloses more than the image: clipped back to it.
    assert enclosing_pixel_box(0, 0, 100, 100, 30, 100, 100) == (0, 0, 100, 100)
    assert enclosing_pixel_box(90, 90, 100, 100, 45, 100, 100)[2:] == (100, 100)


def test_render_writes_qwen_bbox_json():
    text = 'Answer: <qwen_box>25,40,33,60,0</qwen_box>'
    assert render_qwen_boxes(text, 504, 504) == 'Answer: {"bbox_2d": [126, 202, 166, 302]}'
    assert has_qwen_boxes(text) and not has_qwen_boxes(render_qwen_boxes(text, 504, 504))


def test_qwen_markers_also_replace_normalized_points():
    text = "Box the class at <point>(62, 33)</point>: {<5><0><100><89>|<0>}"
    marked = to_qwen_markers(text)
    assert marked == "Box the class at <qwen_point>62,33</qwen_point>: <qwen_box>5,0,100,89,0</qwen_box>"
    assert has_qwen_boxes("<qwen_point>1,2</qwen_point>")


def test_render_writes_qwen_point_json():
    text = "at <qwen_point>62,33</qwen_point>"
    assert render_qwen_boxes(text, 504, 504) == 'at {"point_2d": [312, 166]}'
    assert pixel_point(100, 0, 120, 120) == (120, 0)


def test_bigearthnet_renders_on_the_fixed_120_grid():
    # A BigEarthNet box prompt after unit_list normalization, rendered on its 120 x 120 patch grid.
    normalized = normalize_boxes("Box the class at <point>(0.65, 0.84)</point>. [0.25 0.59, 0.9 1.0]", "unit_list")
    rendered = render_qwen_boxes(to_qwen_markers(normalized.text), *validate_box_grid([120, 120]))
    assert rendered == 'Box the class at {"point_2d": [78, 101]}. {"bbox_2d": [30, 71, 108, 120]}'


@pytest.mark.parametrize("grid", [[120], [0, 120], [120.0, 120], "120x120", [True, 120], None])
def test_box_grid_is_validated(grid):
    with pytest.raises(ValueError, match="box_grid"):
        validate_box_grid(grid)
