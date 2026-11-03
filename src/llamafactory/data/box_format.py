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

r"""Rewrite the bounding boxes in a message to one format.

Every box becomes ``{<x1><y1><x2><y2>|<angle>}``: integer coordinates in [0, 100], and an
integer angle in [0, 180]. This is GeoChat's format: the rectangle is rotated about its centre
by the angle, so a box with no stated angle is upright and gets the default angle of 0.

Source formats, set per dataset as ``box_format`` in ``dataset_info.json``:

- ``geochat``, ``percent``: ``{<x1><y1><x2><y2>}``, with or without ``|<angle>``, in [0, 100].
- ``unit_list``: ``[x1 y1, x2 y2]`` in [0, 1], and ``<point>(x, y)</point>`` in [0, 1].

A box that is out of range, has reversed corners, or is cut off makes the message invalid.
Such boxes are never repaired: a value like ``<981>`` is probably truncated, so clipping it
would produce a wrong label that looks correct.
"""

import math
import re
from collections import Counter
from typing import NamedTuple


COORDINATE_MAX = 100
ANGLE_MAX = 180

_INTEGER = r"\s*(\d+)\s*"
_NUMBER = r"\s*(\d+(?:\.\d+)?)\s*"
_BRACED_BOX = re.compile(rf"\{{<{_INTEGER}><{_INTEGER}><{_INTEGER}><{_INTEGER}>(?:\|<{_INTEGER}>)?\}}")
_BRACE_OPEN = re.compile(r"\{\s*<")
_UNIT_LIST = re.compile(rf"\[{_NUMBER}[ ,]{_NUMBER},{_NUMBER}[ ,]{_NUMBER}\]")
_UNIT_POINT = re.compile(rf"<point>\s*\({_NUMBER},{_NUMBER}\)\s*</point>")


class NormalizedText(NamedTuple):
    text: str
    valid: bool
    stats: Counter


def format_box(x1: int, y1: int, x2: int, y2: int, angle: int) -> str:
    return f"{{<{x1}><{y1}><{x2}><{y2}>|<{angle}>}}"


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def canonicalize_angle(x1: int, y1: int, x2: int, y2: int, angle: int) -> tuple[int, int, int, int, int]:
    r"""Describe the same rectangle with an angle below 90, where that fits the format.

    Rotating about the centre gives R(c, w, h, a) = R(c, h, w, a - 90), and a half-turn maps a
    rectangle onto itself. Half-units are rounded, moving each edge by at most 0.5. A box whose
    rewritten corners would leave [0, 100] is returned unchanged, as clipping changes its shape.
    """
    if angle == ANGLE_MAX:
        return x1, y1, x2, y2, 0
    if angle < 90:
        return x1, y1, x2, y2, angle

    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half_width, half_height = (x2 - x1) / 2, (y2 - y1) / 2
    corners = (
        _round_half_up(center_x - half_height),
        _round_half_up(center_y - half_width),
        _round_half_up(center_x + half_height),
        _round_half_up(center_y + half_width),
    )
    if min(corners) < 0 or max(corners) > COORDINATE_MAX:
        return x1, y1, x2, y2, angle
    return (*corners, angle - 90)


def _is_ordered(x1: float, y1: float, x2: float, y2: float) -> bool:
    return x1 <= x2 and y1 <= y2


def _finish_box(x1: int, y1: int, x2: int, y2: int, angle: int, canonical_angle: bool, stats: Counter) -> str:
    if canonical_angle:
        box = canonicalize_angle(x1, y1, x2, y2, angle)
        if box != (x1, y1, x2, y2, angle):
            stats["rewritten"] += 1
        elif angle >= 90:
            stats["kept_outside_range"] += 1
        x1, y1, x2, y2, angle = box
    stats["boxes"] += 1
    return format_box(x1, y1, x2, y2, angle)


def _normalize_braced(text: str, default_angle: int, canonical_angle: bool, stats: Counter) -> tuple[str, bool]:
    valid = True

    def replace(match: re.Match) -> str:
        nonlocal valid
        x1, y1, x2, y2 = (int(value) for value in match.group(1, 2, 3, 4))
        if match.group(5) is None:
            angle = default_angle
            stats["defaulted"] += 1
        else:
            angle = int(match.group(5))
        if max(x1, y1, x2, y2) > COORDINATE_MAX or angle > ANGLE_MAX or not _is_ordered(x1, y1, x2, y2):
            valid = False
            stats["invalid"] += 1
            return match.group(0)
        return _finish_box(x1, y1, x2, y2, angle, canonical_angle, stats)

    converted = _BRACED_BOX.sub(replace, text)
    unmatched = len(_BRACE_OPEN.findall(text)) - len(_BRACED_BOX.findall(text))
    if unmatched > 0:  # for example a negative value, or an answer cut off inside a box
        valid = False
        stats["invalid"] += unmatched
    return converted, valid


def _normalize_unit(text: str, default_angle: int, canonical_angle: bool, stats: Counter) -> tuple[str, bool]:
    valid = True

    def replace_box(match: re.Match) -> str:
        nonlocal valid
        values = tuple(float(value) for value in match.groups())
        if max(values) > 1 or not _is_ordered(*values):
            valid = False
            stats["invalid"] += 1
            return match.group(0)
        stats["defaulted"] += 1
        x1, y1, x2, y2 = (_round_half_up(value * COORDINATE_MAX) for value in values)
        return _finish_box(x1, y1, x2, y2, default_angle, canonical_angle, stats)

    def replace_point(match: re.Match) -> str:
        nonlocal valid
        x, y = (float(value) for value in match.groups())
        if max(x, y) > 1:
            valid = False
            stats["invalid"] += 1
            return match.group(0)
        stats["points"] += 1
        return f"<point>({_round_half_up(x * COORDINATE_MAX)}, {_round_half_up(y * COORDINATE_MAX)})</point>"

    return _UNIT_POINT.sub(replace_point, _UNIT_LIST.sub(replace_box, text)), valid


_NORMALIZERS = {
    "geochat": _normalize_braced,
    "percent": _normalize_braced,
    "unit_list": _normalize_unit,
}
BOX_FORMATS = tuple(_NORMALIZERS)


def normalize_boxes(
    text: str, box_format: str, default_angle: int = 0, canonical_angle: bool = False
) -> NormalizedText:
    r"""Convert the boxes in ``text``. Text without boxes is returned as it is."""
    if box_format not in _NORMALIZERS:
        raise ValueError(f"Unknown box_format {box_format!r}; expected one of {BOX_FORMATS}.")
    if not 0 <= default_angle <= ANGLE_MAX:
        raise ValueError(f"default_box_angle must be in [0, {ANGLE_MAX}], got {default_angle}.")

    stats = Counter()
    converted, valid = _NORMALIZERS[box_format](text, default_angle, canonical_angle, stats)
    return NormalizedText(converted, valid, stats)
