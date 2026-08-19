from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from loop_geometry import (  # noqa: E402
    angle_spectrum,
    context_retention_rate,
    holonomy,
    minimal_rotation,
    quadratic_form_matrix,
    signature,
    unit,
)
from wm_schemes import WMConfig, exp_sample, green_mask  # noqa: E402


def test_minimal_rotation_maps_endpoints_and_is_special_orthogonal() -> None:
    x = np.array([1.0, 0.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0, 0.0])
    rotation = minimal_rotation(x, y)

    assert np.allclose(rotation @ unit(x), unit(y), atol=1e-10)
    assert np.allclose(rotation.T @ rotation, np.eye(4), atol=1e-10)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10)


def test_signature_degenerates_to_one_rotation_plane() -> None:
    rotation = minimal_rotation(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
    )
    matrix = quadratic_form_matrix(rotation)

    assert signature(matrix) == (2, 0, 2)
    assert len(angle_spectrum(rotation)) == 1


def test_holonomy_fixes_the_base_point() -> None:
    chain = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    path_holonomy = holonomy(chain)

    assert np.allclose(path_holonomy @ unit(chain[0]), unit(chain[0]), atol=1e-10)
    assert np.isclose(np.linalg.det(path_holonomy), 1.0, atol=1e-10)


def test_context_retention_detects_intact_windows() -> None:
    original = list(range(10))
    untouched = original.copy()
    periodic_edits = [100 + i if i % 2 == 0 else token for i, token in enumerate(original)]

    assert context_retention_rate(original, untouched, h=1) == 1.0
    assert context_retention_rate(original, periodic_edits, h=1) == 0.0


def test_watermark_primitives_are_deterministic() -> None:
    cfg = WMConfig("KGW", key=20260819, vocab_size=64, h=1)
    first = green_mask(cfg, [7])
    second = green_mask(cfg, [7])

    assert first.dtype == np.bool_
    assert np.array_equal(first, second)

    exp_cfg = WMConfig("EXP", key=20260819, vocab_size=4, h=1)
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    assert exp_sample(exp_cfg, probabilities, [3]) == exp_sample(
        exp_cfg, probabilities, [3]
    )
