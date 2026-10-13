"""The point + line homography solver recovers a known camera exactly."""
import numpy as np

from rinkiq.autocal import fit_points_lines
from rinkiq.rink import pixels_to_rink


def _camera():
    # a plausible broadcast-like mapping rink feet → pixels
    H_rink_to_px = np.array([[6.0, 1.2, 40.0], [0.3, -5.5, 700.0], [0.0004, 0.004, 1.0]])
    return np.linalg.inv(H_rink_to_px), H_rink_to_px


def _image_line_for_rink_x(X, H_rink_to_px):
    a = pixels_to_rink(np.float64([[X, 5.0], [X, 80.0]]), H_rink_to_px)
    p, q = np.append(a[0], 1), np.append(a[1], 1)
    l = np.cross(p, q)
    return l / np.linalg.norm(l[:2])


def test_three_points_and_a_line_recover_the_camera():
    H_true, H_inv = _camera()
    rink_pts = np.float64([[80, 20.5], [120, 64.5], [31, 64.5]])
    img_pts = pixels_to_rink(rink_pts, H_inv)
    lines = [_image_line_for_rink_x(75, H_inv), _image_line_for_rink_x(100, H_inv)]
    H = fit_points_lines(img_pts, rink_pts, lines, [75.0, 100.0])
    probe = pixels_to_rink(np.float64([[40, 40], [160, 70], [100, 10]]), H_inv)
    assert np.allclose(pixels_to_rink(probe, H), [[40, 40], [160, 70], [100, 10]], atol=1e-3)


def test_points_only_still_works():
    H_true, H_inv = _camera()
    rink_pts = np.float64([[80, 20.5], [120, 20.5], [100, 64.5], [31, 64.5], [169, 20.5]])  # not in a row
    H = fit_points_lines(pixels_to_rink(rink_pts, H_inv), rink_pts, [], [])
    assert np.allclose(pixels_to_rink(pixels_to_rink(rink_pts, H_inv), H), rink_pts, atol=1e-3)


def test_too_few_constraints_returns_none():
    _, H_inv = _camera()
    rink_pts = np.float64([[80, 20.5]])
    assert fit_points_lines(pixels_to_rink(rink_pts, H_inv), rink_pts, [], []) is None
