import numpy as np

from rinkiq.pose import facing_from_keypoints
from rinkiq.rink import LANDMARKS, Calibration, pixels_to_rink, snap_correction, zone_of


def test_homography_round_trip():
    # A synthetic camera: rink feet → pixels with a known perspective matrix.
    H_true = np.array([[5.0, 0.8, 100], [0.2, -4.0, 600], [0.0, 0.001, 1]])
    names = ["Neutral dot · left · far", "Neutral dot · left · near", "Neutral dot · right · far",
             "Neutral dot · right · near", "Center ice dot"]
    rink = np.array([LANDMARKS[n] for n in names])
    px = pixels_to_rink(rink, H_true)
    cal = Calibration.from_pairs([(tuple(p), n) for p, n in zip(px, names)])
    assert cal.reprojection_error_ft() < 1e-3
    assert np.allclose(pixels_to_rink(px, cal.H), rink, atol=1e-3)


def test_dot_snapping_fixes_drift():
    truth = np.array([[31.0, 20.5], [31.0, 64.5]])
    drifted = truth + np.array([30.0, -8.0])  # 31 ft of drift
    C = snap_correction(drifted, [True, True])
    fixed = pixels_to_rink(drifted, C)
    assert np.allclose(fixed, truth, atol=0.5)


def test_zones():
    assert zone_of(150, attacks_right=True) == "OZ"
    assert zone_of(150, attacks_right=False) == "DZ"
    assert zone_of(100, attacks_right=True) == "NZ"


def test_facing_toward_camera():
    k = np.zeros((17, 2))
    c = np.ones(17)
    k[5], k[6] = [60, 50], [40, 50]      # left shoulder on image right → facing the camera
    k[11], k[12] = [57, 75], [43, 75]
    k[0] = [50, 40]
    v, _ = facing_from_keypoints(k, c)
    assert v[1] > 0.7                     # +y = toward the camera
