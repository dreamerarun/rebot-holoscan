"""The live programs must never run with the motorbridge READ-ONLY GUARD installed (2026-11-23: a live calibration
imported calibrate_camera -> real_arm when saving, which blocked every motor command for the rest of the run)."""
import importlib
import sys

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/holoscan_app")
from motorbridge import Motor


def guarded():
    try:
        Motor.send_pos_vel(None, 0.0, 0.0)
    except RuntimeError as e:
        return "READ-ONLY GUARD" in str(e)
    except Exception:
        return False
    return False


for mod in ("real_live", "real_teleop", "motion_plan", "perception", "homography", "robot_calibrate", "collector"):
    importlib.import_module(mod)
    assert not guarded(), f"importing {mod} installed the read-only guard"
from homography import solve  # noqa: E402,F401  what robot_calibrate._finish imports when it saves
assert not guarded(), "homography installed the guard"
print("live import chain: read-only guard NOT installed (OK)")
import calibrate_camera  # noqa: E402,F401  the read-only tool must still install it on purpose
assert guarded(), "calibrate_camera should install the guard"
print("calibrate_camera still read-only (OK)")
