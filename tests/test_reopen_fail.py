"""Offline: a reopen that FAILS must leave read() returning 'no feedback' (not crash), then recover later."""
import sys, types
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/holoscan_app")
import real_live

class FakeMotor:
    def send_pos_vel(s, *a): pass
    def request_feedback(s): pass
    def get_state(s): return types.SimpleNamespace(pos=0.0, vel=0.0, torq=0.1, status_code=1, t_mos=30, t_rotor=30)

arm = real_live.RealArmLive.__new__(real_live.RealArmLive)
arm.dry, arm.enabled, arm.grip_enabled, arm.vlim = False, True, False, 1.0
arm.cfg = {"joints": [{"name": f"joint{i}"} for i in range(1, 7)]}
state = {"fail": True}
def fake_open():
    if state["fail"]:
        raise OSError("port busy")
    arm.arm = [(f"joint{i}", FakeMotor()) for i in range(1, 7)]; arm.grip = ("gripper", FakeMotor())
    arm.ctrl = types.SimpleNamespace(close=lambda: None, poll_feedback_once=lambda: None)
arm.open = fake_open; arm._poll = lambda w: None
arm.arm, arm.grip, arm.ctrl = [], None, None
arm._last_reopen = 0.0
r = arm.read()                                   # reopen attempt fails
assert set(r) == {f"joint{i}" for i in range(1, 7)} | {"gripper"} and all(v is None for v in r.values())
arm._last_reopen = 0.0
r = arm.read()                                   # still failing: the old code crashed here (TypeError)
assert all(v is None for v in r.values())
arm.disable_arm(print)                           # must not crash with grip None
state["fail"] = False; arm._last_reopen = 0.0; arm.enabled = True
arm.read()                                       # reopen succeeds
assert all(v is not None for v in arm.read().values())
print("failed-reopen test: PASS (no crash, recovers)")
