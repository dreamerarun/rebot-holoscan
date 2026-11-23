"""Offline: a motor send that times out must drop the bridge (read -> None -> HOLD) and reopen it (no hardware)."""
import sys, time, types
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/holoscan_app")
import real_live

class FakeMotor:
    def __init__(s, fail): s.fail = fail
    def send_pos_vel(s, *a):
        if s.fail[0]: raise RuntimeError("dm-serial write failed: Operation timed out")
    def request_feedback(s): pass
    def get_state(s): return types.SimpleNamespace(pos=0.0, vel=0.0, torq=0.1, status_code=1, t_mos=30, t_rotor=30)

fail = [False]
arm = real_live.RealArmLive.__new__(real_live.RealArmLive)
arm.dry, arm.enabled, arm.grip_enabled, arm.vlim = False, True, False, 1.0
opened = []
def fake_open():
    arm.arm = [(f"joint{i}", FakeMotor(fail)) for i in range(1, 7)]; arm.grip = ("gripper", FakeMotor(fail))
    arm.ctrl = types.SimpleNamespace(close=lambda: None, poll_feedback_once=lambda: None); opened.append(1)
arm.open = fake_open; arm._poll = lambda w: None
fake_open()
arm.cfg = None
real_live.sim_to_motor_rad = lambda cfg, q7: {f"joint{i}": 0.0 for i in range(1, 7)} | {"gripper": 0.0}

arm.send([0.0] * 7)
assert arm.ctrl is not None and all(v is not None for v in arm.read().values()), "healthy send/read"
fail[0] = True
arm.send([0.0] * 7)
assert arm.ctrl is None, "failed send must drop the bridge"
assert all(v is None for v in arm.read().values()), "reads must report no feedback (-> SafetyFilter HOLD)"
fail[0] = False
arm._last_reopen = 0.0
st = arm.read()                               # reopen attempt
assert len(opened) == 2 and arm.ctrl is not None, "bridge must be reopened"
assert all(v is not None for v in arm.read().values()), "feedback back after reopen"
print("send-failure test: PASS (drop -> no feedback -> reopen)")
