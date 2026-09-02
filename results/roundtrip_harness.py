"""Round-trip transfer test: a gesture found in Open Skate, run on the phone.

Open Skate optimised this gesture for a -360 degree roll about the deck's long
axis. It has no idea what that trick is called. The rig's OCR reader names
whatever the real game does. If the two agree, the physics transfers.

NO PUSH is fired, deliberately: the gesture was optimised against a board at
rest at the reset anchor, which is how the capture corpus was recorded. The
rig's execute_gesture_recipe always pushes first, which would test something
else.

Screenshots are taken directly rather than through the rig's capture helper,
which blocked here waiting on a notification window.
"""
import argparse, json, sys, time
from collections import Counter
import numpy as np, cv2
sys.path.insert(0, "/Users/training-server/trueskate-ai/src")
from trueskate_ai.rl.device_worker import DeviceWorker, DEVICES
from trueskate_ai.sim.touch_actions import execute_n_slot_gestures, reset_position
from trueskate_ai.sim.gestures import scale_to_device
from trueskate_ai.sim.trick_info_reader import detect_trick

ap = argparse.ArgumentParser()
ap.add_argument("--library", required=True)
ap.add_argument("--device", default="iPhone_XR")
ap.add_argument("--trials", type=int, default=8)
ap.add_argument("--shots", type=int, default=12)
a = ap.parse_args()

recipe = json.load(open(a.library))["best_gestures"]
cfg = next(d for d in DEVICES if d["name"] == a.device)
dw, dh = cfg["logical_w"], cfg["logical_h"]
pts = [[scale_to_device(x, y, dw, dh) for x, y in g["points"]] for g in recipe["gestures"]]
durs = [g["duration"] for g in recipe["gestures"]]
eases = [(lambda t, p=g["easing_power"]: t ** p) if g["easing_power"] != 1.0 else None
         for g in recipe["gestures"]]

def shot(driver):
    buf = np.frombuffer(driver.get_screenshot_as_png(), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

w = DeviceWorker(cfg); w.connect()
results = []
try:
    for i in range(a.trials):
        try:
            reset_position(w.driver, dw, dh); time.sleep(0.7)
        except Exception as exc:
            print(f"trial {i}: reset failed: {exc}", flush=True)
        try:
            execute_n_slot_gestures(w.driver, gestures_points=pts,
                                    gestures_durations=durs,
                                    delays=recipe.get("delays") or [],
                                    easings=eases)
        except Exception as exc:
            print(f"trial {i}: gesture failed: {exc}", flush=True); continue
        found = None
        for _ in range(a.shots):
            try:
                r = detect_trick(shot(w.driver))
            except Exception:
                r = None
            if r is not None:
                found = r
                break
            time.sleep(0.18)
        name = getattr(found, "trick", None) if found else None
        status = getattr(found, "status", None) if found else None
        print(f"trial {i}: trick={name!r} status={status!r}", flush=True)
        results.append((name, status))
finally:
    w.disconnect()

print("\n=== ROUND-TRIP SUMMARY ===", flush=True)
for k, v in Counter(f"{n} [{s}]" for n, s in results).most_common():
    print(f"  {v:2d}x  {k}", flush=True)
