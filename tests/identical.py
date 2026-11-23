import json, sys
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r.get("measured") and r.get("torque") and r["state"] in ("ON", "PARKING")]
    best, start, prev, cutoff = 0.0, None, None, None
    # stop at the known freeze of the 17:39 log (the last change of measured)
    for i, r in enumerate(rows):
        sig = (tuple(r["measured"]), tuple(r["torque"]))
        if sig == prev:
            if start is None: start = rows[i - 1]["t"]
            run = r["t"] - start
            if run > best: best, at = run, r["t"] - rows[0]["t"]
        else:
            start = None
        prev = sig
    print(f"{f.split('/')[-1]}: {len(rows)} ON rows, longest identical run {best:.2f} s" + (f" at t={at:.0f}s" if best else ""))
