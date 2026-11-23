import json, sys, math
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if r.get("measured")]
# find where measured stops changing at all
last_change = None
for i in range(1, len(rows)):
    if rows[i]["measured"] != rows[i - 1]["measured"]:
        last_change = i
r = rows[last_change]
print("rows", len(rows), "last change of measured at row", last_change, "t", round(r["t"] - rows[0]["t"], 1), "s of", round(rows[-1]["t"] - rows[0]["t"], 1))
print("  state then", r["state"], r.get("mode"), "measured", [round(math.degrees(v), 3) for v in r["measured"][:7]])
for j in (last_change - 300, last_change - 60, last_change - 1):
    q = rows[j]; print("  before", round(q["t"] - rows[0]["t"], 1), q["state"], q.get("mode"), [round(math.degrees(v), 3) for v in q["measured"][:6]])
print("  keys:", list(rows[-1].keys()))
