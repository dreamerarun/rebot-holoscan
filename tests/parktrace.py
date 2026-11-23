import json, sys, math
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if r.get("cmd") and r.get("measured")]
# last PARKING stretch
idx = [i for i, r in enumerate(rows) if r.get("state") == "PARKING"]
if not idx:
    print("no PARKING rows; states seen:", sorted({r.get("state") for r in rows[-400:]})); sys.exit()
end = idx[-1]; start = end
while start > 0 and rows[start - 1].get("state") == "PARKING": start -= 1
print("PARKING rows", start, end, "duration", round(rows[end]["t"] - rows[start]["t"], 2), "s")
for i in list(range(start, end + 1, max(1, (end - start) // 10))) + [end, min(end + 3, len(rows) - 1)]:
    r = rows[i]
    c = [round(math.degrees(v), 1) for v in r["cmd"][:6]]; m = [round(math.degrees(v), 1) for v in r["measured"][:6]]
    print(f"t {r['t'] - rows[start]['t']:5.2f} {r['state']:8s} {r.get('mode','')[:6]:6s} cmd {c} meas {m}")
