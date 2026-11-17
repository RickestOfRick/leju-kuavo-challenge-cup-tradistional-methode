#!/usr/bin/env python3
"""Verify a seed sweep from the recorded bags and print a coloured summary.

Every bag is named seed<S>_p<D>_ok<0|1>_ik<F>.bag, so the verdict is in the name:
    ok=1 and ik=0  -> the parcel completed the chain with no IK failure

    python3 verify_seeds.py                          # scan bags/act
    python3 verify_seeds.py --dir DIR --csv out.csv
    python3 verify_seeds.py --json                   # machine-readable

Exit code 0 only if every seed has all 4 parcels green.
"""

import argparse
import glob
import json
import os
import re
import sys

BAG_DIR = "/root/kuavo_ws/bags/act"
SWEEP_LOG = "/root/kuavo_ws/bags/act/seed_sweep_50_manual_log.csv"
N_PARCELS = 4
GREEN, RED, YELLOW, DIM, RESET = ("\033[92m", "\033[91m", "\033[93m",
                                  "\033[90m", "\033[0m")


def scan(bag_dir):
    """-> {seed: {parcel: (ok, ik)}} from the bag names."""
    found = {}
    for path in glob.glob(os.path.join(bag_dir, "*.bag")):
        m = re.search(r"seed(\d+)_(?:p|pick)(\d+)_ok(\d)_ik(\d+)\.bag$",
                      os.path.basename(path))
        if not m:
            continue
        seed, parcel, ok, ik = map(int, m.groups())
        # keep the best verdict seen for a parcel (a re-record can supersede)
        prev = found.get(seed, {}).get(parcel)
        if prev is None or (ok, -ik) > (prev[0], -prev[1]):
            found.setdefault(seed, {})[parcel] = (ok, ik)
    return found


def main():
    global GREEN, RED, YELLOW, DIM, RESET
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=BAG_DIR)
    ap.add_argument("--sweep", action="store_true",
                    help="show the full sweep run log (no arguments needed)")
    ap.add_argument("--results", default=None,
                    help="CSV run log (seed,parcel,ok,ik) instead of scanning bags")
    ap.add_argument("--compact", action="store_true",
                    help="pack several seeds per line (for 50+ seeds)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    # when piped to a file, keep the ANSI codes out
    if args.no_color or not sys.stdout.isatty():
        GREEN = RED = YELLOW = DIM = RESET = ""

    if args.sweep and not args.results:
        args.results = SWEEP_LOG
        args.compact = True
    if args.results:
        data = {}
        with open(args.results) as f:
            header = f.readline()
            for line in f:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 4 or not parts[0].isdigit():
                    continue
                seed, parcel, ok, ik = (int(parts[0]), int(parts[1]),
                                        int(parts[2] or 0), int(parts[3] or 0))
                prev = data.get(seed, {}).get(parcel)
                if prev is None or (ok, -ik) > (prev[0], -prev[1]):
                    data.setdefault(seed, {})[parcel] = (ok, ik)
        source = "run log %s" % args.results
    else:
        data = scan(args.dir)
        source = "bags in %s" % args.dir
    if not data:
        print("no results (checked %s)" % source)
        return 1

    seeds = sorted(data)
    full_ok = 0
    lines = []
    for seed in seeds:
        marks = []
        good = 0
        for p in range(N_PARCELS):
            v = data[seed].get(p)
            if v is None:
                marks.append(DIM + "·" + RESET)
            elif v[0] == 1 and v[1] == 0:
                marks.append(GREEN + "✓" + RESET)
                good += 1
            elif v[0] == 1:
                marks.append(YELLOW + "!" + RESET)
                good += 1
            else:
                marks.append(RED + "✗" + RESET)
        ok = good == N_PARCELS
        full_ok += int(ok)
        head = "%sseed %-3d%s  %s  %d/%d" % (
            GREEN if ok else RED, seed, RESET,
            " ".join(marks), good, N_PARCELS)
        lines.append((seed, ok, good, head))

    print("\n  source: %s" % source)
    if args.compact:
        print("  %-52s %s" % ("seed   p0 p1 p2 p3", "seed   p0 p1 p2 p3"))
        print("  " + "-" * 72)
        for i in range(0, len(lines), 2):
            cell = []
            for seed, ok, good, head in lines[i:i + 2]:
                marks = " ".join("%s%s%s" % (GREEN if (data[seed].get(p) and
                                                      data[seed][p][0] == 1) else RED,
                                             "✓" if data[seed].get(p) and
                                             data[seed][p][0] == 1 else "✗", RESET)
                                 for p in range(N_PARCELS))
                cell.append("%-4d  %s" % (seed, marks))
            print("  " + "     ".join(cell))
    else:
        print("  %-46s %s" % ("seed", "parcels  p0 p1 p2 p3"))
        print("  " + "-" * 60)
        for seed, ok, good, head in lines:
            print("  " + head)

    total_seeds = len(seeds)
    total_parcels = sum(len(v) for v in data.values())
    good_parcels = sum(1 for s in data.values() for (ok, ik) in s.values() if ok)
    print("  " + "-" * 60)
    bags = len(glob.glob(os.path.join(args.dir, "*.bag")))
    print("  " + "=" * 72)
    print("  %s%d/%d seeds complete%s   %s%d/%d parcels ok%s   %s(%d bags kept)%s"
          % (GREEN if full_ok == total_seeds else YELLOW, full_ok, total_seeds, RESET,
             GREEN if good_parcels == total_parcels else YELLOW,
             good_parcels, total_parcels, RESET, DIM, bags, RESET))

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("seed,parcel,ok,ik\n")
            for seed in seeds:
                for p in range(N_PARCELS):
                    v = data[seed].get(p)
                    f.write("%d,%d,%s,%s\n" % (seed, p,
                                               "" if v is None else v[0],
                                               "" if v is None else v[1]))
        print("  csv -> %s" % args.csv)

    if args.json:
        print(json.dumps({
            "seeds": [{"seed": s, "parcels": {str(p): list(v)
                                              for p, v in data[s].items()}}
                      for s in seeds],
            "seeds_complete": full_ok, "seeds_total": total_seeds,
            "parcels_ok": good_parcels, "parcels_total": total_parcels,
        }, indent=2))

    return 0 if full_ok == total_seeds else 2


if __name__ == "__main__":
    sys.exit(main())
