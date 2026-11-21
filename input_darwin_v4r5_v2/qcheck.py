#!/usr/bin/env python3
"""
qcheck.py - pre-submission check for MPI jobs on DTU's LSF 'hpc' queue.

For each ptile it estimates the earliest time the job could start inside one
enclosure with a single node model (the hpc queue's same[type:model] and
cu[maxcus=1] rules), using:
  - free cores now        (nodestat -F <queue>)
  - enclosure layout      (bmgroup -cu)
  - when running jobs end (bjobs -u all -r, counted to their wall-time limit)

Optionally checks the executable for AVX-512 and an MITgcm run directory for
broken links / missing pickup files.

Examples:
  python3 qcheck.py -n 113
  python3 qcheck.py -n 113 --mem 4 --exe ./mitgcmuv --rundir .
"""
import argparse, glob, math, os, re, subprocess, sys
from collections import defaultdict
from datetime import datetime, timedelta

SKIP_STATES = ("down", "closed", "unavail", "unreach")


def run(cmd, check=True):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        sys.exit(f"Could not run '{' '.join(cmd)}': {e}")
    if check and r.returncode != 0:
        sys.exit(f"'{' '.join(cmd)}' failed:\n{r.stderr}")
    return r.stdout


# ---------------------------------------------------------------- cluster data
def read_nodes(queue, skip_busy):
    """Parse 'nodestat -F <queue>'. Procs column is free:total."""
    nodes = {}
    for line in run(["nodestat", "-F", queue]).splitlines():
        p = line.split()
        if len(p) < 7 or p[0] == "Node" or ":" not in p[2]:
            continue
        name, state, procs, model = p[0], p[1], p[2], p[4]
        try:
            free, total = (int(x) for x in procs.split(":"))
            mem = float(p[5])
        except ValueError:
            continue
        feats = " ".join(p[7:]).strip("()").split()
        st = state.lower()
        usable = not any(s in st for s in SKIP_STATES) and not (skip_busy and st == "busy")
        nodes[name] = dict(name=name, state=state, free=free, total=total,
                           model=model, mem=mem, feats=feats, usable=usable)
    return nodes


def read_enclosures():
    """Parse 'bmgroup -cu' and return (name, [hosts]) for each enclosure line."""
    out = []
    for line in run(["bmgroup", "-cu"]).splitlines():
        p = line.split()
        if len(p) >= 3 and p[1] == "enclosure":
            out.append((p[0], p[2:]))
    return out


def parse_hms(s):
    m = re.match(r"^(\d+):(\d+)(?::(\d+))?$", s)
    if not m:
        return None
    return int(m.group(1)) + int(m.group(2)) / 60 + int(m.group(3) or 0) / 3600


def read_running_jobs():
    """Return {host: [(hours_left, cores), ...]} for all running jobs."""
    out = run(["bjobs", "-u", "all", "-r", "-o", "jobid exec_host time_left"], check=False)
    ends = defaultdict(list)
    for line in out.splitlines():
        p = line.split()
        if len(p) < 3 or not p[0][0].isdigit() or p[1] == "-":
            continue
        left = parse_hms(p[2])
        if left is None:            # no run limit -> may never end
            left = math.inf
        for part in p[1].split(":"):
            if "*" in part:
                k, host = part.split("*", 1)
                k = int(k)
            else:
                k, host = 1, part
            ends[host].append((left, k))
    return ends


# ---------------------------------------------------------------- pre-checks
def count_zmm(exe):
    """Count instructions using AVX-512 (zmm) registers."""
    if not os.path.isfile(exe):
        print(f"  !   executable '{exe}' not found")
        return None
    try:
        proc = subprocess.Popen(["objdump", "-d", exe], stdout=subprocess.PIPE,
                                text=True, errors="ignore")
    except OSError as e:
        print(f"  !   could not run objdump: {e}")
        return None
    n = sum(1 for line in proc.stdout if "zmm" in line)
    proc.wait()
    return n


def check_rundir(d):
    broken = [f for f in sorted(os.listdir(d))
              if os.path.islink(os.path.join(d, f)) and not os.path.exists(os.path.join(d, f))]
    if broken:
        print(f"  !   {len(broken)} broken link(s):")
        for f in broken[:20]:
            print(f"        {f}")
        if len(broken) > 20:
            print("        ...")
    else:
        print("  ok  no broken links")

    datafile = os.path.join(d, "data")
    if not os.path.isfile(datafile):
        print("  ?   no 'data' file found, skipping pickup check")
        return
    with open(datafile, errors="ignore") as fh:
        txt = fh.read()
    m = re.search(r"^\s*niter0\s*=\s*(\d+)", txt, re.I | re.M)
    if not m:
        print("  ?   nIter0 not found in 'data' (maybe startTime is used) - check pickups by hand")
        return
    it = int(m.group(1))
    if it == 0:
        print("  ok  nIter0 = 0 (cold start, no pickup files needed)")
        return
    found = sorted(glob.glob(os.path.join(d, f"pickup*.{it:010d}.data")))
    if not found:
        print(f"  !   nIter0 = {it} but no pickup*.{it:010d}.data files found")
        return
    print(f"  ok  nIter0 = {it}; pickup files found:")
    for f in found:
        tag = "" if os.path.exists(f[:-5] + ".meta") else "   <-- .meta missing!"
        print(f"        {os.path.basename(f)}{tag}")
    print("        (check that every package you use - seaice, ptracers, darwin, ... - has one)")


# ---------------------------------------------------------------- simulation
def free_at(node, t, ends, mem_ok):
    """Free cores on a node t hours from now (worst case: jobs run to their limit)."""
    if not node["usable"] or not mem_ok:
        return 0
    f = node["free"] + sum(k for left, k in ends.get(node["name"], ()) if left <= t)
    return min(f, node["total"])


def fits(frees, k, p, rem):
    """frees sorted descending. Need k-1 nodes with >=p free plus one with >=rem."""
    return len(frees) >= k and (k == 1 or frees[k - 2] >= p) and frees[k - 1] >= rem


def earliest_start(group, k, p, rem, ends, mem_per_core):
    nodes = group["nodes"]
    mem_ok = {n["name"]: n["mem"] >= p * mem_per_core for n in nodes}
    times = {0.0}
    for n in nodes:
        times.update(left for left, _ in ends.get(n["name"], ()) if math.isfinite(left))
    for t in sorted(times):
        frees = sorted((free_at(n, t, ends, mem_ok[n["name"]]) for n in nodes), reverse=True)
        if fits(frees, k, p, rem):
            spare = sum(1 for x in frees if x >= p) - (k if rem == p else k - 1)
            return t, spare
    return math.inf, 0


def fmt_time(t):
    if t == 0:
        return "now"
    if math.isinf(t):
        return "never*"
    mins = int(round(t * 60))
    when = (datetime.now() + timedelta(minutes=mins)).strftime("%a %H:%M")
    return f"{mins // 60}h{mins % 60:02d} ({when})"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Estimate the best ptile for an MPI job on DTU's LSF cluster.")
    ap.add_argument("-n", "--cores", type=int, required=True, help="number of MPI processes (bsub -n)")
    ap.add_argument("--mem", type=float, default=4.0, help="GB per core requested in rusage (default 4)")
    ap.add_argument("-q", "--queue", default="hpc")
    ap.add_argument("--ptiles", default="4,6,8,12,16,20,24,32", help="ptile values to test")
    ap.add_argument("--exe", help="executable to check for AVX-512 instructions")
    ap.add_argument("--rundir", help="MITgcm run directory to check")
    ap.add_argument("--avx512", action="store_true", help="only consider AVX-512 nodes")
    ap.add_argument("--skip-busy", action="store_true", help="treat nodes in 'Busy' state as unusable")
    a = ap.parse_args()
    N = a.cores
    need_avx512 = a.avx512

    # ---- pre-checks
    if a.exe or a.rundir:
        print("=== Pre-checks ===")
        if a.exe:
            n = count_zmm(a.exe)
            if n:
                need_avx512 = True
                print(f"  !   {a.exe} uses AVX-512 ({n} zmm instructions): only AVX-512 nodes considered")
            elif n == 0:
                print(f"  ok  {a.exe} has no AVX-512 instructions: runs on all node types")
        if a.rundir:
            check_rundir(a.rundir)
        print()

    # ---- cluster snapshot
    nodes = read_nodes(a.queue, a.skip_busy)
    ends = read_running_jobs()
    groups = []
    for name, hosts in read_enclosures():
        bymodel = defaultdict(list)
        for h in hosts:
            nd = nodes.get(h)
            if nd is None or (need_avx512 and "avx512" not in nd["feats"]):
                continue
            bymodel[nd["model"]].append(nd)
        for model, lst in bymodel.items():
            lst.sort(key=lambda x: x["name"])
            groups.append(dict(
                label=f"{name}: {lst[0]['name']}.. ({len(lst)} nodes, {model}, {lst[0]['total']} cores)",
                nodes=lst))

    # ---- evaluate each ptile
    ptiles = sorted({int(x) for x in a.ptiles.split(",") if x.strip()})
    print(f"=== Earliest start for a {N}-core job (queue '{a.queue}', {a.mem:g} GB/core) ===")
    print(f"{'ptile':>5} {'nodes':>5}  {'earliest start (worst case)':<28} best enclosure")
    rows = []
    for p in ptiles:
        k = math.ceil(N / p)
        rem = N - (k - 1) * p
        best = None
        for g in groups:
            if p > g["nodes"][0]["total"] or len(g["nodes"]) < k:
                continue
            t, spare = earliest_start(g, k, p, rem, ends, a.mem)
            if best is None or (t, -spare) < (best["t"], -best["spare"]):
                best = dict(t=t, spare=spare, g=g)
        if best is None:
            print(f"{p:>5} {k:>5}  {'-':<28} no enclosure is big enough")
            continue
        rows.append(dict(p=p, k=k, **best))
        print(f"{p:>5} {k:>5}  {fmt_time(best['t']):<28} {best['g']['label']}")

    if not rows:
        sys.exit("\nNo usable enclosure found.")

    # ---- recommendation: earliest start; within 15 min prefer larger ptile (fewer nodes)
    best_t = min(r["t"] for r in rows)
    if math.isinf(best_t):
        sys.exit("\nNo ptile can be placed even after all running jobs end (check memory/nodes).")
    close = [r for r in rows if r["t"] <= best_t + 0.25]
    if best_t == 0:
        safe = [r for r in close if r["spare"] > 0]
        pick = max(safe or close, key=lambda r: r["p"])
        note = "fits right now" + ("" if safe else ", but with no spare nodes (may be taken first)")
    else:
        pick = max(close, key=lambda r: r["p"])
        note = f"earliest worst-case start in {fmt_time(pick['t'])}"
    print(f"\nRecommendation: ptile={pick['p']} ({pick['k']} nodes) - {note}.")
    print("\nPut in your job script:")
    print(f'  #BSUB -n {N}')
    print(f'  #BSUB -R "span[ptile={pick["p"]}]"')
    if need_avx512:
        print('  #BSUB -R "select[avx512]"')

    print("\nNotes:")
    print(" - Times assume running jobs use their full wall-time limit; real starts are often earlier.")
    print(" - Pending jobs with higher priority (and their reservations) are NOT taken into account,")
    print("   so a big job can wait longer than shown.")
    if any(math.isinf(r["t"]) for r in rows):
        print(" * 'never': blocked by jobs without a run limit, or not enough memory per node.")


if __name__ == "__main__":
    main()