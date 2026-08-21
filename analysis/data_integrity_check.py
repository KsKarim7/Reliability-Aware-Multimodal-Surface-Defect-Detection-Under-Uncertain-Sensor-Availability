"""Fresh verification of every canonical CSV. Assumes nothing from prior sessions."""
import csv, os, glob
import numpy as np

base = os.path.expanduser("~/MISDD-MM")
CFG = ["baseline","innov1_only","innov2_only","innov3_only","innov4_only","innov2_3_4","full_model"]
SEEDS = [111,222,333]

def load(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    hdr, body = rows[0], rows[1:]
    return hdr, {r[0].split("-")[-1]: [float(x) for x in r[1:]] for r in body if len(r) > 1}

print("="*70)
print("A. STRUCTURAL INTEGRITY — every canonical CSV")
print("="*70)
problems = []
allfiles = []
for d in ["ablation_results_v5_3nn","ablation_results_v5_eyescandies",
          "ablation_results_v5_missing_rate","ablation_results_v5"]:
    allfiles += sorted(glob.glob(f"{base}/{d}/**/*.csv", recursive=True))
allfiles += sorted(glob.glob(f"{base}/result_diag_gate3_*/**/*.csv", recursive=True))
for f in allfiles:
    with open(f) as fh:
        lines = fh.read().strip().split("\n")
    n = len(lines)
    hdr, d = load(f)
    zeros = [k for k,v in d.items() if v[0]==0 or v[1]==0 or v[2]==0]
    rel = f.replace(base+"/","")
    tag = ""
    if n != 11: tag += f" LINES={n}"
    if zeros: tag += f" ZEROROWS={len(zeros)}:{','.join(zeros[:4])}"
    if tag: problems.append(rel+tag)
print(f"total canonical CSVs found: {len(allfiles)}")
print(f"header of first file: {load(allfiles[0])[0]}")
if problems:
    print("\nfiles with non-standard shape or zero rows:")
    for p in problems: print("  ", p)
else:
    print("all files: 11 lines, no zero rows")

print()
print("="*70)
print("B. MAIN TABLES — per-config per-seed means (recomputed)")
print("="*70)
DS = {}
for name, d in [("MVTec3D","ablation_results_v5_3nn"), ("Eyescandies","ablation_results_v5_eyescandies")]:
    tab = {}
    for s in SEEDS:
        for c in CFG:
            _, dd = load(f"{base}/{d}/seed{s}/{c}.csv")
            tab[(s,c)] = dd
    DS[name] = tab
    print(f"\n--- {name} ---")
    print(f"{'config':<13}" + "".join(f"{s:>9}" for s in SEEDS) + f"{'mean':>9}")
    for c in CFG:
        ms = [np.mean([v[0] for v in tab[(s,c)].values()]) for s in SEEDS]
        print(f"{c:<13}" + "".join(f"{m:>9.2f}" for m in ms) + f"{np.mean(ms):>9.2f}")
    cls = sorted(tab[(111,'baseline')].keys())
    print(f"classes ({len(cls)}): {', '.join(cls)}")

print()
print("="*70)
print("C. PAIRED DIFFERENCES (config - baseline) — recomputed fresh")
print("="*70)
for name, tab in DS.items():
    print(f"\n--- {name} ---")
    print(f"{'config':<13}{'seedMean':>10}{'seedSD(n-1)':>13}{'perClassSD':>12}")
    for c in CFG[1:]:
        sd_ = []
        pc = []
        for s in SEEDS:
            b = np.mean([v[0] for v in tab[(s,'baseline')].values()])
            m = np.mean([v[0] for v in tab[(s,c)].values()])
            sd_.append(m-b)
            for k in tab[(s,c)]:
                pc.append(tab[(s,c)][k][0] - tab[(s,'baseline')][k][0])
        print(f"{c:<13}{np.mean(sd_):>+10.4f}{np.std(sd_,ddof=1):>13.4f}{np.std(pc,ddof=1):>12.4f}")
    bm = [np.mean([v[0] for v in tab[(s,'baseline')].values()]) for s in SEEDS]
    print(f"  baseline seed-means {np.round(bm,2)}  SD(n-1)={np.std(bm,ddof=1):.3f}")

print()
print("="*70)
print("D. 3-NN k-SELECTION DATA — what exists")
print("="*70)
v5 = f"{base}/ablation_results_v5"
for s in SEEDS:
    fs = sorted(glob.glob(f"{v5}/seed{s}/*.csv"))
    print(f"ablation_results_v5/seed{s} (1-NN): {len(fs)} configs {[os.path.basename(x)[:-4] for x in fs]}")
print()
print("per-config seed-111 means: 1-NN (v5) vs 3-NN (v5_3nn)")
print(f"{'config':<13}{'1-NN':>9}{'3-NN':>9}{'delta':>9}")
k1_all, k3_all = {}, {}
for c in CFG:
    p1 = f"{v5}/seed111/{c}.csv"
    if not os.path.exists(p1):
        print(f"{c:<13}{'ABSENT':>9}"); continue
    _, d1 = load(p1); _, d3 = load(f"{base}/ablation_results_v5_3nn/seed111/{c}.csv")
    k1_all[c], k3_all[c] = d1, d3
    m1 = np.mean([v[0] for v in d1.values()]); m3 = np.mean([v[0] for v in d3.values()])
    print(f"{c:<13}{m1:>9.2f}{m3:>9.2f}{m3-m1:>+9.2f}")
print()
print("per-class k1->k3 for full_model and baseline (seed 111):")
for c in ("baseline","full_model"):
    if c in k1_all:
        print(f"  {c}:")
        for cl in sorted(k1_all[c]):
            print(f"    {cl:<13}{k1_all[c][cl][0]:>8.2f} -> {k3_all[c][cl][0]:>8.2f}  {k3_all[c][cl][0]-k1_all[c][cl][0]:>+7.2f}")

print()
print("="*70)
print("E. MISSING-RATE COVERAGE")
print("="*70)
for eta in ["0.3","0.5","0.9"]:
    for s in SEEDS:
        fs = sorted(glob.glob(f"{base}/ablation_results_v5_missing_rate/eta{eta}/seed{s}/*.csv"))
        print(f"eta{eta}/seed{s}: {[os.path.basename(x)[:-4] for x in fs]}")

print()
print("="*70)
print("F. GATE-3 PILOT CSVs")
print("="*70)
for c in ("baseline","full_model"):
    p = f"{base}/result_diag_gate3_{c}/mvtec3d/both/0.7/csv/Seed_111-results.csv"
    _, d = load(p)
    real = {k:v for k,v in d.items() if v[0] > 0}
    zero = [k for k,v in d.items() if v[0] == 0]
    print(f"{c}: {len(d)} rows | real={sorted(real)} | zero-placeholders={sorted(zero)}")
    for k in sorted(real): print(f"    {k:<10}{real[k][0]:>8.2f}")
