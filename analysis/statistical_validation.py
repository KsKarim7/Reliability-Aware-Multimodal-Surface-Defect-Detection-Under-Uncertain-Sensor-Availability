"""Full statistical validation per 02_STATISTICAL_VALIDATION_PROMPT.md (rewritten version).
Equivalence-primary. statsmodels mixed model + hierarchical bootstrap reported side by side.
"""
import csv, os, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

np.random.seed(20260821)
base = os.path.expanduser("~/MISDD-MM")
CFG = ["innov1_only","innov2_only","innov3_only","innov4_only","innov2_3_4","full_model"]
ALL = ["baseline"]+CFG
SEEDS = [111,222,333]
B = 10000

# ---- STEP 1: equivalence margin, FIXED BEFORE ANY ANALYSIS ----
DELTA = 1.0   # pp. Justification is external (see report). Hard-coded here, not data-derived.

def load(p):
    with open(p) as f: rows = list(csv.reader(f))[1:]
    return {r[0].split("-")[-1]: float(r[1]) for r in rows if len(r) > 1}

def dataset(d):
    return {(s,c): load(f"{base}/{d}/seed{s}/{c}.csv") for s in SEEDS for c in ALL}

DS = {"MVTec3D": dataset("ablation_results_v5_3nn"),
      "Eyescandies": dataset("ablation_results_v5_eyescandies")}

def seed_diffs(tab, c):
    out=[]
    for s in SEEDS:
        b=np.mean(list(tab[(s,'baseline')].values())); m=np.mean(list(tab[(s,c)].values()))
        out.append(m-b)
    return np.array(out)

def class_diffs(tab, c):
    """returns long-form (diff, seed, class)"""
    rec=[]
    for s in SEEDS:
        for cl in tab[(s,c)]:
            rec.append((tab[(s,c)][cl]-tab[(s,'baseline')][cl], s, cl))
    return pd.DataFrame(rec, columns=["diff","seed","cls"])

def tost(d, delta):
    n=len(d); m=d.mean(); se=d.std(ddof=1)/np.sqrt(n); df=n-1
    se=max(se,1e-12)
    p_lo=1-stats.t.cdf((m+delta)/se, df)     # H0: mu <= -delta
    p_hi=stats.t.cdf((m-delta)/se, df)       # H0: mu >= +delta
    tc90=stats.t.ppf(0.95,df); tc95=stats.t.ppf(0.975,df)
    return dict(mean=m, sd=d.std(ddof=1), se=se,
                p_lower=p_lo, p_upper=p_hi, p_tost=max(p_lo,p_hi),
                ci90=(m-tc90*se, m+tc90*se), ci95=(m-tc95*se, m+tc95*se),
                equiv=(m-tc90*se > -delta) and (m+tc90*se < delta))

def hboot(df, B=B):
    """hierarchical: resample seeds, then classes within seed"""
    seeds=df.seed.unique(); out=np.empty(B)
    bysd={s: df[df.seed==s]["diff"].values for s in seeds}
    for b in range(B):
        vals=[]
        for s in np.random.choice(seeds, len(seeds), replace=True):
            v=bysd[s]; vals.append(np.random.choice(v, len(v), replace=True))
        out[b]=np.concatenate(vals).mean()
    return np.percentile(out,[2.5,97.5]), out.mean()

def mixed(df):
    """Crossed random effects: diff ~ 1 + (1|seed) + (1|class).
    statsmodels idiom for crossed REs = single dummy group + two variance components."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            d=df.copy(); d["grp"]=1
            vcf={"seed":"0 + C(seed)", "cls":"0 + C(cls)"}
            md=smf.mixedlm("diff ~ 1", d, groups=d["grp"], re_formula="0", vc_formula=vcf)
            r=md.fit(reml=True, method="lbfgs")
            ci=r.conf_int().loc["Intercept"]
            vc=dict(zip(r.model.exog_vc.names, [float(x) for x in r.vcomp])) if len(r.vcomp) else {}
            return dict(est=float(r.params["Intercept"]), se=float(r.bse["Intercept"]),
                        lo=float(ci.iloc[0]), hi=float(ci.iloc[1]), conv=bool(r.converged),
                        vc={**{k:round(v,5) for k,v in vc.items()}, "resid":round(float(r.scale),5)})
        except Exception as e:
            return dict(est=np.nan, se=np.nan, lo=np.nan, hi=np.nan, conv=False, vc={}, err=str(e)[:60])

def bh(p):
    p=np.asarray(p); m=len(p); o=np.argsort(p); adj=np.empty(m); prev=1.0
    for i in range(m-1,-1,-1):
        prev=min(prev, p[o[i]]*m/(i+1)); adj[o[i]]=min(prev,1.0)
    return adj

print("#"*72); print("# STEP 1 — PRE-SPECIFIED MARGIN"); print("#"*72)
print(f"delta = {DELTA} pp (fixed before analysis; justification external)")

RESULTS={}
for name,tab in DS.items():
    print()
    print("#"*72); print(f"# STEPS 2-4 — {name}"); print("#"*72)
    rows=[]; pt=[]; pw=[]
    for c in CFG:
        sd_=seed_diffs(tab,c); cd=class_diffs(tab,c)
        t=tost(sd_,DELTA); bci,bm=hboot(cd); mm=mixed(cd)
        tt=stats.ttest_rel([np.mean(list(tab[(s,c)].values())) for s in SEEDS],
                           [np.mean(list(tab[(s,'baseline')].values())) for s in SEEDS])
        try: w=stats.wilcoxon(sd_)
        except Exception: w=type("x",(),{"pvalue":np.nan})()
        d_cohen=t["mean"]/t["sd"] if t["sd"]>0 else np.nan
        # bootstrap CI for d (seed-level, n=3)
        bd=[]
        for _ in range(2000):
            r=np.random.choice(sd_,3,replace=True)
            bd.append(r.mean()/r.std(ddof=1) if r.std(ddof=1)>0 else np.nan)
        bd=np.array(bd); dci=(np.nanpercentile(bd,2.5), np.nanpercentile(bd,97.5))
        rows.append(dict(cfg=c, **t, boot=bci, bootmean=bm, mm=mm,
                         p_t=tt.pvalue, p_w=w.pvalue, d=d_cohen, dci=dci))
        pt.append(tt.pvalue); pw.append(w.pvalue)
    adj=bh(pt); adjw=bh(pw)
    for r,a,aw in zip(rows,adj,adjw): r["p_t_bh"]=a; r["p_w_bh"]=aw
    RESULTS[name]=rows

    print(f"\n{'config':<13}{'meanD':>8}{'sd':>7}{'90%CI':>18}{'95%CI':>18}{'p_lo':>8}{'p_hi':>8}{'TOST':>8}{'equiv':>7}")
    for r in rows:
        print(f"{r['cfg']:<13}{r['mean']:>+8.3f}{r['sd']:>7.3f}"
              f"{f'[{r[chr(99)+chr(105)+chr(57)+chr(48)][0]:+.3f},{r[chr(99)+chr(105)+chr(57)+chr(48)][1]:+.3f}]':>18}"
              f"{f'[{r[chr(99)+chr(105)+chr(57)+chr(53)][0]:+.3f},{r[chr(99)+chr(105)+chr(57)+chr(53)][1]:+.3f}]':>18}"
              f"{r['p_lower']:>8.4f}{r['p_upper']:>8.4f}{r['p_tost']:>8.4f}{str(r['equiv']):>7}")
    print(f"\n{'config':<13}{'bootstrap95':>22}{'mixedEst':>10}{'mixed95CI':>22}{'conv':>6}")
    for r in rows:
        m=r['mm']
        print(f"{r['cfg']:<13}{f'[{r[chr(98)+chr(111)+chr(111)+chr(116)][0]:+.3f},{r[chr(98)+chr(111)+chr(111)+chr(116)][1]:+.3f}]':>22}"
              f"{m['est']:>+10.3f}{f'[{m[chr(108)+chr(111)]:+.3f},{m[chr(104)+chr(105)]:+.3f}]':>22}{str(m['conv']):>6}")
    print(f"\n{'config':<13}{'paired_t_p':>11}{'t_BH':>8}{'wilcox_p':>10}{'w_BH':>8}{'cohen_d':>9}{'d_boot95':>20}")
    for r in rows:
        print(f"{r['cfg']:<13}{r['p_t']:>11.4f}{r['p_t_bh']:>8.4f}{r['p_w']:>10.4f}{r['p_w_bh']:>8.4f}"
              f"{r['d']:>+9.2f}{f'[{r[chr(100)+chr(99)+chr(105)][0]:+.1f},{r[chr(100)+chr(99)+chr(105)][1]:+.1f}]':>20}")
    widest=max(max(abs(r['ci95'][0]),abs(r['ci95'][1])) for r in rows)
    widest90=max(max(abs(r['ci90'][0]),abs(r['ci90'][1])) for r in rows)
    print(f"\nACHIEVED bound (widest 95% CI endpoint): {widest:.3f} pp")
    print(f"ACHIEVED bound (widest 90% CI endpoint, TOST-equivalent): {widest90:.3f} pp")
    mmv=RESULTS[name][0]['mm']
    print(f"variance components (example, {rows[0]['cfg']}): {rows[0]['mm'].get('vc')}")

print()
print("#"*72); print("# STEP 3b — per-class descriptive notes"); print("#"*72)
for name,tab in DS.items():
    print(f"\n--- {name}: largest |per-class config-baseline| deviations ---")
    dev=[]
    for s in SEEDS:
        for c in CFG:
            for cl in tab[(s,c)]:
                dev.append((tab[(s,c)][cl]-tab[(s,'baseline')][cl], c, s, cl))
    dev.sort(key=lambda x:-abs(x[0]))
    for v,c,s,cl in dev[:6]: print(f"   {v:+.2f}  {c} seed{s} {cl}")
    print(f"   class difficulty range: ", end="")
    cm={cl: np.mean([tab[(s,c)][cl] for s in SEEDS for c in ALL]) for cl in tab[(111,'baseline')]}
    lo=min(cm.items(),key=lambda x:x[1]); hi=max(cm.items(),key=lambda x:x[1])
    print(f"{lo[0]} {lo[1]:.1f} ... {hi[0]} {hi[1]:.1f}")
    # peach/seed333 style check: any class+seed where multiple configs agree in sign & size
    print("   consistent per-class clusters (|mean over configs| > 0.3):")
    for s in SEEDS:
        for cl in tab[(111,'baseline')]:
            vals=[tab[(s,c)][cl]-tab[(s,'baseline')][cl] for c in CFG]
            if abs(np.mean(vals))>0.3:
                print(f"      seed{s} {cl}: mean over 6 configs {np.mean(vals):+.2f} (range {min(vals):+.2f}..{max(vals):+.2f})")

print()
print("#"*72); print("# STEP 5 — CROSS-DATASET"); print("#"*72)
bm={n: np.array([np.mean(list(t[(s,'baseline')].values())) for s in SEEDS]) for n,t in DS.items()}
for n,v in bm.items(): print(f"{n} baseline seed-means {np.round(v,2)} SD(n-1)={v.std(ddof=1):.3f}")
lev=stats.levene(bm["MVTec3D"],bm["Eyescandies"])
F=bm["Eyescandies"].var(ddof=1)/bm["MVTec3D"].var(ddof=1)
pF=2*min(stats.f.cdf(F,2,2),1-stats.f.cdf(F,2,2))
print(f"Levene p={lev.pvalue:.4f} | F(var ratio Eyes/MVTec)={F:.2f} p={pF:.4f}")
print("\nconfig-vs-baseline difference: does it differ between datasets?")
pl=[]
for c in CFG:
    a=seed_diffs(DS["MVTec3D"],c); b=seed_diffs(DS["Eyescandies"],c)
    tt=stats.ttest_ind(a,b)
    pl.append(tt.pvalue)
    print(f"  {c:<13} MVTec {a.mean():+.3f} vs Eyes {b.mean():+.3f}  indep-t p={tt.pvalue:.4f}")
print(f"  BH-adjusted: {np.round(bh(pl),4)}")

print()
print("#"*72); print("# STEP 6 — 3-NN CLAIM"); print("#"*72)
ck=f"{base}/result/mvtec3d/both/0.7/checkpoint"
n_ck=len([f for f in os.listdir(ck)]) if os.path.isdir(ck) else 0
print(f"checkpoint dir exists: {os.path.isdir(ck)} | .pt files: {n_ck}")
print("=> rescoring seeds 222/333 at k=1 is IMPOSSIBLE without retraining\n")
v5=f"{base}/ablation_results_v5"
for s in SEEDS:
    p=f"{v5}/seed{s}"
    print(f"1-NN record seed{s}: {'PRESENT ('+str(len(os.listdir(p)))+' configs)' if os.path.isdir(p) and os.listdir(p) else 'ABSENT'}")
print("\nper-config seed-111 k=1 vs k=3 (both from committed campaign CSVs):")
print(f"{'config':<13}{'1-NN':>8}{'3-NN':>8}{'delta':>8}")
k1={},
K1={};K3={}
for c in ALL:
    p1=f"{v5}/seed111/{c}.csv"
    if not os.path.exists(p1): continue
    d1=load(p1); d3=load(f"{base}/ablation_results_v5_3nn/seed111/{c}.csv")
    K1[c]=d1;K3[c]=d3
    print(f"{c:<13}{np.mean(list(d1.values())):>8.2f}{np.mean(list(d3.values())):>8.2f}"
          f"{np.mean(list(d3.values()))-np.mean(list(d1.values())):>+8.2f}")
print("\nper-class delta, baseline (the config the README table's mean matches):")
dl=[(cl, K3['baseline'][cl]-K1['baseline'][cl]) for cl in sorted(K1['baseline'])]
for cl,v in dl: print(f"   {cl:<13}{v:>+7.2f}")
vals=[v for _,v in dl]
print(f"   mean {np.mean(vals):+.3f} | improved {sum(1 for v in vals if v>0)}/10 | max {max(vals):+.2f} ({dl[int(np.argmax(vals))][0]})")
print(f"   mean excluding largest contributor: {np.mean(sorted(vals)[:-1]):+.3f}")
print("\npaired t on per-class k3-k1 (baseline, n=10 classes, seed111 only):")
tt=stats.ttest_rel([K3['baseline'][c] for c in sorted(K1['baseline'])],
                   [K1['baseline'][c] for c in sorted(K1['baseline'])])
print(f"   t p={tt.pvalue:.4f}  (single seed; NOT a held-out test)")

print()
print("#"*72); print("# STEP 7 — MISSING-RATE"); print("#"*72)
ETAS=[0.3,0.5,0.7,0.9]
def mr(eta,s,c):
    p=(f"{base}/ablation_results_v5_3nn/seed{s}/{c}.csv" if eta==0.7
       else f"{base}/ablation_results_v5_missing_rate/eta{eta}/seed{s}/{c}.csv")
    return np.mean(list(load(p).values()))
print(f"{'eta':>5}{'baseline':>10}{'full':>8}{'gap':>8}{'gap95CI':>20}")
for e in ETAS:
    b=[mr(e,s,'baseline') for s in SEEDS]; f_=[mr(e,s,'full_model') for s in SEEDS]
    g=np.array(f_)-np.array(b); se=g.std(ddof=1)/np.sqrt(3); tc=stats.t.ppf(0.975,2)
    print(f"{e:>5}{np.mean(b):>10.2f}{np.mean(f_):>8.2f}{g.mean():>+8.3f}"
          f"{f'[{g.mean()-tc*se:+.2f},{g.mean()+tc*se:+.2f}]':>20}")
sl={}
for c in ('baseline','full_model'):
    sl[c]=[np.polyfit(ETAS,[mr(e,s,c) for e in ETAS],1)[0] for s in SEEDS]
    print(f"{c} per-seed slopes {np.round(sl[c],2)} mean {np.mean(sl[c]):+.3f} pp/unit-eta")
tt=stats.ttest_rel(sl['baseline'],sl['full_model'])
print(f"slope difference paired-t p={tt.pvalue:.4f}")
for c in ('baseline','full_model'):
    d=[mr(0.9,s,c)-mr(0.3,s,c) for s in SEEDS]
    print(f"{c} absolute degradation eta .3->.9: {np.mean(d):+.2f} pp (per-seed {np.round(d,2)})")

print()
print("#"*72); print("# STEP 8 — GATE-3 PILOT"); print("#"*72)
for c in ("baseline","full_model"):
    pil=load(f"{base}/result_diag_gate3_{c}/mvtec3d/both/0.7/csv/Seed_111-results.csv")
    pil={k:v for k,v in pil.items() if v>0}
    ref=load(f"{base}/ablation_results_v5_3nn/seed111/{c}.csv")
    cls=sorted(pil); d=np.array([pil[k]-ref[k] for k in cls])
    tt=stats.ttest_rel([pil[k] for k in cls],[ref[k] for k in cls])
    se=d.std(ddof=1)/np.sqrt(len(d)); tc=stats.t.ppf(0.975,len(d)-1)
    print(f"{c}: classes={cls}")
    print(f"   pilot {[round(pil[k],2) for k in cls]}")
    print(f"   ref   {[round(ref[k],2) for k in cls]}")
    print(f"   diff  {np.round(d,2)}  mean {d.mean():+.3f} CI95 [{d.mean()-tc*se:+.2f},{d.mean()+tc*se:+.2f}] worst {d.min():+.2f}  paired-t p={tt.pvalue:.3f}")
    print(f"   PRE-REGISTERED criterion (mean>+0.5 AND min>-1.0): {'PASS' if d.mean()>0.5 and d.min()>-1.0 else 'FAIL'}")

print()
print("#"*72); print("# STEP 9 — VERIFICATION"); print("#"*72)
p=f"{base}/ablation_results_v5_3nn/seed111/baseline.csv"
d=load(p)
print(f"re-derived MVTec seed111 baseline mean from 10 per-class rows: {np.mean(list(d.values())):.4f}")
print(f"   (report/CLAUDE value: 77.57)")
print(f"n classes={len(d)}; values={[round(v,2) for v in d.values()]}")
