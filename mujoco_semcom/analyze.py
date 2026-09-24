import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon
import matplotlib.pyplot as plt

def paired_binary(df,method,baseline,outcome="success"):
    p=df.pivot(index="seed",columns="policy",values=outcome).dropna()
    a,b=p[method].astype(int),p[baseline].astype(int)
    w=int(((a==1)&(b==0)).sum()); l=int(((a==0)&(b==1)).sum()); n=w+l
    pv=binomtest(w,n,.5,alternative="greater").pvalue if n else 1.
    return w,l,pv

def paired_cont(df,method,baseline,metric):
    p=df.pivot(index="seed",columns="policy",values=metric).dropna(); d=p[method]-p[baseline]
    if np.allclose(d,0): return float(d.mean()),1.
    return float(d.mean()),float(wilcoxon(d,alternative="two-sided",zero_method="wilcox").pvalue)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("csv",type=Path); ap.add_argument("--outdir",type=Path,default=Path("results")); a=ap.parse_args()
    df=pd.read_csv(a.csv); a.outdir.mkdir(parents=True,exist_ok=True)
    summary=df.groupby(["network_profile","n_distractors","policy"]).agg(
      n=("seed","count"),success_rate=("success","mean"),collision_rate=("collision","mean"),timeout_rate=("timeout","mean"),
      completion_s=("completion_s","mean"),action_gap=("mean_action_gap","mean"),p95_action_gap=("p95_action_gap","mean"),
      mean_aoi_s=("mean_aoi_s","mean"),p95_aoi_s=("p95_aoi_s","mean"),mean_kaoi_m=("mean_kaoi_m","mean"),
      relevant_tx=("sent_relevant_obs_frac","mean"),distractor_tx=("sent_distractor_frac","mean")).reset_index()
    summary.to_csv(a.outdir/"summary.csv",index=False)
    tests=[]
    for (prof,nd),sub in df.groupby(["network_profile","n_distractors"]):
      for base in ["round_robin","aoi","kaoi","error","saoi"]:
       if base not in set(sub.policy): continue
       w,l,p=paired_binary(sub,"action_impact",base); diff,p2=paired_cont(sub,"action_impact",base,"mean_action_gap")
       tests.append(dict(network_profile=prof,n_distractors=nd,baseline=base,success_wins=w,success_losses=l,success_exact_p=p,action_gap_mean_diff=diff,action_gap_wilcoxon_p=p2))
    tests=pd.DataFrame(tests); tests.to_csv(a.outdir/"paired_tests.csv",index=False)
    for prof in sorted(df.network_profile.unique()):
      s=summary[summary.network_profile==prof]; fig,ax=plt.subplots(figsize=(7.2,4.6))
      for pol in ["aoi","kaoi","error","saoi","action_impact"]:
       x=s[s.policy==pol].sort_values("n_distractors")
       if len(x): ax.plot(x.n_distractors,100*x.success_rate,marker="o",label=pol)
      ax.set_xlabel("Number of dynamic distractors"); ax.set_ylabel("Task success rate (%)"); ax.set_ylim(0,105)
      ax.grid(True,alpha=.25); ax.legend(); ax.set_title(prof); fig.tight_layout(); fig.savefig(a.outdir/f"success_vs_distractors_{prof}.png",dpi=180); plt.close(fig)
    print(summary.to_string(index=False)); print("\nPAIRED TESTS\n",tests.to_string(index=False))
if __name__=="__main__": main()
