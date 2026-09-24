from __future__ import annotations
import argparse, heapq
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd
from profiles import get_profile

COMM_DT=.05; MAX_STEPS=240; NOBS=8; NENT=9
LINKS=np.array([.34,.30,.24]); EEF_R=.035
OBS_R=np.array([.07,.07,.07,.06,.06,.06,.06,.06])
SAFE=.25; VMAX=.48; QDMAX=2.1
POLICIES=("round_robin","aoi","kaoi","error","saoi","action_impact","action_impact_error")

def clipnorm(v,m):
    n=float(np.linalg.norm(v))
    return v if n<=m or n<1e-12 else v*(m/n)

def fk(q):
    a=np.cumsum(q[:3])
    return np.array([np.sum(LINKS*np.cos(a)),np.sum(LINKS*np.sin(a))])

def jac(q):
    a=np.cumsum(q[:3]); s=np.sin(a); c=np.cos(a); J=np.empty((2,3))
    for k in range(3):
        J[0,k]=-np.sum(LINKS[k:]*s[k:]); J[1,k]=np.sum(LINKS[k:]*c[k:])
    return J

def ik(q,v,d=.06):
    J=jac(q); A=J@J.T+(d*d)*np.eye(2)
    return np.clip(J.T@np.linalg.solve(A,v),-QDMAX,QDMAX)

def scene(seed,t,n_distractors=5):
    rng=np.random.default_rng(seed); ph=rng.uniform(0,2*np.pi,NOBS+1)
    tb=np.array([.68,-.20])+rng.normal(0,[.018,.018]); w=.42+.08*rng.random()
    target=tb+np.array([.018*np.sin(w*t+ph[0]),.03*np.sin(.7*w*t+ph[0])])
    tv=np.array([.018*w*np.cos(w*t+ph[0]),.03*.7*w*np.cos(.7*w*t+ph[0])])
    base=np.array([[.30,.25],[.43,.08],[.55,-.055]])+rng.normal(0,.018,(3,2))
    pos=np.zeros((NOBS,2)); vel=np.zeros((NOBS,2))
    for i in range(3):
        amp=.15+.025*rng.random(); wi=.82+.22*rng.random()
        pos[i]=base[i]+[.025*np.sin(.55*wi*t+ph[i+1]),amp*np.sin(wi*t+ph[i+1])]
        vel[i]=[.025*.55*wi*np.cos(.55*wi*t+ph[i+1]),amp*wi*np.cos(wi*t+ph[i+1])]
    db=np.array([[-.48,-.48],[-.50,.34],[0,-.68],[-.66,.02],[.05,.73]])+rng.normal(0,.015,(5,2))
    for k in range(5):
        i=3+k; moving=k<n_distractors
        amp=(.18+.035*rng.random()) if moving else .005
        wi=(1.35+.45*rng.random()) if moving else .25
        direction=np.array([1.,.35*(-1 if k%2 else 1)]); direction/=np.linalg.norm(direction)
        pos[i]=db[k]+amp*np.sin(wi*t+ph[i+1])*direction
        vel[i]=amp*wi*np.cos(wi*t+ph[i+1])*direction
    return target,tv,pos,vel

def operator(ee,target,obs):
    to=target-ee; dg=np.linalg.norm(to); gd=to/(dg+1e-9); v=.92*to
    for i,o in enumerate(obs):
        r=o-ee; d=np.linalg.norm(r)+1e-9; surf=d-(EEF_R+OBS_R[i])
        if surf>=SAFE: continue
        rh=r/d; ahead=max(0.,float(np.dot(rh,gd))); rel=.18+1.55*ahead
        if ahead<.05 and surf>.10: rel*=.10
        inv=max(0.,1/max(surf,.012)-1/SAFE); rep=min(.78,.030*inv)*rel
        away=-rh; cross=gd[0]*rh[1]-gd[1]*rh[0]; side=-1. if cross>=0 else 1.
        tang=side*np.array([-rh[1],rh[0]])
        v+=rep*away+.18*rep*tang
    return clipnorm(v,VMAX)

def states(t,o): return np.vstack([t[None,:],o])
def vels(tv,ov): return np.vstack([tv[None,:],ov])

def choose(policy,step,ee,true,truev,rx,stamps):
    ages=(step-stamps)*COMM_DT; err=np.linalg.norm(true-rx,axis=1)
    if policy=="round_robin": return step%NENT
    if policy=="aoi": score=ages
    elif policy=="kaoi": score=ages*np.linalg.norm(truev,axis=1)
    elif policy=="error": score=err
    elif policy=="saoi": score=ages*err
    else:
        u0=operator(ee,rx[0],rx[1:]); score=np.zeros(NENT)
        for i in range(NENT):
            cf=rx.copy(); cf[i]=true[i]
            score[i]=np.linalg.norm(operator(ee,cf[0],cf[1:])-u0)
        if policy=="action_impact_error": score*=.03+err
    return int(np.argmax(score+1e-9*ages+1e-12*(NENT-np.arange(NENT))))

def network(seed,prof,n=MAX_STEPS):
    rng=np.random.default_rng(seed+99173)
    loss=rng.random(n)<prof.get("loss_pct",0)/100
    mean=float(prof.get("delay_ms",0)); jit=float(prof.get("jitter_ms",0))
    dm=rng.normal(mean,jit,n) if jit else np.full(n,mean)
    dm=np.clip(dm,0,mean+4*jit+1e-9)
    return loss,np.ceil(dm/(COMM_DT*1000)).astype(int),dm

def set_mocap(m,d,t,o):
    d.mocap_pos[m.body("target").mocapid]=[t[0],t[1],.08]
    for i in range(NOBS): d.mocap_pos[m.body(f"obs{i}").mocapid]=[o[i,0],o[i,1],.08]

@dataclass
class Packet:
    arrival:int; seq:int; ent:int; gen:int; state:np.ndarray

def run(xml,seed,policy,profile_name,family="paper",n_distractors=5):
    import mujoco
    prof=get_profile(profile_name,family); losses,delays,dms=network(seed,prof)
    m=mujoco.MjModel.from_xml_path(str(xml)); d=mujoco.MjData(m)
    d.qpos[:3]=[2.60,-1.70,-.70]; mujoco.mj_forward(m,d)
    t0,tv0,o0,ov0=scene(seed,0,n_distractors); set_mocap(m,d,t0,o0); mujoco.mj_forward(m,d)
    rx=states(t0,o0).copy(); stamps=np.zeros(NENT,int); pending=[]; seq=0
    sub=int(round(COMM_DT/m.opt.timestep)); action_gap=[]; aois=[]; kao=[]; clr=[]
    sent=np.zeros(NENT,int); success=collision=hold=0
    for step in range(MAX_STEPS):
        tt=step*COMM_DT; tar,tv,obs,ov=scene(seed,tt,n_distractors); tr=states(tar,obs); trv=vels(tv,ov)
        set_mocap(m,d,tar,obs); mujoco.mj_forward(m,d)
        while pending and pending[0][0]<=step:
            _,_,p=heapq.heappop(pending)
            if p.gen>=stamps[p.ent]: rx[p.ent]=p.state; stamps[p.ent]=p.gen
        q=np.asarray(d.qpos[:3]).copy(); ee=np.asarray(d.site("eef").xpos[:2]).copy()
        ent=choose(policy,step,ee,tr,trv,rx,stamps); sent[ent]+=1
        if not losses[step]:
            ds=int(delays[step])
            if ds<=0: rx[ent]=tr[ent]; stamps[ent]=step
            else:
                p=Packet(step+ds,seq,ent,step,tr[ent].copy()); seq+=1
                heapq.heappush(pending,(p.arrival,p.seq,p))
        ages=(step-stamps)*COMM_DT; aois.append(np.mean(ages)); kao.append(np.mean(ages*np.linalg.norm(trv,axis=1)))
        us=operator(ee,rx[0],rx[1:]); ut=operator(ee,tr[0],tr[1:]); action_gap.append(np.linalg.norm(us-ut))
        d.ctrl[:3]=ik(q,us)
        for _ in range(sub): mujoco.mj_step(m,d)
        ee2=np.asarray(d.site("eef").xpos[:2]); clear=np.linalg.norm(obs-ee2[None,:],axis=1)-(EEF_R+OBS_R)
        clr.append(np.min(clear))
        if np.min(clear)<0: collision=1; break
        if np.linalg.norm(ee2-tar)<.06:
            hold+=1
            if hold>=3: success=1; break
        else: hold=0
    n=step+1
    return dict(seed=seed,policy=policy,network_profile=profile_name,n_distractors=n_distractors,
      success=success,collision=collision,timeout=int(not success and not collision),steps=n,completion_s=n*COMM_DT,
      mean_action_gap=float(np.mean(action_gap)),p95_action_gap=float(np.quantile(action_gap,.95)),
      mean_aoi_s=float(np.mean(aois)),p95_aoi_s=float(np.quantile(aois,.95)),mean_kaoi_m=float(np.mean(kao)),
      min_clearance_m=float(np.min(clr)),packet_losses=int(np.sum(losses[:n])),mean_network_delay_ms=float(np.mean(dms[:n])),
      sent_target_frac=float(sent[0]/sent.sum()),sent_relevant_obs_frac=float(sent[1:4].sum()/sent.sum()),
      sent_distractor_frac=float(sent[4:].sum()/sent.sum()))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--episodes",type=int,default=80)
    ap.add_argument("--profiles",nargs="+",default=["wifi_congested","poor_4g"])
    ap.add_argument("--policies",nargs="+",default=list(POLICIES)); ap.add_argument("--profile-family",default="paper",choices=["paper","repo"])
    ap.add_argument("--distractors",nargs="+",type=int,default=[0,3,5]); ap.add_argument("--out",type=Path,default=Path("results/raw_results.csv"))
    ap.add_argument("--xml",type=Path,default=Path(__file__).with_name("models")/"planar_arm.xml"); a=ap.parse_args()
    rows=[]; total=a.episodes*len(a.profiles)*len(a.policies)*len(a.distractors); k=0
    for nd in a.distractors:
      for prof in a.profiles:
       for seed in range(a.episodes):
        for pol in a.policies:
         rows.append(run(a.xml,seed,pol,prof,a.profile_family,nd)); k+=1
        if seed%10==0: print(f"[{k}/{total}] {prof=} {nd=} {seed=}",flush=True)
    df=pd.DataFrame(rows); a.out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
    print(df.groupby(["network_profile","n_distractors","policy"]).agg(success=("success","mean"),collision=("collision","mean"),action_gap=("mean_action_gap","mean"),aoi=("mean_aoi_s","mean"),relevant_tx=("sent_relevant_obs_frac","mean"),distractor_tx=("sent_distractor_frac","mean")).round(4).to_string())
if __name__=="__main__": main()
