"""Verify the included file hashes, panel counts and key manuscript aggregates."""
import csv
import hashlib
import math
from pathlib import Path
from statistics import mean, stdev

root=Path(__file__).resolve().parent
def read(name):
    with (root/name).open(encoding="utf-8",newline="") as f: return list(csv.DictReader(f))
def require(ok, message):
    if not ok: raise ValueError(message)
def close(a,b,tol=1e-8):
    require(math.isclose(float(a),float(b),rel_tol=0,abs_tol=tol),f"{a} != {b} (tolerance: {tol})")
for line in (root/"SHA256SUMS").read_text("ascii").splitlines():
    sha,name=line.split("  ",1)
    require(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,"File digest mismatch: "+name)
panels=read("test-panels.csv")
idx={}
for r in panels:
    key=tuple(r[k] for k in ("model","task","source","target","seed","subset"))
    require(key not in idx,f"Duplicate panel: {key}")
    n,c=int(r["n"]),int(r["correct"])
    require(n>0 and 0<=c<=n and int(r["k"])>0,f"Invalid counts: {key}")
    close(100*c/n, r["oa"])
    idx[key]=100*c/n
D=["PatternNet","RSICD","NWPU45","MLRSNet"]
def harmonic(a,b): return 2*a*b/(a+b)
def scores(m,seeds):
    datasets={}
    for d in D:
        b,n=[mean(idx[m,"B2N",d,d,str(i),k] for i in seeds) for k in ("base","new")]
        datasets[d]={"Base":b,"New":n,"H":harmonic(b,n)}
    b,n=[mean(datasets[d][k] for d in D) for k in ("Base","New")]
    datasets["Overall"]={"Base":b,"New":n,"H":harmonic(b,n)}
    parts=[]
    for task,suffix in [("CD",""),("DG","v2")]:
        ds=[d+suffix for d in D]
        acc={d:mean(idx[m,task,ds[0],d,str(i),"all"] for i in seeds) for d in ds}
        acc["target_mean"]=mean(acc[d] for d in ds[1:]); parts.append(acc)
    return datasets,*parts
tabs=read("tables.csv")
lookup={(r["table"],r["row"],r["metric"]):r for r in tabs}
main={"Prompt-OT":"PromptOT","PLKR":"PLKR","CAPrompt":"CAPrompt","SpecPL":"SpecPL","BRTP":"Full"}
for shown,m in main.items():
    datasets,cd,dg=scores(m,[1,2,3])
    for t,actual in [("T2",{d+"."+k:v for d,row in datasets.items() for k,v in row.items()}),("T3",cd),("T4",dg)]:
        for k,v in actual.items():
            r=lookup[t,shown,k]; close(v,r["checked_value"]);close(v,r["reported"],.00500001)
fig6=read("fig6-ten-seeds.csv")
summary=read("fig6-summary.csv")
require(len(summary)==9 and len(fig6)==90,"Unexpected Figure 6 record count")
def select(m,seeds,key):
    b,c,d=scores({"BRTP":"Full"}.get(m,m),seeds)
    return {"B2N_H":b["Overall"]["H"],"CD":c["target_mean"],"DG":d["target_mean"]}[key]
for r in summary:
    m,k=r["method"],r["column"]
    rows=sorted([v for v in fig6 if (v["method"],v["column"])==(m,k)],key=lambda x:int(x["seed"]))
    require([int(v["seed"]) for v in rows]==list(range(1,11)),"Missing or duplicated seeds")
    vals=[select(m,[i],k) for i in range(1,11)]
    for actual,v in zip(vals,rows): close(actual,v["value"])
    close(stdev(vals),r["sd"]); close(select(m,range(1,11),k),r["estimate"])
_,cd,dg=scores("Full",[1,2,3])
print(f"Verified {len(panels)} panel accuracies, {len(main)} local rows in each main table, and all ten-seed summaries.")
print("BRTP (three seeds): B2N H = %.2f, CD = %.2f, DG = %.2f." %
      (scores("Full",[1,2,3])[0]["Overall"]["H"],cd["target_mean"],dg["target_mean"]))
