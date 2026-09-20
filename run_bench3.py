import json, numpy as np
from proto import *
from proto2 import *
xy,D,dem,Q = make_instance(30, 1); g = cw_giant(D,dem,Q,xy)
for kw in (dict(), dict(memetic=True), dict(quantum=True,sa=True), dict(quantum=True,sa=True,memetic=True)):
    run_gen(D,dem,Q,0,0.6,init_giant=g,**kw)
run_swarm("pso",D,dem,Q,0,0.6,init_giant=g); run_swarm("qpso",D,dem,Q,0,0.6,init_giant=g)
ALG = {
 "PSO (RKE)":     lambda D,dem,Q,s,t,g: run_swarm("pso",D,dem,Q,s,t,init_giant=g),
 "QPSO (RKE)":    lambda D,dem,Q,s,t,g: run_swarm("qpso",D,dem,Q,s,t,init_giant=g),
 "GA":            lambda D,dem,Q,s,t,g: run_gen(D,dem,Q,s,t,init_giant=g),
 "GA-M":          lambda D,dem,Q,s,t,g: run_gen(D,dem,Q,s,t,init_giant=g,memetic=True),
 "QI-EA":         lambda D,dem,Q,s,t,g: run_gen(D,dem,Q,s,t,init_giant=g,quantum=True,sa=True),
 "QI-EA-M":       lambda D,dem,Q,s,t,g: run_gen(D,dem,Q,s,t,init_giant=g,quantum=True,sa=True,memetic=True),
}
plan = [(50,3.0,6),(100,5.0,6),(200,8.0,4),(400,14.0,3)]
import os
out = json.load(open("bench3.json")) if os.path.exists("bench3.json") else {}
out = {int(k):v for k,v in out.items()}
for n,tl,ns in plan:
    xy,D,dem,Q = make_instance(n, 1000+n)
    g = cw_giant(D,dem,Q,xy)
    out.setdefault(n, {"time_limit":tl,"NN":nn_baseline(D,dem,Q),"CW":float(split_cost(g,D,dem,Q)),"runs":{}})
    for name,f in ALG.items():
        if name in out[n]["runs"]: continue
        rs=[]
        for s in range(ns):
            tr = f(D,dem,Q,s,tl,g)
            rs.append({"best":tr.best,"pts":tr.pts})
        out[n]["runs"][name]=rs
        print(n,name,round(np.mean([r["best"] for r in rs]),1),"CW",round(out[n]["CW"],1),flush=True)
        json.dump(out, open("bench3.json","w"))
print("DONE",flush=True)
