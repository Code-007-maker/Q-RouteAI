"""Quick demo: Nearest-Neighbour vs Clarke-Wright vs GA vs QI-EA-M on one 100-stop instance (~20 s)."""
from proto import make_instance, cw_giant, split_cost, nn_baseline
from proto2 import run_gen

xy, D, dem, Q = make_instance(100, 1100)          # 100 customers, fixed seed
g = cw_giant(D, dem, Q, xy)                       # Clarke-Wright starting tour
cw = split_cost(g, D, dem, Q)
print(f"Nearest-neighbour : {nn_baseline(D, dem, Q):8.1f}")
print(f"Clarke-Wright     : {cw:8.1f}  (reference)")

run_gen(D, dem, Q, 0, 0.5, init_giant=g, quantum=True, sa=True)   # warm-up: numba compiles here

engines = {
    "GA":       dict(),
    "QI-EA":    dict(quantum=True, sa=True),
    "GA-M":     dict(memetic=True),
    "QI-EA-M":  dict(quantum=True, sa=True, memetic=True),
}
for name, kw in engines.items():
    tr = run_gen(D, dem, Q, 0, 5.0, init_giant=g, **kw)           # 5 s wall-clock budget
    print(f"{name:<10}        : {tr.best:8.1f}  ({(cw - tr.best) / cw * 100:+.2f} % vs Clarke-Wright)")
