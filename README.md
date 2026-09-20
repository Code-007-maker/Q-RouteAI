# Q-RouteAI — quantum-inspired routing engine (research prototype)

Prototype of the optimisation core behind **Q-RouteAI**, our solution for **Smart India Hackathon 2026, PS ID SIH26137**
(*Quantum-Inspired Intelligent Traffic Route Optimization in Transportation Systems Using Metaheuristic Optimization*, Egreen Quanta).

> **Status:** this repository contains the *engine prototype and its benchmark only* (Python + Numba).
> The full platform (road-graph service, live map UI, disruption simulator, Race Arena) is described in
> [`docs/Q-RouteAI_Detailed_Solution_SIH26137.pdf`](docs/Q-RouteAI_Detailed_Solution_SIH26137.pdf) and is **not** implemented here.

"Quantum-inspired" means the algorithms borrow ideas from quantum-behaved PSO (the exponential jump law of the delta-potential well).
Everything runs on ordinary CPUs; **no quantum hardware or simulator is used.**

---

## Idea in one paragraph

Shortest paths (Dijkstra / A*) are already optimal; the hard, NP-hard part of fleet routing is *sequencing and assigning stops to vehicles* (CVRP / VRPTW).
We therefore use a **Construct → Improve → Repair** pipeline: (1) build a strong starting plan with Clarke-Wright savings,
(2) improve it under a fixed time budget with a quantum-inspired metaheuristic, (3) repair it locally when a road closes or an order arrives.
This repo implements steps 1 and 2 on capacitated routing (CVRP) instances.

## Methods compared

All methods use the same decoder (Prins' optimal split), population size (30), cost function, wall-clock budget, and the same Clarke-Wright starting tour.

| Name | What it is |
|---|---|
| `PSO (RKE)` | Standard PSO on random-key encoded tours |
| `QPSO (RKE)` | Continuous quantum-behaved PSO on random keys (`x = p ± α·|mbest − x|·ln(1/u)`, α: 1.0 → 0.5) |
| `GA` | Permutation GA: tournament selection, order crossover, inversion mutation |
| `GA-M` | GA + windowed 2-opt / Or-opt local search on the best tour (≤ 15 % of run time) |
| `QI-EA` | Evolutionary loop with a **quantum-inspired exponential-jump mutation** (`k = α·L·ln(1/u)`, L = distance to the swarm's consensus tour) and simulated-annealing acceptance |
| `QI-EA-M` | `QI-EA` + the same local search (our engine) |

## Results (measured)

Synthetic CVRP: customers uniform in a 100 × 100 square, depot at the centre, demand 1–20, capacity 100, Euclidean distances.
Budgets 3 / 5 / 8 / 14 s for 50 / 100 / 200 / 400 stops, 6 / 6 / 4 / 3 seeds, one CPU core.
Values are **distance saved vs the Clarke-Wright plan (%)**, mean ± std — higher is better.
Nearest-neighbour is 10–17 % *worse* than Clarke-Wright at these sizes.

| Method | 50 stops | 100 stops | 200 stops | 400 stops |
|---|---|---|---|---|
| PSO (RKE) | 1.52 ± 0.45 | 0.25 ± 0.02 | 0.36 ± 0.06 | 0.29 ± 0.00 |
| QPSO (RKE) | 0.59 ± 0.67 | 0.12 ± 0.12 | 0.03 ± 0.06 | 0.03 ± 0.02 |
| GA | 1.98 ± 0.05 | 0.53 ± 0.28 | 0.51 ± 0.00 | 0.63 ± 0.07 |
| GA-M | 2.82 ± 0.16 | 0.89 ± 0.04 | 0.88 ± 0.00 | 0.94 ± 0.02 |
| QI-EA | 2.68 ± 0.62 | 0.79 ± 0.14 | 0.85 ± 0.06 | 0.76 ± 0.09 |
| **QI-EA-M** | **2.84 ± 0.21** | **1.08 ± 0.03** | 0.88 ± 0.00 | **0.99 ± 0.03** |

(Run 2, Windows / Python 3.14. Run 1, Linux / Python 3.12, agrees within ~0.5 percentage points; raw data in `results_run1_sandbox.json` and `results_run2_windows.json`.)

### What these results support
- Starting from Clarke-Wright matters most. Random-start metaheuristics were **worse** than Clarke-Wright from 100 stops upward in our earlier experiments.
- The quantum-inspired jump operator beat a plain GA in most seeds (`QI-EA` vs `GA`) by roughly 0.1–0.7 percentage points.
- Local search gives the bulk of the gain; with it, `QI-EA-M` and `GA-M` are nearly tied (exactly tied at 200 stops).
- Random-key QPSO (the continuous mode) adds almost nothing over its starting tour.

### What they do **not** support
- That QPSO is generally superior to GA/PSO, or any asymptotic speed-up.
- Anything about real city data: instances are synthetic, budgets are short, seed counts are small (3–6), so p-values are indicative only.
- Time windows are **not** implemented in this prototype (planned in the Prins split, see the PDF, Section 4.3).

## Quick start

```bash
git clone https://github.com/<your-username>/q-routeai.git
cd q-routeai
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python quickstart.py          # ~20 s: NN vs Clarke-Wright vs GA vs QI-EA-M on 100 stops
```

The first run is slower because Numba compiles the kernels (cached afterwards).

### Reproduce the full benchmark (~12 min, one CPU core)

```bash
python run_bench3.py          # writes bench3.json after every method (resumable)
python figs.py bench          # optional: charts from bench3.json (needs matplotlib)
```

- If `bench3.json` already exists, the script **resumes and skips finished methods** — delete it to start from scratch.
- Budgets are wall-clock, so results shift slightly between machines. Avoid heavy background work while it runs.
- Edit `plan = [(n, seconds, seeds), ...]` in `run_bench3.py` to change sizes, budgets, seeds.

### Use your own instance

The engine needs a `(n+1) × (n+1)` distance/time matrix `D` (depot at index 0), a demand array `dem` (`dem[0] = 0`) and a capacity `Q`:

```python
from proto import cw_giant, split_cost
from proto2 import run_gen
g  = cw_giant(D, dem, Q, xy)          # xy: (n+1, 2) coordinates, used only to order routes by angle
tr = run_gen(D, dem, Q, seed=0, tlimit=10.0, init_giant=g, quantum=True, sa=True, memetic=True)
print(tr.best, tr.pts[-1])            # best cost, (time, cost) convergence trace
```

## Repository layout

| File | Purpose |
|---|---|
| `proto.py` | Prins split, local search (2-opt/Or-opt), Clarke-Wright, nearest-neighbour, random-key PSO/QPSO |
| `proto2.py` | Discrete quantum-inspired operator, `GA`/`QI-EA` generational engine (`run_gen`) |
| `run_bench3.py` | Benchmark runner (the numbers above) |
| `quickstart.py` | 20-second demo |
| `figs.py` | Charts and diagrams used in the PDF |
| `results_run*.json` | Raw benchmark data (best cost + convergence trace per run) |
| `docs/` | Detailed solution document (PDF) |

## Limitations and next steps
- Local search re-evaluates the whole tour (`O(n²·w·b)` per pass); **delta evaluation** is the first planned optimisation.
- Add time windows to the split, heterogeneous fleets, multi-depot.
- Benchmark on Solomon, Gehring–Homberger and CVRPLIB X-set instances with ≥ 30 seeds and Wilcoxon tests; compare against OR-Tools as an external reference.
- Road-network layer (OSMnx + SciPy Dijkstra cost matrix), live disruption repair (*Local Wave Recollapse*), CO₂-aware objective — see the PDF.

## References
Sun, Feng & Xu (2004), QPSO · Sun et al. (2012), QPSO convergence analysis · Prins (2004), optimal split ·
Clarke & Wright (1964), savings heuristic · Bean (1994), random keys · Kirkpatrick et al. (1983), simulated annealing.
Full list in the PDF.

## License
MIT — see `LICENSE`.

## Team
<!-- Add team name, members, college and contact here -->
