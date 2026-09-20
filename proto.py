import numpy as np, time
from numba import njit

# ---------------------------------------------------------------- core (numba)
@njit(cache=True)
def split_cost(giant, D, dem, Q):
    """Prins optimal split of a giant tour into capacity-feasible routes. Returns total distance."""
    n = giant.shape[0]
    P = np.full(n + 1, 1e18)
    P[0] = 0.0
    for i in range(n):
        load = 0.0
        cost = 0.0
        for j in range(i + 1, n + 1):
            c = giant[j - 1]
            load += dem[c]
            if load > Q:
                break
            if j == i + 1:
                cost = D[0, c] + D[c, 0]
            else:
                p = giant[j - 2]
                cost = cost - D[p, 0] + D[p, c] + D[c, 0]
            v = P[i] + cost
            if v < P[j]:
                P[j] = v
    return P[n]


@njit(cache=True)
def eval_keys(K, D, dem, Q):
    M = K.shape[0]
    out = np.empty(M)
    for m in range(M):
        g = np.argsort(K[m]) + 1
        out[m] = split_cost(g, D, dem, Q)
    return out


@njit(cache=True)
def eval_perms(Pm, D, dem, Q):
    M = Pm.shape[0]
    out = np.empty(M)
    for m in range(M):
        out[m] = split_cost(Pm[m], D, dem, Q)
    return out


@njit(cache=True)
def local_search(giant, D, dem, Q, w):
    """Windowed 2-opt + Or-opt on the giant tour, evaluated through the optimal split (memetic step)."""
    n = giant.shape[0]
    g = giant.copy()
    best = split_cost(g, D, dem, Q)
    h = g.copy()
    for _ in range(3):
        improved = False
        for i in range(n - 1):
            jmax = min(n, i + w)
            for j in range(i + 1, jmax):
                h[:] = g
                a = i
                b = j
                while a < b:
                    t = h[a]
                    h[a] = h[b]
                    h[b] = t
                    a += 1
                    b -= 1
                c = split_cost(h, D, dem, Q)
                if c < best - 1e-9:
                    best = c
                    g[:] = h
                    improved = True
        rest = np.empty(n, dtype=g.dtype)
        for L in range(1, 4):
            for i in range(n - L + 1):
                m = 0
                for x in range(n):
                    if x < i or x >= i + L:
                        rest[m] = g[x]
                        m += 1
                kmin = max(0, i - w)
                kmax = min(n - L, i + w)
                for k in range(kmin, kmax + 1):
                    if k == i:
                        continue
                    for x in range(k):
                        h[x] = rest[x]
                    for x in range(L):
                        h[k + x] = g[i + x]
                    for x in range(k, n - L):
                        h[x + L] = rest[x]
                    c = split_cost(h, D, dem, Q)
                    if c < best - 1e-9:
                        best = c
                        g[:] = h
                        improved = True
                        m = 0
                        for x in range(n):
                            if x < i or x >= i + L:
                                rest[m] = g[x]
                                m += 1
        if not improved:
            break
    return g, best


@njit(cache=True)
def ox(p1, p2, a, b):
    n = p1.shape[0]
    child = np.zeros(n, dtype=p1.dtype)
    used = np.zeros(n + 2, dtype=np.bool_)
    for i in range(a, b + 1):
        child[i] = p1[i]
        used[p1[i]] = True
    pos = (b + 1) % n
    for t in range(n):
        v = p2[(b + 1 + t) % n]
        if not used[v]:
            child[pos] = v
            used[v] = True
            pos = (pos + 1) % n
    return child


# ---------------------------------------------------------------- instance
def make_instance(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, size=(n + 1, 2))
    xy[0] = (50, 50)
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    dem = np.zeros(n + 1)
    dem[1:] = rng.integers(1, 21, size=n)
    return xy, D, dem, 100.0


def keys_from_giant(g):
    n = len(g)
    k = np.empty(n)
    k[g - 1] = (np.arange(n) + 0.5) / n
    return k


# ---------------------------------------------------------------- algorithms
class Trace:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.pts = []
        self.best = 1e18

    def upd(self, f):
        if f < self.best - 1e-9:
            self.best = f
            self.pts.append((time.perf_counter() - self.t0, f))

    def el(self):
        return time.perf_counter() - self.t0


def nn_baseline(D, dem, Q):
    n = len(dem) - 1
    un = set(range(1, n + 1))
    cur = 0
    g = []
    while un:
        nxt = min(un, key=lambda j: D[cur, j])
        g.append(nxt)
        un.remove(nxt)
        cur = nxt
    return split_cost(np.array(g, dtype=np.int64), D, dem, Q)


PSO_CFG = dict(w0=0.9, w1=0.4, c1=2.0, c2=2.0, vmax=0.25)

def run_swarm(kind, D, dem, Q, seed, tlimit, M=30, memetic=False, sa=False, w_ls=30, init_giant=None):
    """kind in {'pso','qpso'}; random-key encoding."""
    rng = np.random.default_rng(seed)
    n = len(dem) - 1
    tr = Trace()
    X = rng.random((M, n))
    if init_giant is not None:
        base = keys_from_giant(init_giant)
        X[0] = base
        for i in range(1, M):
            X[i] = base + rng.normal(0, rng.uniform(0.005, 0.08), n)
    V = rng.uniform(-0.1, 0.1, (M, n))
    F = eval_keys(X, D, dem, Q)
    PB, PF = X.copy(), F.copy()
    gi = int(np.argmin(PF))
    GB, GF = PB[gi].copy(), PF[gi]
    tr.upd(GF)
    T0 = 0.01 * GF
    it = 0
    last_ls_f = 1e18
    while tr.el() < tlimit:
        prog = tr.el() / tlimit
        if kind == "pso":
            c = PSO_CFG
            wgt = c["w0"] - (c["w0"] - c["w1"]) * prog
            r1, r2 = rng.random((M, n)), rng.random((M, n))
            V = wgt * V + c["c1"] * r1 * (PB - X) + c["c2"] * r2 * (GB - X)
            V = np.clip(V, -c["vmax"], c["vmax"])
            X = X + V
        else:  # QPSO: p = phi*pbest + (1-phi)*gbest ; x = p +/- alpha*|mbest-x|*ln(1/u)
            alpha = 1.0 - 0.5 * prog
            mbest = PB.mean(axis=0)
            phi = rng.random((M, n))
            p = phi * PB + (1 - phi) * GB
            u = rng.random((M, n)) + 1e-12
            sgn = np.where(rng.random((M, n)) < 0.5, -1.0, 1.0)
            X = p + sgn * alpha * np.abs(mbest - X) * np.log(1.0 / u)
        F = eval_keys(X, D, dem, Q)
        if sa:
            T = T0 * (1 - prog) ** 2 + 1e-9
            better = F < PF
            worse_ok = (~better) & (rng.random(M) < np.exp(-(F - PF) / T))
            upd = better | worse_ok
        else:
            upd = F < PF
        PB[upd] = X[upd]
        PF[upd] = F[upd]
        bi = int(np.argmin(F))
        if F[bi] < GF:
            GF, GB = F[bi], X[bi].copy()
        tr.upd(GF)
        it += 1
        if memetic and it % 5 == 0 and GF < last_ls_f - 1e-9:
            g = (np.argsort(GB) + 1).astype(np.int64)
            g2, f2 = local_search(g, D, dem, Q, w_ls)
            last_ls_f = f2
            if f2 < GF - 1e-9:
                GF = f2
                GB = keys_from_giant(g2)
                j = int(np.argmin(PF))
                PB[j] = GB.copy()
                PF[j] = GF
                tr.upd(GF)
            else:
                last_ls_f = GF
    return tr


def run_ga(D, dem, Q, seed, tlimit, M=30, pm=0.25):
    rng = np.random.default_rng(seed)
    n = len(dem) - 1
    tr = Trace()
    Pm = np.array([rng.permutation(n) + 1 for _ in range(M)], dtype=np.int64)
    F = eval_perms(Pm, D, dem, Q)
    tr.upd(F.min())
    while tr.el() < tlimit:
        order = np.argsort(F)
        new = [Pm[order[0]].copy(), Pm[order[1]].copy()]
        while len(new) < M:
            i1 = min(rng.integers(0, M, 3), key=lambda i: F[i])
            i2 = min(rng.integers(0, M, 3), key=lambda i: F[i])
            a, b = sorted(rng.integers(0, n, 2))
            c = ox(Pm[i1], Pm[i2], a, b)
            if rng.random() < pm:
                x, y = sorted(rng.integers(0, n, 2))
                c[x:y + 1] = c[x:y + 1][::-1].copy()
            new.append(c)
        Pm = np.array(new, dtype=np.int64)
        F = eval_perms(Pm, D, dem, Q)
        tr.upd(F.min())
    return tr


ALGS = {
    "GA": lambda D, dem, Q, s, t: run_ga(D, dem, Q, s, t),
    "PSO": lambda D, dem, Q, s, t: run_swarm("pso", D, dem, Q, s, t),
    "QPSO": lambda D, dem, Q, s, t: run_swarm("qpso", D, dem, Q, s, t),
    "PSO-M": lambda D, dem, Q, s, t: run_swarm("pso", D, dem, Q, s, t, memetic=True),
    "QPSO-M": lambda D, dem, Q, s, t: run_swarm("qpso", D, dem, Q, s, t, memetic=True),
    "AS-QPSO-SA-M": lambda D, dem, Q, s, t: run_swarm("qpso", D, dem, Q, s, t, memetic=True, sa=True),
}


def clarke_wright(D, dem, Q):
    """Parallel Clarke-Wright savings heuristic (industry-standard greedy)."""
    n = len(dem) - 1
    routes = {i: [i] for i in range(1, n + 1)}
    route_of = {i: i for i in range(1, n + 1)}
    load = {i: dem[i] for i in range(1, n + 1)}
    iu, ju = np.triu_indices(n, 1)
    sav = D[0, iu + 1] + D[0, ju + 1] - D[iu + 1, ju + 1]
    for idx in np.argsort(-sav):
        if sav[idx] <= 0:
            break
        i, j = int(iu[idx]) + 1, int(ju[idx]) + 1
        ri, rj = route_of[i], route_of[j]
        if ri == rj or load[ri] + load[rj] > Q:
            continue
        a, b = routes[ri], routes[rj]
        if a[-1] == i and b[0] == j: new = a + b
        elif a[0] == i and b[-1] == j: new = b + a
        elif a[-1] == i and b[-1] == j: new = a + b[::-1]
        elif a[0] == i and b[0] == j: new = a[::-1] + b
        else: continue
        routes[ri] = new; del routes[rj]; load[ri] += load[rj]
        for c in b: route_of[c] = ri
    return routes


def cw_giant(D, dem, Q, xy):
    routes = clarke_wright(D, dem, Q)
    def ang(r):
        return np.arctan2(np.mean([xy[c, 1] for c in r]) - xy[0, 1], np.mean([xy[c, 0] for c in r]) - xy[0, 0])
    g = []
    for r in sorted(routes.values(), key=ang):
        g += r
    return np.array(g, dtype=np.int64)
