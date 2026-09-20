import numpy as np, time
from numba import njit
from proto import (split_cost, eval_perms, local_search, ox, Trace, make_instance, nn_baseline)


@njit(cache=True)
def seed_nb(s):
    np.random.seed(s)


@njit(cache=True)
def breakpoint_dist(a, b):
    """number of adjacent pairs of a that are not adjacent in b (0..n-1) -> swarm 'spread' L"""
    n = a.shape[0]
    pos = np.empty(n + 2, dtype=np.int64)
    for i in range(n):
        pos[b[i]] = i
    cnt = 0
    for i in range(n - 1):
        d = pos[a[i]] - pos[a[i + 1]]
        if d != 1 and d != -1:
            cnt += 1
    return cnt


@njit(cache=True)
def consensus(PB):
    """discrete 'mean-best' position: customers ordered by mean position over all pbest tours"""
    M, n = PB.shape
    s = np.zeros(n)
    for m in range(M):
        for i in range(n):
            s[PB[m, i] - 1] += i
    return (np.argsort(s) + 1).astype(np.int64)


@njit(cache=True)
def perturb(x, k):
    n = x.shape[0]
    y = x.copy()
    for _ in range(k):
        r = np.random.random()
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i > j:
            i, j = j, i
        if r < 0.34:
            t = y[i]; y[i] = y[j]; y[j] = t
        elif r < 0.67:
            a = i; b = j
            while a < b:
                t = y[a]; y[a] = y[b]; y[b] = t
                a += 1; b -= 1
        else:
            v = y[i]
            for q in range(i, j):
                y[q] = y[q + 1]
            y[j] = v
    return y


@njit(cache=True)
def dq_step(X, PB, GB, alpha, D, dem, Q, pmix):
    """Discrete QPSO move: attractor p = OX(pbest_i, gbest) with random fraction phi;
    jump size k ~ alpha * L * ln(1/u)  (exponential law of the delta-well), L = |x - mbest| (breakpoint distance)."""
    M, n = X.shape
    cons = consensus(PB)
    Xn = np.empty_like(X)
    for m in range(M):
        L = breakpoint_dist(X[m], cons)
        phi = np.random.random()
        seg = max(1, int(phi * n))
        a = np.random.randint(0, n - seg + 1)
        if np.random.random() < pmix:
            p = ox(PB[m], PB[np.random.randint(0, M)], a, a + seg - 1)
        else:
            p = ox(PB[m], GB, a, a + seg - 1)
        u = np.random.random() + 1e-12
        k = int(alpha * L * np.log(1.0 / u) + 0.5)
        if k < 1:
            k = 1
        if k > n // 2:
            k = n // 2
        Xn[m] = perturb(p, k)
    return Xn, eval_perms(Xn, D, dem, Q)


def run_dq(D, dem, Q, seed, tlimit, M=30, memetic=False, sa=False, adaptive=False, w_ls=30, pmix=0.0, a0=1.0, a1=0.5):
    seed_nb(seed)
    rng = np.random.default_rng(seed)
    n = len(dem) - 1
    tr = Trace()
    X = np.array([rng.permutation(n) + 1 for _ in range(M)], dtype=np.int64)
    F = eval_perms(X, D, dem, Q)
    PB, PF = X.copy(), F.copy()
    gi = int(np.argmin(PF)); GB, GF = PB[gi].copy(), PF[gi]
    tr.upd(GF)
    T0 = 0.01 * GF
    it = 0; boost = 1.0; stall = 0; last_ls_f = 1e18
    while tr.el() < tlimit:
        prog = tr.el() / tlimit
        alpha = min(1.7, (a0 - (a0 - a1) * prog) * boost)
        X, F = dq_step(X, PB, GB, alpha, D, dem, Q, pmix)
        if sa:
            T = T0 * (1 - prog) ** 2 + 1e-9
            better = F < PF
            with np.errstate(over="ignore"):
                worse_ok = (~better) & (rng.random(M) < np.exp(np.minimum(0.0, -(F - PF) / T)))
            upd = better | worse_ok
        else:
            upd = F < PF
        PB[upd] = X[upd]; PF[upd] = F[upd]
        bi = int(np.argmin(F))
        if F[bi] < GF - 1e-9:
            GF, GB = F[bi], X[bi].copy(); stall = 0; boost = 1.0
        else:
            stall += 1
            if adaptive and stall % 15 == 0:
                boost = min(2.0, boost * 1.15)
        tr.upd(GF)
        it += 1
        if memetic and it % 5 == 0 and GF < last_ls_f - 1e-9:
            g2, f2 = local_search(GB.copy(), D, dem, Q, w_ls)
            last_ls_f = f2
            if f2 < GF - 1e-9:
                GF, GB = f2, g2.copy()
                j = int(np.argmin(PF)); PB[j] = GB.copy(); PF[j] = GF
                tr.upd(GF)
                stall = 0; boost = 1.0
            else:
                last_ls_f = GF
    return tr


def run_ga_m(D, dem, Q, seed, tlimit, M=30, pm=0.25, w_ls=30):
    """GA + identical memetic step (local search on the best individual every 5 generations)"""
    rng = np.random.default_rng(seed)
    n = len(dem) - 1
    tr = Trace()
    Pm = np.array([rng.permutation(n) + 1 for _ in range(M)], dtype=np.int64)
    F = eval_perms(Pm, D, dem, Q)
    tr.upd(F.min())
    gen = 0; last_ls_f = 1e18
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
        gen += 1
        bi = int(np.argmin(F))
        if gen % 5 == 0 and F[bi] < last_ls_f - 1e-9:
            g2, f2 = local_search(Pm[bi].copy(), D, dem, Q, w_ls)
            last_ls_f = f2
            if f2 < F[bi] - 1e-9:
                Pm[bi] = g2; F[bi] = f2
            else:
                last_ls_f = F[bi]
        tr.upd(F.min())
    return tr


ALGS2 = {
    "GA-M": lambda D, dem, Q, s, t: run_ga_m(D, dem, Q, s, t),
    "DQPSO": lambda D, dem, Q, s, t: run_dq(D, dem, Q, s, t),
    "DQPSO-M": lambda D, dem, Q, s, t: run_dq(D, dem, Q, s, t, memetic=True),
    "AS-QPSO-SA-M*": lambda D, dem, Q, s, t: run_dq(D, dem, Q, s, t, memetic=True, sa=True, adaptive=True),
}


# ---------------------------------------------------------------- jitted generational engines (same infrastructure for GA and hybrid)
@njit(cache=True)
def _tournament(F, k):
    M = F.shape[0]
    best = np.random.randint(0, M)
    for _ in range(k - 1):
        c = np.random.randint(0, M)
        if F[c] < F[best]:
            best = c
    return best


@njit(cache=True)
def generation(Pm, F, D, dem, Q, alpha, quantum, pm, T):
    """One generation. quantum=False -> classic GA (OX + inversion mutation with prob pm).
    quantum=True -> OX + quantum-inspired exponential-jump mutation  k ~ alpha*L*ln(1/u),
    L = breakpoint distance of the child to the population 'mean-best' consensus tour.
    T>0 -> Metropolis (simulated-annealing) acceptance of a child against its first parent."""
    M, n = Pm.shape
    order = np.argsort(F)
    New = np.empty_like(Pm)
    NF = np.empty(M)
    New[0] = Pm[order[0]]; NF[0] = F[order[0]]
    New[1] = Pm[order[1]]; NF[1] = F[order[1]]
    cons = consensus(Pm) if quantum else Pm[0]
    for m in range(2, M):
        i1 = _tournament(F, 3)
        i2 = _tournament(F, 3)
        a = np.random.randint(0, n); b = np.random.randint(0, n)
        if a > b:
            t = a; a = b; b = t
        c = ox(Pm[i1], Pm[i2], a, b)
        if quantum:
            L = breakpoint_dist(c, cons)
            u = np.random.random() + 1e-12
            k = int(alpha * L * np.log(1.0 / u) + 0.5)
            if k > n // 2:
                k = n // 2
            if k < 1 and np.random.random() < 0.5:
                k = 1
            if k > 0:
                c = perturb(c, k)
        else:
            if np.random.random() < pm:
                i = np.random.randint(0, n); j = np.random.randint(0, n)
                if i > j:
                    t = i; i = j; j = t
                while i < j:
                    t = c[i]; c[i] = c[j]; c[j] = t
                    i += 1; j -= 1
        fc = split_cost(c, D, dem, Q)
        if T > 0.0 and fc > F[i1]:
            if np.random.random() > np.exp(-(fc - F[i1]) / T):
                c = Pm[i1].copy(); fc = F[i1]
        New[m] = c; NF[m] = fc
    return New, NF


def run_gen(D, dem, Q, seed, tlimit, quantum=False, memetic=False, sa=False, M=30, w_ls=30, a0=1.0, a1=0.5, pm=0.25, ls_frac=0.15, init_giant=None):
    seed_nb(seed)
    rng = np.random.default_rng(seed)
    n = len(dem) - 1
    tr = Trace()
    Pm = np.array([rng.permutation(n) + 1 for _ in range(M)], dtype=np.int64)
    if init_giant is not None:
        Pm[0] = init_giant
        for i in range(1, M):
            Pm[i] = perturb(init_giant, int(rng.integers(1, max(3, n // 10))))
    F = eval_perms(Pm, D, dem, Q)
    tr.upd(F.min())
    T0 = 0.01 * F.min(); gen = 0; last_ls_f = 1e18; ls_time = 0.0
    while tr.el() < tlimit:
        prog = tr.el() / tlimit
        alpha = a0 - (a0 - a1) * prog
        T = T0 * (1 - prog) ** 2 + 1e-9 if sa else 0.0
        Pm, F = generation(Pm, F, D, dem, Q, alpha, quantum, pm, T)
        gen += 1
        bi = int(np.argmin(F))
        if memetic and gen % 5 == 0 and F[bi] < last_ls_f - 1e-9 and ls_time < ls_frac * tr.el():
            t_ = time.perf_counter()
            g2, f2 = local_search(Pm[bi].copy(), D, dem, Q, w_ls)
            ls_time += time.perf_counter() - t_
            last_ls_f = f2
            if f2 < F[bi] - 1e-9:
                Pm[bi] = g2; F[bi] = f2
            else:
                last_ls_f = F[bi]
        tr.upd(F.min())
    return tr
