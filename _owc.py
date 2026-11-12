"""OWC(진동수주형 파력발전) 시뮬레이터: 물리 엔진 + 베이지안 최적화 + 웹 서버.
Vercel 서버리스 함수용 엔진 (필요: numpy)

상태변수: 수주 변위 eta, 속도 v, 챔버 공기질량 m, 터빈 각속도 w
 수주:  rho_w*Le*eta'' = rho_w*g*(eta_w-eta) - b*eta' - cq|eta'|eta' - p_gauge
 공기:  p = P0*(rho/rho_a)^gamma, rho = m/(V0-Ac*eta)  (등엔트로피, 압축성), dm/dt = 유입-유출
 유로:  챔버 <-> 대기. [밸브 + 터빈 노즐 직렬 유로] || [릴리프 우회 유로(터빈 비통과)]
        직렬 유효면적 1/Aeff^2 = 1/(Cd*th*Av)^2 + 1/(Cd*An)^2,  mdot = Aeff*sqrt(2*rho*dp)
 터빈:  노즐 압력강하 dp_t = mdot^2/(2 rho (Cd An)^2), 공력동력 Pa = dp_t*Q,
        효율 eta_t(phi), phi = 제트속도/(w*r)  (임펄스형 포물선 특성),  T_t = eta_t*Pa/w
 발전기: J*w' = T_t - cg*w - bf*w,  P_el = eta_gen * cg * w^2
"""
import json, math, os
import numpy as np

G, RHO_W, RHO_A, P0, GAM = 9.81, 1025., 1.225, 101325., 1.4
PAR = dict(Ac=25., B=5., V0=100., Le=6., zeta=0.05, cq=500., Av=0.4, An=0.15, Ab=0.25, Cd=0.8,
           r=0.5, J=60., cg=1., bf=0.05, eta_gen=0.95, eta_t_max=0.75, phi_opt=2., tau_v=0.15)
BOUNDS = np.array([[0, 1], [0.1, 1], [0, 1], [0.1, 1], [2000, 20000]], float)  # po,do,pi,di,p_relief[Pa]


def decode(u):
    u = np.clip(np.asarray(u, float), 0, 1)
    return BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])


def make_wave(kind, H, T, seed, t):
    if kind == 'regular':
        return 0.5 * H * np.sin(2 * np.pi * t / T)
    rng = np.random.default_rng(seed)
    fp = 1 / T; f = np.linspace(0.5 * fp, 2.5 * fp, 40); df = f[1] - f[0]
    S = (5 / 16) * H**2 * fp**4 * f**-5 * np.exp(-1.25 * (fp / f)**4)  # Bretschneider
    a = np.sqrt(2 * S * df); ph = rng.uniform(0, 2 * np.pi, 40)
    return (a[:, None] * np.sin(2 * np.pi * f[:, None] * t[None] + ph[:, None])).sum(0)


def simulate(wave, mode='baseline', xp=None, dur=100., warm=20., dt=0.01, rec=0.05, par=None, full=True):
    """wave=dict(kind,H,T,seed). mode: 'baseline'(밸브 상시 개방) | 'predictive'(예측 기반 래칭). xp=[po,do,pi,di,p_relief]."""
    p_ = {**PAR, **(par or {})}; Ac, V0, Le = p_['Ac'], p_['V0'], p_['Le']
    n = int((warm + dur) / dt); t = np.arange(n) * dt
    ew = make_wave(wave['kind'], wave['H'], wave['T'], wave.get('seed', 1), t)
    po, do, pi_, di, pr = (xp if xp is not None else [0, 1, 0, 1, 1e9]) if mode == 'predictive' else (0, 1, 0, 1, 1e9)
    b = 2 * p_['zeta'] * math.sqrt(RHO_W * G * RHO_W * Le); Cd, An, Av, Ab = p_['Cd'], p_['An'], p_['Av'], p_['Ab']
    m = RHO_A * V0; eta = v = 0.; w = 30.; tho = thi = 1.; step = p_['tau_v'] and dt / p_['tau_v']
    zc, periods, Th = None, [], wave['T']; pred_log = []
    E_el = E_a = E_b = 0.; ps = []; rs = max(1, int(rec / dt)); out = {k: [] for k in 't ew eta p w Pel Pa Pb tho thi Th'.split()}
    for k in range(n):
        V = V0 - Ac * eta; rho = m / V; pg = P0 * (rho / RHO_A)**GAM - P0
        # ---- 파도 주기 예측 + 밸브 제어 ----
        if k and ew[k - 1] < 0 <= ew[k]:
            if zc is not None:
                per = t[k] - zc; pred_log.append((float(t[k]), Th, per)); periods.append(per); Th = float(np.mean(periods[-3:]))
            zc = t[k]
        if mode == 'predictive' and zc is not None:
            ph = min((t[k] - zc) / Th, 0.999)
            to = 1. if (ph - po) % 1 < do else 0.; ti = 1. if (ph - pi_) % 1 < di else 0.
        else:
            to = ti = 1.
        tho += max(-step, min(step, to - tho)); thi += max(-step, min(step, ti - thi))
        # ---- 유동 ----
        dp = abs(pg); up, th, sg = (rho, tho, -1) if pg > 0 else (RHO_A, thi, 1)
        Aeff = 1 / math.sqrt(1 / (Cd * th * Av + 1e-6)**2 + 1 / (Cd * An)**2)
        mt = Aeff * math.sqrt(2 * up) * dp / math.sqrt(dp + 50.)          # 터빈 경유 질량유량
        ex = max(dp - pr, 0.); mb = Cd * Ab * math.sqrt(2 * up) * ex / math.sqrt(ex + 50.)  # 우회(릴리프) 유량
        dpt = mt * mt / (2 * up * (Cd * An)**2); Q = mt / up; Pa = dpt * Q; Pb = dp * mb / up
        phi = (Q / (Cd * An)) / (max(w, 5.) * p_['r'])
        et = max(0.05, p_['eta_t_max'] * (1 - ((phi - p_['phi_opt']) / p_['phi_opt'])**2))
        Tt = et * Pa / max(w, 5.); w = max(0., w + dt * (Tt - p_['cg'] * w - p_['bf'] * w) / p_['J'])
        Pel = p_['eta_gen'] * p_['cg'] * w * w
        m += sg * (mt + mb) * dt
        a = (RHO_W * G * (ew[k] - eta) - b * v - p_['cq'] * abs(v) * v - pg) / (RHO_W * Le)
        v += a * dt; eta += v * dt
        if t[k] >= warm: E_el += Pel * dt; E_a += Pa * dt; E_b += Pb * dt; ps.append(pg)
        if full and k % rs == 0 and t[k] >= warm:
            for key, val in zip(out, (t[k] - warm, ew[k], eta, pg, w, Pel, Pa, Pb, tho, thi, Th)): out[key].append(round(float(val), 4))
    Hs = wave['H']; Pw = RHO_W * G**2 * Hs**2 * wave['T'] / ((32 if wave['kind'] == 'regular' else 64) * math.pi) * p_['B']
    pl = [x for x in pred_log if x[0] >= warm]
    err = float(np.mean([abs(a - b) for _, a, b in pl])) if pl else None
    res = dict(P_el=E_el / dur, P_aero=E_a / dur, P_bypass=E_b / dur, P_wave=Pw, eff=E_el / dur / Pw, p_rms=float(np.sqrt(np.mean(np.square(ps)))),
               period_mae=err, periods=[dict(t=a, pred=b, actual=c) for a, b, c in pl[:40]])
    if full: res['series'] = out
    return res


# ---------------- 베이지안 최적화 (GP + Expected Improvement) ----------------
def _k(A, B, ls): return np.exp(-0.5 * ((A[:, None] - B[None])**2).sum(-1) / ls**2)
_erf = np.vectorize(math.erf)
def _Phi(z): return 0.5 * (1 + _erf(z / math.sqrt(2)))
def _phi(z): return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def bayes_opt(f, n_init=8, n_iter=22, d=5, seed=0):
    rng = np.random.default_rng(seed); X = np.vstack([[0, 1, 0, 1, 1], rng.random((n_init, d))]); y = np.array([f(x) for x in X])
    for _ in range(n_iter):
        yn = (y - y.mean()) / (y.std() + 1e-9); best = None
        for ls in (0.15, 0.3, 0.6):
            L = np.linalg.cholesky(_k(X, X, ls) + 1e-5 * np.eye(len(X))); al = np.linalg.solve(L.T, np.linalg.solve(L, yn))
            ll = -0.5 * yn @ al - np.log(np.diag(L)).sum()
            if best is None or ll > best[0]: best = (ll, ls, L, al)
        _, ls, L, al = best
        C = np.vstack([rng.random((2500, d)), np.clip(X[np.argsort(-y)[:3]][:, None] + 0.05 * rng.standard_normal((3, 200, d)), 0, 1).reshape(-1, d)])
        Ks = _k(C, X, ls); mu = Ks @ al; s = np.sqrt(np.maximum(1 - (np.linalg.solve(L, Ks.T)**2).sum(0), 1e-12))
        imp = mu - yn.max() - 0.01; z = imp / s; xn = C[np.argmax(imp * _Phi(z) + s * _phi(z))]
        X = np.vstack([X, xn]); y = np.append(y, f(xn))
    return X, y


def optimize(wave, n_init=8, n_iter=22):
    f = lambda u: simulate(wave, 'predictive', decode(u), dur=80, full=False)['P_el']
    X, y = bayes_opt(f, n_init, n_iter); i = int(np.argmax(y)); xb = decode(X[i]).tolist()
    hist = [dict(i=j + 1, P=float(y[j]), best=float(np.maximum.accumulate(y)[j])) for j in range(len(y))]
    val = []  # 다른 난수 시드로 검증(과적합 방지)
    for s in (11, 12, 13):
        wv = {**wave, 'seed': s}
        val.append(dict(seed=s, baseline=simulate(wv, 'baseline', full=False)['P_el'], optimized=simulate(wv, 'predictive', xb, full=False)['P_el']))
    return dict(x=xb, history=hist, validation=val)
