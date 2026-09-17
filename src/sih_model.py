#!/usr/bin/env python3
"""
sih_model.py  --  window detector + stacked flight classifier + trust layer, evaluated at the FLIGHT level.

Feature engineering on top of windows.csv (all per flight, so every split stays flight-grouped):
  * temporal context: first differences and rolling maxima over 3 windows for the core physics features
  * flight-relative features: sensor-health and consistency features divided by the flight's own median over
    its first 20 s (an investigator compares a recovered log against itself; also absorbs receiver/airframe offsets)

Flight verdict: window probabilities are produced out-of-fold on the training flights (4-fold by flight),
aggregated per flight (max / p90 / mean / fraction > 0.5 / longest run per class, plus raw maxima of gap fractions
and consistency residuals), and a regularised multinomial logistic model is fitted on those aggregates.
The percentile rule from the first version is kept as a comparison ("rule" vs "stacked").

Experiments
  E0  Leakage audit, paired: per repetition, a flight-grouped split (30 training flights per family) and a random-window
      split with the same number of training windows, same detector; per-rep difference reported.
  E1  In-distribution, flight-grouped: 20 train / 20 calibration / 20 test flights per family (balanced test so the
      within-coarse-class mixture matches calibration), R repetitions; extra flights beyond 20 per family are scored
      separately. Window metrics raw vs temperature-scaled; flight-level accuracy, macro-F1, AURC and probability
      calibration (ECE, NLL, Brier) for the rule and for three stacked variants (probability summaries only, raw flight
      features only, full); a sensitivity variant without the derived-topic gap features; class-conditional conformal sets
      at the flight level (alpha 0.10 and 0.20) with abstention decomposed into multi-label, empty, non-singleton and
      operational (non-singleton or verdict outside its set).
  E2  Leave-one-subtype-out with the full stacked pipeline, with the split allocation and conformal ranks exported and a
      matched-size seen-subtype control for every hold-out.
  E3  Whelan live logs as case studies: verdicts, sets, window timeline, receiver-derived disturbance intervals and a
      declared alert rule with detection delay and false alerts.

Rules: receiver self-report (rx_*) columns are excluded from the detector. Simulator truth is never used.
Usage:
  python sih_model.py --features <features dir> --out <reports dir>
"""
import argparse
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
CLASSES = ['nominal', 'spoof', 'gps_degrade', 'sensor_fault']
META = {'t_start', 't_end', 'flight_id', 'source', 'run', 'family', 'subtype', 'onset_s', 'restore_s',
        'home_lat', 'noise_k', 'noise_t', 'noise_p', 'window_label', 'log'}
COARSE = {'nominal': 'nominal', 'spoof': 'gnss_chain', 'gps_degrade': 'gnss_chain', 'sensor_fault': 'non_gnss_fault'}
COARSE_CLASSES = ['nominal', 'gnss_chain', 'non_gnss_fault']
CTX_FEATS = ['posvel_rms', 'step_minus_vel_max', 'step_speed_max', 'gps_baro_dz_rms', 'dv_gps_imu_vecdiff',
             'gps_ekf_dpos_rms', 'gps_gap_frac', 'gps_nofix_frac', 'gps_silent_frac', 'baro_gap_frac', 'mag_gap_frac',
             'baro_raw_std', 'mag_raw_std',
             'tr_gpsHpos0_absmax', 'tr_gpsHpos1_absmax', 'tr_gpsHvel0_absmax', 'tr_gpsVpos_absmax',
             'tr_baroVpos_absmax', 'tr_heading_absmax', 'inn_gpsHpos0_absmax', 'inn_gpsHvel0_absmax']
REL_FEATS = ['baro_raw_std', 'mag_raw_std', 'mag_norm_std', 'mag_norm_mean', 'gyro_std_mean', 'baro_std',
             'posvel_rms', 'gps_ekf_dpos_rms', 'gps_baro_dz_rms', 'acc_horiz_rms', 'inn_gpsHpos0_absmax',
             'inn_gpsHvel0_absmax', 'inn_baroVpos_absmax', 'inn_heading_absmax']
AGG_RAW = ['gps_gap_frac', 'gps_nofix_frac', 'gps_silent_frac', 'baro_gap_frac', 'mag_gap_frac', 'posvel_rms',
           'step_minus_vel_max', 'gps_ekf_dpos_rms',
           'gps_baro_dz_rms', 'tr_gpsHpos0_absmax', 'tr_baroVpos_absmax', 'tr_heading_absmax']


# ------------------------------------------------------------------ feature engineering (per flight)
def engineer(W):
    W = W.sort_values(['flight_id', 't_start']).reset_index(drop=True)
    g = W.groupby('flight_id', sort=False)
    new = {}
    for f in [f for f in CTX_FEATS if f in W.columns]:
        new[f + '_d1'] = g[f].diff()
        new[f + '_rmax3'] = g[f].transform(lambda s: s.rolling(3, min_periods=1).max())
    rel = [f for f in REL_FEATS if f in W.columns]
    early = W[W['t_start'] < 20.0].groupby('flight_id')[rel].median()
    for f in rel:
        base = W['flight_id'].map(early[f]).astype(float)
        new[f + '_rel'] = W[f] / (base.abs() + 1e-3)
    return pd.concat([W, pd.DataFrame(new, index=W.index)], axis=1)


# ------------------------------------------------------------------ models
def make_xgb(seed):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, tree_method='hist', objective='multi:softprob',
                             random_state=seed, n_jobs=-1, verbosity=0)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=5, random_state=seed)


def make_logreg(seed):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced', random_state=seed))


def make_flight_model(seed):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(max_iter=5000, C=0.3, class_weight='balanced', random_state=seed))


def fit(model, X, y):
    yi = np.array([CLASSES.index(v) for v in y])
    counts = np.bincount(yi, minlength=len(CLASSES)).astype(float)
    w = (len(yi) / (len(CLASSES) * np.maximum(counts, 1)))[yi]
    try:
        model.fit(X, yi, sample_weight=w)
    except (TypeError, ValueError):
        model.fit(X, yi)
    return model


def predict_proba(model, X):
    p = model.predict_proba(X)
    full = np.full((len(X), len(CLASSES)), 1e-6)
    cls = getattr(model, 'classes_', None)
    if cls is None:
        cls = model[-1].classes_
    for j, c in enumerate(cls):
        full[:, int(c)] = p[:, j]
    return full / full.sum(1, keepdims=True)


# ------------------------------------------------------------------ calibration metrics
def ece_top(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def brier(p, y):
    oh = np.eye(p.shape[1])[y]
    return float(((p - oh) ** 2).sum(1).mean())


def fit_temperature(p_cal, y_cal):
    logit = np.log(np.clip(p_cal, 1e-12, 1))
    def loss(T):
        z = logit / T
        z = z - z.max(1, keepdims=True)
        q = np.exp(z); q /= q.sum(1, keepdims=True)
        return nll(q, y_cal)
    r = minimize_scalar(loss, bounds=(0.05, 20.0), method='bounded')
    return float(r.x)


def apply_temperature(p, T):
    z = np.log(np.clip(p, 1e-12, 1)) / T
    z = z - z.max(1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(1, keepdims=True)


# ------------------------------------------------------------------ flight-level aggregation
def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def flight_aggregates(W, P):
    """Aggregate window probabilities (and a few raw window features) into one row per flight."""
    rows = []
    raw = [f for f in AGG_RAW if f in W.columns]
    for fid, idx in W.groupby('flight_id', sort=False).indices.items():
        p = P[idx]; w = W.iloc[idx]
        r = {'flight_id': fid}
        for i, c in enumerate(CLASSES):
            col = p[:, i]
            r[f'max_{c}'] = float(col.max()); r[f'p90_{c}'] = float(np.percentile(col, 90))
            r[f'mean_{c}'] = float(col.mean()); r[f'frac_{c}'] = float((col > 0.5).mean())
            r[f'run_{c}'] = float(longest_run(col > 0.5))
        for f in raw:
            r[f'{f}_max'] = float(np.nanmax(w[f])) if np.isfinite(w[f]).any() else np.nan
        rows.append(r)
    A = pd.DataFrame(rows)
    fam = W.groupby('flight_id')['family'].first(); st = W.groupby('flight_id')['subtype'].first()
    A['family'] = A['flight_id'].map(fam); A['subtype'] = A['flight_id'].map(st)
    return A


def finish_scores(S):
    Q = S[[f'q_{c}' for c in CLASSES]].to_numpy()
    S['pred'] = [CLASSES[i] for i in Q.argmax(1)]
    S['conf'] = Q.max(1)
    # coarse level: probabilities summed within each coarse class, argmax and confidence recomputed
    for cc in COARSE_CLASSES:
        S[f'qc_{cc}'] = sum(S[f'q_{c}'] for c in CLASSES if COARSE[c] == cc)
    QC = S[[f'qc_{cc}' for cc in COARSE_CLASSES]].to_numpy()
    S['coarse_family'] = S['family'].map(COARSE)
    S['coarse_pred'] = [COARSE_CLASSES[i] for i in QC.argmax(1)]
    S['coarse_conf'] = QC.max(1)
    return S


def rule_scores(A):
    """First-version fixed rule: evidence = p90 of each non-nominal class; nominal = complement; normalised."""
    S = A[['flight_id', 'family', 'subtype']].copy()
    q = np.stack([A[f'p90_{c}'].to_numpy() for c in CLASSES[1:]], 1)
    qn = np.maximum(0.0, 1.0 - q.max(1))
    v = np.column_stack([qn, q]); v = v / np.maximum(v.sum(1, keepdims=True), 1e-9)
    for i, c in enumerate(CLASSES):
        S[f'q_{c}'] = v[:, i]
    return finish_scores(S)


def oof_probs(make_model, W_tr, feats, seed, n_folds=4):
    """Out-of-fold window probabilities on the training flights (folds are flights, stratified by family)."""
    F_tr = W_tr.groupby('flight_id')['family'].first().reset_index()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    P_oof = np.zeros((len(W_tr), len(CLASSES)))
    fid = W_tr['flight_id'].to_numpy()
    for _, ho_idx in skf.split(F_tr['flight_id'], F_tr['family']):
        ho = set(F_tr['flight_id'].iloc[ho_idx])
        m_ho = np.isin(fid, list(ho)); m_tr = ~m_ho
        model = fit(make_model(seed), W_tr.loc[m_tr, feats], W_tr.loc[m_tr, 'window_label'])
        P_oof[m_ho] = predict_proba(model, W_tr.loc[m_ho, feats])
    return P_oof


def stacked_pipeline(W_tr, feats, seed, n_folds=4):
    """Cross-fit the window model on training flights, fit the flight model on out-of-fold aggregates, then refit
    the window model on all training flights. Returns (window model, flight model, aggregate columns, temperature
    fitted on the out-of-fold training probabilities, so calibration flights are used for conformal thresholds only)."""
    P_oof = oof_probs(make_xgb, W_tr, feats, seed, n_folds)
    T = fit_temperature(P_oof, np.array([CLASSES.index(v) for v in W_tr['window_label']]))
    A_tr = flight_aggregates(W_tr, P_oof)
    agg_cols = [c for c in A_tr.columns if c not in ('flight_id', 'family', 'subtype')]
    fmodel = fit(make_flight_model(seed), A_tr[agg_cols], A_tr['family'])
    wmodel = fit(make_xgb(seed), W_tr[feats], W_tr['window_label'])
    stacked_pipeline.last_oof = P_oof
    return wmodel, fmodel, agg_cols, T


class NoveltyGate:
    """Distributional abstention: flags a flight whose aggregate evidence lies outside the calibration distribution.
    Two scorers on standardised flight aggregates fitted on training flights: Mahalanobis distance with Ledoit-Wolf
    covariance, and an isolation forest. Thresholds are the (1 - q) quantile of the calibration flights' scores, so
    in-distribution flights are withheld by the gate at rate about q."""

    def __init__(self, q=0.05, seed=0):
        self.q, self.seed = q, seed

    def fit(self, A_tr, cols):
        from sklearn.covariance import LedoitWolf
        from sklearn.ensemble import IsolationForest
        X = A_tr[cols].to_numpy(dtype=float)
        self.cols = cols
        self.med = np.nanmedian(X, 0); X = np.where(np.isnan(X), self.med, X)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.lw = LedoitWolf().fit(Z)
        self.iso = IsolationForest(n_estimators=300, random_state=self.seed).fit(Z)
        return self

    def scores(self, A):
        X = A[self.cols].to_numpy(dtype=float); X = np.where(np.isnan(X), self.med, X)
        Z = (X - self.mu) / self.sd
        return {'mahal': self.lw.mahalanobis(Z), 'iso': -self.iso.score_samples(Z)}

    def calibrate(self, A_cal):
        sc = self.scores(A_cal)
        self.th = {k: float(np.quantile(v, 1 - self.q)) for k, v in sc.items()}
        return self

    def flags(self, A):
        sc = self.scores(A)
        return {k: (sc[k] > self.th[k]) for k in sc}


def stacked_scores(W, P, fmodel, agg_cols):
    A = flight_aggregates(W, P)
    S = A[['flight_id', 'family', 'subtype']].copy()
    Q = predict_proba(fmodel, A[agg_cols])
    for i, c in enumerate(CLASSES):
        S[f'q_{c}'] = Q[:, i]
    return finish_scores(S)


PROB_COLS = lambda cols: [c for c in cols if c.split('_')[0] in ('max', 'p90', 'mean', 'frac', 'run')]
RAW_COLS = lambda cols: [c for c in cols if c not in PROB_COLS(cols)]


def flight_calibration(S, level='fine'):
    """Probability calibration of the flight-level probabilities (10 equal-width bins for ECE)."""
    if level == 'fine':
        Q = S[[f'q_{c}' for c in CLASSES]].to_numpy(); y = np.array([CLASSES.index(v) for v in S['family']])
    else:
        Q = S[[f'qc_{c}' for c in COARSE_CLASSES]].to_numpy(); y = np.array([COARSE_CLASSES.index(v) for v in S['coarse_family']])
    return {'ece10': ece_top(Q, y, bins=10), 'nll': nll(Q, y), 'brier': brier(Q, y)}


# ------------------------------------------------------------------ conformal and selective prediction
def conformal_thresholds(S_cal, alpha, level='fine'):
    """Class-conditional split-conformal thresholds. level='coarse' calibrates separately on the summed coarse
    probabilities and coarse labels (a coarse set is never derived by projecting a fine set)."""
    classes, qp, fam = (CLASSES, 'q_', 'family') if level == 'fine' else (COARSE_CLASSES, 'qc_', 'coarse_family')
    th = {}
    for c in classes:
        s = 1.0 - S_cal.loc[S_cal[fam] == c, f'{qp}{c}'].to_numpy()
        n = len(s)
        if n == 0:
            th[c] = 1.0; continue
        k = int(np.ceil((n + 1) * (1 - alpha)))
        th[c] = float(np.sort(s)[min(k, n) - 1]) if k <= n else 1.0
    return th


def conformal_sets(S, th, level='fine'):
    classes, qp = (CLASSES, 'q_') if level == 'fine' else (COARSE_CLASSES, 'qc_')
    return [[c for c in classes if 1.0 - r[f'{qp}{c}'] <= th[c]] for _, r in S.iterrows()]


def conformal_report(S_test, th, tag, level='fine'):
    classes, fam = (CLASSES, 'family') if level == 'fine' else (COARSE_CLASSES, 'coarse_family')
    sets = conformal_sets(S_test, th, level)
    out = []
    for c in classes:
        m = (S_test[fam] == c).to_numpy()
        if not m.any():
            continue
        sel = [s for s, mm in zip(sets, m) if mm]
        preds = S_test.loc[m, 'pred' if level == 'fine' else 'coarse_pred'].tolist()
        out.append({'tag': tag, 'level': level, 'family': c, 'coverage': float(np.mean([c in s for s in sel])),
                    'avg_set_size': float(np.mean([len(s) for s in sel])),
                    'multi_label_rate': float(np.mean([len(s) > 1 for s in sel])),
                    'empty_rate': float(np.mean([len(s) == 0 for s in sel])),
                    'non_singleton_rate': float(np.mean([len(s) != 1 for s in sel])),
                    'verdict_outside_set_rate': float(np.mean([p not in s for p, s in zip(preds, sel)])),
                    'operational_abstain_rate': float(np.mean([(len(s) != 1) or (p not in s) for p, s in zip(preds, sel)])),
                    'n': int(m.sum())})
    return pd.DataFrame(out), sets


def risk_coverage(S):
    s = S.sort_values('conf', ascending=False)
    correct = (s['pred'] == s['family']).to_numpy().astype(float)
    cov = np.arange(1, len(s) + 1) / len(s)
    risk = 1 - np.cumsum(correct) / np.arange(1, len(s) + 1)
    trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return pd.DataFrame({'coverage': cov, 'risk': risk}), float(trap(risk, cov))


# ------------------------------------------------------------------ helpers
def split_flights(F, rng, n_train=20, n_cal=20, n_test=20):
    """Per family: n_train training, n_cal calibration, n_test test flights; anything left over is 'extra' and is scored
    separately so the primary test set has the same per-class mixture as the calibration set."""
    tr, ca, te, ex = [], [], [], []
    for c in CLASSES:
        ids = F.loc[F.family == c, 'flight_id'].tolist(); rng.shuffle(ids)
        tr += ids[:n_train]; ca += ids[n_train:n_train + n_cal]
        te += ids[n_train + n_cal:n_train + n_cal + n_test]; ex += ids[n_train + n_cal + n_test:]
    return set(tr), set(ca), set(te), set(ex)


def window_metrics(P, y, tag):
    yi = np.array([CLASSES.index(v) for v in y])
    return {'tag': tag, 'acc': accuracy_score(yi, P.argmax(1)), 'macro_f1': f1_score(yi, P.argmax(1), average='macro'),
            'ece': ece_top(P, yi), 'nll': nll(P, yi), 'brier': brier(P, yi)}


def sub(W, ids):
    return W[W.flight_id.isin(ids)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--only', default='all', help='comma-separated subset of e0,e1,e2,e3 to run (others are skipped; existing files kept)')
    ap.add_argument('--e2_reps', type=int, default=3, help='seeds for the unseen-subtype experiment')
    args = ap.parse_args()
    RUN = set(['e0', 'e1', 'e2', 'e3', 'e5']) if args.only == 'all' else set(args.only.split(','))
    fdir, out = pathlib.Path(args.features), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    W = engineer(pd.read_csv(fdir / 'windows.csv'))
    feats = [c for c in W.columns if c not in META and not c.startswith('rx_') and pd.api.types.is_numeric_dtype(W[c])]
    Ws = W[W.source == 'sih'].reset_index(drop=True)
    Ww = W[W.source == 'whelan'].reset_index(drop=True)
    F = Ws.groupby('flight_id').agg(family=('family', 'first'), subtype=('subtype', 'first')).reset_index()
    print(f'features used: {len(feats)} (base + context + flight-relative)')
    print('SIH windows:', len(Ws), '| flights:', len(F), '| Whelan windows:', len(Ww))
    print('window labels:', Ws.window_label.value_counts().to_dict())

    # ---------------- E0 leakage audit, paired with matched training-window counts
    e0 = []
    for rep in (range(args.reps) if 'e0' in RUN else []):
        rng = np.random.RandomState(500 + rep)
        tr, ca, te, ex = split_flights(F, rng, n_train=30, n_cal=0, n_test=10 ** 6)
        W_tr, W_te = sub(Ws, tr), sub(Ws, te)
        m = fit(make_xgb(rep), W_tr[feats], W_tr['window_label'])
        g = window_metrics(predict_proba(m, W_te[feats]), W_te['window_label'], 'flight_grouped_split'); g['rep'] = rep; g['n_train_windows'] = len(W_tr)
        idx = rng.permutation(len(Ws)); k = len(W_tr)
        m = fit(make_xgb(rep), Ws.loc[idx[:k], feats], Ws.loc[idx[:k], 'window_label'])
        r = window_metrics(predict_proba(m, Ws.loc[idx[k:], feats]), Ws.loc[idx[k:], 'window_label'], 'random_window_split'); r['rep'] = rep; r['n_train_windows'] = k
        e0 += [g, r]
    if e0:
        E0 = pd.DataFrame(e0); E0.to_csv(out / 'e0_leakage_audit.csv', index=False)
        piv = E0.pivot(index='rep', columns='tag', values='acc'); d = piv['random_window_split'] - piv['flight_grouped_split']
        print('\n[E0] leakage audit (window-level), mean over paired reps:\n', E0.groupby('tag')[['acc', 'macro_f1', 'ece', 'nll', 'brier']].mean().round(4).to_string())
        print(f'[E0] paired accuracy difference random - grouped: {d.mean():.4f} ± {d.std(ddof=0):.4f} over {len(d)} reps')

    # ---------------- E1 in-distribution, flight-grouped, repeated
    GAP_FEATS = [c for c in feats if c.startswith('baro_gap_frac') or c.startswith('mag_gap_frac')]
    feats_nogap = [c for c in feats if c not in GAP_FEATS]
    wm, fm, cf, rc_all, extra_rows = [], [], [], [], []
    for rep in (range(args.reps) if 'e1' in RUN else []):
        rng = np.random.RandomState(100 + rep)
        tr, ca, te, ex = split_flights(F, rng)
        W_tr, W_ca, W_te, W_ex = sub(Ws, tr), sub(Ws, ca), sub(Ws, te), sub(Ws, ex)
        # full stacked pipeline (its out-of-fold window probabilities also give the temperature for the window and rule branches)
        wmodel, fmodel, agg_cols, T = stacked_pipeline(W_tr, feats, rep)
        for name, mk in (('xgb', make_xgb), ('logreg', make_logreg)):
            model = wmodel if name == 'xgb' else fit(mk(rep), W_tr[feats], W_tr['window_label'])
            P_te = predict_proba(model, W_te[feats])
            T_m = T if name == 'xgb' else fit_temperature(oof_probs(mk, W_tr, feats, rep), np.array([CLASSES.index(v) for v in W_tr['window_label']]))
            wm.append({'rep': rep, 'model': name, 'T': T_m, **window_metrics(P_te, W_te['window_label'], 'raw')})
            wm.append({'rep': rep, 'model': name, 'T': T_m, **window_metrics(apply_temperature(P_te, T_m), W_te['window_label'], 'temp_scaled')})
        P_oof = stacked_pipeline.last_oof
        A_tr = flight_aggregates(W_tr, P_oof)
        f_probs = fit(make_flight_model(rep), A_tr[PROB_COLS(agg_cols)], A_tr['family'])
        f_raw = fit(make_flight_model(rep), A_tr[RAW_COLS(agg_cols)], A_tr['family'])
        P_cal, P_te, P_ex = predict_proba(wmodel, W_ca[feats]), predict_proba(wmodel, W_te[feats]), (predict_proba(wmodel, W_ex[feats]) if len(W_ex) else None)
        P_cal_t, P_te_t = apply_temperature(P_cal, T), apply_temperature(P_te, T)
        # sensitivity: full stacked pipeline without the derived-topic gap features
        wmodel_ng, fmodel_ng, agg_cols_ng, _ = stacked_pipeline(W_tr, feats_nogap, rep)
        variants = {
            'rule': (rule_scores(flight_aggregates(W_ca, P_cal_t)), rule_scores(flight_aggregates(W_te, P_te_t)),
                     rule_scores(flight_aggregates(W_ex, apply_temperature(P_ex, T))) if len(W_ex) else None),
            'stacked_probs_only': (None, None, None), 'stacked_raw_only': (None, None, None),
            'stacked': (stacked_scores(W_ca, P_cal, fmodel, agg_cols), stacked_scores(W_te, P_te, fmodel, agg_cols),
                        stacked_scores(W_ex, P_ex, fmodel, agg_cols) if len(W_ex) else None),
            'stacked_no_gap': (stacked_scores(W_ca, predict_proba(wmodel_ng, W_ca[feats_nogap]), fmodel_ng, agg_cols_ng),
                               stacked_scores(W_te, predict_proba(wmodel_ng, W_te[feats_nogap]), fmodel_ng, agg_cols_ng), None),
        }
        for tag, fm_, cols in (('stacked_probs_only', f_probs, PROB_COLS(agg_cols)), ('stacked_raw_only', f_raw, RAW_COLS(agg_cols))):
            def sc(W, P):
                A = flight_aggregates(W, P); S = A[['flight_id', 'family', 'subtype']].copy(); Q = predict_proba(fm_, A[cols])
                for i, c in enumerate(CLASSES):
                    S[f'q_{c}'] = Q[:, i]
                return finish_scores(S)
            variants[tag] = (sc(W_ca, P_cal), sc(W_te, P_te), None)
        thresholds = {}
        for agg, (S_cal, S_te, S_ex) in variants.items():
            rcurve, aurc = risk_coverage(S_te)
            cal_f, cal_c = flight_calibration(S_te, 'fine'), flight_calibration(S_te, 'coarse')
            fm.append({'rep': rep, 'aggregation': agg, 'flight_acc': accuracy_score(S_te.family, S_te.pred),
                       'flight_macro_f1': f1_score(S_te.family, S_te.pred, average='macro'), 'aurc': aurc,
                       'coarse_acc': accuracy_score(S_te.coarse_family, S_te.coarse_pred),
                       'coarse_macro_f1': f1_score(S_te.coarse_family, S_te.coarse_pred, average='macro'),
                       'flight_ece10': cal_f['ece10'], 'flight_nll': cal_f['nll'], 'flight_brier': cal_f['brier'],
                       'coarse_ece10': cal_c['ece10'], 'coarse_nll': cal_c['nll'], 'coarse_brier': cal_c['brier']})
            rcurve['rep'] = rep; rcurve['aggregation'] = agg; rc_all.append(rcurve)
            if agg in ('rule', 'stacked'):
                for alpha in (0.10, 0.20):
                    for level in ('fine', 'coarse'):
                        th = conformal_thresholds(S_cal, alpha, level)
                        thresholds[f'{agg}_{level}_alpha{alpha}'] = th
                        rep_df, sets = conformal_report(S_te, th, f'{agg}_alpha{alpha}', level)
                        rep_df['rep'] = rep; rep_df['aggregation'] = agg; rep_df['alpha'] = alpha
                        cf.append(rep_df)
                        S_te[f'set_{level}_a{alpha}'] = [','.join(x) for x in sets]
                        if S_ex is not None and len(S_ex):
                            ex_df, _ = conformal_report(S_ex, th, f'{agg}_alpha{alpha}', level)
                            ex_df['rep'] = rep; ex_df['aggregation'] = agg; ex_df['alpha'] = alpha
                            ex_df['acc'] = accuracy_score(S_ex.family, S_ex.pred) if level == 'fine' else accuracy_score(S_ex.coarse_family, S_ex.coarse_pred)
                            extra_rows.append(ex_df)
                S_te.assign(rep=rep, aggregation=agg).to_csv(out / f'e1_rep{rep}_{agg}_flight_scores.csv', index=False)
                S_cal.assign(rep=rep, aggregation=agg).to_csv(out / f'e1_rep{rep}_{agg}_calibration_scores.csv', index=False)
        (out / f'e1_rep{rep}_thresholds.json').write_text(json.dumps(thresholds, indent=1))
        if rep == 0:
            imp = getattr(wmodel, 'feature_importances_', None)
            if imp is not None:
                pd.DataFrame({'feature': feats, 'importance': imp}).sort_values('importance', ascending=False)\
                    .to_csv(out / 'e1_rep0_window_feature_importance.csv', index=False)
            lr = fmodel[-1]
            pd.DataFrame(lr.coef_, index=[CLASSES[int(i)] for i in lr.classes_], columns=agg_cols).T\
                .to_csv(out / 'e1_rep0_flight_model_coefficients.csv')
    if not wm:
        WM = FM = CF = None
    else:
        WM, FM, CF = pd.DataFrame(wm), pd.DataFrame(fm), pd.concat(cf, ignore_index=True)
    if WM is not None:
        WM.to_csv(out / 'e1_window_metrics.csv', index=False); FM.to_csv(out / 'e1_flight_metrics.csv', index=False)
    if WM is not None:
        CF.to_csv(out / 'e1_conformal.csv', index=False); pd.concat(rc_all).to_csv(out / 'e1_risk_coverage.csv', index=False)
    if extra_rows:
        pd.concat(extra_rows, ignore_index=True).to_csv(out / 'e1_extra_flights.csv', index=False)
    if WM is not None:
        print('\n[E1] window metrics, mean over reps:\n', WM.groupby(['model', 'tag'])[['acc', 'macro_f1', 'ece', 'nll', 'brier']].mean().round(4).to_string())
        print('\n[E1] flight-level, mean (std) over reps:\n', FM.groupby('aggregation')[['flight_acc', 'flight_macro_f1', 'aurc', 'coarse_acc', 'coarse_macro_f1']].agg(['mean', 'std']).round(3).to_string())
        print('\n[E1] flight-level probability calibration, mean over reps:\n', FM.groupby('aggregation')[['flight_ece10', 'flight_nll', 'flight_brier', 'coarse_ece10', 'coarse_nll', 'coarse_brier']].mean().round(3).to_string())
        print('\n[E1] class-conditional conformal at flight level, mean (std) over reps:\n',
              CF.groupby(['aggregation', 'level', 'alpha', 'family'])[['coverage', 'avg_set_size', 'multi_label_rate', 'empty_rate', 'non_singleton_rate', 'verdict_outside_set_rate', 'operational_abstain_rate']].agg(['mean', 'std']).round(3).to_string())
    if extra_rows:
        EX = pd.concat(extra_rows, ignore_index=True)
        print('\n[E1] extra flights beyond the balanced test set (scored with the same thresholds), mean over reps:\n',
              EX.groupby(['aggregation', 'level', 'alpha', 'family'])[['n', 'acc', 'coverage', 'operational_abstain_rate']].mean().round(3).to_string())

    # ---------------- E2 leave-one-subtype-out: ONE fitted pipeline per hold-out and seed, evaluated on a disjoint
    # seen-subtype test set and on the withheld subtype, so model, training volume and thresholds are held fixed
    def score_set(S, fam, S_cal):
        cfam = COARSE[fam]
        row = {'n': len(S), 'acc': float((S.pred == fam).mean()), 'coarse_acc': float((S.coarse_pred == cfam).mean()),
               'mean_conf': float(S.conf.mean()), 'pred_dist': json.dumps(S.pred.value_counts().to_dict())}
        for alpha in (0.10, 0.20):
            th = conformal_thresholds(S_cal, alpha); sets = conformal_sets(S, th)
            row[f'coverage_a{alpha}'] = float(np.mean([fam in x for x in sets]))
            row[f'operational_abstain_a{alpha}'] = float(np.mean([(len(x) != 1) or (p not in x) for p, x in zip(S.pred, sets)]))
            thc = conformal_thresholds(S_cal, alpha, 'coarse'); csets = conformal_sets(S, thc, 'coarse')
            row[f'coarse_coverage_a{alpha}'] = float(np.mean([cfam in x for x in csets]))
            row[f'coarse_operational_abstain_a{alpha}'] = float(np.mean([(len(x) != 1) or (p not in x) for p, x in zip(S.coarse_pred, csets)]))
        return row

    per_seed, splits, gate_scores_all = [], [], []
    E2_SEEDS = list(range(args.e2_reps))
    for fam in (CLASSES[1:] if 'e2' in RUN else []):
        for st in sorted(F.loc[F.family == fam, 'subtype'].unique()):
            held = F.loc[F.subtype == st, 'flight_id'].tolist()
            for seed in E2_SEEDS:
                rng = np.random.RandomState(700 + seed)
                seen = F.loc[(F.family == fam) & (F.subtype != st), 'flight_id'].tolist(); rng.shuffle(seen)
                n_tr = len(seen) // 3; n_ca = len(seen) // 3
                tr, ca = seen[:n_tr], seen[n_tr:n_tr + n_ca]; seen_test = seen[n_tr + n_ca:]
                alloc = {fam: (n_tr, n_ca, len(seen_test))}
                for c in CLASSES:
                    if c == fam:
                        continue
                    ids = F.loc[F.family == c, 'flight_id'].tolist(); rng.shuffle(ids)
                    tr += ids[:20]; ca += ids[20:40]; alloc[c] = (20, 20, 0)
                W_tr, W_ca = sub(Ws, tr), sub(Ws, ca)
                wmodel, fmodel, agg_cols, T = stacked_pipeline(W_tr, feats, 700 + seed)
                A_tr = flight_aggregates(W_tr, stacked_pipeline.last_oof)
                A_ca = flight_aggregates(W_ca, predict_proba(wmodel, W_ca[feats]))
                A_seen = flight_aggregates(sub(Ws, seen_test), predict_proba(wmodel, sub(Ws, seen_test)[feats]))
                A_held = flight_aggregates(sub(Ws, held), predict_proba(wmodel, sub(Ws, held)[feats]))
                gate = NoveltyGate(q=0.05, seed=700 + seed).fit(A_tr, agg_cols).calibrate(A_ca)
                S_cal = stacked_scores(W_ca, predict_proba(wmodel, W_ca[feats]), fmodel, agg_cols)
                S_seen = stacked_scores(sub(Ws, seen_test), predict_proba(wmodel, sub(Ws, seen_test)[feats]), fmodel, agg_cols)
                S_held = stacked_scores(sub(Ws, held), predict_proba(wmodel, sub(Ws, held)[feats]), fmodel, agg_cols)
                r_seen, r_held = score_set(S_seen, fam, S_cal), score_set(S_held, fam, S_cal)
                # distributional abstention gate: fraction withheld by each scorer, and combined with the conformal policy
                for tag_, S_, A_, r_ in (('seen', S_seen, A_seen, r_seen), ('unseen', S_held, A_held, r_held)):
                    fl = gate.flags(A_)
                    th = conformal_thresholds(S_cal, 0.10); sets = conformal_sets(S_, th)
                    conf_abst = np.array([(len(x) != 1) or (p not in x) for p, x in zip(S_.pred, sets)])
                    for k in ('mahal', 'iso'):
                        r_[f'gate_{k}_withheld'] = float(fl[k].mean())
                        r_[f'gate_{k}_or_conformal_withheld'] = float((fl[k] | conf_abst).mean())
                    S_['gate_mahal'] = fl['mahal']; S_['gate_iso'] = fl['iso']
                # threshold-free separability of each novelty score (and of plain confidence) between the withheld subtype and
                # the seen-subtype test set: AUROC per seed, and the catch rate at a fixed 5% false-abstention budget on seen flights
                from sklearn.metrics import roc_auc_score
                sc_seen, sc_un = gate.scores(A_seen), gate.scores(A_held)
                sc_seen['one_minus_conf'] = 1.0 - S_seen['conf'].to_numpy(); sc_un['one_minus_conf'] = 1.0 - S_held['conf'].to_numpy()
                for k in ('mahal', 'iso', 'one_minus_conf'):
                    y = np.r_[np.ones(len(sc_un[k])), np.zeros(len(sc_seen[k]))]; x = np.r_[sc_un[k], sc_seen[k]]
                    r_held[f'gate_{k}_auroc'] = float(roc_auc_score(y, x)) if len(sc_un[k]) and len(sc_seen[k]) else np.nan
                    cut = float(np.quantile(sc_seen[k], 0.95)) if len(sc_seen[k]) else np.nan
                    r_held[f'gate_{k}_catch_at_5pct_fpr'] = float(np.mean(sc_un[k] > cut)) if len(sc_un[k]) else np.nan
                gate_rows = []
                for set_name, S_, sc in (('seen_test', S_seen, sc_seen), ('unseen', S_held, sc_un), ('calibration', S_cal, {**gate.scores(A_ca), 'one_minus_conf': 1.0 - S_cal['conf'].to_numpy()})):
                    for i, fid in enumerate(S_['flight_id']):
                        gate_rows.append({'held_out_subtype': st, 'seed': seed, 'set': set_name, 'flight_id': fid, 'family': S_['family'].iloc[i],
                                          'mahal': float(sc['mahal'][i]), 'iso': float(sc['iso'][i]), 'one_minus_conf': float(sc['one_minus_conf'][i])})
                gate_scores_all.extend(gate_rows)
                per_seed.append({'held_out_family': fam, 'held_out_subtype': st, 'seed': seed,
                                 **{f'unseen_{k}': v for k, v in r_held.items()}, **{f'seen_{k}': v for k, v in r_seen.items()}})
                for c in CLASSES:
                    n_ca_c = alloc[c][1]
                    splits.append({'held_out_subtype': st, 'seed': seed, 'family': c, 'n_train': alloc[c][0], 'n_cal': n_ca_c,
                                   'n_seen_test': alloc[c][2], 'rank_alpha0.1': int(np.ceil((n_ca_c + 1) * 0.9)),
                                   'rank_alpha0.2': int(np.ceil((n_ca_c + 1) * 0.8)), 'rank_at_max_alpha0.1': int(np.ceil((n_ca_c + 1) * 0.9)) >= n_ca_c})
                if seed == 0:
                    S_held.assign(held_out_subtype=st).to_csv(out / f'e2_held_{st}_flight_scores.csv', index=False)
                    pd.DataFrame({'flight_id': A_ca['flight_id'], **gate.scores(A_ca)}).to_csv(out / f'e2_gate_cal_scores_{st}.csv', index=False)
                    S_seen.assign(held_out_subtype=st).to_csv(out / f'e2_seen_{st}_flight_scores.csv', index=False)
    if per_seed:
        PS = pd.DataFrame(per_seed); PS.to_csv(out / 'e2_per_seed.csv', index=False)
        num = [c for c in PS.columns if c not in ('held_out_family', 'held_out_subtype', 'seed', 'unseen_pred_dist', 'seen_pred_dist')]
        g = PS.groupby(['held_out_family', 'held_out_subtype'], sort=False)
        LOSO = g[num].mean().round(4); LOSO_sd = g[num].std(ddof=0).round(4)
        LOSO.columns = [f'{c}_mean' for c in LOSO.columns]; LOSO_sd.columns = [f'{c}_std' for c in LOSO_sd.columns]
        LOSO = pd.concat([LOSO, LOSO_sd], axis=1).reset_index()
        LOSO['unseen_pred_dist_seed0'] = g['unseen_pred_dist'].first().to_numpy()
        LOSO.to_csv(out / 'e2_leave_one_subtype_out.csv', index=False)
        pd.DataFrame(splits).to_csv(out / 'e2_splits.csv', index=False)
        pd.DataFrame(gate_scores_all).to_csv(out / 'e2_gate_scores.csv', index=False)
        show = ['held_out_family', 'held_out_subtype', 'unseen_n_mean', 'seen_n_mean', 'unseen_acc_mean', 'seen_acc_mean', 'unseen_coarse_acc_mean',
                'seen_coarse_acc_mean', 'unseen_mean_conf_mean', 'seen_mean_conf_mean', 'unseen_coverage_a0.1_mean', 'seen_coverage_a0.1_mean',
                'unseen_coarse_coverage_a0.1_mean', 'seen_coarse_coverage_a0.1_mean', 'unseen_operational_abstain_a0.1_mean']
        gate_show = ['held_out_subtype', 'unseen_gate_mahal_withheld_mean', 'seen_gate_mahal_withheld_mean', 'unseen_gate_iso_withheld_mean', 'seen_gate_iso_withheld_mean',
                     'unseen_gate_mahal_or_conformal_withheld_mean', 'seen_gate_mahal_or_conformal_withheld_mean', 'unseen_operational_abstain_a0.1_mean', 'seen_operational_abstain_a0.1_mean']
        sep_show = ['held_out_subtype', 'unseen_gate_mahal_auroc_mean', 'unseen_gate_mahal_auroc_std', 'unseen_gate_mahal_catch_at_5pct_fpr_mean', 'unseen_gate_iso_auroc_mean',
                    'unseen_gate_iso_catch_at_5pct_fpr_mean', 'unseen_gate_one_minus_conf_auroc_mean', 'unseen_gate_one_minus_conf_catch_at_5pct_fpr_mean']
        print(f'\n[E2] leave-one-subtype-out, one fitted pipeline per hold-out evaluated on the withheld subtype and on a disjoint seen-subtype test set, mean over {len(E2_SEEDS)} seeds:\n',
              LOSO[show].round(3).to_string(index=False))
        print(f'\n[E4] distributional abstention gate (threshold at the 95th percentile of calibration scores), fraction withheld, mean over {len(E2_SEEDS)} seeds:\n',
              LOSO[gate_show].round(3).to_string(index=False))
        print(f'\n[E4] threshold-free separability, withheld subtype against seen-subtype test set (AUROC per seed then averaged; catch rate at the score cut that withholds 5% of seen flights):\n',
              LOSO[sep_show].round(3).to_string(index=False))

    # ---------------- E5 onset localisation against the logged onset (one fitted pipeline, held-out test flights)
    if 'e5' in RUN:
        tr, ca, te, ex = split_flights(F, np.random.RandomState(900), n_train=20, n_cal=20, n_test=10 ** 6)
        W_tr, W_te = sub(Ws, tr), sub(Ws, te | ex)
        wmodel, fmodel, agg_cols, T = stacked_pipeline(W_tr, feats, 900)
        P_te = apply_temperature(predict_proba(wmodel, W_te[feats]), T)
        rows = []
        for fid, idx in W_te.groupby('flight_id', sort=False).indices.items():
            d = W_te.iloc[idx].sort_values('t_start'); p_non = 1.0 - P_te[idx][np.argsort(W_te.iloc[idx]['t_start'].to_numpy()), 0]
            t = d['t_start'].to_numpy(); fam = d['family'].iloc[0]; st = d['subtype'].iloc[0]
            onset = float(d['onset_s'].iloc[0]) if fam != 'nominal' and pd.notna(d['onset_s'].iloc[0]) else np.nan
            restore = float(d['restore_s'].iloc[0]) if pd.notna(d['restore_s'].iloc[0]) else np.nan
            alert = np.zeros(len(d), bool)
            for i in range(1, len(d)):
                if p_non[i] > 0.5 and p_non[i - 1] > 0.5:
                    alert[i - 1] = alert[i] = True
            starts = [t[i] for i in range(len(d)) if alert[i] and (i == 0 or not alert[i - 1])]
            row = {'flight_id': fid, 'family': fam, 'subtype': st, 'onset_s': onset, 'restore_s': restore, 'n_alert_episodes': len(starts),
                   'first_alert_s': starts[0] if starts else np.nan}
            if fam == 'nominal':
                row.update({'false_alert': bool(starts), 'est_onset_s': np.nan, 'onset_error_s': np.nan, 'early_alert': bool(starts)})
            else:
                after = [x for x in starts if x >= onset - 5.0]          # first episode starting at or after onset (5 s tolerance)
                est = after[0] if after else np.nan
                row.update({'est_onset_s': est, 'onset_error_s': (est - onset) if after else np.nan,
                            'detected_within_60s': bool(after) and (est - onset) <= 60.0,
                            'early_alert': any(x < onset - 5.0 for x in starts), 'false_alert': np.nan})
            rows.append(row)
        E5 = pd.DataFrame(rows); E5.to_csv(out / 'e5_onset_localisation.csv', index=False)
        att = E5[E5.family != 'nominal']
        summ = att.groupby(['family', 'subtype']).apply(lambda g: pd.Series({
            'n': len(g), 'detected_within_60s': float(g['detected_within_60s'].fillna(False).mean()),
            'median_abs_error_s': float(g['onset_error_s'].abs().median()) if g['onset_error_s'].notna().any() else np.nan,
            'within_10s': float((g['onset_error_s'].abs() <= 10.0).mean()), 'median_signed_error_s': float(g['onset_error_s'].median()) if g['onset_error_s'].notna().any() else np.nan,
            'early_alert_rate': float(g['early_alert'].mean())})).reset_index()
        nom = E5[E5.family == 'nominal']
        summ = pd.concat([summ, pd.DataFrame([{'family': 'nominal', 'subtype': 'none', 'n': len(nom), 'detected_within_60s': np.nan, 'median_abs_error_s': np.nan,
                                                 'within_10s': np.nan, 'median_signed_error_s': np.nan, 'early_alert_rate': float(nom['false_alert'].mean())}])], ignore_index=True)
        summ.to_csv(out / 'e5_onset_summary.csv', index=False)
        print('\n[E5] onset localisation with the declared alert rule against the logged onset (one fitted pipeline; early alert = any episode starting more than 5 s before onset; for nominal flights the last column is the false-alert rate):\n',
              summ.round(3).to_string(index=False))

    # ---------------- E3 Whelan case studies (stacked pipeline)
    if len(Ww) and 'e3' in RUN:
        tr, ca, te, ex = split_flights(F, np.random.RandomState(0), n_train=40, n_cal=20)
        W_tr, W_ca = sub(Ws, tr), sub(Ws, ca)
        wmodel, fmodel, agg_cols, T = stacked_pipeline(W_tr, feats, 0)
        P_ca = predict_proba(wmodel, W_ca[feats])
        S_cal = stacked_scores(W_ca, P_ca, fmodel, agg_cols)
        th10, th20 = conformal_thresholds(S_cal, 0.10), conformal_thresholds(S_cal, 0.20)
        thc10 = conformal_thresholds(S_cal, 0.10, 'coarse')
        Pw = predict_proba(wmodel, Ww[feats]); Pw_t = apply_temperature(Pw, T)
        Ww_out = Ww[['flight_id', 'family', 't_start', 't_end']].copy()
        for i, c in enumerate(CLASSES):
            Ww_out[f'p_{c}'] = Pw_t[:, i]
        Ww_out['pred'] = [CLASSES[i] for i in Pw_t.argmax(1)]
        Ww_out.to_csv(out / 'e3_whelan_window_timeline.csv', index=False)
        Sw = stacked_scores(Ww, Pw, fmodel, agg_cols)
        Sw['set_a0.10'] = [','.join(x) for x in conformal_sets(Sw, th10)]
        Sw['set_a0.20'] = [','.join(x) for x in conformal_sets(Sw, th20)]
        Sw['coarse_set_a0.10'] = [','.join(x) for x in conformal_sets(Sw, thc10, 'coarse')]
        Sw.to_csv(out / 'e3_whelan_flight_verdicts.csv', index=False)
        # receiver-derived disturbance intervals and a declared alert rule (case-study descriptives)
        ev = []
        for fid, d in Ww.groupby('flight_id', sort=False):
            d = d.sort_values('t_start'); idx = d.index.to_numpy()
            rx_dist = ((d['rx_fix_mean'] < 3) | (d['rx_sats_mean'] < 8)).to_numpy()
            p_non = 1.0 - Pw_t[idx, 0]
            alert = np.zeros(len(d), bool)
            for i in range(1, len(d)):
                if p_non[i] > 0.5 and p_non[i - 1] > 0.5:
                    alert[i - 1] = alert[i] = True
            t = d['t_start'].to_numpy()
            def intervals(mask):
                out_, start_ = [], None
                for i, v in enumerate(mask):
                    if v and start_ is None: start_ = t[i]
                    if (not v or i == len(mask) - 1) and start_ is not None:
                        out_.append((float(start_), float(t[i] + (5.0 if v else 0.0)))); start_ = None
                return out_
            dist_iv, alert_iv = intervals(rx_dist), intervals(alert)
            delay = None
            if dist_iv and alert_iv:
                first = dist_iv[0][0]; later = [a for a in alert_iv if a[1] >= first]
                delay = round(max(0.0, later[0][0] - first), 1) if later else None
            false_alerts = sum(1 for a in alert_iv if not any(a[0] < e and a[1] > s_ for s_, e in dist_iv))
            ev.append({'flight_id': fid, 'receiver_disturbance_intervals_s': json.dumps(dist_iv), 'alert_intervals_s': json.dumps(alert_iv),
                       'detection_delay_s': delay, 'alerts_outside_disturbance': false_alerts, 'n_windows': len(d)})
        pd.DataFrame(ev).to_csv(out / 'e3_whelan_events.csv', index=False)
        print('\n[E3] Whelan receiver-derived disturbance intervals, alert rule (P(non-nominal) > 0.5 for 2 consecutive windows):\n',
              pd.DataFrame(ev).to_string(index=False))
        print('\n[E3] Whelan window predicted-class fractions per flight:\n',
              Ww_out.groupby('flight_id')['pred'].value_counts(normalize=True).round(2).to_string())
        print('\n[E3] Whelan flight verdicts (stacked) with conformal sets:\n',
              Sw[['flight_id', 'family', 'pred', 'conf', 'set_a0.10', 'set_a0.20', 'coarse_pred', 'coarse_set_a0.10']].round(3).to_string(index=False))
    print('\nresults written to', out)


if __name__ == '__main__':
    main()
