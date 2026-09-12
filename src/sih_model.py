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
  E0  Leakage audit: random-window split vs flight-grouped split (window level).
  E1  In-distribution, flight-grouped: 20 train / 20 calibration / 20 test flights per family, R repetitions.
      Window metrics raw vs temperature-scaled; flight accuracy for rule and stacked aggregation; class-conditional
      conformal sets at the flight level (alpha 0.10 and 0.20; with 20 calibration flights per class alpha 0.05 would
      be the degenerate rank-20-of-20 threshold); risk-coverage and AURC.
  E2  Leave-one-subtype-out with the stacked pipeline.
  E3  Whelan live logs as external case studies.

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
META = {'t_start', 't_end', 'flight_id', 'source', 'family', 'subtype', 'onset_s', 'restore_s',
        'home_lat', 'noise_k', 'noise_t', 'noise_p', 'window_label', 'log'}
CTX_FEATS = ['posvel_rms', 'step_minus_vel_max', 'step_speed_max', 'gps_baro_dz_rms', 'dv_gps_minus_imu',
             'gps_ekf_dpos_rms', 'gps_gap_frac', 'baro_gap_frac', 'mag_gap_frac', 'baro_raw_std', 'mag_raw_std',
             'tr_gpsHpos0_absmax', 'tr_gpsHpos1_absmax', 'tr_gpsHvel0_absmax', 'tr_gpsVpos_absmax',
             'tr_baroVpos_absmax', 'tr_heading_absmax', 'inn_gpsHpos0_absmax', 'inn_gpsHvel0_absmax']
REL_FEATS = ['baro_raw_std', 'mag_raw_std', 'mag_norm_std', 'mag_norm_mean', 'gyro_std_mean', 'baro_std',
             'posvel_rms', 'gps_ekf_dpos_rms', 'gps_baro_dz_rms', 'acc_horiz_rms', 'inn_gpsHpos0_absmax',
             'inn_gpsHvel0_absmax', 'inn_baroVpos_absmax', 'inn_heading_absmax']
AGG_RAW = ['gps_gap_frac', 'baro_gap_frac', 'mag_gap_frac', 'posvel_rms', 'step_minus_vel_max', 'gps_ekf_dpos_rms',
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


def stacked_pipeline(W_tr, feats, seed, n_folds=4):
    """Cross-fit the window model on training flights, fit the flight model on out-of-fold aggregates, then refit
    the window model on all training flights. Returns (window model, flight model, aggregate columns)."""
    F_tr = W_tr.groupby('flight_id')['family'].first().reset_index()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    P_oof = np.zeros((len(W_tr), len(CLASSES)))
    fid = W_tr['flight_id'].to_numpy()
    for _, ho_idx in skf.split(F_tr['flight_id'], F_tr['family']):
        ho = set(F_tr['flight_id'].iloc[ho_idx])
        m_ho = np.isin(fid, list(ho)); m_tr = ~m_ho
        model = fit(make_xgb(seed), W_tr.loc[m_tr, feats], W_tr.loc[m_tr, 'window_label'])
        P_oof[m_ho] = predict_proba(model, W_tr.loc[m_ho, feats])
    A_tr = flight_aggregates(W_tr, P_oof)
    agg_cols = [c for c in A_tr.columns if c not in ('flight_id', 'family', 'subtype')]
    fmodel = fit(make_flight_model(seed), A_tr[agg_cols], A_tr['family'])
    wmodel = fit(make_xgb(seed), W_tr[feats], W_tr['window_label'])
    return wmodel, fmodel, agg_cols


def stacked_scores(W, P, fmodel, agg_cols):
    A = flight_aggregates(W, P)
    S = A[['flight_id', 'family', 'subtype']].copy()
    Q = predict_proba(fmodel, A[agg_cols])
    for i, c in enumerate(CLASSES):
        S[f'q_{c}'] = Q[:, i]
    return finish_scores(S)


# ------------------------------------------------------------------ conformal and selective prediction
def conformal_thresholds(S_cal, alpha):
    th = {}
    for c in CLASSES:
        s = 1.0 - S_cal.loc[S_cal.family == c, f'q_{c}'].to_numpy()
        n = len(s)
        if n == 0:
            th[c] = 1.0; continue
        k = int(np.ceil((n + 1) * (1 - alpha)))
        th[c] = float(np.sort(s)[min(k, n) - 1]) if k <= n else 1.0
    return th


def conformal_sets(S, th):
    return [[c for c in CLASSES if 1.0 - r[f'q_{c}'] <= th[c]] for _, r in S.iterrows()]


def conformal_report(S_test, th, tag):
    sets = conformal_sets(S_test, th)
    out = []
    for c in CLASSES:
        m = (S_test.family == c).to_numpy()
        if not m.any():
            continue
        sel = [s for s, mm in zip(sets, m) if mm]
        out.append({'tag': tag, 'family': c, 'coverage': float(np.mean([c in s for s in sel])),
                    'avg_set_size': float(np.mean([len(s) for s in sel])),
                    'abstain_rate': float(np.mean([len(s) != 1 for s in sel])), 'n': int(m.sum())})
    return pd.DataFrame(out), sets


def risk_coverage(S):
    s = S.sort_values('conf', ascending=False)
    correct = (s['pred'] == s['family']).to_numpy().astype(float)
    cov = np.arange(1, len(s) + 1) / len(s)
    risk = 1 - np.cumsum(correct) / np.arange(1, len(s) + 1)
    trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return pd.DataFrame({'coverage': cov, 'risk': risk}), float(trap(risk, cov))


# ------------------------------------------------------------------ helpers
def split_flights(F, rng, n_train=20, n_cal=20):
    tr, ca, te = [], [], []
    for c in CLASSES:
        ids = F.loc[F.family == c, 'flight_id'].tolist(); rng.shuffle(ids)
        tr += ids[:n_train]; ca += ids[n_train:n_train + n_cal]; te += ids[n_train + n_cal:]
    return set(tr), set(ca), set(te)


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
    args = ap.parse_args()
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

    # ---------------- E0 leakage audit
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(Ws)); half = len(Ws) // 2
    m = fit(make_xgb(0), Ws.loc[idx[:half], feats], Ws.loc[idx[:half], 'window_label'])
    e0_rand = window_metrics(predict_proba(m, Ws.loc[idx[half:], feats]), Ws.loc[idx[half:], 'window_label'], 'random_window_split')
    tr, ca, te = split_flights(F, np.random.RandomState(0), n_train=30, n_cal=0)
    m = fit(make_xgb(0), sub(Ws, tr)[feats], sub(Ws, tr)['window_label'])
    e0_grp = window_metrics(predict_proba(m, sub(Ws, te)[feats]), sub(Ws, te)['window_label'], 'flight_grouped_split')
    E0 = pd.DataFrame([e0_rand, e0_grp]); E0.to_csv(out / 'e0_leakage_audit.csv', index=False)
    print('\n[E0] leakage audit (window-level):\n', E0.round(4).to_string(index=False))

    # ---------------- E1 in-distribution, flight-grouped, repeated
    wm, fm, cf, rc_all = [], [], [], []
    for rep in range(args.reps):
        rng = np.random.RandomState(100 + rep)
        tr, ca, te = split_flights(F, rng)
        W_tr, W_ca, W_te = sub(Ws, tr), sub(Ws, ca), sub(Ws, te)
        for name, mk in (('xgb', make_xgb), ('logreg', make_logreg)):
            model = fit(mk(rep), W_tr[feats], W_tr['window_label'])
            P_cal, P_te = predict_proba(model, W_ca[feats]), predict_proba(model, W_te[feats])
            T = fit_temperature(P_cal, np.array([CLASSES.index(v) for v in W_ca['window_label']]))
            wm.append({'rep': rep, 'model': name, 'T': T, **window_metrics(P_te, W_te['window_label'], 'raw')})
            wm.append({'rep': rep, 'model': name, 'T': T, **window_metrics(apply_temperature(P_te, T), W_te['window_label'], 'temp_scaled')})
        wmodel, fmodel, agg_cols = stacked_pipeline(W_tr, feats, rep)
        P_cal, P_te = predict_proba(wmodel, W_ca[feats]), predict_proba(wmodel, W_te[feats])
        T = fit_temperature(P_cal, np.array([CLASSES.index(v) for v in W_ca['window_label']]))
        P_cal_t, P_te_t = apply_temperature(P_cal, T), apply_temperature(P_te, T)
        for agg in ('rule', 'stacked'):
            if agg == 'rule':
                S_cal, S_te = rule_scores(flight_aggregates(W_ca, P_cal_t)), rule_scores(flight_aggregates(W_te, P_te_t))
            else:
                S_cal, S_te = stacked_scores(W_ca, P_cal, fmodel, agg_cols), stacked_scores(W_te, P_te, fmodel, agg_cols)
            rcurve, aurc = risk_coverage(S_te)
            fm.append({'rep': rep, 'aggregation': agg, 'flight_acc': accuracy_score(S_te.family, S_te.pred),
                       'flight_macro_f1': f1_score(S_te.family, S_te.pred, average='macro'), 'aurc': aurc})
            for alpha in (0.10, 0.20):
                th = conformal_thresholds(S_cal, alpha)
                rep_df, _ = conformal_report(S_te, th, f'{agg}_alpha{alpha}')
                rep_df['rep'] = rep; rep_df['aggregation'] = agg; rep_df['alpha'] = alpha
                cf.append(rep_df)
            rcurve['rep'] = rep; rcurve['aggregation'] = agg; rc_all.append(rcurve)
            if rep == 0:
                S_te.to_csv(out / f'e1_rep0_{agg}_flight_scores.csv', index=False)
        if rep == 0:
            imp = getattr(wmodel, 'feature_importances_', None)
            if imp is not None:
                pd.DataFrame({'feature': feats, 'importance': imp}).sort_values('importance', ascending=False)\
                    .to_csv(out / 'e1_rep0_window_feature_importance.csv', index=False)
            lr = fmodel[-1]
            pd.DataFrame(lr.coef_, index=[CLASSES[int(i)] for i in lr.classes_], columns=agg_cols).T\
                .to_csv(out / 'e1_rep0_flight_model_coefficients.csv')
    WM, FM, CF = pd.DataFrame(wm), pd.DataFrame(fm), pd.concat(cf, ignore_index=True)
    WM.to_csv(out / 'e1_window_metrics.csv', index=False); FM.to_csv(out / 'e1_flight_metrics.csv', index=False)
    CF.to_csv(out / 'e1_conformal.csv', index=False); pd.concat(rc_all).to_csv(out / 'e1_risk_coverage.csv', index=False)
    print('\n[E1] window metrics, mean over reps:\n', WM.groupby(['model', 'tag'])[['acc', 'macro_f1', 'ece', 'nll', 'brier']].mean().round(4).to_string())
    print('\n[E1] flight-level, mean (std) over reps:\n', FM.groupby('aggregation')[['flight_acc', 'flight_macro_f1', 'aurc']].agg(['mean', 'std']).round(3).to_string())
    print('\n[E1] class-conditional conformal at flight level, mean over reps:\n',
          CF.groupby(['aggregation', 'alpha', 'family'])[['coverage', 'avg_set_size', 'abstain_rate']].mean().round(3).to_string())

    # ---------------- E2 leave-one-subtype-out (stacked pipeline)
    loso = []
    for fam in CLASSES[1:]:
        for st in sorted(F.loc[F.family == fam, 'subtype'].unique()):
            held = set(F.loc[F.subtype == st, 'flight_id'])
            F_in = F[~F.flight_id.isin(held)]
            rng = np.random.RandomState(7)
            tr, ca = [], []
            for c in CLASSES:
                ids = F_in.loc[F_in.family == c, 'flight_id'].tolist(); rng.shuffle(ids)
                n_tr = min(20, len(ids) // 2); n_ca = min(20, len(ids) - n_tr)
                tr += ids[:n_tr]; ca += ids[n_tr:n_tr + n_ca]
            W_tr, W_ca, W_h = sub(Ws, tr), sub(Ws, ca), sub(Ws, held)
            wmodel, fmodel, agg_cols = stacked_pipeline(W_tr, feats, 7)
            S_cal = stacked_scores(W_ca, predict_proba(wmodel, W_ca[feats]), fmodel, agg_cols)
            S_h = stacked_scores(W_h, predict_proba(wmodel, W_h[feats]), fmodel, agg_cols)
            row = {'held_out_family': fam, 'held_out_subtype': st, 'n_held': len(S_h),
                   'acc_on_held': float((S_h.pred == fam).mean()),
                   'pred_dist_on_held': json.dumps(S_h.pred.value_counts().to_dict()),
                   'mean_conf_on_held': float(S_h.conf.mean())}
            for alpha in (0.10, 0.20):
                th = conformal_thresholds(S_cal, alpha)
                sets = conformal_sets(S_h, th)
                row[f'coverage_a{alpha}'] = float(np.mean([fam in s for s in sets]))
                row[f'abstain_a{alpha}'] = float(np.mean([len(s) != 1 for s in sets]))
            loso.append(row)
            S_h.assign(held_out_subtype=st).to_csv(out / f'e2_held_{st}_flight_scores.csv', index=False)
    LOSO = pd.DataFrame(loso); LOSO.to_csv(out / 'e2_leave_one_subtype_out.csv', index=False)
    print('\n[E2] leave-one-subtype-out (stacked, flight level, unseen subtype):\n', LOSO.round(3).to_string(index=False))

    # ---------------- E3 Whelan case studies (stacked pipeline)
    if len(Ww):
        tr, ca, te = split_flights(F, np.random.RandomState(0), n_train=40, n_cal=20)
        W_tr, W_ca = sub(Ws, tr), sub(Ws, ca)
        wmodel, fmodel, agg_cols = stacked_pipeline(W_tr, feats, 0)
        P_ca = predict_proba(wmodel, W_ca[feats])
        S_cal = stacked_scores(W_ca, P_ca, fmodel, agg_cols)
        th10, th20 = conformal_thresholds(S_cal, 0.10), conformal_thresholds(S_cal, 0.20)
        T = fit_temperature(P_ca, np.array([CLASSES.index(v) for v in W_ca['window_label']]))
        Pw = predict_proba(wmodel, Ww[feats]); Pw_t = apply_temperature(Pw, T)
        Ww_out = Ww[['flight_id', 'family', 't_start', 't_end']].copy()
        for i, c in enumerate(CLASSES):
            Ww_out[f'p_{c}'] = Pw_t[:, i]
        Ww_out['pred'] = [CLASSES[i] for i in Pw_t.argmax(1)]
        Ww_out.to_csv(out / 'e3_whelan_window_timeline.csv', index=False)
        Sw = stacked_scores(Ww, Pw, fmodel, agg_cols)
        Sw['set_a0.10'] = [','.join(s) for s in conformal_sets(Sw, th10)]
        Sw['set_a0.20'] = [','.join(s) for s in conformal_sets(Sw, th20)]
        Sw.to_csv(out / 'e3_whelan_flight_verdicts.csv', index=False)
        print('\n[E3] Whelan window predicted-class fractions per flight:\n',
              Ww_out.groupby('flight_id')['pred'].value_counts(normalize=True).round(2).to_string())
        print('\n[E3] Whelan flight verdicts (stacked) with conformal sets:\n',
              Sw[['flight_id', 'family', 'pred', 'conf', 'set_a0.10', 'set_a0.20']].round(3).to_string(index=False))
    print('\nresults written to', out)


if __name__ == '__main__':
    main()
