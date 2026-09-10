#!/usr/bin/env python3
"""
sih_model.py  --  window detector + trust layer, evaluated at the FLIGHT level.

Experiments
  E0  Leakage audit: random-window split vs flight-grouped split (same model, same features).
  E1  In-distribution, flight-grouped: 20 train / 20 calibration / 20 test flights per family, R repetitions.
      Window metrics (macro-F1, ECE, NLL, Brier) before/after temperature scaling.
      Flight verdicts via a fixed aggregation rule; class-conditional conformal prediction sets at the flight
      level (alpha = 0.10, 0.20; with 20 calibration flights per class alpha = 0.05 would be the degenerate rank-20-of-20 threshold): per-class coverage, set size, abstention; risk-coverage curve and AURC.
  E2  Leave-one-subtype-out: hold out every subtype of every family in turn (train+cal without it, test on it):
      accuracy, conformal coverage and abstention on the unseen subtype.
  E3  Whelan live logs as external case studies: window-class timeline and flight verdict with conformal set.

Rules: receiver self-report (rx_*) columns are excluded from the core detector. Simulator truth is never used.
Usage:
  python sih_model.py --features /content/drive/MyDrive/datasets/features_v1 --out /content/drive/MyDrive/results/v1
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
CLASSES = ['nominal', 'spoof', 'gps_degrade', 'sensor_fault']
META = {'t_start', 't_end', 'flight_id', 'source', 'family', 'subtype', 'onset_s', 'restore_s',
        'home_lat', 'noise_k', 'noise_t', 'noise_p', 'window_label', 'log'}


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
                         LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', random_state=seed))


def fit(model, X, y):
    yi = np.array([CLASSES.index(v) for v in y])
    counts = np.bincount(yi, minlength=len(CLASSES)).astype(float)
    w = (len(yi) / (len(CLASSES) * np.maximum(counts, 1)))[yi]
    try:
        model.fit(X, yi, sample_weight=w)
    except (TypeError, ValueError):
        model.fit(X, yi)          # pipeline models: class_weight='balanced' handles imbalance instead
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


# ------------------------------------------------------------------ flight-level aggregation and conformal
def flight_scores(W, P):
    """Fixed aggregation rule: evidence for a non-nominal class = 90th percentile of its window probability
    over the flight ('any sustained window' logic); nominal = complement of the strongest evidence. Normalised."""
    rows = []
    for fid, idx in W.groupby('flight_id').indices.items():
        p = P[idx]
        q = {c: float(np.percentile(p[:, CLASSES.index(c)], 90)) for c in CLASSES[1:]}
        q['nominal'] = float(max(0.0, 1.0 - max(q.values())))
        v = np.array([q[c] for c in CLASSES]); v = v / max(v.sum(), 1e-9)
        rows.append({'flight_id': fid, **{f'q_{c}': v[i] for i, c in enumerate(CLASSES)}})
    S = pd.DataFrame(rows)
    fam = W.groupby('flight_id')['family'].first()
    sub = W.groupby('flight_id')['subtype'].first()
    S['family'] = S['flight_id'].map(fam); S['subtype'] = S['flight_id'].map(sub)
    S['pred'] = [CLASSES[i] for i in S[[f'q_{c}' for c in CLASSES]].to_numpy().argmax(1)]
    S['conf'] = S[[f'q_{c}' for c in CLASSES]].to_numpy().max(1)
    return S


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
    sets = []
    for _, r in S.iterrows():
        sets.append([c for c in CLASSES if 1.0 - r[f'q_{c}'] <= th[c]])
    return sets


def conformal_report(S_test, th, tag):
    sets = conformal_sets(S_test, th)
    out = []
    for c in CLASSES:
        m = (S_test.family == c).to_numpy()
        if not m.any():
            continue
        cov = np.mean([c in s for s, mm in zip(sets, m) if mm])
        size = np.mean([len(s) for s, mm in zip(sets, m) if mm])
        abst = np.mean([len(s) != 1 for s, mm in zip(sets, m) if mm])
        out.append({'tag': tag, 'family': c, 'coverage': cov, 'avg_set_size': size, 'abstain_rate': abst, 'n': int(m.sum())})
    return pd.DataFrame(out), sets


def risk_coverage(S):
    s = S.sort_values('conf', ascending=False)
    correct = (s['pred'] == s['family']).to_numpy().astype(float)
    cov = np.arange(1, len(s) + 1) / len(s)
    risk = 1 - np.cumsum(correct) / np.arange(1, len(s) + 1)
    aurc = float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(np.trapz(risk, cov))
    return pd.DataFrame({'coverage': cov, 'risk': risk}), aurc


# ------------------------------------------------------------------ helpers
def split_flights(F, rng, n_train=20, n_cal=20):
    tr, ca, te = [], [], []
    for c in CLASSES:
        ids = F.loc[F.family == c, 'flight_id'].tolist()
        rng.shuffle(ids)
        tr += ids[:n_train]; ca += ids[n_train:n_train + n_cal]; te += ids[n_train + n_cal:]
    return set(tr), set(ca), set(te)


def window_metrics(P, y, tag):
    yi = np.array([CLASSES.index(v) for v in y])
    return {'tag': tag, 'acc': accuracy_score(yi, P.argmax(1)), 'macro_f1': f1_score(yi, P.argmax(1), average='macro'),
            'ece': ece_top(P, yi), 'nll': nll(P, yi), 'brier': brier(P, yi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reps', type=int, default=5)
    args = ap.parse_args()
    fdir, out = pathlib.Path(args.features), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    W = pd.read_csv(fdir / 'windows.csv')
    feats = [c for c in W.columns if c not in META and not c.startswith('rx_') and pd.api.types.is_numeric_dtype(W[c])]
    Ws = W[W.source == 'sih'].reset_index(drop=True)
    Ww = W[W.source == 'whelan'].reset_index(drop=True)
    F = Ws.groupby('flight_id').agg(family=('family', 'first'), subtype=('subtype', 'first')).reset_index()
    print(f'features used ({len(feats)}):', feats)
    print('SIH windows:', len(Ws), '| flights:', len(F), '| Whelan windows:', len(Ww))
    print('window labels:', Ws.window_label.value_counts().to_dict())

    # ---------------- E0 leakage audit
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(Ws)); half = len(Ws) // 2
    m = fit(make_xgb(0), Ws.loc[idx[:half], feats], Ws.loc[idx[:half], 'window_label'])
    e0_rand = window_metrics(predict_proba(m, Ws.loc[idx[half:], feats]), Ws.loc[idx[half:], 'window_label'], 'random_window_split')
    tr, ca, te = split_flights(F, np.random.RandomState(0), n_train=30, n_cal=0)
    a, b = Ws.flight_id.isin(tr), Ws.flight_id.isin(te)
    m = fit(make_xgb(0), Ws.loc[a, feats], Ws.loc[a, 'window_label'])
    e0_grp = window_metrics(predict_proba(m, Ws.loc[b, feats]), Ws.loc[b, 'window_label'], 'flight_grouped_split')
    E0 = pd.DataFrame([e0_rand, e0_grp]); E0.to_csv(out / 'e0_leakage_audit.csv', index=False)
    print('\n[E0] leakage audit (window-level):\n', E0.round(4).to_string(index=False))

    # ---------------- E1 in-distribution, flight-grouped, repeated
    wm, fm, cf, rc_all = [], [], [], []
    for rep in range(args.reps):
        rng = np.random.RandomState(100 + rep)
        tr, ca, te = split_flights(F, rng)
        a, b, c = Ws.flight_id.isin(tr), Ws.flight_id.isin(ca), Ws.flight_id.isin(te)
        for name, mk in (('xgb', make_xgb), ('logreg', make_logreg)):
            model = fit(mk(rep), Ws.loc[a, feats], Ws.loc[a, 'window_label'])
            P_cal, P_te = predict_proba(model, Ws.loc[b, feats]), predict_proba(model, Ws.loc[c, feats])
            y_cal = np.array([CLASSES.index(v) for v in Ws.loc[b, 'window_label']])
            T = fit_temperature(P_cal, y_cal)
            wm.append({'rep': rep, 'model': name, 'T': T, **window_metrics(P_te, Ws.loc[c, 'window_label'], 'raw')})
            wm.append({'rep': rep, 'model': name, 'T': T, **window_metrics(apply_temperature(P_te, T), Ws.loc[c, 'window_label'], 'temp_scaled')})
            S_cal = flight_scores(Ws.loc[b].reset_index(drop=True), apply_temperature(P_cal, T))
            S_te = flight_scores(Ws.loc[c].reset_index(drop=True), apply_temperature(P_te, T))
            fm.append({'rep': rep, 'model': name, 'flight_acc': accuracy_score(S_te.family, S_te.pred),
                       'flight_macro_f1': f1_score(S_te.family, S_te.pred, average='macro')})
            for alpha in (0.10, 0.20):
                th = conformal_thresholds(S_cal, alpha)
                rep_df, _ = conformal_report(S_te, th, f'{name}_alpha{alpha}')
                rep_df['rep'] = rep; rep_df['model'] = name; rep_df['alpha'] = alpha
                cf.append(rep_df)
            rcurve, aurc = risk_coverage(S_te)
            rcurve['rep'] = rep; rcurve['model'] = name; rc_all.append(rcurve)
            fm[-1]['aurc'] = aurc
            if rep == 0 and name == 'xgb':
                S_te.to_csv(out / 'e1_rep0_xgb_flight_scores.csv', index=False)
                imp = getattr(model, 'feature_importances_', None)
                if imp is not None:
                    pd.DataFrame({'feature': feats, 'importance': imp}).sort_values('importance', ascending=False)\
                        .to_csv(out / 'e1_rep0_xgb_feature_importance.csv', index=False)
    WM, FM, CF = pd.DataFrame(wm), pd.DataFrame(fm), pd.concat(cf, ignore_index=True)
    WM.to_csv(out / 'e1_window_metrics.csv', index=False); FM.to_csv(out / 'e1_flight_metrics.csv', index=False)
    CF.to_csv(out / 'e1_conformal.csv', index=False); pd.concat(rc_all).to_csv(out / 'e1_risk_coverage.csv', index=False)
    print('\n[E1] window metrics, mean over reps:\n', WM.groupby(['model', 'tag'])[['acc', 'macro_f1', 'ece', 'nll', 'brier']].mean().round(4).to_string())
    print('\n[E1] flight-level, mean (std) over reps:\n', FM.groupby('model')[['flight_acc', 'flight_macro_f1', 'aurc']].agg(['mean', 'std']).round(3).to_string())
    print('\n[E1] class-conditional conformal at flight level, mean over reps:\n',
          CF.groupby(['model', 'alpha', 'family'])[['coverage', 'avg_set_size', 'abstain_rate']].mean().round(3).to_string())

    # ---------------- E2 leave-one-subtype-out
    loso = []
    for fam in CLASSES[1:]:
        for sub in sorted(F.loc[F.family == fam, 'subtype'].unique()):
            held = set(F.loc[F.subtype == sub, 'flight_id'])
            F_in = F[~F.flight_id.isin(held)]
            rng = np.random.RandomState(7)
            tr, ca, te = [], [], []
            for c in CLASSES:
                ids = F_in.loc[F_in.family == c, 'flight_id'].tolist(); rng.shuffle(ids)
                n_tr = min(20, len(ids) // 2); n_ca = min(20, len(ids) - n_tr)
                tr += ids[:n_tr]; ca += ids[n_tr:n_tr + n_ca]; te += ids[n_tr + n_ca:]
            a, b = Ws.flight_id.isin(tr), Ws.flight_id.isin(ca)
            h = Ws.flight_id.isin(held)
            model = fit(make_xgb(7), Ws.loc[a, feats], Ws.loc[a, 'window_label'])
            P_cal = predict_proba(model, Ws.loc[b, feats])
            T = fit_temperature(P_cal, np.array([CLASSES.index(v) for v in Ws.loc[b, 'window_label']]))
            S_cal = flight_scores(Ws.loc[b].reset_index(drop=True), apply_temperature(P_cal, T))
            S_h = flight_scores(Ws.loc[h].reset_index(drop=True), apply_temperature(predict_proba(model, Ws.loc[h, feats]), T))
            row = {'held_out_family': fam, 'held_out_subtype': sub, 'n_held': len(S_h),
                   'acc_on_held': float((S_h.pred == fam).mean()),
                   'pred_dist_on_held': json.dumps(S_h.pred.value_counts().to_dict()),
                   'mean_conf_on_held': float(S_h.conf.mean())}
            for alpha in (0.10, 0.20):
                th = conformal_thresholds(S_cal, alpha)
                sets = conformal_sets(S_h, th)
                row[f'coverage_a{alpha}'] = float(np.mean([fam in s for s in sets]))
                row[f'abstain_a{alpha}'] = float(np.mean([len(s) != 1 for s in sets]))
                row[f'empty_a{alpha}'] = float(np.mean([len(s) == 0 for s in sets]))
            loso.append(row)
            S_h.assign(held_out_subtype=sub).to_csv(out / f'e2_held_{sub}_flight_scores.csv', index=False)
    LOSO = pd.DataFrame(loso); LOSO.to_csv(out / 'e2_leave_one_subtype_out.csv', index=False)
    print('\n[E2] leave-one-subtype-out (flight level, unseen subtype):\n', LOSO.round(3).to_string(index=False))

    # ---------------- E3 Whelan case studies
    if len(Ww):
        tr, ca, te = split_flights(F, np.random.RandomState(0), n_train=40, n_cal=20)
        a, b = Ws.flight_id.isin(tr), Ws.flight_id.isin(ca)
        model = fit(make_xgb(0), Ws.loc[a, feats], Ws.loc[a, 'window_label'])
        P_cal = predict_proba(model, Ws.loc[b, feats])
        T = fit_temperature(P_cal, np.array([CLASSES.index(v) for v in Ws.loc[b, 'window_label']]))
        S_cal = flight_scores(Ws.loc[b].reset_index(drop=True), apply_temperature(P_cal, T))
        th10, th20 = conformal_thresholds(S_cal, 0.10), conformal_thresholds(S_cal, 0.20)
        Pw = apply_temperature(predict_proba(model, Ww[feats]), T)
        Ww_out = Ww[['flight_id', 'family', 't_start', 't_end']].copy()
        for i, c in enumerate(CLASSES):
            Ww_out[f'p_{c}'] = Pw[:, i]
        Ww_out['pred'] = [CLASSES[i] for i in Pw.argmax(1)]
        Ww_out.to_csv(out / 'e3_whelan_window_timeline.csv', index=False)
        Sw = flight_scores(Ww, Pw)
        Sw['set_a0.10'] = [','.join(s) for s in conformal_sets(Sw, th10)]
        Sw['set_a0.20'] = [','.join(s) for s in conformal_sets(Sw, th20)]
        Sw.to_csv(out / 'e3_whelan_flight_verdicts.csv', index=False)
        nanfrac = Ww[feats].isna().mean().sort_values(ascending=False)
        print('\n[E3] Whelan feature availability (NaN fraction, worst 8):\n', nanfrac.head(8).round(2).to_string())
        print('\n[E3] Whelan window predicted-class fractions per flight:\n',
              Ww_out.groupby('flight_id')['pred'].value_counts(normalize=True).round(2).to_string())
        print('\n[E3] Whelan flight verdicts with conformal sets:\n',
              Sw[['flight_id', 'family', 'pred', 'conf', 'set_a0.10', 'set_a0.20']].round(3).to_string(index=False))
    print('\nresults written to', out)


if __name__ == '__main__':
    main()
