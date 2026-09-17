"""
Threshold-free separability diagnostics for the distributional abstention gate.

The E4 table reports the fraction of flights withheld at one operating point
(the 95th percentile of calibration scores). That single number cannot separate
two different situations: the score does not distinguish the held-out subtype
from seen-subtype flights at all, or it does distinguish them but the threshold
sits in the wrong place. These helpers answer that directly.

For each gate score and each hold-out, the module reports:

  auroc            separability of held-out flights from seen-subtype test
                   flights, 0.5 meaning no separation
  auroc_lo/hi      Hanley-McNeil 95% interval (approximate; the sample sizes
                   here are 11-32 per side, so the intervals are wide)
  mw_p             two-sided Mann-Whitney p-value
  tpr_at_fpr05     fraction of held-out flights caught at the threshold that
                   withholds 5% of seen-subtype test flights, i.e. the best
                   available operating point at a fixed false-abstention budget
  withheld_unseen  fraction withheld at the 95th percentile of calibration
  withheld_seen    scores, for reconciliation with the existing E4 table

Three scores are computed on the same flight-level aggregates the verdict uses:

  mahalanobis      squared Mahalanobis distance, Ledoit-Wolf covariance
  iforest          negated isolation-forest score_samples
  one_minus_conf   1 minus the stacked model's max class probability, included
                   as the free baseline a reviewer will ask about

A PCA variant is included as a second diagnostic. With roughly 40-60 training
flights and 120 aggregate features the sample covariance is rank deficient, so
a low AUROC in the full feature space may reflect a poor covariance estimate
rather than an absence of signal. If AUROC rises substantially under PCA the
limitation is estimation; if it stays near 0.5 the aggregates do not carry the
signal and the gate cannot be rescued by a different threshold.

All scores are oriented so that larger means more novel.

Reading the output for one hold-out
-----------------------------------
  auroc near 0.5, interval spanning 0.5, tpr_at_fpr low, best_feature_perm_p
  not small
      The aggregates do not separate this subtype. No threshold rescues the
      gate. Report the miss as a property of the feature space.

  auroc high but withheld_unseen_q95 low
      The score separates and the 95th-percentile cut is in the wrong place.
      The current E4 number understates the gate. Re-state the operating point.

  auroc near 0.5 but best_feature_auroc well above best_feature_null_q95 with a
  small permutation p
      Individual aggregates carry the subtype but the 120-dimensional distance
      buries it. The limitation is the covariance estimate, not the information.
      Compare against the PCA row before concluding anything.

  withheld_seen_q95 high wherever the gate fires
      The catch is being bought with false abstention on in-distribution
      flights. Quote both columns together or the result is not honest.

Sample sizes here are 11-32 flights per side and the affected family's
calibration set is about 10 flights, so the 95th percentile is close to the
maximum calibration score. Both facts belong in the table caption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

__all__ = [
    "auroc_with_ci",
    "tpr_at_fixed_fpr",
    "gate_separability",
    "per_feature_auroc",
    "best_feature_null",
    "summarise_seeds",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def auroc_with_ci(pos_scores, neg_scores, alpha=0.05):
    """AUROC of pos vs neg with a Hanley-McNeil interval and Mann-Whitney p.

    pos_scores are the held-out (novel) flights, neg_scores the seen-subtype
    test flights. Ties contribute 0.5, via mid-rank U. Returns NaNs when either
    side is empty.

    The Hanley-McNeil variance assumes exponential score distributions and is
    approximate at these sample sizes. It is reported so the table carries an
    interval rather than a bare point estimate, not as an exact interval.
    """
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    n_pos, n_neg = pos.size, neg.size

    out = {
        "auroc": np.nan,
        "auroc_lo": np.nan,
        "auroc_hi": np.nan,
        "mw_p": np.nan,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
    if n_pos == 0 or n_neg == 0:
        return out

    # mid-rank U for pos, so ties count as 0.5
    u_stat, _ = mannwhitneyu(pos, neg, alternative="greater")
    auroc = float(u_stat) / (n_pos * n_neg)
    auroc = min(max(auroc, 0.0), 1.0)

    try:
        _, p_two = mannwhitneyu(pos, neg, alternative="two-sided")
        p_two = float(p_two)
    except ValueError:
        p_two = np.nan

    q1 = auroc / (2.0 - auroc) if auroc < 1.0 else 1.0
    q2 = (2.0 * auroc**2) / (1.0 + auroc)
    var = (
        auroc * (1.0 - auroc)
        + (n_pos - 1) * (q1 - auroc**2)
        + (n_neg - 1) * (q2 - auroc**2)
    ) / (n_pos * n_neg)
    se = float(np.sqrt(max(var, 0.0)))
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else float(
        abs(np.sqrt(2) * _erfinv(1 - alpha))
    )

    out.update(
        auroc=auroc,
        auroc_lo=min(max(auroc - z * se, 0.0), 1.0),
        auroc_hi=min(max(auroc + z * se, 0.0), 1.0),
        mw_p=p_two,
    )
    return out


def _erfinv(y):
    from scipy.special import erfinv

    return erfinv(y)


def tpr_at_fixed_fpr(pos_scores, neg_scores, fpr=0.05):
    """Fraction of pos caught at the smallest threshold with FPR <= `fpr` on neg.

    The threshold is taken from the seen-subtype test scores themselves, so this
    is the most favourable operating point available at that false-abstention
    budget. It answers whether a different cut would have rescued the gate.
    """
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return {"tpr_at_fpr": np.nan, "threshold_at_fpr": np.nan, "realised_fpr": np.nan}

    # candidate thresholds: just above each neg score, plus one above the max
    candidates = np.unique(np.concatenate([neg, [neg.max() + 1.0]]))
    best = None
    for thr in candidates:
        realised_fpr = float(np.mean(neg > thr))
        if realised_fpr <= fpr + _EPS:
            tpr = float(np.mean(pos > thr))
            if best is None or tpr > best["tpr_at_fpr"]:
                best = {
                    "tpr_at_fpr": tpr,
                    "threshold_at_fpr": float(thr),
                    "realised_fpr": realised_fpr,
                }
    if best is None:
        best = {"tpr_at_fpr": 0.0, "threshold_at_fpr": np.nan, "realised_fpr": np.nan}
    return best


# --------------------------------------------------------------------------
# scorers
# --------------------------------------------------------------------------
def _standardise(train, blocks):
    """Drop zero-variance training columns, then z-score every block on train."""
    train = np.asarray(train, dtype=float)
    keep = np.std(train, axis=0) > _EPS
    if keep.sum() == 0:
        raise ValueError("every training feature is constant; nothing to score on")
    mu = train[:, keep].mean(axis=0)
    sd = train[:, keep].std(axis=0)
    sd = np.where(sd > _EPS, sd, 1.0)

    def _t(x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x.reshape(0, int(keep.sum()))
        return (x[:, keep] - mu) / sd

    return _t(train), [_t(b) for b in blocks], int(keep.sum())


def _mahalanobis_scores(train_z, blocks_z, pca_k=None, seed=0):
    """Squared Mahalanobis distance from the training distribution.

    With pca_k set, the distance is taken in the leading PCA subspace fitted on
    the training flights, capped at n_train - 1 components.
    """
    if pca_k is not None:
        n_comp = int(min(pca_k, train_z.shape[0] - 1, train_z.shape[1]))
        if n_comp < 1:
            return None
        pca = PCA(n_components=n_comp, random_state=seed).fit(train_z)
        train_z = pca.transform(train_z)
        blocks_z = [pca.transform(b) if b.size else b.reshape(0, n_comp) for b in blocks_z]

    lw = LedoitWolf(store_precision=True, assume_centered=False).fit(train_z)
    return [lw.mahalanobis(b) if b.size else np.array([]) for b in blocks_z]


def per_feature_auroc(unseen_X, seen_X, feature_names=None, top=5):
    """Univariate separability of each aggregate feature, unseen vs seen test.

    This is the check the multivariate scores cannot give. Mahalanobis in 120
    dimensions fitted on 40 flights can drown a signal that sits in two or three
    features. If no single feature separates and neither multivariate score
    separates, the aggregates genuinely do not carry the subtype; if a feature
    reaches AUROC 0.9 while Mahalanobis sits at 0.5, the covariance estimate is
    the limitation.

    Direction is ignored: the reported value is max(a, 1 - a), so a feature that
    is systematically lower on the held-out subtype scores just as high.
    """
    unseen = np.asarray(unseen_X, dtype=float)
    seen = np.asarray(seen_X, dtype=float)
    if unseen.size == 0 or seen.size == 0:
        return pd.DataFrame(columns=["feature", "auroc_abs", "auroc_raw", "mw_p"])
    n_feat = unseen.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_feat)]

    recs = []
    for j in range(n_feat):
        a = unseen[:, j]
        b = seen[:, j]
        if not (np.isfinite(a).any() and np.isfinite(b).any()):
            continue
        if np.nanstd(np.concatenate([a, b])) <= _EPS:
            continue
        res = auroc_with_ci(a, b)
        if not np.isfinite(res["auroc"]):
            continue
        recs.append(
            {
                "feature": feature_names[j],
                "auroc_abs": max(res["auroc"], 1.0 - res["auroc"]),
                "auroc_raw": res["auroc"],
                "mw_p": res["mw_p"],
            }
        )
    out = pd.DataFrame(recs).sort_values("auroc_abs", ascending=False)
    return out.head(top) if top else out


def best_feature_null(unseen_X, seen_X, n_perm=200, seed=0):
    """Permutation null for the best-single-feature AUROC.

    Scanning 120 aggregates over ~35 flights, the largest per-feature AUROC runs
    around 0.8 under pure noise, so the raw maximum cannot be read as evidence.
    This shuffles the unseen/seen labels `n_perm` times, records the maximum
    per-feature AUROC each time, and returns the 95th percentile of that null
    plus an empirical one-sided p-value for the observed maximum.

    Read the observed maximum against `null_q95`, never against 0.5.
    """
    unseen = np.asarray(unseen_X, dtype=float)
    seen = np.asarray(seen_X, dtype=float)
    if unseen.size == 0 or seen.size == 0:
        return {"best_feature_null_q95": np.nan, "best_feature_perm_p": np.nan}

    pooled = np.vstack([unseen, seen])
    n_pos = unseen.shape[0]
    keep = np.nanstd(pooled, axis=0) > _EPS
    pooled = pooled[:, keep]
    if pooled.shape[1] == 0:
        return {"best_feature_null_q95": np.nan, "best_feature_perm_p": np.nan}

    def _max_auc(mat, idx_pos):
        mask = np.zeros(mat.shape[0], dtype=bool)
        mask[idx_pos] = True
        # rank-based AUROC for every column at once, ties at mid-rank
        ranks = np.apply_along_axis(_midrank, 0, mat)
        r_pos = ranks[mask].sum(axis=0)
        n_neg = mat.shape[0] - n_pos
        auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(np.nanmax(np.maximum(auc, 1.0 - auc)))

    observed = _max_auc(pooled, np.arange(n_pos))

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        idx = rng.permutation(pooled.shape[0])[:n_pos]
        null[b] = _max_auc(pooled, idx)

    return {
        "best_feature_auroc_observed": observed,
        "best_feature_null_q95": float(np.percentile(null, 95)),
        "best_feature_perm_p": float((1 + np.sum(null >= observed)) / (n_perm + 1)),
    }


def _midrank(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < xs.size:
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def _iforest_scores(train_z, blocks_z, seed=0):
    """Negated isolation-forest score_samples, so larger means more novel."""
    n_train = train_z.shape[0]
    if n_train < 4:
        return None
    iso = IsolationForest(
        n_estimators=300,
        max_samples=min(256, n_train),
        random_state=seed,
        n_jobs=1,
    ).fit(train_z)
    return [-iso.score_samples(b) if b.size else np.array([]) for b in blocks_z]


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def gate_separability(
    train_X,
    calib_X,
    unseen_X,
    seen_X,
    held_out_subtype,
    held_out_family=None,
    seed=0,
    conf_unseen=None,
    conf_seen=None,
    conf_calib=None,
    pca_k=10,
    q=95.0,
    fpr_budget=0.05,
    feature_names=None,
    n_perm=200,
):
    """Separability of every gate score for one hold-out and one seed.

    Parameters
    ----------
    train_X, calib_X, unseen_X, seen_X
        Flight-level aggregate matrices for this hold-out and seed: the training
        flights the gate is fitted on, the calibration flights the 95th
        percentile comes from, the withheld-subtype flights, and the disjoint
        seen-subtype test flights. Same columns, same order, in the same units
        the verdict uses. Pass them exactly as the existing E4 code does.
    conf_unseen, conf_seen, conf_calib
        Optional max predicted class probability per flight, from the stacked
        model in the same fit. Supplying them adds the one_minus_conf baseline.
    pca_k
        Components for the PCA sensitivity. None disables it.
    feature_names
        Column names of the aggregate matrices, used for the best-single-feature
        diagnostic. Pass the same list the model uses.
    q
        Percentile of calibration scores used for the reported operating point.
        Keep at 95.0 to reconcile with the existing table.

    Returns a list of one row-dict per score.
    """
    train_z, (calib_z, unseen_z, seen_z), n_feat = _standardise(
        train_X, [calib_X, unseen_X, seen_X]
    )

    scorers = {}

    mah = _mahalanobis_scores(train_z, [calib_z, unseen_z, seen_z], seed=seed)
    if mah is not None:
        scorers["mahalanobis"] = mah

    if pca_k is not None:
        mah_pca = _mahalanobis_scores(
            train_z, [calib_z, unseen_z, seen_z], pca_k=pca_k, seed=seed
        )
        if mah_pca is not None:
            scorers[f"mahalanobis_pca{int(min(pca_k, train_z.shape[0] - 1, n_feat))}"] = mah_pca

    iso = _iforest_scores(train_z, [calib_z, unseen_z, seen_z], seed=seed)
    if iso is not None:
        scorers["iforest"] = iso

    if conf_unseen is not None and conf_seen is not None:
        c_cal = (
            1.0 - np.asarray(conf_calib, dtype=float)
            if conf_calib is not None
            else np.array([])
        )
        scorers["one_minus_conf"] = [
            c_cal,
            1.0 - np.asarray(conf_unseen, dtype=float),
            1.0 - np.asarray(conf_seen, dtype=float),
        ]

    # best single aggregate feature, as a floor on what any score could achieve
    feat_tbl = per_feature_auroc(
        np.asarray(unseen_X, dtype=float),
        np.asarray(seen_X, dtype=float),
        feature_names=feature_names,
        top=1,
    )
    if len(feat_tbl):
        best_feat = str(feat_tbl.iloc[0]["feature"])
        best_feat_auroc = float(feat_tbl.iloc[0]["auroc_abs"])
        null = best_feature_null(unseen_X, seen_X, n_perm=n_perm, seed=seed)
    else:
        best_feat, best_feat_auroc = None, np.nan
        null = {"best_feature_null_q95": np.nan, "best_feature_perm_p": np.nan}

    rows = []
    for name, (s_cal, s_unseen, s_seen) in scorers.items():
        row = {
            "held_out_family": held_out_family,
            "held_out_subtype": held_out_subtype,
            "seed": seed,
            "score": name,
            "n_train": int(train_z.shape[0]),
            "n_features": n_feat,
            "n_calib": int(np.asarray(s_cal).size),
            "best_feature": best_feat,
            "best_feature_auroc": best_feat_auroc,
            "best_feature_null_q95": null.get("best_feature_null_q95", np.nan),
            "best_feature_perm_p": null.get("best_feature_perm_p", np.nan),
        }
        row.update(auroc_with_ci(s_unseen, s_seen))
        row.update(tpr_at_fixed_fpr(s_unseen, s_seen, fpr=fpr_budget))

        # operating point currently reported in the E4 table
        s_cal = np.asarray(s_cal, dtype=float)
        s_cal = s_cal[np.isfinite(s_cal)]
        su = np.asarray(s_unseen, dtype=float)
        ss = np.asarray(s_seen, dtype=float)
        if s_cal.size and su.size and ss.size:
            thr = float(np.percentile(s_cal, q))
            row["threshold_q95"] = thr
            row["withheld_unseen_q95"] = float(np.mean(su > thr))
            row["withheld_seen_q95"] = float(np.mean(ss > thr))
        else:
            row["threshold_q95"] = np.nan
            row["withheld_unseen_q95"] = np.nan
            row["withheld_seen_q95"] = np.nan

        rows.append(row)

    return rows


def summarise_seeds(rows, by=("held_out_family", "held_out_subtype", "score")):
    """Mean and std across seeds, matching how the E2 tables are reported.

    AUROC is averaged across per-seed values rather than pooled, because each
    seed fits a different pipeline and the raw score scales are not comparable.
    """
    # rows may already be a DataFrame; list() on one yields column names, not rows
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if df.empty:
        return df
    by = [c for c in by if c in df.columns and df[c].notna().any()]
    value_cols = [
        "auroc",
        "auroc_lo",
        "auroc_hi",
        "mw_p",
        "tpr_at_fpr",
        "realised_fpr",
        "withheld_unseen_q95",
        "withheld_seen_q95",
        "best_feature_auroc",
        "best_feature_null_q95",
        "best_feature_perm_p",
        "n_train",
        "n_features",
        "n_pos",
        "n_neg",
    ]
    value_cols = [c for c in value_cols if c in df.columns]
    agg = df.groupby(by, dropna=False)[value_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    return agg.reset_index()
