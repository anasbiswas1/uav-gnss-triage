#!/usr/bin/env python3
"""
sih_report.py  --  manuscript tables and figures from a results folder written by sih_model.py.

Reads only committed result tables (reports/<run>/), so every number in the manuscript traces to a file in the
repository. Two optional inputs read raw data: --manifests (dataset composition table) and --whelan_dir (receiver
self-report channel table for the three live logs).

Usage:
  python sih_report.py --reports reports/v3 --figures figures/v3 \
      [--manifests data/sih/sih_flights_v2 data/sih/sih_flights_v2_nofix] [--whelan_dir data/features/features_v3/whelan_live]
Outputs: <reports>/tables/*.csv, <reports>/tables/tables.md, <figures>/*.png and *.pdf
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

CLASSES = ['nominal', 'spoof', 'gps_degrade', 'sensor_fault']
SUBTYPE_LABEL = {'jump': 'Jump', 'drift_incoherent': 'Incoherent drift', 'drift_coherent': 'Coherent drift', 'off_then_ok': 'Receiver silence',
                 'stuck_then_ok': 'Frozen receiver output', 'no_fix_then_ok': 'No-fix reporting', 'baro_stuck': 'Frozen barometer', 'mag_stuck': 'Frozen magnetometer'}
COARSE = ['nominal', 'gnss_chain', 'non_gnss_fault']
LABEL = {'nominal': 'Nominal', 'spoof': 'Spoof', 'gps_degrade': 'GNSS degradation', 'sensor_fault': 'Sensor fault',
         'gnss_chain': 'GNSS-chain inconsistency', 'non_gnss_fault': 'Non-GNSS fault',
         'rule': 'Percentile rule', 'stacked': 'Stacked', 'xgb': 'XGBoost', 'logreg': 'Logistic'}


def ms(x, digits=3):
    return f'{np.mean(x):.{digits}f} ± {np.std(x, ddof=0):.{digits}f}'


def md_table(df, floatfmt='.3f'):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(str(c) for c in cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f'{v:{floatfmt}}' if isinstance(v, (float, np.floating)) and not pd.isna(v) else str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reports', required=True)
    ap.add_argument('--figures', required=True)
    ap.add_argument('--manifests', nargs='*', default=[])
    ap.add_argument('--whelan_dir', default=None)
    args = ap.parse_args()
    R, FIG = pathlib.Path(args.reports), pathlib.Path(args.figures)
    T = R / 'tables'; T.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    md = []

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 120, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans'})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(FIG / f'{name}.png', bbox_inches='tight'); fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight'); plt.close(fig)

    # ---------------------------------------------------------------- T1 dataset composition
    if args.manifests:
        rows = []
        for m in args.manifests:
            for line in open(pathlib.Path(m) / 'manifest.jsonl'):
                if line.strip():
                    r = json.loads(line)
                    if r.get('ok'):
                        rows.append({'run': pathlib.Path(m).name, 'family': r['family'], 'subtype': r['subtype'],
                                     'duration_s': r['verify'].get('duration_s'), 'firmware': r.get('firmware_commit', '')[:10],
                                     'onset_logged': r['verify'].get('onset_sim_s') is not None})
        D = pd.DataFrame(rows)
        t1 = D.groupby(['family', 'subtype']).agg(flights=('duration_s', 'size'), median_duration_s=('duration_s', 'median'),
                                                   min_duration_s=('duration_s', 'min'), max_duration_s=('duration_s', 'max')).reset_index()
        t1['family'] = t1['family'].map(LABEL)
        t1.to_csv(T / 't1_dataset_composition.csv', index=False)
        md.append('## Table 1. Generated dataset composition\n\n' + md_table(t1, '.0f') +
                  f'\n\nTotal flights: {len(D)}; firmware commits: {sorted(D.firmware.unique())}; '
                  f'onset logged for {int(D[D.family != "nominal"].onset_logged.sum())} of {int((D.family != "nominal").sum())} attacked flights.')

    # ---------------------------------------------------------------- T2 leakage audit
    e0 = pd.read_csv(R / 'e0_leakage_audit.csv')
    e0['tag'] = e0['tag'].map({'random_window_split': 'Random-window split', 'flight_grouped_split': 'Flight-grouped split'})
    g = e0.groupby('tag')
    t2 = pd.DataFrame({'Split': list(g.groups), 'Training windows': [int(v['n_train_windows'].mean()) for _, v in g],
                       'Accuracy': [ms(v['acc']) for _, v in g], 'Macro-F1': [ms(v['macro_f1']) for _, v in g],
                       'ECE': [ms(v['ece']) for _, v in g], 'NLL': [ms(v['nll']) for _, v in g], 'Brier': [ms(v['brier']) for _, v in g]})
    piv = e0.pivot(index='rep', columns='tag', values='acc'); d = piv['Random-window split'] - piv['Flight-grouped split']
    t2.to_csv(T / 't2_leakage_audit.csv', index=False)
    md.append('## Table 2. Leakage audit: same detector, matched training-window counts, paired over repetitions\n\n' + md_table(t2) +
              f'\n\nPaired accuracy difference (random-window minus flight-grouped): {d.mean():.3f} ± {d.std(ddof=0):.3f} over {len(d)} repetitions.')
    e0m = e0.groupby('tag')[['acc', 'macro_f1', 'ece']].mean().reset_index()
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(3); w = 0.36
    for i, (_, r) in enumerate(e0m.iterrows()):
        ax.bar(x + (i - 0.5) * w, [r['acc'], r['macro_f1'], r['ece']], w, label=r['tag'])
    ax.set_xticks(x); ax.set_xticklabels(['Accuracy', 'Macro-F1', 'ECE']); ax.set_ylim(0, 1); ax.legend(frameon=False)
    ax.set_title('Window-level metrics by split protocol')
    save(fig, 'f_leakage_audit')

    # ---------------------------------------------------------------- T3 window metrics
    wm = pd.read_csv(R / 'e1_window_metrics.csv')
    g = wm.groupby(['model', 'tag'])
    t3 = pd.DataFrame({'Model': [LABEL[m] for m, _ in g.groups], 'Probabilities': [t for _, t in g.groups],
                       'Accuracy': [ms(v['acc']) for _, v in g], 'Macro-F1': [ms(v['macro_f1']) for _, v in g],
                       'ECE': [ms(v['ece']) for _, v in g], 'NLL': [ms(v['nll']) for _, v in g], 'Brier': [ms(v['brier']) for _, v in g]})
    t3['Probabilities'] = t3['Probabilities'].map({'raw': 'raw', 'temp_scaled': 'temperature-scaled'})
    t3.to_csv(T / 't3_window_metrics.csv', index=False)
    md.append('## Table 3. Window-level detector, mean ± std over 5 flight-grouped repetitions\n\n' + md_table(t3))

    # ---------------------------------------------------------------- T4 flight-level
    fm = pd.read_csv(R / 'e1_flight_metrics.csv')
    order = [a for a in ('rule', 'stacked_probs_only', 'stacked_raw_only', 'stacked', 'stacked_no_gap') if a in set(fm.aggregation)]
    LABEL.update({'stacked_probs_only': 'Stacked, probability summaries only', 'stacked_raw_only': 'Stacked, raw flight features only',
                  'stacked': 'Stacked, full', 'stacked_no_gap': 'Stacked, full, without derived-topic gap features'})
    rows = []
    for a in order:
        v = fm[fm.aggregation == a]
        rows.append({'Aggregation': LABEL[a], 'Flight accuracy': ms(v['flight_acc']), 'Flight macro-F1': ms(v['flight_macro_f1']), 'AURC': ms(v['aurc']),
                     'Coarse accuracy': ms(v['coarse_acc']), 'Coarse macro-F1': ms(v['coarse_macro_f1'])})
    t4 = pd.DataFrame(rows); t4.to_csv(T / 't4_flight_level.csv', index=False)
    md.append('## Table 4. Flight-level verdicts, mean ± std over 5 repetitions (20 train / 20 calibration / 20 test flights per family)\n\n' + md_table(t4))
    rows = []
    for a in order:
        v = fm[fm.aggregation == a]
        rows.append({'Aggregation': LABEL[a], 'Fine ECE (10 bins)': ms(v['flight_ece10']), 'Fine NLL': ms(v['flight_nll']), 'Fine Brier': ms(v['flight_brier']),
                     'Coarse ECE (10 bins)': ms(v['coarse_ece10']), 'Coarse NLL': ms(v['coarse_nll']), 'Coarse Brier': ms(v['coarse_brier'])})
    t4b = pd.DataFrame(rows); t4b.to_csv(T / 't4b_flight_calibration.csv', index=False)
    md.append('## Table 4b. Flight-level probability calibration of the verdict probabilities (test flights), mean ± std over 5 repetitions\n\n' + md_table(t4b))
    # reliability diagram from pooled test flights (all repetitions), full stacked and rule
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for ax, a in zip(axes, ('rule', 'stacked')):
        files = sorted(R.glob(f'e1_rep*_{a}_flight_scores.csv'))
        if not files:
            continue
        S = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        conf = S['conf'].to_numpy(); correct = (S['pred'] == S['family']).to_numpy().astype(float)
        edges = np.linspace(0, 1, 11); mids, accs, cnts = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum():
                mids.append(conf[m].mean()); accs.append(correct[m].mean()); cnts.append(int(m.sum()))
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8); ax.plot(mids, accs, 'o-', ms=4)
        for x_, y_, n_ in zip(mids, accs, cnts):
            ax.annotate(str(n_), (x_, y_), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=6)
        ax.set_title(LABEL[a]); ax.set_xlabel('Flight confidence'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel('Fraction correct')
    save(fig, 'f_flight_reliability')

    # ---------------------------------------------------------------- T5 conformal
    cf = pd.read_csv(R / 'e1_conformal.csv')
    cols5 = ['coverage', 'avg_set_size', 'multi_label_rate', 'empty_rate', 'non_singleton_rate', 'verdict_outside_set_rate', 'operational_abstain_rate']
    gm = cf.groupby(['aggregation', 'level', 'alpha', 'family'])[cols5]
    agg = gm.mean().reset_index(); sd = gm.std(ddof=0).reset_index()
    t5 = agg[['aggregation', 'level', 'alpha', 'family']].copy()
    for c in cols5:
        t5[c] = [f'{m_:.3f} ± {s_:.3f}' for m_, s_ in zip(agg[c], sd[c])]
    t5['aggregation'] = t5['aggregation'].map(LABEL); t5['family'] = t5['family'].map(LABEL)
    t5 = t5.rename(columns={'aggregation': 'Aggregation', 'level': 'Level', 'family': 'Class', 'coverage': 'Coverage', 'avg_set_size': 'Mean set size',
                            'multi_label_rate': 'Multi-label', 'empty_rate': 'Empty', 'non_singleton_rate': 'Non-singleton', 'verdict_outside_set_rate': 'Verdict outside set',
                            'operational_abstain_rate': 'Operational abstention'})
    t5.to_csv(T / 't5_conformal.csv', index=False)
    md.append('## Table 5. Class-conditional conformal prediction sets at the flight level, mean ± std over 5 repetitions (balanced test sets)\n\n' + md_table(t5))
    agg['aggregation'] = agg['aggregation'].map(LABEL); agg['family'] = agg['family'].map(LABEL)
    sub = agg[(agg.aggregation == LABEL['stacked']) & (agg.alpha == 0.1)]
    sub = sub.assign(abstain_rate=sub['operational_abstain_rate'])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, gridspec_kw={'wspace': 0.12, 'width_ratios': [4, 3]})
    wrap = {'GNSS degradation': 'GNSS\ndegradation', 'GNSS-chain inconsistency': 'GNSS-chain\ninconsistency', 'Non-GNSS fault': 'Non-GNSS\nfault', 'Sensor fault': 'Sensor\nfault'}
    for ax, lvl, classes in ((axes[0], 'fine', CLASSES), (axes[1], 'coarse', COARSE)):
        d = sub[sub.level == lvl].set_index('family').reindex([LABEL[c] for c in classes])
        x = np.arange(len(d)); w = 0.38
        ax.bar(x - w / 2, d['coverage'], w, label='Coverage'); ax.bar(x + w / 2, d['abstain_rate'], w, label='Operational abstention')
        ax.axhline(0.9, ls='--', lw=0.8, color='k', label='Target coverage 0.90'); ax.set_xticks(x)
        ax.set_xticklabels([wrap.get(n, n) for n in d.index], fontsize=8)
        ax.set_title(f'{lvl.capitalize()} level, alpha = 0.10'); ax.set_ylim(0, 1.05)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.04))
    save(fig, 'f_conformal_stacked')

    if (R / 'e1_extra_flights.csv').exists():
        ex = pd.read_csv(R / 'e1_extra_flights.csv')
        exm = ex.groupby(['aggregation', 'level', 'alpha', 'family'])[['n', 'acc', 'coverage', 'operational_abstain_rate']].mean().reset_index()
        exm['aggregation'] = exm['aggregation'].map(LABEL); exm['family'] = exm['family'].map(LABEL)
        t5b = exm.rename(columns={'aggregation': 'Aggregation', 'level': 'Level', 'family': 'Class', 'n': 'Flights per rep', 'acc': 'Accuracy',
                                  'coverage': 'Coverage', 'operational_abstain_rate': 'Operational abstention'})
        t5b.to_csv(T / 't5b_extra_flights.csv', index=False)
        md.append('## Table 5b. Flights beyond the balanced test set (GNSS degradation), scored with the same thresholds, mean over 5 repetitions\n\n' + md_table(t5b))

    # ---------------------------------------------------------------- F risk-coverage
    rc = pd.read_csv(R / 'e1_risk_coverage.csv')
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    grid = np.linspace(0.05, 1.0, 96)
    for a in ('rule', 'stacked'):
        if a not in set(rc.aggregation):
            continue
        curves = []
        for rep, d in rc[rc.aggregation == a].groupby('rep'):
            d = d.sort_values('coverage')
            curves.append(np.interp(grid, d['coverage'], d['risk']))
        c = np.array(curves)
        ax.plot(grid, c.mean(0), label=LABEL[a]); ax.fill_between(grid, c.min(0), c.max(0), alpha=0.2)
    ax.set_xlabel('Coverage (fraction of flights given a verdict)'); ax.set_ylabel('Selective risk (error rate)')
    ax.set_title('Risk-coverage, flight level'); ax.legend(frameon=False)
    save(fig, 'f_risk_coverage')

    # ---------------------------------------------------------------- T6 leave-one-subtype-out (single-fit design)
    lo = pd.read_csv(R / 'e2_leave_one_subtype_out.csv')
    ps = pd.read_csv(R / 'e2_per_seed.csv') if (R / 'e2_per_seed.csv').exists() else None
    n_seeds = int(ps['seed'].nunique()) if ps is not None else 1
    pm = lambda a, b: [f'{m_:.3f} ± {s_:.3f}' for m_, s_ in zip(lo[a], lo[b])]
    t6 = pd.DataFrame({'Held-out family': lo['held_out_family'].map(LABEL), 'Held-out subtype': lo['held_out_subtype'].map(lambda v: SUBTYPE_LABEL.get(v, v)),
                       'n unseen': lo['unseen_n_mean'].round(0).astype(int), 'n seen test': lo['seen_n_mean'].round(0).astype(int),
                       'Fine accuracy, unseen': pm('unseen_acc_mean', 'unseen_acc_std'), 'Fine accuracy, seen': pm('seen_acc_mean', 'seen_acc_std'),
                       'Coarse accuracy, unseen': pm('unseen_coarse_acc_mean', 'unseen_coarse_acc_std'), 'Coarse accuracy, seen': pm('seen_coarse_acc_mean', 'seen_coarse_acc_std'),
                       'Mean confidence, unseen': pm('unseen_mean_conf_mean', 'unseen_mean_conf_std'), 'Mean confidence, seen': pm('seen_mean_conf_mean', 'seen_mean_conf_std')})
    t6.to_csv(T / 't6_leave_one_subtype_out.csv', index=False)
    md.append(f'## Table 6. Unseen-subtype evaluation: one fitted pipeline per hold-out (single training and calibration set) evaluated on the withheld subtype and on a '
              f'disjoint seen-subtype test set from the same family, mean ± std over {n_seeds} seeds\n\n' + md_table(t6))
    t6b = pd.DataFrame({'Held-out subtype': lo['held_out_subtype'].map(lambda v: SUBTYPE_LABEL.get(v, v)),
                        'Fine coverage (0.10), unseen': pm('unseen_coverage_a0.1_mean', 'unseen_coverage_a0.1_std'), 'Fine coverage (0.10), seen': pm('seen_coverage_a0.1_mean', 'seen_coverage_a0.1_std'),
                        'Coarse coverage (0.10), unseen': pm('unseen_coarse_coverage_a0.1_mean', 'unseen_coarse_coverage_a0.1_std'), 'Coarse coverage (0.10), seen': pm('seen_coarse_coverage_a0.1_mean', 'seen_coarse_coverage_a0.1_std'),
                        'Operational abstention (0.10), unseen': pm('unseen_operational_abstain_a0.1_mean', 'unseen_operational_abstain_a0.1_std'), 'Operational abstention (0.10), seen': pm('seen_operational_abstain_a0.1_mean', 'seen_operational_abstain_a0.1_std'),
                        'Predicted as (unseen, seed 0)': lo['unseen_pred_dist_seed0']})
    t6b.to_csv(T / 't6b_unseen_vs_seen_coverage.csv', index=False)
    md.append('## Table 6b. Same fitted pipelines: conformal coverage and operational abstention on the withheld subtype and on the seen-subtype test set\n\n' + md_table(t6b))
    if (R / 'e2_splits.csv').exists():
        sp = pd.read_csv(R / 'e2_splits.csv'); sp = sp[sp.seed == 0].drop(columns=['seed']); sp['family'] = sp['family'].map(LABEL)
        sp['held_out_subtype'] = sp['held_out_subtype'].map(lambda v: SUBTYPE_LABEL.get(v, v))
        t6c = sp.rename(columns={'held_out_subtype': 'Held-out subtype', 'family': 'Family', 'n_train': 'Train', 'n_cal': 'Calibration', 'n_seen_test': 'Seen test',
                                 'rank_alpha0.1': 'Rank (0.10)', 'rank_alpha0.2': 'Rank (0.20)', 'rank_at_max_alpha0.1': 'Threshold at max score (0.10)'})
        t6c.to_csv(T / 't6c_holdout_allocation.csv', index=False)
        md.append('## Table 6c. Hold-out allocation per family (identical across seeds), test-set sizes, and the conformal order-statistic ranks; where the rank equals the '
                  'calibration size the threshold is the largest calibration score\n\n' + md_table(t6c, '.0f'))
    if 'unseen_gate_mahal_withheld_mean' in lo.columns:
        t6d = pd.DataFrame({'Held-out subtype': lo['held_out_subtype'].map(lambda v: SUBTYPE_LABEL.get(v, v)),
                            'Mahalanobis gate, unseen': pm('unseen_gate_mahal_withheld_mean', 'unseen_gate_mahal_withheld_std'),
                            'Mahalanobis gate, seen': pm('seen_gate_mahal_withheld_mean', 'seen_gate_mahal_withheld_std'),
                            'Isolation-forest gate, unseen': pm('unseen_gate_iso_withheld_mean', 'unseen_gate_iso_withheld_std'),
                            'Isolation-forest gate, seen': pm('seen_gate_iso_withheld_mean', 'seen_gate_iso_withheld_std'),
                            'Conformal policy alone, unseen': pm('unseen_operational_abstain_a0.1_mean', 'unseen_operational_abstain_a0.1_std'),
                            'Conformal or Mahalanobis gate, unseen': pm('unseen_gate_mahal_or_conformal_withheld_mean', 'unseen_gate_mahal_or_conformal_withheld_std'),
                            'Conformal or Mahalanobis gate, seen': pm('seen_gate_mahal_or_conformal_withheld_mean', 'seen_gate_mahal_or_conformal_withheld_std')})
        t6d.to_csv(T / 't6d_distributional_gate.csv', index=False)
        md.append('## Table 6d. Distributional abstention gate: fraction of flights withheld on the withheld subtype and on the seen-subtype test set (thresholds at the '
                  '95th percentile of the calibration flights), and the combined policy\n\n' + md_table(t6d))
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        x = np.arange(len(lo)); w = 0.26
        ax.bar(x - w, lo['unseen_operational_abstain_a0.1_mean'], w, yerr=lo['unseen_operational_abstain_a0.1_std'], label='Conformal policy alone', capsize=2)
        ax.bar(x, lo['unseen_gate_mahal_withheld_mean'], w, yerr=lo['unseen_gate_mahal_withheld_std'], label='Mahalanobis gate', capsize=2)
        ax.bar(x + w, lo['unseen_gate_mahal_or_conformal_withheld_mean'], w, yerr=lo['unseen_gate_mahal_or_conformal_withheld_std'], label='Combined', capsize=2)
        ax.plot(x, lo['seen_gate_mahal_or_conformal_withheld_mean'], 'kv', ms=5, label='Combined, seen-subtype test set')
        ax.set_xticks(x); ax.set_xticklabels([SUBTYPE_LABEL.get(v, v) for v in lo['held_out_subtype']], rotation=25, ha='right', fontsize=7); ax.set_ylim(0, 1.05)
        ax.set_ylabel('Fraction withheld'); ax.set_title('Withheld subtype: what each abstention mechanism catches', pad=24)
        ax.legend(frameon=False, ncol=4, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 1.0))
        save(fig, 'f_distributional_gate')
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(lo)); w = 0.2
    ax.bar(x - 1.5 * w, lo['unseen_acc_mean'], w, yerr=lo['unseen_acc_std'], label='Fine accuracy, unseen', capsize=2)
    ax.bar(x - 0.5 * w, lo['seen_acc_mean'], w, yerr=lo['seen_acc_std'], label='Fine accuracy, seen test', capsize=2)
    ax.bar(x + 0.5 * w, lo['unseen_coverage_a0.1_mean'], w, yerr=lo['unseen_coverage_a0.1_std'], label='Fine coverage (0.10), unseen', capsize=2)
    ax.bar(x + 1.5 * w, lo['seen_coverage_a0.1_mean'], w, yerr=lo['seen_coverage_a0.1_std'], label='Fine coverage (0.10), seen test', capsize=2)
    ax.plot(x, lo['unseen_mean_conf_mean'], 'k_', ms=14, mew=2, label='Mean confidence, unseen')
    ax.axhline(0.9, ls='--', lw=0.8, color='k')
    ax.set_xticks(x); ax.set_xticklabels([SUBTYPE_LABEL.get(v, v) for v in lo['held_out_subtype']], rotation=25, ha='right', fontsize=7); ax.set_ylim(0, 1.05)
    ax.set_title('Same fitted pipeline on the withheld subtype and on a seen-subtype test set', pad=28)
    ax.legend(frameon=False, ncol=3, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    save(fig, 'f_unseen_subtype_collapse')

    # ---------------------------------------------------------------- T7 Whelan
    wv = pd.read_csv(R / 'e3_whelan_flight_verdicts.csv')
    wv['case'] = wv['flight_id'].str.extract(r'ace-(\w+?)-')[0].str.capitalize()
    t7 = wv[['case', 'family', 'pred', 'conf', 'set_a0.10', 'set_a0.20', 'coarse_pred', 'coarse_set_a0.10']].copy()
    for c in ('set_a0.10', 'set_a0.20', 'coarse_set_a0.10'):
        t7[c] = t7[c].fillna('(empty)').replace('', '(empty)')
    t7['family'] = t7['family'].map(LABEL); t7['pred'] = t7['pred'].map(LABEL); t7['coarse_pred'] = t7['coarse_pred'].map(LABEL)
    t7 = t7.rename(columns={'case': 'Log', 'family': 'Reference label', 'pred': 'Verdict', 'conf': 'Confidence',
                            'set_a0.10': 'Set (0.10)', 'set_a0.20': 'Set (0.20)', 'coarse_pred': 'Coarse verdict', 'coarse_set_a0.10': 'Coarse set (0.10)'})
    t7.to_csv(T / 't7_whelan_verdicts.csv', index=False)
    md.append('## Table 7. Whelan live logs: physics-channel verdicts (detector trained on simulation only)\n\n' + md_table(t7))
    if args.whelan_dir:
        try:
            from pyulog import ULog
            rows = []
            for f in sorted(pathlib.Path(args.whelan_dir).glob('*.ulg')):
                u = ULog(str(f)); g = next(d for d in u.data_list if d.name in ('vehicle_gps_position', 'sensor_gps'))
                lat = np.asarray(g.data.get('lat', g.data.get('latitude_deg')), float); lon = np.asarray(g.data.get('lon', g.data.get('longitude_deg')), float)
                if np.nanmax(np.abs(lat)) > 1000:
                    lat, lon = lat / 1e7, lon / 1e7
                jump = float(np.max(np.hypot(np.diff(lat) * 111320, np.diff(lon) * 111320 * np.cos(np.radians(lat[:-1])))))
                rows.append({'Log': f.name.split('-')[1].capitalize(), 'fix_type min': int(np.min(g.data['fix_type'])),
                             'satellites min': int(np.min(g.data['satellites_used'])), 'satellites median': int(np.median(g.data['satellites_used'])),
                             'noise max': int(np.max(g.data['noise_per_ms'])), 'jamming indicator max': int(np.max(g.data['jamming_indicator'])),
                             'eph max (m)': float(np.max(g.data['eph'])), 'speed variance median': float(np.median(g.data['s_variance_m_s'])),
                             'speed variance max': float(np.max(g.data['s_variance_m_s'])), 'max fix-to-fix jump (m)': jump,
                             'lat range (deg)': f'{lat.min():.5f} to {lat.max():.5f}', 'lon range (deg)': f'{lon.min():.5f} to {lon.max():.5f}'})
            t7b = pd.DataFrame(rows); t7b.to_csv(T / 't7b_whelan_receiver_channel.csv', index=False)
            md.append('## Table 7b. Whelan live logs: receiver self-report channel\n\n' + md_table(t7b, '.2f'))
        except Exception as e:
            md.append(f'(receiver-channel table skipped: {type(e).__name__}: {e})')
    tl = pd.read_csv(R / 'e3_whelan_window_timeline.csv')
    events = pd.read_csv(R / 'e3_whelan_events.csv').set_index('flight_id') if (R / 'e3_whelan_events.csv').exists() else None
    if events is not None:
        t7c = events.reset_index().copy(); t7c['flight_id'] = t7c['flight_id'].str.extract(r'ace-(\w+?)-')[0].str.capitalize()
        t7c = t7c.rename(columns={'flight_id': 'Log', 'receiver_disturbance_intervals_s': 'Receiver disturbance intervals (s)', 'alert_intervals_s': 'Alert intervals (s)',
                                  'detection_delay_s': 'Detection delay (s)', 'alerts_outside_disturbance': 'Alerts outside disturbance', 'n_windows': 'Windows'})
        t7c.to_csv(T / 't7c_whelan_events.csv', index=False)
        md.append('## Table 7c. Live logs: receiver-derived disturbance intervals (fix type below 3 or fewer than 8 satellites) and alerts from the declared rule '
                  '(P(non-nominal) above 0.5 in two consecutive windows)\n\n' + md_table(t7c, '.1f'))
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.0), sharex=False)
    for ax, (fid, d) in zip(axes, tl.groupby('flight_id', sort=False)):
        d = d.sort_values('t_start')
        ax.stackplot(d['t_start'], [d[f'p_{c}'] for c in CLASSES], labels=[LABEL[c] for c in CLASSES], alpha=0.85)
        if events is not None and fid in events.index:
            for s_, e_ in json.loads(events.loc[fid, 'receiver_disturbance_intervals_s']):
                ax.axvspan(s_, e_, color='k', alpha=0.15, lw=0)
        ax.set_ylim(0, 1); ax.set_ylabel('P(class)'); ax.set_title(fid.replace('whelan_', '').split('_2033')[0], fontsize=8)
    axes[-1].set_xlabel('Time since log start (s)'); axes[0].legend(frameon=False, ncol=4, fontsize=7, loc='lower left')
    axes[-1].text(0.99, 0.02, 'shaded: receiver-derived disturbance', transform=axes[-1].transAxes, ha='right', fontsize=6)
    save(fig, 'f_whelan_timelines')

    # ---------------------------------------------------------------- T8 feature importance, T9 flight-model coefficients
    if (R / 'e1_rep0_window_feature_importance.csv').exists():
        imp = pd.read_csv(R / 'e1_rep0_window_feature_importance.csv').head(15)
        imp.to_csv(T / 't8_window_feature_importance_top15.csv', index=False)
        md.append('## Table 8. Window detector, top-15 feature importance (repetition 0)\n\n' + md_table(imp, '.4f'))
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        ax.barh(imp['feature'][::-1], imp['importance'][::-1]); ax.set_xlabel('Gain importance'); ax.set_title('Window detector: top-15 features')
        save(fig, 'f_feature_importance')
    coef = pd.read_csv(R / 'e1_rep0_flight_model_coefficients.csv', index_col=0)
    coef.columns = [LABEL.get(c, c) for c in coef.columns]
    top = coef.abs().max(axis=1).sort_values(ascending=False).head(20).index
    t9 = coef.loc[top].reset_index().rename(columns={'index': 'Aggregate feature'})
    t9.to_csv(T / 't9_flight_model_coefficients_top20.csv', index=False)
    md.append('## Table 9. Flight-level model, standardised coefficients of the 20 most influential aggregates (repetition 0)\n\n' + md_table(t9))
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    im = ax.imshow(coef.loc[top].to_numpy(), aspect='auto', cmap='coolwarm', vmin=-np.abs(coef.loc[top].to_numpy()).max(), vmax=np.abs(coef.loc[top].to_numpy()).max())
    ax.set_xticks(range(coef.shape[1])); ax.set_xticklabels(coef.columns, rotation=20, ha='right'); ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, label='Coefficient'); ax.set_title('Flight-level model coefficients')
    save(fig, 'f_flight_model_coefficients')

    (T / 'tables.md').write_text('\n\n'.join(md) + '\n')
    print('tables ->', T, '| figures ->', FIG)
    print('\n\n'.join(md))


if __name__ == '__main__':
    main()
