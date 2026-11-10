"""All-trial geometric constraint margins at each trial's worst recorded step."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
STEP_PATH = OUT.parent / 'residual_feature_audit_20261109/normal_step_cluster_audit.csv'
TRIAL_PATH = OUT / 'all_trials.csv'
COLORS = {'cc_sf': '#bd5427', 'cc_mf': '#26739e'}


def render(worst, language):
    ko = language == 'ko'
    if ko:
        font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family': 'Malgun Gothic' if ko else 'DejaVu Sans',
                         'font.size': 12, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 8), sharey=True)
    fig.subplots_adjust(left=.09, right=.96, bottom=.28, top=.79, wspace=.26)
    title = ('전체 trial의 constraint margin: 이전 장애물 기준과 전체 형상 기준' if ko else
             'Constraint margins over all trials: previous obstacle versus full geometry')
    fig.text(.09, .945, title, fontsize=18, weight='bold')
    subtitle = ('DoN 0.85 · 방법별 50 trials · 각 trial에서 실제 전체 margin이 가장 작았던 한 스텝' if ko else
                'DoN 0.85 · 50 trials per method · one step per trial: the smallest recorded full-geometry margin')
    fig.text(.09, .897, subtitle, fontsize=11, color='#5d6771')
    for ax, method in zip(axes, ('cc_sf', 'cc_mf')):
        data = worst[worst.source == method].sort_values('seed')
        color = COLORS[method]
        event = data.specified_tradeoff_event.to_numpy(bool)
        values = [data.keep_margin_mm.to_numpy(), data.fresh_margin_mm.to_numpy()]
        ax.axhspan(-2.45, 0, color='#faeeee', zorder=0)
        ax.axhline(0, color='#8b3434', lw=1.3, zorder=2)
        # Fixed, identical horizontal offsets for both columns of each trial.
        jitter = (((data.seed.to_numpy() * 17) % 50) / 49 - .5) * .34
        for column, yy in enumerate(values):
            ax.scatter(column + jitter[~event], yy[~event], c='#a5adb5', edgecolor='white',
                       linewidth=.35, s=32, alpha=.82, zorder=3)
            ax.scatter(column + jitter[event], yy[event], c=color, edgecolor='white',
                       linewidth=.45, s=42, alpha=.95, zorder=4)
            median = float(np.median(yy))
            ax.plot([column-.23, column+.23], [median, median], color='#34424f', lw=1.6, zorder=5)
        neg_keep = int(np.sum(values[0] < 0))
        neg_fresh = int(np.sum(values[1] < 0))
        ax.set_title(method.upper().replace('_', '-'), fontsize=16, color=color, weight='bold', pad=12)
        ax.set_xlim(-.45, 1.45)
        ax.set_ylim(-2.45, 3.35)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['이전 최근접 장애물\n기준 margin', '전체 장애물\n기준 margin'] if ko else
                           ['Previous nearest\nobstacle margin', 'Full-geometry\nmargin'], fontsize=11)
        ax.grid(axis='y', alpha=.16)
        ax.text(0, -2.30, f'음수 {neg_keep}/50' if ko else f'Negative: {neg_keep}/50',
                ha='center', fontsize=11, color='#6c4040')
        ax.text(1, -2.30, f'음수 {neg_fresh}/50' if ko else f'Negative: {neg_fresh}/50',
                ha='center', fontsize=11, color='#6c4040')
        ax.text(.98, .965, f'표시한 사건: {event.sum()}/50' if ko else f'Specified event: {event.sum()}/50',
                transform=ax.transAxes, ha='right', fontsize=11, color=color)
    axes[0].set_ylabel('기하학적 guard margin (mm)' if ko else 'Geometric guard margin (mm)', fontsize=12)
    fig.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor='#a5adb5', markeredgecolor='white',
                               markersize=7, label='나머지 trial' if ko else 'Other trials'),
                        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['cc_sf'], markeredgecolor='white',
                               markersize=7, label='정의한 사건 발생 (SF)' if ko else 'Specified event (SF)'),
                        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['cc_mf'], markeredgecolor='white',
                               markersize=7, label='정의한 사건 발생 (MF)' if ko else 'Specified event (MF)'),
                        Line2D([0], [0], color='#34424f', lw=1.6, label='중앙값' if ko else 'Median')],
               loc='lower left', bbox_to_anchor=(.08, .178), frameon=False, ncol=4, fontsize=10)
    notes = ([
        'Margin = signed clearance - 0.05 mm. 0 이상이면 guard 만족, 음수이면 guard 위반. 점 하나는 trial 하나입니다.',
        '두 열은 같은 실제 다음 자세를 평가합니다. 색 있는 점은 이전 장애물의 clearance가 증가했지만 다른 장애물의 guard에 진입한 trial입니다.',
        '실패 trial만 고르지 않았습니다. 실제 주행 길이는 방법마다 다릅니다. 이 값은 기하학적 guard margin이며 optimizer 내부 chance residual은 아닙니다.'
    ] if ko else [
        'Margin = signed clearance - 0.05 mm. Nonnegative means guard satisfied; negative means guard violated. One dot represents one trial.',
        'Both columns evaluate the same actual next pose. Colored trials improve the previous obstacle clearance while crossing another obstacle guard.',
        'All trials are included; traversal lengths differ. These are geometric guard margins, not native optimized chance residuals.'
    ])
    for y, note in zip((.145, .100, .055), notes):
        fig.text(.09, y, note, fontsize=9, color='#5d6771')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'population_guard_margins_don085_{language}.{ext}', dpi=220, facecolor='white')
    plt.close(fig)


def main():
    steps = pd.read_csv(STEP_PATH)
    trials = pd.read_csv(TRIAL_PATH)
    selected = steps.groupby(['source', 'seed'], sort=True).d_fresh_cluster_next_mm.idxmin()
    worst = steps.loc[selected].copy()
    worst = worst.merge(trials[np.isclose(trials.don, .85)][['source', 'seed', 'specified_tradeoff_event']],
                        on=['source', 'seed'], validate='one_to_one')
    worst['keep_margin_mm'] = worst.d_old_cluster_next_mm - worst.d_col_mm
    worst['fresh_margin_mm'] = worst.d_fresh_cluster_next_mm - worst.d_col_mm
    assert len(worst) == 100 and (worst.groupby('source').size() == 50).all()
    assert (worst.keep_margin_mm >= worst.fresh_margin_mm - 1e-10).all()
    assert (worst.keep_margin_mm >= 0).all()
    assert worst[worst.specified_tradeoff_event].terminal_guard_entry.all()
    worst.to_csv(OUT / 'worst_recorded_margins_don085.csv', index=False)
    summary = []
    for method in ('cc_sf', 'cc_mf'):
        d = worst[worst.source == method]
        summary.append(dict(source=method, trials=len(d), negative_keep=int((d.keep_margin_mm<0).sum()),
            negative_fresh=int((d.fresh_margin_mm<0).sum()), specified_event=int(d.specified_tradeoff_event.sum()),
            keep_margin_median_mm=float(d.keep_margin_mm.median()), fresh_margin_median_mm=float(d.fresh_margin_mm.median()),
            keep_margin_min_mm=float(d.keep_margin_mm.min()), fresh_margin_min_mm=float(d.fresh_margin_mm.min())))
    for language in ('ko', 'en'):
        render(worst, language)
    report = dict(
        scope='All100 trials at DoN.85, including failures. One record each: successful-solver applied step with minimum fresh full-geometry signed clearance; first chronological occurrence breaks exact ties.',
        margins='m_keep=d_jk(x_next)-d_col, m_fresh=min_j d_j(x_next)-d_col, j_k nearest obstacle at actual current pose. Both margins use the same actual next pose. Units mm; d_col=.05mm.',
        selection_caveat='Worst recorded geometric margin, not a random sample. Exposure/traversal durations differ across methods. No optimizer-native feature or chance constraint is recovered.',
        points='Horizontal deterministic offsets avoid overlap, without changing y-values. Same seed has the same horizontal offset in both columns. Colored points mark the separately defined clearance-improvement guard event.',
        summaries=summary,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (STEP_PATH, TRIAL_PATH)},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / 'population_margin_manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
