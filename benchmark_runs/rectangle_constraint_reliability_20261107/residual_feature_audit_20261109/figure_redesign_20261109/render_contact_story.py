"""Actual footprint overview and contact zooms, with unexaggerated geometry."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent
OLD, NEW = '#8c949e', '#253e58'
GREEN, RED = '#14735b', '#bc4237'
SF, MF = '#bd5427', '#26739e'


def clip_left(poly):
    """Exact polygon portion inside x<=0, for the penetration highlight."""
    result = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        inside_a, inside_b = a[0] <= 0, b[0] <= 0
        if inside_a:
            result.append(a)
        if inside_a != inside_b:
            result.append(a + (-a[0] / (b[0] - a[0])) * (b - a))
    return np.array(result)


def footprint(ax, vertices, old=False, fill=True):
    ax.add_patch(Polygon(vertices, closed=True,
        facecolor='#e9edf1' if fill and not old else 'none',
        edgecolor=OLD if old else NEW, linewidth=1.9 if old else 2.2,
        linestyle=(0, (4, 2)) if old else '-', zorder=4 if old else 3))


def frame(ax):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def label_badge(ax, position, label, color):
    ax.text(*position, label, ha='center', va='center', color='white',
            fontsize=13, weight='bold', zorder=12,
            bbox=dict(boxstyle='circle,pad=.23', facecolor=color, edgecolor='white', linewidth=1.2))


def render(data, counts, language):
    ko = language == 'ko'
    if ko:
        font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family': 'Malgun Gothic' if ko else 'DejaVu Sans',
                         'font.size': 12, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.unicode_minus': False})
    before, after = [1000 * np.array(f['world_vertices_m']) for f in data['frames']]
    wall_y = 1000 * data['wall_surfaces']['bottom_y_m']
    a_before, a_after = before[1], after[1]
    b_before, b_after = before[0], after[0]
    a_values = [a_before[1] - wall_y, a_after[1] - wall_y]
    b_values = [b_before[0], b_after[0]]
    fig = plt.figure(figsize=(15.2, 8.6), facecolor='white')
    title = '아래 벽의 여유를 늘린 0.1초, 왼쪽 벽은 침범' if ko else 'More clearance to one wall, penetration of the other — in 0.1 s'
    fig.text(.045, .947, title, fontsize=24 if ko else 21, weight='bold', color='#1b2631')
    subtitle = ('DoN 0.85 · CC-SF seed 10 · 실제 기록 자세 4.6 → 4.7 s' if ko
                else 'DoN 0.85 · CC-SF seed 10 · actual recorded poses at 4.6 and 4.7 s')
    fig.text(.045, .899, subtitle, fontsize=13, color='#5a6570')

    for x, w in ((.04, .40), (.46, .24), (.72, .24)):
        fig.add_artist(FancyBboxPatch((x, .255), w, .585, transform=fig.transFigure,
            boxstyle='round,pad=.009,rounding_size=.012', facecolor='white',
            edgecolor='#dde2e7', linewidth=1, zorder=-5))
    fig.text(.06, .802, '실제 형상과 두 접촉부' if ko else 'Actual footprint and wall contacts', fontsize=15, weight='bold')
    fig.legend(handles=[Line2D([0], [0], color=OLD, lw=2, ls='--', label='4.6 s'),
                        Line2D([0], [0], color=NEW, lw=2, label='4.7 s')],
               loc='upper left', bbox_to_anchor=(.052, .784), frameon=False, ncol=2,
               handlelength=2.3, columnspacing=1.4, fontsize=11)
    ax = fig.add_axes([.058, .295, .365, .44])
    ax.add_patch(Rectangle((-18, 430), 18, 235, facecolor='#f6e4e1', edgecolor='none'))
    ax.add_patch(Rectangle((0, 430), 215, 20, facecolor='#e3f0e9', edgecolor='none'))
    ax.add_patch(Rectangle((150, 600), 65, 65, facecolor='#eceff2', edgecolor='none'))
    ax.plot([0, 0], [430, 665], color=RED, lw=2)
    ax.plot([0, 215], [450, 450], color=GREEN, lw=2)
    footprint(ax, before, old=True)
    footprint(ax, after)
    ax.scatter([a_before[0], b_before[0]], [a_before[1], b_before[1]],
               s=16, facecolor='white', edgecolor=OLD, zorder=8)
    ax.scatter([a_after[0], b_after[0]], [a_after[1], b_after[1]],
               s=20, c=[GREEN, RED], zorder=9)
    label_badge(ax, (112, 438), 'A', GREEN)
    ax.annotate('', xy=a_after, xytext=(108, 444), arrowprops=dict(arrowstyle='-', color=GREEN, lw=1.1))
    label_badge(ax, (-11, 589), 'B', RED)
    ax.annotate('', xy=b_after, xytext=(-8, 584), arrowprops=dict(arrowstyle='-', color=RED, lw=1.1))
    ax.text(160, 438, '아래 벽' if ko else 'Bottom wall', color=GREEN, fontsize=10, ha='center')
    ax.text(18, 617, '왼쪽 벽' if ko else 'Left wall', color=RED, fontsize=10, ha='left')
    ax.text(99, 549, 'CC-SF', color=NEW, fontsize=14, weight='bold', ha='center')
    ax.set(xlim=(-18, 215), ylim=(430, 665))
    frame(ax)
    fig.text(.24, .276, '중심 이동 1.42 mm · 회전 +2.37°' if ko else 'Center displacement 1.42 mm · rotation +2.37°',
             ha='center', fontsize=11, color='#5a6570')

    fig.text(.48, .802, 'A  아래쪽 벽' if ko else 'A  Bottom wall', color=GREEN, fontsize=16, weight='bold')
    fig.text(.48, .755, '기존 최근접 벽: 여유 증가' if ko else 'Previously closest: gap increases', fontsize=11, color='#515b65')
    axa = fig.add_axes([.475, .465, .21, .245])
    axa.axhspan(448.5, 450, facecolor='#e3f0e9', zorder=0)
    axa.axhline(450, color=GREEN, lw=2)
    footprint(axa, before, old=True)
    footprint(axa, after)
    for point, color in [(a_before, OLD), (a_after, NEW)]:
        axa.plot([point[0], point[0]], [450, point[1]], color=color, lw=1.2, zorder=8)
        axa.scatter([point[0]], [point[1]], s=24, color=color, zorder=10)
        axa.plot([point[0] - .12, point[0] + .12], [450, 450], color=color, lw=1.2, zorder=8)
    axa.annotate('4.6 s', xy=a_before, xytext=(85.5, 452.5), color=OLD, fontsize=10,
                 arrowprops=dict(arrowstyle='-', color=OLD, lw=1))
    axa.annotate('4.7 s', xy=a_after, xytext=(91.0, 453.4), color=NEW, fontsize=10,
                 arrowprops=dict(arrowstyle='-', color=NEW, lw=1))
    axa.plot([85.6, 86.6], [449.2, 449.2], color='#515b65', lw=1.3)
    axa.text(86.1, 448.8, '1 mm', color='#515b65', fontsize=8, ha='center')
    axa.set(xlim=(85, 94), ylim=(448.5, 454))
    frame(axa)
    fig.text(.58, .435, '접촉부 확대 · 실제 형상' if ko else 'Magnified contact · actual geometry', ha='center', fontsize=9, color='#78828c')
    fig.text(.58, .371, f'{a_values[0]:.3f} → {a_values[1]:.3f} mm', ha='center', fontsize=17, weight='bold', color='#253e58')
    fig.text(.58, .307, f'여유 +{a_values[1]-a_values[0]:.3f} mm' if ko else f'Gap +{a_values[1]-a_values[0]:.3f} mm',
             ha='center', fontsize=16, color=GREEN, weight='bold')

    fig.text(.74, .802, 'B  왼쪽 벽' if ko else 'B  Left wall', color=RED, fontsize=16, weight='bold')
    fig.text(.74, .755, '다른 근접 벽: 경계 침범' if ko else 'Other nearby wall: penetration', fontsize=11, color='#515b65')
    axb = fig.add_axes([.738, .457, .205, .268])
    axb.axvspan(-1.5, 0, facecolor='#f6e4e1', zorder=0)
    axb.axvline(0, color=RED, lw=2)
    footprint(axb, before, old=True)
    footprint(axb, after)
    penetration = clip_left(after)
    assert len(penetration) == 3
    axb.add_patch(Polygon(penetration, closed=True, facecolor=RED, edgecolor=RED, zorder=6))
    for point, color in [(b_before, OLD), (b_after, RED)]:
        axb.plot([0, point[0]], [point[1], point[1]], color=color, lw=1.2, zorder=8)
        axb.scatter([point[0]], [point[1]], s=24, color=color, zorder=10)
    axb.annotate('4.6 s', xy=b_before, xytext=(1.5, 574.0), color=OLD, fontsize=10,
                 arrowprops=dict(arrowstyle='-', color=OLD, lw=1))
    axb.annotate('4.7 s', xy=b_after, xytext=(1.1, 568.8), color=NEW, fontsize=10,
                 arrowprops=dict(arrowstyle='-', color=NEW, lw=1))
    axb.plot([1.9, 2.9], [568.2, 568.2], color='#515b65', lw=1.3, zorder=11)
    axb.text(2.4, 567.8, '1 mm', color='#515b65', fontsize=8, ha='center')
    axb.set(xlim=(-1.5, 3.5), ylim=(567.5, 575))
    frame(axb)
    fig.text(.84, .435, '빨간 면적 = 실제 침범 부분' if ko else 'Red area = actual penetration', ha='center', fontsize=9, color='#78828c')
    fig.text(.84, .371, f'{b_values[0]:.3f} → -{abs(b_values[1]):.3f} mm', ha='center', fontsize=17, weight='bold', color='#253e58')
    fig.text(.84, .307, f'{abs(b_values[1]):.3f} mm 침범' if ko else f'{abs(b_values[1]):.3f} mm penetration',
             ha='center', fontsize=16, color=RED, weight='bold')

    fig.add_artist(Line2D([.04, .96], [.215, .215], transform=fig.transFigure, color='#dde2e7', lw=1))
    fig.text(.045, .166, '기존 벽 여유 증가 + 다른 벽 guard 진입' if ko else 'Old-wall gap improves; another guard is crossed', fontsize=12, weight='bold')
    fig.text(.045, .122, '다른 벽 거리 < 0.05 mm · 동일한 50개 seed' if ko else 'Other-wall distance < 0.05 mm · all 50 matched seeds', fontsize=10, color='#68727c')
    for x, method, color in [(.56, 'cc_sf', SF), (.84, 'cc_mf', MF)]:
        g = next(row for row in counts['groups'] if row['source'] == method)
        fig.text(x, .157, f'{method.upper().replace("_", "-")}   {g["tradeoff_guard_entry_trials"]}/50',
                 ha='center', fontsize=23, weight='bold', color=color)
        fig.text(x, .115, f'{100*g["trial_fraction"]:.0f}%'+('의 trial' if ko else ' of trials'), ha='center', fontsize=12, color='#68727c')
    note = ('실제 기록 두 자세만 사용. 이동·회전은 과장하지 않았으며, 접촉부만 확대했습니다. '
            '이 패턴은 실행 궤적의 관측 결과이며 optimizer 내부 feature 선택을 복원한 결과는 아닙니다.' if ko else
            'Two actual recorded poses; no exaggerated motion or rotation. Only the contact views are magnified. '
            'Observed motion does not recover the optimizer’s internal feature choice.')
    fig.text(.045, .045, note, fontsize=9, color='#68727c')
    stem = f'clearance_tradeoff_contact_story_{language}'
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'{stem}.{extension}', dpi=220, facecolor='white')
    plt.close(fig)


def main():
    path = OUT / 'case_geometry.json'
    counts_path = OUT.parent / 'clearance_tradeoff_summary.json'
    geometry = json.loads(path.read_text(encoding='utf-8'))
    counts = json.loads(counts_path.read_text(encoding='utf-8'))
    for language in ('ko', 'en'):
        render(geometry, counts, language)
    report = dict(
        source_case='DoN .85, CC-SF seed10, actual recorded poses at4.6s and4.7s',
        selection='Previously selected representative: median SF failure time in17 MF-only-success pairs; smallest seed among ties. Not maximum clearance gap or maximum improvement.',
        diagram='Rigid transform of recorded footprint only; no interpolation, geometry exaggeration, displaced copies, new simulations, optimizer calls or probes. Both contact windows keep equal aspect ratio and show1mm scale bars.',
        old_vs_new='Dashed grey =4.6s, solid dark =4.7s. Green A is previously nearest bottom wall; red B is left boundary wall. The red filled triangle is the exact polygon portion x<=0.',
        population='Same retrospective geometric event as clearance_tradeoff_summary.json: old nearest obstacle gap increases and stays>=d_col, actual global gap falls<d_col. All50 trials percontroller; not all non-events are successes.',
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, counts_path)},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / 'contact_story_manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
