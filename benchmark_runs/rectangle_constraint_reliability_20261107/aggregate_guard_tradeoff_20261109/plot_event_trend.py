"""Publication-size Korean/English event-incidence figures and count table."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
COLORS = {"cc_sf": "#BD5427", "cc_mf": "#26739E"}


def main():
    data = pd.read_csv(OUT / "cohort_summary.csv")
    assert len(data) == 10 and data.trial_count.eq(50).all()
    assert np.allclose(data.specified_event_percent, 100 * data.specified_event_trials / 50)
    wide = []
    for don, group in data.groupby("don", sort=True):
        row = {"DoN": don}
        for source, prefix in (("cc_sf", "CC_SF"), ("cc_mf", "CC_MF")):
            r = group[group.source.eq(source)].iloc[0]
            row.update({f"{prefix}_event": int(r.specified_event_trials),
                        f"{prefix}_other_guard_failure": int(r.other_guard_failed_trials),
                        f"{prefix}_goal": int(r.successful_trials),
                        f"{prefix}_total": int(r.trial_count),
                        f"{prefix}_event_percent": float(r.specified_event_percent),
                        f"{prefix}_Wilson95_lower_percent": float(r.wilson95_lower_percent),
                        f"{prefix}_Wilson95_upper_percent": float(r.wilson95_upper_percent)})
            assert row[f"{prefix}_event"] + row[f"{prefix}_other_guard_failure"] + row[f"{prefix}_goal"] == 50
        wide.append(row)
    pd.DataFrame(wide).to_csv(OUT / "event_trend_counts_wide.csv", index=False)

    font_manager.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
    font_manager.fontManager.addfont("C:/Windows/Fonts/malgunbd.ttf")
    definitions = {
        "ko": dict(
            title="기존 장애물의 여유가 증가했지만 다른 장애물의 guard에 진입한 trial",
            subtitle="DoN 0.83–0.87  |  방법·DoN별 50 trial  |  실제 기록된 자세의 기하학적 평가",
            ylabel="해당 이벤트가 관측된 trial 비율", xlabel="DoN (로봇 너비 / 통로 너비)",
            note1="이벤트: 이전 최인접 장애물의 clearance는 증가하며 guard 밖에 유지되지만, 다음 자세의 전체 clearance는 guard 아래로 감소.",
            note2=r"$\Delta d_{\mathrm{old}}>0,\quad d_{\mathrm{old,next}}\geq d_{\mathrm{col}},\quad d_{\mathrm{global,next}}<d_{\mathrm{col}}$"
                  "    ($d_{col}=0.05$ mm).  표식: 이벤트 trial 수 / 전체 50 trial.",
            note3="전체 500 trial 포함. 실제 최적화 후보의 constraint 위반율이 아닌, 기록된 주행의 기하학적 패턴 비율.",
            font="Malgun Gothic"),
        "en": dict(
            title="Increasing clearance to one obstacle while entering another obstacle's guard",
            subtitle="DoN 0.83–0.87  |  50 trials per controller and DoN  |  Geometry at actual recorded poses",
            ylabel="Trials with the specified event", xlabel="DoN (robot width / passage width)",
            note1="Event: clearance to the previously closest obstacle increases and stays outside its guard, while global next-pose clearance falls below the guard.",
            note2=r"$\Delta d_{\mathrm{old}}>0,\quad d_{\mathrm{old,next}}\geq d_{\mathrm{col}},\quad d_{\mathrm{global,next}}<d_{\mathrm{col}}$"
                  "    ($d_{col}=0.05$ mm).  Labels: event trials / all 50 trials.",
            note3="All 500 trials included. This is an executed-geometry pattern, not the constraint violation rate of optimized candidate states.",
            font="DejaVu Sans"),
    }
    outputs = []
    for language, text in definitions.items():
        with plt.rc_context({"font.family": text["font"], "font.size": 15,
                             "axes.unicode_minus": False, "pdf.fonttype": 42,
                             "svg.fonttype": "path", "axes.spines.top": False,
                             "axes.spines.right": False, "axes.edgecolor": "#555555",
                             "axes.labelcolor": "#242424", "text.color": "#242424",
                             "xtick.color": "#444444", "ytick.color": "#444444"}):
            fig, ax = plt.subplots(figsize=(14, 7))
            fig.subplots_adjust(left=.102, right=.963, bottom=.29, top=.79)
            fig.text(.102, .947, text["title"], fontsize=20, fontweight="bold", va="top")
            fig.text(.102, .883, text["subtitle"], fontsize=13.5, color="#555555", va="top")
            for source, label in (("cc_sf", "CC-SF"), ("cc_mf", "CC-MF")):
                g = data[data.source.eq(source)].sort_values("don")
                ax.plot(g.don, g.specified_event_percent, label=label,
                        color=COLORS[source], linewidth=2.8,
                        linestyle="-" if source == "cc_sf" else (0, (5, 3)),
                        marker="o", markersize=9, markeredgewidth=2,
                        markerfacecolor=COLORS[source] if source == "cc_sf" else "white",
                        clip_on=False, zorder=4)
                for r in g.itertuples(index=False):
                    offset, align = (0, 14), "center"
                    if r.don == .83:
                        offset, align = ((16, -7), "left") if source == "cc_sf" else ((-15, 15), "right")
                    elif r.don == .87:
                        offset, align = ((-15, -24), "right") if source == "cc_sf" else ((15, 13), "left")
                    elif source == "cc_mf" and r.don == .86:
                        offset = (0, -25)
                    ax.annotate(f"{r.specified_event_trials}/50", (r.don, r.specified_event_percent),
                                xytext=offset, textcoords="offset points", ha=align,
                                va="center", fontsize=14, fontweight="bold",
                                color=COLORS[source], annotation_clip=False)
            ax.set_xlim(.8275, .8725)
            ax.set_ylim(0, 100)
            ax.set_xticks([.83, .84, .85, .86, .87], ["0.83", "0.84", "0.85", "0.86", "0.87"])
            ax.set_yticks(np.arange(0, 101, 20))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.tick_params(axis="both", labelsize=14, length=5, pad=8)
            ax.set_xlabel(text["xlabel"], fontsize=15, labelpad=12)
            ax.set_ylabel(text["ylabel"], fontsize=15, labelpad=13)
            ax.grid(axis="y", color="#DADADA", linewidth=.8, alpha=.8)
            ax.set_axisbelow(True)
            ax.legend(loc="upper left", frameon=False, fontsize=16, handlelength=3,
                      borderaxespad=.35, labelspacing=.65)
            fig.text(.102, .158, text["note1"], fontsize=11.2, va="top")
            fig.text(.102, .112, text["note2"], fontsize=11.6, va="top")
            fig.text(.102, .060, text["note3"], fontsize=10.8, color="#555555", va="top")
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bounds = fig.bbox
            for artist in list(fig.texts) + list(ax.texts):
                b = artist.get_window_extent(renderer)
                if not (bounds.x0 <= b.x0 and b.x1 <= bounds.x1
                        and bounds.y0 <= b.y0 and b.y1 <= bounds.y1):
                    raise AssertionError(f"Text outside canvas: {artist.get_text()}")
            for extension in ("png", "pdf", "svg"):
                path = OUT / f"guard_tradeoff_trend_{language}.{extension}"
                fig.savefig(path, dpi=220, facecolor="white")
                outputs.append(path)
            plt.close(fig)
    audit = dict(input="cohort_summary.csv",
                 input_sha256=hashlib.sha256((OUT / "cohort_summary.csv").read_bytes()).hexdigest(),
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 scope="All five DoNs; 50 trials per controller/DoN; zero event represented as zero",
                 primary_intervals="Not plotted per requested design; Wilson95 intervals remain in companion CSV",
                 exported_files=[p.name for p in outputs],
                 automated_checks="10 plotted points match count/50; all event+other guard+goal counts sum to50; all authored text within canvas",
                 visual_inspection="Pending exported PNG review")
    (OUT / "event_trend_figure_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("Saved Korean and English PNG/PDF/SVG plus event_trend_counts_wide.csv", flush=True)


if __name__ == "__main__":
    main()
