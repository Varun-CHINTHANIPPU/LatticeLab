"""Publication-style, reproducible figures from existing LatticeLab CSV reports.

No attack execution. Inputs are the validated statistics and timing aggregate CSVs.
"""
import argparse
import csv
import math
from pathlib import Path


PALETTE = {"LLL": "#355C7D", "BKZ-20": "#D17A22", "BKZ-30": "#238B73"}
MARKERS = {"LLL": "o", "BKZ-20": "s", "BKZ-30": "^"}


def read_rows(path, required):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    return rows


def load_data(statistics_csv, timing_csv):
    stats = read_rows(statistics_csv, ("n", "reduction", "seeds", "trials", "rate", "wilson95_low", "wilson95_high"))
    timing = read_rows(timing_csv, ("n", "reduction", "seeds", "trials", "median_seconds", "p05_seconds", "p25_seconds", "p75_seconds", "p95_seconds"))
    def parse(rows, fields, is_stats):
        parsed = {}
        for row in rows:
            try:
                n, trials, seeds = int(row["n"]), int(row["trials"]), int(row["seeds"])
                label = row["reduction"]
                vals = {field: float(row[field]) for field in fields}
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid numerical figure input") from exc
            if n <= 0 or trials <= 0 or seeds <= 0 or not label or not all(math.isfinite(v) for v in vals.values()):
                raise ValueError("invalid figure input values")
            key = (n, label)
            if key in parsed:
                raise ValueError(f"duplicate configuration {key}")
            if is_stats and not (0 <= vals["wilson95_low"] <= vals["rate"] <= vals["wilson95_high"] <= 1):
                raise ValueError(f"invalid confidence interval for {key}")
            if not is_stats and not (0 <= vals["p05_seconds"] <= vals["p25_seconds"] <= vals["median_seconds"] <= vals["p75_seconds"] <= vals["p95_seconds"]):
                raise ValueError(f"invalid timing quantiles for {key}")
            parsed[key] = {"n": n, "reduction": label, "trials": trials, "seeds": seeds, **vals}
        return parsed
    s = parse(stats, ("rate", "wilson95_low", "wilson95_high"), True)
    t = parse(timing, ("median_seconds", "p05_seconds", "p25_seconds", "p75_seconds", "p95_seconds"), False)
    if s.keys() != t.keys():
        raise ValueError("statistics and timing configuration sets differ")
    for key in s:
        if (s[key]["trials"], s[key]["seeds"]) != (t[key]["trials"], t[key]["seeds"]):
            raise ValueError(f"trial/seed counts differ for {key}")
    return s, t


def render(statistics_csv, timing_csv, outdir="results/figures", formats=("png", "pdf")):
    stats, times = load_data(statistics_csv, timing_csv)
    if not formats or any(fmt not in ("png", "pdf", "svg") for fmt in formats):
        raise ValueError("formats must be png, pdf, or svg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 240,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    labels = sorted({label for _, label in stats},
                    key=lambda label: (label != "LLL", int(label.split("-")[-1]) if "-" in label else 0))
    dims = sorted({n for n, _ in stats})
    colors = {label: PALETTE.get(label, f"C{i}") for i, label in enumerate(labels)}

    # Recovery with asymmetric Wilson intervals.
    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    for label in labels:
        points = [stats[(n, label)] for n in dims if (n, label) in stats]
        xs = [p["n"] for p in points]
        ys = [p["rate"] for p in points]
        ax.errorbar(xs, ys,
                    yerr=[[p["rate"] - p["wilson95_low"] for p in points],
                          [p["wilson95_high"] - p["rate"] for p in points]],
                    label=label, marker=MARKERS.get(label, "D"), capsize=4,
                    linewidth=1.8, color=colors[label])
    ax.set(xlabel="LWE dimension n", ylabel="Secret recovery probability",
           title="Paired primal attacks: recovery with Wilson 95% intervals")
    ax.set_xticks(dims)
    ax.set_ylim(-.045, 1.065)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    _save(fig, destination, "recovery_wilson", formats, plt)

    # Timing: median with interquartile band and p05-p95 whiskers.
    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    for label in labels:
        points = [times[(n, label)] for n in dims if (n, label) in times]
        xs = [p["n"] for p in points]
        ys = [p["median_seconds"] for p in points]
        ax.errorbar(xs, ys,
                    yerr=[[p["median_seconds"] - p["p05_seconds"] for p in points],
                          [p["p95_seconds"] - p["median_seconds"] for p in points]],
                    color=colors[label], marker=MARKERS.get(label, "D"),
                    linewidth=1.8, capsize=4, label=label)
        if len(xs) > 1:
            ax.fill_between(xs, [p["p25_seconds"] for p in points],
                            [p["p75_seconds"] for p in points],
                            color=colors[label], alpha=.12)
    ax.set(xlabel="LWE dimension n", ylabel="Wall-clock seconds per attack",
           title="Runtime: median, IQR shading and p05–p95 whiskers")
    ax.set_xticks(dims)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    _save(fig, destination, "timing_quantiles", formats, plt)

    # Tradeoff: each point is one configuration, not independent observations.
    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    for label in labels:
        for n in dims:
            if (n, label) not in stats:
                continue
            s, t = stats[(n, label)], times[(n, label)]
            ax.errorbar(t["median_seconds"], s["rate"],
                        yerr=[[s["rate"] - s["wilson95_low"]],
                              [s["wilson95_high"] - s["rate"]]],
                        fmt=MARKERS.get(label, "D"), capsize=3,
                        color=colors[label], markersize=7)
            ax.annotate(f"{label}, n={n}", (t["median_seconds"], s["rate"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Median wall-clock seconds per attack",
           ylabel="Secret recovery probability",
           title="Recovery–runtime trade-off (same experimental conditions)")
    ax.set_ylim(-.05, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    _save(fig, destination, "recovery_runtime", formats, plt)
    return [destination / f"{name}.{fmt}" for name in ("recovery_wilson", "timing_quantiles", "recovery_runtime") for fmt in formats]


def _save(fig, outdir, stem, formats, plt):
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", required=True, help="statistics summary.csv")
    parser.add_argument("--timing", required=True, help="timing_summary.csv")
    parser.add_argument("--outdir", default="results/figures")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"])
    args = parser.parse_args()
    try:
        render(args.statistics, args.timing, args.outdir, tuple(args.formats))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
