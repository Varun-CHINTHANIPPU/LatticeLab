"""Turn generated CSV results into reproducible figures (matplotlib required)."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failrate", default="results/failrate.csv")
    parser.add_argument("--attack", default="results/attack.csv")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()
    if not Path(args.failrate).is_file() and not Path(args.attack).is_file():
        parser.error(f"no input CSV found: {args.failrate!r} or {args.attack!r}")
    import matplotlib.pyplot as plt

    destination = Path(args.outdir)
    destination.mkdir(parents=True, exist_ok=True)
    if Path(args.failrate).exists():
        with open(args.failrate, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            parser.error(f"failure-rate CSV is empty: {args.failrate}")
        xs = [float(r["sigma"]) for r in rows]
        measured = [float(r["measured"]) for r in rows]
        predicted = [float(r["predicted"]) for r in rows]
        figure, ax = plt.subplots()
        ax.plot(xs, measured, marker="o", label="Observed")
        ax.plot(xs, predicted, marker="s", label="Across-key prediction")
        ax.set_xlabel("Rounded-Gaussian sigma")
        ax.set_ylabel("Decryption failure probability")
        ax.set_title("LWE decryption failure: measured vs predicted")
        ax.legend()
        figure.savefig(destination / "failrate.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {destination / 'failrate.png'}")
    if Path(args.attack).exists():
        with open(args.attack, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            parser.error(f"attack CSV is empty: {args.attack}")
        figure, ax = plt.subplots()
        for label in sorted(set(r["reduction"] for r in rows)):
            selected = sorted((r for r in rows if r["reduction"] == label), key=lambda r: int(r["n"]))
            ax.plot([int(r["n"]) for r in selected], [float(r["success_rate"]) for r in selected],
                    marker="o", label=label)
        ax.set_xlabel("LWE dimension n")
        ax.set_ylabel("Recovery fraction")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Primal recovery under a fixed trial budget")
        ax.legend()
        figure.savefig(destination / "attack.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {destination / 'attack.png'}")


if __name__ == "__main__":
    main()
