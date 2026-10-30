#!/usr/bin/env python3
"""Plot peak RAM usage from the experiment results CSV."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter


REQUIRED_COLUMNS = {
    "Language",
    "Filter",
    "N_KEYS",
    "Peak_RSS_KB",
}
LANGUAGE_COLORS = {
    "python": "tab:blue",
    "c": "tab:orange",
}
FILTER_STYLES = {
    "learned": "-",
    "classical": "--",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot peak RAM usage against the number of keys."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results.csv"),
        help="experiment results CSV (default: results.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.png"),
        help="chart image path (default: results.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also display the chart interactively",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input CSV does not exist: {args.input}")

    results = pd.read_csv(args.input)
    results.columns = results.columns.astype(str).str.strip()
    missing_columns = REQUIRED_COLUMNS.difference(results.columns)
    if missing_columns:
        parser.error(
            "input CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    results["Language"] = results["Language"].astype(str).str.strip().str.lower()
    results["Filter"] = results["Filter"].astype(str).str.strip().str.lower()
    results["N_KEYS"] = pd.to_numeric(results["N_KEYS"], errors="coerce")
    results["Peak_RSS_KB"] = pd.to_numeric(
        results["Peak_RSS_KB"], errors="coerce"
    )
    results = results.dropna(subset=["N_KEYS", "Peak_RSS_KB"])
    results = results[
        results["Language"].isin(LANGUAGE_COLORS)
        & results["Filter"].isin(FILTER_STYLES)
    ]
    if results.empty:
        parser.error("input CSV has no rows with valid language, filter, and RAM data")

    # Average duplicate measurements at the same key count, if present.
    grouped = (
        results.groupby(["Language", "Filter", "N_KEYS"], as_index=False)["Peak_RSS_KB"]
        .mean()
        .sort_values("N_KEYS")
    )

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for language in ("python", "c"):
        for filter_type in ("learned", "classical"):
            series = grouped[
                (grouped["Language"] == language)
                & (grouped["Filter"] == filter_type)
            ]
            if series.empty:
                continue

            ax.plot(
                series["N_KEYS"],
                series["Peak_RSS_KB"] / 1000.0,
                color=LANGUAGE_COLORS[language],
                linestyle=FILTER_STYLES[filter_type],
                marker="o",
                label=f"{language.title()} {filter_type.title()}",
            )

    ax.set_title("Peak RAM Usage vs Number of Keys")
    ax.set_xlabel("Number of keys")
    ax.set_ylabel("Peak RAM usage (MB)")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(title="Implementation / filter")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Chart saved to {args.output}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
