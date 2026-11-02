#!/usr/bin/env python3
"""Plot memory usage and average query latency from experiment results."""

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
    "Query_Avg_us",
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
        description="Plot memory usage and average query latency against number of keys."
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
        help="base chart image path; _memory and _time are added (default: results.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also display the chart interactively",
    )
    parser.add_argument(
        "--log-time",
        action="store_true",
        help="use a logarithmic Y axis for the query latency chart",
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
    results["Query_Avg_us"] = pd.to_numeric(
        results["Query_Avg_us"], errors="coerce"
    )
    results = results[
        results["Language"].isin(LANGUAGE_COLORS)
        & results["Filter"].isin(FILTER_STYLES)
    ]
    figures = []
    for column, suffix, title, ylabel, scale, y_format in (
        (
            "Peak_RSS_KB",
            "memory",
            "Peak RAM Usage vs Number of Keys",
            "Peak RAM usage (MB)",
            1000.0,
            "{x:,.0f}",
        ),
        (
            "Query_Avg_us",
            "time",
            "Average Query Latency vs Number of Keys",
            "Average query latency (us)",
            1.0,
            "{x:,.2f}",
        ),
    ):
        metric_results = results.dropna(subset=["N_KEYS", column])
        if metric_results.empty:
            parser.error(f"input CSV has no valid {suffix} data")

        # Average duplicate measurements at the same key count, if present.
        grouped = (
            metric_results.groupby(
                ["Language", "Filter", "N_KEYS"], as_index=False
            )[column]
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
                    series[column] / scale,
                    color=LANGUAGE_COLORS[language],
                    linestyle=FILTER_STYLES[filter_type],
                    marker="o",
                    label=f"{language.title()} {filter_type.title()}",
                )

        ax.set_title(title)
        ax.set_xlabel("Number of keys")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        if args.log_time and suffix == "time":
            ax.set_yscale("log")
            y_format = "{x:,g}"
        ax.yaxis.set_major_formatter(StrMethodFormatter(y_format))
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(title="Implementation / filter")

        output = args.output.with_name(
            f"{args.output.stem}_{suffix}{args.output.suffix}"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")
        print(f"Chart saved to {output}")
        figures.append(fig)

    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
