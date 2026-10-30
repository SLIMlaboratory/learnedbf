import argparse
import random
from pathlib import Path

import pandas as pd

from learnedbf import complexity_measures as cpl

parser = argparse.ArgumentParser()
parser.add_argument("--keys", default="data/cicids2018_KEYS_keys_dedup.csv")
parser.add_argument(
    "--nonkeys",
    nargs="+",
    default=["data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv"],
)
parser.add_argument("--max-keys", type=int, default=600_000,
                    help="maximum key rows to load (default: benchmark maximum)")
parser.add_argument("--max-nonkeys", type=int, default=900_000,
                    help="maximum training non-key rows to load (default: benchmark maximum)")
parser.add_argument("--nonkey-offset", type=int, default=900_000,
                    help="skip this many leading non-keys, which are the benchmark test split")
parser.add_argument("--chunksize", type=int, default=100_000,
                    help="CSV rows read at a time")
parser.add_argument("--threshold", type=float, default=0.35)
parser.add_argument("--seed", type=int, default=None,
                    help="shuffle column removal order (default: file order)")
parser.add_argument("--out_dir", default="data")
args = parser.parse_args()

sources = [args.keys] + args.nonkeys

def load_limited_sources(paths, source_ids, label, limit, skip_rows=0):
    """Load a bounded row range across CSVs without reading each file at once."""
    frames = []
    remaining = limit
    rows_to_skip = skip_rows

    for path, source_id in zip(paths, source_ids):
        if remaining <= 0:
            break

        for chunk in pd.read_csv(path, chunksize=args.chunksize):
            if rows_to_skip:
                skipped = min(rows_to_skip, len(chunk))
                chunk = chunk.iloc[skipped:]
                rows_to_skip -= skipped

            if chunk.empty:
                continue

            chunk = chunk.iloc[:remaining].copy()
            chunk["label"] = label
            chunk["src"] = source_id
            frames.append(chunk)
            remaining -= len(chunk)
            if remaining <= 0:
                break

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


keys = load_limited_sources([args.keys], [0], 1, args.max_keys)
nonkeys = load_limited_sources(
    args.nonkeys,
    range(1, len(args.nonkeys) + 1),
    -1,
    args.max_nonkeys,
    skip_rows=args.nonkey_offset,
)
if keys.empty or nonkeys.empty:
    parser.error("The selected input files did not contain any rows in the requested ranges.")

data = pd.concat([keys, nonkeys], ignore_index=True)
del keys, nonkeys

columns = [c for c in data.columns if c not in ("label", "src")]
if args.seed is not None:
    random.Random(args.seed).shuffle(columns)

f1v = cpl.F1v()

while columns:
    # duplicates are removed on the current columns plus the label
    df = data[columns + ["label", "src"]].drop_duplicates(columns + ["label"])
    X = df[columns].astype(float).values
    y = ((df["label"] + 1) // 2).astype(bool).values

    score = f1v.compute(X, y)
    print(f"{len(columns)} columns, {len(df)} rows, F1v = {score:.4f}")
    print(f"  columns: {columns}", flush=True)

    if score < args.threshold:
        break

    columns = columns[1:]

print(f"\nRetained columns ({len(columns)}): {columns}")

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
final = data[columns + ["label", "src"]].drop_duplicates(columns + ["label"])
for i, source in enumerate(sources):
    out = out_dir / f"{Path(source).stem}_f1v.csv"
    final.loc[final["src"] == i, columns].to_csv(out, index=False)
    print(f"Written {out}")
