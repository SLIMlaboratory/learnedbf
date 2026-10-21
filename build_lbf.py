
import numpy as np
import pandas as pd


from learnedbf import LBF
from learnedbf.classifiers import ScoredLinearSVC
from learnedbf.BF import ClassicalBloomFilter

def load_csv(path, num_rows, expected_columns=None):
    df = pd.read_csv(path, nrows=num_rows)

    # Column consistency check
    if expected_columns is not None:
        if list(df.columns) != expected_columns:
            raise ValueError(
                f"Column mismatch in {path}\n"
                f"Diff: {set(df.columns).symmetric_difference(expected_columns)}"
            )
    else:
        expected_columns = list(df.columns)

    # Convert using column names
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X = df.to_numpy(dtype=np.float32, copy=False)
    return X, expected_columns


if __name__ == "__main__":
    # using argc and argv, fecth csv paths and other params from command line
    import argparse
    import sys
    
    # first argument is key csv path, second argument is non-key csv path,
    # third argument is n_keys, fourth argument is n_nonkeys
    parser = argparse.ArgumentParser(description="Build and evaluate a learned"
                                     " Bloom filter from CSV data.")
    parser.add_argument("--key_csv_path", type=str,
                        help="Path to the CSV file containing key rows.")
    parser.add_argument("--nonkey_csv_path", type=str,
                        help="Path to the CSV file containing non-key rows.")
    parser.add_argument("--n_keys", type=int,
                        help="Number of keys to insert into the learned Bloom filter.")
    parser.add_argument("--n_nonkeys", type=int,
                        help="Number of non-keys to build the learned Bloom filter.")
    parser.add_argument("--n_test_nonkeys", type=int,
                            help="Number of non-keys to probe against the Bloom filter.")
    parser.add_argument("--file_name", type=str,
                        help="Base name for output files.")
    parser.add_argument("--epsilon", type=float,
                        help="Desired false positive rate.")

    args = parser.parse_args()

    X_keys, _ = load_csv(args.key_csv_path, args.n_keys)
    X_nonkeys, _ = load_csv(args.nonkey_csv_path, args.n_nonkeys+args.n_test_nonkeys)
    y_keys = np.ones(len(X_keys), dtype=np.int8)
    y_nonkeys = np.zeros(len(X_nonkeys), dtype=np.int8)

    X_test_nonkeys = X_nonkeys[:args.n_test_nonkeys]
    y_test_nonkeys = y_nonkeys[:args.n_test_nonkeys]

    X_nonkeys = X_nonkeys[args.n_test_nonkeys:]
    y_nonkeys = y_nonkeys[args.n_test_nonkeys:]

    # Combine keys and non-keys into a single dataset
    X = np.vstack((X_keys, X_nonkeys))
    y = np.concatenate((y_keys, y_nonkeys)).astype(bool)

    hyper={'C':[0.01, 0.1, 1, 5, 10]}
    hyper={'C':[0.01]}
    
    learned_filter = LBF(
        epsilon=args.epsilon,
        classifier=ScoredLinearSVC(max_iter=1000, random_state=42),
        threshold_test_size=0.2,
        hyperparameters=hyper
    )
    learned_filter.fit(X, y)

    learned_filter.export(f"{args.file_name}.learned.json")

    fpr_nonkeys = learned_filter.estimate_FPR(X_test_nonkeys)
    size = sum(learned_filter.get_size().values())

    with open(f"{args.file_name}.learned.fpr", "w") as f:
        f.write(f"FPR: {fpr_nonkeys}\n")
        f.write(f"size: {size}\n")

    classical_filter = ClassicalBloomFilter(epsilon=args.epsilon, n=args.n_keys)
    classical_filter.fit(X_keys)

    classical_filter.export(f"{args.file_name}.classical.json")
    classical_fpr = classical_filter.estimate_FPR(X_test_nonkeys)

    with open(f"{args.file_name}.classical.fpr", "w") as f:
        f.write(f"FPR: {classical_fpr}\n")
        f.write(f"size: {classical_filter.get_size()}\n")
