"""Generate toy LBF models and CSV inputs for exercising read_lbf.c.

Run with the project environment. Outputs are written under data/:

* lbf_toy.json
* lbf_toy_keys.csv
* lbf_toy_nonkeys.csv
* lbf_toy_16d.json
* lbf_toy_16d_keys.csv
* lbf_toy_16d_nonkeys.csv
"""

import csv
import json
from pathlib import Path

import numpy as np

from learnedbf import LBF
from learnedbf.classifiers import ScoredLinearSVC


def json_default(value: object) -> object:
    """Convert NumPy scalars/arrays that the model's serializer leaves behind."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_cases(path: Path, values: np.ndarray) -> None:
    """Write float64 feature rows with a CSV header."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([f"feature_{index}" for index in range(values.shape[1])])
        writer.writerows(
            [format(float(value), ".17g") for value in row]
            for row in values
        )


def export_filter(learned_filter: LBF, model_path: Path) -> None:
    """Write an LBF JSON representation, converting NumPy scalar values."""
    with model_path.open("w", encoding="utf-8") as model_file:
        json.dump(learned_filter.to_json(), model_file, default=json_default)


def build_and_export_case(output_dir: Path, name: str, objects: np.ndarray,
                          labels: np.ndarray) -> LBF:
    """Fit a linear-SVC-backed LBF and write model/key/non-key artifacts."""
    classifier = ScoredLinearSVC(random_state=522812,
                                 max_iter=100000,
                                 tol=0.1,
                                 C=0.1)
    classifier.fit(objects, labels)

    learned_filter = LBF(classifier=classifier, epsilon=0.1, n=len(objects))
    learned_filter.fit(objects, labels)
    if learned_filter.backup_filter_ is None:
        raise RuntimeError(f"Expected {name} LBF to have a backup Bloom filter")

    model_path = output_dir / f"{name}.json"
    keys_path = output_dir / f"{name}_keys.csv"
    nonkeys_path = output_dir / f"{name}_nonkeys.csv"
    export_filter(learned_filter, model_path)
    write_cases(keys_path, objects[labels])
    write_cases(nonkeys_path, objects[~labels])

    print(f"Exported LBF model: {model_path}")
    print(f"Exported {int(labels.sum())} keys: {keys_path}")
    print(f"Exported {int((~labels).sum())} non-keys: {nonkeys_path}")
    print(f"Threshold: {learned_filter.threshold:.17g}")
    return learned_filter


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Small, one-feature case mirroring test_json_round_trip_with_backup_filter.
    objects = np.arange(1, 10, dtype=np.float64).reshape(-1, 1)
    labels = np.array([True, False, False, False, False,
                       False, True, True, True], dtype=bool)

    simple_filter = build_and_export_case(output_dir, "lbf_toy", objects, labels)
    print(f"Predictions on small toy cases: {simple_filter.predict(objects).tolist()}")

    # Larger multi-feature case: 255 keys and 1024 non-keys, all explicit
    # float64. The overlapping distributions also exercise the backup filter.
    rng = np.random.default_rng(522812)
    key_cases = rng.normal(loc=0.5, scale=1.0, size=(255, 16))
    nonkey_cases = rng.normal(loc=-0.5, scale=1.0, size=(1024, 16))
    vector_objects = np.vstack((key_cases, nonkey_cases)).astype(np.float64)
    vector_labels = np.concatenate((np.ones(255, dtype=bool),
                                    np.zeros(1024, dtype=bool)))
    permutation = rng.permutation(len(vector_objects))
    vector_objects = vector_objects[permutation]
    vector_labels = vector_labels[permutation]
    vector_filter = build_and_export_case(output_dir, "lbf_toy_16d",
                                          vector_objects, vector_labels)
    predictions = vector_filter.predict(vector_objects)
    print(f"16-feature predictions: {int(predictions.sum())}/{len(predictions)} positive")


if __name__ == "__main__":
    main()