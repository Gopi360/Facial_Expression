"""Split ear photos into train and test sides by run, so near-identical burst photos never leak across."""
import random
from collections import defaultdict
from pathlib import Path


def run_id(path: Path) -> str:
    return path.name.split("_")[0]


def split_by_run(
    paths: list[Path], test_share: float = 0.2, seed: int = 0
) -> tuple[list[Path], list[Path]]:
    """Return (train, test). A run's files all go to one side; the label is the parent folder name."""
    runs: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        runs[run_id(p)].append(p)

    order = sorted(runs)
    random.Random(seed).shuffle(order)

    labels_runs: dict[str, set[str]] = defaultdict(set)
    for rid, files in runs.items():
        for p in files:
            labels_runs[p.parent.name].add(rid)

    test_runs: list[str] = []
    # First make sure every label that has two or more runs is tested.
    for label in sorted(labels_runs):
        if len(labels_runs[label]) < 2:
            continue
        if any(r in test_runs for r in labels_runs[label]):
            continue
        test_runs.append(next(r for r in order if r in labels_runs[label]))

    wanted = test_share * len(paths)
    count = sum(len(runs[r]) for r in test_runs)
    for rid in order:
        if count >= wanted:
            break
        if rid not in test_runs:
            test_runs.append(rid)
            count += len(runs[rid])

    test_set = set(test_runs)
    train = [p for p in paths if run_id(p) not in test_set]
    test = [p for p in paths if run_id(p) in test_set]
    return train, test
