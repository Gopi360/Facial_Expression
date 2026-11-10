from pathlib import Path

from training.split import split_by_run


def _paths(runs_per_label=5, files_per_run=10, labels=("clear", "covered")):
    out = []
    for li, label in enumerate(labels):
        for r in range(runs_per_label):
            run_id = f"2026111{li}-1000{r:02d}"
            for n in range(files_per_run):
                out.append(Path(f"dataset/ears/{label}/{run_id}_{n:05d}_left.png"))
    return out


def _run(p):
    return p.name.split("_")[0]


def test_runs_never_split():
    train, test = split_by_run(_paths())
    assert {_run(p) for p in train}.isdisjoint({_run(p) for p in test})


def test_test_share_reached():
    paths = _paths(runs_per_label=5)  # 10 runs of 10 files
    train, test = split_by_run(paths)
    assert len(test) >= 20
    assert len(train) + len(test) == len(paths)


def test_each_label_tested():
    train, test = split_by_run(_paths(runs_per_label=2))
    assert {p.parent.name for p in test} == {"clear", "covered"}


def test_same_seed_same_split():
    paths = _paths()
    assert split_by_run(paths, seed=3) == split_by_run(paths, seed=3)


def test_run_shared_by_both_labels_not_split():
    paths = _paths()
    shared = "20260101-000000"
    paths += [Path(f"dataset/ears/clear/{shared}_{n:05d}_left.png") for n in range(5)]
    paths += [Path(f"dataset/ears/covered/{shared}_{n:05d}_right.png") for n in range(5)]
    for seed in range(5):
        train, test = split_by_run(paths, seed=seed)
        assert not ({_run(p) for p in train} & {_run(p) for p in test})
