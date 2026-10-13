# mainbreak

Which water mains will burst in the next twelve months.

The network has 8000 mains. About one in ten bursts in a given year, and every
burst means a road dug up and a street without water. The crews can inspect a
limited number of mains ahead of time, so they need a ranked list.

## Layout

- `data/mains.csv` is the table, one row per main.
- `mainbreak/data.py` loads it and makes the fixed train/test split.
- `mainbreak/pipeline.py` builds a pipeline for one of three families: `logreg`,
  `rf` or `hgb`.
- `mainbreak/evaluate.py` holds the two scores every run is judged by.
- `mainbreak/train.py` trains the chosen model and writes `artifacts/`.
- `configs/sweep.json` lists the configurations to compare.

## Training

```
python3 -m mainbreak.train
```

This writes `artifacts/model.joblib` and `artifacts/model_meta.json`. Running it
again overwrites both, so the project only ever remembers its latest run.
