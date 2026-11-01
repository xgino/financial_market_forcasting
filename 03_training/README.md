# 03_training: train your own models

Every player trains one of your own models, walk-forward: one checkpoint per cutoff date, each trained only
on results that had finished by that date. Tuning (GridSearchCV) happens inside the training data only.

- `gbm.py` gradient boosting on all assets together. Notebook style: run it cell by cell in VS Code,
  or as part of `python main.py 3`. Watch it live in the dashboard (page Training).

Add a trainer: a file with `SETTINGS` and a class based on `Trainer` with `train(tl, progress, fresh)`.
Save checkpoints in `data/checkpoints/<name>/` with an `index.csv`, write progress to
`data/training/<name>/log.csv`. Then add a model file with the same name in `04_models/` that loads them.