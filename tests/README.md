# tests

`python main.py test` runs every `test_*` function in here, on fake random prices (no download needed).
- `test_core.py` timeline hides the future, wallet can't be edited, every model file loads, scores reward skill
- `test_models.py` no model and no trainer can see the future
On random prices no model can have skill, so "nothing found" is the right answer there.