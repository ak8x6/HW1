# Homework 1 — The ML Framework

This homework accompanies **Session 02: The ML Framework**. You will move from a
Python warm-up to building **logistic regression from scratch in PyTorch**, and
finish with a **prediction contest**.

## 📘 The assignment

Do the work in the notebook **[`homework1.ipynb`](homework1.ipynb)** — that's the
assignment. A read-only **[`homework1.pdf`](homework1.pdf)** copy is included for
easy reading/printing, but you submit/run the notebook.

---

## Setup

```bash
# from this folder (final_lectures/hw1)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook homework1.ipynb
```

You do **not** need a GPU. Everything runs on CPU in seconds.

**Dataset (Part 2).** The [MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)
set (19,020 events, 10 numeric features, binary gamma-vs-hadron label), bundled as
`data/magic04.data` — everything runs offline. The **Part 3 contest** uses a
separate dataset (TBD); its files are added to `data/` when released.

---

## Structure

| Part | Topic | What you do |
|------|-------|-------------|
| **Part 1** | PyTorch basics | What PyTorch adds over plain Python: tensors, vectorization, autograd. Assumes you can already program (new to Python? see the [Google Python tutorial](https://developers.google.com/edu/python)). |
| **Part 2** | Logistic regression from scratch | Build a `LinearClassifier` (`nn.Module`), implement sigmoid, BCE loss, accuracy, evaluate, and the gradient-descent loop, then run a **learning-rate ablation** (loss curves for different learning rates). |
| **Part 3** | Contest | Train your best model and produce a `submission.csv` we will score. |

The folder is intentionally minimal — the notebook is **self-contained** (it
defines its own data/plotting helpers, so there is no `utils.py` to import):

```
hw1/
├── homework1.ipynb           <- the assignment (work here)
├── homework1.pdf             <- read-only PDF copy of the assignment
├── hw1_tests.py              <- the only support file: self-checks, an
│                                environment check, and the contest validator
├── requirements.txt          <- packages to install
├── reading/                  <- start here: Session 02 summary + logistic-regression reference
├── data/
│   ├── magic04.data          <- Part 2 dataset (MAGIC), bundled
│   ├── contest_train.csv     <- Part 3 training data (added when released)
│   └── contest_test.csv      <- Part 3 test set, labels hidden (added when released)
└── contest/
    └── sample_submission.csv <- example of the expected submission format
```

---

## Self-checking your work

Throughout Part 2 you will see cells like:

```python
from hw1_tests import check_sigmoid
check_sigmoid(sigmoid)        # pass YOUR function
```

These run lightweight unit tests against your implementation and print
`PASS` / `FAIL` with a hint. You can also run everything at once at the end:

```python
from hw1_tests import run_all
run_all(globals())
```

The tests check **correctness of the building blocks** (shapes, known values,
edge cases). Passing them does not guarantee a high contest score — that depends
on your modeling choices.

---

## Part 3 — Contest (dataset TBD)

You may use **any model and any library** (your Part 2 model, scikit-learn, …) —
you're graded on predictions, not implementation. When the contest dataset is
released you'll get two files in `data/`:

- `contest_train.csv` — labeled training data (a `label` column + feature columns).
- `contest_test.csv` — the test set: an `example_id` column + the same features,
  with **labels hidden**.

You will train a model, predict a label for every `example_id`, and save a CSV with
columns **`example_id,prediction`** to `contest/submission.csv` (see
`contest/sample_submission.csv`). Validate the format before submitting:

```bash
python hw1_tests.py contest/submission.csv
```

A submission that does not pass the validator will not be scored. The exact metric,
deadline, and leaderboard will be announced with the dataset.
