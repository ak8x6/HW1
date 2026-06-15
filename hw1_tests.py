"""
hw1_tests.py — the ONE support file for Homework 1.

There is no separate utils module. This single file holds everything the notebook
needs besides your own code:

  1. DATA & PLOTTING HELPERS used by the notebook
     (load_data, add_polynomial_features, plot_* ).
  2. SELF-CHECK unit tests for your Part 2 functions
     (sigmoid, predict_proba, bce_loss, accuracy).
  3. An ENVIRONMENT check (are the required packages importable?).
  4. The contest SUBMISSION-FORMAT validator (also runnable from the terminal).

The file is intentionally large — that's by design, so the homework folder stays
minimal (just the notebook, this file, requirements.txt, and the contest data).

--------------------------------------------------------------------------------
USING THE HELPERS (inside the notebook)
--------------------------------------------------------------------------------
    from hw1_tests import load_data, add_polynomial_features, plot_loss_curve

--------------------------------------------------------------------------------
USING THE SELF-CHECKS (inside the notebook)
--------------------------------------------------------------------------------
    from hw1_tests import check_sigmoid
    check_sigmoid(sigmoid)          # pass YOUR function object

    from hw1_tests import run_all
    run_all(globals())             # runs every check, prints a summary

--------------------------------------------------------------------------------
ENVIRONMENT CHECK
--------------------------------------------------------------------------------
    from hw1_tests import check_environment
    check_environment()

--------------------------------------------------------------------------------
VALIDATING A CONTEST SUBMISSION (terminal, when the contest opens)
--------------------------------------------------------------------------------
    python hw1_tests.py contest/submission.csv

Submission columns: example_id, prediction (0/1), probability (P(label=1) in [0,1]).
Exit code 0 = valid, 1 = invalid.
"""
from __future__ import annotations

import math

# torch/numpy are needed for the helpers and self-checks; the validator works
# without them. matplotlib is optional (plots are skipped if it's missing).
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
try:
    import torch
except Exception:  # pragma: no cover
    torch = None
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# ============================================================================
# 1. DATA & PLOTTING HELPERS  (used by the notebook; you don't edit these)
# ----------------------------------------------------------------------------
# DATASET (Part 2): MAGIC Gamma Telescope (UCI) — 19,020 events, 10 continuous
# features, binary target (gamma signal = 1 vs hadron background = 0). Bundled as
# data/magic04.data, so everything runs offline.
# ============================================================================
_MAGIC_FEATURES = ["fLength", "fWidth", "fSize", "fConc", "fConc1",
                   "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"]
_CLASSES = ["hadron", "gamma"]  # index 0 = hadron (h), 1 = gamma (g, signal)


def _load_raw(seed: int = 0):
    """Return (X, y, feature_names, classes).  X: (N,D) float32, y: (N,) 0/1.

    Reads the bundled MAGIC file (data/magic04.data): 10 numeric feature columns
    plus a class column ('g' = gamma -> 1, 'h' = hadron -> 0).
    """
    import os
    import pandas as pd

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "data", "magic04.data")
    if not os.path.exists(path):
        raise RuntimeError(f"MAGIC data not found at {path}")
    df = pd.read_csv(path, header=None, names=_MAGIC_FEATURES + ["class"])
    y = (df["class"].astype(str).str.strip() == "g").to_numpy(dtype=np.float32)
    X = df[_MAGIC_FEATURES].to_numpy(dtype=np.float32)
    return X, y, list(_MAGIC_FEATURES), list(_CLASSES)


def load_data(val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0,
              standardize: bool = True):
    """Load the dataset and split into train/val/test.

    Returns a dict of torch tensors (X_* and y_* with shape (N, 1)) plus
    feature_names, mean, std, and classes. The scaler is fit on TRAIN only
    (no leakage — see the lecture appendix).
    """
    X, y, feature_names, classes = _load_raw(seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test, n_val = int(len(X) * test_frac), int(len(X) * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    X_tr, X_va, X_te = X[train_idx], X[val_idx], X[test_idx]
    y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]
    mean, std = X_tr.mean(0), X_tr.std(0) + 1e-8
    if standardize:
        X_tr, X_va, X_te = (X_tr - mean) / std, (X_va - mean) / std, (X_te - mean) / std

    def t(a):
        return torch.tensor(a, dtype=torch.float32)

    return {
        "X_train": t(X_tr), "y_train": t(y_tr).reshape(-1, 1),
        "X_val": t(X_va), "y_val": t(y_va).reshape(-1, 1),
        "X_test": t(X_te), "y_test": t(y_te).reshape(-1, 1),
        "feature_names": list(feature_names), "mean": t(mean), "std": t(std),
        "classes": list(classes),
    }


def add_polynomial_features(X, degree: int = 2):
    """Append squared terms to X (used in the model-complexity ablation)."""
    feats = [X]
    if degree >= 2:
        feats.append(X ** 2)
    return torch.cat(feats, dim=1)


def plot_loss_curve(history, title: str = "Training loss"):
    """history: a list of per-epoch losses, or a dict label -> list."""
    if plt is None:
        print("matplotlib not available")
        return
    plt.figure(figsize=(6, 4))
    if isinstance(history, dict):
        for label, vals in history.items():
            plt.plot(vals, label=str(label))
        plt.legend()
    else:
        plt.plot(history)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(title)
    plt.grid(alpha=0.3); plt.show()


def plot_lr_sweep(results, title: str = "Learning-rate ablation"):
    """results: dict learning_rate -> final validation metric."""
    if plt is None:
        print("matplotlib not available")
        return
    lrs = sorted(results)
    plt.figure(figsize=(6, 4))
    plt.plot(lrs, [results[k] for k in lrs], marker="o")
    plt.xscale("log"); plt.xlabel("learning rate"); plt.ylabel("validation metric")
    plt.title(title); plt.grid(alpha=0.3); plt.show()


def plot_data_2d(X, y, title: str = "Data"):
    """Scatter a 2-feature dataset, colored by class. Tensors or arrays OK."""
    if plt is None:
        print("matplotlib not available")
        return
    Xn = np.asarray(X.detach() if hasattr(X, "detach") else X)
    yn = np.asarray(y.detach() if hasattr(y, "detach") else y).reshape(-1)
    if Xn.shape[1] != 2:
        print(f"plot_data_2d needs 2 features, got {Xn.shape[1]} — skipping")
        return
    plt.figure(figsize=(6, 5))
    for cls, mk in zip([0, 1], ["o", "^"]):
        m = yn == cls
        plt.scatter(Xn[m, 0], Xn[m, 1], s=12, alpha=0.6, marker=mk, label=f"class {cls}")
    plt.xlabel("feature 0"); plt.ylabel("feature 1"); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.show()


def plot_decision_boundary(predict_fn, X, y, title: str = "Decision boundary"):
    """Plot a model's decision boundary over a 2-feature dataset.

    predict_fn: callable taking an (M, 2) float tensor -> (M, 1) probabilities.
    Only works when X has exactly 2 columns (e.g. the placeholder dataset).
    """
    if plt is None:
        print("matplotlib not available")
        return
    Xn = np.asarray(X.detach() if hasattr(X, "detach") else X)
    yn = np.asarray(y.detach() if hasattr(y, "detach") else y).reshape(-1)
    if Xn.shape[1] != 2:
        print(f"plot_decision_boundary needs 2 features, got {Xn.shape[1]} — skipping")
        return
    x0min, x0max = Xn[:, 0].min() - 0.5, Xn[:, 0].max() + 0.5
    x1min, x1max = Xn[:, 1].min() - 0.5, Xn[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x0min, x0max, 200),
                         np.linspace(x1min, x1max, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        probs = predict_fn(grid).reshape(xx.shape).numpy()
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, probs, levels=20, alpha=0.3, cmap="coolwarm")
    plt.contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=1.5)
    for cls, mk in zip([0, 1], ["o", "^"]):
        m = yn == cls
        plt.scatter(Xn[m, 0], Xn[m, 1], s=12, alpha=0.7, marker=mk, label=f"class {cls}")
    plt.xlabel("feature 0"); plt.ylabel("feature 1"); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.show()


# ============================================================================
# 2. SELF-CHECKS  (Part 2 building blocks)
# ----------------------------------------------------------------------------
# Part 1 is informative only — it has no graded exercises, so there are no
# checks for it here.
# ============================================================================
GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def _result(name, ok, msg=""):
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"[{tag}] {name}" + (f"  -> {msg}" if msg else ""))
    return ok


def _guard(name, fn):
    if fn is None:
        _result(name, False, "function not found — did you define it?")
        return False
    return True


def _close(a, b, tol=1e-4):
    return torch.allclose(torch.as_tensor(a, dtype=torch.float32),
                          torch.as_tensor(b, dtype=torch.float32), atol=tol)


def check_sigmoid(sigmoid):
    name = "sigmoid"
    if not _guard(name, sigmoid):
        return False
    try:
        z = torch.tensor([-100.0, 0.0, 100.0, 1.0])
        out = sigmoid(z)
        exp = torch.tensor([0.0, 0.5, 1.0, 1 / (1 + math.exp(-1))])
        if out.shape != z.shape:
            return _result(name, False, f"shape {tuple(out.shape)} != {tuple(z.shape)}")
        if not _close(out, exp, tol=1e-4):
            return _result(name, False, f"values off: got {out.tolist()}")
        return _result(name, True)
    except NotImplementedError:
        return _result(name, False, "still a TODO stub")
    except Exception as e:  # noqa
        return _result(name, False, f"raised {type(e).__name__}: {e}")


def check_model(LinearClassifier):
    """Check the student's LinearClassifier (nn.Module): parameters + forward pass."""
    name = "LinearClassifier"
    if not _guard(name, LinearClassifier):
        return False
    try:
        import torch.nn as nn

        m = LinearClassifier(2)
        if not isinstance(m, nn.Module):
            return _result(name, False, "LinearClassifier should subclass torch.nn.Module")
        if not (hasattr(m, "w") and hasattr(m, "b")):
            return _result(name, False, "model needs parameters .w (n_features,1) and .b (1,)")
        if tuple(m.w.shape) != (2, 1):
            return _result(name, False, f".w should be (2,1) for n_features=2, got {tuple(m.w.shape)}")

        # set known parameters so the forward pass is deterministic
        m.w = nn.Parameter(torch.tensor([[0.5], [-0.5]]))
        m.b = nn.Parameter(torch.tensor([0.0]))
        X = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
        p = m(X)  # nn.Module __call__ -> forward
        expp = torch.sigmoid(torch.tensor([[-0.5], [0.0]]))
        if tuple(p.shape) != (2, 1):
            return _result(name, False, f"forward shape {tuple(p.shape)} != (2,1)")
        if not _close(p, expp):
            return _result(name, False, "forward(X) != sigmoid(X @ w + b)")
        return _result(name, True)
    except NotImplementedError:
        return _result(name, False, "still a TODO stub (model or sigmoid not implemented)")
    except Exception as e:  # noqa
        return _result(name, False, f"raised {type(e).__name__}: {e}")


def check_bce_loss(bce_loss):
    name = "bce_loss"
    if not _guard(name, bce_loss):
        return False
    try:
        y_pred = torch.tensor([[0.9], [0.1], [0.8]])
        y_true = torch.tensor([[1.0], [0.0], [1.0]])
        out = float(bce_loss(y_pred, y_true))
        exp = float(torch.nn.functional.binary_cross_entropy(y_pred, y_true))
        if not math.isclose(out, exp, abs_tol=1e-4):
            return _result(name, False, f"got {out:.4f}, expected mean BCE {exp:.4f}")
        hi = float(bce_loss(torch.tensor([[0.01]]), torch.tensor([[1.0]])))
        lo = float(bce_loss(torch.tensor([[0.99]]), torch.tensor([[1.0]])))
        if not hi > lo:
            return _result(name, False, "confident-wrong should cost more than confident-right")
        return _result(name, True)
    except NotImplementedError:
        return _result(name, False, "still a TODO stub")
    except Exception as e:  # noqa
        return _result(name, False, f"raised {type(e).__name__}: {e}")


def check_accuracy(accuracy):
    name = "accuracy"
    if not _guard(name, accuracy):
        return False
    try:
        y_pred = torch.tensor([[0.9], [0.2], [0.6], [0.4]])  # -> 1,0,1,0
        y_true = torch.tensor([[1.0], [0.0], [0.0], [0.0]])  # 3/4 correct
        out = float(accuracy(y_pred, y_true))
        if not math.isclose(out, 0.75, abs_tol=1e-6):
            return _result(name, False, f"got {out}, expected 0.75 (threshold at 0.5)")
        return _result(name, True)
    except NotImplementedError:
        return _result(name, False, "still a TODO stub")
    except Exception as e:  # noqa
        return _result(name, False, f"raised {type(e).__name__}: {e}")


def check_evaluate(evaluate):
    name = "evaluate"
    if not _guard(name, evaluate):
        return False
    try:
        import torch.nn as nn

        class _FixedModel(nn.Module):
            """Stand-in model returning fixed probabilities, so we can check that
            evaluate composes the student's bce_loss / accuracy correctly."""
            def forward(self, X):
                return torch.tensor([[0.9], [0.2], [0.6], [0.4]])

        y = torch.tensor([[1.0], [0.0], [0.0], [0.0]])           # -> preds 1,0,1,0 => 3/4
        out = evaluate(_FixedModel(), torch.zeros((4, 1)), y)
        if not isinstance(out, dict):
            return _result(name, False, "should return a dict {'loss':..., 'accuracy':...}")
        if "loss" not in out or "accuracy" not in out:
            return _result(name, False, f"dict needs keys 'loss' and 'accuracy', got {list(out)}")
        p = torch.tensor([[0.9], [0.2], [0.6], [0.4]])
        exp_loss = float(torch.nn.functional.binary_cross_entropy(p, y))
        if not math.isclose(float(out["loss"]), exp_loss, abs_tol=1e-4):
            return _result(name, False, f"loss {out['loss']} != mean BCE {exp_loss:.4f}")
        if not math.isclose(float(out["accuracy"]), 0.75, abs_tol=1e-6):
            return _result(name, False, f"accuracy {out['accuracy']} != 0.75")
        return _result(name, True)
    except NotImplementedError:
        return _result(name, False, "still a TODO stub (evaluate, or bce_loss/accuracy)")
    except Exception as e:  # noqa
        return _result(name, False, f"raised {type(e).__name__}: {e}")


_REGISTRY = {
    "sigmoid": check_sigmoid,
    "LinearClassifier": check_model,
    "bce_loss": check_bce_loss,
    "accuracy": check_accuracy,
    "evaluate": check_evaluate,
}


def run_all(namespace):
    """Pass `globals()` from the notebook. Looks up each expected Part 2 function
    by name, runs its check, and prints a summary."""
    if torch is None:
        print("PyTorch is not installed — run the install cell at the top of the "
              "notebook, then restart the kernel.")
        return False
    print("Running Homework 1 self-checks (Part 2 building blocks)...\n")
    passed = 0
    for fn_name, checker in _REGISTRY.items():
        fn = namespace.get(fn_name, None)
        if checker(fn):
            passed += 1
    total = len(_REGISTRY)
    bar = GREEN if passed == total else YELLOW
    print(f"\n{bar}{passed}/{total} checks passed{RESET}")
    return passed == total


# ============================================================================
# 3. ENVIRONMENT CHECK
# ============================================================================
def check_environment():
    """Print which required packages are importable. Returns True if all present."""
    import importlib
    ok = True
    for name in ["numpy", "pandas", "matplotlib", "sklearn", "torch"]:
        try:
            m = importlib.import_module(name)
            print(f"OK  {name:12s} {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"MISSING  {name}  <- run the install cell at the top of the notebook")
            ok = False
    return ok


# ============================================================================
# 4. CONTEST SUBMISSION VALIDATOR  (Python function + command line)
# ============================================================================
EXPECTED_HEADER = ["example_id", "prediction", "probability"]


def validate_submission(path, test_file=None, verbose=True):
    """Validate a contest submission CSV. Returns True if valid, else False.

    Format (see contest/sample_submission.csv):
        example_id,prediction,probability
        0,1,0.87
        1,0,0.12
    * Header must be exactly:  example_id,prediction,probability
    * example_id  : integer, unique, no missing rows.
    * prediction  : the hard 0/1 label (an integer, exactly 0 or 1).
    * probability : P(label = 1), a float in [0, 1].

    If data/contest_test.csv is available (or passed as test_file), every
    example_id is cross-checked against it (no missing or extra rows).
    """
    import csv
    import os

    def fail(msg):
        if verbose:
            print(f"INVALID: {msg}")
        return False

    if not os.path.exists(path):
        return fail(f"file not found: {path}")

    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return fail("file is empty")

    header = [h.strip() for h in rows[0]]
    if header != EXPECTED_HEADER:
        return fail(f"header must be {EXPECTED_HEADER}, got {header}")

    body = rows[1:]
    if not body:
        return fail("no data rows")

    ids = []
    for i, row in enumerate(body, start=2):
        if len(row) != 3:
            return fail(f"line {i}: expected 3 columns, got {len(row)}")
        id_str, pred_str, prob_str = (c.strip() for c in row)
        # example_id : integer
        try:
            ids.append(int(id_str))
        except ValueError:
            return fail(f"line {i}: example_id '{id_str}' is not an integer")
        # prediction : integer-valued 0 or 1
        try:
            pred = float(pred_str)
        except ValueError:
            return fail(f"line {i}: prediction '{pred_str}' is not a number")
        if not pred.is_integer() or int(pred) not in (0, 1):
            return fail(f"line {i}: prediction {pred_str} must be the integer 0 or 1")
        # probability : float in [0, 1]
        try:
            prob = float(prob_str)
        except ValueError:
            return fail(f"line {i}: probability '{prob_str}' is not a number")
        if not (0.0 <= prob <= 1.0):
            return fail(f"line {i}: probability {prob_str} must be between 0 and 1")

    if len(ids) != len(set(ids)):
        return fail("duplicate example_ids found")

    if test_file is None:
        guess = os.path.join(os.path.dirname(os.path.abspath(path)),
                             "..", "data", "contest_test.csv")
        test_file = guess if os.path.exists(guess) else None

    matched = False
    if test_file and os.path.exists(test_file):
        with open(test_file, newline="") as f:
            tr = list(csv.reader(f))
        thead = [h.strip() for h in tr[0]]
        if "example_id" not in thead:
            if verbose:
                print(f"WARNING: {test_file} has no 'example_id' column; skipping cross-check")
        else:
            col = thead.index("example_id")
            test_ids = {int(r[col]) for r in tr[1:] if r}
            sub_ids = set(ids)
            missing, extra = test_ids - sub_ids, sub_ids - test_ids
            if missing:
                return fail(f"{len(missing)} ids from the test set are missing "
                            f"(e.g. {sorted(missing)[:5]})")
            if extra:
                return fail(f"{len(extra)} ids not in the test set "
                            f"(e.g. {sorted(extra)[:5]})")
            matched = True

    if verbose:
        print(f"VALID: {len(ids)} rows, header OK, example_ids unique"
              + (" and matched to test set" if matched else ""))
    return True


def _cli():
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Validate a Homework 1 contest submission.")
    ap.add_argument("path", help="path to your submission .csv")
    ap.add_argument("--test-file", default=None,
                    help="path to contest_test.csv to cross-check example_ids")
    args = ap.parse_args()
    ok = validate_submission(args.path, test_file=args.test_file)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _cli()
