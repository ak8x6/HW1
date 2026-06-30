# Logistic Regression from Scratch — PyTorch Implementation

> **Completed by [Ahmad Kassem](https://github.com/ak8x6)** as part of the AI Tech Fellow program at [LebNet](https://github.com/TechFellows26) (TechFellows26).

---

## 📌 Overview

This project implements **binary logistic regression from scratch using PyTorch**, following the core ML pipeline: **model → loss → optimization → generalization**. It includes a full implementation of the training loop, custom loss functions, and a prediction contest submission.

## 🧠 What I Built

### Part 1 — PyTorch Fundamentals
- Explored tensors, vectorized operations, and automatic differentiation (`autograd`)
- Understood the computational graph and gradient flow in PyTorch

### Part 2 — Logistic Regression from Scratch
- Implemented the **sigmoid activation function**
- Built **binary cross-entropy (BCE) loss** from scratch
- Created a `LinearClassifier` as a custom `nn.Module`
- Wrote the full **gradient descent training loop**
- Conducted a **learning-rate ablation study** with loss curve analysis
- Evaluated model performance with accuracy metrics on held-out test data

### Part 3 — Prediction Contest
- Trained and optimized a classifier on the provided contest dataset
- Generated predictions and produced a validated `submission.csv`

## 📊 Dataset

- **MAGIC Gamma Telescope** — 19,020 events, 10 numeric features, binary classification (gamma vs. hadron)
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Core language |
| **PyTorch** | Model building, autograd, tensor operations |
| **NumPy** | Numerical computing |
| **Pandas** | Data loading and manipulation |
| **Matplotlib** | Visualization and loss curve plotting |
| **scikit-learn** | Data preprocessing and evaluation utilities |

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/ak8x6/logistic-regression-pytorch.git
cd logistic-regression-pytorch

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook homework1.ipynb
```

> **Note:** No GPU required — everything runs on CPU in seconds.

## 📁 Project Structure

```
├── homework1.ipynb           ← Main notebook with all solutions
├── homework1.pdf             ← Read-only PDF version of the assignment
├── hw1_tests.py              ← Unit tests and contest submission validator
├── requirements.txt          ← Python dependencies
├── reading/
│   ├── Session_02_Summary.pdf
│   └── 220-logistic-regression.pdf
├── data/
│   ├── magic04.data          ← MAGIC Gamma Telescope dataset
│   ├── contest_train.csv     ← Contest training data
│   └── contest_test.csv      ← Contest test set (labels hidden)
└── contest/
    ├── sample_submission.csv ← Expected submission format
    └── submission.csv        ← My contest predictions
```

## 🏷️ Attribution

This project is a fork of the original assignment repository by **[TechFellows26](https://github.com/TechFellows26/HW1)** (LebNet). The assignment structure, datasets, and test suite were provided by the course instructors. All solutions and implementations are my own work.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
