# -*- coding: utf-8 -*-

from __future__ import annotations

# ------------------------------------------------------------------ #
# 0. IMPORTS                                                         #
# ------------------------------------------------------------------ #

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score

# ------------------------------------------------------------------ #
# 1. USER CONFIGURATION                                              #
# ------------------------------------------------------------------ #

CSV = Path("mechanical_feasibility_dataset.csv")      # Insert dataset file name

TARGET_COLUMN = "Mechanical output"  # Target column name
POSITIVE_LABEL = "Success"           # Positive class label

# Range of regularization parameters to test
Cs = np.logspace(-3, 3, 50)

# ------------------------------------------------------------------ #
# 2. LOAD DATASET                                                    #
# ------------------------------------------------------------------ #

if not CSV.is_file():
    raise FileNotFoundError(f"{CSV.resolve()} missing")

df = pd.read_csv(CSV, sep=";")

# Clean labels
df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(str)
    .str.strip()
)

# Binary target
df["Target"] = (
    df[TARGET_COLUMN]
    .str.lower()
    .eq(POSITIVE_LABEL.lower())
    .astype(int)
)

# Automatically detect feature columns
FEATURE_COLUMNS = [
    c for c in df.columns
    if c not in [TARGET_COLUMN, "Target"]
]

# Ensure ternary system
if len(FEATURE_COLUMNS) != 3:
    raise ValueError(
        f"Expected exactly 3 feature columns, got {len(FEATURE_COLUMNS)}"
    )

# Feature matrix and target vector
X = df[FEATURE_COLUMNS].to_numpy(float)
y = df["Target"].to_numpy(int)

print("\nDataset loaded")
print(f"Dataset : {CSV.name}")
print(f"Features: {FEATURE_COLUMNS}")
print(f"Samples : {len(df)}")
print(f"Positive samples: {(y == 1).sum()}")
print(f"Negative samples: {(y == 0).sum()}")

# ------------------------------------------------------------------ #
# 3. LOOCV SEARCH FOR BEST C (λ)                                     #
# ------------------------------------------------------------------ #

loo = LeaveOneOut()

mean_scores = []

print("\nScanning regularization parameter\n")

for C in Cs:

    model = LogisticRegression(
        class_weight="balanced",
        penalty="l1",
        solver="liblinear",
        random_state=1,
        max_iter=1000,
        C=C,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=loo,
        scoring="accuracy",
    )

    mean_score = scores.mean()
    mean_scores.append(mean_score)

    print(
        f"C = {C:10.5g}   "
        f"λ = {1/C:10.5g}   "
        f"LOOCV Accuracy = {mean_score:.4f}"
    )


# ------------------------------------------------------------------ #
# 5. PLOT                                                            #
# ------------------------------------------------------------------ #

lambdas = 1 / Cs

plt.figure(figsize=(8, 5))

plt.semilogx(
    lambdas,
    mean_scores,
    marker="o",
    linewidth=2,
)

plt.xlabel(r"Regularization strength ($\lambda$)")
plt.ylabel("LOOCV accuracy")

plt.grid(True)

plt.tight_layout()
plt.show()