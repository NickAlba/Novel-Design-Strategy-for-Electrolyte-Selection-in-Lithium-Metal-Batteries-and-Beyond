# -*- coding: utf-8 -*-

from __future__ import annotations

# ------------------------------------------------------------------ #
# 0. IMPORTS & STYLE                                                 #
# ------------------------------------------------------------------ #

import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, classification_report

from pysr import PySRRegressor
import sympy as sp

# ------------------------------------------------------------------ #
# STYLE                                                              #
# ------------------------------------------------------------------ #

cbf = [
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#D55E00",
    "#000000",
]

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cbf)

plt.rc("text", usetex=True)
plt.rc("font", family="sans", size=18)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)

CLR_SUCC = "#000000"
CLR_FAIL = "#FF0000"

# ------------------------------------------------------------------ #
# LATEX-SAFE LABELS                                                  #
# ------------------------------------------------------------------ #

def latex_safe(text: str) -> str:
    """
    Escape LaTeX-sensitive characters for plotting labels.
    """

    replacements = {
        "%": r"\%",
        "_": r"\_",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{", 
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

# ------------------------------------------------------------------ #
# 1. USER CONFIGURATION                                              #
# ------------------------------------------------------------------ #

CSV = Path("mechanical_feasibility_dataset.csv")     #insert dataset file name

TARGET_COLUMN = "Mechanical output"    #insert label of TARGET column (e.g., "Mechanical output")
POSITIVE_LABEL = "Success"  #insert label of POSITIVE output in TARGET column (e.g., "Success")

TOTAL_COMPOSITION = 100

GRID_POINTS = 351
GRID_PADDING = 5

# ------------------------------------------------------------------ #
# 2. OUTPUT DIRECTORY                                                #
# ------------------------------------------------------------------ #

dataset_name = CSV.stem

OUTPUT_DIR = Path("outputs") / dataset_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nOutput directory: {OUTPUT_DIR.resolve()}")

# ------------------------------------------------------------------ #
# 3. LOAD DATASET                                                    #
# ------------------------------------------------------------------ #

if not CSV.is_file():
    raise FileNotFoundError(f"{CSV.resolve()} missing")

df = pd.read_csv(CSV, sep=";")

# Clean original labels
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

# Automatic class labels
positive_class_label = POSITIVE_LABEL

negative_candidates = [
    v for v in df[TARGET_COLUMN].unique()
    if v.lower() != POSITIVE_LABEL.lower()
]

if len(negative_candidates) != 1:
    raise ValueError(
        "Could not uniquely determine negative class label."
    )

negative_class_label = negative_candidates[0]

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

X1_NAME, X2_NAME, X3_NAME = FEATURE_COLUMNS

# LaTeX-safe labels
X1_LABEL = latex_safe(X1_NAME)
X2_LABEL = latex_safe(X2_NAME)
X3_LABEL = latex_safe(X3_NAME)



# Feature matrix
X = df[FEATURE_COLUMNS].to_numpy(float)

# Target vector
y = df["Target"].to_numpy(int)

print("\nDataset loaded")
print(f"Dataset: {CSV.name}")
print(f"Features: {FEATURE_COLUMNS}")
print(f"Samples: {len(df)}")
print(f"{positive_class_label}: " f"{(df['Target'] == 1).sum()}")
print(f"{negative_class_label}: " f"{(df['Target'] == 0).sum()}")

# ------------------------------------------------------------------ #
# 4. TRAIN BASELINE MODELS                                           #
# ------------------------------------------------------------------ #

models = {
    "Logistic regression": LogisticRegression(
        class_weight="balanced",
        penalty="l1",
        solver="liblinear",
        C=100,
        random_state=1,
        max_iter=1000,
    ),

    "SVM-RBF": SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        class_weight="balanced",
    ),

    "QDA": QDA(reg_param=1e-3),
}

loo = LeaveOneOut()

print("\nBalanced accuracy (LOOCV)")

timings = {}

for name, mdl in models.items():

    start = time.perf_counter()

    y_pred = cross_val_predict(mdl, X, y, cv=loo)

    bacc = balanced_accuracy_score(y, y_pred)

    mdl.fit(X, y)

    elapsed = time.perf_counter() - start

    timings[name] = elapsed

    print(
        f"{name:22s}: "
        f"{bacc:.3f}   "
        f"[Computing time = {elapsed:.2f} s]"
    )

# ------------------------------------------------------------------ #
# 5. SYMBOLIC REGRESSION                                             #
# ------------------------------------------------------------------ #

print("\nFitting PySR...")

start = time.perf_counter()

# Recast target to [-1, +1]
y_pm1 = y * 2 - 1

sym = PySRRegressor(
    niterations=250,
    population_size=50,
    procs=4,

    elementwise_loss="L2MarginLoss()",

    model_selection="best",

    maxsize=12,
    parsimony=1e-2,

    random_state=1,

    binary_operators=["+", "-", "*"],
    unary_operators=[],

    progress=False,
    verbosity=0,
)

with warnings.catch_warnings():

    warnings.filterwarnings("ignore", category=UserWarning)

    sym.fit(X, y_pm1)

elapsed = time.perf_counter() - start

timings["Symbolic regression"] = elapsed

expr = sym.get_best()["sympy_format"]

expr_pretty = (
    str(expr)
    .replace("x0", X1_NAME)
    .replace("x1", X2_NAME)
    .replace("x2", X3_NAME)
)

print("\nBest symbolic expression:")
print("g(x) =", expr_pretty)

# Symbolic -> numpy
x0, x1, x2 = sp.symbols("x0 x1 x2")

g_np = sp.lambdify(
    (x0, x1, x2),
    expr,
    modules="numpy"
)

def pysr_proba(v: np.ndarray) -> float:

    g = np.clip(g_np(*v), -6, 6)

    return 1 / (1 + np.exp(-g))

models["Symbolic regression"] = sym

# ------------------------------------------------------------------ #
# 6. ADAPTIVE GRID                                                   #
# ------------------------------------------------------------------ #

x_min = max(0, df[X1_NAME].min() - GRID_PADDING)
x_max = min(TOTAL_COMPOSITION, df[X1_NAME].max() + GRID_PADDING)

y_min = max(0, df[X2_NAME].min() - GRID_PADDING)
y_max = min(TOTAL_COMPOSITION, df[X2_NAME].max() + GRID_PADDING)

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, GRID_POINTS),
    np.linspace(y_min, y_max, GRID_POINTS)
)

flat2 = np.c_[xx.ravel(), yy.ravel()]

# Reconstruct third component
x3 = TOTAL_COMPOSITION - flat2[:, 0] - flat2[:, 1]

# Valid compositional region
mask = (
    (x3 >= 0)
    &
    (x3 <= TOTAL_COMPOSITION)
)

flat3 = np.c_[flat2, x3]

# ------------------------------------------------------------------ #
# 7. PROBABILITY FUNCTION                                            #
# ------------------------------------------------------------------ #

def proba(model_name: str, X3: np.ndarray) -> np.ndarray:

    if model_name == "Symbolic regression":

        return np.fromiter(
            (pysr_proba(v) for v in X3),
            dtype=float,
            count=len(X3)
        )

    return models[model_name].predict_proba(X3)[:, 1]

# ------------------------------------------------------------------ #
# 8. FIGURE 1 - RAW DATA                                             #
# ------------------------------------------------------------------ #

fig1, ax1 = plt.subplots(figsize=(5.5, 5.5))

ax1.scatter(
    df[X1_NAME],
    df[X2_NAME],

    c=df["Target"].map({
        1: CLR_SUCC,
        0: CLR_FAIL
    }),

    s=70,
    marker="o",
)

ax1.set_xlabel(X1_LABEL)
ax1.set_ylabel(X2_LABEL)

ax1.set_aspect("equal", adjustable="box")

ax1.grid(ls=":", alpha=0.4)

ax1.legend(handles=[

    Line2D(
        [],
        [],
        color=CLR_SUCC,
        marker='o',
        linestyle='',
        label=positive_class_label,
        markersize=10
    ),

    Line2D(
        [],
        [],
        color=CLR_FAIL,
        marker='o',
        linestyle='',
        label=negative_class_label,
        markersize=10
    )
])

plt.tight_layout()

fig1.savefig(
    OUTPUT_DIR / "figure1_raw_data.png",
    dpi=300,
    bbox_inches="tight"
)

# ------------------------------------------------------------------ #
# 9. FIGURE 2 - DECISION BOUNDARIES                                  #
# ------------------------------------------------------------------ #

fig2, ax2 = plt.subplots(figsize=(6, 6))

ax2.scatter(
    df[X1_NAME],
    df[X2_NAME],

    c=df["Target"].map({
        1: CLR_SUCC,
        0: CLR_FAIL
    }),

    s=70,
    marker="o"
)

styles = {
    "Logistic regression": "dashdot",
    "SVM-RBF": "solid",
    "QDA": "dotted",
    "Symbolic regression": "dashed",
}

legend_handles = []

for i, (name, ls) in enumerate(styles.items()):

    Z = np.full(xx.size, np.nan)

    Z[mask] = proba(name, flat3[mask])

    ax2.contour(
        xx,
        yy,
        Z.reshape(xx.shape),

        levels=[0.5],

        colors=cbf[i],
        linewidths=2,
        linestyles=ls,
    )

    legend_handles.append(

        Line2D(
            [],
            [],
            color=cbf[i],
            ls=ls,
            lw=2,
            label=f"{name}  $p=0.5$"
        )
    )

ax2.set_xlabel(X1_LABEL)
ax2.set_ylabel(X2_LABEL)

ax2.set_aspect('equal', adjustable='box')

ax2.grid(ls=":", alpha=.4)

ax2.legend(
    handles=[

        *legend_handles,

        Line2D(
            [],
            [],
            color=CLR_SUCC,
            marker='o',
            linestyle='',
            label=positive_class_label,
            markersize=8
        ),

        Line2D(
            [],
            [],
            color=CLR_FAIL,
            marker='o',
            linestyle='',
            label=negative_class_label,
            markersize=8
        ),
    ],

    fontsize=14
)

plt.tight_layout()

fig2.savefig(
    OUTPUT_DIR / "figure2_decision_boundaries.png",
    dpi=300,
    bbox_inches="tight"
)

# ------------------------------------------------------------------ #
# 10. FIGURE 3 - SVM PROBABILITY FIELD                               #
# ------------------------------------------------------------------ #

best = "SVM-RBF"

Z = np.full(xx.size, np.nan)

Z[mask] = proba(best, flat3[mask])

fig3, ax3 = plt.subplots(figsize=(6, 6))

cf = ax3.contourf(
    xx,
    yy,
    Z.reshape(xx.shape),

    levels=np.linspace(0, 1, 21),

    cmap="Blues",

    vmin=0,
    vmax=1,

    alpha=.85
)

fig3.colorbar(
    cf,
    ax=ax3,
    label=rf"$P_{{\rm succ}}$ ({best})",
    shrink=0.82
)

ax3.scatter(
    df[X1_NAME],
    df[X2_NAME],

    c=df["Target"].map({
        1: CLR_SUCC,
        0: CLR_FAIL
    }),

    s=70,
    marker="o"
)

ax3.set_xlabel(X1_LABEL)
ax3.set_ylabel(X2_LABEL)

ax3.set_aspect('equal', adjustable='box')

ax3.grid(ls=":", alpha=.4)

plt.tight_layout()

fig3.savefig(
    OUTPUT_DIR / "figure3_probability_field.png",
    dpi=300,
    bbox_inches="tight"
)

# ------------------------------------------------------------------ #
# 11. FIGURE 4 - SYMBOLIC SCORE MAP                                  #
# ------------------------------------------------------------------ #

Zg_raw = np.full(xx.size, np.nan)

Zg_raw[mask] = g_np(
    flat3[mask, 0],
    flat3[mask, 1],
    flat3[mask, 2]
)

lims = 2.0

Zg = np.clip(Zg_raw, -lims, lims)

fig4, ax4 = plt.subplots(figsize=(6, 6))

cg = ax4.contourf(
    xx,
    yy,
    Zg.reshape(xx.shape),

    levels=np.linspace(-lims, lims, 41),

    cmap="PuOr",

    vmin=-lims,
    vmax=lims,

    alpha=.9
)

fig4.colorbar(
    cg,
    ax=ax4,

    label=r"Symbolic regression raw score $g(x)$",

    shrink=0.82,

    ticks=np.arange(-2.0, 2.1, 0.5)
)

# Auxiliary contours
ax4.contour(
    xx,
    yy,
    Zg.reshape(xx.shape),

    levels=[-1, 0, 1],

    colors=["grey", "k", "grey"],

    linewidths=[1.2, 2.0, 1.2],

    linestyles=["dashed", "solid", "dashed"]
)

# Data points
ax4.scatter(
    df[X1_NAME],
    df[X2_NAME],

    c=df["Target"].map({
        1: CLR_SUCC,
        0: CLR_FAIL
    }),

    s=70,
    marker="o"
)

ax4.set_xlabel(X1_LABEL)
ax4.set_ylabel(X2_LABEL)

ax4.set_aspect('equal', adjustable='box')

ax4.grid(ls=":", alpha=.4)

ax4.legend(handles=[

    Line2D(
        [],
        [],
        color=CLR_SUCC,
        marker='o',
        linestyle='',
        label=positive_class_label,
        markersize=8
    ),

    Line2D(
        [],
        [],
        color=CLR_FAIL,
        marker='o',
        linestyle='',
        label=negative_class_label,
        markersize=8
    ),

    Line2D(
        [],
        [],
        color='k',
        ls='-',
        lw=2,
        label=r"$g(x)=0$"
    ),

    Line2D(
        [],
        [],
        color='grey',
        ls='--',
        lw=1.2,
        label=r"$g(x)=\pm1$"
    )

], loc="upper right", fontsize=11)

plt.tight_layout()

fig4.savefig(
    OUTPUT_DIR / "figure4_symbolic_score.png",
    dpi=300,
    bbox_inches="tight"
)

# ------------------------------------------------------------------ #
# 12. FIGURE 5 - SYMBOLIC SCORE ONLY                                 #
# ------------------------------------------------------------------ #

fig5, ax5 = plt.subplots(figsize=(6, 6))

cg2 = ax5.contourf(
    xx,
    yy,
    Zg.reshape(xx.shape),

    levels=np.linspace(-lims, lims, 41),

    cmap="PuOr",

    vmin=-lims,
    vmax=lims,

    alpha=.9
)

fig5.colorbar(
    cg2,
    ax=ax5,

    label=r"$g(x)$",

    shrink=0.82,

    ticks=np.arange(-2.0, 2.1, 0.5)
)

ax5.contour(
    xx,
    yy,
    Zg.reshape(xx.shape),

    levels=[0],

    colors="k",

    linewidths=2
)

ax5.set_xlabel(X1_LABEL)
ax5.set_ylabel(X2_LABEL)

ax5.set_aspect('equal', adjustable='box')

ax5.grid(ls=":", alpha=.4)

plt.tight_layout()

fig5.savefig(
    OUTPUT_DIR / "figure5_symbolic_score_only.png",
    dpi=300,
    bbox_inches="tight"
)

# ------------------------------------------------------------------ #
# 13. SHOW FIGURES                                                   #
# ------------------------------------------------------------------ #

plt.show()

# ------------------------------------------------------------------ #
# 14. METRIC SUMMARY                                                 #
# ------------------------------------------------------------------ #

print("\nMetric definitions")

print(" BalAcc : mean of per-class recalls")
print(" Prec   : TP / (TP + FP)")
print(" Recall : TP / (TP + FN)")
print(" F1     : harmonic mean of Precision and Recall")

print("\nMacro-averaged precision / recall / F1")

hdr = (
    f"{'Model':22s} | "
    f"BalAcc  Prec  Recall   F1   Computing time[s]"
)

print(hdr)

print("-" * len(hdr))

with warnings.catch_warnings():

    warnings.filterwarnings("ignore", category=UserWarning)

    for name in [
        "Logistic regression",
        "SVM-RBF",
        "QDA",
        "Symbolic regression"
    ]:

        if name == "Symbolic regression":

            yhat = (
                g_np(
                    X[:, 0],
                    X[:, 1],
                    X[:, 2]
                ) > 0
            ).astype(int)

        else:

            yhat = cross_val_predict(
                models[name],
                X,
                y,
                cv=loo
            )

        rep = classification_report(
            y,
            yhat,

            labels=[0, 1],

            output_dict=True,

            zero_division=0
        )["macro avg"]

        bal = balanced_accuracy_score(y, yhat)

        elapsed = timings[name]

        print(
            f"{name:22s} | "
            f"{bal:6.3f}  "
            f"{rep['precision']:.3f}  "
            f"{rep['recall']:.3f}  "
            f"{rep['f1-score']:.3f}   "
            f"{elapsed:7.4f}"
        )

# ------------------------------------------------------------------ #
# 15. IDENTIFY BEST MODEL                                            #
# ------------------------------------------------------------------ #

best_name = None
best_bal = -1

for m in [
    "Logistic regression",
    "SVM-RBF",
    "QDA",
    "Symbolic regression"
]:

    if m == "Symbolic regression":

        y_pred = (
            g_np(
                X[:, 0],
                X[:, 1],
                X[:, 2]
            ) > 0
        ).astype(int)

    else:

        y_pred = cross_val_predict(
            models[m],
            X,
            y,
            cv=loo
        )

    bal_m = balanced_accuracy_score(y, y_pred)

    if bal_m > best_bal:

        best_bal = bal_m
        best_name = m

print(
    f"\nBEST model -> "
    f"{best_name}  "
    f"(based on BalAcc = {best_bal:.3f})"
)

print(f"\nFigures saved in:\n{OUTPUT_DIR.resolve()}")
