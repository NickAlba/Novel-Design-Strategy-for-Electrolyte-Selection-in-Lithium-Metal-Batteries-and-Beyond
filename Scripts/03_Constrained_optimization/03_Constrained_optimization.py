# -*- coding: utf-8 -*-

import time
import warnings
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
# ── Additional imports for logistic regression and optimization ──────────
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from scipy.optimize import differential_evolution
from scipy.optimize import NonlinearConstraint

start_time = time.time()

# ── 0.  Matplotlib style ────────────────────────────────────────────────
cbf = ["#0072B2", "#009E73", "#CC79A7", "#F0E442",
       "#56B4E9", "#E69F00", "#D55E00", "#000000"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cbf)
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=18)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)

warnings.filterwarnings("ignore", category=FutureWarning, module="pysr")
warnings.filterwarnings("ignore", category=UserWarning,  module="pysr")

# ── 1.  Directories ─────────────────────────────────────────────────────
base_dir        = Path(__file__).resolve().parent
plots_dir       = base_dir / "plots"
scatter_parent  = plots_dir / "scatter_plots"
contour_parent  = plots_dir / "contour_plots"
normalized_scatter_dir = scatter_parent / "normalized"
not_normalized_scatter_dir = scatter_parent / "not_normalized"
normalized_contour_dir = contour_parent / "normalized"
not_normalized_contour_dir = contour_parent / "not_normalized"

scatter_parent.mkdir(parents=True, exist_ok=True)
contour_parent.mkdir(parents=True, exist_ok=True)
normalized_scatter_dir.mkdir(parents=True, exist_ok=True)
not_normalized_scatter_dir.mkdir(parents=True, exist_ok=True)
normalized_contour_dir.mkdir(parents=True, exist_ok=True)
not_normalized_contour_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# 2. USER CONFIGURATION                                              #
# ------------------------------------------------------------------ #

csv_path = base_dir / "optimization_dataset.csv"     #insert dataset file name
if not csv_path.exists():
    raise FileNotFoundError(f"CSV file not found at: {csv_path}\n"
                            "Please place your conductivity.csv next to the script.")
# Load full dataset
df_full = pd.read_csv(csv_path, sep=";")

#Select C for logistic regression
C_LOG_REG = 100

# Compute component percentages
# df_full["LiTFSI%"]  = 100 - df_full["PVDF-HFP%"] - df_full["Jeff%"]
df_full["PVDF-HFP%"] /= 100
df_full["Jeff%"]     /= 100
df_full["LiTFSI%"]   /= 100

X = df_full[["PVDF-HFP%", "Jeff%", "LiTFSI%"]].values
y_sigma          = df_full["sigma"].values
y_log_sigma      = np.log10(y_sigma)
y_mech_success   = df_full["mechanical_success"].values


# ── 3.  Train‑test split & target normalisation ────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

(
    X_train, X_test,
    y_train_sigma,   y_test_sigma,
    y_train_log_sigma, y_test_log_sigma,
    y_train_mech,    y_test_mech
) = train_test_split(
    X,
    y_sigma,
    y_log_sigma,
    y_mech_success,
    test_size=0.2,
    random_state=1,
    stratify=y_mech_success
)
    
scaler_sigma     = MinMaxScaler()
scaler_log_sigma = MinMaxScaler()

y_train_sigma_norm     = scaler_sigma.fit_transform(y_train_sigma.reshape(-1, 1))
y_train_log_sigma_norm = scaler_log_sigma.fit_transform(y_train_log_sigma.reshape(-1, 1))
y_test_sigma_norm     = scaler_sigma.transform(y_test_sigma.reshape(-1, 1))
y_test_log_sigma_norm = scaler_log_sigma.transform(y_test_log_sigma.reshape(-1, 1))



# ── Logistic Regression for Mechanical Success ───────────────────────────
print("\n" + "=" * 60)
print("🔧 Training Logistic Regression for Mechanical Success...")

# Train logistic regression model
logistic_model = LogisticRegression(class_weight="balanced", penalty="l1", solver= "liblinear", C = C_LOG_REG, random_state=1, max_iter=1000)
logistic_model.fit(X_train, y_train_mech)

# Predictions
y_pred_mech_train = logistic_model.predict(X_train)
y_pred_mech_test = logistic_model.predict(X_test)
y_pred_mech_proba_train = logistic_model.predict_proba(X_train)[:, 1]
y_pred_mech_proba_test = logistic_model.predict_proba(X_test)[:, 1]

# Metrics
train_accuracy = accuracy_score(y_train_mech, y_pred_mech_train)
test_accuracy = accuracy_score(y_test_mech, y_pred_mech_test)
train_auc = roc_auc_score(y_train_mech, y_pred_mech_proba_train)
test_auc = roc_auc_score(y_test_mech, y_pred_mech_proba_test)

print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")
print(f"Training AUC: {train_auc:.3f}")
print(f"Test AUC: {test_auc:.3f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test_mech, y_pred_mech_test))

# Predict probabilities for all samples
probabilities = logistic_model.predict_proba(X)[:, 1]  # Column 1 = probability of success

# Add to DataFrame for analysis
df_full['predicted_mech_success'] = probabilities

# Sort by predicted probability descending
df_sorted = df_full.sort_values(by='predicted_mech_success', ascending=False)

# Display
print("\n📊 Predicted Mechanical Success Probabilities:")
print(df_sorted[['Formulazione', 'PVDF-HFP%', 'Jeff%', 'LiTFSI%', 'sigma', 'mechanical_success', 'predicted_mech_success']])
df_sorted.to_csv(plots_dir / "predicted_mechanical_success.csv", index=False)


#%%
# ── 4.  Models ──────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


models = {
    "Linear":    LinearRegression(),
    "Poly-2":    make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression()),
    # "Poly-3":    make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False), LinearRegression()), # FC - Commented out as requested
    # "Poly-4":    make_pipeline(StandardScaler(), PolynomialFeatures(4, include_bias=False), LinearRegression()), # FC - Commented out as requested
    "RF":        RandomForestRegressor(n_estimators=300, max_depth=None, random_state=1),
    "SVR_RBF": make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)),
    "GPR":     make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)), n_restarts_optimizer=9, random_state=1))
}

# --- PySR --------------------------------------------------------------
try:
    from pysr import PySRRegressor
    models["PySR"] = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        maxsize=12,                        
        parsimony=1e-3,                   
        model_selection="best",
        population_size=500,
        ncycles_per_iteration=100,
        verbosity=0,
        progress=False,
        output_jax_format=False,
        temp_equation_file=True,
    )
except ImportError:
    print("[INFO] PySR not installed – symbolic regression skipped.")

# ── 5.  Helper functions ────────────────────────────────────────────────
def get_plot_limits(values, margin_fraction=0.1):
    mi, mx = values.min(), values.max()
    margin = (mx - mi) * margin_fraction or np.abs(mi) * margin_fraction or 0.1
    return [mi - margin, mx + margin]

def contour_plot(X_arr, z_arr, title, out_path, cbar_label, is_normalized):
    """Generates and saves a contour plot."""
    fig, ax = plt.subplots(figsize=(7, 6))
    tri = Triangulation(X_arr[:, 0], X_arr[:, 1])
    levels = np.linspace(0, 1, 15) if is_normalized else 15
    contour = ax.tricontourf(tri, z_arr, levels=levels, cmap="viridis", alpha=0.9)
    ax.tricontour(tri, z_arr, levels=levels, colors='k', linewidths=0.5)
    ax.scatter(X_arr[:, 0], X_arr[:, 1], c="green", s=21, zorder=5, label='Data Points')
    ax.set_xlabel(r"PVDF-HFP (\%)")
    ax.set_ylabel(r"Jeffamine (\%)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_aspect('equal', adjustable='box')
    cbar = fig.colorbar(contour)
    if is_normalized:
         cbar.set_ticks(np.arange(0, 1.1, 0.2))
    cbar.set_label(cbar_label)
    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"   -> Contour plot saved to: {out_path}")


def consolidated_scatter_plot(plot_data, case_name, label, is_norm, save_dir):
    """Generates a single figure with scatter subplots for all models for easy comparison."""
    n_models = len(plot_data)
    if n_models == 0: return

    # FC - Set grid to 2 rows and 3 columns for PowerPoint, as requested.
    nrows, ncols = 2, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    all_y_true, all_y_pred = [], []
    for model_name in plot_data:
        d = plot_data[model_name]
        if is_norm:
            all_y_true.extend(d['y_train_norm']); all_y_true.extend(d['y_test_norm'])
            all_y_pred.extend(d['y_pred_train_norm']); all_y_pred.extend(d['y_pred_test_norm'])
        else:
            all_y_true.extend(d['y_train_orig']); all_y_true.extend(d['y_test_orig'])
            all_y_pred.extend(d['y_pred_train_orig']); all_y_pred.extend(d['y_pred_test_orig'])

    if is_norm:
        lim = [-0.05, 1.05]
        common_xlabel = f"Measured {label}"
        common_ylabel = f"Predicted {label}"
        plot_fname = f"{case_name}_scatter_consolidated.png"
    else:
        lim = get_plot_limits(np.hstack([all_y_true, all_y_pred]))
        if ("sigma" in case_name.lower()) and ("log" not in case_name.lower()):
            lim[0] = max(0, lim[0])
        if 'log' in case_name:
            common_xlabel = r"Measured $\log_{10} [\sigma / (\mathrm{S}\,\mathrm{cm}^{-1})]$"
            common_ylabel = r"Predicted $\log_{10} [\sigma / (\mathrm{S}\,\mathrm{cm}^{-1})]$"
        else:
            common_xlabel = r"Measured $\sigma \ (\mathrm{S}/\mathrm{cm})$"
            common_ylabel = r"Predicted $\sigma \ (\mathrm{S}/\mathrm{cm})$"
        plot_fname = f"{case_name.replace('_norm', '')}_scatter_consolidated.png"

    fig.supxlabel(common_xlabel, fontsize=22)
    fig.supylabel(common_ylabel, fontsize=22)

    for i, model_name in enumerate(plot_data):
        ax = axes[i]
        d = plot_data[model_name]
        if is_norm:
            y_train, y_test = d['y_train_norm'], d['y_test_norm']
            y_pred_train, y_pred_test = d['y_pred_train_norm'], d['y_pred_test_norm']
        else:
            y_train, y_test = d['y_train_orig'], d['y_test_orig']
            y_pred_train, y_pred_test = d['y_pred_train_orig'], d['y_pred_test_orig']
        ax.scatter(y_train, y_pred_train, c="#009E73", marker='o', s=60, label="Train", alpha=.7)
        ax.scatter(y_test, y_pred_test, c="#D55E00", marker='^', s=70, label="Test", alpha=.9)
        ax.plot(lim, lim, "k--", lw=1.3)
        ax.set_title(f"{model_name}")
        ax.legend(loc='upper left')
        r2, rmse, mae = d['r2_orig'], d['rmse_orig'], d['mae_orig']
        metrics_text = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3g}\nMAE = {mae:.3g}"
        ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(True)
        ax.set_aspect('equal', adjustable='box')
        ax.set(xlim=lim, ylim=lim)

    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout(rect=[0.04, 0.04, 1, 0.98])
    out_path = save_dir / plot_fname
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"   -> Consolidated scatter plot saved to: {out_path}")

# ── 6.  Analysis cases ──────────────────────────────────────────────────
analysis_cases = {
    "sigma_norm": {
        "y_train": y_train_sigma_norm, "y_test": y_test_sigma, "y_test_norm": y_test_sigma_norm,
        "scaler": scaler_sigma, "label": r"$\sigma_n$ (a.u.)"
    },
    "log_sigma_norm": {
        "y_train": y_train_log_sigma_norm, "y_test": y_test_log_sigma, "y_test_norm": y_test_log_sigma_norm,
        "scaler": scaler_log_sigma, "label": r"$\log_{10} \sigma_n$ (a.u.)"
    }
}

# ── 7.  Analysis Loop ─────────────────────────────────────────────────────
all_rows = []
all_plot_data = {}

print("\n" + "=" * 60)
print("🔬 Generating experimental data contour plots & model fits...")

for case_name, data in analysis_cases.items():
    print(f"\n🟡 Running case: {case_name}")
    print("‾" * (16 + len(case_name)))

    # 7.1 — decide which arrays and scaler to use for this case
    if case_name == "sigma_norm":
        ytr_raw, yte_raw       = y_train_sigma,   y_test_sigma
        ytr_norm, yte_norm     = y_train_sigma_norm.ravel(), y_test_sigma_norm.ravel()
        scaler = scaler_sigma
    else:  # "log_sigma_norm"
        ytr_raw, yte_raw       = y_train_log_sigma,   y_test_log_sigma
        ytr_norm, yte_norm     = y_train_log_sigma_norm.ravel(), y_test_log_sigma_norm.ravel()
        scaler = scaler_log_sigma

    # 7.2 — drop NaNs in the raw targets
    mask_tr = ~np.isnan(ytr_raw)
    mask_te = ~np.isnan(yte_raw)

    # report how many points survived
    print(f"  • train: kept {mask_tr.sum():>2d}/{len(mask_tr):>2d} points")
    print(f"  • test : kept {mask_te.sum():>2d}/{len(mask_te):>2d} points")

    # 7.3 — subset X and both forms of y
    Xtr = X_train[mask_tr]
    Xte = X_test[mask_te]

    ytr_raw_clean  = ytr_raw[mask_tr]
    yte_raw_clean  = yte_raw[mask_te]
    ytr_norm_clean = ytr_norm[mask_tr]
    yte_norm_clean = yte_norm[mask_te]

    # 7.4 — contour plots (over the full field, dropping NaNs)
    y_full_raw = y_log_sigma if "log" in case_name else y_sigma
    mask_full  = ~np.isnan(y_full_raw)
    y_full_norm = scaler.transform(y_full_raw.reshape(-1,1)).ravel()

    contour_plot(
        X_arr=X[mask_full],
        z_arr=y_full_norm[mask_full],
        title=f"Experimental {data['label']}",
        out_path=normalized_contour_dir / f"{case_name}_exp_contour.png",
        cbar_label=data["label"],
        is_normalized=True
    )
    raw_lbl = (r"$\log_{10}[\sigma\,(\mathrm{S/cm})]$"
               if "log" in case_name
               else r"$\sigma\ (\mathrm{S/cm})$")
    contour_plot(
        X_arr=X[mask_full],
        z_arr=y_full_raw[mask_full],
        title=f"Experimental {raw_lbl}",
        out_path=not_normalized_contour_dir / f"{case_name.replace('_norm','')}_exp_contour.png",
        cbar_label=raw_lbl,
        is_normalized=False
    )

    # 7.5 — fit & evaluate each model
    all_plot_data[case_name] = {}
    for model_name, model in models.items():
        m = deepcopy(model)
        m.fit(Xtr, ytr_norm_clean)

        # --- time measurement ---
        t0 = time.perf_counter()
        m.fit(Xtr, ytr_norm_clean)
        y_pred_tr_norm = m.predict(Xtr)
        y_pred_te_norm = m.predict(Xte)
        elapsed = time.perf_counter() - t0
         # ------------------------

        # back to raw σ
        y_pred_tr_raw = scaler.inverse_transform(y_pred_tr_norm.reshape(-1,1)).ravel()
        y_pred_te_raw = scaler.inverse_transform(y_pred_te_norm.reshape(-1,1)).ravel()

        # metrics on the raw scale
        r2   = r2_score(yte_raw_clean, y_pred_te_raw)
        rmse = np.sqrt(mean_squared_error(yte_raw_clean, y_pred_te_raw))
        mae  = mean_absolute_error(yte_raw_clean, y_pred_te_raw)

        # collect
        all_rows.append({
            "Case": case_name,
            "Model": model_name,
            "R2":    r2,
            "RMSE":  rmse,
            "MAE":   mae,
            "Time_sec": elapsed,
        })
        all_plot_data[case_name][model_name] = {
            "y_train_orig":      ytr_raw_clean,
            "y_test_orig":       yte_raw_clean,
            "y_pred_train_orig": y_pred_tr_raw,
            "y_pred_test_orig":  y_pred_te_raw,
            "y_train_norm":      ytr_norm_clean,
            "y_test_norm":       yte_norm_clean,
            "y_pred_train_norm": y_pred_tr_norm,
            "y_pred_test_norm":  y_pred_te_norm,
            "r2_orig":           r2,
            "rmse_orig":         rmse,
            "mae_orig":          mae,
            "time_sec":          elapsed,
        }

        print(f"   ✓ [{case_name}] {model_name:<8}  R²={r2:.3f} Computing time={elapsed:.3f} s")


# ── 8.  Consolidated Plots & Metrics ───────────────────────────────────
print("\n" + "=" * 60)
print("📊 Generating consolidated scatter plots...")
for case_name, plot_data_for_case in all_plot_data.items():
    if not plot_data_for_case: continue
    label = analysis_cases[case_name]['label']
    consolidated_scatter_plot(plot_data_for_case, case_name, label, is_norm=False, save_dir=not_normalized_scatter_dir)
    consolidated_scatter_plot(plot_data_for_case, case_name, label, is_norm=True, save_dir=normalized_scatter_dir)

# ── 9.  Metrics – single consolidated table ────────────────────────────
print("\n" + "=" * 60)
print("🔹 Consolidated metrics (test set, original scale):")
metrics_df = pd.DataFrame(all_rows).sort_values(["Case", "R2"], ascending=[True, False])
print(metrics_df.to_string(index=False, float_format="%.4g"))
metrics_df.to_csv(plots_dir / "model_metrics_summary.csv", index=False)
print(f"\n📄 Metrics CSV saved to: {plots_dir / 'model_metrics_summary.csv'}")
print("🏁 Workflow completed successfully.")

# ── 10. Model Selection & Optimization ─────────────────────────────────
print("\n" + "=" * 60)
print("🎯 Model Selection & Optimization...")

# Select best models for each case based on R² score
best_models = {}
for case_name in analysis_cases.keys():
    case_metrics = metrics_df[metrics_df['Case'] == case_name]
    best_model_name = case_metrics.loc[case_metrics['R2'].idxmax(), 'Model']
    best_r2 = case_metrics['R2'].max()
    best_models[case_name] = {
        'name': best_model_name,
        'r2': best_r2
    }
    print(f"🏆 Best model for {case_name}: {best_model_name} (R² = {best_r2:.3f})")

# Retrain best models on full training data for optimization
trained_best_models = {}
for case_name, best_info in best_models.items():
    model_name = best_info['name']
    
    # Get the appropriate data for this case
    if case_name == "sigma_norm":
        ytr_raw, ytr_norm = y_train_sigma, y_train_sigma_norm.ravel()
        scaler = scaler_sigma
    else:  #"log_sigma_norm":
        ytr_raw, ytr_norm = y_train_log_sigma, y_train_log_sigma_norm.ravel()
        scaler = scaler_log_sigma
    
    # Clean data (remove NaNs)
    mask_tr = ~np.isnan(ytr_raw)
    Xtr_clean = X_train[mask_tr]
    ytr_norm_clean = ytr_norm[mask_tr]
    
    # Retrain the best model
    best_model = deepcopy(models[model_name])
    best_model.fit(Xtr_clean, ytr_norm_clean)
    
    trained_best_models[case_name] = {
        'model': best_model,
        'scaler': scaler,
        'name': model_name
    }
#%%    
# ------------------------------------------------------------------ #
# 10.5. 3D IONIC CONDUCTIVITY SURFACE + MECHANICAL SUCCESS OVERLAY     #
# ------------------------------------------------------------------ #

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

"""
3D plot of ionic conductivity vs composition.

Parameters
----------
trained_model_dict : dict
    Dictionary with keys 'model', 'scaler', 'name'.
mech_model : optional
    Logistic regression model predicting mechanical success (0-1 probability).
grid_res : int
    Number of points per axis in the grid.
"""

def plot_conductivity_surface(trained_model_dict, mech_model=None, grid_res=50, target_type="sigma"):
    model = trained_model_dict['model']
    model_name = trained_model_dict['name']

    # Generate grid over compositional space (fractions 0–1)
    pvdf_range = np.linspace(0, 1, grid_res) 
    jeff_range = np.linspace(0, 1, grid_res)
    PVDF, JEFF = np.meshgrid(pvdf_range, jeff_range)
    LITFSI = 1 - PVDF - JEFF
    LITFSI[LITFSI < 0] = 0  # mask negative compositions
    mask = (PVDF + JEFF) > 1
    
    X3 = np.column_stack([PVDF.ravel(), JEFF.ravel(), LITFSI.ravel()])

    # Scale features if needed
    if hasattr(model, "named_steps") and 'standardscaler' in model.named_steps:
        X3_scaled = model.named_steps['standardscaler'].transform(X3)
    else:
        X3_scaled = X3

    # Predict conductivity
    y3_norm = model.predict(X3_scaled)
    scaler = trained_model_dict.get('scaler', None)
    if scaler is not None:
        try:
            y3_raw = scaler.inverse_transform(y3_norm.reshape(-1,1)).ravel()
        except Exception:
            y3_raw = y3_norm
    else:
        y3_raw = y3_norm

    # Predict mechanical success probability
    if mech_model is not None:
        prob_mech = mech_model.predict_proba(X3)[:, 1].reshape(PVDF.shape)
    else:
        prob_mech = np.ones_like(PVDF)

    # Reshape conductivity for plotting
    Z = y3_raw.reshape(PVDF.shape)
    
    # Apply mask to Z and prob_mech
    Z_masked = np.ma.array(Z, mask=mask)
    prob_mech_masked = np.ma.array(prob_mech, mask=mask)

    # Convert to percentages for plotting
    PVDF_pct  = PVDF * 100
    JEFF_pct  = JEFF * 100

    # Plot
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        PVDF_pct, JEFF_pct, Z_masked,
        facecolors=cm.viridis_r(prob_mech_masked),
        rstride=1, cstride=1,
        linewidth=0, antialiased=False, shade=False, alpha = 0.7
    )

    # Colorbar for mechanical success
    m = cm.ScalarMappable(cmap=cm.viridis_r)
    m.set_array(prob_mech)
    fig.colorbar(m, ax=ax, shrink=0.5, aspect=10, label='Mechanical Feasibility')

    # Labels depend on target type
    ax.set_xlabel(r"$x_{1}$", fontsize = 18)
    ax.set_ylabel(r"$x_{2}$", fontsize = 18)

    if target_type == "sigma":
        ax.set_zlabel(r"Predicted $\sigma_{ion}$ (S.cm$^{-1}$)", fontsize = 11.5)
        ax.set_title(f"{model_name} — Predicted Conductivity Surface")
    elif target_type == "log_sigma":
        ax.set_zlabel(r"Predicted $\log_{10}(\sigma)$", fontsize = 11.5)
        ax.set_title(f"{model_name} — Predicted log Conductivity Surface")

    plt.tight_layout()
    ax.view_init(elev=25, azim=45) 
    plt.show()


plot_conductivity_surface(trained_best_models['sigma_norm'], mech_model=logistic_model)
plot_conductivity_surface(trained_best_models['log_sigma_norm'], mech_model=logistic_model, target_type="log_sigma")

#%%

# ── 11. Optimization with Mechanical Success Constraint ────────────────
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0.*")

print("\n" + "=" * 60)
print("🔍 Optimization with Mechanical Success Constraint...")

def objective_function(x, case_name, minimize=True):
    """
    Objective function for optimization.
    x: [PVDF-HFP%, Jeff%, LiTFSI%] (normalized between 0-1)
    """
    x_reshaped = x.reshape(1, -1)
    
    # Get prediction from best model
    model_info = trained_best_models[case_name]
    pred_norm = model_info['model'].predict(x_reshaped)[0]
    
    # Convert back to original scale
    pred_orig = model_info['scaler'].inverse_transform([[pred_norm]])[0][0]
    
    # For log_sigma case, we want to maximize sigma (minimize negative log_sigma)
    # For sigma case, we want to maximize sigma (minimize negative sigma)
    if "log" in case_name:
        return -pred_orig if minimize else pred_orig  # pred_orig is log10(sigma)
    else:
        return -pred_orig if minimize else pred_orig  # pred_orig is sigma

def mechanical_success_constraint(x):
    """
    Constraint function: mechanical success probability must be >= threshold
    Returns positive value when constraint is satisfied
    """
    x_reshaped = x.reshape(1, -1)
    prob = logistic_model.predict_proba(x_reshaped)[0, 1]  # Probability of success
    return prob - mech_success_threshold

def composition_constraint(x):
    """
    Constraint: sum of components should equal 1 (within tolerance)
    """
    return 1.0 - np.sum(x)  # Should be close to 0

# Set mechanical success threshold
mech_success_threshold = 0.9
print(f"Mechanical success threshold: {mech_success_threshold:.1%}")

# Define bounds (components must sum to 1 and be non-negative)
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

# Define constraints
constraints = [
    NonlinearConstraint(mechanical_success_constraint, 0.0, np.inf),  # Mech success >= threshold
    NonlinearConstraint(composition_constraint, -0.01, 0.01)  # Sum = 1 (±1% tolerance)
]

# Optimization results storage
optimization_results = {}

for case_name in analysis_cases.keys():
    print(f"\n🔍 Optimizing for case: {case_name}")
    print(f"   Using best model: {trained_best_models[case_name]['name']}")
    
    try:
        # Run optimization
        result = differential_evolution(
            lambda x: objective_function(x, case_name, minimize=True),
            bounds,
            constraints=constraints,
            seed=42,
            maxiter=1000,
            popsize=15,
            atol=1e-6,
            tol=1e-6
        )
        
        if result.success:
            optimal_composition = result.x
            optimal_value = -result.fun  # Convert back from minimization
            
            # Get mechanical success probability
            mech_prob = logistic_model.predict_proba(optimal_composition.reshape(1, -1))[0, 1]

                
            # Store results
            optimization_results[case_name] = {
                'composition': optimal_composition,
                'value': optimal_value,
                'mech_probability': mech_prob,
                'success': True,
                'message': result.message
            }
            
            print(f"   ✅ Optimization successful!")
            print(f"   📍 Optimal composition:")
            print(f"      PVDF-HFP: {optimal_composition[0]:.1%}")
            print(f"      Jeffamine: {optimal_composition[1]:.1%}")
            print(f"      LiTFSI: {optimal_composition[2]:.1%}")
            print(f"   📊 Predicted {'log₁₀(σ)' if 'log' in case_name else 'σ'}: {optimal_value:.10f}")
            print(f"   🔧 Mechanical success probability: {mech_prob:.1%}")
            
        else:
            optimization_results[case_name] = {
                'success': False,
                'message': result.message
            }
            print(f"   ❌ Optimization failed: {result.message}")
            
    except Exception as e:
        optimization_results[case_name] = {
            'success': False,
            'message': str(e)
        }
        print(f"   ❌ Optimization error: {e}")

# ── 12. Save Optimization Results ─────────────────────────────────────
print("\n" + "=" * 60)
print("💾 Saving Optimization Results...")

# Create results DataFrame
opt_results_list = []
for case_name, result in optimization_results.items():
    if result['success']:
        opt_results_list.append({
            'Case': case_name,
            'Best_Model': trained_best_models[case_name]['name'],
            'PVDF_HFP_percent': result['composition'][0],
            'Jeff_percent': result['composition'][1],
            'LiTFSI_percent': result['composition'][2],
            'Predicted_Value': result['value'],
            'Mech_Success_Prob': result['mech_probability'],
            'Optimization_Success': True,
            'Message': result['message']
        })
    else:
        opt_results_list.append({
            'Case': case_name,
            'Best_Model': trained_best_models[case_name]['name'],
            'PVDF_HFP_percent': np.nan,
            'Jeff_percent': np.nan,
            'LiTFSI_percent': np.nan,
            'Predicted_Value': np.nan,
            'Mech_Success_Prob': np.nan,
            'Optimization_Success': False,
            'Message': result['message']
        })

opt_results_df = pd.DataFrame(opt_results_list)
opt_results_df.to_csv(plots_dir / "optimization_results.csv", index=False)

print("📊 Optimization Results Summary:")
print(opt_results_df.to_string(index=False, float_format="%.4g"))
print(f"\n📄 Results saved to: {plots_dir / 'optimization_results.csv'}")

# ── 13. Comparison with Existing Formulations ────────────────────────
print("\n" + "=" * 60)
print("🔍 Comparing with Existing Formulations...")

# Compare optimized results with existing successful formulations
successful_formulations = df_full[df_full['mechanical_success'] == 1].copy()

if len(successful_formulations) > 0:
    print(f"\n📈 Found {len(successful_formulations)} existing successful formulations:")
    
    for case_name, result in optimization_results.items():
        if result['success']:
            print(f"\n🎯 Case: {case_name}")
            print(f"   Optimized prediction: {result['value']:.10f}")
            
            # Compare with existing successful formulations
            target_col = 'sigma' if case_name == 'sigma_norm' else None
            if target_col and target_col in successful_formulations.columns:
                if case_name == 'log_sigma_norm':
                    existing_values = np.log10(successful_formulations[target_col].dropna())
                else:
                    existing_values = successful_formulations[target_col].dropna()
                
                if len(existing_values) > 0:
                    max_existing = existing_values.max()
                    mean_existing = existing_values.mean()
                    
                    print(f"   Best existing: {max_existing:.10f}")
                    print(f"   Mean existing: {mean_existing:.10f}")
                    print(f"   Improvement over best: {((result['value'] - max_existing) / max_existing * 100):+.1f}%")
                    print(f"   Improvement over mean: {((result['value'] - mean_existing) / mean_existing * 100):+.1f}%")

print(f"\n🏁 Optimization workflow completed successfully!")

end_time = time.time()
elapsed_time = end_time - start_time
mins, secs = divmod(elapsed_time, 60)
print(f"\n⏱️ Total run time: {int(mins)} min {secs:.1f} sec")