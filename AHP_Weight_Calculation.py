# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 11:24:45 2026

@author: acer
"""

import numpy as np
import pandas as pd

# -----------------------------
# USER INPUT
# -----------------------------
CSV_PATH = r"I:\My Drive\Personal Webpage\Github\MCDM_Analysis\Priority_Matrix.csv"   # <-- change this

# -----------------------------
# AHP helpers
# -----------------------------
RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
}

def fill_reciprocals(A: np.ndarray) -> np.ndarray:
    """Fill missing A[i,j] using reciprocity A[i,j] = 1/A[j,i]."""
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 1.0
            elif np.isnan(A[i, j]) and not np.isnan(A[j, i]):
                A[i, j] = 1.0 / A[j, i]
    return A

def ahp_weights_eigen(A: np.ndarray):
    """Eigenvector weights + consistency stats."""
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argmax(eigvals.real)
    lambda_max = eigvals.real[idx]
    w = eigvecs[:, idx].real
    w = np.abs(w)
    w = w / w.sum()

    n = A.shape[0]
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, None)
    CR = (CI / RI) if (RI is not None and RI > 0) else np.nan
    return w, float(lambda_max), float(CI), float(CR)

def ahp_weights_geomean(A: np.ndarray):
    """Geometric mean weights (robust alternative)."""
    gm = np.prod(A, axis=1) ** (1.0 / A.shape[0])
    w = gm / gm.sum()
    return w

# -----------------------------
# Load CSV matrix
# -----------------------------
df = pd.read_csv(CSV_PATH)

# First column is the row names (e.g., Model/Criteria)
row_names = df.iloc[:, 0].astype(str).tolist()
col_names = df.columns[1:].astype(str).tolist()

# Ensure square matrix
if len(row_names) != len(col_names):
    raise ValueError(
        f"Matrix not square: {len(row_names)} rows vs {len(col_names)} columns.\n"
        "Make sure the first column is row labels and remaining headers match criteria names."
    )

# Numeric matrix (blank -> NaN)
A = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

# Fill reciprocals
A = fill_reciprocals(A)

# Validate no NaNs remain
if np.isnan(A).any():
    nan_locs = np.argwhere(np.isnan(A))
    print("Still missing entries after filling reciprocals at:")
    for i, j in nan_locs[:20]:
        print(f"  ({row_names[i]}, {col_names[j]})")
    raise ValueError("Matrix still has missing values. Provide at least one triangle (upper or lower).")

# Validate positivity
if (A <= 0).any():
    raise ValueError("AHP matrix must be positive (>0). Check your CSV for zeros/negative values.")

# Validate reciprocity (optional strict check)
tol = 1e-6
for i in range(A.shape[0]):
    for j in range(A.shape[0]):
        if i != j:
            if abs(A[i, j] * A[j, i] - 1.0) > 1e-3:
                print(f"Warning: reciprocity not exact at ({row_names[i]}, {row_names[j]}): "
                      f"Aij={A[i,j]:.4g}, Aji={A[j,i]:.4g}, product={A[i,j]*A[j,i]:.4g}")

# -----------------------------
# Compute weights
# -----------------------------
w_eig, lambda_max, CI, CR = ahp_weights_eigen(A)
w_gm = ahp_weights_geomean(A)

out = pd.DataFrame({
    "Criteria": row_names,
    "Weight_Eigen": w_eig,
    "Weight_GeoMean": w_gm
}).sort_values("Weight_Eigen", ascending=False).reset_index(drop=True)

print("\n=== AHP Weights (sorted by Eigen weights) ===")
print(out.to_string(index=False))

print("\n=== Consistency ===")
print(f"n = {A.shape[0]}")
print(f"lambda_max = {lambda_max:.6f}")
print(f"CI = {CI:.6f}")
print(f"CR = {CR:.6f}  (Rule of thumb: CR < 0.10 is acceptable)")
