"""Fixed clinical feature preprocessing for the Z-Alizadeh Sani extension dataset.

BUGS FIXED vs. the original version:
  1. BBB column: the raw dataset contains "LBBB" / "RBBB" (not just Y/N), so
     mapping it through YN_MAP silently produced NaN for every LBBB/RBBB row.
     It now maps LBBB/RBBB -> 1, N -> 0.
  2. Any unmapped categorical value (e.g. a typo in Sex/VHD) became NaN and
     was passed straight to the model. Mapped categorical columns are now
     filled with 0 (the negative class) after mapping, so inference never
     crashes or feeds NaN into the trees.
  3. LIME robustness: binary-column detection broke on columns containing
     NaN (np.unique(nan) is not in {0,1} but is also not continuous data),
     and background.std() could yield NaN -> NaN distances. Both guarded.
  4. LIME output: x_encoded.iloc[i] selects the i-th ROW of the one-row
     DataFrame, so float(...) raised TypeError on pandas >= 2.0. Changed to
     iloc[0, i] to read feature i's actual value.
  6. The intercept column of the design matrix was appended UNWEIGHTED
     (np.ones) after multiplying the data by sqrt(w), biasing the weighted
     least-squares fit and flipping contribution signs.
  5. contributions had shape (1, n) because x_encoded.values is one row, so
     np.argsort returned a 2-D index array; iterating it produced 1-element
     arrays instead of scalars (broke 'feature'/'contribution'/'value' and,
     combined with bug 4, made lime_explanation crash on EVERY input).
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

YN_COLS = ["Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP",
           "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur",
           "Dyspnea", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang", "LVH",
           "Poor R Progression"]

# FIX 1: BBB handled separately — the dataset uses LBBB / RBBB / N, not Y/N.
BBB_COL = "BBB"
BBB_MAP = {"n": 0, "lbbb": 1, "rbbb": 1, "y": 1}

TARGETS = ["LAD", "LCX", "RCA", "Cath"]  # never inputs (leakage guard)

SEX_MAP = {"male": 1, "fmale": 0, "female": 0}
VHD_MAP = {"n": 0, "mild": 1, "moderate": 2, "severe": 3}
YN_MAP = {"y": 1, "n": 0}


class ClinicalPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        mapped = []
        for c in YN_COLS:
            if c in X.columns:
                X[c] = X[c].astype(str).str.strip().str.lower().map(YN_MAP)
                mapped.append(c)
        # FIX 1 (BBB): map LBBB/RBBB/N instead of plain Y/N.
        if BBB_COL in X.columns:
            X[BBB_COL] = (X[BBB_COL].astype(str).str.strip().str.lower()
                          .map(BBB_MAP))
            mapped.append(BBB_COL)
        if "Sex" in X.columns:
            X["Sex"] = X["Sex"].astype(str).str.strip().str.lower().map(SEX_MAP)
            mapped.append("Sex")
        if "VHD" in X.columns:
            X["VHD"] = X["VHD"].astype(str).str.strip().str.lower().map(VHD_MAP)
            mapped.append("VHD")

        # FIX 2: never let an unmapped categorical value reach the model as NaN.
        for c in mapped:
            n_bad = int(X[c].isna().sum())
            if n_bad:
                # replace unmapped values with the negative class (0)
                X[c] = X[c].fillna(0)
        return X.astype(float)


# --------------------------------------------------------------------------
# LIME — two robustness guards (rest of the implementation was sound:
# sqrt-weighted ridge regression on kernel-weighted neighbors is correct).
# --------------------------------------------------------------------------
def lime_explanation(model_pipe, background, x_encoded, n_samples=800,
                     kernel_width=0.75, top_k=10, rng=None):
    rng = rng or np.random.default_rng(42)
    # FIX 3a: drop NaN rows from the background so binary detection and std
    #         are computed on clean data.
    background = background.dropna(axis=0, how="any")
    n_features = background.shape[1]
    # ITEM 1: accept a Series, a one-row DataFrame or a NumPy row and always
    # normalise to a one-row DataFrame whose columns are in BACKGROUND order.
    # This makes iloc[0, i] and background.columns[i] provably the same axis.
    if isinstance(x_encoded, pd.Series):
        x_encoded = x_encoded.to_frame().T
    elif not isinstance(x_encoded, pd.DataFrame):
        x_encoded = pd.DataFrame(np.atleast_2d(x_encoded), columns=background.columns)
    x_encoded = x_encoded.reindex(columns=background.columns).astype(float)
    Z = np.empty((n_samples, n_features))
    Z[0] = x_encoded.values.ravel()
    for j, col in enumerate(background.columns):
        colvals = background[col].values
        uniq = np.unique(colvals[~pd.isna(colvals)])
        if len(uniq) and set(uniq).issubset({0.0, 1.0}):
            Z[1:, j] = rng.binomial(1, 0.5, n_samples - 1)
        else:
            Z[1:, j] = rng.choice(colvals, n_samples - 1, replace=True)
    Zdf = pd.DataFrame(Z, columns=background.columns)
    y_prob = model_pipe.predict_proba(Zdf)[:, 1]
    std = background.std(ddof=0).replace(0, 1).fillna(1).values  # FIX 3b
    d = np.sqrt((((Z - x_encoded.values.ravel()) / std) ** 2).sum(axis=1))
    w = np.exp(-(d ** 2) / (kernel_width ** 2))
    W = np.sqrt(w)
    Xw, yw = Z * W[:, None], y_prob * W
    Xd = np.column_stack([Xw, W])  # FIX 6: intercept column must ALSO be weighted (sqrt(w)*1); np.ones biased every slope
    coef, *_ = np.linalg.lstsq(Xd.T @ Xd + 1e-3 * np.eye(n_features + 1),
                               Xd.T @ yw, rcond=None)
    contributions = ((x_encoded.values - background.mean().values) * coef[:n_features]).ravel()  # FIX 5: was shape (1, n); iterating it produced array indices
    # INVARIANT (item 1): contributions[i], background.columns[i] and
    # x_encoded.iloc[0, i] all index the SAME feature axis (column order of
    # `background`, enforced by the reindex above). Do not reorder one without
    # the others.
    assert contributions.shape == (n_features,)
    order = np.argsort(-np.abs(contributions))[:top_k]
    return [{"feature": background.columns[i],
             "contribution": round(float(contributions[i]), 4),
             "direction": "increases risk" if contributions[i] > 0 else "decreases risk",
             "value": round(float(x_encoded.iloc[0, i]), 3)} for i in order]  # FIX 4: iloc[i] indexed a ROW; iloc[0, i] gets feature i's value


# --------------------------------------------------------------------------
# ITEM 2 — CORS. Without this the browser blocks every fetch() from
# http://localhost:5500 to http://localhost:8000 and the UI shows
# "API unreachable". Call add_cors(app) in app/main.py right after
# `app = FastAPI(...)`.
# Verify: DevTools > Network > GET /  -> 200 with an
#         `access-control-allow-origin` response header.
# --------------------------------------------------------------------------
def add_cors(app, origins=None):
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or [
            "http://localhost:5500",
            "http://127.0.0.1:5500",
            "http://localhost:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app
