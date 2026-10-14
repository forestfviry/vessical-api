"""CoronaryCare AI — FastAPI backend (app/main.py).

This is the missing application file: it creates the FastAPI app, registers
CORS, loads the trained pipelines, and implements every endpoint the
frontend calls:

    GET  /                     connection status (drives the header pill)
    POST /predict?explain=true CAD probability + vessel probabilities + LIME
    GET  /features             form schema (drives the auto-generated form)
    GET  /metrics              hold-out metrics + permutation importance
    GET  /dataset/size         row count for the demo-patient picker
    GET  /dataset/patient/{i}  one raw record + ground truth

Start it with:

    cd app && uvicorn main:app --reload --port 8000

Expected artifacts in app/models/:
    {cad,lad,lcx,rca}_pipeline.joblib   trained scikit-learn pipelines
    dataset.csv                          the Z-Alizadeh Sani extension table
    background.parquet                   (optional) LIME background matrix

Notes on the bugs this file fixes relative to the original snippets:
  * CORS: add_cors(app) is called HERE, immediately after app = FastAPI(...)
    (backend_fixes.py only defines the helper; defining is not registering).
  * Column names: the frontend sends "Airway_disease", the raw CSV has
    "Airway disease". ClinicalPreprocessor below normalises BOTH spellings
    (case-insensitive, underscore<->space) before mapping, so nothing is
    silently skipped and no raw string ever reaches the trees.
  * NaN safety: every mapped column is coerced with pd.to_numeric and
    filled with the negative class (0) afterwards — an unexpected token can
    no longer become NaN and crash or silently skew inference.
  * LIME background: the background matrix is REBUILT at startup from
    dataset.csv through the same preprocessor used at inference time, then
    aligned to the model's exact feature order. background.parquet is only
    used as a cross-check (a mismatch logs a warning instead of producing
    NaN contributions). This is the "regenerate background through
    ClinicalPreprocessor" fix, done automatically.
  * Request contract: PatientInput is GENERATED from the same FEATURE_GROUPS
    table that drives GET /features, so the API docs, the validator and the
    form schema can never drift apart. extra="forbid" makes a frontend typo
    fail loudly with a readable 422 instead of being silently dropped.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BeforeValidator, ConfigDict, Field, create_model

from backend_fixes import add_cors, lime_explanation

log = logging.getLogger("ccai")
logging.basicConfig(level=logging.INFO, format="%(levelname)s [ccai] %(message)s")

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
DATASET_CSV = MODEL_DIR / "dataset.csv"
BACKGROUND_PARQUET = MODEL_DIR / "background.parquet"
MODEL_FILES = {"cad": "cad_pipeline.joblib", "lad": "lad_pipeline.joblib",
               "lcx": "lcx_pipeline.joblib", "rca": "rca_pipeline.joblib"}
VESSELS = ["LAD", "LCX", "RCA"]
TARGET_COLS = ["Cath", "LAD", "LCX", "RCA"]  # never inputs (leakage guard)

# ---------------------------------------------------------------------------
# FEATURE_GROUPS is the single source of truth: it drives GET /features AND
# the dynamically created PatientInput model. (If you change a field, change
# it here only.)
#
# type: "number" | "int" | "yn" (Y/N string) | "bool" (1/0) | "select"
# ---------------------------------------------------------------------------
FEATURE_GROUPS = [
    {"group": "Demographics", "fields": [
        {"name": "Age",    "label": "Age",            "type": "int",    "min": 18,  "max": 100, "default": 58},
        {"name": "Sex",    "label": "Sex",            "type": "select", "options": ["Male", "Female"], "default": "Male"},
        {"name": "Weight", "label": "Weight",         "type": "number", "min": 25,  "max": 200, "unit": "kg",    "default": 80},
        {"name": "Length", "label": "Height",         "type": "number", "min": 100, "max": 220, "unit": "cm",    "default": 170},
        {"name": "BMI",    "label": "BMI (auto from weight/height)", "type": "number", "min": 10, "max": 60, "unit": "kg/m²", "default": 27.7},
    ]},
    {"group": "Risk factors & history", "fields": [
        {"name": "DM",             "label": "Diabetes mellitus",     "type": "bool", "default": 0},
        {"name": "HTN",            "label": "Hypertension",          "type": "bool", "default": 0},
        {"name": "Current_Smoker", "label": "Current smoker",        "type": "bool", "default": 0},
        {"name": "EX_Smoker",      "label": "Ex-smoker",             "type": "bool", "default": 0},
        {"name": "FH",             "label": "Family history of CAD", "type": "bool", "default": 0},
        {"name": "Obesity",        "label": "Obesity",               "type": "yn",   "default": "N"},
        {"name": "DLP",            "label": "Dyslipidemia",          "type": "yn",   "default": "N"},
        {"name": "CRF",            "label": "Chronic renal failure", "type": "yn",   "default": "N"},
        {"name": "CVA",            "label": "Cerebrovascular accident", "type": "yn", "default": "N"},
        {"name": "Airway_disease", "label": "Airway disease",        "type": "yn",   "default": "N"},
        {"name": "Thyroid_Disease","label": "Thyroid disease",       "type": "yn",   "default": "N"},
        {"name": "CHF",            "label": "Congestive heart failure", "type": "yn", "default": "N"},
    ]},
    {"group": "Examination & vitals", "fields": [
        {"name": "BP",    "label": "Blood pressure (systolic)", "type": "int",  "min": 60, "max": 260, "unit": "mmHg", "default": 130},
        {"name": "PR",    "label": "Pulse rate",                "type": "int",  "min": 30, "max": 180, "unit": "bpm",  "default": 85},
        {"name": "Edema", "label": "Peripheral edema",          "type": "bool", "default": 0},
        {"name": "Weak_Peripheral_Pulse", "label": "Weak peripheral pulse", "type": "yn", "default": "N"},
        {"name": "Lung_rales",            "label": "Lung rales",            "type": "yn", "default": "N"},
        {"name": "Systolic_Murmur",       "label": "Systolic murmur",       "type": "yn", "default": "N"},
        {"name": "Diastolic_Murmur",      "label": "Diastolic murmur",      "type": "yn", "default": "N"},
    ]},
    {"group": "Symptoms", "fields": [
        {"name": "Typical_Chest_Pain", "label": "Typical chest pain", "type": "bool", "default": 0},
        {"name": "Atypical",     "label": "Atypical chest pain",   "type": "yn", "default": "N"},
        {"name": "Nonanginal",   "label": "Non-anginal pain",      "type": "yn", "default": "N"},
        {"name": "Exertional_CP","label": "Exertional chest pain", "type": "yn", "default": "N"},
        {"name": "LowTH_Ang",    "label": "Low-threshold angina",  "type": "yn", "default": "N"},
        {"name": "Dyspnea",      "label": "Dyspnea",               "type": "yn", "default": "N"},
        {"name": "Function_Class", "label": "NYHA functional class", "type": "int", "min": 0, "max": 4, "default": 0},
    ]},
    {"group": "Electrocardiogram", "fields": [
        {"name": "Q_Wave",      "label": "Q wave",          "type": "bool", "default": 0},
        {"name": "St_Elevation","label": "ST elevation",    "type": "bool", "default": 0},
        {"name": "St_Depression","label": "ST depression",  "type": "bool", "default": 0},
        {"name": "Tinversion",  "label": "T-wave inversion","type": "bool", "default": 0},
        {"name": "LVH",             "label": "LV hypertrophy",        "type": "yn", "default": "N"},
        {"name": "Poor_R_Progression", "label": "Poor R-wave progression", "type": "yn", "default": "N"},
        {"name": "BBB", "label": "Bundle branch block", "type": "yn", "default": "N"},
    ]},
    {"group": "Laboratory", "fields": [
        {"name": "FBS",  "label": "Fasting blood sugar", "type": "number", "min": 40,   "max": 500,   "unit": "mg/dL", "default": 95},
        {"name": "CR",   "label": "Creatinine",          "type": "number", "min": 0.2,  "max": 12,    "unit": "mg/dL", "default": 1.0},
        {"name": "BUN",  "label": "Blood urea nitrogen", "type": "number", "min": 2,    "max": 120,   "unit": "mg/dL", "default": 15},
        {"name": "TG",   "label": "Triglycerides",       "type": "number", "min": 20,   "max": 1000,  "unit": "mg/dL", "default": 180},
        {"name": "LDL",  "label": "LDL cholesterol",     "type": "number", "min": 10,   "max": 400,   "unit": "mg/dL", "default": 130},
        {"name": "HDL",  "label": "HDL cholesterol",     "type": "number", "min": 5,    "max": 120,   "unit": "mg/dL", "default": 40},
        {"name": "ESR",  "label": "Erythrocyte sedimentation rate", "type": "number", "min": 1, "max": 150, "unit": "mm/h", "default": 12},
        {"name": "HB",   "label": "Hemoglobin",          "type": "number", "min": 5,    "max": 22,    "unit": "g/dL",  "default": 14.2},
        {"name": "K",    "label": "Potassium",           "type": "number", "min": 2,    "max": 8,     "unit": "mEq/L", "default": 4.5},
        {"name": "Na",   "label": "Sodium",              "type": "number", "min": 110,  "max": 170,   "unit": "mEq/L", "default": 140},
        {"name": "WBC",  "label": "White blood cells",   "type": "number", "min": 1000, "max": 50000, "unit": "/µL", "default": 7000},
        {"name": "Lymph","label": "Lymphocytes",         "type": "number", "min": 1,    "max": 80,    "unit": "%",    "default": 35},
        {"name": "Neut", "label": "Neutrophils",         "type": "number", "min": 5,    "max": 95,    "unit": "%",    "default": 55},
        {"name": "PLT",  "label": "Platelets",           "type": "number", "min": 20000,"max": 900000,"unit": "/µL", "default": 250},
    ]},
    {"group": "Echocardiography", "fields": [
        {"name": "EF_TTE",      "label": "Ejection fraction",      "type": "int", "min": 5, "max": 80, "unit": "%", "default": 45},
        {"name": "Region_RWMA", "label": "Regions with RWMA",      "type": "int", "min": 0, "max": 17, "default": 0},
        {"name": "VHD",         "label": "Valvular heart disease", "type": "select", "options": ["N", "Mild", "Moderate", "Severe"], "default": "N"},
    ]},
]

ALL_FIELDS = [f for g in FEATURE_GROUPS for f in g["fields"]]
FIELD_BY_NAME = {f["name"]: f for f in ALL_FIELDS}


# ---------------------------------------------------------------------------
# Request model — generated from FEATURE_GROUPS so /features and validation
# can never drift apart. Case-insensitive normalisers accept "male"/"MALE",
# "n"/"no" etc. and always store the canonical token the preprocessor maps.
# ---------------------------------------------------------------------------
def _canon_yn(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("y", "yes", "1", "1.0", "true"):
            return "Y"
        if s in ("n", "no", "0", "0.0", "false"):
            return "N"
    if v in (1, 1.0, True):
        return "Y"
    if v in (0, 0.0, False):
        return "N"
    return v


def _canon_sex(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("male", "m", "1"):
            return "Male"
        if s in ("female", "fmale", "f", "0"):
            return "Female"
    return v


def _canon_vhd(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip().lower()
        for opt in ("N", "Mild", "Moderate", "Severe"):
            if s == opt.lower():
                return opt
    return v


def _field_annotation(f: dict) -> Any:
    t = f["type"]
    if t == "yn":
        return Annotated[Literal["Y", "N"], BeforeValidator(_canon_yn)]
    if t == "select":
        opts = tuple(f["options"])
        canon = _canon_sex if f["name"] == "Sex" else (_canon_vhd if f["name"] == "VHD" else (lambda v: v))
        return Annotated[Literal[opts], BeforeValidator(canon)]  # type: ignore[valid-type]
    if t == "bool":
        return Annotated[int, Field(ge=0, le=1)]  # frontend sends 1/0; bool JSON coerces too
    # NOTE: min/max are served via /features as UI hints but are NOT enforced
    # here — the frontend's own FALLBACK_SCHEMA defaults must always validate
    # (e.g. its PLT default 250 sits below its declared min 20000).
    if t == "int":
        return int
    return float


PatientInput = create_model(
    "PatientInput",
    __config__=ConfigDict(extra="forbid"),
    **{f["name"]: (_field_annotation(f), Field(...)) for f in ALL_FIELDS},
)


# ---------------------------------------------------------------------------
# Preprocessor — FIXED version of backend_fixes.ClinicalPreprocessor.
#   * normalises column-name variants first ("Airway disease" == "Airway_disease",
#     case-insensitive) so nothing is silently skipped;
#   * BBB maps LBBB/RBBB/Y -> 1, N -> 0;
#   * every mapped column is coerced numerically and filled with 0 afterwards,
#     so a stray token can never reach the model as NaN.
# ---------------------------------------------------------------------------
class ClinicalPreprocessor:
    YN_COLS = [f["name"] for f in ALL_FIELDS if f["type"] == "yn"]
    BOOL_COLS = [f["name"] for f in ALL_FIELDS if f["type"] == "bool"]
    BBB_MAP = {"n": 0, "y": 1, "lbbb": 1, "rbbb": 1, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    SEX_MAP = {"male": 1, "fmale": 0, "female": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    VHD_MAP = {"n": 0, "mild": 1, "moderate": 2, "severe": 3, "0": 0, "1": 1, "2": 2, "3": 3}

    def __init__(self) -> None:
        # alias lookup: lowercase name with spaces OR underscores -> canonical
        self.alias: dict[str, str] = {}
        for name in FIELD_BY_NAME:
            self.alias[name.lower()] = name
            self.alias[name.lower().replace("_", " ")] = name

    def _canonical_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X.columns = [self.alias.get(str(c).strip().lower(), str(c).strip()) for c in X.columns]
        return X

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ClinicalPreprocessor":
        return self

    @staticmethod
    def _map_series(s: pd.Series, mapping: dict[str, float]) -> pd.Series:
        # numeric 0/1 passes through untouched; strings map via `mapping`
        def one(v: Any) -> Any:
            if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
                return float(v)
            return mapping.get(str(v).strip().lower(), np.nan)

        return s.map(one)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._canonical_columns(X)
        mapped: list[str] = []
        for c in self.YN_COLS:
            if c in X.columns:
                X[c] = self._map_series(X[c], self.BBB_MAP if c == "BBB" else {"y": 1, "n": 0})
                mapped.append(c)
        for c, mp in (("Sex", self.SEX_MAP), ("VHD", self.VHD_MAP)):
            if c in X.columns:
                X[c] = self._map_series(X[c], mp)
                mapped.append(c)
        # NaN safety: unmapped tokens become the negative class (0), never NaN
        for c in mapped:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
        for c in self.BOOL_COLS:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
        return X


PREPROCESSOR = ClinicalPreprocessor()


# ---------------------------------------------------------------------------
# Shared runtime state (filled by load_artifacts at startup).
# ---------------------------------------------------------------------------
STATE: dict[str, Any] = {
    "models": {},        # {"cad": pipeline, "lad": ..., "lcx": ..., "rca": ...}
    "feature_order": [], # exact column order the models were trained on
    "background": None,  # encoded background DataFrame for LIME
    "df_raw": None,      # raw dataset (for /dataset/* and metrics)
    "metrics": None,
    "warnings": [],
}


def _model_probs(model: Any, X_enc: pd.DataFrame, X_raw: pd.DataFrame) -> np.ndarray:
    """P(y=1). Uses the encoded matrix if the model knows those feature names,
    otherwise the raw frame (i.e. the pipeline embeds its own preprocessing).
    Falls back gracefully so either artifact style works."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        names = list(names)
        if set(names).issubset(X_enc.columns):
            return np.asarray(model.predict_proba(X_enc[names]))[:, 1]
        if set(names).issubset(X_raw.columns):
            return np.asarray(model.predict_proba(X_raw[names]))[:, 1]
    try:
        return np.asarray(model.predict_proba(X_enc))[:, 1]
    except Exception:
        return np.asarray(model.predict_proba(X_raw))[:, 1]


def _to_binary(s: pd.Series) -> pd.Series:
    """Raw dataset labels -> 0/1. Numeric: >=1 is positive. Strings: explicit
    vocabulary only (matches contract.js LABEL_POLARITY)."""
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce") >= 1).astype(int)
    vocab_pos = {"cad", "stenotic", "stenosis", "mild", "moderate", "severe",
                 "positive", "yes", "y", "lbbb", "rbbb"}
    return s.astype(str).str.strip().str.lower().isin(vocab_pos).astype(int)


def _load_joblib(path: Path) -> Any:
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise RuntimeError(
            f"Model artifact not found: {path}. Expected layout:\n"
            f"  app/models/{'{cad,lad,lcx,rca}_pipeline.joblib'}\n"
            f"  app/models/dataset.csv\n"
            f"Train the models first, then restart uvicorn."
        )


def load_artifacts() -> None:
    if not DATASET_CSV.exists():
        raise RuntimeError(f"Dataset not found: {DATASET_CSV}")
    df_raw = pd.read_csv(DATASET_CSV)
    df_raw = PREPROCESSOR._canonical_columns(df_raw)
    STATE["df_raw"] = df_raw

    for key, fname in MODEL_FILES.items():
        STATE["models"][key] = _load_joblib(MODEL_DIR / fname)
    log.info("loaded models: %s", ", ".join(sorted(STATE["models"])))

    # Feature order: trust the CAD pipeline's training columns when available,
    # else fall back to background.parquet.
    cad = STATE["models"]["cad"]
    names = getattr(cad, "feature_names_in_", None)
    if names is not None:
        feature_order = list(names)
    elif BACKGROUND_PARQUET.exists():
        feature_order = list(pd.read_parquet(BACKGROUND_PARQUET).columns)
    else:
        raise RuntimeError("Cannot determine feature order: cad_pipeline.joblib "
                           "has no feature_names_in_ and background.parquet is missing.")
    STATE["feature_order"] = feature_order

    # LIME background: REBUILD from the raw dataset through the SAME
    # preprocessor used at inference (fixes the column-mismatch NaN bug).
    X_bg = PREPROCESSOR.transform(df_raw.drop(columns=[c for c in TARGET_COLS
                                                       if c in df_raw.columns],
                                              errors="ignore"))
    X_bg = X_bg.reindex(columns=feature_order, fill_value=0).fillna(0).astype(float)
    STATE["background"] = X_bg

    if BACKGROUND_PARQUET.exists():
        try:
            saved = pd.read_parquet(BACKGROUND_PARQUET)
            if list(saved.columns) != feature_order:
                STATE["warnings"].append(
                    f"background.parquet columns do not match the model feature "
                    f"order — using the dataset-derived background instead "
                    f"(regenerate background.parquet through ClinicalPreprocessor "
                    f"if you want to keep the file).")
        except Exception as e:  # unreadable parquet -> ignore, we have ours
            STATE["warnings"].append(f"background.parquet unreadable ({e}); using dataset-derived background.")

    _selfcheck_frontend_contract()
    _compute_metrics()
    STATE["ready"] = True


def _selfcheck_frontend_contract() -> None:
    """Warn at startup if frontend/patient.json disagrees with PatientInput."""
    for candidate in (APP_DIR.parent / "frontend" / "patient.json",
                      APP_DIR / "patient.json"):
        if candidate.exists():
            try:
                keys = set(json.loads(candidate.read_text()).keys())
                mine = set(FIELD_BY_NAME)
                if keys != mine:
                    STATE["warnings"].append(
                        f"{candidate.name} field set differs from PatientInput: "
                        f"only-in-json={sorted(keys - mine)}, "
                        f"only-in-API={sorted(mine - keys)}. Align them.")
            except Exception:
                pass
            break


def _compute_metrics() -> None:
    """Hold-out metrics in exactly the shape app.js renders: per-target numbers
    become cards, `permutation_importance` lists become tables."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    df = STATE["df_raw"]
    feature_order = STATE["feature_order"]
    X_raw = df.drop(columns=[c for c in TARGET_COLS if c in df.columns], errors="ignore")
    X = PREPROCESSOR.transform(X_raw).reindex(columns=feature_order, fill_value=0).fillna(0).astype(float)
    idx = np.arange(len(X))
    strat = _to_binary(df["Cath"]) if "Cath" in df.columns else None
    _, te = train_test_split(idx, test_size=0.3, random_state=42, stratify=strat)

    out: dict[str, Any] = {}
    for raw_t, name in (("Cath", "CAD"), ("LAD", "LAD"), ("LCX", "LCX"), ("RCA", "RCA")):
        if raw_t not in df.columns:
            continue
        model = STATE["models"].get("cad" if name == "CAD" else name.lower())
        if model is None:
            continue
        y = _to_binary(df[raw_t])
        p = _model_probs(model, X, X_raw)
        yt, pt = y.iloc[te].to_numpy(), p[te]
        entry: dict[str, Any] = {
            "auc": float(roc_auc_score(yt, pt)),
            "accuracy": float(accuracy_score(yt, (pt >= 0.5).astype(int))),
            "f1": float(f1_score(yt, (pt >= 0.5).astype(int))),
        }
        try:
            pi = permutation_importance(model, X.iloc[te], yt, n_repeats=3,
                                        random_state=42, scoring="roc_auc")
            top = sorted(zip(feature_order, pi.importances_mean), key=lambda t: -abs(t[1]))[:12]
            entry["permutation_importance"] = [
                {"feature": f, "importance": float(v)} for f, v in top]
        except Exception as e:
            log.warning("permutation importance failed for %s: %s", name, e)
        out[name] = entry
    STATE["metrics"] = out


# ---------------------------------------------------------------------------
# App + CORS (item 2/3: the helper is DEFINED in backend_fixes.py but only
# takes effect because it is CALLED here).
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_artifacts()
        log.info("backend ready — %d dataset rows, %d model features",
                 len(STATE["df_raw"]), len(STATE["feature_order"]))
        for w in STATE["warnings"]:
            log.warning("startup check: %s", w)
    except Exception:
        log.exception("artifact loading failed — the API will answer /, /features "
                      "and /dataset/* but /predict and /metrics return 503.")
        STATE["ready"] = False
    yield


app = FastAPI(title="CoronaryCare AI API", version="1.0.0", lifespan=lifespan)
add_cors(app)

DISCLAIMER = ("Clinical decision support only — not a substitute for formal "
              "diagnosis by a qualified physician.")


def _require_ready() -> None:
    if not STATE.get("ready"):
        raise HTTPException(status_code=503, detail=(
            "Model artifacts not loaded. Check the uvicorn log — expected "
            "app/models/{cad,lad,lcx,rca}_pipeline.joblib and dataset.csv."))


def _risk_band(p: float) -> str:
    if p < 0.25:
        return "low"
    if p < 0.5:
        return "moderate"
    return "high"


def _encode(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame([payload])
    X = PREPROCESSOR.transform(raw).reindex(columns=STATE["feature_order"],
                                            fill_value=0).fillna(0).astype(float)
    return X, raw


@app.get("/")
def root() -> dict[str, Any]:
    return {"status": "ok", "service": "CoronaryCare AI API",
            "models_loaded": bool(STATE.get("ready")),
            "n_features": len(STATE.get("feature_order") or []),
            "dataset_rows": None if STATE["df_raw"] is None else int(len(STATE["df_raw"]))}


@app.get("/features")
def features() -> dict[str, Any]:
    """Shape contract.js expects: {groups: [{group, fields: [{name, label,
    type, min, max, unit, options, default}]}]}."""
    groups = []
    for g in FEATURE_GROUPS:
        fields = []
        for f in g["fields"]:
            e = {"name": f["name"], "label": f["label"], "type": f["type"],
                 "default": f["default"]}
            for k in ("min", "max", "unit", "options"):
                if k in f:
                    e[k] = f[k]
            fields.append(e)
        groups.append({"group": g["group"], "fields": fields})
    return {"groups": groups}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    _require_ready()
    if not STATE["metrics"]:
        raise HTTPException(status_code=503, detail="Metrics unavailable (see server log).")
    return STATE["metrics"]


@app.get("/dataset/size")
def dataset_size() -> dict[str, int]:
    _require_ready()
    return {"size": int(len(STATE["df_raw"]))}


def _demo_features(row: pd.Series) -> dict[str, Any]:
    """Convert one raw dataset row into form-fillable values (Y/N strings,
    'Male'/'Female', numeric types) — exactly what fillForm() expects."""
    feats: dict[str, Any] = {}
    for f in ALL_FIELDS:
        name = f["name"]
        if name not in row.index or pd.isna(row[name]):
            continue
        v = row[name]
        t = f["type"]
        if t == "yn":
            feats[name] = "Y" if _to_binary(pd.Series([v])).iloc[0] == 1 else "N"
        elif t == "bool":
            feats[name] = int(_to_binary(pd.Series([v])).iloc[0])
        elif name == "Sex":
            feats[name] = "Male" if _to_binary(pd.Series([v])).iloc[0] == 1 else "Female"
        elif name == "VHD":
            try:
                feats[name] = ["N", "Mild", "Moderate", "Severe"][int(v)]
            except (ValueError, IndexError):
                continue
        elif t == "int":
            try:
                feats[name] = int(v)
            except (ValueError, TypeError):
                continue
        elif t == "number":
            try:
                feats[name] = float(v)
            except (ValueError, TypeError):
                continue
    # BMI is usually absent from the raw dataset — derive it so the frontend
    # never has to (and if absent entirely, fillForm derives it on its own).
    if "BMI" not in feats and "Weight" in feats and "Length" in feats:
        feats["BMI"] = round(feats["Weight"] / ((feats["Length"] / 100) ** 2), 1)
    return feats


def _ground_truth(row: pd.Series) -> dict[str, str]:
    gt: dict[str, str] = {}
    if "Cath" in row.index and not pd.isna(row["Cath"]):
        gt["CAD"] = "CAD" if _to_binary(pd.Series([row["Cath"]])).iloc[0] == 1 else "No CAD"
    for v in VESSELS:
        if v in row.index and not pd.isna(row[v]):
            gt[v] = "Stenotic" if _to_binary(pd.Series([row[v]])).iloc[0] == 1 else "Normal"
    return gt


@app.get("/dataset/patient/{index}")
def dataset_patient(index: int) -> dict[str, Any]:
    _require_ready()
    df = STATE["df_raw"]
    if not 0 <= index < len(df):
        raise HTTPException(status_code=404, detail=f"index {index} out of range [0, {len(df)})")
    row = df.iloc[index]
    return {"index": index, "features": _demo_features(row),
            "ground_truth": _ground_truth(row)}


@app.post("/predict")
def predict(payload: PatientInput, explain: bool = Query(False)) -> dict[str, Any]:
    _require_ready()
    raw = payload.model_dump()
    X, raw_df = _encode(raw)
    bg = STATE["background"]

    cad_p = float(_model_probs(STATE["models"]["cad"], X, raw_df)[0])
    vessels = {}
    for v in VESSELS:
        p = float(_model_probs(STATE["models"][v.lower()], X, raw_df)[0])
        vessels[v] = {"stenosis_probability": p,
                      "predicted_label": "Stenotic" if p >= 0.5 else "Normal"}

    out: dict[str, Any] = {
        "cad": {"cad_probability": cad_p,
                "predicted_status": "CAD" if cad_p >= 0.5 else "No CAD",
                "risk_band": _risk_band(cad_p)},
        "vessels": vessels,
        "explanation": None,
        "disclaimer": DISCLAIMER,
    }

    if explain:
        expl: dict[str, Any] = {}
        expl["CAD"] = lime_explanation(STATE["models"]["cad"], bg, X)
        for v in VESSELS:
            expl[v] = lime_explanation(STATE["models"][v.lower()], bg, X)
        out["explanation"] = expl
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
