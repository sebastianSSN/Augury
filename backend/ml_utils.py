"""
Shared ML utilities: preprocessing, model I/O, and the core training pipeline.
Imported by both routers/ml.py and tasks.py — no circular dependencies.
"""
import io, os, time
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    r2_score, mean_absolute_error, mean_squared_error,
)
from fastapi import HTTPException

from logger import get_logger

log = get_logger("ml_utils")

_MODEL_DIR    = os.environ.get("MODEL_DIR", ".")
MAX_FILE_BYTES = 50 * 1024 * 1024

# Per-user in-memory model cache  {user_id: {model, pipeline, metrics, trained}}
_cache: dict[int, dict] = {}


# ── File helpers ──────────────────────────────────────────────────────────────

def user_dir(user_id: int) -> str:
    path = os.path.join(_MODEL_DIR, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


def model_paths(user_id: int) -> tuple[str, str, str]:
    d = user_dir(user_id)
    return (
        os.path.join(d, "model.joblib"),
        os.path.join(d, "pipeline.joblib"),
        os.path.join(d, "metrics.joblib"),
    )


def load_state(user_id: int) -> dict:
    if user_id in _cache:
        return _cache[user_id]
    mp, pp, xp = model_paths(user_id)
    if all(os.path.exists(p) for p in (mp, pp, xp)):
        state = {
            "model":    joblib.load(mp),
            "pipeline": joblib.load(pp),
            "metrics":  joblib.load(xp),
            "trained":  True,
        }
        _cache[user_id] = state
        return state
    return {"model": None, "pipeline": None, "metrics": None, "trained": False}


def invalidate(user_id: int):
    _cache.pop(user_id, None)


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Legacy helper — prefer dataset_io.load_dataset for new code."""
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo CSV inválido.")


# ── Column analysis ───────────────────────────────────────────────────────────

def infer_column_info(df: pd.DataFrame) -> list:
    info = []
    for col in df.columns:
        series   = df[col].dropna()
        is_num   = pd.api.types.is_numeric_dtype(series)
        n_unique = int(series.nunique())
        col_type = "numeric" if is_num else "categorical"
        if is_num and n_unique <= 15:
            col_type = "numeric_categorical"
        info.append({
            "name":    col,
            "type":    col_type,
            "n_unique": n_unique,
            "missing": int(df[col].isna().sum()),
            "samples": [str(v) for v in series.head(5).tolist()],
        })
    return info


def get_drop_suggestions(df: pd.DataFrame) -> list:
    suggestions = []
    total_rows  = len(df)
    for col in df.columns:
        n_unique          = df[col].nunique()
        cardinality_ratio = n_unique / total_rows if total_rows > 0 else 0
        if n_unique > 1000 or cardinality_ratio > 0.7:
            suggestions.append({
                "column":     col,
                "reason":     f"Alta cardinalidad ({n_unique} únicos)",
                "confidence": min(0.95, cardinality_ratio + 0.2),
                "type":       "cardinality",
            })
            continue
        if any(kw in col.lower() for kw in ["id", "uuid", "key", "codigo"]):
            suggestions.append({
                "column":     col,
                "reason":     "Parece ser identificador",
                "confidence": 0.9,
                "type":       "identifier",
            })
    return suggestions


# ── V1 preprocessing (legacy — kept for backward compat with old saved models) ─

def _preprocess_fit_v1(df: pd.DataFrame, feature_cols: list, target_col: str):
    df       = df.copy()
    encoders = {}
    y_raw     = df[target_col].fillna("unknown").astype(str)
    le_target = LabelEncoder()
    y         = le_target.fit_transform(y_raw)
    encoders["__target__"] = le_target
    X = df[feature_cols].copy()
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X[col]):
            fill_val      = float(X[col].median())
            X[col]        = X[col].fillna(fill_val)
            encoders[col] = {"type": "numeric", "fill": fill_val}
        else:
            X[col]        = X[col].fillna("missing").astype(str)
            le            = LabelEncoder()
            X[col]        = le.fit_transform(X[col])
            encoders[col] = {"type": "categorical", "le": le}
    return X.values, y, encoders


def _preprocess_transform_v1(df: pd.DataFrame, feature_cols: list, encoders: dict) -> np.ndarray:
    X = df[feature_cols].copy()
    for col in feature_cols:
        enc = encoders[col]
        if enc["type"] == "numeric":
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(enc["fill"])
        else:
            le    = enc["le"]
            known = set(le.classes_)
            vals  = X[col].fillna("missing").astype(str)
            vals  = vals.apply(lambda v: v if v in known else le.classes_[0])
            X[col] = le.transform(vals)
    return X.values


# ── V2 preprocessing (sklearn ColumnTransformer) ─────────────────────────────

def _detect_task_type(y_series: pd.Series) -> str:
    """Numeric target with > 20 unique values → regression, otherwise classification."""
    if pd.api.types.is_numeric_dtype(y_series) and y_series.nunique() > 20:
        return "regression"
    return "classification"


def _convert_datetime_cols(df: pd.DataFrame, feature_cols: list) -> tuple[pd.DataFrame, list]:
    """
    Detect datetime-like string columns and convert them to year (numeric).
    Returns modified df and list of converted column names.
    """
    df = df.copy()
    datetime_cols: list[str] = []
    for col in feature_cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.year.fillna(0).astype(int)
            datetime_cols.append(col)
        elif df[col].dtype == object:
            sample = df[col].dropna().head(30)
            if len(sample) < 5:
                continue
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() >= len(sample) * 0.8:
                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.year.fillna(0).astype(int)
                    datetime_cols.append(col)
            except Exception:
                pass
    return df, datetime_cols


def _build_column_transformer(df: pd.DataFrame, feature_cols: list) -> tuple[ColumnTransformer, dict]:
    """
    Build a fitted-ready ColumnTransformer dispatching columns by type.
    - Numeric → StandardScaler (after median imputation)
    - Low-cardinality categorical (≤50 unique) → OneHotEncoder
    - High-cardinality categorical (>50 unique) → OrdinalEncoder
    Returns (ColumnTransformer, feature_meta).
    """
    numeric_cols:  list[str] = []
    cat_low_cols:  list[str] = []
    cat_high_cols: list[str] = []
    feature_meta:  dict      = {}

    for col in feature_cols:
        series = df[col].dropna()
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            feature_meta[col] = {"type": "numeric", "uniques": None}
        else:
            n_unique = int(series.nunique())
            if n_unique <= 50:
                cat_low_cols.append(col)
                uniques = sorted(series.astype(str).unique().tolist())[:100]
                feature_meta[col] = {"type": "categorical", "uniques": uniques}
            else:
                cat_high_cols.append(col)
                feature_meta[col] = {"type": "categorical_high", "uniques": None}

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            SklearnPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]),
            numeric_cols,
        ))
    if cat_low_cols:
        transformers.append((
            "cat_low",
            SklearnPipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_low_cols,
        ))
    if cat_high_cols:
        transformers.append((
            "cat_high",
            SklearnPipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ord",     OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )),
            ]),
            cat_high_cols,
        ))

    ct = ColumnTransformer(transformers, remainder="drop")
    return ct, feature_meta


def preprocess_fit_v2(
    df: pd.DataFrame, feature_cols: list, target_col: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    V2 preprocessing pipeline using sklearn ColumnTransformer.
    Returns (X_array, y_array, pipeline_dict) where pipeline_dict version=2.
    """
    df = df.copy()
    df, datetime_cols = _convert_datetime_cols(df, feature_cols)

    task_type = _detect_task_type(df[target_col].dropna())

    if task_type == "regression":
        y         = df[target_col].values.astype(float)
        le_target = None
        classes: list = []
    else:
        y_raw     = df[target_col].fillna("unknown").astype(str)
        le_target = LabelEncoder()
        y         = le_target.fit_transform(y_raw)
        classes   = list(le_target.classes_)

    ct, feature_meta = _build_column_transformer(df, feature_cols)
    X = ct.fit_transform(df[feature_cols])

    pipeline_dict = {
        "version":        2,
        "feature_cols":   feature_cols,
        "target_col":     target_col,
        "task_type":      task_type,
        "target_encoder": le_target,
        "classes":        classes,
        "preprocessor":   ct,
        "feature_meta":   feature_meta,
        "datetime_cols":  datetime_cols,
    }
    return X, y, pipeline_dict


def preprocess_transform(df: pd.DataFrame, feature_cols: list, pipeline_data: dict) -> np.ndarray:
    """
    Transform features for inference. Handles v1 (legacy) and v2 (sklearn) pipeline formats.
    Always pass the full pipeline dict (not the inner encoders).
    """
    version = pipeline_data.get("version", 1)
    if version >= 2:
        df2 = df.copy()
        for col in pipeline_data.get("datetime_cols", []):
            if col not in df2.columns:
                continue
            if pd.api.types.is_datetime64_any_dtype(df2[col]):
                df2[col] = df2[col].dt.year.fillna(0).astype(int)
            elif df2[col].dtype == object:
                try:
                    df2[col] = pd.to_datetime(df2[col], errors="coerce").dt.year.fillna(0).astype(int)
                except Exception:
                    df2[col] = 0
        return pipeline_data["preprocessor"].transform(df2[feature_cols])
    # Legacy v1
    return _preprocess_transform_v1(df, feature_cols, pipeline_data["encoders"])


def get_target_encoder(pipeline_data: dict):
    """Return the LabelEncoder for the target column regardless of pipeline version."""
    if pipeline_data.get("version", 1) >= 2:
        return pipeline_data["target_encoder"]
    return pipeline_data["encoders"]["__target__"]


def get_feature_uniques(pipeline_data: dict) -> dict:
    """Return {col: [unique_values] or None} for building frontend prediction forms."""
    version = pipeline_data.get("version", 1)
    if version >= 2:
        return {col: meta.get("uniques") for col, meta in pipeline_data["feature_meta"].items()}
    encoders = pipeline_data.get("encoders", {})
    return {
        col: list(encoders[col]["le"].classes_)
        if encoders[col]["type"] == "categorical" else None
        for col in pipeline_data.get("feature_cols", [])
    }


# ── Training error ────────────────────────────────────────────────────────────

class TrainingError(ValueError):
    """Raised for invalid training configurations."""


# ── Algorithm registry ────────────────────────────────────────────────────────

ALGORITHMS: dict[str, dict] = {
    "random_forest": {
        "label": "Random Forest",
        "task":  "classification",
        "build": lambda: RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
    },
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "task":  "classification",
        "build": lambda: GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    },
    "logistic_regression": {
        "label": "Regresión Logística",
        "task":  "classification",
        "build": lambda: LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
    },
    "random_forest_regressor": {
        "label": "Random Forest (Regresión)",
        "task":  "regression",
        "build": lambda: RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
    },
    "gradient_boosting_regressor": {
        "label": "Gradient Boosting (Regresión)",
        "task":  "regression",
        "build": lambda: GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    },
    "ridge_regression": {
        "label": "Ridge Regression",
        "task":  "regression",
        "build": lambda: Ridge(alpha=1.0),
    },
}


def get_classifier(algorithm: str):
    if algorithm not in ALGORITHMS:
        raise TrainingError(
            f"Algoritmo '{algorithm}' no soportado. Opciones: {list(ALGORITHMS.keys())}"
        )
    return ALGORITHMS[algorithm]["build"]()


# ── Feature importance mapping ────────────────────────────────────────────────

def _map_importances(
    raw_imp: np.ndarray, pipeline_data: dict, feature_cols: list
) -> dict:
    """
    Map post-transformation importances back to original column names.
    For OHE-expanded columns, importances are summed back to the source column.
    """
    preprocessor = pipeline_data.get("preprocessor")
    if preprocessor is None:
        total = raw_imp.sum() or 1
        return dict(zip(feature_cols, [round(float(v / total), 4) for v in raw_imp]))

    try:
        col_imp: dict[str, float] = {col: 0.0 for col in feature_cols}
        idx = 0
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if name == "cat_low":
                ohe = transformer.named_steps["ohe"]
                for col, cats in zip(cols, ohe.categories_):
                    n = len(cats)
                    col_imp[col] += float(raw_imp[idx: idx + n].sum())
                    idx += n
            else:  # "num" or "cat_high" — 1-to-1 mapping
                for col in cols:
                    if idx < len(raw_imp):
                        col_imp[col] += float(raw_imp[idx])
                        idx += 1

        total = sum(col_imp.values()) or 1
        return {col: round(v / total, 4) for col, v in col_imp.items()}
    except Exception:
        return {col: round(1 / len(feature_cols), 4) for col in feature_cols}


# ── Core training pipeline ────────────────────────────────────────────────────

def train_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    algorithm: str = "random_forest",
) -> tuple[object, dict, dict, dict]:
    """
    Full v2 training pipeline. Returns (clf, pipeline_dict, metrics_dict, feature_uniques_dict).
    Raises TrainingError for invalid configurations.
    """
    df = df.dropna(subset=[target_col])
    if len(df) < 10:
        raise TrainingError("Muy pocas filas con target válido (mínimo 10).")

    task_type = _detect_task_type(df[target_col].dropna())
    algo_meta = ALGORITHMS.get(algorithm)
    if algo_meta is None:
        raise TrainingError(f"Algoritmo '{algorithm}' no soportado.")

    # Auto-correct algorithm / task mismatch
    algorithm_note = None
    if algo_meta["task"] != task_type:
        fallback = "random_forest" if task_type == "classification" else "random_forest_regressor"
        algorithm_note = (
            f"Se detectó tarea de {task_type}; '{algorithm}' no es compatible. "
            f"Se usó '{fallback}' automáticamente."
        )
        log.warning(algorithm_note)
        algorithm = fallback

    X, y, pipeline_data = preprocess_fit_v2(df, feature_cols, target_col)
    classes = pipeline_data["classes"]

    if task_type == "classification":
        if len(classes) < 2:
            raise TrainingError("El target necesita al menos 2 clases.")
        if len(classes) > 50:
            raise TrainingError(
                f"Demasiadas clases ({len(classes)}). ¿Elegiste una columna continua como target?"
            )

    stratify = y if (task_type == "classification" and len(classes) <= 20) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    t0  = time.perf_counter()
    clf = get_classifier(algorithm)
    clf.fit(X_train, y_train)
    elapsed = round(time.perf_counter() - t0, 2)
    log.info(f"{ALGORITHMS[algorithm]['label']} trained in {elapsed}s")

    y_pred = clf.predict(X_test)

    # Cross-validation (up to 5-fold, at least 10 samples per fold)
    n_splits  = min(5, len(df) // 10)
    cv_scores = np.array([0.0])
    if n_splits >= 2:
        cv_scoring = "accuracy" if task_type == "classification" else "r2"
        cv_cls = StratifiedKFold if task_type == "classification" else KFold
        cv = cv_cls(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            cv_scores = cross_val_score(clf, X, y, cv=cv, scoring=cv_scoring, n_jobs=-1)
        except Exception as e:
            log.warning(f"Cross-validation skipped: {e}")

    # Task-specific metrics
    if task_type == "classification":
        test_score  = round(float(accuracy_score(y_test, y_pred)), 4)
        target_dist = df[target_col].astype(str).value_counts().to_dict()
        task_metrics: dict = {
            "accuracy":            test_score,
            "cv_accuracy_mean":    round(float(cv_scores.mean()), 4),
            "cv_accuracy_std":     round(float(cv_scores.std()), 4),
            "target_distribution": {str(k): int(v) for k, v in target_dist.items()},
            "report":              classification_report(
                y_test, y_pred, target_names=classes, output_dict=True
            ),
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2  = round(float(r2_score(y_test, y_pred)), 4)
        task_metrics = {
            "accuracy":   r2,   # unified key for backward compat
            "r2":         r2,
            "mae":        round(float(mean_absolute_error(y_test, y_pred)), 4),
            "rmse":       round(float(np.sqrt(mse)), 4),
            "cv_r2_mean": round(float(cv_scores.mean()), 4),
            "cv_r2_std":  round(float(cv_scores.std()), 4),
        }

    # Feature importances
    if hasattr(clf, "feature_importances_"):
        raw_imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        coef    = np.abs(clf.coef_)
        raw_imp = coef.mean(axis=0) if coef.ndim > 1 else coef.ravel()
    else:
        raw_imp = np.ones(X.shape[1]) / X.shape[1]

    importances = _map_importances(raw_imp, pipeline_data, feature_cols)

    metrics: dict = {
        "task_type":           task_type,
        "total_samples":       len(df),
        "train_samples":       len(X_train),
        "test_samples":        len(X_test),
        "target_col":          target_col,
        "feature_cols":        feature_cols,
        "classes":             classes,
        "algorithm":           algorithm,
        "algorithm_label":     ALGORITHMS[algorithm]["label"],
        "training_time_s":     elapsed,
        "feature_importances": importances,
        **task_metrics,
    }
    if algorithm_note:
        metrics["algorithm_note"] = algorithm_note

    feature_uniques = get_feature_uniques(pipeline_data)
    return clf, pipeline_data, metrics, feature_uniques
