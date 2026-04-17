from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import io, joblib, os
import uvicorn

app = FastAPI(title="Mini SaaS IA — Generic Classifier", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH    = "model.joblib"
PIPELINE_PATH = "pipeline.joblib"
METRICS_PATH  = "metrics.joblib"

state = {"model": None, "pipeline": None, "metrics": None, "trained": False}

def load_state():
    if all(os.path.exists(p) for p in [MODEL_PATH, PIPELINE_PATH, METRICS_PATH]):
        state["model"]    = joblib.load(MODEL_PATH)
        state["pipeline"] = joblib.load(PIPELINE_PATH)
        state["metrics"]  = joblib.load(METRICS_PATH)
        state["trained"]  = True
        print("✓ Modelo genérico cargado desde disco")

load_state()


# ── Utilidades ────────────────────────────────────────────────────────────────

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
            "name":     col,
            "type":     col_type,
            "n_unique": n_unique,
            "missing":  int(df[col].isna().sum()),
            "samples":  [str(v) for v in series.head(5).tolist()],
        })
    return info


def get_drop_suggestions(df: pd.DataFrame) -> list:
    """
    Analyzes columns and returns list of suggested drops with reasoning.
    Returns: [{"column": str, "reason": str, "confidence": float, "type": str}]

    Heuristics: high cardinality + ID-like names (simple & effective)
    """
    suggestions = []
    total_rows = len(df)

    for col in df.columns:
        # Heuristic 1: High cardinality (>70% unique values or >1000 unique)
        n_unique = df[col].nunique()
        cardinality_ratio = n_unique / total_rows if total_rows > 0 else 0
        if n_unique > 1000 or cardinality_ratio > 0.7:
            suggestions.append({
                "column": col,
                "reason": f"Alta cardinalidad ({n_unique} únicos)",
                "confidence": min(0.95, cardinality_ratio + 0.2),
                "type": "cardinality"
            })
            continue

        # Heuristic 2: ID-like column names
        col_lower = col.lower()
        id_keywords = ["id", "uuid", "key", "codigo"]
        if any(kw in col_lower for kw in id_keywords):
            suggestions.append({
                "column": col,
                "reason": f"Parece ser identificador",
                "confidence": 0.9,
                "type": "identifier"
            })

    return suggestions


def preprocess_fit(df: pd.DataFrame, feature_cols: list, target_col: str):
    df = df.copy()
    encoders = {}

    y_raw     = df[target_col].fillna("unknown").astype(str)
    le_target = LabelEncoder()
    y         = le_target.fit_transform(y_raw)
    encoders["__target__"] = le_target

    X = df[feature_cols].copy()
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X[col]):
            fill_val = float(X[col].median())
            X[col]   = X[col].fillna(fill_val)
            encoders[col] = {"type": "numeric", "fill": fill_val}
        else:
            X[col] = X[col].fillna("missing").astype(str)
            le     = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = {"type": "categorical", "le": le}

    return X.values, y, encoders


def preprocess_transform(df: pd.DataFrame, feature_cols: list, encoders: dict):
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    info = {"status": "ok", "version": "2.0.0", "model_ready": state["trained"]}
    if state["trained"]:
        p = state["pipeline"]
        info["target_col"]   = p["target_col"]
        info["feature_cols"] = p["feature_cols"]
        info["classes"]      = p["classes"]
    return info


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Paso 1 — analiza el CSV y devuelve info de columnas. No entrena nada."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo CSV inválido.")

    if len(df) < 20:
        raise HTTPException(status_code=400, detail="El CSV necesita al menos 20 filas.")
    if len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="El CSV necesita al menos 2 columnas.")

    return {
        "rows":        len(df),
        "columns":     len(df.columns),
        "column_info": infer_column_info(df),
        "preview":     df.head(5).fillna("").to_dict(orient="records"),
    }


@app.post("/suggest-drops")
async def suggest_drops(file: UploadFile = File(...)):
    """
    Analyzes CSV and suggests columns to exclude.
    Detects: high-cardinality columns, ID-like names.
    Called immediately after CSV upload, before target selection.
    """
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo CSV inválido.")

    if len(df) < 20:
        raise HTTPException(status_code=400, detail="CSV necesita ≥20 filas.")

    suggestions = get_drop_suggestions(df)

    return {
        "total_columns": len(df.columns),
        "suggestions": suggestions,
        "message": f"Detectadas {len(suggestions)} columnas potencialmente irrelevantes"
    }


@app.post("/train")
async def train(
    file:       UploadFile = File(...),
    target_col: str        = Form(...),
    drop_cols:  str        = Form(""),
):
    """Paso 2 — entrena con el CSV y el target elegido por el usuario."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo CSV inválido.")

    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Columna '{target_col}' no existe.")

    ignore       = {c.strip() for c in drop_cols.split(",") if c.strip()}
    ignore.add(target_col)
    feature_cols = [c for c in df.columns if c not in ignore]

    if not feature_cols:
        raise HTTPException(status_code=400, detail="No quedan columnas para features.")

    df = df.dropna(subset=[target_col])
    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Muy pocas filas con target válido.")

    X, y, encoders = preprocess_fit(df, feature_cols, target_col)
    le_target      = encoders["__target__"]
    classes        = list(le_target.classes_)

    if len(classes) < 2:
        raise HTTPException(status_code=400, detail="El target necesita al menos 2 clases.")
    if len(classes) > 50:
        raise HTTPException(
            status_code=400,
            detail=f"Demasiadas clases ({len(classes)}). ¿Elegiste una columna continua como target?"
        )

    stratify = y if len(classes) <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred      = clf.predict(X_test)
    acc         = accuracy_score(y_test, y_pred)
    importances = dict(zip(
        feature_cols,
        [round(float(v), 4) for v in clf.feature_importances_]
    ))
    target_dist = df[target_col].astype(str).value_counts().to_dict()

    metrics = {
        "accuracy":            round(acc, 4),
        "total_samples":       len(df),
        "train_samples":       len(X_train),
        "test_samples":        len(X_test),
        "target_col":          target_col,
        "feature_cols":        feature_cols,
        "classes":             classes,
        "target_distribution": {str(k): int(v) for k, v in target_dist.items()},
        "feature_importances": importances,
        "report":              classification_report(
            y_test, y_pred, target_names=classes, output_dict=True
        ),
    }

    pipeline = {
        "feature_cols": feature_cols,
        "target_col":   target_col,
        "encoders":     encoders,
        "classes":      classes,
    }

    joblib.dump(clf,      MODEL_PATH)
    joblib.dump(pipeline, PIPELINE_PATH)
    joblib.dump(metrics,  METRICS_PATH)
    state.update({"model": clf, "pipeline": pipeline, "metrics": metrics, "trained": True})

    # Build feature unique values dict for frontend
    feature_uniques = {}
    for col in feature_cols:
        enc = encoders[col]
        if enc["type"] == "categorical":
            # Get all unique values from the categorical encoder
            feature_uniques[col] = list(enc["le"].classes_)
        else:
            # Numeric features don't need unique values list
            feature_uniques[col] = None

    return {
        "status": "trained",
        "accuracy": round(acc, 4),
        "metrics": metrics,
        "feature_uniques": feature_uniques
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predicción batch: CSV sin columna target → devuelve predicciones."""
    if not state["trained"]:
        raise HTTPException(status_code=400, detail="Modelo no entrenado. POST /train primero.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo CSV inválido.")

    pipeline     = state["pipeline"]
    feature_cols = pipeline["feature_cols"]
    encoders     = pipeline["encoders"]
    classes      = pipeline["classes"]
    clf          = state["model"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columnas faltantes: {missing}")

    X     = preprocess_transform(df, feature_cols, encoders)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    le    = encoders["__target__"]

    results = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        label = le.inverse_transform([pred])[0]
        results.append({
            "row":           i + 1,
            "prediction":    str(label),
            "confidence":    round(float(max(prob)), 4),
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(classes, prob)},
        })

    return {"total": len(results), "target_col": pipeline["target_col"],
            "classes": classes, "predictions": results}


@app.post("/predict-single")
async def predict_single(data: dict):
    """Predicción para una fila enviada como JSON: {"col1": val1, "col2": val2, ...}"""
    if not state["trained"]:
        raise HTTPException(status_code=400, detail="Modelo no entrenado.")

    pipeline     = state["pipeline"]
    feature_cols = pipeline["feature_cols"]
    encoders     = pipeline["encoders"]
    classes      = pipeline["classes"]
    clf          = state["model"]

    missing = [c for c in feature_cols if c not in data]
    if missing:
        raise HTTPException(status_code=400, detail=f"Campos requeridos faltantes: {missing}")

    df    = pd.DataFrame([data])
    X     = preprocess_transform(df, feature_cols, encoders)
    pred  = clf.predict(X)[0]
    prob  = clf.predict_proba(X)[0]
    label = encoders["__target__"].inverse_transform([pred])[0]

    return {
        "prediction":    str(label),
        "confidence":    round(float(max(prob)), 4),
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(classes, prob)},
    }


@app.get("/model-info")
def model_info():
    if not state["trained"]:
        return {"trained": False}
    return {"trained": True, **state["metrics"]}


@app.delete("/model")
def delete_model():
    for path in [MODEL_PATH, PIPELINE_PATH, METRICS_PATH]:
        if os.path.exists(path):
            os.remove(path)
    state.update({"model": None, "pipeline": None, "metrics": None, "trained": False})
    return {"status": "deleted"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
