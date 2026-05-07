import io, json, os, uuid
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from celery_app import celery_app
from database import get_db
from dataset_io import load_dataset
from dependencies import get_current_user
from logger import get_logger
from ml_utils import (
    ALGORITHMS,
    MAX_FILE_BYTES,
    get_drop_suggestions,
    get_feature_uniques,
    get_target_encoder,
    infer_column_info,
    invalidate,
    load_state,
    model_paths,
    preprocess_transform,
)
from tasks import train_model
import models, schemas

router  = APIRouter(tags=["ml"])
log     = get_logger("ml")
limiter = Limiter(key_func=get_remote_address)

_MODEL_DIR = os.environ.get("MODEL_DIR", ".")


# ── /analyze ──────────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
):
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="El archivo supera el límite de 50 MB.")

    df, norm_report = load_dataset(content, file.filename or "dataset.csv")

    if len(df) < 20:
        raise HTTPException(status_code=400, detail="El dataset necesita al menos 20 filas.")
    if len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="El dataset necesita al menos 2 columnas.")

    return {
        "rows":          len(df),
        "columns":       len(df.columns),
        "column_info":   infer_column_info(df),
        "preview":       df.head(5).fillna("").to_dict(orient="records"),
        "normalization": norm_report,
    }


# ── /suggest-drops ────────────────────────────────────────────────────────────

@router.post("/suggest-drops")
async def suggest_drops(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
):
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="El archivo supera el límite de 50 MB.")

    df, norm_report = load_dataset(content, file.filename or "dataset.csv")

    if len(df) < 20:
        raise HTTPException(status_code=400, detail="CSV necesita ≥20 filas.")

    suggestions = get_drop_suggestions(df)
    return {
        "total_columns": len(df.columns),
        "suggestions":   suggestions,
        "normalization": norm_report,
        "message":       f"Detectadas {len(suggestions)} columnas potencialmente irrelevantes",
    }


# ── /train — queues async job ─────────────────────────────────────────────────

@router.post("/train")
@limiter.limit("10/hour")
async def train(
    request:    Request,
    file:       UploadFile = File(...),
    target_col: str        = Form(...),
    drop_cols:  str        = Form(""),
    algorithm:  str        = Form("random_forest"),
    current_user: models.User = Depends(get_current_user),
):
    if algorithm not in ALGORITHMS:
        raise HTTPException(
            status_code=400,
            detail=f"Algoritmo '{algorithm}' no válido. Opciones: {list(ALGORITHMS.keys())}",
        )
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="El archivo supera el límite de 50 MB.")

    # Normalize the dataset now so the worker gets clean data
    df, norm_report = load_dataset(content, file.filename or "dataset.csv")

    # Remap target_col / drop_cols to their normalized names (if renamed)
    rename_map = {r["from"]: r["to"] for r in norm_report.get("column_renames", [])}
    target_col_norm = rename_map.get(target_col, target_col)
    drop_cols_norm  = ",".join(
        rename_map.get(c.strip(), c.strip())
        for c in drop_cols.split(",") if c.strip()
    )

    if target_col_norm not in df.columns:
        raise HTTPException(status_code=400, detail=f"Columna '{target_col}' no existe en el dataset.")

    ignore       = {c.strip() for c in drop_cols_norm.split(",") if c.strip()} | {target_col_norm}
    feature_cols = [c for c in df.columns if c not in ignore]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="No quedan columnas para features.")

    # Persist the *normalized* CSV so the worker reads clean data
    temp_dir  = os.path.join(_MODEL_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    job_id   = str(uuid.uuid4())
    csv_path = os.path.join(temp_dir, f"{job_id}.csv")
    df.to_csv(csv_path, index=False)

    train_model.apply_async(
        kwargs=dict(
            user_id    = current_user.id,
            csv_path   = csv_path,
            target_col = target_col_norm,
            drop_cols  = drop_cols_norm,
            filename   = file.filename or "dataset.csv",
            algorithm  = algorithm,
        ),
        task_id=job_id,
    )

    log.info(f"Training queued — user={current_user.id} job_id={job_id}")
    return {"job_id": job_id, "status": "queued", "normalization": norm_report}


# ── /train/status/{job_id} ────────────────────────────────────────────────────

@router.get("/train/status/{job_id}")
def train_status(
    job_id: str,
    current_user: models.User = Depends(get_current_user),
):
    result = celery_app.AsyncResult(job_id)

    if result.state in ("PENDING", "RECEIVED"):
        return {"status": "pending"}

    if result.state == "STARTED":
        meta = result.info or {}
        return {"status": "started", "detail": meta.get("status", "")}

    if result.state == "SUCCESS":
        invalidate(current_user.id)   # flush web-process cache so next load reads fresh model
        return {"status": "done", **result.get()}

    if result.state == "FAILURE":
        return {"status": "failed", "error": str(result.result)}

    return {"status": result.state.lower()}


# ── Prediction helpers ────────────────────────────────────────────────────────

def _load_predict_df(content: bytes, filename: str, pipeline: dict) -> pd.DataFrame:
    """Load prediction CSV, normalize column names, and validate feature columns."""
    df, _ = load_dataset(content, filename)
    feature_cols = pipeline["feature_cols"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columnas faltantes en el archivo: {missing}")
    return df


def _predict_rows(clf, pipeline: dict, df: pd.DataFrame) -> tuple[list, str]:
    """Run inference and return (results_list, task_type)."""
    feature_cols = pipeline["feature_cols"]
    classes      = pipeline["classes"]
    task_type    = pipeline.get("task_type", "classification")

    X     = preprocess_transform(df, feature_cols, pipeline)
    preds = clf.predict(X)

    if task_type == "classification":
        le    = get_target_encoder(pipeline)
        probs = clf.predict_proba(X)
        results = [
            {
                "row":           i + 1,
                "prediction":    str(le.inverse_transform([pred])[0]),
                "confidence":    round(float(max(prob)), 4),
                "probabilities": {cls: round(float(p), 4) for cls, p in zip(classes, prob)},
            }
            for i, (pred, prob) in enumerate(zip(preds, probs))
        ]
    else:
        results = [
            {"row": i + 1, "prediction": round(float(pred), 4)}
            for i, pred in enumerate(preds)
        ]

    return results, task_type


# ── /predict ──────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
):
    state = load_state(current_user.id)
    if not state["trained"]:
        raise HTTPException(status_code=400, detail="No tienes ningún modelo entrenado. POST /train primero.")

    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="El archivo supera el límite de 50 MB.")

    pipeline = state["pipeline"]
    df       = _load_predict_df(content, file.filename or "predict.csv", pipeline)
    results, task_type = _predict_rows(state["model"], pipeline, df)

    return {
        "total":      len(results),
        "task_type":  task_type,
        "target_col": pipeline["target_col"],
        "classes":    pipeline["classes"],
        "predictions": results,
    }


# ── /predict-csv — batch predict, returns CSV download ───────────────────────

@router.post("/predict-csv")
async def predict_csv(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
):
    state = load_state(current_user.id)
    if not state["trained"]:
        raise HTTPException(status_code=400, detail="No tienes ningún modelo entrenado. POST /train primero.")

    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="El archivo supera el límite de 50 MB.")

    pipeline     = state["pipeline"]
    clf          = state["model"]
    feature_cols = pipeline["feature_cols"]
    classes      = pipeline["classes"]
    task_type    = pipeline.get("task_type", "classification")

    df = _load_predict_df(content, file.filename or "predict.csv", pipeline)
    X  = preprocess_transform(df, feature_cols, pipeline)

    out = df.copy()
    if task_type == "classification":
        le          = get_target_encoder(pipeline)
        preds       = clf.predict(X)
        probs       = clf.predict_proba(X)
        out["prediction"] = le.inverse_transform(preds)
        out["confidence"] = [round(float(max(p)), 4) for p in probs]
        for cls, col_probs in zip(classes, zip(*probs)):
            out[f"prob_{cls}"] = [round(float(p), 4) for p in col_probs]
    else:
        preds = clf.predict(X)
        out["prediction"] = [round(float(p), 4) for p in preds]

    csv_bytes = out.to_csv(index=False).encode()
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


# ── /algorithms ───────────────────────────────────────────────────────────────

@router.get("/algorithms")
def list_algorithms():
    return [{"id": k, "label": v["label"], "task": v["task"]} for k, v in ALGORITHMS.items()]


# ── /predict-single ───────────────────────────────────────────────────────────

@router.post("/predict-single")
async def predict_single(
    data: dict,
    current_user: models.User = Depends(get_current_user),
):
    state = load_state(current_user.id)
    if not state["trained"]:
        raise HTTPException(status_code=400, detail="Modelo no entrenado.")

    pipeline     = state["pipeline"]
    feature_cols = pipeline["feature_cols"]
    classes      = pipeline["classes"]
    task_type    = pipeline.get("task_type", "classification")
    clf          = state["model"]

    # Normalize incoming keys so they match trained column names
    from dataset_io import clean_col_name
    norm_data = {clean_col_name(k): v for k, v in data.items()}

    missing = [c for c in feature_cols if c not in norm_data]
    if missing:
        raise HTTPException(status_code=400, detail=f"Campos requeridos faltantes: {missing}")

    df   = pd.DataFrame([norm_data])
    X    = preprocess_transform(df, feature_cols, pipeline)
    pred = clf.predict(X)[0]

    if task_type == "classification":
        le    = get_target_encoder(pipeline)
        prob  = clf.predict_proba(X)[0]
        label = le.inverse_transform([pred])[0]
        return {
            "prediction":    str(label),
            "confidence":    round(float(max(prob)), 4),
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(classes, prob)},
        }
    else:
        return {"prediction": round(float(pred), 4)}


# ── /model-info ───────────────────────────────────────────────────────────────

@router.get("/model-info")
def model_info(current_user: models.User = Depends(get_current_user)):
    state = load_state(current_user.id)
    if not state["trained"]:
        return {"trained": False}
    return {"trained": True, **state["metrics"]}


# ── /model-history ────────────────────────────────────────────────────────────

@router.get("/model-history", response_model=list[schemas.ModelRecordOut])
def model_history(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return (
        db.query(models.ModelRecord)
        .filter(models.ModelRecord.user_id == current_user.id)
        .order_by(models.ModelRecord.created_at.desc())
        .all()
    )


# ── /model DELETE ─────────────────────────────────────────────────────────────

@router.delete("/model")
def delete_model(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    mp, pp, xp = model_paths(current_user.id)
    for path in (mp, pp, xp):
        if os.path.exists(path):
            os.remove(path)
    invalidate(current_user.id)

    db.query(models.ModelRecord).filter(
        models.ModelRecord.user_id == current_user.id
    ).update({"is_active": False})
    db.commit()
    log.info(f"Model deleted — user={current_user.id}")

    return {"status": "deleted"}
