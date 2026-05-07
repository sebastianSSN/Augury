"""
Celery tasks. Runs in a separate worker process.
"""
import json, os
import joblib
import pandas as pd

from celery_app import celery_app
from database import SessionLocal
from logger import get_logger
import models
from ml_utils import model_paths, invalidate, train_dataframe, TrainingError

log = get_logger("tasks")


@celery_app.task(bind=True, name="tasks.train_model")
def train_model(
    self,
    user_id: int,
    csv_path: str,
    target_col: str,
    drop_cols: str,
    filename: str,
    algorithm: str = "random_forest",
) -> dict:
    """
    Full training pipeline. Runs in Celery worker, separate from the web process.
    Returns the training result dict that is stored in the Celery result backend.
    """
    log.info(f"Task started — user={user_id} target={target_col} file={filename}")
    self.update_state(state="STARTED", meta={"status": "Cargando dataset..."})

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        _cleanup(csv_path)
        raise RuntimeError(f"No se pudo leer el CSV: {exc}") from exc

    ignore       = {c.strip() for c in drop_cols.split(",") if c.strip()} | {target_col}
    feature_cols = [c for c in df.columns if c not in ignore]

    if not feature_cols:
        _cleanup(csv_path)
        raise RuntimeError("No quedan columnas para features tras eliminar las excluidas.")

    self.update_state(state="STARTED", meta={"status": "Entrenando modelo..."})

    try:
        clf, pipeline, metrics, feature_uniques = train_dataframe(df, feature_cols, target_col, algorithm)
    except TrainingError as exc:
        _cleanup(csv_path)
        raise RuntimeError(str(exc)) from exc

    # Persist model to disk (per-user directory)
    mp, pp, xp = model_paths(user_id)
    joblib.dump(clf,      mp)
    joblib.dump(pipeline, pp)
    joblib.dump(metrics,  xp)
    invalidate(user_id)

    # Persist metadata to DB
    db = SessionLocal()
    try:
        db.query(models.ModelRecord).filter(
            models.ModelRecord.user_id == user_id
        ).update({"is_active": False})
        record = models.ModelRecord(
            user_id      = user_id,
            dataset_name = filename,
            target_col   = target_col,
            feature_cols = json.dumps(feature_cols),
            n_classes    = len(metrics["classes"]),
            accuracy     = metrics["accuracy"],
            n_samples    = metrics["total_samples"],
            is_active    = True,
        )
        db.add(record)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    _cleanup(csv_path)
    log.info(f"Task done — user={user_id} accuracy={metrics['accuracy']}")

    return {
        "accuracy":        metrics["accuracy"],
        "metrics":         metrics,
        "feature_uniques": feature_uniques,
    }


def _cleanup(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
