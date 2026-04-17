# Mini SaaS IA — Clasificador genérico v2.0.0

Plataforma no-code para entrenar modelos ML de clasificación con cualquier dataset CSV. Sube datos, elige columna target, obtén predicciones.

## 🎯 Características

- **Genérico**: Funciona con cualquier CSV (Titanic, Iris, churn, spam, etc.)
- **Sin configuración**: Detección automática de tipos de columnas (numérico/categórico)
- **UI moderna**: Interfaz interactiva con 4 pasos guiados
- **RandomForest**: Modelo robusto + feature importances
- **Batch + Single**: Predicciones para CSV completo o registros individuales
- **Cero dependencias frontend**: React por CDN, abre directo sin build

---

## 📁 Estructura del proyecto

```
mini-saas-ia/
├── backend/
│   ├── main.py              ← API FastAPI + Random Forest
│   ├── requirements.txt     ← dependencias Python
│   └── titanic_sample.csv   ← dataset de prueba
└── frontend/
    └── index.html           ← UI React (HTML puro, no build)
```

---

## ⚡ Quickstart

### 1️⃣ Instalar dependencias

```bash
cd backend
pip install -r requirements.txt
```

### 2️⃣ Levantar el backend

```bash
uvicorn main:app --reload --port 8000
```

Verás: `Application startup complete`

Swagger UI: http://localhost:8000/docs

### 3️⃣ Abrir el frontend

```bash
# Mac/Linux
open frontend/index.html

# Windows
start frontend/index.html

# O simplemente abre desde el navegador: file:///ruta/a/frontend/index.html
```

---

## 🔄 Flujo de uso (4 pasos)

### Paso 1 — Subir CSV
Arrastra o selecciona un CSV cualquiera. El backend analiza columnas automáticamente.

### Paso 2 — Configurar
1. Haz clic en la columna que deseas **predecir** (target)
2. Las otras serán **features** automáticamente
3. Clic derecho en columnas para ignorarlas

### Paso 3 — Métricas
Entrena RandomForest. Verás:
- Accuracy en test set
- Importancia de cada feature
- Distribución de la clase target
- 📊 Gráficos interactivos

### Paso 4 — Predicciones
Ingresa valores para cada feature → obtén predicción + confianza + probabilidades por clase

---

## 📡 API Endpoints

### `POST /analyze`
Analiza un CSV sin entrenar.

**Params**: `file` (CSV)

**Respuesta**:
```json
{
  "rows": 891,
  "columns": 8,
  "column_info": [
    {
      "name": "Survived",
      "type": "numeric",
      "n_unique": 2,
      "missing": 0,
      "samples": ["0", "1"]
    }
  ],
  "preview": [...]
}
```

---

### `POST /train`
Entrena RandomForest con el CSV y columna target elegida.

**Params**:
- `file` (CSV)
- `target_col` (string) — columna a predecir
- `drop_cols` (string, opcional) — columnas a ignorar, separadas por coma

**Respuesta**:
```json
{
  "status": "trained",
  "accuracy": 0.8134,
  "metrics": {
    "accuracy": 0.8134,
    "total_samples": 891,
    "train_samples": 712,
    "test_samples": 179,
    "target_col": "Survived",
    "feature_cols": ["Pclass", "Sex", "Age", ...],
    "classes": ["0", "1"],
    "target_distribution": {"0": 549, "1": 342},
    "feature_importances": {
      "Sex": 0.2845,
      "Pclass": 0.1923,
      ...
    },
    "report": {...}
  }
}
```

Guarda automáticamente `model.joblib`, `pipeline.joblib`, `metrics.joblib`

---

### `POST /predict`
Predicción **batch**: CSV sin columna target → devuelve predicciones para cada fila.

**Params**: `file` (CSV)

**Respuesta**:
```json
{
  "total": 10,
  "target_col": "Survived",
  "classes": ["0", "1"],
  "predictions": [
    {
      "row": 1,
      "prediction": "1",
      "confidence": 0.8921,
      "probabilities": {"0": 0.1079, "1": 0.8921}
    }
  ]
}
```

---

### `POST /predict-single`
Predicción para **un registro**: envía JSON con valores.

**Body**:
```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 38,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 71.2833,
  "Embarked": "C"
}
```

**Respuesta**:
```json
{
  "prediction": "1",
  "confidence": 0.9234,
  "probabilities": {"0": 0.0766, "1": 0.9234}
}
```

---

### `GET /model-info`
Métricas del modelo actual (accuracy, features, distribución de clases, etc.)

**Respuesta**:
```json
{
  "trained": true,
  "accuracy": 0.8134,
  "feature_importances": {...},
  "target_distribution": {...}
}
```

---

### `GET /`
Health check.

**Respuesta**:
```json
{
  "status": "ok",
  "version": "2.0.0",
  "model_ready": true,
  "target_col": "Survived",
  "feature_cols": [...],
  "classes": ["0", "1"]
}
```

---

### `DELETE /model`
Borra el modelo entrenado y archivos (joblib).

**Respuesta**:
```json
{"status": "deleted"}
```

---

## 📊 Requisitos del CSV

**Para entrenamiento** (`/train`):
- Mínimo 10 filas con target válido
- Mínimo 2 columnas
- Una columna target con 2–50 clases únicas
- Valores faltantes → se rellenan automáticamente (mediana para numérico, "missing" para categórico)

**Para predicción** (`/predict`):
- Mismas columnas que el modelo (menos target)
- Mismo formato (tipos de datos consistentes)

**Tipos soportados**:
- Numérico: `int`, `float`
- Categórico: `string`
- Valores faltantes: se manejan automáticamente

---

## 💡 Ejemplos de datasets

Prueba con cualquiera de estos:

| Dataset | Link | Target |
|---------|------|--------|
| **Titanic** | [kaggle.com/titanic](https://www.kaggle.com/c/titanic) | Survived (0/1) |
| **Iris** | [archive.ics.uci.edu/iris](http://archive.ics.uci.edu/ml/datasets/iris) | Species |
| **Churn** | [Churn_Modelling.csv](https://www.kaggle.com/shrutimechlearn/churn-modelling) | Exited (0/1) |
| **Heart Disease** | [UCI ML](http://archive.ics.uci.edu/ml/datasets/Heart+Disease) | HeartDisease (0/1) |
| **Spam/Ham** | [SMS Spam](https://www.kaggle.com/uciml/sms-spam-collection-dataset) | Category |

---

## 🛠 Arquitectura

**Backend**:
- FastAPI con CORS habilitado
- Preprocesamiento automático (LabelEncoder)
- Random Forest 150 estimadores
- Split 80/20 con stratify

**Frontend**:
- React 18 por CDN
- Componentes funcionales, hooks
- Drag-drop nativo
- Gráficos de barras CSS puro

**Persistencia**:
- Modelos guardados en disk (`.joblib`)
- Reutiliza modelo anterior si existe
- `/model` DELETE para limpiar

---

## 🚀 Siguientes pasos (Fase 2 — AWS)

- [ ] Dockerizar backend
- [ ] Upload a Amazon ECR
- [ ] Deploy como AWS Lambda (Mangum)
- [ ] Guardar CSVs en S3
- [ ] Servir modelo desde SageMaker endpoint
- [ ] Agregar API Gateway + Cognito auth
