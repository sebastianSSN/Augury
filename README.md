# Augury — ML Predictions Platform v3.1

Plataforma no-code para entrenar modelos de ML y obtener predicciones con cualquier dataset.  
Multi-usuario con autenticación JWT, entrenamiento asíncrono via Celery, clasificación y regresión automáticas, predicción individual y batch con descarga CSV.

---

## Características

- **Genérico** — Funciona con cualquier CSV (Titanic, Iris, churn, heart disease, etc.)
- **Multi-usuario** — Cada cuenta tiene su propio modelo aislado
- **Autenticación JWT** — Register / Login / sesión persistente
- **Entrenamiento asíncrono** — Celery + Redis, el worker entrena sin bloquear la API
- **Tres algoritmos** — Random Forest · Gradient Boosting · Regresión Logística (seleccionable)
- **Predicción batch** — Sube un CSV y descarga las predicciones en CSV
- **Predicción individual** — Formulario interactivo con autocompletado para variables categóricas
- **Rate limiting** — Protección contra abuso (slowapi)
- **Tests** — 31 tests automatizados, CI/CD con GitHub Actions + Render

---

## Estructura del proyecto

```
augury/
├── backend/
│   ├── main.py              ← App FastAPI, middlewares, rate limiting
│   ├── auth.py              ← JWT + bcrypt
│   ├── models.py            ← ORM: User, ModelRecord
│   ├── schemas.py           ← Pydantic schemas
│   ├── database.py          ← SQLAlchemy + PostgreSQL
│   ├── dependencies.py      ← Dependencia get_current_user
│   ├── ml_utils.py          ← Pipeline ML: preprocesamiento, entrenamiento, algoritmos
│   ├── celery_app.py        ← Configuración Celery + Redis
│   ├── tasks.py             ← Tarea Celery: train_model
│   ├── logger.py            ← Logging estructurado (JSON en prod)
│   ├── routers/
│   │   ├── auth.py          ← POST /auth/register, /auth/login, GET /auth/me
│   │   └── ml.py            ← Todos los endpoints ML
│   ├── tests/
│   │   ├── conftest.py      ← Fixtures: SQLite in-memory, Celery eager, auth
│   │   ├── test_auth.py     ← 9 tests de autenticación
│   │   └── test_ml.py       ← 22 tests de endpoints ML
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example         ← Variables de entorno documentadas
├── frontend/
│   └── index.html           ← UI React (sin build, CDN)
├── docker-compose.yml       ← PostgreSQL + Redis + backend + worker + frontend
└── .github/workflows/ci.yml ← CI (tests) + CD (deploy Render)
```

---

## Puesta en marcha

### Opción A — Docker Compose (recomendada)

Es la forma más rápida: levanta PostgreSQL, Redis, el backend, el worker Celery y el frontend con un solo comando.

**Requisitos**: Docker Desktop instalado y corriendo.

**1. Clonar el repositorio**

```bash
git clone https://github.com/tu-usuario/augury.git
cd augury
```

**2. Crear el archivo de variables de entorno**

```bash
cp backend/.env.example backend/.env
```

Edita `backend/.env` y configura al menos `SECRET_KEY`:

```env
SECRET_KEY=cambia-esto-por-una-clave-segura-de-32-caracteres
DATABASE_URL=postgresql://saas_user:saas_pass@db:5432/saas_db
REDIS_URL=redis://redis:6379/0
MODEL_DIR=/app/models
ALLOWED_ORIGINS=http://localhost
LOG_FORMAT=text
```

> Para generar una `SECRET_KEY` segura: `python -c "import secrets; print(secrets.token_hex(32))"`

**3. Levantar todos los servicios**

```bash
docker compose up --build
docker compose logs -f backend worker
```

La primera vez descarga las imágenes y construye el backend (~2-3 minutos).  
Cuando veas `Application startup complete`, todo está listo.

**4. Abrir la app**

| Servicio | URL |
|----------|-----|
| Frontend | http://localhost:3000 |
| API / Swagger | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

**5. Detener los servicios**

```bash
docker compose down
```

Para borrar también los datos (PostgreSQL + modelos guardados):

```bash
docker compose down -v
```

---

### Opción B — Ejecución manual (sin Docker)

Útil para desarrollo local si ya tienes PostgreSQL y Redis instalados.

**Requisitos**: Python 3.10+, PostgreSQL 14+, Redis 7+

**1. Clonar el repositorio**

```bash
git clone https://github.com/tu-usuario/augury.git
cd augury
```

**2. Crear y activar entorno virtual**

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac / Linux
python -m venv .venv
source .venv/bin/activate
```

**3. Instalar dependencias**

```bash
cd backend
pip install -r requirements.txt
```

**4. Configurar variables de entorno**

```bash
cp .env.example .env
```

Edita `.env`:

```env
SECRET_KEY=cambia-esto-por-una-clave-segura
DATABASE_URL=postgresql://usuario:contraseña@localhost:5432/saas_db
REDIS_URL=redis://localhost:6379/0
MODEL_DIR=./tmp_models
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:5500
LOG_FORMAT=text
```

> Asegúrate de haber creado la base de datos: `createdb saas_db`

**5. Levantar el backend**

En una terminal:

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Verás `Application startup complete`. Las tablas se crean automáticamente al arrancar.

**6. Levantar el worker Celery**

En una segunda terminal (con el venv activado):

```bash
cd backend
celery -A celery_app worker --concurrency=2 --loglevel=info
```

Verás `celery@... ready.`

**7. Abrir el frontend**

```bash
# Windows
start frontend\index.html

# Mac
open frontend/index.html
```

O sirve la carpeta con cualquier servidor estático:

```bash
# Python
python -m http.server 3000 --directory frontend

# Node
npx serve frontend -p 3000
```

> Si usas `file://`, el navegador puede bloquear peticiones a localhost.  
> Recomendado: usar un servidor estático en el puerto 3000.

---

## Variables de entorno

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `SECRET_KEY` | Clave para firmar JWT. **Cambiar siempre.** | — |
| `DATABASE_URL` | URL de conexión PostgreSQL | `postgresql://...` |
| `REDIS_URL` | URL de conexión Redis | `redis://localhost:6379/0` |
| `MODEL_DIR` | Directorio donde se guardan los modelos | `.` |
| `ALLOWED_ORIGINS` | Orígenes CORS permitidos (coma-separados) | `http://localhost:3000` |
| `LOG_FORMAT` | `text` (legible) o `json` (producción) | `text` |
| `TOKEN_EXPIRE_DAYS` | Días de validez del token JWT | `7` |

---

## Flujo de uso

### Paso 1 — Crear cuenta / iniciar sesión

Al abrir la app verás la pantalla de autenticación.  
Regístrate con email y contraseña (mínimo 8 caracteres). El token se guarda en `localStorage` y se usa automáticamente.

### Paso 2 — Subir CSV

Arrastra un CSV o haz clic para seleccionarlo. El backend analiza las columnas automáticamente:
- Detecta tipos (numérico / categórico)
- Cuenta valores únicos y vacíos
- Muestra 5 filas de preview
- Sugiere columnas a excluir (IDs, alta cardinalidad)

**Requisitos del CSV**:
- Mínimo 20 filas y 2 columnas
- Target con 2–50 clases únicas
- Tamaño máximo: 50 MB

### Paso 3 — Configurar y entrenar

1. **Haz clic** en la columna que quieres predecir (target)
2. **Clic derecho** en columnas para ignorarlas (o usa el botón "Excluir sugerencias")
3. **Elige el algoritmo**: Random Forest · Gradient Boosting · Regresión Logística
4. **Pulsa Entrenar** — el job se envía al worker Celery y la UI hace polling hasta completar

### Paso 4 — Ver métricas

Una vez entrenado verás:
- Accuracy en test set (split 80/20 automático)
- Algoritmo utilizado
- Importancia de cada feature
- Distribución de la columna target
- Número de muestras, clases y features

### Paso 5 — Predecir

**Predicción individual**: Rellena el formulario con valores para cada feature → obtienes predicción + confianza + probabilidades por clase.

**Predicción batch**: Sube un CSV con las mismas columnas features → descarga `predictions.csv` con las predicciones añadidas.

---

## Algoritmos disponibles

| Algoritmo | Cuándo usarlo |
|-----------|---------------|
| **Random Forest** | Buena opción general. Robusto con datos ruidosos y missing values |
| **Gradient Boosting** | Mayor precisión en datasets limpios y bien estructurados |
| **Regresión Logística** | Datasets linealmente separables. Rápido e interpretable |

---

## API Endpoints

Todos los endpoints ML requieren autenticación: `Authorization: Bearer <token>`

### Autenticación

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/auth/register` | Crear cuenta (`email`, `password`) |
| `POST` | `/auth/login` | Obtener token (`email`, `password`) |
| `GET` | `/auth/me` | Info del usuario autenticado |

### ML

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/analyze` | Analiza columnas de un CSV |
| `POST` | `/suggest-drops` | Sugiere columnas a excluir |
| `POST` | `/train` | Encola entrenamiento (devuelve `job_id`) |
| `GET` | `/train/status/{job_id}` | Estado del job de entrenamiento |
| `GET` | `/algorithms` | Lista algoritmos disponibles |
| `GET` | `/model-info` | Métricas del modelo actual |
| `GET` | `/model-history` | Historial de entrenamientos |
| `POST` | `/predict-single` | Predicción para un registro (JSON) |
| `POST` | `/predict` | Predicción batch (CSV → JSON) |
| `POST` | `/predict-csv` | Predicción batch (CSV → CSV descargable) |
| `DELETE` | `/model` | Borra el modelo actual |

### Health

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Versión de la API |
| `GET` | `/health` | Health check |

Documentación interactiva completa en: `http://localhost:8000/docs`

---

## Rate limiting

| Endpoint | Límite |
|----------|--------|
| `POST /auth/register` | 10 por hora por IP |
| `POST /auth/login` | 20 por hora por IP |
| `POST /train` | 10 por hora por IP |
| Resto | 200 por hora por IP |

---

## Tests

```bash
cd backend
pytest tests/ -v
```

**31 tests** cubriendo: registro/login/auth, analyze, suggest-drops, train (3 algoritmos), predict-single, predict-csv, model-info, delete, aislamiento multi-usuario.

Los tests usan SQLite in-memory (sin PostgreSQL) y Celery en modo eager (sin Redis). No necesitan infraestructura externa.

---

## CI/CD

Cada push a `main` o Pull Request ejecuta automáticamente los tests en GitHub Actions.  
Los push a `main` también disparan un deploy automático en Render (requiere el secret `RENDER_DEPLOY_HOOK`).

---

## Datasets de prueba

| Dataset | Dónde encontrarlo | Target sugerido |
|---------|-------------------|-----------------|
| Titanic | [Kaggle](https://www.kaggle.com/c/titanic) | `Survived` |
| Iris | [UCI ML](https://archive.ics.uci.edu/ml/datasets/iris) | `Species` |
| Heart Disease | [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) | `HeartDisease` |
| Customer Churn | [Kaggle](https://www.kaggle.com/shrutimechlearn/churn-modelling) | `Exited` |
| Spam/Ham | [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset) | `v1` |

---

## Arquitectura

```
┌─────────────┐     HTTP      ┌──────────────────────┐
│  Frontend   │ ──────────►  │  FastAPI (port 8000)  │
│ (index.html)│ ◄──────────  │  + slowapi limits     │
└─────────────┘              └──────────┬─────────────┘
                                        │ apply_async
                             ┌──────────▼─────────────┐
                             │   Redis (broker)        │
                             └──────────┬─────────────┘
                                        │ consume
                             ┌──────────▼─────────────┐
                             │   Celery Worker         │
                             │   train_model task      │
                             │   → sklearn fit()       │
                             │   → joblib dump         │
                             └──────────┬─────────────┘
                                        │ writes
                             ┌──────────▼─────────────┐
                             │  PostgreSQL             │
                             │  users + model_records  │
                             └─────────────────────────┘
```

**Modelos en disco**: `{MODEL_DIR}/{user_id}/model.joblib` — cada usuario tiene su directorio aislado.
