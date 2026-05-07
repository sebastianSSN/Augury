import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import time

from database import Base, engine
from routers import auth, ml
from logger import get_logger

log = get_logger("app")

Base.metadata.create_all(bind=engine)
log.info("Database tables verified")

# ── Rate limiter (shared across routers via app.state) ────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/hour"])

app = FastAPI(
    title="Augury — ML Predictions Platform",
    version="3.1.0",
    description="Plataforma no-code de predicciones ML con autenticación multi-usuario.",
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

_ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    log.warning(f"Rate limit exceeded: {request.method} {request.url.path} from {request.client.host}")
    return JSONResponse(
        status_code=429,
        content={"detail": f"Demasiadas peticiones. Límite: {exc.detail}. Intenta más tarde."},
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - start) * 1000, 1)
    log.info(f"{request.method} {request.url.path} → {response.status_code} ({ms}ms)")
    return response


app.include_router(auth.router)
app.include_router(ml.router)


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "version": "3.1.0"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy"}
