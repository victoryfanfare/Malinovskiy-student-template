from __future__ import annotations

import io
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from .core import (
    summarize_dataset,
    missing_table,
    compute_quality_flags,
    generate_json_summary,
    top_categories,
    correlation_matrix,
    DatasetSummary,
)

# ---------- Настройка логирования ----------

# Создаем логгер
logger = logging.getLogger("eda_api")
logger.setLevel(logging.INFO)

# Создаем обработчик для файла
file_handler = logging.FileHandler("logs/api.log")
file_handler.setLevel(logging.INFO)

# Создаем обработчик для консоли
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Создаем форматтер для JSON логов
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **getattr(record, "extra_data", {})
        }
        return json.dumps(log_record)

# Настраиваем форматтеры
json_formatter = JsonFormatter()
file_handler.setFormatter(json_formatter)

# Простой форматтер для консоли
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Добавляем обработчики к логгеру
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Создаем папку для логов
Path("logs").mkdir(exist_ok=True)

# ---------- Хранение метрик ----------

class MetricsStore:
    """Класс для хранения метрик сервиса в памяти"""
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0
        self.endpoint_stats = defaultdict(lambda: {
            "count": 0,
            "total_latency_ms": 0.0,
            "last_status": None,
            "last_timestamp": None
        })
        self.last_responses = []
    
    def record_request(self, endpoint: str, status: int, latency_ms: float):
        """Запись информации о запросе"""
        self.total_requests += 1
        
        if 200 <= status < 300:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_latency_ms += latency_ms
        
        stats = self.endpoint_stats[endpoint]
        stats["count"] += 1
        stats["total_latency_ms"] += latency_ms
        stats["last_status"] = status
        stats["last_timestamp"] = time.time()
        
        # Сохраняем последние 10 ответов
        self.last_responses.append({
            "endpoint": endpoint,
            "status": status,
            "timestamp": time.time(),
            "latency_ms": latency_ms
        })
        if len(self.last_responses) > 10:
            self.last_responses.pop(0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получение всех метрик"""
        avg_latency = self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0
        
        endpoint_details = {}
        for endpoint, stats in self.endpoint_stats.items():
            endpoint_avg = stats["total_latency_ms"] / stats["count"] if stats["count"] > 0 else 0
            endpoint_details[endpoint] = {
                "total_requests": stats["count"],
                "avg_latency_ms": round(endpoint_avg, 2),
                "last_status": stats["last_status"],
                "last_request": stats["last_timestamp"]
            }
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.successful_requests / self.total_requests * 100, 2) if self.total_requests > 0 else 0,
            "avg_latency_ms": round(avg_latency, 2),
            "endpoint_stats": endpoint_details,
            "last_responses": self.last_responses[-5:] if self.last_responses else []
        }

# Создаем глобальное хранилище метрик
metrics_store = MetricsStore()

# ---------- Создание FastAPI приложения ----------

app = FastAPI(
    title="EDA CLI HTTP Service",
    description="HTTP-сервис для анализа качества датасетов",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Middleware для логирования и метрик
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware для логирования всех запросов"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Вызываем следующий middleware/обработчик
    response = await call_next(request)
    
    # Рассчитываем время выполнения
    latency_ms = (time.time() - start_time) * 1000
    
    # Получаем информацию о запросе
    endpoint = request.url.path
    status_code = response.status_code
    
    # Записываем метрики
    metrics_store.record_request(endpoint, status_code, latency_ms)
    
    # Формируем данные для лога
    log_data = {
        "request_id": request_id,
        "endpoint": endpoint,
        "method": request.method,
        "status": status_code,
        "latency_ms": round(latency_ms, 2),
        "client_host": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    # Логируем запрос
    logger.info(f"Request processed", extra={"extra_data": log_data})
    
    # Добавляем request_id в заголовки ответа
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Processing-Time"] = f"{latency_ms:.2f}ms"
    
    return response

# ---------- Pydantic модели ----------

class QualityRequest(BaseModel):
    """Запрос для оценки качества по агрегированным признакам."""
    n_rows: int = Field(..., ge=0, description="Число строк")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(..., ge=0.0, le=1.0, description="Максимальная доля пропусков")
    numeric_cols: int = Field(..., ge=0, description="Количество числовых колонок")
    categorical_cols: int = Field(..., ge=0, description="Количество категориальных колонок")

class QualityResponse(BaseModel):
    """Ответ с оценкой качества."""
    ok_for_model: bool = Field(..., description="Подходит ли для обучения модели")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Оценка качества 0-1")
    message: str = Field(..., description="Пояснение")
    latency_ms: float = Field(..., ge=0.0, description="Время обработки в мс")
    flags: Dict[str, Any] = Field(default_factory=dict, description="Флаги качества")
    dataset_shape: Dict[str, int] = Field(default_factory=dict, description="Размеры датасета")
    request_id: Optional[str] = Field(None, description="Идентификатор запроса")

class DatasetSummaryResponse(BaseModel):
    """Ответ с полной сводкой датасета."""
    dataset_info: Dict[str, Any]
    quality_summary: Dict[str, Any]
    issues: Dict[str, Any]
    problematic_columns: List[str]
    column_types: Dict[str, List[str]]
    processing_time_ms: float
    request_id: Optional[str] = Field(None, description="Идентификатор запроса")

# ---------- Эндпоинты ----------

@app.get("/")
async def root(request: Request):
    """Корневой эндпоинт."""
    return {
        "service": "EDA CLI HTTP Service",
        "version": "1.0.0",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Проверка здоровья сервиса"},
            {"path": "/quality", "method": "POST", "description": "Оценка качества по агрегированным признакам"},
            {"path": "/quality-from-csv", "method": "POST", "description": "Оценка качества по CSV файлу"},
            {"path": "/quality-flags-from-csv", "method": "POST", "description": "Полные флаги качества из CSV"},
            {"path": "/summary-from-csv", "method": "POST", "description": "Полная JSON сводка датасета"},
            {"path": "/top-categories-from-csv", "method": "POST", "description": "Топ категорий из CSV"},
            {"path": "/head-from-csv", "method": "POST", "description": "Первые N строк CSV"},
            {"path": "/sample-from-csv", "method": "POST", "description": "Случайная выборка из CSV"},
            {"path": "/metrics", "method": "GET", "description": "Метрики сервиса"},
        ],
        "request_id": request.headers.get("X-Request-ID")
    }

@app.get("/health")
async def health():
    """Проверка здоровья сервиса."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0",
        "service": "eda-cli-api"
    }

@app.post("/quality", response_model=QualityResponse)
async def quality(req: QualityRequest, request: Request):
    """Оценка качества по агрегированным признакам."""
    start_time = time.time()
    
    # Используем вашу логику из compute_quality_flags
    score = 1.0
    
    # Штраф за пропуски
    score -= req.max_missing_share * 0.5
    
    # Штраф за маленький датасет
    if req.n_rows < 100:
        score -= 0.2
    
    # Штраф за много колонок
    if req.n_cols > 100:
        score -= 0.1
    
    # Нормализуем скор
    score = max(0.0, min(1.0, score))
    score = round(score, 3)
    
    latency_ms = (time.time() - start_time) * 1000
    ok_for_model = score >= 0.6
    
    # Логируем результат
    log_data = {
        "endpoint": "/quality",
        "n_rows": req.n_rows,
        "n_cols": req.n_cols,
        "ok_for_model": ok_for_model,
        "quality_score": score,
        "latency_ms": round(latency_ms, 2)
    }
    logger.info("Quality assessment completed", extra={"extra_data": log_data})
    
    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message="Оценка на основе агрегированных признаков",
        latency_ms=round(latency_ms, 2),
        flags={
            "too_few_rows": req.n_rows < 100,
            "too_many_columns": req.n_cols > 100,
            "too_many_missing": req.max_missing_share > 0.5,
        },
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
        request_id=request.headers.get("X-Request-ID")
    )

@app.post("/quality-from-csv", response_model=QualityResponse)
async def quality_from_csv(
    file: UploadFile = File(...),
    sep: str = Form(","),
    encoding: str = Form("utf-8"),
    request: Request = None
):
    """Оценка качества по CSV файлу."""
    start_time = time.time()
    
    try:
        # Читаем CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), sep=sep, encoding=encoding)
        
        # Используем функции из вашего ядра
        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df, df)
        
        latency_ms = (time.time() - start_time) * 1000
        ok_for_model = flags["quality_score"] >= 0.6
        
        # Логируем результат
        log_data = {
            "endpoint": "/quality-from-csv",
            "filename": file.filename,
            "n_rows": summary.n_rows,
            "n_cols": summary.n_cols,
            "ok_for_model": ok_for_model,
            "quality_score": flags["quality_score"],
            "latency_ms": round(latency_ms, 2),
            "has_issues": any([
                flags["too_few_rows"],
                flags["too_many_columns"],
                flags["too_many_missing"],
                flags["has_constant_columns"],
                flags["has_high_cardinality_categoricals"],
                flags["has_suspicious_id_duplicates"],
                flags["has_many_zero_values"],
            ])
        }
        logger.info("CSV quality assessment completed", extra={"extra_data": log_data})
        
        return QualityResponse(
            ok_for_model=ok_for_model,
            quality_score=flags["quality_score"],
            message="Оценка на основе полного анализа CSV",
            latency_ms=round(latency_ms, 2),
            flags=flags,
            dataset_shape={"n_rows": summary.n_rows, "n_cols": summary.n_cols},
            request_id=request.headers.get("X-Request-ID") if request else None
        )
        
    except Exception as e:
        # Логируем ошибку
        logger.error(f"Error processing CSV: {str(e)}", extra={
            "extra_data": {
                "endpoint": "/quality-from-csv",
                "filename": file.filename if hasattr(file, 'filename') else 'unknown',
                "error": str(e)
            }
        })
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка обработки CSV: {str(e)}"
        )

@app.post("/quality-flags-from-csv")
async def quality_flags_from_csv(
    file: UploadFile = File(...),
    sep: str = Form(","),
    encoding: str = Form("utf-8"),
    request: Request = None
):
    """
    Полный набор флагов качества из CSV.
    Использует все новые эвристики из HW03.
    """
    start_time = time.time()
    
    try:
        # Читаем CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), sep=sep, encoding=encoding)
        
        # Используем функции из ядра с новыми эвристиками
        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df, df)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Формируем ответ с упором на новые флаги из HW03
        response = {
            "filename": file.filename,
            "dataset_shape": {
                "n_rows": summary.n_rows,
                "n_cols": summary.n_cols
            },
            "quality_score": flags["quality_score"],
            "processing_time_ms": round(latency_ms, 2),
            "request_id": request.headers.get("X-Request-ID") if request else None,
            "flags": {
                # Существующие флаги
                "too_few_rows": flags["too_few_rows"],
                "too_many_columns": flags["too_many_columns"],
                "too_many_missing": flags["too_many_missing"],
                "max_missing_share": flags["max_missing_share"],
                
                # НОВЫЕ ФЛАГИ ИЗ HW03
                "has_constant_columns": flags["has_constant_columns"],
                "has_high_cardinality_categoricals": flags["has_high_cardinality_categoricals"],
                "has_suspicious_id_duplicates": flags["has_suspicious_id_duplicates"],
                "has_many_zero_values": flags["has_many_zero_values"],
            },
            "details": {
                "constant_columns": flags.get("constant_columns", []),
                "high_cardinality_columns": flags.get("high_cardinality_columns", []),
                "id_duplicates": flags.get("id_duplicates", {}),
                "many_zero_columns": flags.get("many_zero_columns", {}),
            }
        }
        
        # Логируем результат
        log_data = {
            "endpoint": "/quality-flags-from-csv",
            "filename": file.filename,
            "n_rows": summary.n_rows,
            "n_cols": summary.n_cols,
            "quality_score": flags["quality_score"],
            "latency_ms": round(latency_ms, 2),
            "has_constant_columns": flags["has_constant_columns"],
            "has_high_cardinality": flags["has_high_cardinality_categoricals"],
            "has_id_duplicates": flags["has_suspicious_id_duplicates"],
            "has_many_zeros": flags["has_many_zero_values"]
        }
        logger.info("Full quality flags assessment completed", extra={"extra_data": log_data})
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing CSV for quality flags: {str(e)}", extra={
            "extra_data": {
                "endpoint": "/quality-flags-from-csv",
                "filename": file.filename if hasattr(file, 'filename') else 'unknown',
                "error": str(e)
            }
        })
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка обработки CSV: {str(e)}"
        )

@app.post("/summary-from-csv", response_model=DatasetSummaryResponse)
async def summary_from_csv(
    file: UploadFile = File(...),
    sep: str = Form(","),
    encoding: str = Form("utf-8"),
    min_missing_share: float = Form(0.3),
    request: Request = None
):
    """
    Полная JSON-сводка датасета.
    Использует функцию generate_json_summary из HW03.
    """
    start_time = time.time()
    
    try:
        # Читаем CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), sep=sep, encoding=encoding)
        
        # Используем функции из ядра
        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        quality_flags = compute_quality_flags(summary, missing_df, df)
        
        # Находим проблемные колонки по пропускам
        problematic_cols = []
        if not missing_df.empty:
            problematic_cols = missing_df[
                missing_df["missing_share"] > min_missing_share
            ].index.tolist()
        
        # Генерируем JSON-сводку (функция из HW03)
        json_summary = generate_json_summary(
            summary, missing_df, quality_flags, df, problematic_cols
        )
        
        latency_ms = (time.time() - start_time) * 1000
        json_summary["processing_time_ms"] = round(latency_ms, 2)
        json_summary["request_id"] = request.headers.get("X-Request-ID") if request else None
        
        # Логируем результат
        logger.info("Dataset summary generated", extra={
            "extra_data": {
                "endpoint": "/summary-from-csv",
                "filename": file.filename,
                "n_rows": summary.n_rows,
                "n_cols": summary.n_cols,
                "processing_time_ms": round(latency_ms, 2)
            }
        })
        
        return json_summary
        
    except Exception as e:
        logger.error(f"Error generating dataset summary: {str(e)}", extra={
            "extra_data": {
                "endpoint": "/summary-from-csv",
                "filename": file.filename if hasattr(file, 'filename') else 'unknown',
                "error": str(e)
            }
        })
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка обработки CSV: {str(e)}"
        )

@app.get("/metrics")
async def get_metrics():
    """
    Эндпоинт для получения метрик сервиса.
    Возвращает статистику по работе сервиса.
    """
    metrics = metrics_store.get_metrics()
    
    # Добавляем дополнительную информацию
    metrics.update({
        "service": "eda-cli-http",
        "uptime": "N/A",  # В реальном приложении можно добавить расчет аптайма
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "log_file": "logs/api.log",
        "note": "Метрики собираются в памяти и сбрасываются при перезапуске сервиса"
    })
    
    # Логируем запрос метрик
    logger.info("Metrics requested", extra={
        "extra_data": {
            "endpoint": "/metrics",
            "total_requests": metrics["total_requests"],
            "success_rate": metrics["success_rate"]
        }
    })
    
    return metrics

@app.post("/top-categories-from-csv")
async def top_categories_from_csv(
    file: UploadFile = File(...),
    sep: str = Form(","),
    encoding: str = Form("utf-8"),
    max_columns: int = Form(5),
    top_k: int = Form(5),
    request: Request = None
):
    """
    Топ-категории для категориальных колонок.
    Использует функцию top_categories из ядра.
    """
    start_time = time.time()
    
    try:
        # Читаем CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), sep=sep, encoding=encoding)
        
        # Получаем топ категорий
        top_cats = top_categories(df, max_columns=max_columns, top_k=top_k)
        
        # Преобразуем результат для JSON
        result = {}
        for col_name, df_top in top_cats.items():
            result[col_name] = df_top.to_dict(orient="records")
        
        latency_ms = (time.time() - start_time) * 1000
        
        response = {
            "filename": file.filename,
            "parameters": {
                "max_columns": max_columns,
                "top_k": top_k
            },
            "top_categories": result,
            "processing_time_ms": round(latency_ms, 2),
            "columns_analyzed": list(top_cats.keys()),
            "request_id": request.headers.get("X-Request-ID") if request else None
        }
        
        logger.info("Top categories analysis completed", extra={
            "extra_data": {
                "endpoint": "/top-categories-from-csv",
                "filename": file.filename,
                "columns_analyzed": list(top_cats.keys()),
                "processing_time_ms": round(latency_ms, 2)
            }
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing top categories: {str(e)}", extra={
            "extra_data": {
                "endpoint": "/top-categories-from-csv",
                "filename": file.filename if hasattr(file, 'filename') else 'unknown',
                "error": str(e)
            }
        })
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка обработки CSV: {str(e)}"
        )

@app.post("/head-from-csv")
async def head_from_csv(
    file: UploadFile = File(...),
    sep: str = Form(","),
    encoding: str = Form("utf-8"),
    n: int = Form(5, ge=1, le=1000),
    request: Request = None
):
    """
    Первые N строк CSV файла.
    Аналог CLI команды head.
    """
    try:
        # Читаем CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), sep=sep, encoding=encoding)
        
        # Берем первые N строк
        head_df = df.head(n)
        
        # Конвертируем в словарь, заменяя NaN на None
        result = head_df.where(pd.notnull(head_df), None).to_dict(orient="records")
        
        response = {
            "filename": file.filename,
            "n_rows": n,
            "total_rows": len(df),
            "columns": list(df.columns),
            "data": result,
            "request_id": request.headers.get("X-Request-ID") if request else None
        }
        
        logger.info("Head rows extracted", extra={
            "extra_data": {
                "endpoint": "/head-from-csv",
                "filename": file.filename,
                "n_rows": n,
                "total_rows": len(df)
            }
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Error extracting head rows: {str(e)}", extra={
            "extra_data": {
                "endpoint": "/head-from-csv",
                "filename": file.filename if hasattr(file, 'filename') else 'unknown',
                "error": str(e)
            }
        })
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка обработки CSV: {str(e)}"
        )

@app.post("/sample-from-csv")
async def sample_from_csv(
    file: UploadFile = File(...),
    sep: str = Form(","),
    encoding: str = Form("utf-8"),
    n: int = Form(10, ge=1, le=1000),
    random_state: int = Form(42),
    request: Request = None
):
    """
    Случайная выборка из CSV файла.
    Аналог CLI команды sample.
    """
    try:
        # Читаем CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), sep=sep, encoding=encoding)
        
        # Если запрошено больше строк, чем есть в файле
        if n > len(df):
            n = len(df)
        
        # Берем случайную выборку
        sample_df = df.sample(n=n, random_state=random_state)
        
        # Конвертируем в словарь, заменяя NaN на None
        result = sample_df.where(pd.notnull(sample_df), None).to_dict(orient="records")
        
        response = {
            "filename": file.filename,
            "sample_size": n,
            "total_rows": len(df),
            "random_state": random_state,
            "sample_fraction": round(n / len(df), 4),
            "columns": list(df.columns),
            "data": result,
            "request_id": request.headers.get("X-Request-ID") if request else None
        }
        
        logger.info("Random sample extracted", extra={
            "extra_data": {
                "endpoint": "/sample-from-csv",
                "filename": file.filename,
                "sample_size": n,
                "total_rows": len(df),
                "random_state": random_state
            }
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Error extracting random sample: {str(e)}", extra={
            "extra_data": {
                "endpoint": "/sample-from-csv",
                "filename": file.filename if hasattr(file, 'filename') else 'unknown',
                "error": str(e)
            }
        })
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка обработки CSV: {str(e)}"
        )