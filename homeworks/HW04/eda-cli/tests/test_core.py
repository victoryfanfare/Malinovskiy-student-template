from __future__ import annotations

import pandas as pd
import json
import io
from pathlib import Path
import tempfile
import pytest
from fastapi.testclient import TestClient

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
    generate_json_summary,
)

from eda_cli.api import app

client = TestClient(app)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_compute_quality_flags_new_heuristics():
    """Тест новых эвристик качества данных (HW03 требование)"""
    
    # Тест 1: DataFrame с константной колонкой
    df_const = pd.DataFrame({
        'id': [1, 2, 3],
        'constant_col': [10, 10, 10],  # константная колонка
        'normal_col': [1, 2, 3]
    })
    
    summary = summarize_dataset(df_const)
    missing_df = missing_table(df_const)
    flags = compute_quality_flags(summary, missing_df, df_const)
    
    assert flags['has_constant_columns'] == True
    assert 'constant_col' in flags['constant_columns']
    assert flags['quality_score'] < 1.0  # качество должно быть снижено
    
    # Тест 2: DataFrame с дубликатами ID
    df_dup = pd.DataFrame({
        'user_id': [101, 102, 101, 103],  # дубликат 101
        'item_id': [1, 2, 3, 4],
        'value': [10, 20, 30, 40]
    })
    
    summary = summarize_dataset(df_dup)
    missing_df = missing_table(df_dup)
    flags = compute_quality_flags(summary, missing_df, df_dup)
    
    assert flags['has_suspicious_id_duplicates'] == True
    assert 'user_id' in flags['id_duplicates']
    assert flags['id_duplicates']['user_id']['duplicate_count'] == 2
    
    # Тест 3: DataFrame с нулевыми значениями
    df_zeros = pd.DataFrame({
        'col1': [0, 0, 0, 1, 2],  # 60% нулей (>30%)
        'col2': [0, 0, 1, 2, 3],  # 40% нулей (>30%)
        'col3': [1, 2, 3, 4, 5]   # 0% нулей
    })
    
    summary = summarize_dataset(df_zeros)
    missing_df = missing_table(df_zeros)
    flags = compute_quality_flags(summary, missing_df, df_zeros)
    
    assert flags['has_many_zero_values'] == True
    assert 'col1' in flags['many_zero_columns']
    assert 'col2' in flags['many_zero_columns']
    assert 'col3' not in flags['many_zero_columns']
    
    # Проверяем правильность расчета доли нулей
    assert flags['many_zero_columns']['col1']['zero_share'] == 0.6
    assert flags['many_zero_columns']['col2']['zero_share'] == 0.4
    
    # Тест 4: DataFrame с высокой кардинальностью
    df_high_card = pd.DataFrame({
        'id': list(range(150)),  # 150 уникальных значений
        'category': ['A'] * 50 + ['B'] * 50 + ['C'] * 50,  # 3 категории
        'value': list(range(150))
    })
    
    summary = summarize_dataset(df_high_card)
    missing_df = missing_table(df_high_card)
    flags = compute_quality_flags(summary, missing_df, df_high_card)
    
    # id имеет 150 уникальных значений, что > порога 100
    assert flags['has_high_cardinality_categoricals'] == True
    high_card_cols = [col['column'] for col in flags['high_cardinality_columns']]
    assert 'id' in high_card_cols
    assert 'category' not in high_card_cols  # только 3 уникальных значения
    
    # Тест 5: Все проблемы вместе
    df_all_problems = pd.DataFrame({
        'id': [1, 1, 2],  # дубликаты
        'constant': [5, 5, 5],  # константа
        'zeros': [0, 0, 1],  # много нулей
        'many_unique': list(range(150))  # высокая кардинальность
    })
    
    summary = summarize_dataset(df_all_problems)
    missing_df = missing_table(df_all_problems)
    flags = compute_quality_flags(summary, missing_df, df_all_problems)
    
    # Проверяем все флаги
    assert flags['has_constant_columns'] == True
    assert flags['has_suspicious_id_duplicates'] == True
    assert flags['has_many_zero_values'] == True
    assert flags['has_high_cardinality_categoricals'] == True
    
    # Quality score должен быть очень низким
    assert flags['quality_score'] < 0.5


def test_generate_json_summary():
    """Тест генерации JSON-сводки"""
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'C', 'B']
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    quality_flags = compute_quality_flags(summary, missing_df, df)
    problematic_cols = []
    
    json_summ = generate_json_summary(summary, missing_df, quality_flags, df, problematic_cols)
    
    # Проверяем структуру JSON
    assert 'dataset_info' in json_summ
    assert 'quality_summary' in json_summ
    assert 'issues' in json_summ
    assert 'problematic_columns' in json_summ
    assert 'column_types' in json_summ
    
    # Проверяем конкретные значения
    assert json_summ['dataset_info']['n_rows'] == 5
    assert json_summ['dataset_info']['n_cols'] == 3
    assert 'quality_score' in json_summ['quality_summary']
    
    # Проверяем, что новые эвристики присутствуют в issues
    assert 'has_constant_columns' in json_summ['issues']
    assert 'has_high_cardinality_categoricals' in json_summ['issues']
    assert 'has_suspicious_id_duplicates' in json_summ['issues']
    assert 'has_many_zero_values' in json_summ['issues']


def test_top_categories_with_parameters():
    """Тест функции top_categories с параметрами"""
    df = pd.DataFrame({
        'category1': ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'E', 'F', 'G'],
        'category2': ['X', 'X', 'Y', 'Y', 'Z', 'Z', 'X', 'Y', 'Z', 'X'],
        'numeric': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    
    # Тест с ограничением количества колонок
    top_cats = top_categories(df, max_columns=1, top_k=3)
    assert len(top_cats) == 1  # только первая категориальная колонка
    assert 'category1' in top_cats
    
    # Тест с ограничением top_k
    top_cats = top_categories(df, max_columns=2, top_k=2)
    assert len(top_cats) == 2
    assert 'category1' in top_cats
    assert 'category2' in top_cats
    assert len(top_cats['category1']) == 2  # только top-2 значения
    assert len(top_cats['category2']) == 2


def test_summary_with_zero_share():
    """Тест расчета доли нулей в ColumnSummary"""
    df = pd.DataFrame({
        'col_with_zeros': [0, 0, 1, 2, 3],
        'col_without_zeros': [1, 2, 3, 4, 5],
        'string_col': ['A', 'B', 'C', 'D', 'E']
    })
    
    summary = summarize_dataset(df)
    
    # Находим колонки
    col_zeros = next(c for c in summary.columns if c.name == 'col_with_zeros')
    col_no_zeros = next(c for c in summary.columns if c.name == 'col_without_zeros')
    col_string = next(c for c in summary.columns if c.name == 'string_col')
    
    # Проверяем zero_share для числовых колонок
    assert col_zeros.zero_share == 0.4  # 2 нуля из 5
    assert col_no_zeros.zero_share == 0.0  # нет нулей
    assert col_string.zero_share is None  # не числовая колонка


# ---------- НОВЫЕ ТЕСТЫ ДЛЯ API (HW04) ----------

def test_api_health_endpoint():
    """Тест эндпоинта /health"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_api_quality_endpoint():
    """Тест эндпоинта /quality"""
    request_data = {
        "n_rows": 1000,
        "n_cols": 10,
        "max_missing_share": 0.1,
        "numeric_cols": 5,
        "categorical_cols": 3
    }
    
    response = client.post("/quality", json=request_data)
    assert response.status_code == 200
    data = response.json()
    
    assert "quality_score" in data
    assert "ok_for_model" in data
    assert "latency_ms" in data
    assert "flags" in data
    assert "dataset_shape" in data
    assert isinstance(data["quality_score"], float)
    assert 0 <= data["quality_score"] <= 1


def test_api_quality_from_csv_endpoint():
    """Тест эндпоинта /quality-from-csv"""
    # Создаем тестовый CSV
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value": [10, 20, 30, 40, 50],
        "category": ["A", "B", "A", "C", "B"]
    })
    
    csv_bytes = df.to_csv(index=False).encode()
    
    response = client.post(
        "/quality-from-csv",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
        data={"sep": ",", "encoding": "utf-8"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Проверяем, что возвращаются флаги из HW03
    flags = data["flags"]
    assert "has_constant_columns" in flags
    assert "has_high_cardinality_categoricals" in flags
    assert "has_suspicious_id_duplicates" in flags
    assert "has_many_zero_values" in flags


def test_api_quality_flags_from_csv_endpoint():
    """Тест нового эндпоинта /quality-flags-from-csv (основной для HW04)"""
    # Создаем CSV с проблемами для тестирования новых эвристик
    df = pd.DataFrame({
        "user_id": [1, 2, 1, 3, 4],  # Дубликаты ID
        "constant_col": [5, 5, 5, 5, 5],  # Константная колонка
        "zeros_col": [0, 0, 0, 1, 2],  # 60% нулей
        "category": list(range(150))[:5]  # Высокая кардинальность (но ограничено 5 строками)
    })
    
    csv_bytes = df.to_csv(index=False).encode()
    
    response = client.post(
        "/quality-flags-from-csv",
        files={"file": ("test_flags.csv", csv_bytes, "text/csv")},
        data={"sep": ",", "encoding": "utf-8"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Проверяем структуру ответа
    assert "flags" in data
    assert "details" in data
    assert "quality_score" in data
    
    # Проверяем новые флаги из HW03
    flags = data["flags"]
    assert "has_constant_columns" in flags
    assert "has_suspicious_id_duplicates" in flags
    assert "has_many_zero_values" in flags
    
    # Должно быть True для константной колонки и дубликатов ID
    assert flags["has_constant_columns"] is True
    assert flags["has_suspicious_id_duplicates"] is True
    assert flags["has_many_zero_values"] is True
    
    # Проверяем детали
    details = data["details"]
    assert "constant_columns" in details
    assert "id_duplicates" in details
    assert "many_zero_columns" in details


def test_api_summary_from_csv_endpoint():
    """Тест эндпоинта /summary-from-csv"""
    df = pd.DataFrame({
        "id": list(range(50)),
        "feature1": [i * 1.5 for i in range(50)],
        "category": ["A", "B", "C"] * 16 + ["A", "B"]
    })
    
    csv_bytes = df.to_csv(index=False).encode()
    
    response = client.post(
        "/summary-from-csv",
        files={"file": ("test_summary.csv", csv_bytes, "text/csv")},
        data={"sep": ",", "encoding": "utf-8", "min_missing_share": 0.3}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Проверяем структуру JSON-сводки
    assert "dataset_info" in data
    assert "quality_summary" in data
    assert "issues" in data
    assert "column_types" in data
    
    # Проверяем, что новые эвристики присутствуют
    issues = data["issues"]
    assert "has_constant_columns" in issues
    assert "has_high_cardinality_categoricals" in issues
    assert "has_suspicious_id_duplicates" in issues
    assert "has_many_zero_values" in issues


def test_api_invalid_csv():
    """Тест обработки невалидного CSV"""
    # Отправляем не CSV файл
    response = client.post(
        "/quality-from-csv",
        files={"file": ("test.txt", b"not a csv content", "text/plain")}
    )
    
    # Должен вернуть ошибку 400
    assert response.status_code == 400


def test_api_empty_csv():
    """Тест обработки пустого CSV"""
    df = pd.DataFrame()
    csv_bytes = df.to_csv(index=False).encode()
    
    response = client.post(
        "/quality-from-csv",
        files={"file": ("empty.csv", csv_bytes, "text/csv")}
    )
    
    # Должен вернуть ошибку 400
    assert response.status_code == 400


def test_api_top_categories_from_csv():
    """Тест эндпоинта /top-categories-from-csv"""
    df = pd.DataFrame({
        "city": ["Moscow", "Moscow", "SPb", "Moscow", "SPb", "Kazan"] * 5,
        "category": ["A", "B", "A", "A", "B", "C"] * 5,
        "value": list(range(30))
    })
    
    csv_bytes = df.to_csv(index=False).encode()
    
    response = client.post(
        "/top-categories-from-csv",
        files={"file": ("test_categories.csv", csv_bytes, "text/csv")},
        data={"sep": ",", "encoding": "utf-8", "max_columns": 2, "top_k": 3}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "top_categories" in data
    assert "columns_analyzed" in data
    assert "city" in data["top_categories"]
    assert "category" in data["top_categories"]


def test_api_metrics_endpoint():
    """Тест эндпоинта /metrics"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    
    assert "service" in data
    assert "endpoints_available" in data
    assert "total_requests" in data
    assert "avg_latency_ms" in data
    assert isinstance(data["total_requests"], int)


def test_api_head_from_csv():
    """Тест эндпоинта /head-from-csv"""
    df = pd.DataFrame({
        "col1": list(range(20)),
        "col2": [f"value_{i}" for i in range(20)]
    })
    
    csv_bytes = df.to_csv(index=False).encode()
    
    response = client.post(
        "/head-from-csv",
        files={"file": ("test_head.csv", csv_bytes, "text/csv")},
        data={"sep": ",", "encoding": "utf-8", "n": 5}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert data["n_rows"] == 5
    assert len(data["data"]) == 5


def test_api_sample_from_csv():
    """Тест эндпоинта /sample-from-csv"""
    df = pd.DataFrame({
        "id": list(range(30)),
        "value": [i * 2 for i in range(30)]
    })
    
    csv_bytes = df.to_csv(index=False).encode()
    
    response = client.post(
        "/sample-from-csv",
        files={"file": ("test_sample.csv", csv_bytes, "text/csv")},
        data={"sep": ",", "encoding": "utf-8", "n": 10, "random_state": 42}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert data["sample_size"] == 10
    assert data["random_state"] == 42
    assert len(data["data"]) == 10