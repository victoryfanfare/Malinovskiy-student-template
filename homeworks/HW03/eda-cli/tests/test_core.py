from __future__ import annotations

import pandas as pd
import json
from pathlib import Path

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
    generate_json_summary,
)


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