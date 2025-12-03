from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence
import json

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    is_categorical: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    zero_share: Optional[float] = None  # доля нулей для числовых колонок

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета по колонкам:
    - количество строк/столбцов;
    - типы;
    - пропуски;
    - количество уникальных;
    - несколько примерных значений;
    - базовые числовые статистики (для numeric).
    """
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        # Примерные значения выводим как строки
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        is_categorical = bool(ptypes.is_categorical_dtype(s) or ptypes.is_object_dtype(s))
        
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None
        zero_share_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())
            # Доля нулей
            zero_count = (s == 0).sum()
            zero_share_val = float(zero_count / n_rows) if n_rows > 0 else 0.0

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                is_categorical=is_categorical,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
                zero_share=zero_share_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: count/share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляция Пирсона для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для категориальных/строковых колонок считает top-k значений.
    Возвращает словарь: колонка -> DataFrame со столбцами value/count/share.
    """
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result


def compute_quality_flags(summary: DatasetSummary, missing_df: pd.DataFrame, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Простейшие эвристики «качества» данных:
    - слишком много пропусков;
    - подозрительно мало строк;
    - константные колонки;
    - высокая кардинальность категориальных признаков;
    - дубликаты в ID-колонках;
    - много нулей в числовых колонках.
    """
    flags: Dict[str, Any] = {}
    
    #Существующие эвристики
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5
    
    #НОВЫЕ ЭВРИСТИКИ
    
    # 2.1. Проверка на константные колонки
    constant_columns = []
    for col in df.columns:
        if df[col].nunique(dropna=True) == 1:
            constant_columns.append(col)
    flags["has_constant_columns"] = len(constant_columns) > 0
    flags["constant_columns"] = constant_columns
    
    # 2.2. Проверка на высокую кардинальность категориальных признаков
    high_cardinality_cols = []
    card_threshold = 100  # порог для высокой кардинальности
    
    for col_summary in summary.columns:
        if col_summary.is_categorical and col_summary.unique > card_threshold:
            high_cardinality_cols.append({
                "column": col_summary.name,
                "unique_count": col_summary.unique,
                "threshold": card_threshold
            })
    
    flags["has_high_cardinality_categoricals"] = len(high_cardinality_cols) > 0
    flags["high_cardinality_columns"] = high_cardinality_cols
    
    # 2.3. Проверка на дубликаты в ID-подобных колонках
    id_duplicates = {}
    # Считаем, что колонки с 'id' в названии - это идентификаторы
    for col in df.columns:
        if 'id' in col.lower() or col.endswith('_id'):
            duplicate_mask = df.duplicated(subset=[col], keep=False)
            duplicates = df[duplicate_mask]
            if not duplicates.empty:
                duplicate_count = duplicates[col].nunique()
                total_unique = df[col].nunique()
                id_duplicates[col] = {
                    "duplicate_count": int(duplicate_count),
                    "total_unique": int(total_unique),
                    "duplicate_share": float(duplicate_count / total_unique) if total_unique > 0 else 0.0
                }
    
    flags["has_suspicious_id_duplicates"] = len(id_duplicates) > 0
    flags["id_duplicates"] = id_duplicates
    
    # 2.4. Проверка на много нулей в числовых колонках
    zero_threshold = 0.3  # 30% нулей
    many_zero_cols = {}
    
    for col_summary in summary.columns:
        if col_summary.is_numeric and col_summary.zero_share is not None:
            if col_summary.zero_share > zero_threshold:
                many_zero_cols[col_summary.name] = {
                    "zero_share": col_summary.zero_share,
                    "threshold": zero_threshold
                }
    
    flags["has_many_zero_values"] = len(many_zero_cols) > 0
    flags["many_zero_columns"] = many_zero_cols
    
    # 3. Расчёт скор-а качества с учётом новых эвристик
    score = 1.0
    
    # Штрафы за разные проблемы
    if flags["too_few_rows"]:
        score -= 0.2
    if flags["too_many_columns"]:
        score -= 0.1
    
    # Штраф за пропуски пропорционален их доле
    score -= max_missing_share * 0.5
    
    # Штрафы за новые эвристики
    if flags["has_constant_columns"]:
        score -= 0.15
    if flags["has_high_cardinality_categoricals"]:
        score -= 0.1
    if flags["has_suspicious_id_duplicates"]:
        score -= 0.2
    if flags["has_many_zero_values"]:
        score -= 0.1
    
    # Гарантируем диапазон 0-1
    score = max(0.0, min(1.0, score))
    flags["quality_score"] = round(score, 3)

    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """
    Превращает DatasetSummary в табличку для более удобного вывода.
    """
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        row = {
            "name": col.name,
            "dtype": col.dtype,
            "non_null": col.non_null,
            "missing": col.missing,
            "missing_share": f"{col.missing_share:.2%}",
            "unique": col.unique,
            "is_numeric": col.is_numeric,
            "is_categorical": col.is_categorical,
        }
        
        if col.is_numeric:
            row.update({
                "min": f"{col.min:.2f}" if col.min is not None else "N/A",
                "max": f"{col.max:.2f}" if col.max is not None else "N/A",
                "mean": f"{col.mean:.2f}" if col.mean is not None else "N/A",
                "std": f"{col.std:.2f}" if col.std is not None else "N/A",
                "zero_share": f"{col.zero_share:.2%}" if col.zero_share is not None else "N/A",
            })
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_json_summary(
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    quality_flags: Dict[str, Any],
    df: pd.DataFrame,
    problematic_cols: List[str]
) -> Dict[str, Any]:
    """
    Генерирует JSON-сводку по датасету.
    """
    json_summary = {
        "dataset_info": {
            "n_rows": summary.n_rows,
            "n_cols": summary.n_cols,
            "file_size": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        },
        "quality_summary": {
            "quality_score": quality_flags["quality_score"],
            "max_missing_share": quality_flags["max_missing_share"],
            "has_issues": any([
                quality_flags["too_few_rows"],
                quality_flags["too_many_columns"],
                quality_flags["too_many_missing"],
                quality_flags["has_constant_columns"],
                quality_flags["has_high_cardinality_categoricals"],
                quality_flags["has_suspicious_id_duplicates"],
                quality_flags["has_many_zero_values"],
            ])
        },
        "issues": {
            "too_few_rows": quality_flags["too_few_rows"],
            "too_many_columns": quality_flags["too_many_columns"],
            "too_many_missing": quality_flags["too_many_missing"],
            "has_constant_columns": {
                "flag": quality_flags["has_constant_columns"],
                "columns": quality_flags["constant_columns"]
            },
            "has_high_cardinality_categoricals": {
                "flag": quality_flags["has_high_cardinality_categoricals"],
                "columns": quality_flags["high_cardinality_columns"]
            },
            "has_suspicious_id_duplicates": {
                "flag": quality_flags["has_suspicious_id_duplicates"],
                "details": quality_flags["id_duplicates"]
            },
            "has_many_zero_values": {
                "flag": quality_flags["has_many_zero_values"],
                "columns": quality_flags["many_zero_columns"]
            }
        },
        "problematic_columns": problematic_cols,
        "column_types": {
            "numeric": [col.name for col in summary.columns if col.is_numeric],
            "categorical": [col.name for col in summary.columns if col.is_categorical],
            "other": [col.name for col in summary.columns if not col.is_numeric and not col.is_categorical]
        }
    }
    
    return json_summary