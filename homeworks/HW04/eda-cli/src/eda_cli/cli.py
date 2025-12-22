from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
    generate_json_summary,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
    plot_top_categories_chart,
    plot_boxplots_numeric,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Обзор датасета: {Path(path).name}")
    typer.echo("=" * 50)
    typer.echo(f"Строк: {summary.n_rows:,}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    
    # Подсчитываем типы колонок
    numeric_count = sum(1 for col in summary.columns if col.is_numeric)
    categorical_count = sum(1 for col in summary.columns if col.is_categorical)
    other_count = summary.n_cols - numeric_count - categorical_count
    
    typer.echo(f"Числовых колонок: {numeric_count}")
    typer.echo(f"Категориальных колонок: {categorical_count}")
    typer.echo(f"Прочих колонок: {other_count}")
    
    typer.echo("\nОбзор колонок:")
    typer.echo("=" * 50)
    typer.echo(summary_df.to_string(index=False, max_rows=20))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    # Основные параметры
    max_hist_columns: int = typer.Option(
        6, 
        help="Максимум числовых колонок для гистограмм."
    ),
    # НОВЫЕ ПАРАМЕТРЫ
    top_k_categories: int = typer.Option(
        5,
        help="Сколько top-значений выводить для категориальных признаков."
    ),
    report_title: str = typer.Option(
        "EDA-отчёт",
        help="Заголовок отчёта."
    ),
    min_missing_share: float = typer.Option(
        0.3,
        help="Порог доли пропусков, выше которого колонка считается проблемной."
    ),
    max_categories: int = typer.Option(
        5,
        help="Максимум категориальных колонок для анализа."
    ),
    # Дополнительные параметры
    include_boxplots: bool = typer.Option(
        False,
        help="Включить боксплоты для числовых колонок."
    ),
    include_category_charts: bool = typer.Option(
        False,
        help="Включить графики для категориальных признаков."
    ),
    json_summary: bool = typer.Option(
        False,
        help="Сохранить JSON-сводку по датасету."
    ),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # Показываем прогресс
    with typer.progressbar([
        "Загрузка данных",
        "Анализ колонок", 
        "Анализ пропусков",
        "Анализ качества",
        "Генерация графиков",
        "Создание отчёта"
    ], label="Генерация отчёта") as progress:
        # 1. Обзор
        progress.update(0)
        summary = summarize_dataset(df)
        summary_df = flatten_summary_for_print(summary)
        
        progress.update(1)
        missing_df = missing_table(df)
        
        progress.update(2)
        corr_df = correlation_matrix(df)
        
        progress.update(3)
        # Используем новый параметр max_categories для top_categories
        top_cats = top_categories(df, max_columns=max_categories, top_k=top_k_categories)

        # 2. Качество в целом
        quality_flags = compute_quality_flags(summary, missing_df, df)

        # 3. Находим проблемные колонки по пропускам
        problematic_cols = []
        if not missing_df.empty:
            problematic_cols = missing_df[
                missing_df["missing_share"] > min_missing_share
            ].index.tolist()

        # 4. Сохраняем табличные артефакты
        summary_df.to_csv(out_root / "summary.csv", index=False)
        if not missing_df.empty:
            missing_df.to_csv(out_root / "missing.csv", index=True)
        if not corr_df.empty:
            corr_df.to_csv(out_root / "correlation.csv", index=True)
        save_top_categories_tables(top_cats, out_root / "top_categories")

        # 5. Генерация JSON-сводки (если запрошено)
        if json_summary:
            json_summ = generate_json_summary(
                summary, missing_df, quality_flags, df, problematic_cols
            )
            with open(out_root / "summary.json", "w", encoding="utf-8") as f:
                json.dump(json_summ, f, indent=2, ensure_ascii=False)

        # 6. Генерация графиков
        progress.update(4)
        plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
        plot_missing_matrix(df, out_root / "missing_matrix.png")
        plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")
        
        if include_boxplots:
            plot_boxplots_numeric(df, out_root / "boxplots_numeric.png")
        
        if include_category_charts and top_cats:
            plot_top_categories_chart(top_cats, out_root)

        # 7. Markdown-отчёт с новыми параметрами
        progress.update(5)
        md_path = out_root / "report.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write(f"# {report_title}\n\n")
            f.write(f"## Информация о файле\n\n")
            f.write(f"- **Файл**: `{Path(path).name}`\n")
            f.write(f"- **Разделитель**: `{sep}`\n")
            f.write(f"- **Кодировка**: `{encoding}`\n\n")
            
            f.write(f"## Параметры генерации\n\n")
            f.write(f"- **top_k_categories**: `{top_k_categories}`\n")
            f.write(f"- **min_missing_share**: `{min_missing_share:.1%}`\n")
            f.write(f"- **max_categories**: `{max_categories}`\n")
            f.write(f"- **max_hist_columns**: `{max_hist_columns}`\n")
            f.write(f"- **include_boxplots**: `{include_boxplots}`\n")
            f.write(f"- **include_category_charts**: `{include_category_charts}`\n")
            f.write(f"- **json_summary**: `{json_summary}`\n\n")
            
            f.write(f"## Общая информация\n\n")
            f.write(f"- **Строк**: `{summary.n_rows:,}`\n")
            f.write(f"- **Столбцов**: `{summary.n_cols}`\n")
            
            # Подсчитываем типы колонок
            numeric_count = sum(1 for col in summary.columns if col.is_numeric)
            categorical_count = sum(1 for col in summary.columns if col.is_categorical)
            
            f.write(f"- **Числовых колонок**: `{numeric_count}`\n")
            f.write(f"- **Категориальных колонок**: `{categorical_count}`\n")
            f.write(f"- **Прочих колонок**: `{summary.n_cols - numeric_count - categorical_count}`\n\n")

            f.write("## Качество данных (эвристики)\n\n")
            f.write(f"### Общая оценка: **{quality_flags['quality_score']:.3f}/1.000**\n\n")
            
            # Основные метрики
            f.write("### Основные метрики\n\n")
            f.write(f"- Максимальная доля пропусков: **{quality_flags['max_missing_share']:.2%}**\n")
            f.write(f"- Слишком мало строк (<100): **{quality_flags['too_few_rows']}**\n")
            f.write(f"- Слишком много колонок (>100): **{quality_flags['too_many_columns']}**\n")
            f.write(f"- Слишком много пропусков (>50%): **{quality_flags['too_many_missing']}**\n\n")
            
            # Новые эвристики
            f.write("### Новые эвристики (HW03)\n\n")
            
            # Константные колонки
            f.write(f"#### Константные колонки\n")
            f.write(f"- Есть константные колонки: **{quality_flags['has_constant_columns']}**\n")
            if quality_flags['has_constant_columns']:
                f.write(f"- Константные колонки: `{', '.join(quality_flags['constant_columns'])}`\n")
            f.write("\n")
            
            # Высокая кардинальность
            f.write(f"#### Высокая кардинальность (>100 уникальных)\n")
            f.write(f"- Есть категории с высокой кардинальностью: **{quality_flags['has_high_cardinality_categoricals']}**\n")
            if quality_flags['has_high_cardinality_categoricals']:
                f.write("- Колонки с высокой кардинальностью:\n")
                for col_info in quality_flags['high_cardinality_columns']:
                    f.write(f"  - `{col_info['column']}`: {col_info['unique_count']:,} уникальных значений\n")
            f.write("\n")
            
            # Дубликаты ID
            f.write(f"#### Дубликаты ID\n")
            f.write(f"- Есть дубликаты ID: **{quality_flags['has_suspicious_id_duplicates']}**\n")
            if quality_flags['has_suspicious_id_duplicates']:
                f.write("- Проблемные ID-колонки:\n")
                for col, info in quality_flags['id_duplicates'].items():
                    f.write(f"  - `{col}`: {info['duplicate_count']} дубликатов ({info['duplicate_share']:.1%} от уникальных)\n")
            f.write("\n")
            
            # Много нулей
            f.write(f"#### Много нулей (>30%)\n")
            f.write(f"- Есть колонки с избытком нулей: **{quality_flags['has_many_zero_values']}**\n")
            if quality_flags['has_many_zero_values']:
                f.write("- Колонки с избытком нулей:\n")
                for col, info in quality_flags['many_zero_columns'].items():
                    f.write(f"  - `{col}`: {info['zero_share']:.1%} нулей (порог: {info['threshold']:.0%})\n")
            f.write("\n")
            
            f.write("## Проблемные колонки\n\n")
            if problematic_cols:
                f.write(f"Колонки с большой долей пропусков (>**{min_missing_share:.0%}**):\n")
                for col in problematic_cols:
                    share = missing_df.loc[col, "missing_share"]
                    f.write(f"- **`{col}`**: {share:.1%} пропусков\n")
            else:
                f.write("Нет колонок с большой долей пропусков.\n")
            f.write("\n")

            f.write("## Детальная информация по колонкам\n\n")
            f.write("Полная информация доступна в файле `summary.csv`.\n\n")

            f.write("## Пропуски\n\n")
            if missing_df.empty:
                f.write("Пропусков нет или датасет пуст.\n\n")
            else:
                f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

            f.write("## Корреляция числовых признаков\n\n")
            if corr_df.empty:
                f.write("ℹНедостаточно числовых колонок для корреляции.\n\n")
            else:
                f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

            f.write("## Категориальные признаки\n\n")
            if not top_cats:
                f.write("ℹКатегориальные/строковые признаки не найдены.\n\n")
            else:
                f.write(f"Топ-**{top_k_categories}** категорий по колонкам (всего анализировано **{max_categories}** колонок):\n")
                f.write("См. файлы в папке `top_categories/`.\n")
                if include_category_charts:
                    f.write("А также графики в `top_categories_*.png`.\n")
                f.write("\n")

            f.write("## Графики\n\n")
            f.write(f"### Гистограммы\n")
            f.write(f"Построены для первых **{max_hist_columns}** числовых колонок.\n")
            f.write("См. файлы `hist_*.png`.\n\n")
            
            if include_boxplots:
                f.write(f"### Боксплоты\n")
                f.write("См. файл `boxplots_numeric.png`.\n\n")
            
            f.write("### Матрица пропусков\n")
            f.write("См. файл `missing_matrix.png`.\n\n")
            
            f.write("### Тепловая карта корреляции\n")
            f.write("См. файл `correlation_heatmap.png`.\n\n")
            
            if json_summary:
                f.write("## JSON-сводка\n\n")
                f.write("См. файл `summary.json`.\n\n")

            f.write("---\n")
            f.write(f"*Отчёт сгенерирован с помощью `eda-cli`*\n")
            f.write(f"*Параметры: top_k={top_k_categories}, missing_threshold={min_missing_share:.0%}*\n")

    typer.echo(f"\nОтчёт успешно сгенерирован в каталоге: {out_root}")
    typer.echo(f"Основной отчёт: {md_path}")
    typer.echo("Табличные файлы:")
    typer.echo("  - summary.csv")
    if not missing_df.empty:
        typer.echo("  - missing.csv")
    if not corr_df.empty:
        typer.echo("  - correlation.csv")
    if top_cats:
        typer.echo("  - top_categories/*.csv")
    if json_summary:
        typer.echo("  - summary.json")
    
    typer.echo("Графики:")
    typer.echo("  - hist_*.png")
    typer.echo("  - missing_matrix.png")
    typer.echo("  - correlation_heatmap.png")
    if include_boxplots:
        typer.echo("  - boxplots_numeric.png")
    if include_category_charts and top_cats:
        typer.echo("  - top_categories_*.png")


@app.command()
def head(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    n: int = typer.Option(5, help="Количество строк для вывода."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Вывести первые N строк CSV-файла.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    
    typer.echo(f"Первые {n} строк из файла '{Path(path).name}':")
    typer.echo("=" * 60)
    
    # Используем rich для красивого вывода таблицы
    try:
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        
        # Добавляем колонки
        for col in df.columns[:10]:  # Ограничиваем количество колонок для отображения
            table.add_column(col)
        
        # Добавляем строки
        for i in range(min(n, len(df))):
            row_values = []
            for col in df.columns[:10]:
                val = df.iloc[i][col]
                if pd.isna(val):
                    row_values.append("[red]NaN[/red]")
                elif isinstance(val, (int, float)) and not isinstance(val, bool):
                    row_values.append(f"{val:.2f}" if abs(val) < 10000 else f"{val:,.0f}")
                else:
                    row_values.append(str(val)[:30])  # Обрезаем длинные строки
            
            table.add_row(*row_values)
        
        console.print(table)
        
        # Выводим статистику
        typer.echo(f"\nСтатистика:")
        typer.echo(f"  - Всего строк: {len(df):,}")
        typer.echo(f"  - Всего колонок: {len(df.columns)}")
        typer.echo(f"  - Показано строк: {min(n, len(df))}")
        typer.echo(f"  - Показано колонок: {min(10, len(df.columns))}")
        
        if len(df.columns) > 10:
            typer.echo(f"  - [yellow]Предупреждение:[/yellow] Показаны только первые 10 из {len(df.columns)} колонок")
        
    except ImportError:
        # Если rich не установлен, используем простой вывод
        typer.echo(df.head(n).to_string(index=False))
        typer.echo(f"\nВсего строк: {len(df):,}, колонок: {len(df.columns)}")


@app.command()
def sample(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    n: int = typer.Option(10, help="Количество строк для вывода."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    random_state: int = typer.Option(42, help="Seed для воспроизводимости."),
) -> None:
    """
    Вывести случайную выборку N строк CSV-файла.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    
    if n > len(df):
        typer.echo(f"  Запрошено {n} строк, но в файле только {len(df)}")
        n = len(df)
    
    sample_df = df.sample(n=n, random_state=random_state)
    
    typer.echo(f" Случайная выборка {n} строк из файла '{Path(path).name}':")
    typer.echo("=" * 60)
    typer.echo(sample_df.to_string(index=False))
    
    typer.echo(f"\n Статистика выборки:")
    typer.echo(f"  - Всего строк в файле: {len(df):,}")
    typer.echo(f"  - Размер выборки: {n}")
    typer.echo(f"  - Random seed: {random_state}")
    typer.echo(f"  - Доля выборки: {n/len(df):.1%}")


def main():
    """Точка входа в приложение."""
    app()


if __name__ == "__main__":
    main()