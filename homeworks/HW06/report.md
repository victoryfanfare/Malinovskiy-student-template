# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

# Датасет 02: Сложная бинарная классификация
- **Выбранный датасет**: `S06-hw-dataset-02.csv`
- **Размер**: (18000 строк, 39 столбцов)
- **Целевая переменная**: `target` (бинарная классификация, 0/1)
- **Распределение классов**:
  - Класс 0: 13273 образцов (73.74%)
  - Класс 1: 4727 образцов (26.26%)
- **Признаки**:
  - 37 числовых признаков (float64)
  - 1 столбец `id` (int64)
  - 2 признака `x_int_1`, `x_int_2` (float64, но с целочисленными значениями)
- **Пропуски**: отсутствуют

# Датасет 04: Дисбалансированная классификация
- **Выбранный датасет**: `S06-hw-dataset-04.csv`
- **Размер**: (25000 строк, 62 столбца)
- **Целевая переменная**: `target` (бинарная классификация, 0/1)
- **Распределение классов**:
  - Класс 0: 23770 образцов (95.08%)
  - Класс 1: 1230 образцов (4.92%)
- **Коэффициент дисбаланса**: 19.3 : 1 (сильный дисбаланс)
- **Признаки**:
  - 60 числовых признаков (float64)
  - 1 столбец `id` (int64)
- **Пропуски**: отсутствуют

## 2. Protocol

- **Разбиение**: train/test = 75% / 25%
- **Random_state**: 42 (для воспроизводимости)
- **Стратификация**: Да (сохранение пропорций классов)

# Подбор гиперпараметров:
- **Метод**: GridSearchCV
- **Кросс-валидация**:
  - Decision Tree — 5 фолдов (StratifiedKFold)
  - Random Forest, Gradient Boosting — 3 фолда
- **Оптимизируемые метрики**:
  - Для датасета 02: ROC-AUC
  - Для датасета 04: Average Precision (учитывает дисбаланс)

# Метрики качества:
1. **Accuracy** — доля правильных предсказаний
2. **F1-score** — баланс между precision и recall
3. **ROC-AUC** — для сбалансированных данных (датасет 02)
4. **Average Precision** — для дисбалансированных данных (датасет 04)

## 3. Models

### Baseline модели (для обоих датасетов):

1. **DummyClassifier**
   - Стратегия: `most_frequent`
   - Цель: установка нижней границы качества

2. **LogisticRegression**
   - Pipeline со StandardScaler
   - Для датасета 04: `class_weight='balanced'`

# Модели недели 6

# Для датасета 02:

1. **DecisionTreeClassifier**
   - Контроль сложности: `max_depth`, `min_samples_leaf`, `ccp_alpha`
   - Подбираемые параметры:
     - `max_depth` = [3, 5, 7, None]
     - `min_samples_leaf` = [1, 5, 10]
     - `ccp_alpha` = [0, 0.001, 0.01]
     - `criterion` = ['gini', 'entropy']
   - **Лучшие параметры**:
     ```
     {'ccp_alpha': 0.001, 'criterion': 'entropy',
      'max_depth': None, 'min_samples_leaf': 10,
      'min_samples_split': 2}
     ```

2. **RandomForestClassifier**
   - Подбираемые параметры:
     - `n_estimators` = [100, 200]
     - `max_depth` = [5, 10, None]
     - `min_samples_leaf` = [1, 5, 10]
     - `max_features` = ['sqrt', 'log2', 0.5]
     - `bootstrap` = [True, False]
   - **Лучшие параметры**:
     ```
     {'bootstrap': False, 'max_depth': None,
      'max_features': 'sqrt', 'min_samples_leaf': 1,
      'n_estimators': 200}
     ```

3. **GradientBoostingClassifier**
   - Подбираемые параметры:
     - `n_estimators` = [100, 200]
     - `learning_rate` = [0.01, 0.05, 0.1]
     - `max_depth` = [3, 5, 7]
     - `min_samples_leaf` = [1, 5, 10]
     - `subsample` = [0.8, 1.0]
   - **Лучшие параметры**:
     ```
     {'learning_rate': 0.05, 'max_depth': 7,
      'min_samples_leaf': 5, 'n_estimators': 200,
      'subsample': 0.8}
     ```

4. **StackingClassifier** (опционально)
   - Базовые модели: лучшие LogisticRegression, RandomForest, GradientBoosting
   - Метамодель: LogisticRegression
   - `cv = 5`

# Для датасета 04:

1. **DecisionTreeClassifier**
   - Учет дисбаланса: `class_weight = [None, 'balanced']`
   - **Лучшие параметры**:
     ```
     {'ccp_alpha': 0.001, 'class_weight': None,
      'criterion': 'entropy', 'max_depth': None,
      'min_samples_leaf': 10, 'min_samples_split': 2}
     ```

2. **RandomForestClassifier**
   - Учет дисбаланса: `class_weight = ['balanced', 'balanced_subsample']`
   - **Лучшие параметры**:
     ```
     {'class_weight': 'balanced', 'max_depth': None,
      'max_features': 'sqrt', 'min_samples_leaf': 5,
      'n_estimators': 200}
     ```

3. **GradientBoostingClassifier**
   - Без встроенной обработки дисбаланса
   - **Лучшие параметры**:
     ```
     {'learning_rate': 0.05, 'max_depth': 5,
      'min_samples_leaf': 20, 'n_estimators': 200,
      'subsample': 0.8}
     ```

4. **StackingClassifier** (опционально)
   - Аналогично датасету 02, но с balanced-метамоделью

## 4. Results

# Датасет 02 (сбалансированный):

| Model | Test Accuracy | Test F1 | Test ROC-AUC | Test Avg Precision |
|------|---------------|---------|--------------|--------------------|
| Dummy Classifier | 0.7373 | 0.0000 | 0.5000 | 0.2627 |
| Logistic Regression | 0.8160 | 0.5714 | 0.8009 | 0.6597 |
| Decision Tree | 0.8187 | 0.6205 | 0.8289 | 0.6795 |
| Random Forest | 0.9002 | 0.7817 | 0.9307 | 0.8785 |
| Gradient Boosting | 0.9093 | 0.8106 | 0.9314 | 0.8812 |
| Stacking Classifier | **0.9153** | **0.8291** | **0.9340** | **0.8815** |

**Победитель для датасета 02**: **Stacking Classifier**

# Датасет 04 (дисбалансированный):

| Model | Test Accuracy | Test F1 | Test Avg Precision | Test ROC-AUC |
|------|---------------|---------|--------------------|--------------|
| Dummy Classifier | 0.9509 | 0.0000 | 0.0491 | 0.5000 |
| Logistic Regression (balanced) | 0.7787 | 0.2576 | 0.4575 | 0.8418 |
| Decision Tree | 0.9678 | 0.5659 | 0.5572 | 0.8474 |
| Random Forest (balanced) | 0.9757 | 0.6724 | 0.7891 | 0.9105 |
| Gradient Boosting | **0.9802** | **0.7500** | 0.7924 | 0.9004 |
| Stacking Classifier | 0.9603 | 0.6612 | **0.8018** | **0.9120** |

**Победитель для датасета 04**: **Stacking Classifier (по Average Precision)**

## 5. Analysis

### Устойчивость моделей

Для проверки устойчивости выполнено 5 прогонов с разными `random_state`.

**Датасет 02**:
- Random Forest: Accuracy = 0.8996 ± 0.0023, F1 = 0.7793 ± 0.0059
- Gradient Boosting: Accuracy = 0.9070 ± 0.0045, F1 = 0.8052 ± 0.0111

**Датасет 04**:
- Random Forest (balanced): Accuracy = 0.9748 ± 0.0005, F1 = 0.6597 ± 0.0091

**Вывод**: модели демонстрируют хорошую устойчивость; Random Forest более стабилен.

### Ошибки лучших моделей

**Датасет 02 (Stacking Classifier)**:
- Accuracy: 0.9153
- F1-score: 0.8291
- Умеренное переобучение (разница train/test accuracy ≈ 0.0847)

**Датасет 04 (Stacking Classifier)**:
- Accuracy: 0.9603
- F1-score: 0.6612
- Average Precision: 0.8018
- Минимальное переобучение (≈ 0.0244)

### Интерпретация (Permutation Importance)

Результаты сохранены в файлах:
- `permutation_importance_02.json`
- `permutation_importance_04.json`

**Выводы**:
- Для датасета 02 выявлены устойчивые и интерпретируемые признаки
- Для датасета 04 модель находит более сложные закономерности из-за сильного дисбаланса
- Важность признаков существенно различается между датасетами

## 6. Conclusion

### Ключевые выводы

1. **Эффективность ансамблей**
   - Stacking Classifier показал лучшие результаты на обоих датасетах
   - Ансамбли значительно превосходят одиночные модели

2. **Учет дисбаланса**
   - Accuracy вводит в заблуждение при сильном дисбалансе
   - F1-score и Average Precision — приоритетные метрики
   - `class_weight='balanced'` критичен для деревьев и леса

3. **Контроль сложности**
   - `ccp_alpha=0.001` эффективно снижает переобучение
   - Для дисбалансированных данных нужны более консервативные параметры

4. **Честный эксперимент**
   - Фиксированный random_state
   - Стратификация
   - GridSearchCV с кросс-валидацией
   - Сохранение всех артефактов

5. **Практические рекомендации**
   - Всегда начинать с baseline
   - Использовать ансамбли для табличных данных
   - Выбирать метрики под бизнес-задачу
   - Документировать и сохранять все этапы эксперимента
