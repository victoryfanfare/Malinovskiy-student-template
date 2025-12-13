"""
Клиент для тестирования EDA CLI HTTP сервиса.
Отправляет несколько запросов к разным эндпоинтам и выводит сводку.
"""

import httpx
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime
import sys

BASE_URL = "http://localhost:8000"

class EDAClient:
    """Клиент для взаимодействия с EDA API"""
    
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
        self.results = []
    
    def test_health(self):
        """Тест эндпоинта health"""
        print("Тестирование /health...")
        try:
            response = self.client.get(f"{self.base_url}/health")
            result = {
                "endpoint": "/health",
                "status": response.status_code,
                "latency_ms": response.elapsed.total_seconds() * 1000,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            self.results.append(result)
            
            print(f"Статус: {response.status_code}")
            print(f"Время: {result['latency_ms']:.2f} мс")
            if result['success']:
                print(f"Данные: {json.dumps(result['data'], indent=2)}")
            print()
            return result
        except Exception as e:
            print(f"Ошибка: {e}")
            print()
            return None
    
    def test_quality(self, n_rows=1000, n_cols=10, missing_share=0.1):
        """Тест эндпоинта quality"""
        print(f"Тестирование /quality (n_rows={n_rows}, n_cols={n_cols})...")
        try:
            data = {
                "n_rows": n_rows,
                "n_cols": n_cols,
                "max_missing_share": missing_share,
                "numeric_cols": n_cols // 2,
                "categorical_cols": n_cols // 2
            }
            
            response = self.client.post(f"{self.base_url}/quality", json=data)
            result_data = response.json() if response.status_code == 200 else None
            
            result = {
                "endpoint": "/quality",
                "status": response.status_code,
                "latency_ms": response.elapsed.total_seconds() * 1000,
                "success": response.status_code == 200,
                "quality_score": result_data.get("quality_score") if result_data else None,
                "ok_for_model": result_data.get("ok_for_model") if result_data else None,
                "data": result_data
            }
            self.results.append(result)
            
            print(f"Статус: {response.status_code}")
            print(f"Время: {result['latency_ms']:.2f} мс")
            if result['success']:
                print(f"Quality Score: {result['quality_score']:.3f}")
                print(f"OK for model: {result['ok_for_model']}")
                print(f"Флагов: {len(result_data.get('flags', {}))}")
            print()
            return result
        except Exception as e:
            print(f"Ошибка: {e}")
            print()
            return None
    
    def test_quality_from_csv(self, csv_path):
        """Тест эндпоинта quality-from-csv"""
        print(f"Тестирование /quality-from-csv с файлом {csv_path.name}...")
        try:
            with open(csv_path, "rb") as f:
                response = self.client.post(
                    f"{self.base_url}/quality-from-csv",
                    files={"file": (csv_path.name, f, "text/csv")},
                    data={"sep": ",", "encoding": "utf-8"}
                )
            
            result_data = response.json() if response.status_code == 200 else None
            
            result = {
                "endpoint": "/quality-from-csv",
                "status": response.status_code,
                "latency_ms": response.elapsed.total_seconds() * 1000,
                "success": response.status_code == 200,
                "filename": csv_path.name,
                "quality_score": result_data.get("quality_score") if result_data else None,
                "ok_for_model": result_data.get("ok_for_model") if result_data else None,
                "data": result_data
            }
            self.results.append(result)
            
            print(f"Статус: {response.status_code}")
            print(f"Время: {result['latency_ms']:.2f} мс")
            if result['success']:
                print(f"Quality Score: {result['quality_score']:.3f}")
                print(f"OK for model: {result['ok_for_model']}")
                
                # Показываем новые флаги из HW03
                flags = result_data.get('flags', {})
                print(f"   Новые флаги из HW03:")
                print(f"      - Константные колонки: {flags.get('has_constant_columns', 'N/A')}")
                print(f"      - Высокая кардинальность: {flags.get('has_high_cardinality_categoricals', 'N/A')}")
                print(f"      - Дубликаты ID: {flags.get('has_suspicious_id_duplicates', 'N/A')}")
                print(f"      - Много нулей: {flags.get('has_many_zero_values', 'N/A')}")
            print()
            return result
        except Exception as e:
            print(f"Ошибка: {e}")
            print()
            return None
    
    def test_quality_flags_from_csv(self, csv_path):
        """Тест нового эндпоинта quality-flags-from-csv"""
        print(f"Тестирование /quality-flags-from-csv с файлом {csv_path.name}...")
        try:
            with open(csv_path, "rb") as f:
                response = self.client.post(
                    f"{self.base_url}/quality-flags-from-csv",
                    files={"file": (csv_path.name, f, "text/csv")},
                    data={"sep": ",", "encoding": "utf-8"}
                )
            
            result_data = response.json() if response.status_code == 200 else None
            
            result = {
                "endpoint": "/quality-flags-from-csv",
                "status": response.status_code,
                "latency_ms": response.elapsed.total_seconds() * 1000,
                "success": response.status_code == 200,
                "filename": csv_path.name,
                "quality_score": result_data.get("quality_score") if result_data else None,
                "data": result_data
            }
            self.results.append(result)
            
            print(f"Статус: {response.status_code}")
            print(f"Время: {result['latency_ms']:.2f} мс")
            if result['success']:
                print(f"   📈 Quality Score: {result['quality_score']:.3f}")
                
                # Показываем подробности
                flags = result_data.get('flags', {})
                details = result_data.get('details', {})
                
                print(f"Основные флаги:")
                for flag_name, flag_value in flags.items():
                    if flag_name not in ['quality_score', 'max_missing_share']:
                        print(f"      - {flag_name}: {flag_value}")
                
                if details.get('constant_columns'):
                    print(f"Константные колонки: {', '.join(details['constant_columns'])}")
                
                if details.get('id_duplicates'):
                    print(f"Дубликаты ID:")
                    for col, info in details['id_duplicates'].items():
                        print(f"      - {col}: {info.get('duplicate_count', 0)} дубликатов")
            print()
            return result
        except Exception as e:
            print(f"Ошибка: {e}")
            print()
            return None
    
    def test_metrics(self):
        """Тест эндпоинта metrics"""
        print("Тестирование /metrics...")
        try:
            response = self.client.get(f"{self.base_url}/metrics")
            result_data = response.json() if response.status_code == 200 else None
            
            result = {
                "endpoint": "/metrics",
                "status": response.status_code,
                "latency_ms": response.elapsed.total_seconds() * 1000,
                "success": response.status_code == 200,
                "data": result_data
            }
            self.results.append(result)
            
            print(f"Статус: {response.status_code}")
            print(f"Время: {result['latency_ms']:.2f} мс")
            if result['success']:
                print(f" Всего запросов: {result_data.get('total_requests', 0)}")
                print(f" Успешных: {result_data.get('successful_requests', 0)}")
                print(f" Неуспешных: {result_data.get('failed_requests', 0)}")
                print(f" Успешность: {result_data.get('success_rate', 0)}%")
                print(f" Среднее время: {result_data.get('avg_latency_ms', 0):.2f} мс")
            print()
            return result
        except Exception as e:
            print(f" Ошибка: {e}")
            print()
            return None
    
    def print_summary(self):
        """Вывод сводки по всем тестам"""
        print("=" * 80)
        print("СВОДКА ПО ТЕСТИРОВАНИЮ")
        print("=" * 80)
        
        if not self.results:
            print("Нет результатов для отображения")
            return
        
        successful = sum(1 for r in self.results if r.get('success'))
        total = len(self.results)
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print(f"Успешных тестов: {successful}/{total} ({success_rate:.1f}%)")
        print()
        
        # Сводка по эндпоинтам
        print("Результаты по эндпоинтам:")
        print("-" * 80)
        
        for result in self.results:
            status_icon = "+" if result.get('success') else "-"
            endpoint = result.get('endpoint', 'unknown')
            status = result.get('status', 'N/A')
            latency = result.get('latency_ms', 0)
            
            quality_info = ""
            if result.get('quality_score') is not None:
                quality_info = f" | Quality: {result['quality_score']:.3f}"
            
            print(f"{status_icon} {endpoint:30} | Статус: {status:4} | Время: {latency:6.2f} мс{quality_info}")
        
        print()
        
        # Сводка по времени
        avg_latency = sum(r.get('latency_ms', 0) for r in self.results) / total if total > 0 else 0
        max_latency = max((r.get('latency_ms', 0) for r in self.results), default=0)
        min_latency = min((r.get('latency_ms', 0) for r in self.results if r.get('latency_ms', 0) > 0), default=0)
        
        print("⏱️  Статистика по времени:")
        print(f"   Среднее: {avg_latency:.2f} мс")
        print(f"   Минимальное: {min_latency:.2f} мс")
        print(f"   Максимальное: {max_latency:.2f} мс")
        print()
        
        # Проверка новых флагов из HW03
        print("Проверка новых эвристик из HW03:")
        hw03_flags_present = False
        
        for result in self.results:
            if result.get('success') and result.get('data'):
                data = result['data']
                flags = data.get('flags', {}) if isinstance(data, dict) else {}
                
                hw03_flags = [
                    'has_constant_columns',
                    'has_high_cardinality_categoricals', 
                    'has_suspicious_id_duplicates',
                    'has_many_zero_values'
                ]
                
                for flag in hw03_flags:
                    if flag in flags:
                        hw03_flags_present = True
                        print(f" {flag} присутствует в ответе от {result['endpoint']}")
        
        if not hw03_flags_present:
            print(" Новые флаги из HW03 не обнаружены")
        
        print()
        print("=" * 80)
        print(f"Тестирование завершено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

def create_test_files():
    """Создание тестовых CSV файлов"""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    files = []
    
    # 1. Простой датасет
    simple_df = pd.DataFrame({
        "id": list(range(100)),
        "value": [i * 1.5 for i in range(100)],
        "category": ["A", "B", "C", "D"] * 25,
        "score": [i % 100 for i in range(100)]
    })
    simple_path = test_dir / "simple_dataset.csv"
    simple_df.to_csv(simple_path, index=False)
    files.append(simple_path)
    
    # 2. Датасет с проблемами (для тестирования новых эвристик)
    problem_df = pd.DataFrame({
        "user_id": [1, 2, 3, 1, 4, 5, 2, 6, 7, 8] * 10,  # Дубликаты ID
        "constant_feature": [0.5] * 100,  # Константная колонка
        "zero_feature": [0] * 60 + list(range(40)),  # 60% нулей
        "high_card_feature": list(range(100)),  # Высокая кардинальность (100 уникальных)
        "normal_feature": [i * 0.1 for i in range(100)],
        "category": ["A", "B", "C", "D", "E"] * 20,
        "missing_values": [1 if i % 10 == 0 else None for i in range(100)]  # 10% пропусков
    })
    problem_path = test_dir / "problem_dataset.csv"
    problem_df.to_csv(problem_path, index=False)
    files.append(problem_path)
    
    # 3. Маленький датасет
    small_df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["Moscow", "SPb", "Kazan"]
    })
    small_path = test_dir / "small_dataset.csv"
    small_df.to_csv(small_path, index=False)
    files.append(small_path)
    
    print(f"Создано {len(files)} тестовых файлов в папке {test_dir}")
    return files

def main():
    """Основная функция"""
    print("=" * 80)
    print("🚀 КЛИЕНТ ДЛЯ ТЕСТИРОВАНИЯ EDA CLI HTTP СЕРВИСА")
    print("=" * 80)
    print()
    
    # Проверяем, запущен ли сервис
    print("🔍 Проверка доступности сервиса...")
    try:
        client = EDAClient()
        health_result = client.test_health()
        
        if not health_result or not health_result.get('success'):
            print("❌ Сервис не доступен!")
            print(f"   Убедитесь, что сервис запущен: uv run uvicorn eda_cli.api:app --reload --port 8000")
            sys.exit(1)
        
        print("✅ Сервис доступен!")
        print()
        
        # Создаем тестовые файлы
        print("Создание тестовых файлов...")
        test_files = create_test_files()
        print()
        
        # Запускаем тесты
        print("Запуск тестов...")
        print()
        
        # Тест 1: Эндпоинт quality с разными параметрами
        client.test_quality(n_rows=1000, n_cols=10, missing_share=0.1)
        client.test_quality(n_rows=50, n_cols=20, missing_share=0.5)  # Маленький датасет с пропусками
        client.test_quality(n_rows=5000, n_cols=5, missing_share=0.01)  # Большой датасет
        
        # Тест 2: Эндпоинты с CSV файлами
        for csv_file in test_files:
            client.test_quality_from_csv(csv_file)
            client.test_quality_flags_from_csv(csv_file)  # Новый эндпоинт из HW04
        
        # Тест 3: Эндпоинт metrics
        client.test_metrics()
        
        # Выводим сводку
        client.print_summary()
        
        # Показываем ссылки на документацию
        print()
        print("Дополнительная информация:")
        print(f"   Swagger UI: {BASE_URL}/docs")
        print(f"   ReDoc: {BASE_URL}/redoc")
        print(f"   Метрики: {BASE_URL}/metrics")
        print(f"   Логи: logs/api.log")
        
    except httpx.ConnectError:
        print("Не удалось подключиться к сервису!")
        print(f"   Убедитесь, что сервис запущен: uv run uvicorn eda_cli.api:app --reload --port 8000")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()