> **WindScope · интерактивное демо раннего предупреждения · 17.11.2026**
>
> Новое приложение находится в [windscope/](windscope/). Запуск из этой папки: `./scripts/start_demo.ps1`; основной экран — [http://localhost:3100](http://localhost:3100).
>
> [Инструкция демонстрации, результаты и ограничения](windscope/docs/demo_release.md). Персональная модель вычисляет ранние сигналы во время исторического replay. Полная проверка: **4 из 11 отказов**, семь пропусков; score не является вероятностью. Исходный исследовательский проект ниже сохранён для сравнения.

# SCADA Predictive Maintenance — MVP

**CARE H7 · Wind Farm A · 16.11.2026**

Цель: по наблюдаемым SCADA-показателям оценивать риск **подтверждённой неисправности оборудования в ближайшие 7 или 30 дней**. Сначала проверяем возможность достоверно сформировать метки; только затем обучаем модели. MVP не предназначен для управления реальным оборудованием и не заменяет промышленную систему диагностики.

## Streamlit Dashboard · CARE H7

1. Откройте PowerShell в корне проекта. Если окружения ещё нет, выполните `python -m venv .venv` с Python 3.11–3.14.
2. Установите зависимости и запустите Dashboard.

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

   В активированном окружении команда запуска — `streamlit run app.py`. Откройте [локальный Dashboard](http://localhost:8501).
3. В **Overview** выберите турбину и исторический файл. **SCADA Monitoring** показывает исходные датчики и позволяет сравнить два канала. **Failure Prediction** показывает архивные score обеих моделей, пороги validation и первое сохранённое предупреждение. **Model Explainability** показывает gain, глобальный и локальный TreeSHAP и историю значимых датчиков.
4. Выберите **Воспроизведение** и двигайте ползунок UTC. Прогнозы и отметки отказа появляются по времени исторического среза. Показания учитывают десятиминутную задержку доступности. Итоговые метрики теста и глобальные объяснения доступны в режиме **История**.

Dashboard читает `results/care_h7_windows_20261116/training/` и `data/CARE_To_Compare/Wind Farm A/datasets/`. Для пересчёта локального SHAP выбранного прогноза используются сохранённые `pipeline.joblib` обеих моделей и `cache/care_h7_v6/features.parquet`. При отсутствии кеша доступны архивные локальные объяснения отдельных тестовых точек. При отсутствии исходного CSV для тестовых файлов доступен почасовой архив части датчиков. Для переноса можно указать переменные окружения `SCADA_DASHBOARD_RUN` (папка запуска, содержащая `training/`), `SCADA_DASHBOARD_RAW` (папка Wind Farm A), `SCADA_DASHBOARD_FEATURES` (файл Parquet с признаками). Загружайте `pipeline.joblib` только из доверенного локального эксперимента.

Сохранённые прогнозы покрывают validation и test. Score некалиброваны. При порогах validation Logistic Regression обнаружила один из двух тестовых отказов, LightGBM — ноль из двух. SHAP объясняет оценки LightGBM. Для установления причин отказов нужны отдельные данные. [Отчёт эксперимента](results/care_h7_windows_20261116/report.md)

Локальные проверки Dashboard — `.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_dashboard.py -v`.

## CARE H7 в Windows

Первая исследовательская тренировка на Wind Farm A выполнена локально с сохранённым кешем `cache/care_h7_v6`. Разбиение и параметры берутся из `configs/care_h7.json`. Отчёт содержит результаты обеих моделей, пороги, оговорки и ссылки на сохранённые артефакты. [Результаты эксперимента](results/care_h7_windows_20261116/report.md)

Выполняйте из корня проекта в PowerShell. Для нового окружения сначала запустите `python -m venv .venv` с Python 3.11–3.14 и установите зависимости командой `.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt`.

```powershell
$python = '.\.venv\Scripts\python.exe'
$run = "results/care_h7_windows_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
& $python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
& $python -m scada_pdm.cli train --config configs/care_h7.json --cache cache/care_h7_v6 --output "$run/data_gate" --audit-only
if ($LASTEXITCODE -ne 0) { throw 'Data gate failed' }
$gate = Get-Content -Raw "$run/data_gate/metrics.json" | ConvertFrom-Json
if ($gate.horizons.h7.status -ne 'eligible') { throw 'H7 is ineligible. See the data gate report.' }
& $python -m scada_pdm.cli train --config configs/care_h7.json --cache cache/care_h7_v6 --output "$run/training"
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
& $python -m scada_pdm.cli plot-scada --run "$run/training"
```

В `training/h7/` находятся модели вместе с предобработкой, прогнозы и разбор предупреждений. Метрики записаны в `training/metrics.json` и `training/metrics.csv`. Графики SCADA по тестовым файлам находятся в `training/scada_plots/`. Для их повторного построения укажите новый каталог в `plot-scada --output`.

## CARE H7 в Google Colab

Для первого реального эксперимента используется CARE v6, Wind Farm A, горизонт семь дней. Проверено независимое разбиение по турбинам и повторам телеметрии. Test содержит одну турбину и два размеченных эпизода. [Аудит H7 и ограничения оценки](docs/care_h7_audit.md)

1. Скопируйте `scada_pdm/`, `scripts/`, `configs/`, `notebooks/` и `requirements-ml.txt` в `MyDrive/scada-pdm-mvp/`. При наличии данных загрузите `data/CARE_To_Compare/Wind Farm A/` с обоими файлами метаданных и 22 CSV в `datasets/`. Ячейка 3 может получить недостающие файлы адресными запросами к ZIP. [CARE v6 на Zenodo](https://zenodo.org/records/15846963)
2. Откройте [CARE_H7_Training_Colab.ipynb](notebooks/CARE_H7_Training_Colab.ipynb) через **File → Open notebook → Google Drive**. В **Runtime → Change runtime type** выберите CPU. Оставьте `RUN_TRAINING = False`.
3. Запустите ячейки **1–5**. Они подключат Drive, установят зависимости, проверят данные, соберут Parquet-кеш, разметят H7 и проверят утечки между выборками. В ячейке 5 проверьте статус H7 и размеры train, validation и test.
4. Если статус `eligible`, вставьте отображённый путь кеша в `EXISTING_CACHE`, установите `RUN_TRAINING = True` и выполните ячейки **1–7**. Метрики Logistic Regression и LightGBM, предобработка, графики SHAP, gain и история score с показаниями датчиков сохраняются в `MyDrive/scada-pdm-runs/care-h7/<run_id>/`.

Команда локальной проверки разбиения принимает `--audit-only`. Порядок команд приведён в [отчёте H7](docs/care_h7_audit.md#локальные-проверки-и-запуск). Конфигурация и [ноутбук EDP](notebooks/SCADA_Training_Colab.ipynb) доступны отдельно.

## EDP SCADA в Google Colab

В публичном архиве Mendeley Version 1 доступны два изображения журнала отказов. Их сверенная таблица из 28 записей уже входит в проект. Для расчёта покрытия H7/H30 нужно получить исходную телеметрию за оба года. [Mendeley Version 1](https://data.mendeley.com/datasets/zjxjnjp3xs/1) · [Аудит доступных файлов](docs/edp_data_audit.md#проверка-mendeley-version-1)

1. **Загрузите проект в Drive.** Скопируйте `scada_pdm/`, `configs/`, `sources/`, `notebooks/` и `requirements-ml.txt` в `MyDrive/scada-pdm-mvp/` с сохранением структуры. Журнал `sources/edp_mendeley_v1_failures.csv` копируется вместе с исходниками. Скачайте годовые SCADA из EDP и положите их в `MyDrive/scada-pdm-mvp/data/edp/mendeley_v1/` под именами `signals_2016.xlsx` и `signals_2017.xlsx`. Если получили CSV, укажите фактический путь и разделитель в [конфигурации](configs/edp_mendeley_v1.json). При доступном прямом URL ячейка 3 может скачать файл сама после настройки `url` и `max_bytes` в конфигурации. [SCADA 2016](https://edp.com/en/innovation/data/wind-farm-1-wind-turbine-scada-signals-2016) · [SCADA 2017](https://edp.com/en/innovation/data/wind-farm-1-wind-turbine-scada-signals-2017)
2. **Откройте Notebook.** В Colab выберите **File → Open notebook → Google Drive**, затем `scada-pdm-mvp/notebooks/SCADA_Training_Colab.ipynb`. В **Runtime → Change runtime type** выберите CPU.
3. **Запустите ячейки 1–5 по порядку.** Ячейка 1 подключает Drive, 2 устанавливает зависимости, 3 проверяет входы или скачивает файлы по настроенным URL, 4 создаёт Parquet и `failure_review_queue.csv`, 5 формирует отчёт по train/validation/test. При первом запуске оставьте `RUN_TRAINING = False`. Если файла не хватает, ячейка 3 напечатает нужный путь.
4. **Подтвердите метки до обучения.** Сверьте эпизоды, период полноты журнала, восстановление и известные простои. Укажите подготовленные CSV в конфигурации и повторите ячейки 1–5 с пустым `EXISTING_CACHE`, чтобы пересчитать метки. Прошедшие проверку горизонты получат статус `eligible`. [Поля для сверки](docs/edp_data_audit.md#подготовка-исходных-edp-и-подтверждение-эпизодов)
5. **Обучите модели в Colab.** Укажите проверенный кеш в `EXISTING_CACHE`, поменяйте `RUN_TRAINING = True` и выполните ячейки 1–7. Ячейка 5 обучит Logistic Regression и LightGBM для пригодных горизонтов, 6 покажет метрики и кривые, 7 — gain и TreeSHAP. Кеш и отчёты сохраняются в `MyDrive/scada-pdm-runs/<run_id>/`. [Notebook](notebooks/SCADA_Training_Colab.ipynb)

Из корня проекта, в отдельном окружении Python 3.11–3.14:

```bash
python -m pip install -r requirements-ml.txt
python -m unittest discover -s tests -v
python -m scada_pdm.cli smoke --output results/edp_smoke_repeat
```

Для проверки готовности оригинальных файлов локально запустите:

```bash
python -m scada_pdm.cli download --config configs/edp_mendeley_v1.json --output results/edp_v1_sources.json
python -m scada_pdm.cli prepare --config configs/edp_mendeley_v1.json --cache cache/edp_v1
python -m scada_pdm.cli train --config configs/edp_mendeley_v1.json --cache cache/edp_v1 --output results/edp_v1_audit --audit-only
```

В режиме готового кеша используйте конфигурацию, с которой он был создан. После изменения файлов сверки пересоздайте кеш. Score хранится как некалиброванная оценка риска.

## Что уже сделано

- Проверены официальная статья/FAQ CARE v6 и исходники трёх GitHub-решений.
- Выявлен методологический риск готового 14-дневного Notebook: использован `event_start_id` (старт аномалии), а физический отказ относится к `event_end`.
- Для Wind Farm A `status_type_id` частично реконструирован по журналу будущих отказов. Использование в качестве online-признака или фильтра предсказываемых строк недопустимо.
- Проверены 22 оригинальных файла SCADA Wind Farm A из CARE v6. Итоги H7/H30, примеры и решение по данным приведены в [локальном аудите](docs/local_data_gate.md).
- Добавлена частичная загрузка Wind Farm A и проверка границ событий по timestamp и ID строки.
- Подготовлены [исходный аудит данных](docs/data_audit.md) и [план реализации](docs/implementation_plan.md).

## Выбранные источники

1. [CARE v6 / Zenodo](https://zenodo.org/records/15846963): 95 событий, 45 событий с неисправностями, 50 нормальных; Wind Farm A — 22 события. Используем прежде всего как benchmark и потенциальный источник supervised-окон.
2. [EDP SCADA / Mendeley v2](https://data.mendeley.com/datasets/zjxjnjp3xs/2): 2016–2017, реальные журналы 28 отказов и рядов SCADA. **Сильный резервный кандидат в основной supervised-датасет** при проблемах с метками CARE.
3. [EnergyFaultDetector](https://github.com/AEFDI/EnergyFaultDetector): готовый anomaly-detector, предобработка, CARE-score; **не** готовый 7/30-дневный failure classifier.
4. [CARE PDM v2](https://github.com/PeterImoniose/care-wind-turbine-predictive-maintenance-v2): образец структуры пайплайна / кэша / валидации; задача преимущественно anomaly detection.
5. [14-day failure notebook](https://github.com/diegotamargo/Wind-turbine-failure-predictor): концепция supervised baseline, **метки требуют исправления и переоценки**.

## Структура проекта

```
scada-pdm-mvp/
├── README.md
├── .gitignore
├── requirements-ml.txt                # зависимости CPU-пайплайна
├── configs/                          # оригинальные EDP и справочный фрагмент
├── sources/edp_mendeley_v1_failures.csv
├── scada_pdm/                        # загрузка, разметка, обучение и SHAP
├── notebooks/CARE_H7_Training_Colab.ipynb
├── notebooks/SCADA_Training_Colab.ipynb
├── docs/
│   ├── data_audit.md                  # подробные findings / риски / источники
│   ├── edp_data_audit.md              # проверка EDP и результаты smoke
│   ├── care_h7_audit.md              # дубли, разметка и разбиение CARE H7
│   ├── implementation_plan.md         # этапы и критерии готовности
│   ├── codex_next_task.md             # исходное задание для локального аудита
│   └── local_data_gate.md             # результаты и решение по H7/H30
├── scripts/
│   ├── care_readiness_audit.py        # проверка метаданных и покрытия H7/H30
│   └── fetch_care_a.py                # адресная загрузка файлов Wind Farm A
└── tests/
    ├── test_care_readiness_audit.py
    ├── test_edp_pipeline.py
    ├── test_farm_a_h7.py
    └── test_fetch_care_a.py
```

Папки `data/`, `cache/`, `artifacts/`, `results/` будут созданы локально по мере необходимости и исключены из Git.

## Быстрая проверка CARE без датасета

Работает на Python 3.10+ без установки пакетов:

```bash
python -m unittest discover -s tests -p "test_*care*.py" -v
python scripts/care_readiness_audit.py --help
```

## Данные Wind Farm A и запуск аудита

Команды выполняются из папки `scada-pdm-mvp` в PowerShell или Bash. Если `data/CARE_To_Compare/Wind Farm A` уже подготовлена, начните с аудита. Для загрузки Wind Farm A нужен доступ к Zenodo и около 0,81 ГБ на диске. Загрузчик запрашивает отдельные диапазоны ZIP, проверяет CRC-32 каждого файла и переиспользует уже проверенные файлы при повторном запуске. Источник зафиксирован на [CARE v6, Zenodo 15846963](https://zenodo.org/records/15846963).

```bash
python scripts/fetch_care_a.py --output data
python scripts/care_readiness_audit.py --raw-dir data/CARE_To_Compare --farm A --horizons 7 30 --output results/care_a_metadata.json
python scripts/care_readiness_audit.py --raw-dir data/CARE_To_Compare --farm A --scan-series --horizons 7 30 --output results/care_a_coverage.json
```

Аудит считает метки внутри отдельных файлов. Для `anomaly` отсчёт идёт от `event_end`, а строки с `t >= event_end` исключаются. Ноль требует полного окна внутри размеченного нормального интервала. Разрывы 10-минутного ряда и короткий горизонт наблюдения ведут к цензурированию. В JSON есть количества строк и событий, сверка границ по ID, а также пересечения временных диапазонов файлов той же турбины. Команды для получения всего архива и проверки MD5 приведены в [локальном аудите](docs/local_data_gate.md#полный-архив-care-v6).

## Google Colab

В обычном CPU runtime с Python 3.10+ распакуйте исходники проекта в `/content/scada-pdm-mvp`. Запустите эту ячейку. Если Wind Farm A уже лежит на Google Drive, укажите её путь в `raw_dir` и пропустите вызов `fetch_care_a.py`.

```python
from pathlib import Path
import subprocess
import sys

project = Path("/content/scada-pdm-mvp")
raw_dir = project / "data" / "CARE_To_Compare"
report = project / "results" / "care_a_coverage.json"

subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*care*.py", "-v"],
               cwd=project, check=True)
subprocess.run([sys.executable, "scripts/fetch_care_a.py", "--output", str(raw_dir.parent)],
               cwd=project, check=True)
subprocess.run([sys.executable, "scripts/care_readiness_audit.py", "--raw-dir", str(raw_dir),
                "--farm", "A", "--scan-series", "--horizons", "7", "30", "--output", str(report)],
               cwd=project, check=True)
```

Перед завершением сессии скачайте JSON или сохраните его на подключённый Drive. Для аудита CARE достаточно стандартной библиотеки Python. [Справка Google Colab](https://research.google.com/colaboratory/faq.html)

## Обучение и интерфейс — после data gate

- Базовая модель: Logistic Regression, затем LightGBM (на лаговых/агрегированных признаках).
- Разделение по независимым событиям/турбинам, пороги на train/validation; отдельный untouched test.
- Метрики: PR-AUC, ROC-AUC, recall отказов, ложные тревоги/турбина-месяц, lead time, калибровка.
- Интерфейс: Streamlit/Plotly, режим исторического replay; прогноз строго из информации, доступной в момент t.

**Не выдавать аномальность за вероятность отказа.** Надпись «риск 70%» допускается только после оценки и калибровки на независимых событиях.

