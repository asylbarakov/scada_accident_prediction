# Развёртывание и проверка SCADA ML baseline SWTFPS

Дата: **16 ноября 2026 года**. Область работы: существующая pretrained-модель, последовательный inference и offline-анализ двух исторических сценариев. Обучение, FastAPI, frontend, dashboard, WebSocket, ESP32, ThingSpeak и Oracle APEX в эту работу не входят.

Исходная архитектурная оценка: [research_and_architecture.md](research_and_architecture.md). Настоящий отчёт дополняет её runtime-проверкой, а не заменяет результаты исследования.

## 1. Источники и структура проекта

Использованы [GitHub SWTFPS](https://github.com/SaraMahyan/SWTFPS_Wind-Turbine-Failure-Prediction), [Zenodo v1.0.0](https://zenodo.org/records/23441231), [исследовательская статья](https://wes.copernicus.org/articles/11/3983/2026/) и [Kaggle Wind Farm A](https://www.kaggle.com/datasets/azizkasimov/wind-turbine-scada-data-for-early-fault-detection/data?select=Wind%20Farm%20A), dataset version **2**, основная CSV-группа.

В `reference/SWTFPS_Wind-Turbine-Failure-Prediction` клонирован tag `v1.0.0`, commit `b0922f9ab22a65b582d8c0d0e810d8e959500e76`. Авторские исходники и артефакт сохранены без изменений. Внутренний `.git` reference-репозитория отделён от нашего кода; `reference/`, raw data и wheelhouse исключены из будущего Git-индекса основного проекта.

```text
reference/SWTFPS_Wind-Turbine-Failure-Prediction/  # оригинальный release
data/raw/wind_farm_a_v2/                         # неизменённые 0.csv, 25.csv, metadata
data/scenarios_manifest.json                    # версия, пути, размеры, SHA-256
baseline/inference.py                           # mapper, интерфейс и model adapter
baseline/run.py                                 # последовательный inference CLI
baseline/evaluation.py                          # offline annotation metrics
baseline/analyze.py                             # offline evaluation и PNG-графики
tests/                                         # контракты и artifact integration tests
scripts/prepare_baseline.ps1                    # clone, проверка данных, сборка
scripts/fetch_wheels.py                         # verified PyPI wheelhouse
scripts/run_baseline.ps1                        # ограниченные контейнерные запуски
Dockerfile.baseline
requirements-baseline.txt                       # прямые зависимости
requirements-baseline.lock.txt                  # все 19 Python packages
artifacts/ml_baseline/                          # predictions, inspection, metrics, plots
```

## 2. Какой артефакт запускается

Путь в оригинальном репозитории: [`Codes/Models' Packeges/prototype_model_best.pkl`](https://github.com/SaraMahyan/SWTFPS_Wind-Turbine-Failure-Prediction/blob/b0922f9ab22a65b582d8c0d0e810d8e959500e76/Codes/Models'%20Packeges/prototype_model_best.pkl).

SHA-256: `98584786a6a4421f74e599743378c67aadaeaa67ffbfdd582a14f5d235617fd8`. Adapter отказывается загружать любой другой digest. Модели прототипа из `Prototype Model Packeges` не используются.

Статическое исследование до загрузки подтвердило:

- Сохранённый объект описывает sklearn `Pipeline`: `MinMaxScaler` и `XGBClassifier`.
- `_sklearn_version` равна `1.7.2`.
- В двух UBJSON version arrays XGBoost содержится `[3, 1, 2]`. Это уточняет неизвестную ранее версию XGBoost.
- В артефакте присутствует модуль `numpy._core.multiarray`, соответствующий NumPy 2.x.
- Вход — семь признаков, objective — `binary:logistic`.

Результат статического review сохранён в [static_review.json](../artifacts/ml_baseline/static_review.json). Статический анализ не доказывает безопасность pickle; запуск ограничен отдельно на уровне контейнера.

## 3. Окружение и зависимости

Основное окружение — Python **3.12.12**, отдельный venv `/opt/venv` внутри Linux Docker image `wind-scada-baseline:1.0`. Системный Windows Python 3.7 не используется для inference и не изменён. Он может выполнять только стандартный downloader wheels.

| Компонент | Зафиксированная версия | Основание |
| --- | --- | --- |
| Python | 3.12.12 | Отдельный поддерживаемый runtime, совместимый с выбранными wheels |
| scikit-learn | 1.7.2 | Версия сохранённого pipeline |
| xgboost-cpu | 3.1.2 | Совпадает с версией XGBoost внутри артефакта; предоставляет модуль `xgboost` |
| NumPy | 2.3.4 | Выбранный совместимый NumPy 2.x; точная training-версия не установлена |
| SciPy | 1.16.3 | Runtime dependency sklearn/XGBoost |
| pandas | 2.3.3 | DataFrame с фиксированными именами и порядком features |
| joblib | 1.5.2 | Чтение исходного формата артефакта |
| threadpoolctl | 3.6.0 | Dependency sklearn, контроль CPU threads |
| matplotlib | 3.10.7 | Только offline-графики, не inference |
| libgomp1 | Системный пакет Debian bookworm | OpenMP runtime XGBoost |

Дополнительные Python dependencies, включая библиотеки графиков и timezone data, перечислены с точными версиями в [lockfile](../requirements-baseline.lock.txt). CatBoost, LightGBM, SHAP, LIME, pyserial, TensorFlow, FastAPI и frontend packages для выбранной модели не требуются.

Использован официальный [CPU-only пакет XGBoost](https://xgboost.readthedocs.io/en/release_3.1.0/install.html#minimal-installation-cpu-only). GPU/CUDA/NCCL не устанавливаются. Версии библиотек, кроме явно сохранённых в артефакте, являются проверенным окружением deployment, а не заявлением о точном воспроизведении training environment авторов.

Базовый образ закреплён digest `sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c`. Downloader получает wheels из PyPI metadata API, проверяет опубликованный SHA-256 и сохраняет [wheel_manifest.json](../artifacts/ml_baseline/wheel_manifest.json). `pip install` внутри сборки работает с `--no-index` из локального wheelhouse. Это позволяет повторить установку без обращения к PyPI из контейнера; первоначальная подготовка wheels и системного image требует сети.

## 4. Изоляция недоверенного pickle

`venv` сам по себе не ограничивает исполнение pickle. Поэтому pickle загружается только при запуске через [run_baseline.ps1](../scripts/run_baseline.ps1) с параметрами:

- `--network none`, без опубликованных портов;
- пользователь `10001:10001`, `--cap-drop ALL`, `no-new-privileges`;
- read-only root filesystem и read-only mounts к нашему коду, одному артефакту и выбранным CSV;
- writable `/tmp` в ограниченном tmpfs и writable output directory;
- лимиты 2 CPU, 2 GiB RAM, 128 процессов;
- без Docker socket, пользовательской home directory, SSH/cloud credentials или передачи host environment variables.

Контейнер inference получает **только `0.csv` и `25.csv`**, а не `event_info.csv` или `feature_description.csv`. Контейнер offline evaluation получает annotations и predictions, но **не получает модель**. Исходная колонка `status_type_id` остаётся в raw CSV для сохранения данных, однако reader исключает её из `Measurement.values`, а mapper выбирает только семь разрешённых признаков.

Параметры реально созданных контейнеров записываются в `containment_Test.json`, `containment_Infer.json` и `containment_Evaluate.json`; `runtime.json` дополнительно фиксирует UID и доступные сетевые интерфейсы. В image не копируются reference code, данные или pickle. При сборке не импортируются авторские scripts.

Важно: `ScadaXGBoostAdapter` не создаёт OS sandbox самостоятельно. Для будущей интеграции его следует запускать в таком же ограниченном worker/container. Прямой вызов loader на обычном хосте не даёт описанных гарантий.

## 5. Подготовленные сценарии

Raw files скопированы из проверенной исследовательской выборки Kaggle v2 без преобразования. Setup умеет повторно скачать те же файлы через versioned Kaggle endpoint и сверить hashes. Основная CSV-группа использует `;`; `comma_` файлы запрещены reader и не используются.

| Сценарий | Asset | Строк всего | Train | Prediction | Prediction timestamps |
| --- | --- | --- | --- | --- | --- |
| `0.csv` | `0` | 54 986 | 52 148 | 2 838 | `2023-08-04 06:10:00`–`2023-08-24 06:10:00` |
| `25.csv` | `11` | 54 712 | 52 289 | 2 423 | `2023-05-23 06:50:00`–`2023-06-09 02:30:00` |

Event 0: label **anomaly**, description **Generator bearing failure**, аннотированный интервал `2023-08-06 06:10:00`–`2023-08-20 06:10:00`, row IDs `52436`–`54447` включительно. Это интервал аномалии, связанный с отказом, а не подтверждённый timestamp возникновения физического отказа.

Event 25: label **normal**; исходный normal interval `2023-05-23 06:50:00`–`2023-06-05 02:30:00`. Вся prediction-часть не содержит аннотированного anomaly event; её использование как normal scenario отмечено отдельно от точных границ normal annotation.

Оригинальные `event_info.csv`, `feature_description.csv` и README сохранены. Units не изменяются: ветер в m/s, температуры в °C, напряжение в V; `sensor_50` сохраняется как исходное значение с единицей **Wh по metadata**, без подмены W/kW. Исторические timestamps остаются naive dataset time, timezone не присваивается.

SHA-256 и размеры всех пяти подготовленных файлов находятся в [scenarios_manifest.json](../data/scenarios_manifest.json). Данные и исходники монтируются read-only, а predictions пишутся в отдельные файлы.

## 6. Точный inference contract

Порядок входа подтверждается сохранёнными `feature_names_in_`, `n_features_in_` и исходным [`prototype_model.py`](https://github.com/SaraMahyan/SWTFPS_Wind-Turbine-Failure-Prediction/blob/b0922f9ab22a65b582d8c0d0e810d8e959500e76/Prototype%20Experiment/New%20Sensors%20Control_With_Overall_Performance%20copy/Codes/prototype_model.py):

```text
sensor_0_avg
sensor_1_avg
sensor_43_avg
sensor_11_avg
wind_speed_3_avg
voltage = (sensor_32_avg + sensor_33_avg + sensor_34_avg) / 3
sensor_50
```

Mapper создаёт одну строку DataFrame float64 в исходных SCADA units. Затем вызываются исходные `pipeline.predict` и `pipeline.predict_proba`. Дополнительные scaling, clipping, fitting или domain mapping прототипа не выполняются. Сохранённый `MinMaxScaler` применяется самим pipeline ровно один раз. Runtime CPU selection ограничена `n_jobs=1`, `device=cpu`; веса и файл модели не меняются.

**Runtime inspection подтвердила** `classes_=[0,1]`, семь входов, steps `scaler/xgb`, `clip=False`, `n_samples_seen_=16000`. Сохранённые scaler bounds:

| Признак | data_min | data_max |
| --- | --- | --- |
| `sensor_0_avg` | 10 | 34 |
| `sensor_1_avg` | 9,8 | 351,8 |
| `sensor_43_avg` | 22 | 40 |
| `sensor_11_avg` | 34 | **66** |
| `wind_speed_3_avg` | 1,2 | 12,6 |
| `voltage` | **394,7** | 401,9 |
| `sensor_50` | −1987 | 262752 |

Эти значения отличаются от hardcoded reference bounds в некоторых prototype scripts, например gearbox max 67 и voltage min около 394,633. Поэтому ручное восстановление scaler по prototype script было бы неверным. Значения вне сохранённых min/max не обрезаются: это поведение исходного pipeline. Полные коэффициенты и booster config сохранены в [model_inspection.json](../artifacts/ml_baseline/model_inspection.json).

Последовательность обработки: открыть CSV → прочитать текущую строку → проверить возрастание timestamp/ID → извлечь разрешённые sensors → получить prediction → записать output → перейти к следующей строке. Future window, event labels, fault boundaries и status не подаются модели. На временных разрывах новые точки не создаются. Ошибка порядка завершает run; missing/NaN/Inf обязательных входов записываются как `invalid_input` с пустым классом и score.

Модель является **бинарным табличным классификатором** для каждой текущей строки. Классы сопоставляются с авторским target `failure` (`0` / `1`), который создавался в исследовательской подготовке данных; это не диагноз «generator bearing» и не survival/RUL forecast. У неё нет подтверждённого prediction horizon. `class_1_score` — output `predict_proba` класса 1, **без подтверждённой калибровки**. Его нельзя интерпретировать как вероятность отказа в ближайшие часы/дни.

В авторском `Codes/creating_datasets_with_faliure_colon.py` target расширял anomaly interval на 36 row IDs перед `event_start_id`, что названо 6 часами. При временных пропусках 36 строк не равны 6 часам. Такое offline target engineering не делает классификатор калиброванным прогнозом на горизонт 6 часов. В текущей проверке этот target не пересоздавался и не использовался как feature; evaluation сравнивает с исходным аннотированным интервалом.

Сохраняются timestamps, asset/scenario/row IDs, partition, status, predicted class, оба class scores, семь исходных/derived feature values, model ID/digest, preprocessing/schema version и score semantics. Невалидное измерение не превращается в «normal».

## 7. Результаты запуска и проверки

**Deployment выполнен успешно:** исходный pipeline загружается без version mismatch warnings, выдаёт классы и scores на всех **109 698** строках. Дополнительное обучение или изменение weights не выполнялось. Invalid inputs в этих двух сценариях не обнаружены. Полный offline sequential run занял **287,31 секунды** в текущем Docker/WSL2 окружении: event 0 — 140,75 секунды, event 25 — 145,27 секунды. Это измерение одного запуска, не production latency SLO.

В [журнале тестов](../artifacts/ml_baseline/tests.log) — **12 успешно пройденных тестов**: порядок/единицы features и среднее трёх фаз; missing/NaN/Inf/bool inputs; ленивое чтение и запрет нарушения хронологии; исключение status/labels; совпадение последовательных классов/scores с batch-вызовом оригинального pipeline; корректный offline lead и episodes при gaps; nullable результат ошибки.

**Результаты только исходных prediction-частей:**

| Показатель | Event 0: anomaly | Event 25: normal |
| --- | --- | --- |
| Число точек | 2 838 | 2 423 |
| Невалидные входы | 0 | 0 |
| Positive class 1 points | **2 268 (79,92 %)** | **1 842 (76,02 %)** |
| Positive episodes, разрывы учитываются | 143 | **184** |
| Первая positive-точка в prediction | `2023-08-04 06:10:00` | `2023-05-23 06:50:00` |
| До anomaly interval | 278 points | Не применимо |
| Внутри anomaly interval | 1 455 points | Не применимо |
| После anomaly interval | 535 points | Не применимо |
| Lead относительно начала annotation | **2 880 минут / 48 часов** | Не определён |
| Positive state продолжается из train | Да | Да |
| Normal false-alert proxy | Не определён для всего fault-сценария | **1 842 points / 184 episodes** |

**Главный вывод по качеству:** модель технически работает, но на normal prediction-сценарии выдаёт предупреждение для 76,02 % точек. Поэтому текущий artifact пригоден как технический baseline и проверка интеграционного пути, а не как подтверждённая система эксплуатационных предупреждений.

В event 0 предупреждения действительно есть **до** начала annotated interval. Однако первая positive-точка совпадает с началом prediction-окна, а positive state уже присутствует в конце train. Следовательно, 48 часов — наблюдаемый сдвиг первой доступной positive-точки этого окна относительно annotation start; это **не установленный момент начала новой тревоги** и не доказанное упреждение физического отказа.

В полной истории event 0 первая positive-точка находится ещё в `2022-08-04 06:10:00`, а event 25 — в `2022-05-23 06:50:00`, то есть в первых строках обоих файлов. Для всей истории positive fractions равны 49,81 % и 52,21 % соответственно. Годовой разрыв между ранним предупреждением и annotation не следует трактовать как полезный прогноз.

Pointwise agreement с исходной annotation в prediction-частях:

| Сценарий | TN | FP вне anomaly interval | FN внутри interval | TP | Annotation precision | Annotation recall |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 13 | 813 | 557 | 1455 | 64,15 % | 72,32 % |
| 25 | 581 | 1842 | 0 | 0 | 0 % | Не определён: нет positive ground truth |

Для event 0 число 813 включает **278 ранних** и **535 поздних** annotation-negative points. Оно не означает 813 физически подтверждённых ложных тревог. Для event 25 доступная разметка normal позволяет использовать 1842 как false-alert proxy, но подтверждающих эксплуатационных fault logs для независимой проверки нет.

### Артефакты и визуальное сравнение

- [Event 0: все 54 986 sequential predictions](../artifacts/ml_baseline/event_0_predictions.csv).
- [Event 25: все 54 712 sequential predictions](../artifacts/ml_baseline/event_25_predictions.csv).
- [Все offline metrics и episodes: evaluation.json](../artifacts/ml_baseline/evaluation.json).
- [Эпизоды предупреждений event 0, prediction](../artifacts/ml_baseline/event_0_prediction_warning_episodes.csv) и [event 25](../artifacts/ml_baseline/event_25_prediction_warning_episodes.csv).
- [Event 0: график prediction-части и anomaly interval](../artifacts/ml_baseline/event_0_prediction.png).
- [Event 25: график normal prediction-части](../artifacts/ml_baseline/event_25_prediction.png).
- Полный годовой контекст: [event 0](../artifacts/ml_baseline/event_0_all.png), [event 25](../artifacts/ml_baseline/event_25_all.png).
- [Run manifest, hashes и durations](../artifacts/ml_baseline/run_manifest.json), [runtime versions](../artifacts/ml_baseline/runtime.json), [полный freeze](../artifacts/ml_baseline/environment.freeze.txt).

На графиках синяя линия — class 1 score, красные штрихи — предсказанный класс 1. Оранжевая область event 0 соответствует исходной anomaly annotation. Отдельные панели показывают температуры, wind speed, mean voltage и `sensor_50`. Разрывы времени не соединяются искусственными измерениями. График normal-сценария показывает многочисленные высокие scores при отсутствии anomaly annotation, что визуально подтверждает проблему специфичности baseline.

### Что не сработало и что осталось ограничением

| Проверка / попытка | Результат |
| --- | --- |
| Прямая первоначальная установка зависимостей через сеть внутри Docker build | Загрузка wheels затянулась; сборка остановлена. Использован verified local wheelhouse и offline pip install. Точная причина задержки сетевого клиента не установлена; это не ошибка совместимости модели |
| Использование глобального Windows Python 3.7 | Не выполнялось: выбранные современные dependencies требуют отдельного runtime; создан Python 3.12.12 venv в контейнере |
| Загрузка именно SCADA XGBoost artifact | Успешна; feature order, scaler и classes подтверждены runtime inspection |
| Воспроизведение полезного early-warning поведения на normal-сценарии | **Не подтверждено:** 76,02 % positive points; порог не подгонялся, feature scaling не менялся |
| Независимое воспроизведение опубликованных research metrics | Не выполнено: точная training lineage и исходный experiment split отсутствуют; два event-сценария не являются независимым benchmark |

Нельзя исправлять высокую долю тревог произвольным дополнительным scaler, clipping или подбором threshold по этим же событиям. Возможные причины включают отличие balanced training distribution от исходной телеметрии, версии labels/data и domain/data quality effects; причинный вклад каждой из них здесь не установлен.

## 8. Методика offline comparison

Offline-анализ начинается **после** получения predictions. Для каждого сценария результаты разделяются на `all`, `train` и исходную `prediction` часть. Основная таблица выше относится к `prediction`; полная история сохранена отдельно. Model fitting не выполнялся.

Для event 0 истинная pointwise annotation задаётся интервалом `[event_start, event_end]` включительно. Для normal-сценария 25 отсутствует anomaly annotation. Считаются positive points, episodes, первая positive-точка, предупреждения до/внутри/после интервала и pointwise confusion matrix. Episode — последовательные positive-точки с шагом ровно 600 секунд; отрицательная/невалидная точка или временной разрыв прерывает episode.

**Lead to annotation start** вычисляется как `event_start − first_positive_timestamp` только если первая positive-точка выбранной части раньше начала anomaly interval. Это наблюдаемый сдвиг относительно аннотации, а не доказанное время упреждения физического отказа. Если предупреждений до начала нет, значение `null`. Отдельно фиксируется, продолжается ли предупреждение из train в prediction; тогда начало alarm episode может быть left-censored в пределах prediction-окна.

Семантика ложных тревог:

- Для normal-сценария positive points/episodes — **false-alert proxy относительно доступной разметки**, не физически подтверждённые ошибки.
- Для fault-сценария positive points вне interval — annotation-negative alerts. Ранние предупреждения до interval могут быть желательными; pointwise FP не следует автоматически называть физическими false positives.
- Precision/recall/accuracy в `evaluation.json` показывают согласие с выбранной interval annotation, а не независимую оценку качества модели.

Исходный training script использовал balanced random sampling и shuffled stratified split; точные training rows не восстановлены. Сценарии могут пересекаться с обучением исходной модели. Из двух сценариев нельзя вывести обобщённые accuracy, false-alarm rate или гарантии lead time. Метрики статьи здесь не воспроизводятся.

## 9. Воспроизведение на этом компьютере

Рабочий каталог — `D:\predict_defect`. Требуются Git, запущенный Docker Desktop с Linux engine и Python для downloader wheels; глобальные ML packages не нужны.

```powershell
Set-Location D:\predict_defect
.\scripts\prepare_baseline.ps1
.\scripts\run_baseline.ps1 -Stage Freeze
.\scripts\run_baseline.ps1 -Stage Test
.\scripts\run_baseline.ps1 -Stage Infer
.\scripts\run_baseline.ps1 -Stage Evaluate
```

Setup не перезаписывает существующие raw files или изменённый reference checkout: при hash/commit mismatch останавливается с ошибкой. Сеть используется только для получения публичных sources/dependencies; inference и evaluation работают с `network=none`.

Короткий smoke run в отдельную output directory:

```powershell
.\scripts\run_baseline.ps1 -Stage Infer -Partition prediction -Limit 100 -OutputDirectory artifacts/ml_baseline_smoke
```

Для полного повторения используется `Limit=0` по умолчанию. Полный run сохраняет новую версию файлов predictions в выбранном output directory; для сравнения нескольких запусков следует задавать разные directories. Оригинальные данные и модель остаются read-only.

Эквивалентные Python entry points внутри подготовленного restricted container:

```text
python -m baseline.run --model /model/baseline.pkl --input /input/0.csv --input /input/25.csv --output /output --partition all
python -m baseline.analyze --predictions /output --annotations /annotations/event_info.csv
python -m unittest discover -s tests -v
```

Не следует запускать `joblib.load` на хосте или выполнять авторский training script ради inference.

## 10. Подключение к будущему приложению

Интерфейс определён в [inference.py](../baseline/inference.py): `PredictionAdapter.predict(Measurement) -> Prediction`. `Measurement` содержит timestamp, turbine/scenario/row IDs и только текущие sensor values. `Prediction` содержит явный status, model provenance и структурированный результат. `to_dict()` выдаёт JSON-compatible объект.

Future replay-worker может создавать `Measurement` из уже committed SCADA-записи и передавать adapter. FastAPI позднее будет читать сохранённый результат; загружать pickle в API request handler не требуется. Сохранённый scaler принадлежит adapter, frontend его не повторяет.

Коллега может предоставить другой adapter с тем же контрактом; model ID, artifact digest, preprocessing version и score semantics должны измениться вместе с моделью. Если новой модели потребуется past window, интерфейс следует расширить явным causal history context с `window_end <= source_time`; текущий XGBoost окна не использует. Offline evaluation и rendering отделены от prediction path и не должны попадать в online worker.

## 11. Ограничения и дальнейшие действия

Прежде чем считать baseline пригодным для операционного использования, ML-инженеру необходимо подтвердить lineage обучающей выборки, определить target и prediction horizon, выполнить chronological/group-aware validation с учётом перекрытия event-файлов, оценить false alarms и при необходимости калибровку scores. Обучение и эти исследовательские улучшения в текущей работе не выполнялись.

Неполные APEX/ESP32 компоненты намеренно не развёртывались. Следующий технический шаг после рассмотрения результатов — согласование model contract с коллегой и подключение готового adapter к будущему replay-worker. Full-stack разработка ещё не начата.
