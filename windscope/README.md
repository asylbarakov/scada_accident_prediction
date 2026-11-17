> **Актуальное демо · 17 ноября 2026**
>
> Откройте [WindScope](http://localhost:3100): персональная модель, ранний сигнал, два графика и проверка по историческому отказу. Первый запуск: `scripts/start_demo.ps1`. Предыдущий интерфейс доступен по `/monitoring`.
>
> [Как провести демонстрацию и что проверено](docs/demo_release.md). Исторический результат модели: **4 из 11 отказов**, семь пропусков; это исследовательский результат, не промышленная гарантия.
# WindScope — мониторинг и прогноз отказа H7

Рабочий локальный MVP: FastAPI, PostgreSQL, persistent historical replay, WebSockets и Next.js. Подключён экспериментальный прогноз отказа на **7 суток**: сохранённая Logistic Regression из `project`, отдельный изолированный inference-сервис и почасовой расчёт по истории SCADA. Исходный XGBoost SWTFPS доступен для прежних сессий и сравнения. Это воспроизведение истории, а не подключение к действующей SCADA.

Для проверки H7 добавлены отложенные тестовые события **13, 22 и 72** турбины 21. Сценарии 0 и 25 сохранены. Новые сессии через API по умолчанию используют H7; в настройках интерфейса можно выбрать модель нового запуска. Классификация типа поломки не реализуется. Score не является калиброванной вероятностью.

Подробности, ограничения, проверочный диапазон Event 22 и контракт: [Интеграция CARE H7](docs/h7_integration.md). Воспроизводимая проверка: `./scripts/test_h7.ps1`. На исходном отложенном тесте H7 обнаружила **1 из 2** эпизодов; подключение не означает улучшения точности модели. Приведённые ниже результаты прежнего baseline относятся к XGBoost, а не к H7.

Для движущихся графиков и новых прогнозов используйте «Включить мониторинг H7»: весь диапазон выбранного события, понятная скорость в секундах на точку и явно обозначенный повтор по концу истории. [Непрерывный мониторинг и чтение прогнозов](docs/continuous_monitoring.md).

## Запуск на Windows

Если после запуска Docker страница показывает недоступность API, выполните из корня `docker compose up -d`. Постоянные сервисы имеют автоматический перезапуск, интерфейс повторяет подключение при временной недоступности backend. [Диагностика и восстановление](docs/startup_recovery.md).

Нужны Docker Desktop в режиме Linux containers, PowerShell, Git и Python 3 для первоначальной подготовки baseline. Docker должен быть запущен; при первой установке требуется Internet для официальных images, packages и публичных исходных данных. Python environment модели находится в контейнере, host Python не загружает pickle.

```powershell
Set-Location D:\predict_defect
.\scripts\start_app.ps1
```

Скрипт проверяет reference commit и SHA256 модели/данных, при необходимости подготавливает исходный image `wind-scada-baseline:1.0`, затем собирает и запускает Compose. `init` выполняет Alembic migration и идемпотентный импорт. Первый импорт 109 698 строк может занять несколько минут.

- Приложение: [http://localhost:3100](http://localhost:3100)
- OpenAPI / Swagger: [http://localhost:8100/docs](http://localhost:8100/docs)
- Health: [http://localhost:8100/api/v1/health](http://localhost:8100/api/v1/health)

Порты привязаны к `127.0.0.1`. PostgreSQL и ML не опубликованы на host. `.env.example` содержит локальный dev password; Compose работает и без `.env`. При выборе своего пароля используйте URL-safe значение, поскольку оно входит в `DATABASE_URL`. После инициализации volume изменение переменной не меняет пароль существующей PostgreSQL роли.

Если baseline image уже подготовлен, эквивалентный ручной запуск:

```powershell
.\scripts\prepare_baseline.ps1 -SkipBuild
docker compose up -d --build
docker compose ps
```

## Работа с replay

1. Выберите `WT-11 · Event 25` или `WT-00 · Event 0`.
2. Нажмите «Запустить». При 600× исходные 10 минут соответствуют примерно одной секунде обработки. Доступны 1×, 60×, 600×, 3 600× и 36 000×.
3. Используйте «Пауза», «Продолжить», скорость и «Сбросить replay». В «Настройки» можно создать отдельную сессию внутри prediction-части или выбрать всю prediction-часть, настроив confirmation/cooldown. Старые сессии остаются в БД.
4. В режиме Replay сервер выдаёт только уже наблюдаемую историю и доступные аннотации. «Исторический анализ» явно открывает полную историю и ground truth; будущие predictions не рассчитываются заранее.
5. «Турбина» показывает charts с поиском и категориями для 81 исходного sensor column. avg/min/max/std сгруппированы; recommended presets содержат датчики с одинаковыми units. «Модель и события» разделяет raw predictions и аннотации.
6. «Предупреждения» содержит filters, sensor evidence, suggested action и «Принять в работу». Открыть измерения можно из карточки alert. Checkbox «Архив всех запусков» явно открывает сохранённую историю; она не меняет текущий статус парка.

## Alert policy и диагностика v2

По умолчанию alert активируется после **3 последовательных class 1**, закрывается после **3 последовательных class 0**, cooldown — **60 source minutes**. Гап >10 минут или невалидный input прерывает подтверждение; открытый эпизод закрывается с причиной `data_gap` / `data_quality`. Reset завершает эпизод с `replay_reset`. Это не подтверждение физического восстановления. Настройки неизменны внутри generation и задаются для новой сессии. ML/scaler/weights/decision threshold не меняются.

Диагностика объясняет текущий signal и сохраняет actual sensor evidence. При отсутствии причины: «Обнаружено отклонение в работе оборудования. Причина не определена.» Экспериментальное правило проверяет повышение температуры относительно **только прошлых 12 source часов** (не менее 24 finite points, ≥6 robust MAD scales); это статистическая зона проверки генератора/редуктора, не подтверждённый отказ и не промышленный safety limit. Critical не генерируется. Class 0 не означает физическую исправность. Evidence alert фиксируется на момент активации; live diagnostic хранится отдельно.

Полная оценка default policy: Event 0 — 2 268/2 838 raw positives и 23 alerts; Event 25 — 1 842/2 423 raw positives и 31 alert. Первый Event 0 alert задержан на 20 минут, всё ещё до начала annotation. На normal-сценарии warnings остаются длительными. Меньше уведомлений **не означает улучшенную точность ML**. Подробные durations, delays и ограничения — в [отчёте v2](docs/ux_and_alerts_improvements.md).

После обновления исходников существующей установки:

```powershell
docker compose up -d --build
```

`init` применит migration `002` и проверит ранее импортированные данные. Volume не удаляется, модель не переобучается, прежние predictions остаются. v1 raw episodes сохранены в старой таблице `alerts`, автоматически не объявляются v2 operational alerts. Для новых правил создайте новую сессию в UI; запущенные при restart сессии восстановятся paused.

Воспроизвести offline evaluation без network/model load:

```powershell
.\scripts\evaluate_alerts.ps1
# Другой набор правил — в отдельный каталог, без изменения модели:
.\scripts\evaluate_alerts.ps1 -Activate 5 -Resolve 3 -Cooldown 60 -OutputDirectory artifacts/policy_5_3_60
```

Скрипт читает сохранённые настоящие baseline predictions и оригинальные CSV read-only; сохраняет JSON и сравнительный scientific plot. Параметры не подбирались для максимизации metrics этих двух событий.

Cursor, скорость, generation и состояние сохраняются в PostgreSQL. Refresh восстанавливает сессию. После перезапуска worker активные сессии восстанавливаются **на паузе**; продолжение запускается пользователем. Разрывы времени сохраняются: worker ждёт исходный delta / speed, chart разрывает линию, gap-event имеет `synthetic=false`. На высокой скорости обработка ограничена фактической производительностью, точки не пропускаются.

Время набора хранится без timezone: источник её не сообщает. Отдельные `processed_at` / `emitted_at` — UTC. Условная UTC шкала в chart используется только для стабильного отображения исходных дат, без утверждения timezone источника.

## Модель и ограничения

Используется неизменённый `baseline/inference.py`, проверяющий digest опубликованного `prototype_model_best.pkl`. Pipeline уже содержит MinMaxScaler, дополнительного scaling и обучения нет. Порядок features: `sensor_0_avg`, `sensor_1_avg`, `sensor_43_avg`, `sensor_11_avg`, `wind_speed_3_avg`, `voltage` (mean `sensor_32_avg`, `sensor_33_avg`, `sensor_34_avg`), `sensor_50`.

Class 1 отображается как сырой сигнал модели, class 0 — отсутствие сырого сигнала. Итоговые alerts вычисляет отдельная policy. Score **не является калиброванной вероятностью отказа**, горизонт не задан, численного forecast нет. Critical severity baseline не определяет. В исходной оценке 76,02 % prediction-точек normal-сценария получили class 1; ранний сигнал Event 0 уже существовал до fault interval. Платформа демонстрирует интеграцию, но её warnings не подходят для самостоятельных эксплуатационных решений.

`sensor_50` сохраняет metadata unit `Wh`; описание источника неоднозначно, автоматического перевода в kW нет. Для мощности можно выбрать исходные `power_29_*`, `power_30_*`. Исходные повреждённые символы units сохранены в `source_unit`; только известные формы `�C` / `�` отображаются как `°C` / `°`.

ML выполняется non-root в read-only контейнере на отдельной internal network, без database credentials, raw CSV, event annotations и внешнего egress. Worker передаёт только текущие девять raw feature values и идентификаторы measurement. Контейнерная изоляция ограничивает риск pickle, но не превращает его в безопасный формат.

## Проверки

После запуска приложения:

```powershell
.\scripts\test_app.ps1
.\scripts\run_baseline.ps1 -Stage Test
```

Первый скрипт запускает 25 backend tests, 3 frontend behavior tests, реальные HTTP/WebSocket E2E для обоих сценариев, сравнение с сохранённым baseline, alert confirmation/acknowledgement/generation isolation и восстановление после restart API/worker. Он создаёт отдельные короткие test sessions, не удаляя данные, и сохраняет JSON evidence в `artifacts/full_stack/`. Существующие golden predictions должны находиться в `artifacts/ml_baseline/`; при их отсутствии выполните исходный baseline workflow, описанный в `docs/ml_baseline_deployment.md`. Второй скрипт запускает 12 сохранённых baseline tests в изоляции.

E2E сравнивает первые четыре prediction-точки каждого сценария с прежними CSV (class и score, tolerance `1e-7`); отдельный тест проверяет настоящий gap. Это проверка интеграции, не повторная оценка всех 109 698 строк. Frontend Docker build выполняет production compilation и TypeScript check.

## Остановка и диагностика

```powershell
docker compose logs --tail 100 api worker ml
docker compose logs --tail 100 init
docker compose stop
docker compose up -d
```

Named volume `windscope_pgdata` сохраняет импорт, сессии, predictions и events. При `docker compose down` volume также сохраняется. Для повторного импорта достаточно перезапустить `init`; SHA256 и source row count проверяются до пропуска существующего сценария. Отсутствие данных, несовпадение hash или неполный импорт являются ошибкой запуска.

## Структура и документация

| Путь | Назначение |
| --- | --- |
| `backend/app/` | API, use cases replay, worker, ingestion, SQLAlchemy models, ML HTTP boundary |
| `backend/migrations/` | Alembic schema version |
| `backend/tests/` | Unit/repository tests, настоящие E2E и restart tests |
| `frontend/src/` | Русский dashboard и ECharts |
| `baseline/` | Ранее проверенный inference adapter и offline validation; сохранены |
| `reference/` | Исходный SWTFPS release, без изменений |
| `data/raw/`, `data/scenarios_manifest.json` | Оригиналы CSV и provenance hashes |
| `artifacts/ml_baseline/`, `artifacts/full_stack/` | Предыдущие результаты и новые evidence |

- [Research и архитектура](docs/research_and_architecture.md)
- [Deployment исходного baseline](docs/ml_baseline_deployment.md)
- [Дизайн-система и извлечённые tokens KEGOC AI](docs/design_system.md)
- [Результаты реализации, проверки и ограничения](docs/implementation_progress.md)
- [WindScope v2: UX, diagnostics, alert rules и оценка](docs/ux_and_alerts_improvements.md)
- [API, данные и замена модели](docs/api_and_data_contract.md)

Зависимости: `backend/requirements.txt` — direct dependencies, `backend/requirements.lock.txt` — точные Python versions всего image; `requirements-baseline.lock.txt` — baseline; `frontend/package-lock.json` — npm lock. Backend основан на совместимом baseline image. Oracle APEX/ESP32 инфраструктура не запускается.

Это **локальный single-user MVP без authentication, TLS и production hardening**. Перед сетевым deployment нужны access control, reverse proxy/TLS, secrets management, backups, retention, нагрузочные проверки и эксплуатационный мониторинг. Следующая продуктовая работа — согласовать model contract и quality criteria с ML-инженером, затем заменить adapter и оценить новую модель на отложенных турбинах.

Источники: [GitHub SWTFPS](https://github.com/SaraMahyan/SWTFPS_Wind-Turbine-Failure-Prediction), [Zenodo release](https://zenodo.org/records/23441231), [статья WES](https://wes.copernicus.org/articles/11/3983/2026/), [Kaggle dataset](https://www.kaggle.com/datasets/azizkasimov/wind-turbine-scada-data-for-early-fault-detection), [визуальный reference KEGOC AI](https://kegoc.ai/).
