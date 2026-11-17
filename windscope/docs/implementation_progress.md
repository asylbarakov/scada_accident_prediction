# WindScope — результаты реализации MVP

Дата проверки: **16 ноября 2026 года**. Проект: `D:\predict_defect`. Реализован и запущен локальный end-to-end application на реальных данных, с исходным XGBoost. Предыдущие research, baseline source, модель, raw CSV и offline results сохранены. Новый full-stack scope реализован по последнему заданию пользователя.

## Статус этапов

| Этап | Результат | Статус |
| --- | --- | --- |
| Foundation | Изучены предыдущие reports и adapter; через браузер извлечены реальные CSS tokens KEGOC AI; создана собственная дизайн-система, API и schema | Выполнен |
| Backend | FastAPI REST, PostgreSQL, Alembic, проверяемый ingestion, sensor/turbine catalog, persistent sessions, transactional replay, predictions/alerts/event journal, WebSockets | Выполнен |
| ML | Повторно используется существующий adapter и проверенный artifact; inference в отдельном ограниченном контейнере; обучение не запускалось | Выполнен |
| Frontend | Русский Next.js dashboard, monitoring, faults/predictions, ECharts, выбор sensors, replay controls, time/range/mode selection, live updates и refresh/reconnect | Выполнен |
| Validation | Production frontend build/TypeScript, 10 backend tests, 12 baseline tests, реальные E2E для двух сценариев, gap и restart checks | Выполнен |
| Production deployment | Authentication, public hosting, эксплуатационные SLA, нагрузочное тестирование большого парка | Вне scope локального MVP |

Приложение: `http://localhost:3100`; OpenAPI: `http://localhost:8100/docs`. Запуск и повторяемые проверки описаны в [README](../README.md). [API/data contract](api_and_data_contract.md) объясняет endpoint semantics, schema и замену модели; [design system](design_system.md) отделяет проверенные значения референса от наших design decisions.

## Что реализовано

### Источники и хранение

`app/ingest.py` читает только primary Kaggle v2 CSV и сверяет SHA256 по `data/scenarios_manifest.json`. В PostgreSQL импортированы **109 698 source records**, 81 sensor column и два сценария: Event 0 / asset 0 и Event 25 / asset 11. Это исторические записи двух assets, не live подключение к wind farm. Original timestamps, partition, row IDs, status_type_id, units и annotations сохранены; source timezone неизвестна и не приписывается Казахстану/UTC.

Хранилище разделяет raw measurements, original scenario annotations, replay state, calculated predictions, warning episodes и ordered stream events. JSONB позволяет хранить исходный широкий набор sensors без 81 nullable ORM column. Индексы поддерживают source time / session queries. PostgreSQL volume сохраняется при остановке контейнеров. Импорт и initial Alembic migration повторно выполнены при финальных frontend restarts без дублирования measurements.

### Persistent historical replay

Основные use cases находятся в `backend/app/service.py`, clock policies — в `domain.py`, фоновая обработка — в `worker.py`.

- Replay range ограничен исходной prediction-частью; training history используется как доступная предыстория, без inference во время её просмотра.
- Реальный source delta делится на speed; Start/Pause/Resume/Reset/Speed сохраняются на сервере. Pause и смена скорости сохраняют оставшийся source interval.
- Row locking, leases и revision/generation fencing исключают сохранение результата, рассчитанного одновременно с Pause/Reset. Unique constraint исключает повторный prediction для одной точки generation.
- Missing timestamps не заполняются. Gap публикуется явно; предупреждения, разделённые gap, становятся отдельными эпизодами.
- Reset создаёт новую generation и скрывает прежние результаты, сохраняя audit records. REST и WS backlog фильтруются по generation.
- При restart worker сохраняет cursor, переводит running sessions в paused и освобождает старые leases. Resume продолжает со следующего source row.
- WebSocket передаёт snapshot, durable backlog по sequence, затем новые события. Клиент восстанавливает соединение и отбрасывает дубликаты.

В Replay future telemetry и annotations скрыты сервером. До начала fault interval его аннотации нет; будущий конец ongoing interval также скрыт. Analysis mode явно открывает полную историю и не вызывает inference. На frontend generation/mode/session guards предотвращают перенос данных предыдущего выбора.

### Реальный model inference

`backend/app/ml_server.py` оборачивает **неизменённый** `baseline.inference.ScadaXGBoostAdapter`. Сохранён scaler в Pipeline; feature order и artifact digest проверяются перед загрузкой. Нет дополнительного scaling, fitting или threshold tuning.

Worker передаёт только текущие девять raw feature values для семи входов модели; voltage — mean трёх phase voltages. Inference process не имеет CSV, annotations, database credentials или внешнего сетевого доступа. Контейнер non-root, root filesystem read-only, capabilities dropped. Это ограничение executable pickle boundary, не гарантия безопасности произвольного artifact.

Prediction хранит timestamp набора, UTC обработки, class, scores, version/digest/preprocessing, status и `prediction_horizon=null`. Недоступность ML сохраняется как unavailable с unknown UI state, без выдуманного normal result. Class 1 создаёт model warning, critical severity не выводится из бинарного результата.

### Frontend и визуальное поведение

Оригинальная композиция WindScope использует проверенные KEGOC colors, industrial surfaces, компактные borders, читаемые statuses и раздельные часы source/processing. Логотипы и assets референса не копировались. Next.js/TypeScript + Tailwind CSS + ECharts работают с настоящим API.

Есть park summary/table, turbine detail, 81 sensor columns и mean voltage, zoomable telemetry и class score charts, warning episodes, prediction history и ground truth. В source unit `sensor_50` сохранено `Wh`; для мощности доступны исходные `power_*` columns. Повреждённые degree symbols нормализуются только при display, оригинал остаётся в API `source_unit` и raw files. Charts сохраняют user zoom при live updates и разрывают линии только на реальных gaps; downsampling не создаёт ложные gaps.

## Проверки и evidence

Выполнены на локальных Docker containers, не на mock API.

| Проверка | Наблюдаемый результат | Evidence |
| --- | --- | --- |
| Backend unit/repository | 10 tests passed: time/speed policies, actual gap anchors, annotation clipping, transactions/fencing, idempotent commands, stale revision, Reset/generation | `artifacts/full_stack/tests.log`, `backend/tests/test_domain.py`, `test_service.py` |
| Существующий baseline | Все 12 tests passed повторно; включает реальное загрузившееся Pipeline inference | `artifacts/full_stack/baseline_tests.log` |
| Event 25 E2E | Первые 4 prediction-точки: class и score совпали с прежним CSV (tolerance `1e-7`); Pause/Resume/Reset и backlog reconnect прошли | `artifacts/full_stack/full_stack_e2e.json` |
| Event 0 E2E | Те же проверки на fault scenario; future annotations до наблюдения отсутствуют | Тот же JSON |
| Actual missing timestamp | Replay через настоящий gap создал ровно 2 исходных predictions, gap-event `synthetic=false`, 0 synthetic points | Тот же JSON |
| Restart API + worker | Running session восстановлена paused с тем же cursor/generation; Resume продвинул cursor | `artifacts/full_stack/restart_checkpoint.json`, `restart_verified.json` |
| Frontend production build | Next production compile, TypeScript check и standalone image прошли | `frontend/Dockerfile`, build output |
| Provenance | Reference checkout чистый, commit совпал; model и пять исходных файлов совпали с SHA256 manifest | Повторный `scripts/prepare_baseline.ps1 -SkipBuild` |
| Host endpoints | UI, proxied scenarios и API health ответили HTTP 200; ML и PostgreSQL healthy | Compose / HTTP checks |

Full-stack E2E проверяет ограниченные участки и один gap, а не повторно все source rows. Полный offline baseline run 109 698 rows относится к предыдущему deployment, описанному в [baseline report](ml_baseline_deployment.md). Эти доказательства не являются оценкой reliability ML или нагрузочной сертификацией.

Браузером проверены live predictions, создание нового range через календарный control, переключение сценариев, Reset, исторический режим и отображение источников/units. Normal replay был запущен на 600× и поставлен на паузу на позиции 89/148; после refresh восстановились та же позиция, source timestamp `2023-05-23 21:30` и paused state. Resume отдельно проверен automated E2E. Event 0 до `2023-08-06 06:10` не показывает fault annotation в Replay; analysis открывает `Generator bearing failure` и весь интервал, возврат в Replay снова скрывает его.

Обнаружена и воспроизведена ошибка Reset: сервер очищал видимость, а клиент merge сохранял старые predictions. После явной очистки points/predictions/annotations/alerts на смене generation тот же browser path показал пустую prediction history и нулевой progress. Запоздалые polling responses также ограничены session/mode/generation.

Visual QA выполнена на desktop 1440 px, mobile 390×844 и обычном browser pane 870 px. Root horizontal overflow отсутствует (на mobile `scrollWidth=375`, viewport 390); таблица прокручивается внутри panel, replay controls остаются доступны. Browser console после проверки не содержала captured errors/warnings. Это ручная browser QA, не автоматизированный regression suite. Viewport override сброшен; вкладка preview сохранена. Evidence: [desktop dashboard](../artifacts/full_stack/dashboard.jpg), [mobile monitoring](../artifacts/full_stack/mobile.jpg), [checklist](../artifacts/full_stack/browser_qa.json).

## Что переиспользовано

| Компонент | Решение |
| --- | --- |
| Original SWTFPS checkout и pretrained artifact | Сохранены без изменений; training scripts не импортируются |
| `baseline/inference.py` | Используется непосредственно как replaceable PredictionAdapter contract |
| Baseline tests, raw manifest и preparation scripts | Сохранены и повторно проверены |
| Offline golden predictions | Используются для E2E сравнения; не подставляются в UI вместо inference |
| Research и baseline reports | Сохранены, используются как technical foundation |

С нуля разработаны ingestion/DB/migrations, FastAPI endpoints, stateful replay worker, transactional coordination/event journal, HTTP ML boundary, WebSocket delivery/reconnect и весь frontend. Oracle APEX/ESP32, Kafka и Kubernetes не добавлялись.

## Устранённые препятствия

- Optional `uvicorn[standard]` требовал отсутствовавший wheel `httptools`; использован обычный Uvicorn/h11 и explicit websockets, exact dependencies закреплены.
- Internal-only Docker network не давала корректный host preview; добавлена отдельная edge network для frontend/API, database/inference остаются internal.
- Порт 8000 был занят другим локальным приложением; WindScope использует 8100, UI — 3100. Чужие services не изменялись.
- Filtering WS backlog после Reset, gap-aware downsampling и frontend generation cleanup дополнены проверками по найденным boundary cases.

## Известные ограничения

1. **Качество модели:** 76,02 % positive prediction-точек normal Event 25, warning Event 0 уже существовал до fault interval. UI не выдаёт class score за вероятность отказа через 24 часа, не обещает validated lead time и не прогнозирует числовые sensor values.
2. **Dataset:** только два выбранных сценария; неизвестная timezone, gaps и missing values, неоднозначные units некоторых sensors. Это replay anonymized assets, не доказательство переноса на реальный парк.
3. **Local MVP:** нет authentication/TLS/RBAC и multi-tenant isolation. Loopback binding и CORS ограничивают доступ, но не заменяют production access control. Analysis API намеренно доступен локальному пользователю.
4. **Производительность:** один worker, no event retention/archival, JSONB telemetry и polling WS. Ускорение — целевой масштаб времени; при недостатке throughput replay отстаёт, сохраняя все точки. Массовые parallel sessions/long-lived event volumes не нагрузочно протестированы.
5. **ML outage:** unavailable prediction сохраняется, automatic backfill/retry уже завершённых rows отсутствует. Critical severity и forecast horizon поддерживаются только после изменения model contract/policy.
6. **Model replacement:** вход/выход отделены от application, но feature allowlist, status mapping и baseline-specific UI labels требуют согласованного обновления. Registry/hot swap не реализованы.
7. **Migrations/dependencies:** initial migration использует ORM metadata; перед развитием schema её нужно заморозить и добавлять явные revisions. Python/npm versions pinned, Docker base tags пока не все закреплены digest; offline reproducible registry mirror не создан.
8. **Validation:** browser QA дополняет backend E2E; автоматического browser regression suite и CI pipeline пока нет. ML failure recovery и большие объёмы одновременных пользователей не проверялись нагрузочно.

## Следующие шаги

1. Review локального MVP: playback semantics, units, labels и видимость аннотаций; согласовать acceptance criteria.
2. ML-инженеру: улучшить specificity, определить classes/severity/horizon, оценить calibration/lead-time на независимых turbines и предоставить versioned artifact с feature contract.
3. Подключить новый adapter в изоляции и повторить causal/golden E2E; обновить model metadata и quality note.
4. Перед shared deployment добавить access control, TLS/secrets, backup/retention, CI/browser regression и нагрузочную проверку; затем выбирать hosting и масштабирование по измеренной нагрузке.

Источники: [SWTFPS GitHub](https://github.com/SaraMahyan/SWTFPS_Wind-Turbine-Failure-Prediction), [Zenodo](https://zenodo.org/records/23441231), [WES paper](https://wes.copernicus.org/articles/11/3983/2026/), [Kaggle](https://www.kaggle.com/datasets/azizkasimov/wind-turbine-scada-data-for-early-fault-detection), [KEGOC AI](https://kegoc.ai/). Новые внешние research claims в этой реализации не добавлялись; verified baseline findings остаются в предыдущих reports.

## Подключён CARE H7 — 17 ноября 2026

Добавлен отдельный сервис сохранённой Logistic Regression: прогноз аннотированного отказа на 168 часов, расчёт раз в час по прошлой SCADA. Версия модели закреплена за replay-сессией; прежний XGBoost и его история сохранены. Импортированы held-out события 13, 22 и 72 турбины 21. Тип неисправности не классифицируется.

Проверены 644 архивных score, причинное построение признаков из сырых CSV, девять live H7-отрезков через API/worker/WebSocket, 43 автоматических теста и прежний XGBoost E2E. В браузере подтверждены предупреждение Event 22 и сохранение acknowledgement после reload. Качество модели остаётся экспериментальным: 1 из 2 тестовых эпизодов обнаружен.

Подробности, ограничения и воспроизведение: [h7_integration.md](h7_integration.md).