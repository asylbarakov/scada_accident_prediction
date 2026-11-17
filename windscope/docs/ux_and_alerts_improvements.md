# WindScope v2 — UX, диагностика и предупреждения

Дата: 16 ноября 2026. Реализовано поверх существующего MVP в `D:\predict_defect`; FastAPI, PostgreSQL, Compose, replay-worker, WebSockets и исходный XGBoost сохранены. Переобучения, изменения decision threshold и дополнительного scaling не было.

## UX-аудит и результат

Исходные проблемы и phased plan: [ux_audit_v2.md](ux_audit_v2.md). Сырые class 1 выглядели как эксплуатационный диагноз; большой sensor dropdown смешивал физические величины и статистики, а warning episodes не содержали action/acknowledgement. Новый dashboard начинает с требующих внимания турбин, объяснений и evidence, затем показывает replay и измерения.

- Горизонтальная navigation: «Обзор парка», «Турбина», «Предупреждения», «Модель и события».
- Turbine cards разделяют «Наблюдение/Требует проверки», raw ML и физическое состояние «Не оценено». Class 0 не подтверждает исправность.
- Alert center: фильтры turbine/severity/status, source и UTC timestamps, evidence, proposed inspection, lifecycle и переход к source-time окрестности сигнала.
- Sensor browser: русские имена, категории, поиск по названию/CSV key, recommended presets и advanced statistics. avg/min/max/std принадлежат одной физической величине; units получены из metadata и не заменены предположениями.
- Telemetry, raw model score и historical annotation визуально подписаны раздельно. Есть time ranges, zoom, source clock marker, gap breaks и focus на evidence timestamp.
- Новая активация вызывает одну inline notification по persistent ID, без popup на каждом measurement. Reconnect backlog не повторяет уже известные уведомления.

KEGOC AI повторно открыт в браузере: «Карта НЭС» и «SCADA», root CSS variables и computed styles body/headings/nav/KPI вычитаны непосредственно. Palette и компактная операционная композиция использованы без копирования protected assets/полного layout. [Обновлённая design system](design_system.md) отделяет verified values от наших решений. JSON и reference screenshots — `artifacts/v2/kegoc_*`.

## Структура реализации

| Файл | Назначение |
| --- | --- |
| `backend/app/diagnostics.py` | Causal diagnostic observations и data-quality checks; не читает event labels |
| `backend/app/alert_engine.py` | Pure versioned policy, confirmation/hysteresis/cooldown на source clock |
| `backend/app/service.py` | Transactional persistence diagnostics/alerts, acknowledgement, Reset fencing, archive |
| `backend/app/models.py` | Новые Diagnostic, AlertRuntime, OperationalAlert; старые таблицы сохранены |
| `backend/migrations/versions/002_diagnostics_alerts.py` | Additive Alembic migration без изменения baseline predictions |
| `backend/app/sensor_catalog.py` | Русские display names, physical groups/statistics поверх original keys |
| `backend/app/main.py` | Sensors, policy, diagnostics, filtered alerts, acknowledgement REST |
| `backend/app/evaluate_alerts.py` | Offline evaluation сохранённых реальных predictions + causal raw history |
| `frontend/src/hooks/useWindScope.ts` | Прежний persistent replay flow, snapshot/history, generation guards, WebSocket reconnect |
| `frontend/src/components/WindScopeApp.tsx` | Operational dashboard, alert center, explicit historical modes |
| `frontend/src/components/SensorPicker.tsx` | Search/category/advanced selection |
| `frontend/src/components/AlertDetail.tsx` | Evidence, inspection action, confidence limits, lifecycle |
| `frontend/src/components/TelemetryChart.tsx` | Units, scrollable legend, timestamp marker, preserved gaps/zoom |
| `frontend/src/app/tokens.css` | Primitive → semantic → component tokens |

Измерение обрабатывается как прежде: worker получает следующую chronological запись → текущие inputs передаются в ограниченный ML service → commit повторно проверяет lease/revision/generation → сохраняет raw Prediction, Diagnostic, policy state, OperationalAlert и durable events в одной transaction. Внешний HTTP inference остаётся вне DB lock. SQL indexes покрывают session/generation/status и asset/time; диагностика запрашивает максимум 72 предыдущих rows. Alert acknowledgement берёт тот же session lock, чтобы не терять изменения при concurrent tick.

Исходные `baseline/inference.py`, model artifact, reference checkout, raw CSV, manifest и прежние golden predictions не изменены. Old `alerts` остаётся историей v1 raw episodes; v2 использует новую `operational_alerts`, без автоматического объявления старых raw signals итоговыми warnings. Migration 001 ограничена первоначальным набором таблиц, чтобы на новой установке не создавать v2 schema до revision 002.

## Диагностический контракт и честная интерпретация

Diagnostic содержит turbine ID, source timestamp, detection_source, raw model_output, optional suspected_subsystem, category, explanation, evidence с actual values/units, severity, confidence_limitations, suggested_action, rule version и `prediction_horizon=null`.

1. **ML-only:** class 1 даёт наблюдение «Обнаружено отклонение в работе оборудования. Причина не определена.» Без компонентного evidence subsystem=null. Class 0 даёт отсутствие сигнала, а не «оборудование исправно».
2. **Экспериментальное relative-temperature rule:** для каждого доступного generator/bearing/gearbox temperature берутся только прошлые 12 source часов, максимум 72 rows, минимум 24 finite значения. Вычисляются median и MAD; `z=(current−median)/(1.4826×MAD)`. При upward z≥6 и raw class 1 указывается зона проверки generator/gearbox. Flat reference MAD≈0 пропускается. Reference median/count и robust deviation сохраняются в evidence.
3. **Неполный input / ML unavailable:** observation `data_quality`, condition unknown, нет выдуманного class 0; подтверждение policy сбрасывается.
4. **Ground truth:** хранится/просматривается отдельно. Не входит в inputs модели, функцию diagnostics, policy или action online. Future annotation в Replay скрыта; analysis явно открывает полный interval.

Порог 6 MAD — статистическая эвристика этой версии, **не промышленный предел температуры и не доказанная идентификация отказа**. Температура зависит от нагрузки, погоды, cooling и sensor quality; load-normalized reference пока нет. Unit tests проверяют causal/statistical logic и отбрасывание future rows, но не подтверждают физическую диагностическую точность. На normal Event 25 правило также срабатывает; это подчёркивает ограничение специфичности.

Полный electrical/rotor fault localization не реализован: доступность sensors сама по себе не является основанием выдумывать validated diagnostic rule. Эти каналы доступны для анализа, а необъяснимый сигнал остаётся unclassified. Новый ML adapter и набор diagnostic rules можно заменить независимо от replay/storage.

## Alert Engine v2.0

Default policy `consecutive-hysteresis-v2.0`:

| Параметр | Default | Семантика |
| --- | --- | --- |
| activate_points | 3 | Последовательные finite/valid class 1 с исходным шагом 600 s |
| resolve_points | 3 | Три последовательных valid class 0 закрывают открытый эпизод |
| cooldown_minutes | 60 | После resolution/interruption следующая активация не раньше этого source-time срока |
| max_gap_seconds | 600 | Больший delta прерывает continuity; точки не синтезируются |

Это конфигурируемая фильтрация уведомлений, не изменение ML. UI «Настройки» создаёт новую сессию с выбранной политикой. Backend принимает те же поля в `POST /api/v1/replay-sessions.alert_policy`. Допустимы confirmation counts 1..100, cooldown 0..10080 min. Policy snapshot сохраняется для каждой generation/alert; Reset переносит config в новую generation с пустым runtime state. Нельзя незаметно менять правила посреди эпизода.

- Первый сигнал хранится отдельно от времени активации; данные не backdate-ятся к началу серии.
- Только один открытый alert на текущую session/generation; следующие positive points обновляют счётчик/last time, без нового notification.
- Во время cooldown positives могут накапливать confirmation; alert возникает на первой допустимой точке, если серия остаётся непрерывной.
- Gap/data-quality interruption закрывает эпизод с явной причиной, а не утверждением об исчезновении неисправности. Следующее подтверждение начинается заново.
- `Active → Acknowledged → Resolved`. Принятие в работу идемпотентно, сохраняется в БД и WS journal; не меняет ML, severity или ход replay. Resolved нельзя принять заново.
- Reset закрывает открытый alert с `replay_reset`, сохраняет audit, скрывает предыдущую generation из live endpoints.
- Default center использует последние сессии двух scenarios и их текущие generations. «Архив всех запусков» / `archive=true` явно открывает предыдущие records; архив не влияет на current fleet status и UI не даёт менять acknowledgement из архивного view.
- Evidence alert неизменно после activation; текущая Diagnostic обновляется отдельно. Browser QA выявила и regression test зафиксировал эту границу timestamp.

Поддерживаются severity values Observation/Warning/Critical. В этой версии Diagnostic observations имеют Observation; operational подтверждённые последовательные ML signals — Warning. Critical автоматически **не генерируется**, потому что независимые industrial criteria отсутствуют. Warning означает приоритет инженерной проверки, а не установленный риск конкретного физического отказа.

## Offline evaluation — Event 0 и Event 25

Выполнена для всей исходной **prediction-части**, используя сохранённые реальные XGBoost predictions и original raw rows. Перед prediction возможная предыстория прогревает только causal temperature reference; alert state стартует пустым в начале replay. Train rows не входят в метрики. Row IDs и source timestamps выравниваются. Ground truth читается только после последовательного прохода diagnostics/policy.

| Метрика | Event 0 | Event 25 |
| --- | --- | --- |
| Prediction points | 2 838 | 2 423 |
| Raw class 1 | 2 268 (79,92 %) | 1 842 (76,02 %) |
| Generated operational alerts | 23 | 31 |
| Суммарная длительность alert, source min | 23 330 | 18 740 |
| Длительность одного alert, min..max | 30..5 010 | 30..4 320 |
| Alerts на normal-сценарии | Не применимо | 31 |
| Activations до начала fault annotation | 2 | Не применимо |
| Первый alert | 2023-08-04 06:30 | 2023-05-23 07:10 |
| Задержка первого alert относительно raw signal | 20 min | 20 min |
| Confirmation/cooldown delay по всем activations | 20..50 min | 20..50 min |
| First alert до annotation start | 2 860 min (47 h 40 min) | Нет anomaly interval |
| Right-censored alerts на конце ряда | 1 | 1 |
| Relative-temperature diagnostic points | 166 | 187 |

Длительность = source resolution−activation; для незакрытого эпизода last observed−activation, с `right_censored=true`. Cooldown и негативное hysteresis могут держать alert открытым между отдельными positives. Gap обнаруживается при поступлении следующей строки: elapsed span alert до закрытия может включать время без измерений; это не доказательство непрерывного физического отклонения в пропуске. Поэтому меньше карточек не обязательно означает меньше времени под предупреждением.

На Event 25 остаётся 31 normal-scenario alert и 18 740 minutes alert coverage — это существенная проблема baseline. Количество сообщений уменьшено, но **улучшение ML accuracy не заявляется**. Повышение specificity не доказано. Два выбранных сценария не являются независимым benchmark, training lineage не восстановлена, annotation intervals не определяют точный физический onset. Сигнал Event 0 уже присутствовал в начале replay; lead относительно аннотации не является validated early failure prediction.

Evidence: [JSON](../artifacts/v2/alert_evaluation.json), [полное сравнение](../artifacts/v2/alert_policy_comparison.png). Запуск: `scripts/evaluate_alerts.ps1`; параметры меняют только policy и output directory, не веса/threshold/scaler. Default 3/3/60 задан до оценки, не оптимизирован по двум events.

## Проверки

- 25 backend unit/repository tests: исходные replay/visibility contracts плюс causal diagnostics, quality, confirmation/hysteresis/cooldown, immutable evidence, persistent ack/resolution/archive/reset и canonical UTC serialization.
- 3 Node frontend behavior tests: поиск/physical grouping/variants и совместные filters без stale hidden detail.
- 12 существующих baseline tests: original model/scaler/feature mapper/sequential inference сохранены.
- Реальные HTTP + WebSocket + PostgreSQL + XGBoost E2E для Event 0/25: первые 4 predictions совпадают с golden CSV; один alert после третьей positive-точки; evidence timestamp фиксирован; acknowledgement идемпотентен и сохраняется; Reset скрывает alerts/diagnostics, archive сохраняет их с reason.
- Проверен actual gap без synthetic points; restart API/worker восстанавливает cursor paused и продолжает со следующей точки.
- Production Next.js build и TypeScript check проходят, dependencies не расширялись новыми UI framework/runtime packages.

Логи/JSON: `artifacts/v2/tests.log`, `backend_tests_final.log`, `baseline_tests.log`, `full_stack_e2e.json`, `restart_verified.json`. Всего 40 unit/repository/frontend/baseline tests плюс отдельные реальные E2E/restart проверки.

Final API review дополнительно выявил выдачу PostgreSQL timestamptz с session offset +05:00: alert creation/ack/resolution и WS emitted_at нормализованы к +00:00 без изменения момента времени. Regression проверяет этот случай. Source timestamps по-прежнему naive, timezone не выдумывается.

Browser QA выполнена на окончательной UI-сборке: новый Event 0 session запущен и поставлен на паузу на позиции 3/2838; alert #6 активирован в `2023-08-04 06:30` после первого raw signal в 06:10. Карточка хранит evidence ровно 06:30, «Принято в работу» сохранено после refresh. Проверены переход к chart, critical filter без stale details, явный архив прошлых запусков, поиск «мощность» / `power_30`, advanced avg/min/max/std и kW. Будущая Generator bearing failure annotation скрыта в Replay, доступна в analysis и снова скрывается при возврате. Captured browser console не содержала errors/warnings.

Responsive: desktop 1440×1000 (`scrollWidth=1425`), tablet 870×1000 (`855`), mobile 390×844 (`375` в monitor, `390` в alerts). Горизонтального переполнения страницы нет; navigation/table используют собственный scroll. Mobile sensor popover: x=31,8..358,2, внутри viewport. Override сброшен, приложение оставлено на «Обзор парка», replay paused. [Полный checklist](../artifacts/v2/browser_qa.json). Это ручная автоматизация браузера, не полноценный browser regression suite.

| Экран | До | После |
| --- | --- | --- |
| Dashboard | [Исходный](../artifacts/v2/before_dashboard.jpg) | [Оперативный обзор](../artifacts/v2/after_dashboard.jpg) |
| Alerts | [Сырые episodes](../artifacts/v2/before_alerts.jpg) | [Persistent alert center](../artifacts/v2/after_alert_center.jpg) |
| Mobile | Отдельный исходный mobile-снимок не получен | [Alert details](../artifacts/v2/after_alert_mobile.jpg), [sensor browser](../artifacts/v2/after_sensor_browser_mobile.jpg) |
| Tablet | [Исходный monitoring, 855px](../artifacts/v2/before_monitor_tablet.jpg) | [Monitoring](../artifacts/v2/after_monitor_tablet.jpg) |

![WindScope v2 — итоговый оперативный обзор](../artifacts/v2/after_dashboard.jpg)

## Использованные установленные skills

Файлы SKILL.md действительно найдены и прочитаны в локальной среде. Применение:

| Skill | Как использован |
| --- | --- |
| `design-audit` | Critical / Refinement / Polish audit, states/hierarchy/reduction; [audit](ux_audit_v2.md) |
| `frontend-design` | Предметная операционная композиция, original alert-first layout, plain engineer-oriented copy |
| `vercel-react-best-practices` | Parallel independent fetches, dynamic ECharts import, memoized sensor arrays, primitive effect dependencies, generation/session guards |
| `ui-ux-pro-max:design-system` | Three-layer tokens, согласованные component states и responsive sizes |
| `typography-design` | Русская типографика, hierarchy, tabular numbers, focus/readable labels |
| `computer-use:computer-use` + CUA browser docs | Реальный KEGOC inspection, до/после screenshots, functional browser flow и responsive QA |
| `test-driven-development` + `writing-good-tests` | Red/green тесты policy, diagnostics, persistence, sensors, filters, immutable evidence |
| `verification-before-completion` | Свежие builds/tests/endpoints/evidence до итоговых утверждений |
| `systematic-debugging` | Root-cause проверка evidence timestamp, regression перед исправлением |
| `supabase-postgres-best-practices` | Review indexes/joins, сохранение короткой commit transaction без ML HTTP под lock |

Общий design-audit approval gate не останавливал работу: пользователь явно поручил выполнить improvements incrementally. Его отсутствующий `references/audit-template.md` не выдумывался; использована структура самого SKILL.md. Отдельный общий code-review skill в доступном перечне не обнаружен; review выполнен по React/PostgreSQL rules, boundary tests и проверке final source. Security scan не подменял functional code review.

## Ограничения и следующий шаг

- Диагностика экспериментальна и не подтверждает компонентный отказ; false alerts остаются существенными. Нужны ML-инженер, OEM limits, load-conditioned reference и независимые turbines для quality validation.
- UI показывает до 1 000 recent center/archive alerts и 1 200 recent predictions; полноценной pagination/retention archive UI пока нет. API diagnostics/predictions поддерживает bounded queries.
- Нет automatic backfill/retry уже сохранённых unavailable predictions. Статистический reference может пересекать data gaps, но использует только прошлые finite rows в ограниченных 12 часах; это указано как ограничение, continuity policy отдельно прерывает alerts.
- Нет authentication/roles, автор acknowledgement не идентифицируется; есть UTC timestamp. Это локальный single-user MVP, не production deployment.
- Archive/read-only режим отделён в UI; local API намеренно позволяет historical queries. Public deployment требует access control, TLS/secrets/backups и нагрузочной проверки.
- Автоматические tests не являются physical safety validation и не заменяют independent ML evaluation.

Следующий этап — review UX и policy с инженером эксплуатации, согласование диагностических критериев/units, затем интеграция улучшенной модели через прежний adapter contract и независимая оценка alerts/detection delay. Полный dataset/model provenance остаётся в [baseline report](ml_baseline_deployment.md) и [research](research_and_architecture.md).
