# Повторная проверка Kaggle v2 и полноты локальных данных

Дата: 17 ноября 2026 года. Dataset ID: `5929315`. Проверена version **2**, current version на момент проверки — также 2, последнее обновление API — `2025-09-17`.

## Вывод

**Для текущей модели и следующего эксперимента на Wind Farm A все необходимые исходные файлы и дополнительные датчики локально есть.** Все **22 primary CSV** и обе primary metadata-таблицы заново получены напрямую из Kaggle v2 и сравнены по SHA256 с рабочей копией. Все совпали. Подтверждена побайтовая идентичность всех 22 рабочих CARE v6 файлов основным Kaggle v2 файлам; прежняя оговорка о несверенных остальных 20 файлах теперь снята.

**Полного набора всех трёх ферм локально нет:** raw SCADA для B/C отсутствуют. Их metadata проверены, но текущая модель A не совместима с ними по одним именам колонок. Для отдельного расширения на B/C потребуются скачивание, mapping и новый протокол проверки. Доступность датасета не подтверждает качество прогнозов модели.

## 1. Источники и метод проверки

- [Kaggle dataset](https://www.kaggle.com/datasets/azizkasimov/wind-turbine-scada-data-for-early-fault-detection/data).
- [Официальный Kaggle API: dataset view](https://www.kaggle.com/api/v1/datasets/view/azizkasimov/wind-turbine-scada-data-for-early-fault-detection).
- [Официальный Kaggle API: version 2 file listing](https://www.kaggle.com/api/v1/datasets/list/azizkasimov/wind-turbine-scada-data-for-early-fault-detection?datasetVersionNumber=2&pageSize=100).
- [Исходный CARE v6, Zenodo](https://zenodo.org/records/15846963), Gück и Roelofs, CC BY-SA 4.0. Атрибуция и условия: [DATA_LICENSES.md](../project/DATA_LICENSES.md).

Страница Kaggle не отдала читаемый HTML инструменту поиска, поэтому использован официальный публичный API без токенов и аккаунта. Пройдены **все три страницы** списка файлов с `datasetVersionNumber=2`; проверено отсутствие повторных имён и pagination tokens. Один ответ API с pageSize=100 не представляет весь набор.

Каждый из 22 primary CSV полностью прочитан через file-download endpoint с version=2. Проверены размер по inventory и SHA256 всего тела ответа; сравнение не ограничивалось заголовком или началом файла. После совпадения временная копия удалялась, оригинал не перезаписывался. При несовпадении скрипт сохраняет отдельную копию и не меняет рабочие данные. В этой проверке несовпадений primary A не найдено.

Всего по file-download endpoints прочитано **802 923 557 байт**: все primary A CSV, primary metadata трёх ферм, comma event metadata для сравнения и два README. Это полный SHA256-анализ 22 A CSV, а не проверка всех 39,85 GB Kaggle-файлов. Whole-archive checksum Kaggle/Zenodo не вычислялся; серверный SHA256 в listing не предоставлен, remote SHA256 рассчитан из полученного по HTTPS содержимого.

Доказательства:

- [inventory.json](../artifacts/kaggle_v2_inventory_audit/inventory.json): все 204 remote entries, размеры, группы, версия и pagination count.
- [remote_verification.json](../artifacts/kaggle_v2_inventory_audit/remote_verification.json): URL и версия каждого полученного файла, размер, local/remote SHA256, local presence и сравнение.
- [readiness.json](../artifacts/kaggle_v2_inventory_audit/readiness.json): профили каждого A CSV, schema, sensors, временные интервалы, metadata и проверка annotation IDs.
- [application_manifest_checks.json](../artifacts/kaggle_v2_inventory_audit/application_manifest_checks.json): все 16 записей трёх application manifests существуют и соответствуют своим SHA256.
- [verification.json](../artifacts/kaggle_v2_inventory_audit/verification.json): итоговые assertions, hash audit-кода и hashes файлов доказательств; сохранённые JSON передаются через Git побайтно, без преобразования окончаний строк.

`matches_local=false` у дополнительных comma metadata и README.txt означает отсутствие прежнего файла в проверяемом локальном пути (`local_present=false`), а не повреждение primary SCADA. Они сохранены отдельно как provenance; для ML не использовались.

## 2. Полный состав Kaggle version 2

| Ферма | Primary CSV | Размер primary CSV, байт | Comma CSV | Локальные raw primary CSV |
|---|---:|---:|---:|---:|
| A | 22 | 802 864 323 | 22 | **22/22**, каждый проверен |
| B | 15 | 1 365 821 398 | 15 | 0/15 |
| C | 58 | 17 818 749 714 | 58 | 0/58 |
| Всего | **95** | **19 987 435 435** | **95** | **22/95** |

Полный Kaggle inventory — **204 файла, 39 847 429 522 байта**: 190 measurement CSV, 12 metadata CSV и два README. Primary B/C потребовали бы ещё **19 184 571 112 байт**, около 19,2 GB. Эти measurement files не скачивались в этой задаче: они не являются входами текущей модели и следующего A-эксперимента. Metadata B/C получены и совпали с уже подготовленной локальной metadata-копией.

Список A IDs: `0, 3, 10, 13, 14, 17, 22, 24, 25, 26, 38, 40, 42, 45, 51, 68, 69, 71, 72, 73, 84, 92`.

Файлы находятся в `project/data/CARE_To_Compare/Wind Farm A/datasets/`. Metadata A — в родительской папке. Metadata B/C — в `artifacts/nbm_h24_v1/external_metadata/`. Новые служебные копии для сравнения — в `data/raw/kaggle_v2_verified/`; они не входят в application import manifests.

### Почему не следует обучать на всех 204 файлах как на новых примерах

Повторно проверено по Kaggle metadata:

| Проверка | Primary | Comma |
|---|---|---|
| Event 0, annotation start | 2023-08-06 06:10:00 | 2022-08-05 06:10:00 |
| Event 51, label | anomaly | normal |
| A event metadata, поле asset | Есть | Отсутствует |

Группы отличаются версиями дат/разметки, а не только разделителем. Нельзя соединять их как дополнительные независимые поломки или применять comma metadata к primary CSV. Полные comma measurement CSV в этом аудите не получались; equivalence всех comma-файлов не утверждается.

Описание Kaggle/README сообщает 44 anomaly / 51 normal. Фактические primary metadata содержат **45 anomaly / 50 normal**: A — 12/10, B — 6/9, C — 27/31. Описание Zenodo v6 также указывает 45/50 и объясняет исправление Event 51 в истории версий. Для текущей primary-группы используем её metadata, а не старый общий текст.

## 3. Проверка всех строк Wind Farm A

- **1 196 747 исходных строк**, включая повторяющуюся между event-файлами историю. Это не столько же независимых наблюдений эксплуатации.
- Каждый из 22 файлов содержит **86 колонок**: пять служебных и 81 measurement field. Metadata описывает 54 базовых sensor names; avg/min/max/std — разные поля одного канала.
- В каждом файле присутствуют все **20 проверяемых входных каналов**: наборы SWTFPS и personal baseline, дополнительные температуры и `power_30_avg`.
- Timestamp внутри каждого файла строго возрастает; повторных или обратных временных шагов не обнаружено. Номинальные 10-минутные интервалы имеют разрывы — они посчитаны отдельно по каждому CSV.
- Для всех 22 metadata events `event_start_id` и `event_end_id` указывают строки с соответствующими annotation timestamps; asset metadata согласуется с asset в CSV.
- В 81 measurement field найдено **40 пустых/NaN ячеек**: 20 в `sensor_14_avg`, 20 в `sensor_22_avg`. Повторяющиеся строки разных source CSV здесь считаются повторно. Число отсутствующих временных точек в эти 40 ячеек не входит.
- Для 20 проверяемых model/experimental каналов не найдено nonfinite или нечисловых непустых значений. Отсутствие таких значений не означает физическую корректность всех показаний.
- Primary A metadata содержит **12 anomaly и 10 normal segments**. Предыдущая оценка модели использует 11 canonical faults после консервативной обработки Event 51/72. Эта новая проверка не пересчитывает global deduplication и не объявляет 12 segments двенадцатью независимыми отказами.

## 4. Датчики для следующего эксперимента уже доступны

| Канал | Смысл metadata | Непустых конечных значений | Пропусков |
|---|---|---:|---:|
| `sensor_14_avg` | Generator bearing 1, Non-Drive End | 1 196 727 | 20 |
| `sensor_38_avg` | HV transformer phase L1 temperature | 1 196 747 | 0 |
| `sensor_39_avg` | HV transformer phase L2 temperature | 1 196 747 | 0 |
| `sensor_40_avg` | HV transformer phase L3 temperature | 1 196 747 | 0 |
| `power_30_avg` | Grid power | 1 196 747 | 0 |

Все эти колонки есть во всех 22 CSV. Их исключение из четырёхтемпературного baseline было выбором модели, а не отсутствием файлов. Presence/числовая заполненность подтверждены; качество future warning от их добавления ещё не проверялось.

### Мощность и единицы

`power_29_avg` — Possible grid active power, `power_30_avg` — Grid power. Глобальные raw диапазоны: соответственно `0 … 0.9756097561` и `−0.0156585366 … 0.9759512195`. Metadata пишет kW, прежний config — p.u. Авторы CARE прямо предупреждают об **anonymization-related scaling of power-related data**. Поэтому эти числа нельзя объявлять физическими kW без восстановления scaling; величина обратного коэффициента здесь не установлена. Пересчёт в kW и исправление исходных данных не выполнялись.

В temperature metadata повреждён символ градуса; исходные байты сохранены. Физическая интерпретация и качество крайних значений требуют отдельных проверок. Например, в двух generator-bearing channels встречается 205; это наблюдаемое число, не автоматически подтверждённая температура исправного/неисправного подшипника.

CARE также сообщает о недостоверных исходных Min/Max/Std, особенно на B/C, и рекомендует Avg. Это относится к поставленным per-timestamp statistics. Каузальные rolling statistics, вычисляемые из последовательности Avg, являются другой операцией и тоже требуют своей проверки.

## 5. Почему B/C нельзя сразу передать модели A

Числовые имена датчиков анонимизированы отдельно по фермам. Подтверждённый пример `sensor_0`:

- A: Ambient temperature.
- B: Total negative reactive power, kvarh, counter.
- C: ABB-LS Input K1, IL1, A.

Совпадение имени не означает совпадение физической величины. Нужны mapping по descriptions/units/statistics, missing-channel policy, проверка clocks/labels и raw quality, затем отдельная проверка обобщения. Наличие metadata и доступность 73 файлов B/C в Kaggle ещё не являются подготовленным внешним тестом.

## 6. Что лежит на диске, импортируется и показывается

Это три разных уровня:

1. **Исследовательские исходники:** все 22 primary A CSV, теперь независимо сверенные с Kaggle v2.
2. **Application manifests:** семь IDs — `0, 13, 22, 25, 26, 42, 72`. Все 16 file references в трёх manifests проверены по hash. Это проверка файлов, а не повторный import или утверждение о текущем содержимом БД.
3. **Основной демонстрационный UI:** четыре выбранных случая — `0, 25, 26, 42`. Остальные CSV не обязаны появляться на главном экране только потому, что они есть на диске.

Проверка не меняла dataset inputs, app manifests, inference, модели, БД или UI. Новые модели не обучались.

## 7. Воспроизведение

Скрипт использует Python standard library, не загружает pickle/model artifacts и не использует credentials:

```powershell
python -m unittest discover -s tests -p test_kaggle_audit.py -v
python scripts/audit_kaggle_v2.py
```

Первый полный online запуск читает около 803 MB по сети, сверяет все primary A bytes и профилирует 1,2 млн строк. Для повторной проверки сохранённых remote hashes и offline profile:

```powershell
python scripts/audit_kaggle_v2.py --offline
```

На новой установке нужен полный A data directory для сравнения с существующей рабочей копией. Если его нет, script сохраняет полученные файлы отдельно и помечает `matches_local=false`; он не подменяет автоматически пути приложения. Пять тестов проверяют выбор primary без comma, защиту существующих файлов, отдельное сохранение отличий, отказ при truncated response, schema/sensor availability, timestamp gaps и annotation IDs. До реализации все пять тестов падали, после неё проходят.

Рекомендованный следующий шаг — отдельный A-эксперимент с уже имеющимися дополнительными каналами и прежним false-alarm budget. Скачивание B/C — отдельное расширение объёма данных и области модели, а не устранение отсутствующих файлов текущего baseline.
