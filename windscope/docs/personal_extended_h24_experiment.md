# Восемь температур: контролируемое расширение personal baseline

17 ноября 2026 года. Протокол: `care-personal-eight-temperatures-h24-v3`.

## Решение

**Baseline сохраняется.** Добавление второго подшипника генератора и трёх фаз трансформатора не увеличило число обнаруженных отказов: **4/11**. Ложных normal-эпизодов стало **18 вместо 17**. Критерий замены не выполнен; модель, registry, API, UI и БД демо не изменены.

Это один зафиксированный эксперимент на ранее изученных данных A. После результатов параметры, sensor targets и thresholds не подбирались. Он проверяет температурный normal-behaviour подход; его score не является обученным/калиброванным прогнозом вероятности отказа за сутки. Полезность раннего предупреждения оценивается отдельно по событиям в последующие 24 часа.

## Что проверено в качестве данных

Использованы все 22 primary Kaggle v2 CSV Wind Farm A; их байты ранее независимо сверены с Kaggle. [Аудит полноты](kaggle_dataset_inventory_audit.md). В новом запуске SHA256 каждого raw CSV и каждой baseline feature table снова сверены с прежним data gate.

Дополнительные targets:

| Model target | Raw column | Смысл |
|---|---|---|
| generator_nde_temperature | sensor_14_avg | Подшипник генератора Non-Drive End |
| transformer_l1_temperature | sensor_38_avg | HV transformer phase L1 |
| transformer_l2_temperature | sensor_39_avg | HV transformer phase L2 |
| transformer_l3_temperature | sensor_40_avg | HV transformer phase L3 |

Missing/nonfinite превращаются в отсутствующее значение; интерполяции нет. Каждый target относится к `t−10 минут`; на разрыве timestamp stale-показание не переносится. Все конечные значения, включая 205, сохранены: подтверждённых физических границ для их автоматического удаления не предоставлено.

В почасовой delayed истории второго подшипника значения 205 найдены в нескольких source timelines. Во всех этих случаях **certified normal mask равен false**: ни одно такое значение не попало в подтверждённые normal-периоды fitting/calibration. В test Event 40 таких часовых значений 115; они не использовались для настройки модели и не удалялись из inference. Эти числа не являются независимыми измерениями между файлами из-за повторяющейся истории. Нельзя по одному значению доказать ни физический перегрев, ни ошибку датчика.

Подробные missing counts, min/max и 205 counts разделены по source/target/role в [quality.json](../artifacts/personal_extended_v3/quality.json). Они относятся к delayed hourly features и включают потери после временных gaps; поэтому не равны 20 пустым raw-ячейкам sensor_14 из предыдущего аудита.

Мощность в этом эксперименте не менялась: прежний `power_29` с исходной числовой шкалой. CARE сообщает об anonymization-related scaling. Проверка `power_30` должна быть отдельным экспериментом, чтобы не смешивать причины изменения результата.

## Изоляция эффекта добавленных targets

Для каждого source:

1. Загружены четыре собственные frozen normal-регрессора baseline; повторно воспроизведены все их test scores с допуском 10⁻¹² относительно старого CSV.
2. Четыре новых XGBRegressor обучены только на той же certified historical normal истории, с теми же режимными признаками, XGBoost parameters и temporal roles.
3. Bias и scale новых каналов вычислены только по прошлой normal-calibration. Старые calibration и четыре регрессора сохранены.
4. Восемь residuals объединены прежним правилом: maximum positive normalized deviation и trailing mean трёх доступных часов.
5. Порог заново выбран по тому же прошлому normal-бюджету: ≤2 эпизодов/720 normal-часов и ≤5% normal warning time. Ни один test fault label не участвовал в fitting или threshold selection.

Нормальная история и train/calibration boundaries наследуются из проверенного [baseline](personal_h24_experiment.md). Все fitting follow-up заканчиваются до historical boundary, calibration follow-up — строго до prediction start. Индивидуально доступны те же 22 source bundles; разные source clocks не объединяются. Предположение о retrospective availability normal labels остаётся ограничением.

Протокол записан до candidate test evaluation: [protocol.json](../experiments/personal_extended/protocol.json).

## Результаты на всей prediction-части

| Метрика | Baseline: 4 температуры | Candidate: 8 температур |
|---|---:|---:|
| Новые предупреждения в H24 | **4/11** | **4/11** |
| Пропущенные canonical отказы | 7 | 7 |
| Доступные решения | 8 238/8 433 | 8 238/8 433 |
| Availability | 97,69% | 97,69% |
| Известные доступные normal-часы | 2 983 | 2 983 |
| Ложные normal-эпизоды | 17 | 18 |
| Ложных эпизодов / 720 normal-часов | 4,10 | 4,34 |
| Normal warning hours | 74 | 64 |
| Normal warning fraction | 2,48% | 2,15% |
| Unknown warning hours | 102 | 189 |

Меньшая доля warning time при большем числе эпизодов означает более раздробленные сигналы и не считается улучшением. Unknown warnings не объявляются ложными или истинными без подтверждённого исхода.

Остались те же обнаруженные события:

| Event | Baseline lead | Candidate lead |
|---|---|---|
| 26 | 2 ч 20 мин | 2 ч 20 мин |
| 42 | 14 ч 50 мин | 13 ч 50 мин |
| 73 | 17 ч 40 мин | 17 ч 40 мин |
| 84 | 5 ч 30 мин | 5 ч 30 мин |

Новых обнаружений generator/gearbox/transformer failures не получено. Denominator — все 11 canonical событий, как в baseline; новые added targets не изменяют deduplication 51/72 или annotations. Lead time относится к CARE anchor (концу аннотированного аномального интервала), а не независимо установленному времени механического разрушения.

Candidate не прошёл два главных требования: больше четырёх обнаружений и ≤2 false episodes/720h. Thresholds на test не понижались ради получения дополнительных попаданий.

## Проверка исполнения и воспроизведение

- Три новых теста сначала выявили отсутствие реализации, затем прошли: past-only delay, reset на gap, missing/nonfinite без интерполяции; конечный экстремум сохраняется.
- Для всех 22 источников воспроизведены старые baseline test scores.
- Для всех 22 восьмиканальных bundles native save/load дал те же score без расхождений.
- 44 real raw-prefix и model-prefix checks подтвердили отсутствие влияния будущих измерений на более ранний score.
- Проверены temporal embargo/follow-up boundaries новых fit/calibration keys.

Код: [extended.py](../experiments/personal_extended/extended.py), [test_extended.py](../experiments/personal_extended/test_extended.py). Результаты: [metrics.json](../artifacts/personal_extended_v3/metrics.json), [verification.json](../artifacts/personal_extended_v3/verification.json), [fault_results.csv](../artifacts/personal_extended_v3/fault_results.csv), [prediction_scores.csv](../artifacts/personal_extended_v3/prediction_scores.csv).

```powershell
.\scripts\run_personal_extended.ps1 -OutputDirectory artifacts/personal_extended_my_run
```

Нужны полный A data directory, frozen `artifacts/personal_h24_v1` с 22 bundles/features, прежний baseline code и image `windscope-xgb-h24:1.0`. Demo-only установка подготавливает лишь подмножество этих inputs. Скрипт запускается без сети/credentials, с read-only source mounts, ограничениями памяти/CPU и отдельным writable output. Fitting не изменяет старый baseline и приложение.

Полный повтор — `artifacts/personal_extended_v3_reproduction`; итоговая побайтовая сверка сохраняется в [reproduction.json](../artifacts/personal_extended_v3/reproduction.json). Повтор не является новым независимым тестом качества.

## Что следует из этого эксперимента

Подтверждено: недостающие в модели каналы доступны и технически корректно добавлены, но их простое включение в максимум температурных отклонений не решило пропуски. Само присутствие новых датчиков не означает полезный предвестник отказа.

Следующая гипотеза — физически осмысленные отношения и режимы: разница двух подшипников генератора, дисбаланс фаз трансформатора, фактическая мощность относительно доступной. Это отдельные варианты с заранее заданными признаками и прежними ограничениями; их улучшение пока не доказано. Для обоснования generalization всё ещё нужен новый holdout с проверенным mapping, а не повторная настройка на этих 11 событиях. Classification узла и смена UI в этой итерации не выполнялись.
