# UX-аудит WindScope v2

## Critical

- Сырой ML class 1 имеет визуальный вес эксплуатационного alert; class 0 может ошибочно восприниматься как физическая исправность. Разделить raw result, diagnostic observation, persistent alert и ground truth.
- Dashboard начинается с generic KPI, но не отвечает, что проверить. Сначала краткая сводка требующих внимания assets, далее turbine cards и активные alert cards с объяснением/evidence/action.
- У warning episodes нет acknowledgement, понятной причины и перехода к measurement. Добавить alert center, lifecycle и сохранённый detail view.
- Sensor dropdown содержит 81 техническую колонку и повторяет avg/min/max/std. Группировать физические измерения, добавить поиск и advanced statistics; не смешивать units на одной оси.

## Refinement

- Вместо sidebar и больших повторяющихся блоков — компактная горизонтальная navigation, непрерывная полоса статуса/виртуального времени и спокойные surface panels, ближе к проверенному KEGOC visual language.
- Увеличить важный текст с 10–11 px до 13–15 px, оставить mono/tabular numbers для telemetry и UTC/source clocks. Короткие русские labels, технические CSV keys в details/tooltips.
- Observation cyan, Warning amber, Critical red с текстом/иконкой. Class 0 — «Сигнал модели не обнаружен», физическое состояние «Не оценено».

## Polish

- Inline notification только на активацию нового alert, dedup по persistent ID; не на каждый timestep.
- Пустые состояния объясняют, как запустить replay/снять filters. Видимые loading/error/focus states, 40–44 px controls на mobile.
- До/после screenshots: `artifacts/v2/`; browser QA — desktop, tablet, mobile и реальный flow acknowledgement + refresh.

## Проверенный reference и рамки

16 ноября 2026 повторно открыты KEGOC AI «Карта НЭС» и «SCADA». Вычитаны root CSS variables и computed styles body/headings/nav/KPI. JSON: `artifacts/v2/kegoc_tokens.json`; screenshots `kegoc_scada.jpg`, `kegoc_monitoring.jpg`. Наблюдены горизонтальные tabs, hairline divisions, компактные KPI и отдельная лента alerts с временем/severity. Их данные и assets не используются.

Пользователь прямо разрешил выполнить redesign и функциональные улучшения по этапам; дополнительный approval gate из общего design-audit workflow не нужен для уже разрешённого scope. Audit-template reference отсутствует в установленном skill; использована структура Critical / Refinement / Polish из самого SKILL.md.
