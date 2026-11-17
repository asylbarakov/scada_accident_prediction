# Дизайн-система WindScope v2

## Проверенный reference

[KEGOC AI](https://kegoc.ai/) повторно исследован через браузер 16 ноября 2026 года. Открыты «Карта НЭС» и «SCADA»; прочитаны root CSS variables и computed styles body, h1/h2, nav tabs, KPI. Исходный JSON: [kegoc_tokens.json](../artifacts/v2/kegoc_tokens.json). Screenshots: [SCADA](../artifacts/v2/kegoc_scada.jpg), [мониторинг](../artifacts/v2/kegoc_monitoring.jpg). Таблицы на других вкладках отдельно не проверялись; им не приписываются точные значения.

| Реальная CSS variable | Значение | Применение WindScope |
| --- | --- | --- |
| `--bg` / `--bg2` | `#0a0e13` / `#0c1117` | Canvas / header |
| `--panel` / `--card` / `--panel2` | `#0e1319` / `#10161e` / `#121922` | Operational strip / cards / controls |
| `--line` / `--line2` | `#1b232d` / `#252f3b` | Hairline разделители / borders |
| `--text` / `--text2` | `#e6edf3` / `#c3ccd6` | Заголовки / основной текст |
| `--muted` / `--dim` | `#8a96a3` / `#5b6673` | Labels / второстепенная информация |
| `--accent` / `--accentbg` | `#38bdf8` / `rgba(56,189,248,.13)` | Выбор / фокус |
| `--green` | `#34d399` | Состояние соединения |
| `--amber` / `--orange` | `#fbbf24` / `#f59e0b` | Warning / plot markers |
| `--red` | `#ef4444` | Critical / error |
| `--blue` / `--cyan` | `#5b7cfa` / `#22d3ee` | Ряды данных |
| `--sans` / `--mono` | Inter, system-ui / JetBrains Mono, ui-monospace | Текст / technical identifiers |

Computed reference: body `12px/18px`; h1/h2 `600 11.5px/17.25px`; nav buttons `11.5px`, horizontal padding `9px`, gap `6px`; inactive color `rgb(170,180,192)`, active `rgb(232,246,255)`. KPI padding `12px 12px 10px`, radius `0px`; navigation horizontal, активная вкладка выделена, dashboard использует плотные блоки с тонкими разделителями. Отдельная лента уведомлений показывает текст, время и severity. Эти размеры наблюдены у reference, но не являются обязательными размерами доступного WindScope UI.

## Наше оригинальное решение

Операционная композиция: горизонтальная product navigation → источник и processing clock → краткая fleet summary → требующие внимания alerts → turbine cards → persistent replay console → telemetry/diagnostics. Вместо generic KPI с большими числами приоритет отдан объяснению и действию инженера. Protected assets, логотипы, схема сети и полный layout KEGOC не копировались.

- Body 14px/1.45; heading 26px/600, section 16px/600, card 15px; captions 10–12px. Основные действия/объяснения 12–14px. Это наши значения. System font fallback работает без внешней загрузки шрифта.
- Две font roles: sans для UI, mono для feature names/versions; все numbers tabular. Русские labels, короткие all-caps captions с letter-spacing .09em.
- Spacing 4/8/12/16/20/24/32px; radius 4/6/8px; border 1px. Никаких декоративных gradients. Цветные поверхности получены из проверенных primitives с нашей opacity, а не объявлены CSS сайта.
- Tokens в `frontend/src/app/tokens.css`: primitives → semantic aliases → component tokens (`--button-bg`, `--card-border`, `--control-height`). Styles используют semantic/component vars; ECharts имеет согласованные palette values в canvas options.

## Семантика состояния

| Сущность | Label / цвет | Значение |
| --- | --- | --- |
| Соединение | Green dot + «Подключено» | Только транспорт доступен |
| Нет итогового alert | «Наблюдение», neutral | Не является подтверждением физической исправности |
| Diagnostic observation | Cyan | ML signal или экспериментальное статистическое наблюдение |
| Policy-confirmed warning | Amber, отдельная карточка | Последовательные сигналы прошли confirmation; требуется проверка |
| Critical | Red, зарезервировано | Без независимых критериев не активируется |
| ML class 0/1 | Отдельный raw block | Не диагноз, не severity, не calibrated future failure probability |
| Ground truth | «Факты набора» / source label | Историческая аннотация; не используется online inference/diagnostics |

Alert lifecycle: «Активно» → «Принято в работу» → «Закрыто». Acknowledgement не означает исчезновение сигнала; Reset/gap/data-quality interruption закрывают эпизод с явной причиной, не подтверждая восстановление оборудования. Archive отделён notice от текущих alerts.

## Компоненты

- **Alert card:** severity + turbine → объяснение → actual sensor evidence → source activation time/status. По нажатию открывает detail с action, confidence limitations и versioned policy.
- **Turbine card:** asset/scenario, требуется ли проверка, ветер/температура/мощность в исходных units, raw ML отдельной строкой. Нет физического «здоров» по class 0.
- **Replay console:** Start/Pause/Resume/Reset, speed, source clock, progress; настройки range/policy в disclosure. Processing UTC clock находится отдельно.
- **Sensor browser:** поиск по русскому названию или CSV key; 6 categories + recommended/all; физический sensor группирует statistics. Advanced toggle раскрывает min/max/std, неизвестный unit не выдумывается. Presets смешивают только одинаковые units.
- **Charts:** telemetry, model score и ground truth названы раздельно. DataZoom, scrollable legend, tooltip, source timestamp marker, focus на времени alert, 6h/24h/7d/all display ranges. Настоящий gap разрывает series; NaN не интерполируется.
- **Notifications:** одна inline notification на новый persistent alert ID, без модальных popups на каждом timestep; backlog при reconnect не повторяет старые уведомления.

Hover усиливает border/foreground; selected — accent surface; disabled снижает opacity. Focus-visible 2px accent. Кнопки имеют доступные labels; основные mobile actions 40–44px; Escape закрывает sensor browser. Не заявляется полная accessibility certification.

## Responsive и visual evidence

Desktop >1250px: горизонтальная navigation, 2 turbine cards, telemetry + diagnostics column. Tablet ≤950px: summary 2×2, diagnostics ниже charts, сохраняются labels. Mobile ≤700px: product tabs прокручиваются внутри header, cards в одну колонку, replay переносится, charts и alert details занимают ширину панели. Popovers ограничены viewport, таблицы прокручиваются внутри container.

До/после: `artifacts/v2/before_dashboard.jpg`, `before_alerts.jpg`, `before_monitor_tablet.jpg`; исходные screenshots получены в pane шириной 855px. Финальные desktop/mobile/tablet screenshots и точные viewport checks находятся в `artifacts/v2/` и описаны в [отчёте v2](ux_and_alerts_improvements.md). Previous design-system snapshot сохранён в `artifacts/v2/design_system_before.md`.
