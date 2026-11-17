# Следующая задача для Codex на локальной машине

Работай в новой директории `scada-pdm-mvp` с приложенным стартовым пакетом. Никаких изменений в иных рабочих проектах, Git worktree, Docker volumes или незакоммиченных файлах. Не удаляй существующие данные. Не делай `git push` без прямого разрешения.

**Цель:** подтвердить пригодность CARE v6 / Wind Farm A для supervised Failure Prediction на 7/30 дней. Аномальность и физический отказ не смешивать.

1. Проверь `README.md` и `docs/data_audit.md`. Запусти `python -m unittest discover -s tests -v`. Не утверждай, что модель обучена — её пока нет.
2. Определи доступное свободное место и наличие Python 3.11. Для CARE нужен архив 5.5 ГБ и ~19–20 ГБ распакованных файлов. **Перед скачиванием большого архива** предложи точные команды загрузки в `data/` и проверку checksum; сам не скачивай, пока нет отдельного разрешения.
3. Если данные CARE v6 уже лежат локально и доступны по пути `data/CARE_To_Compare`, запусти:
   `python scripts/care_readiness_audit.py --raw-dir data/CARE_To_Compare --farm A --scan-series --horizons 7 30 --output results/care_a_coverage.json`.
4. Просмотри 3–5 `event_info` + SCADA CSV вручную, проверь соответствие `event_end` / `event_end_id` реальной границе отказа и возможность horizon labeling без использования `status_type_id` в качестве будущего предиктора.
5. Если H30 недостаточно покрыт, изучи EDP SCADA v2 на Mendeley (2016–2017; 28 fault rows) как альтернативу для supervised.
6. Сохрани findings в `docs/local_data_gate.md` с агрегированными counts H7/H30, примерами событий, обнаруженными проблемами и рекомендацией CARE/EDP. Верни результаты, не переходи к обучению и UI до решения data gate.

Контроль качества: не соединять anonymized CARE event files по timestamp; `event_start` ≠ начало отказа, ориентир — `event_end` (по статье CARE). Не называть anomaly-score вероятностью отказа. Обязательно отмечать right-censored negative rows и маленькое число независимых failure events.
