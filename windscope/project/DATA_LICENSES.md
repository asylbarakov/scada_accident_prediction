# Источники и лицензии данных

## CARE to Compare v6 · Wind Farm A

Christian Gück и Cyriana Roelofs (Fraunhofer IEE), *Wind Turbine SCADA Data For Early Fault Detection*, Zenodo, версия 6, 9 июля 2025 г., DOI `10.5281/zenodo.15846963`. Для ссылки на методику авторы просят также цитировать Christian Gück, Cyriana M. A. Roelofs, Stefan Faulstich, *CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine Data*, Data 9(12), 138 (2024). [Запись Zenodo](https://zenodo.org/records/15846963) · [Статья](https://doi.org/10.3390/data9120138)

Лицензия **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** подтверждена 16.11.2026. API Zenodo, поле `metadata.license.id`, и DataCite, поле `rightsList`, возвращают `cc-by-sa-4.0`. Лицензия разрешает копирование, распространение и адаптацию с атрибуцией, указанием изменений и той же лицензией при распространении адаптации. Сохраняйте этот файл, ссылку на источник и лицензию при передаче данных дальше. [Zenodo API](https://zenodo.org/api/records/15846963) · [DataCite](https://api.datacite.org/dois/10.5281/zenodo.15846963) · [Условия CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)

| Путь | Что поставляется и какие изменения сделаны |
| --- | --- |
| Full · `data/CARE_To_Compare/Wind Farm A/` | 22 CSV и два файла метаданных исходного подкаталога, байты сохранены. Ветропарки B и C не включены. Исходный архив на Zenodo имеет MD5 `2547b58c21ac8c242d13232860cf500c`. |
| Demo · `data/CARE_To_Compare/Wind Farm A/datasets/` | Из 22 CSV выбраны строки из раздела `prediction`, десять использованных физических каналов и три служебных поля. Содержимое полей сохранено. |
| Full и Demo · `cache/care_h7_v6/` | Подготовленные данные CARE, в том числе признаки из прошлых окон. В Demo только строки признаков, соответствующие сохранённым почасовым прогнозам. |
| Full и Demo · `results/` | Результаты исследования. Атрибуция и CC BY-SA 4.0 распространяются на содержащиеся здесь выдержки и адаптации данных CARE, включая экспорт показаний и графики с ними. |

Эта атрибуция относится к данным CARE и их адаптациям. Источник данных и его авторы не заявляют поддержки проекта KEGOC.

## EDP · Mendeley v1

`sources/edp_mendeley_v1_failures.csv` — сверенная транскрипция двух изображений таблиц, 28 событий. Источник — Krzysztof Kijanowski, Tomasz Barszcz, Wieslaw Staszewski и Phong Dao, *Description of the SCADA dataset of the EDP onshore wind farm in Portugal*, Mendeley Data V1, DOI `10.17632/zjxjnjp3xs.1`. На странице указана CC BY 4.0. Файл входит в Full для отдельного EDP-трека. [Источник](https://data.mendeley.com/datasets/zjxjnjp3xs/1) · [Лицензия CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
