"""Local, read-only Streamlit demo for the archived CARE H7 experiment."""

import os
from pathlib import Path
from datetime import timedelta

import pandas as pd
import streamlit as st

from scada_pdm.dashboard_charts import (
    PLOT_CONFIG, feature_name, importance_figure, prediction_figure, sensor_name, telemetry_figure,
)
from scada_pdm.dashboard_data import (
    MODELS, ArtifactError, alert_summary, explain_point, load_experiment, load_features, load_pipelines,
    load_telemetry, read_artifact, visible_fault, visible_rows,
)

ROOT = Path(__file__).resolve().parent
RUN = Path(os.getenv("SCADA_DASHBOARD_RUN", ROOT / "results" / "care_h7_windows_20261116"))
RAW = Path(os.getenv("SCADA_DASHBOARD_RAW", ROOT / "data" / "CARE_To_Compare" / "Wind Farm A"))
FEATURES = Path(os.getenv("SCADA_DASHBOARD_FEATURES", ROOT / "cache" / "care_h7_v6" / "features.parquet"))
PAGES = {
    "Overview": ("Обзор парка", "Оборудование, предупреждения и результаты эксперимента"),
    "SCADA Monitoring": ("Мониторинг SCADA", "Фактические показания датчиков и их изменение во времени"),
    "Failure Prediction": ("Прогноз отказа", "Архивные оценки H7 и пороги, выбранные на validation"),
    "Model Explainability": ("Объяснимость модели", "Gain, TreeSHAP и исходные показания датчиков"),
}
FOLDS = {"train": "Train", "validation": "Validation", "test": "Test"}

CSS = """
<style>
[data-testid="stAppViewContainer"] {background:#f2f6f4}
[data-testid="stHeader"] {background:transparent}
[data-testid="stMainBlockContainer"] {max-width:1320px;padding-top:2rem;padding-bottom:4rem}
[data-testid="stSidebar"] {background:#e7efeb;border-right:1px solid #d9e5de}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding-top:1.1rem}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.30rem}
[data-testid="stSidebar"] label[data-baseweb="radio"] {padding:.52rem .55rem;border-radius:7px}
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {background:#f8fcfa;box-shadow:0 0 0 1px #d4e4da}
[data-testid="stVerticalBlockBorderWrapper"] {background:#fff;border-color:#dbe7df!important;border-radius:12px!important}
[data-testid="stMetric"] {min-height:85px}
.brand {display:flex;align-items:center;gap:11px;margin:.2rem 0 1.55rem}
.brand-mark {display:grid;place-items:center;width:40px;height:40px;background:#0b6659;color:white;border-radius:11px;
font-size:14px;font-weight:800;letter-spacing:.04em}
.brand-title {font-size:16px;font-weight:780;letter-spacing:-.02em;color:#173730}
.brand-sub {font-size:11px;color:#4a635b;line-height:1.4}
.page-head {display:flex;justify-content:space-between;align-items:flex-start;gap:16px;padding:.3rem 0 .26rem}
.page-head h1 {margin:0;padding:0;font-size:clamp(2rem,4vw,3rem);letter-spacing:-.052em;line-height:1.1}
.page-head p {margin:.43rem 0 0;color:#4e665e;font-size:.94rem}
.study {font-size:.76rem;font-weight:650;white-space:nowrap;border:1px solid #c8ded2;color:#245748;
background:#e7f3ec;padding:.42rem .69rem;border-radius:100px}
.mini-model {color:#516763;font-size:.80rem;font-weight:620;margin-bottom:.1rem}
.card-value {font-size:clamp(1.15rem,2.1vw,1.65rem);font-weight:750;line-height:1.15;letter-spacing:-.035em;color:#172c29}
.card-value.on {color:#087263}
.card-value.off {color:#975322}
.compact-table {width:100%;margin:.5rem 0 .1rem;border-collapse:collapse;font-size:.86rem}
.compact-table th,.compact-table td {padding:.68rem .35rem;text-align:right;border-bottom:1px solid #e1ebe5}
.compact-table th:first-child,.compact-table td:first-child {text-align:left}
.compact-table th {color:#526662;font-size:.75rem}
@media (max-width:680px) {
  [data-testid="stMainBlockContainer"] {padding-left:1rem;padding-right:1rem}
  .page-head {flex-wrap:wrap}.study {order:-1}
}
</style>
"""


@st.cache_resource(show_spinner=False)
def experiment():
    return load_experiment(RUN)


@st.cache_data(show_spinner=False, max_entries=12)
def event_telemetry(event_id):
    return load_telemetry(experiment(), RAW, event_id)


@st.cache_resource(show_spinner=False)
def fitted_models():
    return load_pipelines(experiment())


@st.cache_data(show_spinner=False)
def cached_features():
    return load_features(FEATURES, experiment().inference["lightgbm"]["feature_names"])


@st.cache_data(show_spinner=False)
def explanation_table(filename):
    return read_artifact(RUN / "training" / "h7" / "lightgbm" / filename, pd.read_csv)


def dt(value, *, seconds=False):
    if value is None or pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%d.%m.%Y %H:%M:%S UTC" if seconds else "%d.%m.%Y %H:%M UTC")


def plot(fig, key):
    st.plotly_chart(fig, theme=None, config=PLOT_CONFIG, key=key)


def load_scada(event):
    try:
        return event_telemetry(event.id)
    except ArtifactError as exc:
        st.warning(f"Телеметрия недоступна. {exc}")
        return None


def snapshot_slider(event, saved):
    telemetry = load_scada(event)
    limits = []
    if telemetry is not None:
        start, end = telemetry.frame["timestamp"].agg(["min", "max"])
        delay = pd.Timedelta(minutes=telemetry.available_after_minutes)
        limits.extend([start + delay, end + delay])
    if not saved.empty:
        limits.extend(saved["timestamp"].agg(["min", "max"]))
    if not limits:
        st.info("Для воспроизведения нужна телеметрия или сохранённые прогнозы события.")
        return pd.NaT
    start, end = min(limits).to_pydatetime(), max(limits).to_pydatetime()
    if start == end:
        return pd.Timestamp(start)
    selected = st.slider("Исторический срез (UTC)", min_value=start, max_value=end, value=start,
                         step=timedelta(minutes=10), format="DD.MM.YYYY HH:mm", key=f"cursor_{event.id}")
    return pd.Timestamp(selected)


def show_context(data):
    turbines = sorted({event.turbine for event in data.events.values()}, key=int)
    default = "21" if "21" in turbines else turbines[0]
    columns = st.columns([1, 1, 1.6], vertical_alignment="bottom")
    turbine = columns[0].selectbox("Турбина", turbines, index=turbines.index(default),
                                  format_func=lambda x: f"Турбина {x}", key="turbine")
    events = sorted(event.id for event in data.events.values() if event.turbine == turbine)
    start = events.index(22) if 22 in events else 0
    event_id = columns[1].selectbox("Историческое событие", events, index=start,
                                   format_func=lambda x: f"Событие {x}", key=f"event_{turbine}")
    mode = columns[2].radio("Режим просмотра", ("История", "Воспроизведение"), horizontal=True, key="mode")
    event = data.events[event_id]
    archived = data.predictions.loc[data.predictions["source_event"].eq(event_id)]
    as_of = snapshot_slider(event, archived) if mode == "Воспроизведение" else None
    st.caption(f"{FOLDS[event.fold]} · турбина {event.turbine} · файл {event.id} · H7")
    return event, as_of, visible_rows(archived, as_of), visible_fault(event, as_of)


def score_card(label, value, note, tone=""):
    with st.container(border=True):
        st.html(f'<div class="mini-model">{label}</div><div class="card-value {tone}">{value}</div>')
        st.caption(note)


def no_predictions(event, as_of):
    if event.fold == "train":
        return "Архивные прогнозы сохранены для validation и test."
    if as_of is not None:
        return "До выбранного среза сохранённых прогнозов нет."
    return "Для этого события нет сохранённых прогнозов."


def overview(data, event, as_of, forecasts, fault):
    summary = alert_summary(forecasts, data, as_of)
    row = st.columns(3)
    with row[0]:
        score_card("Горизонт", "H7", "Отказ в следующие 7 суток")
    for column, (model, name) in zip(row[1:], MODELS.items(), strict=True):
        item = summary[model]
        latest, first = item["latest"], item["first_alert"]
        with column:
            if latest is None or item["stale"]:
                score_card(name, "Нет оценки", "До среза нет актуальной точки" if as_of is not None else "Нет архивного прогноза")
            elif item["disabled"]:
                score_card(name, "Оповещения отключены", "Порог validation выше диапазона score", "off")
            elif (as_of is None and first is not None) or (as_of is not None and latest["alert"]):
                score_card(name, "Есть предупреждение", f"Первое · {dt(first)}", "on")
            else:
                note = f"Первое ранее · {dt(first)}" if first is not None else f"Оценка · {dt(latest['timestamp'])}"
                score_card(name, "Без предупреждения", note)
    main, side = st.columns([1.65, 1], gap="medium")
    with main, st.container(border=True):
        st.subheader("Ход прогноза")
        if forecasts.empty:
            st.info(no_predictions(event, as_of))
        else:
            plot(prediction_figure(forecasts, data.inference, data.cadence_minutes, fault, compact=True), "overview_scores")
    with side, st.container(border=True):
        if as_of is None:
            st.subheader("Результаты теста")
            test = data.models["lightgbm"]["test"]
            st.caption(f"{test['eligible_events']} отказа · {test['eligible_rows']} почасовые точки")
            rows = []
            for model, label in MODELS.items():
                m = data.models[model]["test"]
                rows.append(f"<tr><td>{label}</td><td>{m['pr_auc_average_precision']:.4f}</td>"
                            f"<td>{m['roc_auc']:.4f}</td><td>{m['detected_events']}/{m['eligible_events']}</td></tr>")
            st.html('<table class="compact-table" aria-label="Отложенный тест"><thead><tr>'
                    '<th>Модель</th><th>AP</th><th>ROC AUC</th><th>Recall</th></tr></thead><tbody>'
                    + "".join(rows) + '</tbody></table>')
            st.caption("LightGBM не обнаружил тестовые отказы при зафиксированном пороге.")
        else:
            st.subheader("Состояние на срезе")
            st.write(f"Отметка CARE · {dt(fault)}" if fault is not None else "До среза отметка отказа отсутствует.")
            st.caption("Сводные итоги эксперимента доступны в режиме «История».")
    if as_of is None:
        with st.container(border=True):
            st.subheader("Оборудование")
            fleet = []
            for asset in sorted({item.turbine for item in data.events.values()}, key=int):
                files = [item for item in data.events.values() if item.turbine == asset]
                evaluated = data.predictions.loc[data.predictions["turbine_id"].eq(asset)]
                fleet.append({"Турбина": asset, "Выборка": FOLDS[files[0].fold], "Файлов SCADA": len(files),
                              "Событий с архивом score": int(evaluated["source_event"].nunique())})
            st.dataframe(pd.DataFrame(fleet), hide_index=True, width="stretch")


def telemetry_panel(data, event, as_of, fault, *, compare=False, ranked=False):
    telemetry = load_scada(event)
    if telemetry is None:
        return
    visible = visible_rows(telemetry.frame, as_of, telemetry.available_after_minutes)
    sensors = [key for key in data.sensors if key in visible]
    if ranked:
        try:
            order = explanation_table("sensor_importance.csv")["sensor"].tolist()
            sensors = [key for key in order if key in sensors]
        except (ArtifactError, KeyError):
            pass
    if visible.empty or not sensors:
        st.info("До среза нет доступных показаний датчиков.")
        return
    selected = []
    left, right = st.columns(2)
    with left:
        default = sensors.index("ambient_temperature") if "ambient_temperature" in sensors else 0
        selected.append(st.selectbox("Датчик", sensors, index=default, format_func=sensor_name,
                                     key=f"sensor_{event.id}_{compare}"))
    if compare:
        with right:
            secondary = [None, *(key for key in sensors if key != selected[0])]
            reference = st.selectbox("Сравнить с", secondary,
                                     format_func=lambda x: sensor_name(x) if x else "Без сравнения",
                                     key=f"compare_{event.id}_{selected[0]}")
            if reference is not None:
                selected.append(reference)
    with st.container(border=True):
        plot(telemetry_figure(visible, data.sensors, selected, telemetry.stride_minutes, fault), f"telemetry_{compare}")
        scope = "Исходные SCADA · шаг 10 минут" if telemetry.from_raw else "Архив показаний · почасовые тестовые точки"
        st.caption(f"{scope} · {', '.join(data.sensors[key]['column'] for key in selected)}")
        if as_of is not None:
            st.caption(f"Для исторического среза учтена задержка доступности {telemetry.available_after_minutes} мин.")


def failure_prediction(data, event, as_of, forecasts, fault):
    with st.container(border=True):
        st.subheader("Оценки H7 и предупреждения")
        if not forecasts.empty:
            plot(prediction_figure(forecasts, data.inference, data.cadence_minutes, fault), "failure_chart")
        else:
            st.info(no_predictions(event, as_of))
        if fault is not None:
            st.caption(f"Зафиксированный отказ CARE · {dt(fault)}")
        elif as_of is not None:
            st.caption("До среза отметка отказа отсутствует.")
        elif event.related_anchor is not None:
            st.caption(f"Разметка связана с эпизодом {event.related_anchor}. Отдельный якорь для этого файла не использован.")
        else:
            st.caption("Для этого файла нет отдельной отметки отказа в оценке H7.")
    rows = []
    for model, entry in alert_summary(forecasts, data, as_of).items():
        first = entry["first_alert"]
        rows.append({"Модель": MODELS[model], "Порог validation": repr(data.inference[model]["threshold"]),
                     "Первое предупреждение": dt(first),
                     "Оповещения": "Отключены порогом" if entry["disabled"] else "По сохранённым флагам"})
    with st.container(border=True):
        st.subheader("Пороги и первое предупреждение")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Ромб показывает первую сохранённую отметку предупреждения. Порог LightGBM выше диапазона score.")
        if as_of is None:
            result = data.models["lightgbm"]["test"]
            st.caption(f"LightGBM на отложенном тесте · обнаружено {result['detected_events']} из "
                       f"{result['eligible_events']} отказов.")


def rankings():
    try:
        features = explanation_table("feature_importance.csv")
        sensors = explanation_table("sensor_importance.csv")
        for column, (metric, title, unit) in zip(st.columns(2), (
            ("gain", "LightGBM gain", "Суммарный gain"),
            ("mean_abs_shap_log_odds", "Global TreeSHAP", "Среднее |SHAP|, log-odds"),
        ), strict=True):
            with column, st.container(border=True):
                st.subheader(title)
                plot(importance_figure(features, features["feature"].map(feature_name), metric, unit, limit=8), metric)
        with st.container(border=True):
            st.subheader("Важность физических датчиков")
            plot(importance_figure(sensors, sensors["sensor"].map(sensor_name), "mean_abs_grouped_shap_log_odds",
                                   "Средний |суммарный SHAP|, log-odds", limit=10), "sensor_gain")
            st.caption("Вклады признаков одного датчика суммируются для каждой тестовой точки.")
    except (ArtifactError, KeyError) as exc:
        st.warning(f"Глобальные объяснения недоступны. {exc}")


def local_explanation(data, event, forecasts):
    if forecasts.empty:
        st.info("Для локального объяснения нужен архивный прогноз в доступном интервале.")
        return
    points = forecasts["timestamp"].drop_duplicates().sort_values().tolist()
    active = st.selectbox("Момент прогноза", points, index=len(points) - 1, format_func=dt,
                          key=f"explain_timestamp_{event.id}")
    try:
        items, base, scores = explain_point(cached_features(), fitted_models(), data, event, active, forecasts)
        st.caption("Обе модели воспроизвели сохранённые score с исходной предобработкой.")
    except ArtifactError as exc:
        st.warning(f"Повторный расчёт недоступен. {exc}")
        try:
            archive = explanation_table("local_shap.csv")
            pick = archive.loc[archive["turbine_id"].astype(str).eq(event.turbine)
                               & pd.to_datetime(archive["timestamp"], utc=True).eq(active)]
            if pick.empty:
                st.info("Для этого момента нет сохранённого локального SHAP.")
                return
            items, base = pick, pick["base_log_odds"].iloc[0]
            scores = forecasts.loc[forecasts["timestamp"].eq(active)].set_index("model")["score"].to_dict()
            st.caption("Показано сохранённое объяснение выбранной точки.")
        except (ArtifactError, KeyError) as fallback:
            st.info(f"Архив локальных объяснений недоступен. {fallback}")
            return
    with st.container(border=True):
        columns = st.columns(3)
        columns[0].metric("Logistic Regression · score", f"{scores['logistic']:.6f}")
        columns[1].metric("LightGBM · score", f"{scores['lightgbm']:.6f}")
        columns[2].metric("Базовое значение LightGBM", f"{base:.3f} log-odds")
        plot(importance_figure(items, items["feature"].map(feature_name), "shap_log_odds",
                               "Вклад в score модели, log-odds", signed=True, limit=10), "local_shap")


def explainability(data, event, as_of, forecasts, fault):
    sections = ["Локальный SHAP", "История датчика"] if as_of is not None else [
        "Важность", "Локальный SHAP", "История датчика",
    ]
    section = st.radio("Раздел объяснения", sections, horizontal=True, key=f"explain_{as_of is None}")
    if section == "Важность":
        rankings()
    elif section == "Локальный SHAP":
        local_explanation(data, event, forecasts)
    else:
        telemetry_panel(data, event, as_of, fault, ranked=as_of is None)
    st.caption("SHAP объясняет вклад признаков в оценку модели. Для установления причины отказа нужны отдельные данные.")


def main():
    st.set_page_config(page_title="SCADA H7 · Wind Farm A", layout="wide", initial_sidebar_state="expanded")
    st.html(CSS)
    with st.sidebar:
        st.html('<div class="brand"><div class="brand-mark">H7</div><div class="brand-title">SCADA'
                '<div class="brand-sub">Predictive Maintenance</div></div></div>')
        page = st.radio("Раздел", tuple(PAGES), key="page", label_visibility="collapsed")
        st.divider()
        st.caption("CARE v6 · Wind Farm A")
        st.caption("Архивный эксперимент · некалиброванный score")
    title, subtitle = PAGES[page]
    st.html(f'<header class="page-head"><div><h1>{title}</h1><p>{subtitle}</p></div>'
            '<span class="study">Исследовательский режим</span></header>')
    try:
        data = experiment()
    except ArtifactError as exc:
        st.error(f"Не удалось открыть эксперимент CARE H7. {exc}")
        st.stop()
    event, as_of, forecasts, fault = show_context(data)
    if page == "Overview":
        overview(data, event, as_of, forecasts, fault)
    elif page == "SCADA Monitoring":
        telemetry_panel(data, event, as_of, fault, compare=True)
    elif page == "Failure Prediction":
        failure_prediction(data, event, as_of, forecasts, fault)
    else:
        explainability(data, event, as_of, forecasts, fault)


if __name__ == "__main__":
    main()
