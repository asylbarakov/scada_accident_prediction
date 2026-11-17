"""Plotly figures for the read-only CARE H7 dashboard."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .dashboard_data import MODELS, chart_grid

MODEL_COLORS = {"logistic": "#007969", "lightgbm": "#b36322"}
SENSOR_NAMES = {
    "ambient_temperature": "Температура воздуха", "wind_speed": "Скорость ветра",
    "grid_active_power": "Активная мощность", "gearbox_bearing_temperature": "Подшипник редуктора",
    "gearbox_oil_temperature": "Температура масла редуктора",
    "generator_bearing_temperature": "Подшипник генератора",
    "generator_speed": "Обороты генератора", "transformer_temperature": "Трансформатор L1",
    "hydraulic_oil_temperature": "Температура гидравлического масла", "rotor_speed": "Обороты ротора",
}
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}


def sensor_name(sensor):
    return SENSOR_NAMES.get(sensor, sensor)


def feature_name(feature):
    sensor, _, details = feature.partition("__")
    if details == "now":
        return f"{sensor_name(sensor)} · последнее"
    window, _, statistic = details.partition("_")
    kind = {"mean": "среднее", "max": "максимум", "std": "разброс",
            "delta_mean": "откл. от среднего", "trend": "тренд"}.get(statistic, statistic)
    return f"{sensor_name(sensor)} · {kind} {window.removesuffix('h')} ч" if details else feature


def style(fig, height):
    return fig.update_layout(
        height=height, paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Inter, Segoe UI, sans-serif", "size": 12, "color": "#304541"},
        margin={"l": 48, "r": 26, "t": 35, "b": 52},
        legend={"orientation": "h", "y": 1.04, "x": 0, "yanchor": "bottom", "font": {"size": 11}},
        hoverlabel={"bgcolor": "white", "font_color": "#172c29"},
        hovermode="x unified",
    ).update_xaxes(gridcolor="#edf2ef", linecolor="#dae5df", zeroline=False).update_yaxes(
        gridcolor="#e8efeb", linecolor="#dae5df", zeroline=False)


def fault_marker(fig, fault):
    if fault is None:
        return
    value = fault.to_pydatetime()
    fig.add_shape(type="line", x0=value, x1=value, y0=0, y1=1, xref="x", yref="paper",
                  line={"color": "#9b3a45", "dash": "dash", "width": 1.5})
    fig.add_annotation(x=value, y=1, yref="paper", text="Отказ CARE", showarrow=False, xanchor="right",
                       yanchor="top", bgcolor="#fff2f3", borderpad=4, font={"color": "#8c3340", "size": 11})


def prediction_figure(frame, inference, cadence_minutes, fault=None, *, compact=False):
    fig = go.Figure()
    for model, label in MODELS.items():
        group = frame.loc[frame["model"].eq(model)]
        if group.empty:
            continue
        color = MODEL_COLORS[model]
        series = chart_grid(group, "score", cadence_minutes)
        fig.add_trace(go.Scatter(x=series.index, y=series, name=label, connectgaps=False,
                                 mode="lines+markers" if series.notna().sum() == 1 else "lines",
                                 line={"color": color, "width": 2},
                                 hovertemplate="%{y:.8f}<extra>%{fullData.name}</extra>"))
        first = group.loc[group["alert"]].sort_values("timestamp").head(1)
        if not first.empty:
            fig.add_trace(go.Scatter(x=first["timestamp"], y=first["score"], name=f"Первое · {label}",
                                     mode="markers", marker={"color": color, "size": 10, "symbol": "diamond",
                                                             "line": {"color": "white", "width": 1.4}},
                                     showlegend=not compact))
        threshold = inference[model]["threshold"]
        if not compact and 0 <= threshold <= 1:
            fig.add_hline(y=threshold, line_color=color, line_dash="dot", line_width=1)
    fault_marker(fig, fault)
    style(fig, 300 if compact else 425)
    fig.update_xaxes(title_text="Время CARE (UTC)", hoverformat="%d.%m.%Y %H:%M")
    fig.update_yaxes(title_text="H7 score", range=[-0.025, 1.04], tickformat=".2f")
    return fig


def telemetry_figure(frame, sensors, selected, cadence_minutes, fault=None):
    dual = len(selected) == 2 and sensors[selected[0]]["unit"] != sensors[selected[1]]["unit"]
    fig = make_subplots(specs=[[{"secondary_y": dual}]])
    for index, sensor in enumerate(selected):
        unit = sensors[sensor]["unit"]
        values = chart_grid(frame, sensor, cadence_minutes)
        secondary = dual and index == 1
        fig.add_trace(go.Scatter(x=values.index, y=values, name=sensor_name(sensor),
                                 mode="lines+markers" if values.notna().sum() == 1 else "lines",
                                 line={"color": ("#007969", "#b36322")[index], "width": 1.8},
                                 connectgaps=False,
                                 hovertemplate=f"%{{y:.2f}} {unit}<extra>%{{fullData.name}}</extra>"),
                      secondary_y=secondary)
        fig.update_yaxes(title_text=unit, secondary_y=secondary)
    fault_marker(fig, fault)
    style(fig, 435)
    fig.update_xaxes(title_text="Время CARE (UTC)", hoverformat="%d.%m.%Y %H:%M",
                     rangeslider={"visible": True, "thickness": 0.09})
    if dual:
        fig.update_yaxes(showgrid=False, secondary_y=True)
    return fig


def importance_figure(frame, labels, value, axis_title, *, limit=9, signed=False):
    scores = frame.assign(label=labels)
    names = scores[value].abs().nlargest(limit).index
    if signed:
        remainder = scores.loc[~scores.index.isin(names), value].sum()
        scores = pd.concat([scores.loc[names], pd.DataFrame({"label": ["Остальные признаки"], value: [remainder]})],
                           ignore_index=True)
    else:
        scores = scores.loc[names]
    scores = scores.sort_values(value, key=lambda x: x.abs())
    colors = ["#007969" if number >= 0 else "#ad5540" for number in scores[value]]
    fig = go.Figure(go.Bar(x=scores[value], y=scores["label"], orientation="h", marker_color=colors,
                           hovertemplate="%{y}<br>%{x:.5g}<extra></extra>"))
    style(fig, max(270, 54 + len(scores) * 27)).update_layout(hovermode="closest", margin={"l": 10, "r": 20, "t": 10, "b": 47})
    fig.update_xaxes(title_text=axis_title, zeroline=signed, zerolinecolor="#9cafaa")
    fig.update_yaxes(automargin=True, showgrid=False)
    return fig
