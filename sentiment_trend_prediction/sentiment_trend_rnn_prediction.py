import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUTS = [
    PROJECT_DIR / "weibo/search_comments_2026-04-17.jsonl",
    PROJECT_DIR / "xhs/jsonl/search_comments_2026-04-06.jsonl",
    PROJECT_DIR / "douyin/douyin_naixue_simulated_comments.jsonl",
    PROJECT_DIR / "dianping/dianping_naixue_simulated_comments.jsonl",
]
DEFAULT_OUTPUT = SCRIPT_DIR / "sentiment_trend_rnn_prediction.html"
DEFAULT_FUTURE_OUTPUT = SCRIPT_DIR / "sentiment_trend_rnn_prediction_future.html"
DEFAULT_GRANULARITY = "month"
TRAIN_RATIO = 0.8
DEFAULT_WINDOW = 4
DEFAULT_EPOCHS = 260
DEFAULT_HIDDEN_SIZE = 24
DEFAULT_FORECAST_END_DATE = date(2026, 9, 30)
POLARITIES = ("positive", "neutral", "negative")
POLARITY_LABELS = {
    "positive": "正向情感",
    "neutral": "中性情感",
    "negative": "负向情感",
}
POLARITY_COLORS = {
    "positive": "#4FAE78",
    "neutral": "#6E7581",
    "negative": "#E34D4D",
}


class RNNRegressor(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True, nonlinearity="tanh")
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        outputs, _ = self.rnn(inputs)
        return self.output(outputs[:, -1, :])


def iter_json_records(file_path: Path):
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{file_path} is a JSON value, but not a JSON array")
        yield from data
        return

    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        record, index = decoder.raw_decode(text, index)
        yield record


def normalize_timestamp(value):
    if value is None or value == "":
        return None

    timestamp = float(value)
    if timestamp > 10_000_000_000:
        timestamp /= 1000
    return timestamp


def record_date(record: dict, timezone: ZoneInfo):
    timestamp = normalize_timestamp(record.get("create_time"))
    if timestamp is not None:
        return datetime.fromtimestamp(timestamp, timezone).date()

    date_time = record.get("create_date_time")
    if not date_time:
        return None

    return datetime.fromisoformat(str(date_time)).astimezone(timezone).date()


def record_polarity(record: dict):
    sentiment = record.get("attributes", {}).get("sentiment", {})
    polarity = str(sentiment.get("polarity", "")).lower()
    if polarity in POLARITIES:
        return polarity
    return None


def aggregate_daily_counts(input_files, timezone: ZoneInfo):
    daily_counts = defaultdict(Counter)
    source_counts = Counter()
    skipped = Counter()

    for file_path in input_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        source_name = file_path.parts[-2] if file_path.parent.name != "jsonl" else file_path.parts[-3]
        for record in iter_json_records(file_path):
            day = record_date(record, timezone)
            polarity = record_polarity(record)

            if day is None:
                skipped["missing_time"] += 1
                continue
            if polarity is None:
                skipped["missing_polarity"] += 1
                continue

            daily_counts[day][polarity] += 1
            source_counts[source_name] += 1

    return daily_counts, source_counts, skipped


def filter_daily_counts(daily_counts, start_date: date | None, end_date: date | None):
    if start_date is None and end_date is None:
        return daily_counts

    return {
        day: counts
        for day, counts in daily_counts.items()
        if (start_date is None or day >= start_date) and (end_date is None or day <= end_date)
    }


def bucket_start(day: date, granularity: str):
    if granularity == "day":
        return day
    if granularity == "week":
        return date.fromordinal(day.toordinal() - day.weekday())
    if granularity == "month":
        return date(day.year, day.month, 1)
    raise ValueError(f"Unsupported granularity: {granularity}")


def bucket_label(day: date, granularity: str):
    if granularity == "day":
        return day.strftime("%Y-%m-%d")
    if granularity == "week":
        return day.strftime("%Y-W%W")
    if granularity == "month":
        return day.strftime("%Y-%m")
    raise ValueError(f"Unsupported granularity: {granularity}")


def aggregate_by_granularity(daily_counts, granularity: str):
    grouped_counts = defaultdict(Counter)
    for day, counts in daily_counts.items():
        grouped_counts[bucket_start(day, granularity)].update(counts)
    return grouped_counts


def build_rows(grouped_counts, granularity: str):
    rows = []
    for bucket_index, day in enumerate(sorted(grouped_counts)):
        total = sum(grouped_counts[day].values())
        if not total:
            continue

        row = {
            "index": bucket_index,
            "day": day,
            "label": bucket_label(day, granularity),
            "total": total,
        }
        for polarity in POLARITIES:
            row[polarity] = grouped_counts[day][polarity] / total * 100
            row[f"{polarity}_count"] = grouped_counts[day][polarity]
        rows.append(row)
    return rows


def next_bucket(day: date, granularity: str):
    if granularity == "day":
        return day.fromordinal(day.toordinal() + 1)
    if granularity == "week":
        return day.fromordinal(day.toordinal() + 7)
    if granularity == "month":
        if day.month == 12:
            return date(day.year + 1, 1, 1)
        return date(day.year, day.month + 1, 1)
    raise ValueError(f"Unsupported granularity: {granularity}")


def future_bucket_days(last_day: date, end_date: date, granularity: str):
    target_day = bucket_start(end_date, granularity)
    days = []
    current = next_bucket(last_day, granularity)
    while current <= target_day:
        days.append(current)
        current = next_bucket(current, granularity)
    return days


def split_rows(rows):
    if len(rows) < 2:
        raise ValueError("At least two time buckets are required for train/test prediction.")
    split_index = min(max(1, int(len(rows) * TRAIN_RATIO)), len(rows) - 1)
    return split_index, rows[:split_index], rows[split_index:]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clamp_percentage(value: float):
    return min(100.0, max(0.0, value))


def format_stat(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isinf(value):
        return "Inf"
    return f"{value:.{digits}f}"


def choose_window_size(train_length: int, requested: int):
    return max(1, min(requested, train_length - 1))


def build_training_samples(series, window_size):
    samples_x = []
    samples_y = []
    for start in range(len(series) - window_size):
        samples_x.append(series[start:start + window_size])
        samples_y.append(series[start + window_size])
    return np.array(samples_x, dtype=np.float32), np.array(samples_y, dtype=np.float32)


def fit_rnn_series(train_series, full_series, test_length, window_size, epochs, hidden_size):
    actual_window = choose_window_size(len(train_series), window_size)
    if len(train_series) <= 1:
        fallback = [train_series[-1]] * test_length
        return {
            "predicted_test": fallback,
            "predicted_all": [None] * (len(full_series) - test_length) + fallback,
            "window_size": 1,
            "epochs": 0,
            "train_loss": None,
        }

    if len(train_series) <= actual_window:
        actual_window = max(1, len(train_series) - 1)

    sample_x, sample_y = build_training_samples(train_series, actual_window)
    if len(sample_x) == 0:
        fallback = [train_series[-1]] * test_length
        return {
            "predicted_test": fallback,
            "predicted_all": [None] * (len(full_series) - test_length) + fallback,
            "window_size": actual_window,
            "epochs": 0,
            "train_loss": None,
        }

    train_min = float(np.min(train_series))
    train_max = float(np.max(train_series))
    scale = train_max - train_min
    if scale == 0:
        fallback = [train_series[-1]] * test_length
        return {
            "predicted_test": fallback,
            "predicted_all": [None] * (len(full_series) - test_length) + fallback,
            "window_size": actual_window,
            "epochs": 0,
            "train_loss": 0.0,
        }

    sample_x = (sample_x - train_min) / scale
    sample_y = (sample_y - train_min) / scale

    x_tensor = torch.tensor(sample_x[:, :, None], dtype=torch.float32)
    y_tensor = torch.tensor(sample_y[:, None], dtype=torch.float32)

    model = RNNRegressor(hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    final_loss = None
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(x_tensor)
        loss = loss_fn(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    model.eval()
    history = list(full_series[:len(train_series)])
    predicted_test = []
    with torch.no_grad():
        for step in range(test_length):
            window = np.array(history[-actual_window:], dtype=np.float32)
            window = ((window - train_min) / scale).reshape(1, actual_window, 1)
            prediction = model(torch.tensor(window, dtype=torch.float32)).item()
            prediction = clamp_percentage(prediction * scale + train_min)
            predicted_test.append(prediction)
            history.append(full_series[len(train_series) + step])

    predicted_all = [None] * len(train_series) + predicted_test
    return {
        "predicted_test": predicted_test,
        "predicted_all": predicted_all,
        "window_size": actual_window,
        "epochs": epochs,
        "train_loss": final_loss,
    }


def forecast_rnn_series(full_series, forecast_steps, window_size, epochs, hidden_size):
    actual_window = choose_window_size(len(full_series), window_size)
    if len(full_series) <= 1:
        return {"predicted_future": [full_series[-1]] * forecast_steps}

    if len(full_series) <= actual_window:
        actual_window = max(1, len(full_series) - 1)

    sample_x, sample_y = build_training_samples(full_series, actual_window)
    if len(sample_x) == 0:
        return {"predicted_future": [full_series[-1]] * forecast_steps}

    train_min = float(np.min(full_series))
    train_max = float(np.max(full_series))
    scale = train_max - train_min
    if scale == 0:
        return {"predicted_future": [full_series[-1]] * forecast_steps}

    sample_x = (sample_x - train_min) / scale
    sample_y = (sample_y - train_min) / scale
    x_tensor = torch.tensor(sample_x[:, :, None], dtype=torch.float32)
    y_tensor = torch.tensor(sample_y[:, None], dtype=torch.float32)

    model = RNNRegressor(hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(x_tensor)
        loss = loss_fn(predictions, y_tensor)
        loss.backward()
        optimizer.step()

    history = list(full_series)
    predicted_future = []
    model.eval()
    with torch.no_grad():
        for _ in range(forecast_steps):
            window = np.array(history[-actual_window:], dtype=np.float32)
            window = ((window - train_min) / scale).reshape(1, actual_window, 1)
            prediction = model(torch.tensor(window, dtype=torch.float32)).item()
            prediction = clamp_percentage(prediction * scale + train_min)
            predicted_future.append(prediction)
            history.append(prediction)

    return {"predicted_future": predicted_future}


def compute_test_proportion_metrics(test_rows, predictions_by_polarity):
    actual_values = []
    predicted_values = []

    for test_index, row in enumerate(test_rows):
        for polarity in POLARITIES:
            actual_values.append(row[polarity])
            predicted_values.append(predictions_by_polarity[polarity][test_index])

    pair_count = len(actual_values)
    mae = sum(abs(actual - predicted) for actual, predicted in zip(actual_values, predicted_values)) / pair_count if pair_count else 0.0
    accuracy = max(0.0, 1.0 - mae / 100.0)

    overlap = sum(min(actual, predicted) for actual, predicted in zip(actual_values, predicted_values))
    predicted_total = sum(predicted_values)
    actual_total = sum(actual_values)
    precision = overlap / predicted_total if predicted_total else 0.0
    recall = overlap / actual_total if actual_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return accuracy, f1, precision


def build_prediction_figure(rows, train_rows, test_rows, model_results):
    split_index = len(train_rows)
    labels = [row["label"] for row in rows]
    test_labels = [row["label"] for row in test_rows]

    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        row_heights=[0.76, 0.24],
        vertical_spacing=0.12,
    )

    summary_rows = []
    for polarity in POLARITIES:
        actual_all = [row[polarity] for row in rows]
        actual_test = actual_all[split_index:]
        counts_all = [row[f"{polarity}_count"] for row in rows]
        predicted_test = model_results[polarity]["predicted_test"]
        errors = [abs(actual - predicted) for actual, predicted in zip(actual_test, predicted_test)]
        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum(error ** 2 for error in errors) / len(errors))

        fig.add_trace(
            go.Scatter(
                x=labels,
                y=actual_all,
                mode="lines+markers",
                name=f"{POLARITY_LABELS[polarity]} 实际值",
                line=dict(color=POLARITY_COLORS[polarity], width=3),
                marker=dict(size=7, color="white", line=dict(width=2, color=POLARITY_COLORS[polarity])),
                customdata=[
                    [counts_all[index], rows[index]["total"], "训练集" if index < split_index else "测试集"]
                    for index in range(len(rows))
                ],
                hovertemplate=(
                    "%{x}<br>%{customdata[2]} 实际值<br>"
                    + POLARITY_LABELS[polarity]
                    + ": %{y:.1f}%<br>数量: %{customdata[0]} / %{customdata[1]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=test_labels,
                y=predicted_test,
                mode="lines+markers",
                name=f"{POLARITY_LABELS[polarity]} 预测值",
                line=dict(color=POLARITY_COLORS[polarity], width=3, dash="dash"),
                marker=dict(size=8, symbol="diamond", color=POLARITY_COLORS[polarity]),
                customdata=list(zip(actual_test, errors)),
                hovertemplate=(
                    "%{x}<br>RNN 趋势预测<br>"
                    + POLARITY_LABELS[polarity]
                    + ": %{y:.1f}%<br>实际值: %{customdata[0]:.1f}%<br>"
                    + "绝对误差: %{customdata[1]:.1f}%<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        summary_rows.append(
            [
                POLARITY_LABELS[polarity],
                str(model_results[polarity]["window_size"]),
                str(model_results[polarity]["epochs"]),
                format_stat(model_results[polarity]["train_loss"], 4),
                f"{mae:.2f}%",
                f"{rmse:.2f}%",
            ]
        )

    fig.add_trace(
        go.Table(
            header=dict(
                values=["情感类别", "窗口长度", "训练轮数", "最终训练损失", "测试 MAE", "测试 RMSE"],
                fill_color="#F1F5F9",
                line_color="#D8DEE9",
                align="center",
                font=dict(size=14, color="#30333A"),
                height=34,
            ),
            cells=dict(
                values=list(zip(*summary_rows)),
                fill_color="white",
                line_color="#E5E7EB",
                align="center",
                font=dict(size=14, color="#4B5563"),
                height=32,
            ),
        ),
        row=2,
        col=1,
    )

    boundary_label = test_rows[0]["label"]
    fig.add_shape(
        type="line",
        x0=boundary_label,
        x1=boundary_label,
        y0=0,
        y1=100,
        xref="x",
        yref="y",
        line=dict(color="#30333A", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=boundary_label,
        y=97,
        xref="x",
        yref="y",
        text="测试集开始",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=13, color="#30333A"),
        bgcolor="rgba(255,255,255,0.88)",
    )
    fig.add_annotation(
        x=0.01,
        y=1.11,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text=(
            f"训练集: 前 {len(train_rows)} 个时间桶 ({train_rows[0]['label']} 至 {train_rows[-1]['label']})"
            f" · 测试集: 后 {len(test_rows)} 个时间桶 ({test_rows[0]['label']} 至 {test_rows[-1]['label']})"
        ),
        font=dict(size=14, color="#5B6270"),
    )

    fig.update_layout(
        title=dict(text="情感趋势 RNN 预测", x=0.02, y=0.98, font=dict(size=28, color="#30333A")),
        template="plotly_white",
        width=1600,
        height=840,
        margin=dict(l=70, r=50, t=120, b=55),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.67,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
        ),
        font=dict(family="Arial, Heiti SC, Microsoft YaHei, sans-serif", size=15, color="#6B6F78"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, linecolor="#8A8F99", linewidth=1.5, tickfont=dict(size=14), row=1, col=1)
    fig.update_yaxes(
        range=[0, 100],
        ticksuffix="%",
        dtick=20,
        gridcolor="#E3E8EF",
        zeroline=False,
        tickfont=dict(size=14),
        title="情感占比",
        row=1,
        col=1,
    )

    return fig


def build_future_prediction_figure(rows, future_days, future_predictions, granularity: str):
    labels = [row["label"] for row in rows]
    future_labels = [bucket_label(day, granularity) for day in future_days]
    fig = go.Figure()

    for polarity in POLARITIES:
        actual_all = [row[polarity] for row in rows]
        fig.add_trace(go.Scatter(x=labels, y=actual_all, mode="lines+markers", name=f"{POLARITY_LABELS[polarity]} 历史值", line=dict(color=POLARITY_COLORS[polarity], width=3), marker=dict(size=7, color="white", line=dict(width=2, color=POLARITY_COLORS[polarity]))))
        if future_labels:
            fig.add_trace(go.Scatter(x=[labels[-1]] + future_labels, y=[actual_all[-1]] + future_predictions[polarity], mode="lines+markers", name=f"{POLARITY_LABELS[polarity]} 未来预测", line=dict(color=POLARITY_COLORS[polarity], width=3, dash="dash"), marker=dict(size=8, symbol="diamond", color=POLARITY_COLORS[polarity])))

    if future_labels:
        fig.add_shape(type="line", x0=future_labels[0], x1=future_labels[0], y0=0, y1=100, xref="x", yref="y", line=dict(color="#30333A", width=2, dash="dot"))
        fig.add_annotation(x=future_labels[0], y=97, xref="x", yref="y", text="未来预测开始", showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=13, color="#30333A"), bgcolor="rgba(255,255,255,0.88)")

    fig.update_layout(title=dict(text="情感趋势 RNN 未来预测", x=0.02, y=0.98, font=dict(size=28, color="#30333A")), template="plotly_white", width=1600, height=680, margin=dict(l=70, r=50, t=110, b=70), hovermode="x unified", legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5, font=dict(size=14)), font=dict(family="Arial, Heiti SC, Microsoft YaHei, sans-serif", size=15, color="#6B6F78"), plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(showgrid=False, linecolor="#8A8F99", linewidth=1.5, tickfont=dict(size=14))
    fig.update_yaxes(range=[0, 100], ticksuffix="%", dtick=20, gridcolor="#E3E8EF", zeroline=False, tickfont=dict(size=14), title="情感占比")
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Predict sentiment trend with RNN models.")
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUTS,
        help="JSONL/JSON files containing attributes.sentiment.polarity and create_time.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output HTML path.",
    )
    parser.add_argument("--future-output", type=Path, default=DEFAULT_FUTURE_OUTPUT, help="Output HTML path for future forecast.")
    parser.add_argument(
        "--timezone",
        default="Asia/Shanghai",
        help="Timezone used to convert create_time to calendar dates.",
    )
    parser.add_argument(
        "--granularity",
        choices=("day", "week", "month"),
        default=DEFAULT_GRANULARITY,
        help="Time bucket used for aggregation.",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        help="Optional inclusive start date, for example 2025-01-01.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        help="Optional inclusive end date, for example 2026-04-30.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW,
        help="Look-back window size for each RNN sequence.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Training epochs for each polarity model.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="Hidden size of the RNN layer.",
    )
    parser.add_argument("--forecast-end-date", type=date.fromisoformat, default=DEFAULT_FORECAST_END_DATE, help="Inclusive end date for future forecast HTML.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    timezone = ZoneInfo(args.timezone)
    creation_time = datetime.now(timezone)
    daily_counts, source_counts, skipped = aggregate_daily_counts(args.input, timezone)
    daily_counts = filter_daily_counts(daily_counts, args.start_date, args.end_date)

    if not daily_counts:
        raise ValueError("No usable records found. Check create_time and attributes.sentiment.polarity.")

    grouped_counts = aggregate_by_granularity(daily_counts, args.granularity)
    rows = build_rows(grouped_counts, args.granularity)
    split_index, train_rows, test_rows = split_rows(rows)

    model_results = {}
    predictions_by_polarity = {}
    for polarity in POLARITIES:
        full_series = [row[polarity] for row in rows]
        train_series = full_series[:split_index]
        result = fit_rnn_series(
            train_series=train_series,
            full_series=full_series,
            test_length=len(test_rows),
            window_size=args.window_size,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
        )
        model_results[polarity] = result
        predictions_by_polarity[polarity] = result["predicted_test"]

    fig = build_prediction_figure(rows, train_rows, test_rows, model_results)
    accuracy, macro_f1, macro_precision = compute_test_proportion_metrics(test_rows, predictions_by_polarity)
    future_days = future_bucket_days(rows[-1]["day"], args.forecast_end_date, args.granularity)
    future_predictions = {}
    for polarity in POLARITIES:
        future_predictions[polarity] = forecast_rnn_series([row[polarity] for row in rows], len(future_days), args.window_size, args.epochs, args.hidden_size)["predicted_future"]
    future_fig = build_future_prediction_figure(rows, future_days, future_predictions, args.granularity)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    args.future_output.parent.mkdir(parents=True, exist_ok=True)
    future_fig.write_html(args.future_output, include_plotlyjs=True, full_html=True)

    total_used = sum(sum(counts.values()) for counts in grouped_counts.values())
    training_duration = f"{train_rows[0]['label']} to {train_rows[-1]['label']}"
    print(f"output: {args.output}")
    print(f"future output: {args.future_output}")
    print(f"Creation time: {creation_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"date range: {min(daily_counts)} to {max(daily_counts)}")
    print(f"granularity: {args.granularity}")
    print(f"buckets: {len(rows)}")
    print(f"train buckets: {len(train_rows)}")
    print(f"test buckets: {len(test_rows)}")
    print(f"Training duration: {training_duration}")
    print(f"Accuracy rate: {accuracy:.4f}")
    print(f"F1 points: {macro_f1:.4f}")
    print(f"Precision: {macro_precision:.4f}")
    print(f"records used: {total_used}")
    print(f"source counts: {dict(source_counts)}")
    if skipped:
        print(f"skipped: {dict(skipped)}")


if __name__ == "__main__":
    main()
