import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
from scipy.stats import f as f_distribution


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUTS = [
    PROJECT_DIR / "weibo/search_comments_2026-04-17.jsonl",
    PROJECT_DIR / "xhs/jsonl/search_comments_2026-04-06.jsonl",
    PROJECT_DIR / "douyin/douyin_naixue_simulated_comments.jsonl",
    PROJECT_DIR / "dianping/dianping_naixue_simulated_comments.jsonl",
]
DEFAULT_OUTPUT = SCRIPT_DIR / "sentiment_trend_anova_prediction.html"
DEFAULT_FUTURE_OUTPUT = SCRIPT_DIR / "sentiment_trend_anova_prediction_future.html"
DEFAULT_GRANULARITY = "month"
TRAIN_RATIO = 0.8
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


def fit_linear_trend_anova(x_train, y_train, x_all):
    train_count = len(x_train)
    mean_x = sum(x_train) / train_count
    mean_y = sum(y_train) / train_count
    sxx = sum((value - mean_x) ** 2 for value in x_train)
    sxy = sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in zip(x_train, y_train))

    if train_count < 3 or sxx == 0:
        predictions = [mean_y for _ in x_all]
        return {
            "intercept": mean_y,
            "slope": 0.0,
            "predictions": predictions,
            "f_stat": None,
            "p_value": None,
            "r_squared": None,
        }

    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    fitted_train = [intercept + slope * value for value in x_train]
    predictions = [intercept + slope * value for value in x_all]

    ss_model = sum((value - mean_y) ** 2 for value in fitted_train)
    ss_error = sum((actual - fitted) ** 2 for actual, fitted in zip(y_train, fitted_train))
    total_ss = ss_model + ss_error
    r_squared = ss_model / total_ss if total_ss else 0.0

    if ss_error == 0:
        f_stat = math.inf if ss_model > 0 else 0.0
        p_value = 0.0 if math.isinf(f_stat) else 1.0
    else:
        df_model = 1
        df_error = train_count - 2
        ms_model = ss_model / df_model
        ms_error = ss_error / df_error
        f_stat = ms_model / ms_error
        p_value = f_distribution.sf(f_stat, df_model, df_error)

    return {
        "intercept": intercept,
        "slope": slope,
        "predictions": predictions,
        "f_stat": f_stat,
        "p_value": p_value,
        "r_squared": r_squared,
    }


def clamp_percentage(value: float):
    return min(100.0, max(0.0, value))


def format_stat(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isinf(value):
        return "Inf"
    return f"{value:.{digits}f}"


def compute_test_proportion_metrics(rows, test_rows, split_index):
    predictions_by_polarity = {}
    x_all = [row["index"] for row in rows]
    x_train = [row["index"] for row in rows[:split_index]]

    for polarity in POLARITIES:
        model = fit_linear_trend_anova(x_train, [row[polarity] for row in rows[:split_index]], x_all)
        predictions_by_polarity[polarity] = [clamp_percentage(value) for value in model["predictions"][split_index:]]

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


def compute_future_predictions(rows, granularity: str, forecast_end_date: date):
    future_days = future_bucket_days(rows[-1]["day"], forecast_end_date, granularity)
    if not future_days:
        return [], {}

    x_all = [row["index"] for row in rows]
    x_future = list(range(len(rows), len(rows) + len(future_days)))
    predictions = {}
    for polarity in POLARITIES:
        model = fit_linear_trend_anova(x_all, [row[polarity] for row in rows], x_all + x_future)
        predictions[polarity] = [clamp_percentage(value) for value in model["predictions"][-len(future_days):]]
    return future_days, predictions


def build_prediction_figure(rows, train_rows, test_rows, split_index):
    labels = [row["label"] for row in rows]
    x_all = [row["index"] for row in rows]
    x_train = [row["index"] for row in train_rows]
    test_labels = [row["label"] for row in test_rows]

    fig = go.Figure()
    for polarity in POLARITIES:
        actual_all = [row[polarity] for row in rows]
        actual_test = actual_all[split_index:]
        counts_all = [row[f"{polarity}_count"] for row in rows]

        model = fit_linear_trend_anova(x_train, [row[polarity] for row in train_rows], x_all)
        predicted_all = [clamp_percentage(value) for value in model["predictions"]]
        predicted_test = predicted_all[split_index:]
        errors = [abs(actual - predicted) for actual, predicted in zip(actual_test, predicted_test)]
        mae = sum(errors) / len(errors)

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
            )
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
                    "%{x}<br>ANOVA 趋势预测<br>"
                    + POLARITY_LABELS[polarity]
                    + ": %{y:.1f}%<br>实际值: %{customdata[0]:.1f}%<br>"
                    + "绝对误差: %{customdata[1]:.1f}%<extra></extra>"
                ),
            )
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
        title=dict(text="情感趋势 ANOVA 预测", x=0.02, y=0.98, font=dict(size=28, color="#30333A")),
        template="plotly_white",
        width=1600,
        height=680,
        margin=dict(l=70, r=50, t=120, b=70),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
        ),
        font=dict(family="Arial, Heiti SC, Microsoft YaHei, sans-serif", size=15, color="#6B6F78"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, linecolor="#8A8F99", linewidth=1.5, tickfont=dict(size=14))
    fig.update_yaxes(
        range=[0, 100],
        ticksuffix="%",
        dtick=20,
        gridcolor="#E3E8EF",
        zeroline=False,
        tickfont=dict(size=14),
        title="情感占比",
    )

    return fig


def build_future_prediction_figure(rows, future_days, future_predictions, granularity: str):
    labels = [row["label"] for row in rows]
    future_labels = [bucket_label(day, granularity) for day in future_days]
    fig = go.Figure()

    for polarity in POLARITIES:
        actual_all = [row[polarity] for row in rows]
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=actual_all,
                mode="lines+markers",
                name=f"{POLARITY_LABELS[polarity]} 历史值",
                line=dict(color=POLARITY_COLORS[polarity], width=3),
                marker=dict(size=7, color="white", line=dict(width=2, color=POLARITY_COLORS[polarity])),
            )
        )
        if future_labels:
            fig.add_trace(
                go.Scatter(
                    x=[labels[-1]] + future_labels,
                    y=[actual_all[-1]] + future_predictions[polarity],
                    mode="lines+markers",
                    name=f"{POLARITY_LABELS[polarity]} 未来预测",
                    line=dict(color=POLARITY_COLORS[polarity], width=3, dash="dash"),
                    marker=dict(size=8, symbol="diamond", color=POLARITY_COLORS[polarity]),
                )
            )

    if future_labels:
        fig.add_shape(
            type="line",
            x0=future_labels[0],
            x1=future_labels[0],
            y0=0,
            y1=100,
            xref="x",
            yref="y",
            line=dict(color="#30333A", width=2, dash="dot"),
        )
        fig.add_annotation(
            x=future_labels[0],
            y=97,
            xref="x",
            yref="y",
            text="未来预测开始",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=13, color="#30333A"),
            bgcolor="rgba(255,255,255,0.88)",
        )

    fig.update_layout(
        title=dict(text="情感趋势 ANOVA 未来预测", x=0.02, y=0.98, font=dict(size=28, color="#30333A")),
        template="plotly_white",
        width=1600,
        height=680,
        margin=dict(l=70, r=50, t=110, b=70),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5, font=dict(size=14)),
        font=dict(family="Arial, Heiti SC, Microsoft YaHei, sans-serif", size=15, color="#6B6F78"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, linecolor="#8A8F99", linewidth=1.5, tickfont=dict(size=14))
    fig.update_yaxes(range=[0, 100], ticksuffix="%", dtick=20, gridcolor="#E3E8EF", zeroline=False, tickfont=dict(size=14), title="情感占比")
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Predict sentiment trend with ANOVA-tested linear trend lines.")
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
    parser.add_argument(
        "--future-output",
        type=Path,
        default=DEFAULT_FUTURE_OUTPUT,
        help="Output HTML path for future forecast.",
    )
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
        "--forecast-end-date",
        type=date.fromisoformat,
        default=DEFAULT_FORECAST_END_DATE,
        help="Inclusive end date for future forecast HTML.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    creation_time = datetime.now(ZoneInfo(args.timezone))
    timezone = ZoneInfo(args.timezone)
    daily_counts, source_counts, skipped = aggregate_daily_counts(args.input, timezone)
    daily_counts = filter_daily_counts(daily_counts, args.start_date, args.end_date)

    if not daily_counts:
        raise ValueError("No usable records found. Check create_time and attributes.sentiment.polarity.")

    grouped_counts = aggregate_by_granularity(daily_counts, args.granularity)
    rows = build_rows(grouped_counts, args.granularity)
    split_index, train_rows, test_rows = split_rows(rows)
    fig = build_prediction_figure(rows, train_rows, test_rows, split_index)
    accuracy, macro_f1, macro_precision = compute_test_proportion_metrics(rows, test_rows, split_index)
    future_days, future_predictions = compute_future_predictions(rows, args.granularity, args.forecast_end_date)
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
