import argparse
import json
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUTS = [
    BASE_DIR / "weibo/search_comments_2026-04-17.jsonl",
    BASE_DIR / "xhs/jsonl/search_comments_2026-04-06.jsonl",
    BASE_DIR / "douyin/douyin_naixue_simulated_comments.jsonl",
    BASE_DIR / "dianping/dianping_naixue_simulated_comments.jsonl",
]
DEFAULT_OUTPUT = BASE_DIR / "sentiment_trend.html"
DEFAULT_GRANULARITY = "month"
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
    """Read normal JSONL, JSON arrays, or pretty-printed JSON objects joined together."""
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

    # XHS records in this project use milliseconds; Weibo records use seconds.
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
            date = record_date(record, timezone)
            polarity = record_polarity(record)

            if date is None:
                skipped["missing_time"] += 1
                continue
            if polarity is None:
                skipped["missing_polarity"] += 1
                continue

            daily_counts[date][polarity] += 1
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


def build_figure(grouped_counts, granularity: str):
    dates = sorted(grouped_counts)
    totals = {day: sum(grouped_counts[day].values()) for day in dates}
    labels = [bucket_label(day, granularity) for day in dates]

    fig = go.Figure()
    fill_modes = {
        "positive": "tozeroy",
        "neutral": None,
        "negative": "tozeroy",
    }

    for polarity in POLARITIES:
        counts = [grouped_counts[day][polarity] for day in dates]
        percentages = [
            (grouped_counts[day][polarity] / totals[day] * 100) if totals[day] else 0
            for day in dates
        ]

        fig.add_trace(
            go.Scatter(
                x=labels,
                y=percentages,
                mode="lines+markers",
                name=POLARITY_LABELS[polarity],
                line=dict(color=POLARITY_COLORS[polarity], width=3, shape="spline"),
                marker=dict(size=7, color="white", line=dict(width=3, color=POLARITY_COLORS[polarity])),
                fill=fill_modes[polarity],
                fillcolor=hex_to_rgba(POLARITY_COLORS[polarity], 0.12),
                customdata=list(zip(counts, [totals[day] for day in dates])),
                hovertemplate=(
                    "%{x}<br>"
                    + POLARITY_LABELS[polarity]
                    + ": %{y:.1f}%<br>"
                    + "数量: %{customdata[0]} / %{customdata[1]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text="情感趋势变化", x=0.02, y=0.97, font=dict(size=28, color="#30333A")),
        template="plotly_white",
        width=1600,
        height=620,
        margin=dict(l=70, r=50, t=90, b=90),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            xanchor="center",
            x=0.5,
            font=dict(size=16),
        ),
        font=dict(family="Arial, Heiti SC, Microsoft YaHei, sans-serif", size=15, color="#6B6F78"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#8A8F99",
        linewidth=1.5,
        tickfont=dict(size=16),
    )
    fig.update_yaxes(
        range=[0, 100],
        ticksuffix="%",
        dtick=20,
        gridcolor="#E3E8EF",
        zeroline=False,
        tickfont=dict(size=16),
    )

    return fig

def hex_to_rgba(hex_color: str, alpha: float):
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot sentiment polarity trends with Plotly.")
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
        "--timezone",
        default="Asia/Shanghai",
        help="Timezone used to convert create_time to calendar dates.",
    )
    parser.add_argument(
        "--granularity",
        choices=("day", "week", "month"),
        default=DEFAULT_GRANULARITY,
        help="Time bucket used for aggregation. Default is month for full-range trend charts.",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        help="Optional inclusive start date, for example 2026-03-23.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        help="Optional inclusive end date, for example 2026-04-21.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    timezone = ZoneInfo(args.timezone)
    daily_counts, source_counts, skipped = aggregate_daily_counts(args.input, timezone)
    daily_counts = filter_daily_counts(daily_counts, args.start_date, args.end_date)

    if not daily_counts:
        raise ValueError("No usable records found. Check create_time and attributes.sentiment.polarity.")

    grouped_counts = aggregate_by_granularity(daily_counts, args.granularity)
    fig = build_figure(grouped_counts, args.granularity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)

    total_used = sum(sum(counts.values()) for counts in grouped_counts.values())
    print(f"output: {args.output}")
    print(f"date range: {min(daily_counts)} to {max(daily_counts)}")
    print(f"granularity: {args.granularity}")
    print(f"buckets: {len(grouped_counts)}")
    print(f"records used: {total_used}")
    print(f"source counts: {dict(source_counts)}")
    if skipped:
        print(f"skipped: {dict(skipped)}")


if __name__ == "__main__":
    main()
