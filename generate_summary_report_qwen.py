#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_DIR = Path(__file__).resolve().parent
PREDICTION_DIR = PROJECT_DIR / "sentiment_trend_prediction"
RISK_DIR = PROJECT_DIR / "risk_monitoring_visuals"
SENTIMENT_TREND_HTML = PROJECT_DIR / "sentiment_trend.html"

DEFAULT_MODEL = "qwen-turbo"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODEL_SCRIPTS = {
    "anova": PREDICTION_DIR / "sentiment_trend_anova_prediction.py",
    "lstm": PREDICTION_DIR / "sentiment_trend_lstm_prediction.py",
    "rnn": PREDICTION_DIR / "sentiment_trend_rnn_prediction.py",
    "gradientboosting": PREDICTION_DIR / "sentiment_trend_grandientboosting_prediction.py",
}

FUTURE_HTMLS = {
    "anova": PREDICTION_DIR / "sentiment_trend_anova_prediction_future.html",
    "lstm": PREDICTION_DIR / "sentiment_trend_lstm_prediction_future.html",
    "rnn": PREDICTION_DIR / "sentiment_trend_rnn_prediction_future.html",
    "gradientboosting": PREDICTION_DIR / "sentiment_trend_grandientboosting_prediction_future.html",
}


@dataclass
class PlotlyFigure:
    data: list[dict[str, Any]]
    layout: dict[str, Any]


def extract_plotly_figure(html_text: str) -> PlotlyFigure:
    marker = "Plotly.newPlot("
    start = html_text.rfind(marker)
    if start == -1:
        raise ValueError("Plotly.newPlot not found")

    pos = start + len(marker)
    decoder = json.JSONDecoder()

    pos = skip_ws(html_text, pos)
    if html_text[pos] == '"':
        _, consumed = decoder.raw_decode(html_text[pos:])
        pos += consumed
    else:
        raise ValueError("Unexpected Plotly.newPlot id format")

    pos = skip_delimiter(html_text, pos)
    data, consumed = decoder.raw_decode(html_text[pos:])
    pos += consumed

    pos = skip_delimiter(html_text, pos)
    layout, consumed = decoder.raw_decode(html_text[pos:])
    return PlotlyFigure(data=data, layout=layout)


def skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def skip_delimiter(text: str, pos: int) -> int:
    pos = skip_ws(text, pos)
    if text[pos] != ",":
        raise ValueError(f"Expected comma at position {pos}")
    pos += 1
    return skip_ws(text, pos)


def load_plotly_figure(path: Path) -> PlotlyFigure:
    return extract_plotly_figure(path.read_text(encoding="utf-8"))


def find_trace(figure: PlotlyFigure, keywords: list[str]) -> dict[str, Any] | None:
    lowered = [word.lower() for word in keywords]
    for trace in figure.data:
        name = str(trace.get("name", "")).lower()
        if any(word in name for word in lowered):
            return trace
    return None


def last_point(trace: dict[str, Any]) -> dict[str, Any]:
    xs = trace.get("x", [])
    ys = trace.get("y", [])
    if not xs or not ys:
        raise ValueError("Trace has no points")
    return {"x": xs[-1], "y": ys[-1]}


def summarize_sentiment_trend(path: Path) -> dict[str, Any]:
    fig = load_plotly_figure(path)
    positive = last_point(find_trace(fig, ["正向", "positive"]))
    neutral = last_point(find_trace(fig, ["中性", "neutral"]))
    negative = last_point(find_trace(fig, ["负向", "negative"]))
    return {
        "source_file": path.name,
        "latest_bucket": positive["x"],
        "latest_sentiment": {
            "positive": round(float(positive["y"]), 2),
            "neutral": round(float(neutral["y"]), 2),
            "negative": round(float(negative["y"]), 2),
        },
    }


def summarize_distribution_chart(path: Path) -> dict[str, Any]:
    fig = load_plotly_figure(path)
    summaries: list[dict[str, Any]] = []
    for trace in fig.data:
        if trace.get("labels") is not None and trace.get("values") is not None:
            labels = trace.get("labels") or []
            values = trace.get("values") or []
        elif trace.get("orientation") == "h":
            labels = trace.get("y") or []
            values = trace.get("x") or []
        else:
            labels = trace.get("x") or []
            values = trace.get("y") or []
        pairs = []
        for label, value in zip(labels, values):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            pairs.append({"label": str(label), "value": round(number, 2)})
        pairs.sort(key=lambda item: item["value"], reverse=True)
        summaries.append(
            {
                "name": trace.get("name") or path.stem,
                "top_items": pairs[:6],
            }
        )
    return {"source_file": path.name, "traces": summaries}


def summarize_prediction_chart(path: Path) -> dict[str, Any]:
    fig = load_plotly_figure(path)
    summary = {"source_file": path.name, "forecast_end": None, "future_sentiment": {}}
    mapping = {
        "positive": ["正向", "positive"],
        "neutral": ["中性", "neutral"],
        "negative": ["负向", "negative"],
    }
    for key, base_words in mapping.items():
        trace = None
        for candidate in fig.data:
            name = str(candidate.get("name", "")).lower()
            if "未来" in name or "future" in name or "forecast" in name:
                if any(word in name for word in [w.lower() for w in base_words]):
                    trace = candidate
                    break
        if trace is None:
            continue
        point = last_point(trace)
        summary["forecast_end"] = point["x"]
        summary["future_sentiment"][key] = round(float(point["y"]), 2)
    return summary


def run_prediction_script(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["python3", str(path)],
        cwd=str(PROJECT_DIR),
        text=True,
        capture_output=True,
        check=True,
    )
    metrics: dict[str, Any] = {"script": path.name}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def collect_prediction_metrics() -> dict[str, dict[str, Any]]:
    return {name: run_prediction_script(path) for name, path in MODEL_SCRIPTS.items()}


def collect_prediction_forecasts() -> dict[str, dict[str, Any]]:
    return {name: summarize_prediction_chart(path) for name, path in FUTURE_HTMLS.items()}


def collect_visual_summaries() -> dict[str, Any]:
    return {
        "sentiment_trend": summarize_sentiment_trend(SENTIMENT_TREND_HTML),
        "risk_monitoring": {
            "data_source_distribution": summarize_distribution_chart(RISK_DIR / "data_source_distribution.html"),
            "emotional_subdivision_distribution": summarize_distribution_chart(RISK_DIR / "emotional_subdivision_distribution.html"),
            "emotional_trend_day": summarize_sentiment_trend(RISK_DIR / "emotional_trend_day.html"),
            "emotional_trend_week": summarize_sentiment_trend(RISK_DIR / "emotional_trend_week.html"),
            "emotional_trend_month": summarize_sentiment_trend(RISK_DIR / "emotional_trend_month.html"),
        },
    }


def choose_best_model(metrics: dict[str, dict[str, Any]]) -> str:
    scored = []
    for model_name, model_metrics in metrics.items():
        try:
            score = float(model_metrics.get("Accuracy rate", "0"))
        except ValueError:
            score = 0.0
        scored.append((score, model_name))
    scored.sort(reverse=True)
    return scored[0][1]


def build_context_payload(metrics: dict[str, dict[str, Any]], forecasts: dict[str, dict[str, Any]], visuals: dict[str, Any]) -> dict[str, Any]:
    best_model = choose_best_model(metrics)
    latest_month = visuals["risk_monitoring"]["emotional_trend_month"]["latest_bucket"]
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model_by_accuracy": best_model,
        "latest_month_bucket": latest_month,
        "prediction_metrics": metrics,
        "future_prediction_endpoints": forecasts,
        "visual_summaries": visuals,
    }


def build_monthly_prompt(context: dict[str, Any]) -> str:
    return (
        "你是企业舆情与情感风险分析师。请根据下面提供的结构化数据，生成一份中文月报，标题必须包含“Summary Report”。\n"
        "要求：\n"
        "1. 只输出中文正文，不要解释模型调用过程。\n"
        "2. 总长度不少于500字。\n"
        "3. 面向业务汇报场景，语言正式、简洁、可执行。\n"
        "4. 报告必须覆盖最近一个月的情感表现、风险变化、四个预测模型的对比、最佳模型结论、未来走势判断、运营建议。\n"
        "5. 明确引用数据中的比例、趋势方向、模型准确率，不要泛泛而谈。\n"
        "6. 如果不同图表结论存在差异，要指出差异并给出谨慎解释。\n\n"
        f"数据上下文：\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def build_all_time_prompt(context: dict[str, Any]) -> str:
    return (
        "你是企业舆情与情感风险分析师。请根据下面提供的结构化数据，生成一份中文简报，标题必须包含“Summary Report”。\n"
        "要求：\n"
        "1. 只输出中文正文，不要解释模型调用过程。\n"
        "2. 总长度不少于500字。\n"
        "3. 覆盖所有时间数据的整体情感格局、阶段性波动、主要数据源贡献、风险特征、模型预测一致性与分歧、未来到预测终点的走向。\n"
        "4. 必须引用关键数值，例如情感比例、source counts、accuracy、预测终点占比。\n"
        "5. 最后给出管理层可直接采纳的3到5条行动建议。\n"
        "6. 不要写成营销文案，要像分析报告。\n\n"
        f"数据上下文：\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def call_qwen(client: OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你输出的是正式中文分析报告，必须以数据为依据，不夸张，不虚构，不使用项目符号堆砌空话。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def write_report(path: Path, title: str, body: str) -> None:
    content = f"# {title}\n\n{body}\n"
    path.write_text(content, encoding="utf-8")


def sanitize_month_label(label: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Summary Report with Qwen from local prediction and visualization outputs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Qwen model name, default: qwen-turbo")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible endpoint")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY", help="Environment variable containing the API key")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "summary_reports")
    parser.add_argument("--dry-run", action="store_true", help="Only collect and print context, do not call the model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = collect_prediction_metrics()
    forecasts = collect_prediction_forecasts()
    visuals = collect_visual_summaries()
    context = build_context_payload(metrics, forecasts, visuals)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    context_path = args.output_dir / "summary_report_context.json"
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(context, ensure_ascii=False, indent=2))
        print(f"context saved: {context_path}")
        return

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in environment variable: {args.api_key_env}")

    client = build_client(args.base_url, api_key)
    monthly_prompt = build_monthly_prompt(context)
    all_time_prompt = build_all_time_prompt(context)

    monthly_report = call_qwen(client, args.model, monthly_prompt)
    all_time_report = call_qwen(client, args.model, all_time_prompt)

    month_label = sanitize_month_label(context["latest_month_bucket"])
    monthly_path = args.output_dir / f"summary_report_monthly_{month_label}.md"
    all_time_path = args.output_dir / "summary_report_all_time.md"

    write_report(monthly_path, "Summary Report", monthly_report)
    write_report(all_time_path, "Summary Report", all_time_report)

    print(f"context saved: {context_path}")
    print(f"monthly report: {monthly_path}")
    print(f"all-time report: {all_time_path}")
    print(f"model used: {args.model}")


if __name__ == "__main__":
    main()
