#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from generate_summary_report_qwen import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_client,
    call_qwen,
    build_context_payload,
    collect_prediction_forecasts,
    collect_prediction_metrics,
    collect_visual_summaries,
    sanitize_month_label,
)


PROJECT_DIR = Path(__file__).resolve().parent


def build_monthly_prompt(context: dict) -> str:
    return (
        "你是企业舆情治理与服务改进分析师。请根据下面提供的结构化数据，围绕“Recurring Problems and Improvement Suggestions”生成一份中文月报。"
        "标题必须包含英文主题“Recurring Problems and Improvement Suggestions”。\n"
        "要求：\n"
        "1. 重点看负面消息，不要平均用力。\n"
        "2. 特别关注 negative 情感占比，以及情绪细分类中的 Angry、Aversive、Sad、Concern、Questioning。\n"
        "3. 报告聚焦最近一个月，长度不少于500字。\n"
        "4. 内容必须包括：最近一个月的负面风险概况、反复出现的问题类型、可能成因、跨平台表现差异、未来几个月负面走势判断、可执行的改进建议。\n"
        "5. 要引用具体数值，不能空泛，例如负面占比、模型预测值、source counts、细分类数量。\n"
        "6. 如果预测模型存在差异，要指出分歧并给出保守判断。\n"
        "7. 不要写成宣传稿，要像管理层问题诊断报告。\n\n"
        f"数据上下文：\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def build_all_time_prompt(context: dict) -> str:
    return (
        "你是企业舆情治理与服务改进分析师。请根据下面提供的结构化数据，围绕“Recurring Problems and Improvement Suggestions”生成一份中文全时期简报。"
        "标题必须包含英文主题“Recurring Problems and Improvement Suggestions”。\n"
        "要求：\n"
        "1. 重点识别所有时间范围内反复出现的负面问题，而不是做一般性总结。\n"
        "2. 重点关注 negative 情感占比，以及 Angry、Aversive、Sad、Concern、Questioning 相关细分类信号。\n"
        "3. 长度不少于500字。\n"
        "4. 必须覆盖：长期反复问题、负面问题的严重程度排序、数据源差异、趋势变化、预测到2026-09的风险判断、改进优先级建议。\n"
        "5. 要引用具体数值和模型指标，例如 accuracy、future negative ratio、records used、source counts、情绪细分类数量。\n"
        "6. 最后给出3到6条按优先级排列的改进建议，并说明理由。\n"
        "7. 输出中文，风格正式、直接、面向管理层。\n\n"
        f"数据上下文：\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def write_report(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate recurring problems and improvement reports with Qwen.")
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
    context_path = args.output_dir / "recurring_problems_context.json"
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(context, ensure_ascii=False, indent=2))
        print(f"context saved: {context_path}")
        return

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in environment variable: {args.api_key_env}")

    client = build_client(args.base_url, api_key)
    monthly_report = call_qwen(client, args.model, build_monthly_prompt(context))
    all_time_report = call_qwen(client, args.model, build_all_time_prompt(context))

    month_label = sanitize_month_label(context["latest_month_bucket"])
    monthly_path = args.output_dir / f"recurring_problems_monthly_{month_label}.md"
    all_time_path = args.output_dir / "recurring_problems_all_time.md"

    write_report(monthly_path, "Recurring Problems and Improvement Suggestions", monthly_report)
    write_report(all_time_path, "Recurring Problems and Improvement Suggestions", all_time_report)

    print(f"context saved: {context_path}")
    print(f"monthly report: {monthly_path}")
    print(f"all-time report: {all_time_path}")
    print(f"model used: {args.model}")


if __name__ == "__main__":
    main()
