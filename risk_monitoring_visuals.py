import base64
import html
import json
import re
from collections import Counter
from io import BytesIO
from pathlib import Path

import plotly.graph_objects as go

try:
    import jieba
except ImportError:
    jieba = None

try:
    from wordcloud import WordCloud
except ImportError as exc:
    WordCloud = None
    WORDCLOUD_IMPORT_ERROR = exc
else:
    WORDCLOUD_IMPORT_ERROR = None


BASE_DIR = Path("/Users/fujunhan/Desktop/MIS2011/GroupProject")
OUTPUT_DIR = BASE_DIR / "risk_monitoring_visuals"

INPUT_FILES = {
    "weibo": BASE_DIR / "weibo/search_comments_2026-04-17.jsonl",
    "rednote": BASE_DIR / "xhs/jsonl/search_comments_2026-04-06.jsonl",
    "douyin": BASE_DIR / "douyin/douyin_naixue_simulated_comments.jsonl",
    "dianping": BASE_DIR / "dianping/dianping_naixue_simulated_comments.jsonl",
}

OUTPUT_WORD_CLOUD = OUTPUT_DIR / "word_cloud.html"
OUTPUT_EMOTION_PIE = OUTPUT_DIR / "emotional_subdivision_distribution.html"
OUTPUT_SOURCE_BAR = OUTPUT_DIR / "data_source_distribution.html"
SQUARE_SIZE = 390

TONE_LABELS = {
    "平淡语气": "Plain",
    "关切语调": "Concern",
    "开心语调": "Happy",
    "愤怒语调": "Angry",
    "悲伤语调": "Sad",
    "疑问语调": "Questioning",
    "惊奇语调": "Surprise",
    "厌恶语调": "Aversive",
}

SOURCE_LABELS = {
    "rednote": "Rednote",
    "douyin": "Douyin",
    "weibo": "Weibo",
    "dianping": "Dianping",
}

SOURCE_COLORS = {
    "rednote": "#f54e77",
    "douyin": "#111827",
    "weibo": "#ff8a3d",
    "dianping": "#16a34a",
}

TONE_COLORS = {
    "Plain": "#8b95a7",
    "Concern": "#4f7cff",
    "Happy": "#48b878",
    "Angry": "#e34d4d",
    "Sad": "#6c7aa8",
    "Questioning": "#a46be8",
    "Surprise": "#f4a62a",
    "Aversive": "#b25139",
}

STOPWORDS = {
    "奈雪", "新品", "评论", "用户", "微博", "ID", "这个", "那个", "感觉", "真的",
    "一下", "还是", "就是", "可以", "不是", "没有", "有点", "比较", "一个",
    "今天", "现在", "但是", "因为", "所以", "哈哈", "哈哈哈", "已经", "这么",
    "什么", "怎么", "自己", "大家", "起来", "出来", "一下子", "一杯",
}


def iter_json_records(file_path: Path):
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{file_path} is not a JSON array")
        yield from data
        return

    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        record, index = decoder.raw_decode(text, index)
        if isinstance(record, dict):
            yield record


def clean_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<img[^>]*alt=\"([^\"]+)\"[^>]*>", r" \1 ", text)
    text = re.sub(r"</?(a|span)[^>]*>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def record_text(record: dict) -> str:
    attributes = record.get("attributes") or {}
    return clean_text(attributes.get("analyzed_text") or record.get("content") or "")


def record_tone(record: dict) -> str:
    emotion = (record.get("attributes") or {}).get("emotion_fine_grained") or {}
    raw_label = emotion.get("label") or emotion.get("raw_label") or "Unknown"
    return TONE_LABELS.get(raw_label, raw_label)


def load_records() -> list[dict]:
    rows = []
    for source, file_path in INPUT_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        for record in iter_json_records(file_path):
            rows.append({"source": source, "record": record})
    return rows


def tokenize_for_word_cloud(text: str) -> list[str]:
    if jieba:
        tokens = jieba.lcut(text)
    else:
        tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)

    cleaned = []
    for token in tokens:
        token = token.strip()
        if len(token) < 2:
            continue
        if token in STOPWORDS:
            continue
        if re.fullmatch(r"[\W_]+", token):
            continue
        cleaned.append(token)
    return cleaned


def find_chinese_font() -> str | None:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def write_word_cloud_html(rows: list[dict]):
    if WordCloud is None:
        raise RuntimeError(
            "wordcloud is not installed. Install it with: pip install wordcloud jieba"
        ) from WORDCLOUD_IMPORT_ERROR

    all_text = " ".join(record_text(row["record"]) for row in rows)
    token_counts = Counter(tokenize_for_word_cloud(all_text))
    if not token_counts:
        raise ValueError("No usable tokens found for word cloud.")

    word_cloud = WordCloud(
        width=900,
        height=900,
        background_color="white",
        font_path=find_chinese_font(),
        max_words=180,
        prefer_horizontal=0.9,
        colormap="viridis",
        contour_width=2,
        contour_color="#111827",
    ).generate_from_frequencies(token_counts)

    image_buffer = BytesIO()
    word_cloud.to_image().save(image_buffer, format="PNG")
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode("ascii")

    OUTPUT_WORD_CLOUD.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Word Cloud</title>
    <style>
        body {{
            margin: 0;
            min-height: 100vh;
            display: grid;
            place-items: center;
            background: #ffffff;
            font-family: Arial, "Noto Sans SC", sans-serif;
        }}
        .panel {{
            width: min({SQUARE_SIZE}px, calc(100vw - 24px));
            height: min({SQUARE_SIZE}px, calc(100vw - 24px));
            border: 0;
            border-radius: 0;
            padding: 0;
            background: #fff;
            overflow: hidden;
        }}
        h1 {{
            display: none;
        }}
        img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }}
    </style>
</head>
<body>
    <main class="panel">
        <h1>Word cloud</h1>
        <img src="data:image/png;base64,{image_base64}" alt="Comment word cloud">
    </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_emotion_pie_html(rows: list[dict]):
    counts = Counter(record_tone(row["record"]) for row in rows)
    labels = list(TONE_COLORS)
    values = [counts.get(label, 0) for label in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.38,
                sort=False,
                marker=dict(colors=[TONE_COLORS[label] for label in labels]),
                textinfo="label+percent",
                hovertemplate="%{label}<br>Records: %{value}<br>%{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=None,
        width=SQUARE_SIZE,
        height=SQUARE_SIZE,
        margin=dict(l=16, r=16, t=18, b=18),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family='Arial, "Noto Sans SC", sans-serif', size=14, color="#111827"),
        showlegend=False,
    )
    fig.write_html(OUTPUT_EMOTION_PIE, include_plotlyjs=True, full_html=True)


def write_source_bar_html(rows: list[dict]):
    counts = Counter(row["source"] for row in rows)
    sources = ["rednote", "douyin", "weibo", "dianping"]
    labels = [SOURCE_LABELS[source] for source in sources]
    values = [counts.get(source, 0) for source in sources]
    x_axis_max = max(values) * 1.32 if values else 1

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(color=[SOURCE_COLORS[source] for source in sources]),
                text=[f"{value:,}" for value in values],
                textposition="outside",
                hovertemplate="%{y}<br>Records: %{x:,}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=None,
        width=SQUARE_SIZE,
        height=SQUARE_SIZE,
        margin=dict(l=88, r=70, t=28, b=52),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family='Arial, "Noto Sans SC", sans-serif', size=14, color="#111827"),
        xaxis=dict(title="Comment records", range=[0, x_axis_max], showgrid=True, gridcolor="#e5e7eb"),
        yaxis=dict(title="", autorange="reversed"),
    )
    fig.write_html(OUTPUT_SOURCE_BAR, include_plotlyjs=True, full_html=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_records()

    write_word_cloud_html(rows)
    write_emotion_pie_html(rows)
    write_source_bar_html(rows)

    print(f"records: {len(rows):,}")
    print(f"word cloud: {OUTPUT_WORD_CLOUD}")
    print(f"emotion pie: {OUTPUT_EMOTION_PIE}")
    print(f"source bar: {OUTPUT_SOURCE_BAR}")


if __name__ == "__main__":
    main()
