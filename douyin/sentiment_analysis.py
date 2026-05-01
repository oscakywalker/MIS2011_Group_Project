import html
import json
import re
from pathlib import Path
import os

import torch
from tqdm import tqdm
from transformers import pipeline

BASE_DIR = Path("/Users/fujunhan/Desktop/MIS2011/GroupProject")
INPUT_FILE = BASE_DIR / "douyin/douyin_naixue_simulated_comments.jsonl"

SENTIMENT_MODEL = "H-Z-Ning/Senti-RoBERTa-Mini"
EMOTION_MODEL = "Johnson8187/Chinese-Emotion-Small"
BATCH_SIZE = 32

EMOTION_LABEL_MAP = {
    "LABEL_0": "平淡语气",
    "LABEL_1": "关切语调",
    "LABEL_2": "开心语调",
    "LABEL_3": "愤怒语调",
    "LABEL_4": "悲伤语调",
    "LABEL_5": "疑问语调",
    "LABEL_6": "惊奇语调",
    "LABEL_7": "厌恶语调",
}


def map_star_to_polarity(star: int) -> str:
    if star <= 2:
        return "negative"
    if star == 3:
        return "neutral"
    return "positive"


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def load_records(file_path: Path) -> list[dict]:
    raw_text = file_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    if raw_text.startswith("["):
        data = json.loads(raw_text)
        if not isinstance(data, list):
            raise ValueError(f"{file_path} does not contain a JSON array")
        return data

    return [json.loads(line) for line in raw_text.splitlines() if line.strip()]


def clean_weibo_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<img[^>]*alt=\"([^\"]+)\"[^>]*>", r" \1 ", text)
    text = re.sub(r"</?(a|span)[^>]*>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_text(record: dict) -> str:
    content = clean_weibo_text(record.get("content") or "")
    note_id = (record.get("note_id") or "").strip()
    nickname = (record.get("nickname") or "").strip()
    parts = [part for part in [content, f"评论用户:{nickname}" if nickname else "", f"微博ID:{note_id}" if note_id else ""] if part]
    return " | ".join(parts).strip()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    records = load_records(INPUT_FILE)
    if not records:
        raise ValueError(f"No records found in {INPUT_FILE}")

    device = 0 if torch.cuda.is_available() else -1

    sentiment_pipe = pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        device=device,
        truncation=True,
        max_length=256,
    )

    emotion_pipe = pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        tokenizer=EMOTION_MODEL,
        device=device,
        truncation=True,
        max_length=256,
    )

    texts = [build_text(record) for record in records]
    sentiment_results = []
    emotion_results = []

    for batch in tqdm(list(chunked(texts, BATCH_SIZE)), desc="Inference", unit="batch"):
        safe_batch = [text if text else "空文本" for text in batch]
        sentiment_results.extend(sentiment_pipe(safe_batch))
        emotion_results.extend(emotion_pipe(safe_batch))

    for record, text, sent_res, emo_res in tqdm(
        zip(records, texts, sentiment_results, emotion_results),
        total=len(records),
        desc="Annotating",
        unit="row",
    ):
        raw_label = sent_res["label"]
        star = int(raw_label.split("_")[-1]) + 1
        polarity = map_star_to_polarity(star)

        record["attributes"] = {
            "source": "weibo_comment",
            "analyzed_text": text,
            "sentiment": {
                "model": SENTIMENT_MODEL,
                "raw_label": raw_label,
                "star_rating": star,
                "polarity": polarity,
                "score": round(float(sent_res["score"]), 6),
            },
            "emotion_fine_grained": {
                "model": EMOTION_MODEL,
                "raw_label": emo_res["label"],
                "label": EMOTION_LABEL_MAP.get(emo_res["label"], emo_res["label"]),
                "score": round(float(emo_res["score"]), 6),
            },
        }

    tmp_path = INPUT_FILE.with_suffix(INPUT_FILE.suffix + ".tmp")
    backup_path = INPUT_FILE.with_suffix(INPUT_FILE.suffix + ".bak")

    if not backup_path.exists():
        os.replace(INPUT_FILE, backup_path)
    else:
        INPUT_FILE.unlink(missing_ok=True)

    with tmp_path.open("w", encoding="utf-8") as f:
        for record in tqdm(records, desc="Writing", unit="row"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    os.replace(tmp_path, INPUT_FILE)

    print(f"updated: {INPUT_FILE}")
    print(f"backup: {backup_path}")
    print(f"records: {len(records)}")


if __name__ == "__main__":
    main()
