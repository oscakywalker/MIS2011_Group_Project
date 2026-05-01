import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import pipeline

BASE_DIR = Path("/Users/fujunhan/Desktop/MIS2011/GroupProject")
FILES = [
    BASE_DIR / "xhs/jsonl/search_comments_2026-04-06.jsonl",
    BASE_DIR / "xhs/jsonl/search_contents_2026-04-06.jsonl",
]

SENTIMENT_MODEL = "H-Z-Ning/Senti-RoBERTa-Mini"
EMOTION_MODEL = "Johnson8187/Chinese-Emotion-Small"

# H-Z-Ning/Senti-RoBERTa-Mini: 1~5 star
def map_star_to_polarity(star: int) -> str:
    if star <= 2:
        return "negative"
    if star == 3:
        return "neutral"
    return "positive"

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

def build_text(record: dict, file_name: str) -> str:
    if "search_comments" in file_name:
        return (record.get("content") or "").strip()

    title = (record.get("title") or "").strip()
    desc = (record.get("desc") or "").strip()
    text = f"{title} {desc}".strip()
    return text

def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]

def main():
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

    for file_path in FILES:
        with open(file_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        texts = [build_text(r, file_path.name) for r in records]
        text_batches = list(chunked(texts, 32))

        sentiment_results = []
        emotion_results = []

        for batch in tqdm(text_batches, desc=f"Inference {file_path.name}", unit="batch"):
            safe_batch = [t if t else "空文本" for t in batch]
            sentiment_results.extend(sentiment_pipe(safe_batch))
            emotion_results.extend(emotion_pipe(safe_batch))

        for record, text, sent_res, emo_res in tqdm(
            zip(records, texts, sentiment_results, emotion_results),
            total=len(records),
            desc=f"Annotating {file_path.name}",
            unit="row",
        ):
            # sentiment: LABEL_0 ~ LABEL_4 -> 1~5 star
            raw_label = sent_res["label"]
            star = int(raw_label.split("_")[-1]) + 1
            polarity = map_star_to_polarity(star)

            emotion_label = EMOTION_LABEL_MAP.get(emo_res["label"], emo_res["label"])

            record["attributes"] = {
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
                    "label": emotion_label,
                    "score": round(float(emo_res["score"]), 6),
                },
            }

        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")

        if not backup_path.exists():
            os.replace(file_path, backup_path)
        else:
            file_path.unlink(missing_ok=True)

        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in tqdm(records, desc=f"Writing {file_path.name}", unit="row"):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        os.replace(tmp_path, file_path)
        print(f"done: {file_path}")
        print(f"backup: {backup_path}")

if __name__ == "__main__":
    main()
