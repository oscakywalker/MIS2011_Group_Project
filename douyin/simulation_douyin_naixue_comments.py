import json
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


BASE_DIR = Path("/mnt/nvme/fjh/data_simulation")
INPUT_FILE = BASE_DIR / "weibo/search_comments_2026-04-17.jsonl"
OUTPUT_FILE = BASE_DIR / "douyin/douyin_naixue_simulated_comments.jsonl"

MODEL_PATH = "/mnt/nvme/fjh/Qwen2.5-VL-7B-Instruct"
TOTAL_COMMENTS = 1238
BATCH_SIZE = 30

KEYWORDS = ["奈雪新品上市", "奈雪价格太高"]

IP_LOCATIONS = [
    "广东",
    "上海",
    "北京",
    "浙江",
    "江苏",
    "四川",
    "重庆",
    "湖北",
    "湖南",
    "福建",
    "山东",
    "河南",
    "陕西",
    "天津",
    "辽宁",
    "安徽",
    "广西",
]

DOUYIN_NICK_PREFIX = [
    "爱喝奶茶的",
    "今天也想喝",
    "打工人",
    "路过的",
    "嘴馋星人",
    "省钱版",
    "甜品脑袋",
    "奶茶观察员",
    "减肥失败的",
    "路人甲",
]

DOUYIN_NICK_SUFFIX = [
    "小张",
    "阿七",
    "小鱼",
    "葡萄",
    "桃子",
    "啵啵",
    "小林",
    "椰椰",
    "栗子",
    "小羊",
    "橙子",
    "打工仔",
]

FALLBACK_COMMENTS = [
    "奈雪新品上市我先观望一下，颜值是有的，价格也是真的有点劝退",
    "新品看起来挺好喝的，但是奈雪这个价格我真的下不去手",
    "奈雪价格太高了吧，一杯快赶上一顿饭了",
    "新品上了但钱包没跟上，等第二杯半价再说",
    "奈雪这次新品包装挺好看，味道不知道值不值这个价",
    "说实话奈雪新品上市我会心动，但看到价格又冷静了",
    "不是不爱喝，是奈雪价格太高，我选择自己泡茶",
    "新品可以，价格不太可以，打工人真的喝不起了",
    "奈雪新品上市冲热搜可以，冲钱包不行",
    "这个价位我只能说适合拍照，不适合天天喝",
]


def load_json_objects(path: Path) -> list[dict]:
    """
    Supports:
    1. A standard JSON array.
    2. Multiple pretty-printed JSON objects written back-to-back.
    3. Standard JSONL with one compact object per line.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} does not contain a JSON array")
        return data

    decoder = json.JSONDecoder()
    records = []
    idx = 0

    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break

        obj, next_idx = decoder.raw_decode(text, idx)
        if isinstance(obj, dict):
            records.append(obj)
        idx = next_idx

    return records


def weighted_random_datetime() -> datetime:
    """
    Randomly pick a timestamp between 2024-04-22 and 2026-05-12.
    The probability density is higher near 2026-05-12.
    """
    tz = timezone(timedelta(hours=8))
    start_dt = datetime(2024, 4, 22, 0, 0, 0, tzinfo=tz)
    end_dt = datetime(2026, 5, 12, 23, 59, 59, tzinfo=tz)

    total_seconds = int((end_dt - start_dt).total_seconds())
    offset_seconds = int(random.triangular(0, total_seconds, total_seconds))

    return start_dt + timedelta(seconds=offset_seconds)


def extract_json_array(text: str) -> list[str]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    comments = []
    for item in data:
        if isinstance(item, str):
            cleaned = item.strip()
        elif isinstance(item, dict):
            cleaned = str(item.get("content", "")).strip()
        else:
            continue

        if cleaned:
            comments.append(cleaned)

    return comments


def build_prompt(batch_size: int) -> str:
    keyword = random.choice(KEYWORDS)

    return f"""
你是一个中文短视频平台评论生成器。请生成 {batch_size} 条符合抖音评论区风格的中文评论。

主题关键词：{keyword}

要求：
1. 评论围绕“奈雪新品上市”或“奈雪价格太高”。
2. 口吻像真实抖音用户：短句、口语化、有吐槽、有种草、有观望、有玩梗。
3. 不要像广告文案，不要太正式。
4. 长度控制在 6 到 35 个中文字符左右。
5. 可以少量使用表情文字，例如：哈哈哈、救命、笑死、狠狠心动、钱包哭了。
6. 不要出现编号。
7. 不要解释。
8. 只返回 JSON 数组，数组元素是字符串。

示例格式：
[
  "奈雪新品看着好喝但价格劝退",
  "钱包说它不同意",
  "新品上市我先蹲个测评"
]
""".strip()


def generate_comments(model, processor, target_count: int) -> list[str]:
    comments = []
    seen = set()

    with tqdm(total=target_count, desc="Generating", unit="条") as pbar:
        while len(comments) < target_count:
            need = min(BATCH_SIZE, target_count - len(comments))
            prompt = build_prompt(need)

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=900,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.9,
                    repetition_penalty=1.08,
                )

            generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
            output_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            batch_comments = extract_json_array(output_text)
            if not batch_comments:
                batch_comments = random.sample(
                    FALLBACK_COMMENTS,
                    k=min(need, len(FALLBACK_COMMENTS)),
                )

            for comment in batch_comments:
                comment = re.sub(r"\s+", "", comment).strip()
                comment = comment.strip("。.!！,，")
                if not comment or comment in seen:
                    continue

                seen.add(comment)
                comments.append(comment)
                pbar.update(1)

                if len(comments) >= target_count:
                    break

    return comments


def infer_keywords(content: str) -> list[str]:
    keywords = []

    if any(token in content for token in ["新品", "上市", "新口味", "上新"]):
        keywords.append("奈雪新品上市")

    if any(token in content for token in ["贵", "价格", "钱包", "喝不起", "劝退", "不值"]):
        keywords.append("奈雪价格太高")

    if not keywords:
        keywords.append(random.choice(KEYWORDS))

    return keywords


def make_fake_record(content: str, idx: int, source_records: list[dict]) -> dict:
    base = random.choice(source_records) if source_records else {}
    dt = weighted_random_datetime()

    comment_id = f"dy_sim_{idx:08d}"
    user_id = f"dy_user_{random.randint(1000000000, 9999999999)}"
    nickname = random.choice(DOUYIN_NICK_PREFIX) + random.choice(DOUYIN_NICK_SUFFIX)

    return {
        "comment_id": comment_id,
        "create_time": int(dt.timestamp()),
        "create_date_time": dt.strftime("%Y-%m-%d %H:%M:%S+08:00"),
        "note_id": str(base.get("note_id") or f"dy_note_{random.randint(1000000000, 9999999999)}"),
        "content": content,
        "sub_comment_count": str(
            random.choices([0, 1, 2, 3, 4, 5], weights=[60, 18, 10, 6, 4, 2])[0]
        ),
        "comment_like_count": str(
            random.choices(
                list(range(0, 200)),
                weights=[
                    80 if i < 10 else 15 if i < 50 else 4 if i < 120 else 1
                    for i in range(200)
                ],
            )[0]
        ),
        "last_modify_ts": int(time.time() * 1000),
        "ip_location": random.choice(IP_LOCATIONS),
        "user_id": user_id,
        "nickname": nickname,
        "gender": random.choice(["m", "f", ""]),
        "profile_url": f"https://www.douyin.com/user/{user_id}",
        "avatar": "",
        "attributes": {
            "source": "douyin_comment_simulated",
            "platform": "douyin",
            "brand": "奈雪",
            "keywords": infer_keywords(content),
            "generated_by": MODEL_PATH,
            "based_on_structure": str(INPUT_FILE),
            "time_sampling": {
                "start": "2024-04-22 00:00:00+08:00",
                "end": "2026-05-12 23:59:59+08:00",
                "method": "random.triangular(mode=end)",
            },
        },
    }


def main():
    source_records = load_json_objects(INPUT_FILE)

    print(f"loaded source records: {len(source_records)}")
    print(f"loading model from: {MODEL_PATH}")
    print(f"cuda available: {torch.cuda.is_available()}")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    comments = generate_comments(model, processor, TOTAL_COMMENTS)
    records = [
        make_fake_record(content, idx + 1, source_records)
        for idx, content in enumerate(comments)
    ]

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"done: {OUTPUT_FILE}")
    print(f"records: {len(records)}")


if __name__ == "__main__":
    main()