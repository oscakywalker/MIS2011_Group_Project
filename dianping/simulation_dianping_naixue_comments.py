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
OUTPUT_FILE = BASE_DIR / "dianping/dianping_naixue_simulated_comments.jsonl"

MODEL_PATH = "/mnt/nvme/fjh/Qwen2.5-VL-7B-Instruct"
TOTAL_COMMENTS = 1097
BATCH_SIZE = 30

KEYWORDS = ["奈雪新品上市", "奈雪价格太高", "奈雪探店体验", "奈雪门店服务"]

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

DIANPING_NICK_PREFIX = [
    "爱喝奶茶的",
    "今天也想喝",
    "打工人",
    "周末探店的",
    "嘴馋星人",
    "省钱版",
    "甜品脑袋",
    "点评老用户",
    "减肥失败的",
    "附近上班的",
]

DIANPING_NICK_SUFFIX = [
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
    "新品颜值不错，店里出杯也快，口味清爽但价格确实偏高",
    "周末去探店排队有点久，环境还可以，新品适合拍照打卡",
    "服务态度挺好，会主动提醒甜度，整体体验不错就是单价高",
    "新品喝起来层次还行，茶底比较清爽，但性价比一般",
    "门店位置方便，堂食座位偏少，价格对学生党不太友好",
    "包装和杯型很好看，适合拍照，味道没有特别惊艳",
    "店员推荐了新品，入口不腻，不过这个价位会降低复购频率",
    "环境干净，取餐速度正常，奈雪新品上市尝鲜可以",
    "价格比附近饮品店贵一些，胜在环境和服务比较稳定",
    "这次探店体验中规中矩，新品有记忆点但不算必点",
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
你是一个大众点评用户评论生成器。请生成 {batch_size} 条符合大众点评风格的中文探店留言。

主题关键词：{keyword}

要求：
1. 评论围绕奈雪门店探店体验，重点写新品上市、价格偏高、口味、服务、环境、排队、出杯速度、复购意愿。
2. 口吻像真实大众点评用户：具体、生活化、消费后反馈，不要像广告文案。
3. 可以有正面、中性、负面混合，不要全部好评。
4. 长度控制在 18 到 70 个中文字符左右。
5. 不要写星级编号，不要出现“五星好评”“商家回复”等模板话。
6. 不要解释。
7. 只返回 JSON 数组，数组元素是字符串。

示例格式：
[
  "新品颜值挺高，茶底清爽不腻，就是价格比预期高一点",
  "周末探店人比较多，排队十几分钟，店员服务态度还可以",
  "环境干净适合坐一会儿，但新品味道没有惊艳到我"
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

    if any(token in content for token in ["探店", "环境", "门店", "排队", "座位", "堂食", "打卡"]):
        keywords.append("奈雪探店体验")

    if any(token in content for token in ["服务", "店员", "出杯", "取餐", "态度"]):
        keywords.append("奈雪门店服务")

    if not keywords:
        keywords.append(random.choice(KEYWORDS))

    return keywords


def infer_experience_aspects(content: str) -> list[str]:
    aspect_rules = {
        "new_product": ["新品", "上市", "新口味", "上新"],
        "price": ["贵", "价格", "性价比", "钱包", "不值", "劝退"],
        "taste": ["口味", "好喝", "茶底", "甜度", "清爽", "不腻", "味道"],
        "service": ["服务", "店员", "态度", "推荐"],
        "environment": ["环境", "座位", "堂食", "干净", "打卡", "门店"],
        "queue": ["排队", "等", "出杯", "取餐"],
        "repurchase": ["复购", "再来", "下次", "尝鲜"],
    }
    aspects = [
        aspect
        for aspect, tokens in aspect_rules.items()
        if any(token in content for token in tokens)
    ]
    return aspects or ["overall_experience"]


def make_fake_record(content: str, idx: int, source_records: list[dict]) -> dict:
    base = random.choice(source_records) if source_records else {}
    dt = weighted_random_datetime()

    comment_id = f"dp_sim_{idx:08d}"
    user_id = f"dp_user_{random.randint(1000000000, 9999999999)}"
    nickname = random.choice(DIANPING_NICK_PREFIX) + random.choice(DIANPING_NICK_SUFFIX)

    return {
        "comment_id": comment_id,
        "create_time": int(dt.timestamp()),
        "create_date_time": dt.strftime("%Y-%m-%d %H:%M:%S+08:00"),
        "note_id": str(base.get("note_id") or f"dp_shop_{random.randint(1000000000, 9999999999)}"),
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
        "profile_url": f"https://www.dianping.com/member/{user_id}",
        "avatar": "",
        "attributes": {
            "source": "dianping_comment_simulated",
            "platform": "dianping",
            "brand": "奈雪",
            "keywords": infer_keywords(content),
            "experience_aspects": infer_experience_aspects(content),
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