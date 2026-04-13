# download_opus100.py
from huggingface_hub import hf_hub_download
import pandas as pd
import json
import os

os.makedirs("data", exist_ok=True)

# Скачиваем ВСЕ сплиты: train, test, validation
files_to_try = [
    "ru-zh/train-00000-of-00001.parquet",
    "ru-zh/test-00000-of-00001.parquet",
    "ru-zh/validation-00000-of-00001.parquet",
]

count = 0
with open("data/opus100.jsonl", "w", encoding="utf-8") as out:
    for filename in files_to_try:
        try:
            print(f"Скачиваю {filename}...")
            path = hf_hub_download(
                repo_id="Helsinki-NLP/opus-100",
                filename=filename,
                repo_type="dataset"
            )
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                t = row.get("translation", {})
                zh = t.get("zh", "")
                ru = t.get("ru", "")
                if zh and ru:
                    out.write(json.dumps({"zh": zh, "ru": ru}, ensure_ascii=False) + "\n")
                    count += 1
            print(f"  +{len(df)} строк")
        except Exception as e:
            print(f"  {filename}: {e}")

print(f"\nГотово: {count:,} пар → data/opus100.jsonl")# download_opus100.py
from huggingface_hub import hf_hub_download
import pandas as pd
import json
import os

os.makedirs("data", exist_ok=True)

# Скачиваем ВСЕ сплиты: train, test, validation
files_to_try = [
    "ru-zh/train-00000-of-00001.parquet",
    "ru-zh/test-00000-of-00001.parquet",
    "ru-zh/validation-00000-of-00001.parquet",
]

count = 0
with open("data/opus100.jsonl", "w", encoding="utf-8") as out:
    for filename in files_to_try:
        try:
            print(f"Скачиваю {filename}...")
            path = hf_hub_download(
                repo_id="Helsinki-NLP/opus-100",
                filename=filename,
                repo_type="dataset"
            )
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                t = row.get("translation", {})
                zh = t.get("zh", "")
                ru = t.get("ru", "")
                if zh and ru:
                    out.write(json.dumps({"zh": zh, "ru": ru}, ensure_ascii=False) + "\n")
                    count += 1
            print(f"  +{len(df)} строк")
        except Exception as e:
            print(f"  {filename}: {e}")

print(f"\nГотово: {count:,} пар → data/opus100.jsonl")