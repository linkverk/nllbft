# download_opus100.py
import urllib.request
import tarfile
import json
import os

os.makedirs("data", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

url = "https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-v1.0.tar.gz"
tar_path = "tmp/opus-100.tar.gz"

# Скачиваем (~200 MB)
if not os.path.exists(tar_path):
    print(f"Скачиваю OPUS-100 (~200 MB)...")
    def progress(count, block, total):
        mb = count * block / 1e6
        total_mb = total / 1e6 if total > 0 else "?"
        print(f"\r  {mb:.0f}/{total_mb} MB", end="", flush=True)
    urllib.request.urlretrieve(url, tar_path, reporthook=progress)
    print("\n  Скачано!")
else:
    print(f"Уже скачан: {tar_path}")

# Распаковываем только ru-zh файлы
print("Распаковка ru-zh...")
with tarfile.open(tar_path, "r:gz") as tf:
    members = [m for m in tf.getmembers() if "ru-zh" in m.name and m.isfile()]
    print(f"  Найдено {len(members)} файлов:")
    for m in members:
        print(f"    {m.name} ({m.size / 1e6:.1f} MB)")
    tf.extractall("tmp", members=members)

# Ищем train файлы
zh_path = ru_path = None
for m in members:
    path = os.path.join("tmp", m.name)
    if path.endswith(".zh"):
        zh_path = path
    elif path.endswith(".ru"):
        ru_path = path

if not zh_path or not ru_path:
    print("Файлы не найдены!")
    exit(1)

# Конвертируем
print(f"Конвертация...")
count = 0
with open(zh_path, "r", encoding="utf-8") as fz, \
     open(ru_path, "r", encoding="utf-8") as fr, \
     open("data/opus100.jsonl", "w", encoding="utf-8") as out:
    for zh, ru in zip(fz, fr):
        zh, ru = zh.strip(), ru.strip()
        if zh and ru:
            out.write(json.dumps({"zh": zh, "ru": ru}, ensure_ascii=False) + "\n")
            count += 1
print(f"Готово: {count:,} пар → data/opus100.jsonl")