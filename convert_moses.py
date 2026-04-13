# convert_moses.py
import json
with open("OpenSubtitles.ru-zh_CN.zh_CN", "r", encoding="utf-8") as fz, \
     open("OpenSubtitles.ru-zh_CN.ru", "r", encoding="utf-8") as fr, \
     open("data/opensubs.jsonl", "w", encoding="utf-8") as out:
    count = 0
    for zh, ru in zip(fz, fr):
        zh, ru = zh.strip(), ru.strip()
        if zh and ru and len(zh) > 1 and len(ru) > 1:
            out.write(json.dumps({"zh": zh, "ru": ru}, ensure_ascii=False) + "\n")
            count += 1
    print(f"Готово: {count:,} пар")