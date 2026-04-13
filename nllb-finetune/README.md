# NLLB Fine-tune: Перевод Chinese → Russian

Автоматически скачивает и комбинирует лучшие датасеты для zh→ru.

## Быстрый старт (одна команда)
```bash
pip install -r requirements.txt
python train.py
```
Скачает ~2M пар из 4 источников, отфильтрует, обучит NLLB-600M.
RTX 4070 Super: ~2-3 часа.

## Команды
```bash
python train.py                              # скачать данные + обучить
python train.py download                     # только скачать данные
python train.py train                        # только обучить
python train.py translate "你好世界"          # перевести текст
python train.py compare                      # сравнить с MarianMT
python train.py interactive                  # интерактивный режим
python train.py srt input.srt output.srt     # перевести субтитры
python train.py convert file.tmx             # XML/TMX → JSONL
python train.py info                         # статистика данных
```

## Датасеты (автоматически)
| Источник       | Размер | Стиль        | Включён |
|----------------|--------|--------------|---------|
| OPUS-100       | ~1M    | Микс         | ✅      |
| OpenSubtitles  | ~5M    | Разговорный  | ✅      |
| UN Corpus      | ~5M    | Формальный   | ✅      |
| Tatoeba        | ~10K   | Проверенный  | ✅      |

Настрой лимиты в `config.yaml → data.limits`.

## Свои данные
Любой формат — положи файл и добавь в `config.yaml`:
```yaml
data:
  custom_files:
    - "extra/my_data.jsonl"       # {"zh": "...", "ru": "..."}
    - "extra/corpus.tmx"          # TMX (OPUS формат)
    - "extra/data.xml"            # любой XML
    - "extra/pairs.csv"           # CSV/TSV
```

Или из SRT:
```yaml
data:
  srt_zh_dir: "srt/chinese"
  srt_ru_dir: "srt/russian"
```

Скачанные файлы с OPUS (Moses .txt.gz): распакуй, получишь `corpus.zh` + `corpus.ru` — конвертер подхватит автоматически.
