#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════
  NLLB Fine-tune: Перевод Chinese → Russian
══════════════════════════════════════════════════════════════

Датасеты (автоматически скачивает и комбинирует):
  • OPUS-100 zh-ru        ~1M пар   — чистый, микс доменов
  • OpenSubtitles zh-ru   ~5M пар   — разговорный, для видео
  • UN Parallel Corpus    ~5M пар   — формальный, точный
  • Tatoeba zh-ru         ~10K пар  — вручную проверенный
  • Свои файлы            JSONL/TMX/XML/CSV/SRT

Использование:
  python train.py                              — скачать данные + обучить
  python train.py download                     — только скачать данные
  python train.py train                        — только обучить (данные уже есть)
  python train.py translate "你好世界"          — перевести текст
  python train.py compare                      — сравнить с MarianMT
  python train.py interactive                  — интерактивный режим
  python train.py srt input.srt output.srt     — перевести субтитры
  python train.py convert file.tmx             — конвертировать XML/TMX → JSONL
  python train.py info                         — статистика данных
"""

import argparse
import json
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
log = logging.getLogger("nllb")
console = Console()


# ═══════════════════════════════════════════════════════════
# Конфиг
# ═══════════════════════════════════════════════════════════

def load_config(path: str = "config.yaml") -> dict:
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    log.warning(f"{path} не найден, дефолты")
    return {}


def get(cfg, *keys, default=None):
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    return val


# ═══════════════════════════════════════════════════════════
# JSONL I/O
# ═══════════════════════════════════════════════════════════

def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log.info(f"  → {path} ({len(data):,} пар)")


# ═══════════════════════════════════════════════════════════
# Скачивание датасетов
# ═══════════════════════════════════════════════════════════

def download_opus100(limit: int = 1_000_000) -> list[dict]:
    """OPUS-100: чистый параллельный корпус, ~1M zh-ru."""
    from datasets import load_dataset
    log.info(f"[1/4] OPUS-100 (лимит {limit:,})...")
    pairs = []

    # OPUS-100 использует конфиг "ru-zh" (не "zh-ru")
    configs_to_try = [
        ("Helsinki-NLP/opus-100", "ru-zh"),
        ("Helsinki-NLP/opus-100", "zh-en"),  # fallback через en
    ]

    for ds_name, subset in configs_to_try:
        if pairs:
            break
        try:
            ds = load_dataset(ds_name, subset, split="train")
            for item in tqdm(ds, desc="  OPUS-100", total=min(limit, len(ds))):
                if len(pairs) >= limit:
                    break
                t = item.get("translation", item)
                # Ключи зависят от конфига
                zh = t.get("zh", t.get("zh_cn", ""))
                ru = t.get("ru", "")
                if zh and ru:
                    pairs.append({"zh": zh, "ru": ru})
        except Exception as e:
            log.debug(f"  {ds_name}/{subset}: {e}")
            continue

    log.info(f"  OPUS-100: {len(pairs):,} пар")
    return pairs


def download_opensubtitles(limit: int = 5_000_000) -> list[dict]:
    """OpenSubtitles: разговорный стиль, фильмы/сериалы."""
    from datasets import load_dataset
    log.info(f"[2/4] OpenSubtitles (лимит {limit:,})...")
    pairs = []

    # Попробовать разные форматы датасета
    loaders = [
        # Новый Parquet-формат
        lambda: load_dataset("sentence-transformers/parallel-sentences-opensubtitles",
                             "ru", split="train", streaming=True),
        # Прямой OPUS формат
        lambda: load_dataset("open_subtitles", lang1="ru", lang2="zh_cn",
                             split="train", streaming=True),
    ]

    for loader in loaders:
        if pairs:
            break
        try:
            ds = loader()
            for item in tqdm(ds, desc="  OpenSubs", total=limit):
                if len(pairs) >= limit:
                    break
                # sentence-transformers формат
                en = item.get("english", "")
                non_en = item.get("non_english", "")
                # OPUS формат
                t = item.get("translation", {})
                zh = t.get("zh_cn", t.get("zh", non_en if not en else ""))
                ru = t.get("ru", en if not non_en else non_en)

                # sentence-transformers: en + non_english(ru), нет прямого zh-ru
                # Пропускаем если нет прямой пары
                if zh and ru and len(zh) > 2 and len(ru) > 2:
                    pairs.append({"zh": zh, "ru": ru})
        except Exception as e:
            log.debug(f"  OpenSubtitles loader ошибка: {e}")
            continue

    if not pairs:
        log.warning("  OpenSubtitles: не удалось скачать. Скачай вручную с https://opus.nlpl.eu/")

    log.info(f"  OpenSubtitles: {len(pairs):,} пар")
    return pairs


def download_un_corpus(limit: int = 5_000_000) -> list[dict]:
    """UN Parallel Corpus: формальный, высокое качество."""
    from datasets import load_dataset
    log.info(f"[3/4] UN Corpus (лимит {limit:,})...")
    pairs = []

    # UN Corpus: конфиг "ru-zh" (не "zh-ru")
    configs_to_try = [
        ("Helsinki-NLP/un_pc", "ru-zh"),
        ("Helsinki-NLP/un_pc", "en-zh"),  # через en как pivot
    ]

    for ds_name, subset in configs_to_try:
        if pairs:
            break
        try:
            ds = load_dataset(ds_name, subset, split="train", streaming=True)
            for item in tqdm(ds, desc="  UN", total=limit):
                if len(pairs) >= limit:
                    break
                t = item.get("translation", item)
                zh = t.get("zh", "")
                ru = t.get("ru", "")
                if zh and ru:
                    pairs.append({"zh": zh, "ru": ru})
        except Exception as e:
            log.debug(f"  {ds_name}/{subset}: {e}")
            continue

    if not pairs:
        log.warning("  UN Corpus: не удалось скачать. Скачай вручную с https://opus.nlpl.eu/")

    log.info(f"  UN Corpus: {len(pairs):,} пар")
    return pairs


def download_tatoeba(limit: int = 50_000) -> list[dict]:
    """Tatoeba: маленький, но вручную проверенный."""
    from datasets import load_dataset
    log.info(f"[4/4] Tatoeba (лимит {limit:,})...")
    pairs = []

    configs_to_try = [
        ("Helsinki-NLP/tatoeba_mt", "zho-rus"),
        ("Helsinki-NLP/tatoeba_mt", "rus-zho"),
    ]

    for ds_name, subset in configs_to_try:
        if pairs:
            break
        try:
            ds = load_dataset(ds_name, subset, split="train")
            for item in ds:
                if len(pairs) >= limit:
                    break
                # tatoeba_mt формат: sourceString / targetString
                src = item.get("sourceString", "")
                tgt = item.get("targetString", "")
                src_lang = item.get("sourceLang", subset.split("-")[0])
                tgt_lang = item.get("targetLang", subset.split("-")[1])

                if "zho" in src_lang:
                    zh, ru = src, tgt
                else:
                    zh, ru = tgt, src

                if zh and ru:
                    pairs.append({"zh": zh, "ru": ru})
        except Exception as e:
            log.debug(f"  {ds_name}/{subset}: {e}")
            continue

    log.info(f"  Tatoeba: {len(pairs):,} пар")
    return pairs


# ═══════════════════════════════════════════════════════════
# Конвертация XML / TMX / CSV / SRT
# ═══════════════════════════════════════════════════════════

def convert_file(path: str) -> list[dict]:
    """Автоопределение формата и конвертация в пары."""
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext == ".jsonl":
        return load_jsonl(path)

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []

    # Читаем начало файла для определения XML-формата
    if ext in (".xml", ".tmx", ".xliff", ".xlf"):
        return _convert_xml(path)

    if ext in (".csv", ".tsv"):
        return _convert_csv(path, "\t" if ext == ".tsv" else ",")

    if ext == ".srt":
        log.warning(f"  SRT нельзя конвертировать без пары. Используй --srt-zh-dir и --srt-ru-dir")
        return []

    # Попробовать как plain text (два файла рядом: file.zh + file.ru)
    zh_path = path.replace(".ru", ".zh") if ".ru" in path else path + ".zh"
    ru_path = path.replace(".zh", ".ru") if ".zh" in path else path + ".ru"
    if Path(zh_path).exists() and Path(ru_path).exists():
        return _convert_parallel_txt(zh_path, ru_path)

    log.warning(f"  Неизвестный формат: {path}")
    return []


def _convert_xml(path: str) -> list[dict]:
    """TMX / XLIFF / generic XML."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1000).lower()

    pairs = []

    if "<tmx" in head:
        log.info(f"  Формат TMX: {path}")
        tree = ET.parse(path)
        for tu in tree.iter("tu"):
            texts = {}
            for tuv in tu.findall("tuv"):
                lang = tuv.get("{http://www.w3.org/XML/1998/namespace}lang",
                              tuv.get("lang", tuv.get("xml:lang", "")))
                seg = tuv.find("seg")
                if seg is not None and seg.text:
                    texts[lang.lower()[:2]] = seg.text.strip()
            zh = texts.get("zh", "")
            ru = texts.get("ru", "")
            if zh and ru:
                pairs.append({"zh": zh, "ru": ru})

    elif "<xliff" in head:
        log.info(f"  Формат XLIFF: {path}")
        tree = ET.parse(path)
        for elem in tree.iter():
            if elem.tag.endswith("trans-unit") or elem.tag == "trans-unit":
                src = tgt = None
                for child in elem:
                    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if tag == "source":
                        src = child.text
                    elif tag == "target":
                        tgt = child.text
                if src and tgt:
                    pairs.append({"zh": src.strip(), "ru": tgt.strip()})
    else:
        log.info(f"  Generic XML: {path}")
        tree = ET.parse(path)
        # Пробуем найти пары в любой структуре
        for elem in tree.iter():
            zh = elem.findtext("zh") or elem.findtext("source") or elem.get("zh", "")
            ru = elem.findtext("ru") or elem.findtext("target") or elem.get("ru", "")
            if zh and ru:
                pairs.append({"zh": zh.strip(), "ru": ru.strip()})

    log.info(f"  → {len(pairs):,} пар из {Path(path).name}")
    return pairs


def _convert_csv(path: str, delimiter: str) -> list[dict]:
    """CSV/TSV: первая колонка zh, вторая ru."""
    import csv
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                pairs.append({"zh": row[0].strip(), "ru": row[1].strip()})
    log.info(f"  → {len(pairs):,} пар из {Path(path).name}")
    return pairs


def _convert_parallel_txt(zh_path: str, ru_path: str) -> list[dict]:
    """Два параллельных текстовых файла (Moses формат)."""
    pairs = []
    with open(zh_path, "r", encoding="utf-8") as fz, \
         open(ru_path, "r", encoding="utf-8") as fr:
        for zh, ru in zip(fz, fr):
            zh, ru = zh.strip(), ru.strip()
            if zh and ru:
                pairs.append({"zh": zh, "ru": ru})
    log.info(f"  → {len(pairs):,} пар из parallel txt")
    return pairs


# ═══════════════════════════════════════════════════════════
# SRT пары
# ═══════════════════════════════════════════════════════════

def parse_srt(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r"\n\s*\n", content.strip())
    entries = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            try:
                idx = int(lines[0].strip())
                ts = lines[1].strip()
                text = " ".join(lines[2:]).strip()
                if text:
                    entries.append({"index": idx, "timestamp": ts, "text": text})
            except ValueError:
                continue
    return entries


def write_srt(entries: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(f"{e['index']}\n{e['timestamp']}\n{e['text']}\n\n")


def collect_srt_pairs(zh_dir: str, ru_dir: str) -> list[dict]:
    pairs = []
    for zh_file in sorted(Path(zh_dir).glob("*.srt")):
        ru_file = Path(ru_dir) / zh_file.name
        if not ru_file.exists():
            continue
        zh_entries = {e["index"]: e["text"] for e in parse_srt(str(zh_file))}
        ru_entries = {e["index"]: e["text"] for e in parse_srt(str(ru_file))}
        for idx in zh_entries:
            if idx in ru_entries:
                pairs.append({"zh": zh_entries[idx], "ru": ru_entries[idx]})
    log.info(f"  SRT: {len(pairs):,} пар из {zh_dir}")
    return pairs


# ═══════════════════════════════════════════════════════════
# Фильтрация и дедупликация
# ═══════════════════════════════════════════════════════════

def filter_pairs(pairs: list[dict], cfg: dict) -> list[dict]:
    f = get(cfg, "data", "filter", default={})
    min_zh = f.get("min_zh_len", 2)
    min_ru = f.get("min_ru_len", 2)
    max_zh = f.get("max_zh_len", 500)
    max_ru = f.get("max_ru_len", 1000)
    do_dedup = f.get("dedup", True)

    before = len(pairs)

    # Фильтрация по длине
    pairs = [
        p for p in pairs
        if min_zh <= len(p["zh"]) <= max_zh
        and min_ru <= len(p["ru"]) <= max_ru
    ]
    after_filter = len(pairs)

    # Дедупликация
    if do_dedup:
        seen = set()
        unique = []
        for p in pairs:
            key = (p["zh"], p["ru"])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        pairs = unique

    log.info(f"  Фильтрация: {before:,} → {after_filter:,} (длина) → {len(pairs):,} (дедуп)")
    return pairs


# ═══════════════════════════════════════════════════════════
# Главная: скачать + собрать все данные
# ═══════════════════════════════════════════════════════════

def download_all(cfg: dict) -> tuple[list, list]:
    """Скачать и скомбинировать все датасеты из конфига."""
    sources = get(cfg, "data", "sources", default={})
    limits = get(cfg, "data", "limits", default={})
    all_pairs = []

    console.rule("[bold green]Скачивание датасетов")

    # Автоматические датасеты
    if sources.get("opus100", True):
        all_pairs.extend(download_opus100(limits.get("opus100", 1_000_000)))

    if sources.get("opensubtitles", True):
        all_pairs.extend(download_opensubtitles(limits.get("opensubtitles", 500_000)))

    if sources.get("un_corpus", True):
        all_pairs.extend(download_un_corpus(limits.get("un_corpus", 500_000)))

    if sources.get("tatoeba", True):
        all_pairs.extend(download_tatoeba(limits.get("tatoeba", 50_000)))

    # Свои файлы (JSONL / TMX / XML / CSV)
    custom = get(cfg, "data", "custom_files", default=[])
    if custom:
        console.rule("[bold cyan]Свои файлы")
        for fpath in custom:
            if Path(fpath).exists():
                all_pairs.extend(convert_file(fpath))
            else:
                log.warning(f"  Файл не найден: {fpath}")

    # SRT пары
    srt_zh = get(cfg, "data", "srt_zh_dir", default="")
    srt_ru = get(cfg, "data", "srt_ru_dir", default="")
    if srt_zh and srt_ru and Path(srt_zh).exists() and Path(srt_ru).exists():
        console.rule("[bold cyan]SRT данные")
        all_pairs.extend(collect_srt_pairs(srt_zh, srt_ru))

    # Фильтрация
    console.rule("[bold yellow]Фильтрация")
    all_pairs = filter_pairs(all_pairs, cfg)

    # Лимит
    total_limit = get(cfg, "data", "total_limit", default=0)
    if total_limit and len(all_pairs) > total_limit:
        np.random.seed(42)
        np.random.shuffle(all_pairs)
        all_pairs = all_pairs[:total_limit]
        log.info(f"  Лимит: обрезано до {total_limit:,}")

    # Сплит
    np.random.seed(42)
    np.random.shuffle(all_pairs)
    eval_size = get(cfg, "data", "eval_size", default=2000)
    eval_data = all_pairs[-eval_size:]
    train_data = all_pairs[:-eval_size]

    # Сохранение
    console.rule("[bold green]Сохранение")
    train_path = get(cfg, "data", "train_path", default="data/train.jsonl")
    eval_path = get(cfg, "data", "eval_path", default="data/eval.jsonl")
    save_jsonl(train_data, train_path)
    save_jsonl(eval_data, eval_path)

    # Статистика
    _print_stats(train_data, eval_data)

    return train_data, eval_data


def _print_stats(train: list, eval_data: list):
    table = Table(title="Итого")
    table.add_column("", style="bold")
    table.add_column("Кол-во", justify="right")
    table.add_row("Train", f"{len(train):,}")
    table.add_row("Eval", f"{len(eval_data):,}")
    table.add_row("Всего", f"{len(train) + len(eval_data):,}")
    if train:
        avg_zh = sum(len(p["zh"]) for p in train) / len(train)
        avg_ru = sum(len(p["ru"]) for p in train) / len(train)
        table.add_row("Ср. длина ZH", f"{avg_zh:.0f} символов")
        table.add_row("Ср. длина RU", f"{avg_ru:.0f} символов")
    console.print(table)


# ═══════════════════════════════════════════════════════════
# Обучение
# ═══════════════════════════════════════════════════════════

def train(cfg: dict):
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer,
        Seq2SeqTrainingArguments, Seq2SeqTrainer,
        DataCollatorForSeq2Seq, EarlyStoppingCallback,
    )
    from datasets import Dataset
    import sacrebleu

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    gen_cfg = cfg.get("generation", {})
    output_dir = cfg.get("output_dir", "output")

    model_name = model_cfg.get("name", "facebook/nllb-200-distilled-600M")
    src_lang = model_cfg.get("src_lang", "zho_Hans")
    tgt_lang = model_cfg.get("tgt_lang", "rus_Cyrl")

    if torch.cuda.is_available():
        g = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {g.name} ({g.total_memory / 1e9:.1f} GB)")

    # --- Данные ---
    train_path = get(cfg, "data", "train_path", default="data/train.jsonl")
    eval_path = get(cfg, "data", "eval_path", default="data/eval.jsonl")

    if not Path(train_path).exists():
        log.info("Данные не найдены — скачиваю...")
        download_all(cfg)

    train_data = load_jsonl(train_path)
    eval_data = load_jsonl(eval_path) if Path(eval_path).exists() else train_data[-2000:]
    log.info(f"Train: {len(train_data):,} | Eval: {len(eval_data):,}")

    # --- Модель ---
    log.info(f"Модель: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    use_fp16 = train_cfg.get("fp16", True) and torch.cuda.is_available()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if use_fp16 else torch.float32
    )

    if train_cfg.get("use_lora", False):
        from peft import LoraConfig, get_peft_model, TaskType
        lora = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=train_cfg.get("lora_r", 16),
            lora_alpha=train_cfg.get("lora_alpha", 32),
            lora_dropout=train_cfg.get("lora_dropout", 0.05),
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    max_src = train_cfg.get("max_src_length", 128)
    max_tgt = train_cfg.get("max_tgt_length", 192)

    def tokenize(examples):
        tokenizer.src_lang = src_lang
        inp = tokenizer(examples["zh"], max_length=max_src, truncation=True, padding="max_length")
        tokenizer.src_lang = tgt_lang
        lbl = tokenizer(examples["ru"], max_length=max_tgt, truncation=True, padding="max_length")
        tokenizer.src_lang = src_lang
        inp["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in l]
            for l in lbl["input_ids"]
        ]
        return inp

    tokenizer.src_lang = src_lang
    train_ds = Dataset.from_list(train_data).map(tokenize, batched=True, batch_size=1000, remove_columns=["zh", "ru"], desc="Tokenize train")
    eval_ds = Dataset.from_list(eval_data).map(tokenize, batched=True, batch_size=1000, remove_columns=["zh", "ru"], desc="Tokenize eval")

    def compute_metrics(pred):
        preds, labels = pred.predictions, pred.label_ids
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        d_p = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        d_l = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        return {
            "bleu": round(sacrebleu.corpus_bleu(d_p, [d_l]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(d_p, [d_l]).score, 2),
        }

    bs = train_cfg.get("batch_size", 8)
    ga = train_cfg.get("gradient_accumulation", 4)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("epochs", 3),
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        gradient_accumulation_steps=ga,
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.06),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        fp16=use_fp16,
        eval_strategy="steps",
        eval_steps=eval_cfg.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=eval_cfg.get("save_steps", 500),
        logging_steps=eval_cfg.get("logging_steps", 100),
        predict_with_generate=True,
        generation_max_length=max_tgt,
        generation_num_beams=gen_cfg.get("num_beams", 5),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        load_best_model_at_end=True,
        metric_for_best_model=eval_cfg.get("metric", "chrf"),
        greater_is_better=True,
        save_total_limit=3,
        report_to="none",
        seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=eval_cfg.get("early_stopping_patience", 3))],
    )

    console.rule("[bold green]Обучение")
    log.info(f"  Модель:  {model_name}")
    log.info(f"  Данные:  {len(train_data):,} train / {len(eval_data):,} eval")
    log.info(f"  Batch:   {bs} × {ga} = {bs * ga}")
    log.info(f"  Эпохи:   {train_cfg.get('epochs', 3)}")

    result = trainer.train()
    log.info(f"  Loss: {result.metrics['train_loss']:.4f} | Время: {result.metrics['train_runtime']:.0f}с")

    final = os.path.join(output_dir, "final")
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    log.info(f"  Модель сохранена: {final}")

    metrics = trainer.evaluate()
    log.info(f"  BLEU: {metrics.get('eval_bleu', '?')} | chrF: {metrics.get('eval_chrf', '?')}")
    return final


# ═══════════════════════════════════════════════════════════
# Инференс
# ═══════════════════════════════════════════════════════════

def _load_model(model_path: str, cfg: dict):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    mc = cfg.get("model", {})
    tok = AutoTokenizer.from_pretrained(model_path)
    m = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(dev).eval()
    tok.src_lang = mc.get("src_lang", "zho_Hans")
    tid = tok.convert_tokens_to_ids(mc.get("tgt_lang", "rus_Cyrl"))
    return m, tok, dev, tid


def _translate(text, m, tok, dev, tid, cfg):
    gc = cfg.get("generation", {})
    inp = tok(text, return_tensors="pt", max_length=256, truncation=True).to(dev)
    with torch.no_grad():
        out = m.generate(**inp, forced_bos_token_id=tid,
                         max_length=gc.get("max_length", 256),
                         num_beams=gc.get("num_beams", 5))
    return tok.decode(out[0], skip_special_tokens=True)


def cmd_translate(text: str, cfg: dict):
    mp = os.path.join(cfg.get("output_dir", "output"), "final")
    m, tok, dev, tid = _load_model(mp, cfg)
    print(_translate(text, m, tok, dev, tid, cfg))


def cmd_srt(inp: str, out: str, cfg: dict):
    mp = os.path.join(cfg.get("output_dir", "output"), "final")
    m, tok, dev, tid = _load_model(mp, cfg)
    entries = parse_srt(inp)
    log.info(f"Перевожу {len(entries)} субтитров...")
    for e in tqdm(entries, desc="Перевод"):
        e["text"] = _translate(e["text"], m, tok, dev, tid, cfg)
    write_srt(entries, out)
    log.info(f"Сохранено: {out}")


def cmd_compare(cfg: dict):
    mp = os.path.join(cfg.get("output_dir", "output"), "final")
    m, tok, dev, tid = _load_model(mp, cfg)
    tests = ["你好，欢迎来到我的频道", "今天天气很好", "人工智能改变了我们的生活",
             "这个方法比传统方法快三倍", "感谢大家的观看，我们下期再见"]
    try:
        from transformers import MarianMTModel, MarianTokenizer
        mt = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-ru")
        mm = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-ru").to(dev).eval()
        has_m = True
    except Exception:
        has_m = False

    table = Table(title="MarianMT vs Fine-tuned NLLB")
    table.add_column("ZH", style="cyan", max_width=30)
    if has_m:
        table.add_column("MarianMT", style="red", max_width=35)
    table.add_column("NLLB (tuned)", style="green", max_width=35)

    for zh in tests:
        row = [zh]
        if has_m:
            inp = mt(zh, return_tensors="pt", truncation=True).to(dev)
            with torch.no_grad():
                o = mm.generate(**inp, num_beams=5)
            row.append(mt.decode(o[0], skip_special_tokens=True))
        row.append(_translate(zh, m, tok, dev, tid, cfg))
        table.add_row(*row)

    console.print(table)


def cmd_interactive(cfg: dict):
    mp = os.path.join(cfg.get("output_dir", "output"), "final")
    m, tok, dev, tid = _load_model(mp, cfg)
    log.info("Вводи текст на китайском (q — выход):")
    while True:
        text = input("\nZH> ").strip()
        if text.lower() in ("q", "quit", "exit"):
            break
        if text:
            print(f"RU> {_translate(text, m, tok, dev, tid, cfg)}")


def cmd_info(cfg: dict):
    train_path = get(cfg, "data", "train_path", default="data/train.jsonl")
    eval_path = get(cfg, "data", "eval_path", default="data/eval.jsonl")
    if Path(train_path).exists():
        train_data = load_jsonl(train_path)
        eval_data = load_jsonl(eval_path) if Path(eval_path).exists() else []
        _print_stats(train_data, eval_data)
    else:
        log.info("Данных нет. Запусти: python train.py download")


def cmd_convert(path: str):
    pairs = convert_file(path)
    out = Path(path).stem + ".jsonl"
    save_jsonl(pairs, out)


def _apply_gpu_profile(cfg: dict, gpu_flag: str) -> dict:
    """Применяет GPU профиль поверх training конфига."""
    profiles = cfg.get("gpu_profiles", {})

    # Автоопределение GPU
    if gpu_flag == "auto":
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram < 6:
                gpu_flag = "rtx3050"
            else:
                gpu_flag = "rtx4070"
            log.info(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.0f} GB) → профиль {gpu_flag}")
        else:
            gpu_flag = "cpu"
            log.info("GPU не найден → профиль cpu")

    if gpu_flag not in profiles:
        log.warning(f"Профиль '{gpu_flag}' не найден в config.yaml. Доступны: {list(profiles.keys())}")
        return cfg

    profile = profiles[gpu_flag]
    training = cfg.get("training", {})

    # Мержим: профиль перезаписывает training
    for key, value in profile.items():
        training[key] = value

    cfg["training"] = training

    # Красивый вывод
    table = Table(title=f"GPU профиль: {gpu_flag}")
    table.add_column("Параметр", style="cyan")
    table.add_column("Значение", style="green")
    table.add_row("batch_size", str(training.get("batch_size")))
    table.add_row("gradient_accumulation", str(training.get("gradient_accumulation")))
    eff = training.get("batch_size", 1) * training.get("gradient_accumulation", 1)
    table.add_row("effective batch", str(eff))
    table.add_row("fp16", str(training.get("fp16")))
    table.add_row("LoRA", str(training.get("use_lora")))
    table.add_row("max_src_length", str(training.get("max_src_length")))
    table.add_row("max_tgt_length", str(training.get("max_tgt_length")))
    console.print(table)

    return cfg


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="NLLB Fine-tune: перевод zh→ru",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python train.py                              скачать + обучить
  python train.py download                     только скачать
  python train.py train                        только обучить
  python train.py translate "你好世界"          перевести текст
  python train.py compare                      сравнить с MarianMT
  python train.py interactive                  интерактивный режим
  python train.py srt input.srt output.srt     перевести субтитры
  python train.py convert file.tmx             XML/TMX → JSONL
  python train.py info                         статистика данных

GPU профили:
  python train.py --gpu rtx4070                RTX 4070 Super (12 GB) — дефолт
  python train.py --gpu rtx3050                RTX 3050 Mobile (4 GB) — LoRA
  python train.py --gpu cpu                    Без GPU (тест)
""",
    )
    p.add_argument("command", nargs="?", default="all",
                   choices=["all", "download", "train", "translate", "compare",
                            "interactive", "srt", "convert", "info"],
                   help="Команда")
    p.add_argument("args", nargs="*", help="Аргументы команды")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model-path", default=None)
    p.add_argument("--gpu", default="auto",
                   help="GPU профиль: rtx4070 / rtx3050 / cpu / auto (дефолт)")

    args = p.parse_args()
    cfg = load_config(args.config)

    # Применяем GPU профиль
    cfg = _apply_gpu_profile(cfg, args.gpu)

    if args.command == "download":
        download_all(cfg)

    elif args.command == "train":
        train(cfg)

    elif args.command == "all":
        train(cfg)  # автоматически скачает если нет данных

    elif args.command == "translate":
        if not args.args:
            log.error('Укажи текст: python train.py translate "你好"')
            sys.exit(1)
        cmd_translate(args.args[0], cfg)

    elif args.command == "compare":
        cmd_compare(cfg)

    elif args.command == "interactive":
        cmd_interactive(cfg)

    elif args.command == "srt":
        if len(args.args) < 2:
            log.error("python train.py srt input.srt output.srt")
            sys.exit(1)
        cmd_srt(args.args[0], args.args[1], cfg)

    elif args.command == "convert":
        if not args.args:
            log.error("python train.py convert file.tmx")
            sys.exit(1)
        cmd_convert(args.args[0])

    elif args.command == "info":
        cmd_info(cfg)


if __name__ == "__main__":
    main()
