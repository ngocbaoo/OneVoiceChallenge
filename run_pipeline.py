"""
=============================================================================
OneVoice End-to-End Pipeline Runner & Profiler
  vi (text) ─► opus-mt-vi-en (MT) ─► en (text) ─► Piper TTS ─► English speech
=============================================================================
Mục tiêu: triển khai tối ưu #1 (STREAMING + PIPELINING giữa hai tầng) và ĐO
TRỰC TIẾP các chỉ số pipeline (latency, CPU, RAM, throughput, time-to-first-
audio) — thay vì suy luận bằng cách cộng số liệu hai tầng đo rời.

Chạy 2 chế độ trên cùng máy/cùng run để so sánh công bằng:
  A. SEQUENTIAL  : dịch xong 1 câu → tổng hợp 1 câu → câu kế (không gối tầng)
  B. PIPELINED   : MT (producer) và TTS (consumer) chạy SONG SONG qua hàng đợi
                   → TTS xử lý câu i trong khi MT đang dịch câu i+1

Cả hai dùng cùng granularity (1 câu/lần) để cô lập đúng lợi ích của pipelining.
ASR/DNSMOS KHÔNG nằm trong pipeline production nên không đo ở đây.
=============================================================================
"""

import json
import os
import sys
import gc
import time
import wave
import queue
import platform
import statistics
import threading
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import psutil

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from piper import PiperVoice

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MT_MODEL    = "Helsinki-NLP/opus-mt-vi-en"
VOICE_PATH  = Path("models/en_US-lessac-medium.onnx")
BENCHMARK   = Path("benchmark/vi_en_benchmark.json")
OUTPUT_DIR  = Path("results")
AUDIO_DIR   = OUTPUT_DIR / "audio_pipeline"
MAX_NEW_TOKENS = 256
WARMUP_RUNS = 3
DEVICE = "cpu"   # đánh giá CPU-only

OUTPUT_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

_PROC = psutil.Process(os.getpid())


# ─────────────────────────────────────────────────────────────────────────────
# System monitor (CPU system-wide + RAM của tiến trình này)
# ─────────────────────────────────────────────────────────────────────────────
class SystemMonitor:
    def __init__(self, interval=0.25):
        self.interval = interval
        self._cpu, self._rss = [], []
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        psutil.cpu_percent(interval=None)   # prime
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            self._cpu.append(psutil.cpu_percent(interval=None))
            self._rss.append(_PROC.memory_info().rss / 1024 / 1024)
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self):
        def s(lst):
            if not lst:
                return {"mean": 0, "max": 0, "min": 0}
            return {"mean": round(statistics.mean(lst), 2),
                    "max": round(max(lst), 2), "min": round(min(lst), 2)}
        return {"cpu_percent": s(self._cpu), "proc_rss_mb": s(self._rss)}


# ─────────────────────────────────────────────────────────────────────────────
# Stage primitives
# ─────────────────────────────────────────────────────────────────────────────
def translate_one(model, tokenizer, text):
    inputs = tokenizer([text], return_tensors="pt", padding=True,
                       truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def synth_one(voice, text, wav_path):
    with wave.open(str(wav_path), "wb") as wf:
        voice.synthesize_wav(text, wf)
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


# ─────────────────────────────────────────────────────────────────────────────
# Mode A — SEQUENTIAL (no cross-stage overlap)
# ─────────────────────────────────────────────────────────────────────────────
def run_sequential(model, tokenizer, voice, sources, ids, domains):
    mon = SystemMonitor(); mon.start(); gc.collect()
    t0 = time.perf_counter()
    mt_lat, tts_lat, audio_durs = [], [], []
    ttfa = None
    for i, src in enumerate(sources):
        a = time.perf_counter()
        en = translate_one(model, tokenizer, src)
        b = time.perf_counter()
        wav = AUDIO_DIR / f"{ids[i]:03d}_{domains[i]}.wav"
        dur = synth_one(voice, en, wav)
        c = time.perf_counter()
        mt_lat.append((b - a) * 1000)
        tts_lat.append((c - b) * 1000)
        audio_durs.append(dur)
        if ttfa is None:
            ttfa = c - t0                      # first audio ready
    wall = time.perf_counter() - t0
    mon.stop()
    return _pack("sequential", wall, ttfa, mt_lat, tts_lat, audio_durs, mon.summary())


# ─────────────────────────────────────────────────────────────────────────────
# Mode B — PIPELINED (MT producer || TTS consumer)
# ─────────────────────────────────────────────────────────────────────────────
def run_pipelined(model, tokenizer, voice, sources, ids, domains):
    q = queue.Queue(maxsize=8)
    mt_lat, tts_lat, audio_durs = [], [], []
    ttfa = {"t": None}
    mon = SystemMonitor(); mon.start(); gc.collect()
    t0 = time.perf_counter()

    def producer():
        for i, src in enumerate(sources):
            a = time.perf_counter()
            en = translate_one(model, tokenizer, src)
            mt_lat.append((time.perf_counter() - a) * 1000)
            q.put((i, en))
        q.put(None)   # sentinel

    def consumer():
        while True:
            item = q.get()
            if item is None:
                break
            i, en = item
            b = time.perf_counter()
            wav = AUDIO_DIR / f"{ids[i]:03d}_{domains[i]}.wav"
            dur = synth_one(voice, en, wav)
            tts_lat.append((time.perf_counter() - b) * 1000)
            audio_durs.append(dur)
            if ttfa["t"] is None:
                ttfa["t"] = time.perf_counter() - t0

    pt = threading.Thread(target=producer)
    ct = threading.Thread(target=consumer)
    pt.start(); ct.start()
    pt.join(); ct.join()
    wall = time.perf_counter() - t0
    mon.stop()
    return _pack("pipelined", wall, ttfa["t"], mt_lat, tts_lat, audio_durs, mon.summary())


# ─────────────────────────────────────────────────────────────────────────────
def _pack(mode, wall, ttfa, mt_lat, tts_lat, audio_durs, sysstats):
    n = len(mt_lat)
    total_audio = sum(audio_durs)
    return {
        "mode": mode,
        "n_sentences": n,
        "wall_time_s": round(wall, 3),
        "time_to_first_audio_s": round(ttfa, 3) if ttfa else None,
        "throughput_sentences_per_s": round(n / wall, 3),
        "total_audio_s": round(total_audio, 3),
        "end_to_end_rtf": round(wall / total_audio, 4) if total_audio else None,
        "x_real_time": round(total_audio / wall, 2) if wall else None,
        "mt_latency_ms": {
            "mean": round(statistics.mean(mt_lat), 2),
            "median": round(statistics.median(mt_lat), 2),
            "p95": round(float(np.percentile(mt_lat, 95)), 2),
        },
        "tts_latency_ms": {
            "mean": round(statistics.mean(tts_lat), 2),
            "median": round(statistics.median(tts_lat), 2),
            "p95": round(float(np.percentile(tts_lat, 95)), 2),
        },
        "cpu_percent": sysstats["cpu_percent"],
        "proc_rss_mb": sysstats["proc_rss_mb"],
    }


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  OneVoice Pipeline Profiler  —  vi → MT → en → TTS → speech")
    print(f"  Device: {DEVICE.upper()}  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 72)

    with open(BENCHMARK, encoding="utf-8") as f:
        data = json.load(f)
    sources = [d["vi"] for d in data]
    ids     = [d["id"] for d in data]
    domains = [d["domain"] for d in data]
    print(f"\n[1/4] Loaded {len(sources)} Vietnamese sentences.")

    print("\n[2/4] Loading models …")
    rss0 = _PROC.memory_info().rss / 1024 / 1024
    tokenizer = AutoTokenizer.from_pretrained(MT_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL).to(DEVICE).eval()
    voice = PiperVoice.load(str(VOICE_PATH))
    rss1 = _PROC.memory_info().rss / 1024 / 1024
    print(f"      Both models resident — process RSS delta: {rss1 - rss0:.1f} MB")

    print(f"\n[3/4] Warm-up ({WARMUP_RUNS}) …")
    for _ in range(WARMUP_RUNS):
        en = translate_one(model, tokenizer, sources[0])
        synth_one(voice, en, AUDIO_DIR / "_warmup.wav")
    (AUDIO_DIR / "_warmup.wav").unlink(missing_ok=True)
    print("      Done.")

    print("\n[4/4] Profiling …")
    print("      → Mode A: SEQUENTIAL (no overlap) …")
    seq = run_sequential(model, tokenizer, voice, sources, ids, domains)
    print(f"        wall={seq['wall_time_s']}s  TTFA={seq['time_to_first_audio_s']}s  "
          f"thr={seq['throughput_sentences_per_s']}/s  CPU={seq['cpu_percent']['mean']}%")

    print("      → Mode B: PIPELINED (MT || TTS) …")
    pipe = run_pipelined(model, tokenizer, voice, sources, ids, domains)
    print(f"        wall={pipe['wall_time_s']}s  TTFA={pipe['time_to_first_audio_s']}s  "
          f"thr={pipe['throughput_sentences_per_s']}/s  CPU={pipe['cpu_percent']['mean']}%")

    speedup = round(seq["wall_time_s"] / pipe["wall_time_s"], 3)
    wall_saved = round(seq["wall_time_s"] - pipe["wall_time_s"], 2)
    ttfa_seq = seq["time_to_first_audio_s"] or 0
    ttfa_pipe = pipe["time_to_first_audio_s"] or 0

    results = {
        "metadata": {
            "pipeline": "vi → opus-mt-vi-en → en → Piper en_US-lessac-medium → speech",
            "optimization": "#1 streaming + cross-stage pipelining",
            "device": DEVICE.upper(),
            "n_sentences": len(sources),
            "both_models_resident_rss_delta_mb": round(rss1 - rss0, 2),
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "sequential": seq,
        "pipelined": pipe,
        "comparison": {
            "wall_speedup_x": speedup,
            "wall_time_saved_s": wall_saved,
            "throughput_gain_x": round(
                pipe["throughput_sentences_per_s"] / seq["throughput_sentences_per_s"], 3),
            "cpu_mean_delta_pct": round(
                pipe["cpu_percent"]["mean"] - seq["cpu_percent"]["mean"], 2),
        },
    }
    with open(OUTPUT_DIR / "pipeline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("  PIPELINE PROFILE SUMMARY")
    print("=" * 72)
    print(f"  Sentences            : {len(sources)}   |  total audio: {pipe['total_audio_s']}s")
    print(f"  Both models RSS      : {rss1 - rss0:.1f} MB (resident)")
    print("  ─────────────────────────────────────────────────────────────────")
    print(f"  {'Metric':<26}{'SEQUENTIAL':>16}{'PIPELINED':>16}")
    print(f"  {'wall time (s)':<26}{seq['wall_time_s']:>16}{pipe['wall_time_s']:>16}")
    print(f"  {'time-to-first-audio (s)':<26}{ttfa_seq:>16}{ttfa_pipe:>16}")
    print(f"  {'throughput (sent/s)':<26}{seq['throughput_sentences_per_s']:>16}{pipe['throughput_sentences_per_s']:>16}")
    print(f"  {'end-to-end RTF':<26}{seq['end_to_end_rtf']:>16}{pipe['end_to_end_rtf']:>16}")
    print(f"  {'CPU mean %':<26}{seq['cpu_percent']['mean']:>16}{pipe['cpu_percent']['mean']:>16}")
    print(f"  {'CPU max %':<26}{seq['cpu_percent']['max']:>16}{pipe['cpu_percent']['max']:>16}")
    print(f"  {'proc RSS max (MB)':<26}{seq['proc_rss_mb']['max']:>16}{pipe['proc_rss_mb']['max']:>16}")
    print("  ─────────────────────────────────────────────────────────────────")
    print(f"  Wall speedup         : {speedup}×   (saved {wall_saved}s)")
    print(f"  TTFA reduction       : {ttfa_seq:.3f}s → {ttfa_pipe:.3f}s")
    print("=" * 72)
    print(f"  Saved → {(OUTPUT_DIR / 'pipeline_results.json').resolve()}")


if __name__ == "__main__":
    main()
