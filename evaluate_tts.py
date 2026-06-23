"""
=============================================================================
Evaluation Pipeline: Piper TTS  —  en_US-lessac-medium
English Text-to-Speech Benchmark (final stage of the OneVoice vi→en→speech
pipeline). The spoken text is the ENGLISH MT output (`hypothesis`) produced by
the opus-mt-vi-en stage and stored in results/translations.jsonl.
=============================================================================
Metrics:
  - Intelligibility : WER (via Whisper ASR re-transcription, language=en)
                      reference = the MT hypothesis text actually fed to TTS
  - Naturalness     : DNSMOS (non-intrusive predicted MOS: OVRL/SIG/BAK/P808)
  - Performance     : Synthesis latency, Real-Time Factor (RTF), audio dur,
                      throughput, model size, peak RAM, CPU load
Note: Intrusive metrics (PESQ/STOI/MCD/Speaker-sim) require ground-truth
      reference recordings, which a self-made *text* benchmark does not have,
      and are therefore out of scope here.
=============================================================================
"""

import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import gc
import time
import wave
import platform
import statistics
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import librosa

from piper import PiperVoice
import jiwer
from faster_whisper import WhisperModel
from speechmos import dnsmos

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
VOICE_PATH   = Path("models/en_US-lessac-medium.onnx")
VOICE_NAME   = "en_US-lessac-medium"
# Spoken text = English MT output (hypothesis) from the opus-mt-vi-en stage.
BENCHMARK    = Path("results/translations.jsonl")
TEXT_FIELD   = "hypothesis"       # the English text the TTS must speak
OUTPUT_DIR   = Path("results")
AUDIO_DIR    = OUTPUT_DIR / "audio"
ASR_MODEL    = "small"            # faster-whisper, English
ASR_LANG     = "en"
ASR_BEAM     = 5
WARMUP_RUNS  = 3
SR_SYNTH     = 22050              # Piper output sample rate (from voice config)

OUTPUT_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_file_size_mb(path) -> float:
    try:
        return os.path.getsize(path) / 1024 / 1024
    except Exception:
        return 0.0


class SystemMonitor:
    """Polls CPU / RAM every 500 ms in a background thread."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self._cpu_samples = []
        self._ram_samples = []
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            self._cpu_samples.append(psutil.cpu_percent(interval=None))
            self._ram_samples.append(psutil.virtual_memory().used / 1024 / 1024)
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self):
        def safe(lst):
            if not lst:
                return {"mean": 0, "max": 0, "min": 0}
            return {"mean": round(statistics.mean(lst), 2),
                    "max":  round(max(lst), 2),
                    "min":  round(min(lst), 2)}
        return {"cpu_percent": safe(self._cpu_samples),
                "ram_used_mb": safe(self._ram_samples)}


# ── WER normalisation ─────────────────────────────────────────────────────────
# Lowercase, strip punctuation (keep Vietnamese letters + digits), collapse ws.
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def wer_pair(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0
    if not h:
        return 1.0
    return jiwer.wer(r, h)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_benchmark():
    """Load the English MT hypotheses (translations.jsonl) as TTS inputs."""
    data = []
    with open(BENCHMARK, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            data.append({"id": row["id"], "domain": row["domain"],
                         "text": row[TEXT_FIELD]})
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Synthesis
# ─────────────────────────────────────────────────────────────────────────────

def synthesize(voice, text, wav_path):
    """Synthesize text to a WAV file; return (synth_time_s, audio_dur_s)."""
    t0 = time.perf_counter()
    with wave.open(str(wav_path), "wb") as wf:
        voice.synthesize_wav(text, wf)
    synth_time = time.perf_counter() - t0
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        audio_dur = frames / float(rate)
    return synth_time, audio_dur


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"  Piper TTS  ({VOICE_NAME})  –  Benchmark Evaluation")
    print(f"  Device : CPU  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading benchmark data …")
    data = load_benchmark()
    domains = sorted({d["domain"] for d in data})
    print(f"      Loaded {len(data)} sentences across {len(domains)} domains.")

    # ── 2. Load TTS voice ─────────────────────────────────────────────────────
    print(f"\n[2/7] Loading Piper voice: {VOICE_NAME} …")
    ram_before = psutil.virtual_memory().used / 1024 / 1024
    voice = PiperVoice.load(str(VOICE_PATH))
    ram_after = psutil.virtual_memory().used / 1024 / 1024
    model_ram_mb = ram_after - ram_before
    model_size_mb = get_file_size_mb(VOICE_PATH)
    print(f"      Model size : {model_size_mb:.1f} MB (onnx)")
    print(f"      RAM delta  : {model_ram_mb:.1f} MB")

    # ── 3. Warm-up ────────────────────────────────────────────────────────────
    print(f"\n[3/7] Warming up ({WARMUP_RUNS} runs) …")
    tmp = AUDIO_DIR / "_warmup.wav"
    for _ in range(WARMUP_RUNS):
        synthesize(voice, data[0]["text"], tmp)
    if tmp.exists():
        tmp.unlink()
    print("      Done.")

    # ── 4. Synthesize all sentences (with monitoring) ─────────────────────────
    print("\n[4/7] Synthesizing benchmark sentences …")
    monitor = SystemMonitor(interval=0.5)
    monitor.start()
    gc.collect()

    rows = []
    synth_times, audio_durs, rtfs = [], [], []
    t_start = time.perf_counter()
    for i, d in enumerate(data):
        wav_path = AUDIO_DIR / f"{d['id']:03d}_{d['domain']}.wav"
        st, dur = synthesize(voice, d["text"], wav_path)
        rtf = st / dur if dur > 0 else 0.0
        synth_times.append(st)
        audio_durs.append(dur)
        rtfs.append(rtf)
        rows.append({"id": d["id"], "domain": d["domain"], "text": d["text"],
                     "wav": str(wav_path), "synth_time_s": round(st, 4),
                     "audio_dur_s": round(dur, 4), "rtf": round(rtf, 4)})
        if (i + 1) % 20 == 0 or i + 1 == len(data):
            print(f"      Synthesized {i + 1}/{len(data)} …")
    total_synth_time = time.perf_counter() - t_start

    monitor.stop()
    sys_stats = monitor.summary()

    total_audio = sum(audio_durs)
    avg_latency_ms = statistics.mean(synth_times) * 1000
    median_latency_ms = statistics.median(synth_times) * 1000
    p95_latency_ms = float(np.percentile(synth_times, 95)) * 1000
    overall_rtf = total_synth_time / total_audio if total_audio > 0 else 0.0
    throughput_sps = len(data) / total_synth_time
    audio_per_s = total_audio / total_synth_time   # seconds of audio per second

    print(f"      Total synth time : {total_synth_time:.1f}s for {total_audio:.1f}s audio")
    print(f"      Overall RTF      : {overall_rtf:.4f}  ({audio_per_s:.1f}x real-time)")

    # ── 5. ASR re-transcription → WER ─────────────────────────────────────────
    print(f"\n[5/7] Loading Whisper ASR ({ASR_MODEL}) and transcribing …")
    asr = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
    asr_total_time = 0.0
    for r in rows:
        t0 = time.perf_counter()
        segs, _ = asr.transcribe(r["wav"], language=ASR_LANG, beam_size=ASR_BEAM)
        hyp = "".join(s.text for s in segs).strip()
        asr_total_time += time.perf_counter() - t0
        r["asr_transcript"] = hyp
        r["wer"] = round(wer_pair(r["text"], hyp), 4)
    wer_scores = [r["wer"] for r in rows]
    mean_wer = statistics.mean(wer_scores) * 100
    median_wer = statistics.median(wer_scores) * 100
    print(f"      Mean WER : {mean_wer:.2f}%   (ASR took {asr_total_time:.1f}s)")

    # ── 6. DNSMOS (non-intrusive predicted MOS) ───────────────────────────────
    print("\n[6/7] Computing DNSMOS (non-intrusive predicted MOS) …")
    for r in rows:
        wav, _ = librosa.load(r["wav"], sr=16000, mono=True)
        wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
        m = dnsmos.run(wav, sr=16000)
        r["dnsmos_ovrl"] = round(float(m["ovrl_mos"]), 4)
        r["dnsmos_sig"]  = round(float(m["sig_mos"]), 4)
        r["dnsmos_bak"]  = round(float(m["bak_mos"]), 4)
        r["dnsmos_p808"] = round(float(m["p808_mos"]), 4)
    ovrl = statistics.mean(r["dnsmos_ovrl"] for r in rows)
    sig  = statistics.mean(r["dnsmos_sig"]  for r in rows)
    bak  = statistics.mean(r["dnsmos_bak"]  for r in rows)
    p808 = statistics.mean(r["dnsmos_p808"] for r in rows)
    print(f"      DNSMOS OVRL={ovrl:.3f}  SIG={sig:.3f}  BAK={bak:.3f}  P808={p808:.3f}")

    # ── Per-domain breakdown ──────────────────────────────────────────────────
    breakdown = {}
    for dom in domains:
        idx = [r for r in rows if r["domain"] == dom]
        breakdown[dom] = {
            "n": len(idx),
            "wer_%": round(statistics.mean(x["wer"] for x in idx) * 100, 2),
            "dnsmos_ovrl": round(statistics.mean(x["dnsmos_ovrl"] for x in idx), 3),
            "rtf": round(statistics.mean(x["rtf"] for x in idx), 4),
        }

    # ── 7. Assemble + save ────────────────────────────────────────────────────
    print("\n[7/7] Saving outputs …")
    results = {
        "metadata": {
            "voice":          VOICE_NAME,
            "engine":         "Piper TTS (onnxruntime)",
            "task":           "English text-to-speech (speaks opus-mt-vi-en output)",
            "tts_input":      f"{TEXT_FIELD} (English MT output)",
            "wer_reference":  "MT hypothesis text fed to TTS",
            "benchmark":      str(BENCHMARK),
            "n_sentences":    len(data),
            "n_domains":      len(domains),
            "device":         "CPU",
            "asr_model":      f"faster-whisper-{ASR_MODEL}",
            "dnsmos":         "speechmos DNSMOS (P.808/P.835)",
            "sample_rate":    SR_SYNTH,
            "timestamp":      datetime.now().isoformat(),
            "python_version": sys.version,
            "platform":       platform.platform(),
        },
        "intelligibility": {
            "WER_mean_%":   round(mean_wer, 4),
            "WER_median_%": round(median_wer, 4),
            "asr_model":    f"faster-whisper-{ASR_MODEL}",
        },
        "naturalness_dnsmos": {
            "OVRL_mos": round(ovrl, 4),
            "SIG_mos":  round(sig, 4),
            "BAK_mos":  round(bak, 4),
            "P808_mos": round(p808, 4),
        },
        "performance_metrics": {
            "model_size_mb":          round(model_size_mb, 2),
            "ram_delta_on_load_mb":   round(model_ram_mb, 2),
            "peak_ram_used_mb":       sys_stats["ram_used_mb"]["max"],
            "total_synth_time_s":     round(total_synth_time, 3),
            "total_audio_s":          round(total_audio, 3),
            "overall_rtf":            round(overall_rtf, 4),
            "x_real_time":            round(audio_per_s, 2),
            "avg_latency_ms":         round(avg_latency_ms, 3),
            "median_latency_ms":      round(median_latency_ms, 3),
            "p95_latency_ms":         round(p95_latency_ms, 3),
            "throughput_sentences_per_s": round(throughput_sps, 3),
            "system_monitor":         sys_stats,
        },
        "per_domain": breakdown,
    }

    with open(OUTPUT_DIR / "tts_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "tts_synthesis.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TTS EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Voice          : {VOICE_NAME}")
    print(f"  Sentences      : {len(data)}  |  Audio: {total_audio:.1f}s")
    print()
    print("  ── Intelligibility ─────────────────────────────────────────────")
    print(f"  WER (mean)     : {mean_wer:.2f}%   (ASR: faster-whisper-{ASR_MODEL})")
    print(f"  WER (median)   : {median_wer:.2f}%")
    print()
    print("  ── Naturalness (DNSMOS, non-intrusive MOS) ─────────────────────")
    print(f"  OVRL / SIG / BAK / P808 : {ovrl:.3f} / {sig:.3f} / {bak:.3f} / {p808:.3f}")
    print()
    print("  ── Performance ─────────────────────────────────────────────────")
    print(f"  Model size     : {model_size_mb:.1f} MB")
    print(f"  Overall RTF    : {overall_rtf:.4f}  ({audio_per_s:.1f}x real-time)")
    print(f"  Avg latency    : {avg_latency_ms:.1f} ms/sentence")
    print(f"  p95 latency    : {p95_latency_ms:.1f} ms/sentence")
    print(f"  Throughput     : {throughput_sps:.2f} sentences/s")
    print(f"  CPU (mean/max) : {sys_stats['cpu_percent']['mean']}% / {sys_stats['cpu_percent']['max']}%")
    print()
    print("  ── Per-domain (WER% / DNSMOS-OVRL / RTF) ───────────────────────")
    for dom, v in breakdown.items():
        print(f"  {dom:<14} WER={v['wer_%']:6.2f}%  OVRL={v['dnsmos_ovrl']:.3f}  RTF={v['rtf']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
