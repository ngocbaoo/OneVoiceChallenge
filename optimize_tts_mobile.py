"""
Mobile optimization — Stage 2: TTS (Piper en_US-lessac-medium)
  Baseline : ONNX fp32 (current)
  Optimized: ONNX INT8 (dynamic quantization via onnxruntime)
Measures: on-disk size, process RSS, per-sentence latency, RTF, and a WER
sanity check on a subset (faster-whisper) to confirm intelligibility holds.
"""
import json, os, sys, time, gc, wave, statistics, shutil, re
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import psutil
from onnxruntime.quantization import quantize_dynamic, QuantType
from piper import PiperVoice
import jiwer
from faster_whisper import WhisperModel

FP32 = Path("models/en_US-lessac-medium.onnx")
INT8 = Path("models/en_US-lessac-medium.int8.onnx")
CFG  = Path("models/en_US-lessac-medium.onnx.json")
TRANS = Path("results/translations.jsonl")
AUDIO = Path("results/audio_mobile"); AUDIO.mkdir(parents=True, exist_ok=True)
OUT = Path("results"); OUT.mkdir(exist_ok=True)
WER_SUBSET = 30          # sentences for WER sanity check
PROC = psutil.Process(os.getpid())

def size_mb(p): return os.path.getsize(p)/1024/1024
def rss(): return PROC.memory_info().rss/1024/1024

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE); _WS = re.compile(r"\s+")
def norm(t): return _WS.sub(" ", _PUNCT.sub(" ", t.lower().strip())).strip()

# ── data ──
rows = [json.loads(l) for l in open(TRANS, encoding="utf-8")]
texts = [r["hypothesis"] for r in rows]
ids   = [r["id"] for r in rows]
print(f"[data] {len(texts)} English sentences (MT hypotheses)")

# ── quantize fp32 -> int8 ──
if not INT8.exists():
    print("[quantize] dynamic INT8 quantization of Piper ONNX ...")
    quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QInt8)
    shutil.copyfile(CFG, str(INT8)+".json")  # Piper needs matching <model>.json
print(f"[quantize] fp32={size_mb(FP32):.1f} MB  int8={size_mb(INT8):.1f} MB")

def synth(voice, text, path):
    t0 = time.perf_counter()
    with wave.open(str(path), "wb") as wf:
        voice.synthesize_wav(text, wf)
    st = time.perf_counter()-t0
    with wave.open(str(path), "rb") as wf:
        dur = wf.getnframes()/float(wf.getframerate())
    return st, dur

def bench(model_path, tag):
    r0 = rss()
    voice = PiperVoice.load(str(model_path))
    r1 = rss()
    for _ in range(3): synth(voice, texts[0], AUDIO/"_w.wav")  # warmup
    lat, durs = [], []
    gc.collect()
    for i, t in enumerate(texts):
        st, dur = synth(voice, t, AUDIO/f"{tag}_{ids[i]:03d}.wav")
        lat.append(st*1000); durs.append(dur)
    total_audio = sum(durs); total_synth = sum(lat)/1000
    return {"variant":tag,"size_mb":round(size_mb(model_path),1),"ram_load_mb":round(r1-r0,1),
            "lat_mean_ms":round(statistics.mean(lat),1),"lat_p95_ms":round(float(np.percentile(lat,95)),1),
            "rtf":round(total_synth/total_audio,4),"x_real_time":round(total_audio/total_synth,2)}

print("\n[bench] fp32 ...");  base = bench(FP32, "fp32"); print("  ", base)
print("\n[bench] int8 ...");  opt  = bench(INT8, "int8"); print("  ", opt)

# ── WER sanity check on subset (both variants) ──
print(f"\n[wer] re-transcribing {WER_SUBSET} sentences with faster-whisper ...")
asr = WhisperModel("small", device="cpu", compute_type="int8")
def wer_for(tag):
    ws = []
    for i in range(WER_SUBSET):
        segs,_ = asr.transcribe(str(AUDIO/f"{tag}_{ids[i]:03d}.wav"), language="en", beam_size=5)
        hyp = "".join(s.text for s in segs).strip()
        r = norm(texts[i]); h = norm(hyp)
        ws.append(jiwer.wer(r,h) if r else 0.0)
    return round(statistics.mean(ws)*100,2)
wer_fp32 = wer_for("fp32"); wer_int8 = wer_for("int8")
print(f"  WER fp32={wer_fp32}%  int8={wer_int8}%")

result = {
    "stage":"TTS (Piper en_US-lessac-medium)",
    "baseline": {**base, "wer_subset_%": wer_fp32},
    "optimized": {**opt, "wer_subset_%": wer_int8},
    "wer_subset_n": WER_SUBSET,
    "deltas": {
        "size_reduction_x": round(base["size_mb"]/opt["size_mb"],2),
        "speedup_x": round(base["lat_mean_ms"]/opt["lat_mean_ms"],2),
        "wer_delta_pp": round(wer_int8-wer_fp32,2),
    },
}
json.dump(result, open(OUT/"mobile_tts_results.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

print("\n"+"="*64)
print("  TTS MOBILE OPTIMIZATION — ONNX fp32  vs  ONNX int8")
print("="*64)
print(f"  {'':20}{'fp32':>14}{'int8':>14}")
print(f"  {'size (MB)':20}{base['size_mb']:>14}{opt['size_mb']:>14}")
print(f"  {'RAM load (MB)':20}{base['ram_load_mb']:>14}{opt['ram_load_mb']:>14}")
print(f"  {'latency mean (ms)':20}{base['lat_mean_ms']:>14}{opt['lat_mean_ms']:>14}")
print(f"  {'RTF':20}{base['rtf']:>14}{opt['rtf']:>14}")
print(f"  {'WER subset (%)':20}{wer_fp32:>14}{wer_int8:>14}")
print("-"*64)
print(f"  size : {result['deltas']['size_reduction_x']}x smaller")
print(f"  speed: {result['deltas']['speedup_x']}x   WER delta: {result['deltas']['wer_delta_pp']:+} pp")
print("="*64)
