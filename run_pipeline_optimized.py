"""
OneVoice — RECOMMENDED mobile pipeline, end-to-end measurement
  MT : CTranslate2 INT8  (models/mt_ct2_int8)   ← optimized
  TTS: Piper ONNX fp32   (en_US-lessac-medium)  ← kept fp32 (real-time)
Mode: SEQUENTIAL per-sentence + streaming output (the recommended runtime;
      cross-stage parallelism was shown to hurt on a saturated CPU).
Measures wall time, time-to-first-audio, throughput, RTF, CPU, process RSS,
and prints the improvement vs the fp32 baseline (results/pipeline_results.json).
"""
import json, os, sys, time, gc, wave, statistics, threading
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import psutil
import ctranslate2
from transformers import AutoTokenizer
from piper import PiperVoice

MT_HF     = "Helsinki-NLP/opus-mt-vi-en"     # tokenizer only
CT2_DIR   = Path("models/mt_ct2_int8")
VOICE     = Path("models/en_US-lessac-medium.onnx")
BENCH     = Path("benchmark/vi_en_benchmark.json")
OUT       = Path("results"); OUT.mkdir(exist_ok=True)
AUDIO     = OUT / "audio_opt"; AUDIO.mkdir(exist_ok=True)
MAX_NEW   = 256
PROC      = psutil.Process(os.getpid())


class Monitor:
    def __init__(s, iv=0.25): s.iv, s.cpu, s.rss, s.run = iv, [], [], False
    def start(s):
        s.run=True; psutil.cpu_percent(None)
        s.t=threading.Thread(target=s._p, daemon=True); s.t.start()
    def _p(s):
        while s.run:
            s.cpu.append(psutil.cpu_percent(None)); s.rss.append(PROC.memory_info().rss/1024/1024)
            time.sleep(s.iv)
    def stop(s):
        s.run=False; s.t.join(timeout=2)
    def summ(s):
        f=lambda l:{"mean":round(statistics.mean(l),2),"max":round(max(l),2),"min":round(min(l),2)} if l else {}
        return {"cpu_percent":f(s.cpu),"proc_rss_mb":f(s.rss)}


def main():
    print("="*72)
    print("  OneVoice RECOMMENDED pipeline  —  MT(CT2 int8) + TTS(ONNX fp32)")
    print("="*72)

    data = json.load(open(BENCH, encoding="utf-8"))
    sources = [d["vi"] for d in data]; ids=[d["id"] for d in data]; domains=[d["domain"] for d in data]
    print(f"\n[1/3] {len(sources)} Vietnamese sentences")

    print("[2/3] Loading models (MT CT2 int8 + TTS fp32) …")
    r0 = PROC.memory_info().rss/1024/1024
    tok = AutoTokenizer.from_pretrained(MT_HF)
    translator = ctranslate2.Translator(str(CT2_DIR), device="cpu", compute_type="int8")
    voice = PiperVoice.load(str(VOICE))
    r1 = PROC.memory_info().rss/1024/1024
    print(f"      Both models resident — RSS delta: {r1-r0:.1f} MB")

    def translate(text):
        src = tok.convert_ids_to_tokens(tok.encode(text))
        res = translator.translate_batch([src], max_decoding_length=MAX_NEW)
        return tok.convert_tokens_to_string(res[0].hypotheses[0])

    def synth(text, path):
        with wave.open(str(path), "wb") as wf:
            voice.synthesize_wav(text, wf)
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes()/float(wf.getframerate())

    # warm-up
    for _ in range(3):
        synth(translate(sources[0]), AUDIO/"_w.wav")
    (AUDIO/"_w.wav").unlink(missing_ok=True)

    print("[3/3] Profiling end-to-end (sequential + streaming) …")
    mon = Monitor(); mon.start(); gc.collect()
    t0 = time.perf_counter()
    mt_lat, tts_lat, durs = [], [], []
    ttfa = None
    for i, s in enumerate(sources):
        a = time.perf_counter()
        en = translate(s)
        b = time.perf_counter()
        dur = synth(en, AUDIO/f"{ids[i]:03d}_{domains[i]}.wav")
        c = time.perf_counter()
        mt_lat.append((b-a)*1000); tts_lat.append((c-b)*1000); durs.append(dur)
        if ttfa is None: ttfa = c - t0
    wall = time.perf_counter()-t0
    mon.stop(); st = mon.summ()

    total_audio = sum(durs)
    opt = {
        "config": "MT CTranslate2-int8 + TTS ONNX-fp32",
        "mode": "sequential + streaming",
        "n_sentences": len(sources),
        "both_models_rss_delta_mb": round(r1-r0,1),
        "wall_time_s": round(wall,3),
        "time_to_first_audio_s": round(ttfa,3),
        "throughput_sentences_per_s": round(len(sources)/wall,3),
        "total_audio_s": round(total_audio,3),
        "end_to_end_rtf": round(wall/total_audio,4),
        "x_real_time": round(total_audio/wall,2),
        "mt_latency_ms": {"mean":round(statistics.mean(mt_lat),2),"median":round(statistics.median(mt_lat),2),"p95":round(float(np.percentile(mt_lat,95)),2)},
        "tts_latency_ms": {"mean":round(statistics.mean(tts_lat),2),"median":round(statistics.median(tts_lat),2),"p95":round(float(np.percentile(tts_lat,95)),2)},
        "cpu_percent": st["cpu_percent"],
        "proc_rss_mb": st["proc_rss_mb"],
    }

    # baseline (fp32 sequential) from earlier run
    base = None
    bp = OUT/"pipeline_results.json"
    if bp.exists():
        base = json.load(open(bp, encoding="utf-8")).get("sequential")

    result = {"optimized": opt, "baseline_fp32_sequential": base}
    if base:
        result["improvement"] = {
            "wall_speedup_x": round(base["wall_time_s"]/opt["wall_time_s"],2),
            "throughput_gain_x": round(opt["throughput_sentences_per_s"]/base["throughput_sentences_per_s"],2),
            "mt_latency_speedup_x": round(base["mt_latency_ms"]["mean"]/opt["mt_latency_ms"]["mean"],2),
            "rss_reduction_x": round(base["proc_rss_mb"]["max"]/opt["proc_rss_mb"]["max"],2),
            "cpu_mean_delta_pct": round(opt["cpu_percent"]["mean"]-base["cpu_percent"]["mean"],2),
        }
    json.dump(result, open(OUT/"pipeline_optimized_results.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n"+"="*72)
    print("  RECOMMENDED PIPELINE — END-TO-END RESULT")
    print("="*72)
    print(f"  Sentences        : {len(sources)}  |  audio: {total_audio:.1f}s  |  RSS: {r1-r0:.0f} MB")
    if base:
        print(f"  {'Metric':<24}{'BASELINE fp32':>16}{'OPTIMIZED':>16}")
        print(f"  {'wall time (s)':<24}{base['wall_time_s']:>16}{opt['wall_time_s']:>16}")
        print(f"  {'throughput (sent/s)':<24}{base['throughput_sentences_per_s']:>16}{opt['throughput_sentences_per_s']:>16}")
        print(f"  {'MT latency mean (ms)':<24}{base['mt_latency_ms']['mean']:>16}{opt['mt_latency_ms']['mean']:>16}")
        print(f"  {'TTS latency mean (ms)':<24}{base['tts_latency_ms']['mean']:>16}{opt['tts_latency_ms']['mean']:>16}")
        print(f"  {'time-to-first-audio (s)':<24}{base['time_to_first_audio_s']:>16}{opt['time_to_first_audio_s']:>16}")
        print(f"  {'CPU mean %':<24}{base['cpu_percent']['mean']:>16}{opt['cpu_percent']['mean']:>16}")
        print(f"  {'proc RSS max (MB)':<24}{base['proc_rss_mb']['max']:>16}{opt['proc_rss_mb']['max']:>16}")
        print("-"*72)
        im = result["improvement"]
        print(f"  Wall speedup     : {im['wall_speedup_x']}×    MT latency: {im['mt_latency_speedup_x']}× faster")
        print(f"  RSS reduction    : {im['rss_reduction_x']}×    CPU mean Δ: {im['cpu_mean_delta_pct']:+} pp")
    print("="*72)
    print(f"  Saved → {(OUT/'pipeline_optimized_results.json').resolve()}")


if __name__ == "__main__":
    main()
