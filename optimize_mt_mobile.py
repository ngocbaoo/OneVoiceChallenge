"""
Mobile optimization — Stage 1: MT (opus-mt-vi-en)
  Baseline : PyTorch fp32 (current pipeline)
  Optimized: CTranslate2 INT8  (mobile-friendly runtime + quantization)
Measures: on-disk size, process RSS, per-sentence latency, BLEU (quality guard).
"""
import json, os, sys, time, gc, statistics, shutil
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import psutil
import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ctranslate2
from ctranslate2.converters import TransformersConverter

MODEL = "Helsinki-NLP/opus-mt-vi-en"
CT2_DIR = Path("models/mt_ct2_int8")
BENCH = Path("benchmark/vi_en_benchmark.json")
OUT = Path("results"); OUT.mkdir(exist_ok=True)
MAX_NEW = 256
PROC = psutil.Process(os.getpid())

def dir_size_mb(p):
    p = Path(p)
    if p.is_file(): return os.path.getsize(p)/1024/1024
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())/1024/1024

def rss(): return PROC.memory_info().rss/1024/1024

# ── data ──
data = json.load(open(BENCH, encoding="utf-8"))
sources = [d["vi"] for d in data]
refs = [d["ref_en"] for d in data]
print(f"[data] {len(sources)} sentences")

# ── convert to CT2 int8 ──
if not CT2_DIR.exists():
    print("[convert] Helsinki opus-mt-vi-en -> CTranslate2 INT8 ...")
    TransformersConverter(MODEL).convert(str(CT2_DIR), quantization="int8", force=True)
print(f"[convert] CT2 dir size: {dir_size_mb(CT2_DIR):.1f} MB")

tok = AutoTokenizer.from_pretrained(MODEL)

# ─────────────────────────────────────────────────────────────────────────────
# Baseline: PyTorch fp32
# ─────────────────────────────────────────────────────────────────────────────
def bench_pytorch():
    r0 = rss()
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = n_params*4/1024/1024
    r1 = rss()
    for _ in range(3):  # warmup
        with torch.no_grad():
            ii = tok([sources[0]], return_tensors="pt", padding=True, truncation=True, max_length=512)
            model.generate(**ii, max_new_tokens=MAX_NEW)
    lat, hyps = [], []
    gc.collect()
    for s in sources:
        t0 = time.perf_counter()
        with torch.no_grad():
            ii = tok([s], return_tensors="pt", padding=True, truncation=True, max_length=512)
            out = model.generate(**ii, max_new_tokens=MAX_NEW)
        hyps.append(tok.batch_decode(out, skip_special_tokens=True)[0])
        lat.append((time.perf_counter()-t0)*1000)
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    del model; gc.collect()
    return {"runtime":"PyTorch fp32","size_mb":round(size_mb,1),"ram_load_mb":round(r1-r0,1),
            "lat_mean_ms":round(statistics.mean(lat),1),"lat_median_ms":round(statistics.median(lat),1),
            "lat_p95_ms":round(float(np.percentile(lat,95)),1),"bleu":round(bleu,2)}, hyps

# ─────────────────────────────────────────────────────────────────────────────
# Optimized: CTranslate2 int8
# ─────────────────────────────────────────────────────────────────────────────
def bench_ct2():
    r0 = rss()
    tr = ctranslate2.Translator(str(CT2_DIR), device="cpu", compute_type="int8")
    r1 = rss()
    def translate(text):
        src = tok.convert_ids_to_tokens(tok.encode(text))
        res = tr.translate_batch([src], max_decoding_length=MAX_NEW)
        return tok.convert_tokens_to_string(res[0].hypotheses[0])
    for _ in range(3): translate(sources[0])  # warmup
    lat, hyps = [], []
    gc.collect()
    for s in sources:
        t0 = time.perf_counter()
        hyps.append(translate(s))
        lat.append((time.perf_counter()-t0)*1000)
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    return {"runtime":"CTranslate2 int8","size_mb":round(dir_size_mb(CT2_DIR),1),"ram_load_mb":round(r1-r0,1),
            "lat_mean_ms":round(statistics.mean(lat),1),"lat_median_ms":round(statistics.median(lat),1),
            "lat_p95_ms":round(float(np.percentile(lat,95)),1),"bleu":round(bleu,2)}, hyps

print("\n[bench] PyTorch fp32 baseline ...")
base, base_h = bench_pytorch()
print("  ", base)
print("\n[bench] CTranslate2 int8 ...")
opt, opt_h = bench_ct2()
print("  ", opt)

# agreement between fp32 and int8 outputs (how many identical)
identical = sum(1 for a,b in zip(base_h,opt_h) if a.strip()==b.strip())

result = {
    "stage":"MT (opus-mt-vi-en)",
    "baseline": base,
    "optimized": opt,
    "deltas": {
        "size_reduction_x": round(base["size_mb"]/opt["size_mb"],2),
        "ram_reduction_x": round(base["ram_load_mb"]/opt["ram_load_mb"],2) if opt["ram_load_mb"]>0 else None,
        "speedup_x": round(base["lat_mean_ms"]/opt["lat_mean_ms"],2),
        "bleu_delta": round(opt["bleu"]-base["bleu"],2),
        "identical_outputs_pct": identical,
    },
}
json.dump(result, open(OUT/"mobile_mt_results.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

print("\n"+"="*64)
print("  MT MOBILE OPTIMIZATION — PyTorch fp32  vs  CTranslate2 int8")
print("="*64)
print(f"  {'':20}{'fp32':>14}{'int8':>14}")
print(f"  {'size (MB)':20}{base['size_mb']:>14}{opt['size_mb']:>14}")
print(f"  {'RAM load (MB)':20}{base['ram_load_mb']:>14}{opt['ram_load_mb']:>14}")
print(f"  {'latency mean (ms)':20}{base['lat_mean_ms']:>14}{opt['lat_mean_ms']:>14}")
print(f"  {'BLEU':20}{base['bleu']:>14}{opt['bleu']:>14}")
print("-"*64)
print(f"  size  : {result['deltas']['size_reduction_x']}x smaller")
print(f"  speed : {result['deltas']['speedup_x']}x faster")
print(f"  BLEU  : {result['deltas']['bleu_delta']:+} (identical outputs: {identical}/100)")
print("="*64)
