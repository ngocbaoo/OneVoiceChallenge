"""
=============================================================================
Evaluation Pipeline: Helsinki-NLP/opus-mt-vi-en
Vietnamese → English Machine Translation Benchmark
=============================================================================
Metrics:
  - Translation Quality : BLEU, spBLEU, chrF, chrF++, WER, COMET
  - Performance         : Inference latency, Model size, Peak RAM, CPU load
=============================================================================
"""

import json
import os
import sys

# Force UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import time
import platform
import gc
import statistics
import traceback
from datetime import datetime
from pathlib import Path

import psutil
import threading
import numpy as np

# ── Torch ────────────────────────────────────────────────────────────────────
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Translation-quality metrics ───────────────────────────────────────────────
import sacrebleu
from jiwer import wer as compute_wer

# ── Optional COMET ────────────────────────────────────────────────────────────
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("[WARN] unbabel-comet not installed – COMET scores will be skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME   = "Helsinki-NLP/opus-mt-vi-en"
BENCHMARK    = Path("benchmark/vi_en_benchmark.json")
OUTPUT_DIR   = Path("results")
COMET_MODEL  = "Unbabel/wmt22-comet-da"        # fallback: "Unbabel/wmt21-cometinho-da"
BATCH_SIZE   = 8
WARMUP_RUNS  = 3   # warm-up passes (not counted in latency)
MAX_NEW_TOKENS = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_model_size_mb(model):
    """Return total number of parameters and rough size in MB (float32 equiv.)."""
    n_params = sum(p.numel() for p in model.parameters())
    # actual dtype stored on disk — MarianMT uses float32
    bytes_per_param = 4   # float32
    size_mb = n_params * bytes_per_param / 1024 / 1024
    return n_params, size_mb


def get_file_size_mb(path: str) -> float:
    """Return the size of a file on disk in MB."""
    try:
        return os.path.getsize(path) / 1024 / 1024
    except Exception:
        return 0.0


class SystemMonitor:
    """Polls CPU / RAM / GPU every 500 ms in a background thread."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self._cpu_samples  = []
        self._ram_samples  = []
        self._gpu_samples  = []
        self._vram_samples = []
        self._running = False
        self._thread  = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            self._cpu_samples.append(psutil.cpu_percent(interval=None))
            mem = psutil.virtual_memory()
            self._ram_samples.append(mem.used / 1024 / 1024)

            if DEVICE == "cuda":
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util  = pynvml.nvmlDeviceGetUtilizationRates(h)
                    vmem  = pynvml.nvmlDeviceGetMemoryInfo(h)
                    self._gpu_samples.append(util.gpu)
                    self._vram_samples.append(vmem.used / 1024 / 1024)
                except Exception:
                    pass
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self):
        def safe_stats(lst):
            if not lst:
                return {"mean": 0, "max": 0, "min": 0}
            return {"mean": round(statistics.mean(lst), 2),
                    "max":  round(max(lst), 2),
                    "min":  round(min(lst), 2)}
        return {
            "cpu_percent":      safe_stats(self._cpu_samples),
            "ram_used_mb":      safe_stats(self._ram_samples),
            "gpu_util_percent": safe_stats(self._gpu_samples),
            "vram_used_mb":     safe_stats(self._vram_samples),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_benchmark():
    with open(BENCHMARK, encoding="utf-8") as f:
        data = json.load(f)
    sources    = [d["vi"]     for d in data]
    references = [d["ref_en"] for d in data]
    domains    = [d["domain"] for d in data]
    ids        = [d["id"]     for d in data]
    return data, sources, references, domains, ids


# ─────────────────────────────────────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────────────────────────────────────

def translate_batch(model, tokenizer, texts, device):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded


def translate_all(model, tokenizer, sources, device, batch_size=BATCH_SIZE):
    translations = []
    latencies    = []   # per-sentence latency (ms)
    total_tokens = 0

    n = len(sources)
    for start in range(0, n, batch_size):
        batch   = sources[start : start + batch_size]
        t0      = time.perf_counter()
        hyps    = translate_batch(model, tokenizer, batch, device)
        t1      = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        per_sent   = elapsed_ms / len(batch)
        latencies.extend([per_sent] * len(batch))
        translations.extend(hyps)

        # count tokens generated (rough)
        for h in hyps:
            total_tokens += len(tokenizer.encode(h))

        print(f"  Translated {min(start + batch_size, n)}/{n} sentences …")

    return translations, latencies, total_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu(hypotheses, references):
    """SacreBLEU corpus BLEU (tokenize=13a, the WMT standard)."""
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return result.score, str(result)

def compute_spbleu(hypotheses, references):
    """spBLEU (SentencePiece BLEU – flores tokenizer)."""
    result = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="flores101")
    return result.score, str(result)

def compute_chrf(hypotheses, references, plus=False):
    """chrF or chrF++ (word_order=2 for chrF++)."""
    word_order = 2 if plus else 0
    result = sacrebleu.corpus_chrf(hypotheses, [references], word_order=word_order)
    return result.score, str(result)

def compute_wer_score(hypotheses, references):
    """Word Error Rate (treating reference as ground-truth, hypothesis as ASR output analogy)."""
    scores = []
    for h, r in zip(hypotheses, references):
        try:
            scores.append(compute_wer(r, h))
        except Exception:
            scores.append(1.0)
    return statistics.mean(scores) * 100   # in %

def compute_comet_score(sources, hypotheses, references, model_path=None):
    if not COMET_AVAILABLE:
        return None, None
    try:
        if model_path is None:
            print(f"  Downloading COMET model ({COMET_MODEL}) …")
            model_path = download_model(COMET_MODEL)
        comet_model = load_from_checkpoint(model_path)
        data = [{"src": s, "mt": h, "ref": r}
                for s, h, r in zip(sources, hypotheses, references)]
        gpus = 1 if DEVICE == "cuda" else 0
        output = comet_model.predict(data, batch_size=8, gpus=gpus, progress_bar=True)
        scores = output.scores
        mean_score = float(np.mean(scores))
        return mean_score, scores
    except Exception as e:
        print(f"  [WARN] COMET scoring failed: {e}")
        traceback.print_exc()
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Per-domain breakdown
# ─────────────────────────────────────────────────────────────────────────────

def domain_breakdown(hypotheses, references, domains):
    unique_domains = sorted(set(domains))
    breakdown = {}
    for dom in unique_domains:
        idxs = [i for i, d in enumerate(domains) if d == dom]
        hyp_d = [hypotheses[i] for i in idxs]
        ref_d = [references[i] for i in idxs]
        bleu_score, _ = compute_bleu(hyp_d, ref_d)
        chrf_score, _ = compute_chrf(hyp_d, ref_d)
        breakdown[dom] = {
            "n_sentences": len(idxs),
            "bleu":        round(bleu_score, 2),
            "chrf":        round(chrf_score, 2),
        }
    return breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Helsinki-NLP/opus-mt-vi-en  –  Benchmark Evaluation")
    print(f"  Device : {DEVICE.upper()}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("\n[1/7] Loading benchmark data …")
    data, sources, references, domains, ids = load_benchmark()
    print(f"      Loaded {len(sources)} sentence pairs across {len(set(domains))} domains.")

    # ── 2. Load model ────────────────────────────────────────────────────────
    print(f"\n[2/7] Loading model: {MODEL_NAME} …")
    ram_before_model = psutil.virtual_memory().used / 1024 / 1024

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    ram_after_model = psutil.virtual_memory().used / 1024 / 1024
    model_ram_mb    = ram_after_model - ram_before_model

    n_params, param_size_mb = get_model_size_mb(model)
    print(f"      Parameters : {n_params:,}")
    print(f"      Param size : {param_size_mb:.1f} MB  (float32 equiv.)")
    print(f"      RAM delta  : {model_ram_mb:.1f} MB")

    # ── 3. Warm-up ────────────────────────────────────────────────────────────
    print(f"\n[3/7] Warming up ({WARMUP_RUNS} runs) …")
    warmup_texts = sources[:min(WARMUP_RUNS, len(sources))]
    for i in range(WARMUP_RUNS):
        _ = translate_batch(model, tokenizer, warmup_texts[:2], DEVICE)
    print("      Done.")

    # ── 4. Translate (with system monitoring) ────────────────────────────────
    print("\n[4/7] Translating benchmark sentences …")
    monitor = SystemMonitor(interval=0.5)
    monitor.start()

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    translations, latencies, total_tokens = translate_all(
        model, tokenizer, sources, DEVICE, BATCH_SIZE
    )
    t_end = time.perf_counter()

    monitor.stop()
    sys_stats = monitor.summary()

    total_time_s       = t_end - t_start
    avg_latency_ms     = statistics.mean(latencies)
    median_latency_ms  = statistics.median(latencies)
    p95_latency_ms     = float(np.percentile(latencies, 95))
    throughput_sps     = len(sources) / total_time_s

    peak_ram_mb = sys_stats["ram_used_mb"]["max"]
    if DEVICE == "cuda":
        try:
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        except Exception:
            peak_vram_mb = 0.0
    else:
        peak_vram_mb = 0.0

    print(f"      Total time   : {total_time_s:.1f}s")
    print(f"      Avg latency  : {avg_latency_ms:.1f} ms/sentence")
    print(f"      Throughput   : {throughput_sps:.2f} sentences/s")

    # ── 5. Quality metrics ───────────────────────────────────────────────────
    print("\n[5/7] Computing translation-quality metrics …")

    print("  [*] BLEU ...")
    bleu_score, bleu_str       = compute_bleu(translations, references)

    print("  [*] spBLEU ...")
    spbleu_score, spbleu_str   = compute_spbleu(translations, references)

    print("  [*] chrF ...")
    chrf_score, chrf_str       = compute_chrf(translations, references, plus=False)

    print("  [*] chrF++ ...")
    chrfpp_score, chrfpp_str   = compute_chrf(translations, references, plus=True)

    print("  [*] WER ...")
    wer_score                  = compute_wer_score(translations, references)

    print("  [*] COMET ...")
    comet_score, comet_scores  = compute_comet_score(sources, translations, references)

    print("  [*] Per-domain breakdown ...")
    breakdown = domain_breakdown(translations, references, domains)

    # ── 6. Build results dict ────────────────────────────────────────────────
    print("\n[6/7] Assembling results …")

    results = {
        "metadata": {
            "model":          MODEL_NAME,
            "task":           "vi → en",
            "benchmark":      str(BENCHMARK),
            "n_sentences":    len(sources),
            "n_domains":      len(set(domains)),
            "device":         DEVICE.upper(),
            "timestamp":      datetime.now().isoformat(),
            "python_version": sys.version,
            "platform":       platform.platform(),
            "torch_version":  torch.__version__,
        },
        "quality_metrics": {
            "BLEU":      {"score": round(bleu_score, 4),    "detail": bleu_str},
            "spBLEU":    {"score": round(spbleu_score, 4),  "detail": spbleu_str},
            "chrF":      {"score": round(chrf_score, 4),    "detail": chrf_str},
            "chrF++":    {"score": round(chrfpp_score, 4),  "detail": chrfpp_str},
            "WER_%":     {"score": round(wer_score, 4)},
            "COMET":     {"score": round(comet_score, 4) if comet_score is not None else None,
                          "model": COMET_MODEL},
        },
        "performance_metrics": {
            "model_parameters":      n_params,
            "model_size_mb_float32": round(param_size_mb, 2),
            "ram_delta_on_load_mb":  round(model_ram_mb, 2),
            "peak_ram_used_mb":      round(peak_ram_mb, 2),
            "peak_vram_used_mb":     round(peak_vram_mb, 2),
            "total_inference_time_s":round(total_time_s, 3),
            "avg_latency_ms":        round(avg_latency_ms, 3),
            "median_latency_ms":     round(median_latency_ms, 3),
            "p95_latency_ms":        round(p95_latency_ms, 3),
            "throughput_sentences_per_s": round(throughput_sps, 3),
            "total_output_tokens":   total_tokens,
            "system_monitor":        sys_stats,
        },
        "per_domain": breakdown,
    }

    # ── 7. Save ───────────────────────────────────────────────────────────────
    print("\n[7/7] Saving outputs …")
    results_path = OUTPUT_DIR / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"      Results JSON → {results_path}")

    # Save sentence-level translations
    sent_path = OUTPUT_DIR / "translations.jsonl"
    with open(sent_path, "w", encoding="utf-8") as f:
        for i, (src, hyp, ref, dom) in enumerate(
            zip(sources, translations, references, domains)
        ):
            comet_s = None
            if comet_scores is not None:
                try:
                    comet_s = round(float(comet_scores[i]), 4)
                except Exception:
                    pass
            row = {
                "id": ids[i], "domain": dom,
                "source": src, "hypothesis": hyp, "reference": ref,
                "comet_score": comet_s,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"      Translations → {sent_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Model          : {MODEL_NAME}")
    print(f"  Sentences      : {len(sources)}")
    print(f"  Device         : {DEVICE.upper()}")
    print()
    print("  ── Translation Quality ──────────────────────────────────────────")
    print(f"  BLEU           : {bleu_score:.2f}")
    print(f"  spBLEU         : {spbleu_score:.2f}")
    print(f"  chrF           : {chrf_score:.2f}")
    print(f"  chrF++         : {chrfpp_score:.2f}")
    print(f"  WER            : {wer_score:.2f}%")
    if comet_score is not None:
        print(f"  COMET          : {comet_score:.4f}  (model: {COMET_MODEL})")
    else:
        print("  COMET          : N/A")
    print()
    print("  ── Performance ─────────────────────────────────────────────────")
    print(f"  Parameters     : {n_params:,}")
    print(f"  Model Size     : {param_size_mb:.1f} MB (fp32 equiv.)")
    print(f"  RAM delta      : {model_ram_mb:.1f} MB")
    print(f"  Peak RAM       : {peak_ram_mb:.1f} MB")
    if DEVICE == "cuda":
        print(f"  Peak VRAM      : {peak_vram_mb:.1f} MB")
    print(f"  Total time     : {total_time_s:.1f}s")
    print(f"  Avg latency    : {avg_latency_ms:.1f} ms/sentence")
    print(f"  Median latency : {median_latency_ms:.1f} ms/sentence")
    print(f"  p95 latency    : {p95_latency_ms:.1f} ms/sentence")
    print(f"  Throughput     : {throughput_sps:.2f} sentences/s")
    print(f"  CPU (mean/max) : {sys_stats['cpu_percent']['mean']}% / {sys_stats['cpu_percent']['max']}%")
    print(f"  RAM (mean/max) : {sys_stats['ram_used_mb']['mean']} / {sys_stats['ram_used_mb']['max']} MB")
    if sys_stats["gpu_util_percent"]["max"] > 0:
        print(f"  GPU util       : {sys_stats['gpu_util_percent']['mean']}% / {sys_stats['gpu_util_percent']['max']}%")
    print()
    print("  ── Per-domain BLEU / chrF ──────────────────────────────────────")
    for dom, vals in breakdown.items():
        print(f"  {dom:<15} BLEU={vals['bleu']:5.2f}  chrF={vals['chrf']:5.2f}  (n={vals['n_sentences']})")
    print("=" * 70)
    print(f"  Full results → {results_path.resolve()}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
