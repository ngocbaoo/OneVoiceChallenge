"""
Accuracy of the RECOMMENDED pipeline (MT CT2-int8 + TTS fp32).
  MT  : BLEU, chrF, chrF++  vs human reference (sacrebleu)
  TTS : WER (ASR round-trip) on the audio produced by the optimized pipeline
Compares against the fp32 baselines (report_mt.md BLEU 37.08, report_tts.md WER 1.34%).
"""
import json, sys, statistics, re
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

import ctranslate2
from transformers import AutoTokenizer
import sacrebleu
import jiwer
from faster_whisper import WhisperModel

CT2_DIR = Path("models/mt_ct2_int8")
MT_HF   = "Helsinki-NLP/opus-mt-vi-en"
BENCH   = Path("benchmark/vi_en_benchmark.json")
AUDIO   = Path("results/audio_opt")          # wavs from run_pipeline_optimized.py
OUT     = Path("results"); OUT.mkdir(exist_ok=True)
MAX_NEW = 256

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE); _WS = re.compile(r"\s+")
def norm(t): return _WS.sub(" ", _PUNCT.sub(" ", t.lower().strip())).strip()

data = json.load(open(BENCH, encoding="utf-8"))
sources = [d["vi"] for d in data]; refs=[d["ref_en"] for d in data]
ids=[d["id"] for d in data]; domains=[d["domain"] for d in data]
print(f"[data] {len(sources)} sentences")

# ── MT accuracy (CT2 int8) ──
print("[mt] translating with CTranslate2 int8 …")
tok = AutoTokenizer.from_pretrained(MT_HF)
tr = ctranslate2.Translator(str(CT2_DIR), device="cpu", compute_type="int8")
def translate(text):
    src = tok.convert_ids_to_tokens(tok.encode(text))
    res = tr.translate_batch([src], max_decoding_length=MAX_NEW)
    return tok.convert_tokens_to_string(res[0].hypotheses[0])
hyps = [translate(s) for s in sources]

bleu  = sacrebleu.corpus_bleu(hyps, [refs]).score
chrf  = sacrebleu.corpus_chrf(hyps, [refs], word_order=0).score
chrfpp= sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
print(f"[mt] BLEU={bleu:.2f}  chrF={chrf:.2f}  chrF++={chrfpp:.2f}")

# per-domain BLEU
dom_bleu = {}
for dom in sorted(set(domains)):
    idx = [i for i,d in enumerate(domains) if d==dom]
    dom_bleu[dom] = round(sacrebleu.corpus_bleu([hyps[i] for i in idx], [[refs[i] for i in idx]]).score, 2)

# ── TTS accuracy (WER round-trip on optimized-pipeline audio) ──
print("[tts] re-transcribing optimized-pipeline audio with faster-whisper …")
asr = WhisperModel("small", device="cpu", compute_type="int8")
wers = []
missing = 0
for i in range(len(sources)):
    wav = AUDIO / f"{ids[i]:03d}_{domains[i]}.wav"
    if not wav.exists():
        missing += 1; continue
    segs,_ = asr.transcribe(str(wav), language="en", beam_size=5)
    asr_txt = "".join(s.text for s in segs).strip()
    r = norm(hyps[i]); h = norm(asr_txt)
    wers.append(jiwer.wer(r,h) if r else 0.0)
wer_mean = statistics.mean(wers)*100
wer_med  = statistics.median(wers)*100
print(f"[tts] WER mean={wer_mean:.2f}%  median={wer_med:.2f}%  (n={len(wers)}, missing={missing})")

result = {
    "pipeline": "RECOMMENDED (MT CTranslate2-int8 + TTS Piper-fp32)",
    "mt_accuracy": {
        "BLEU": round(bleu,2), "chrF": round(chrf,2), "chrF++": round(chrfpp,2),
        "vs_fp32_baseline": {"BLEU_fp32": 37.08, "BLEU_delta": round(bleu-37.08,2)},
        "per_domain_bleu": dom_bleu,
    },
    "tts_accuracy": {
        "WER_mean_%": round(wer_mean,2), "WER_median_%": round(wer_med,2),
        "asr": "faster-whisper-small (en, beam=5)", "n": len(wers),
        "vs_fp32_baseline": {"WER_fp32_%": 1.34, "WER_delta_pp": round(wer_mean-1.34,2)},
    },
}
json.dump(result, open(OUT/"pipeline_accuracy_results.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

print("\n"+"="*64)
print("  RECOMMENDED PIPELINE — ACCURACY")
print("="*64)
print(f"  MT  : BLEU {bleu:.2f} (fp32 37.08, Δ{bleu-37.08:+.2f}) | chrF {chrf:.2f} | chrF++ {chrfpp:.2f}")
print(f"  TTS : WER {wer_mean:.2f}% (fp32 1.34%, Δ{wer_mean-1.34:+.2f}pp) | median {wer_med:.2f}%")
print("="*64)
