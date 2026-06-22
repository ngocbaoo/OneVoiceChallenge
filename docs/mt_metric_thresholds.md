# Ngưỡng Tham Chiếu Các Metrics Đánh Giá Dịch Máy

> Tổng hợp ngưỡng chấp nhận được (acceptable) và xuất sắc (excellent) cho các metrics phổ biến trong đánh giá chất lượng dịch máy (Machine Translation Evaluation), kèm trích dẫn nguồn.

---

## 1. BLEU (Bilingual Evaluation Understudy)

**Thang điểm:** 0–100 (↑ tốt hơn)

| Mức BLEU | Diễn giải |
|----------|-----------|
| < 10 | Gần như vô dụng (almost useless) |
| 10–19 | Khó nắm bắt ý chính (hard to get the gist) |
| 20–29 | Hiểu được ý chính nhưng nhiều lỗi ngữ pháp |
| **30–40** | **Bản dịch hiểu được đến tốt (understandable to good) ✅** |
| 40–50 | Chất lượng cao (high quality) |
| 50–60 | Rất cao, đầy đủ và lưu loát (very high quality, adequate, and fluent) |
| > 60 | Thường tốt hơn con người (quality often better than human) |

---

## 2. spBLEU (SentencePiece BLEU)

**Thang điểm:** 0–100 (↑ tốt hơn)

spBLEU là biến thể của BLEU sử dụng tokenizer SentencePiece theo chuẩn FLORES-101, đảm bảo so sánh nhất quán đa ngôn ngữ mà không phụ thuộc vào tokenizer riêng cho từng ngôn ngữ.

| Mức spBLEU | Diễn giải |
|------------|-----------|
| < 20 | Kém (low-resource hoặc cặp ngôn ngữ khó) |
| 20–30 | Trung bình |
| **30–40** | **Tốt ✅** |
| 40–50 | Chất lượng cao |
| > 50 | Xuất sắc |

---

## 3. chrF / chrF++

**Thang điểm:** 0–100 (↑ tốt hơn)

| Mức chrF/chrF++ | Diễn giải |
|-----------------|-----------|
| < 40 | Kém |
| 40–50 | Trung bình |
| **50–60** | **Khá tốt (state-of-the-art cho nhiều cặp ngôn ngữ) ✅** |
| **> 60** | **Tốt đến rất tốt ✅✅** |
| > 70 | Xuất sắc |

### Lưu ý

chrF++ thường cho score thấp hơn chrF vài điểm vì bổ sung word n-grams đánh giá thêm thứ tự từ. Chênh lệch lớn giữa chrF và chrF++ cho thấy mô hình giữ đúng nội dung nhưng thứ tự từ chưa tối ưu.

---

## 4. WER (Word Error Rate)

**Thang điểm:** 0–∞% (↓ tốt hơn)

### Ngưỡng trong ASR (Speech Recognition)

| Mức WER | Diễn giải |
|---------|-----------|
| 0–5% | Xuất sắc |
| 5–10% | Tốt |
| 10–20% | Khá |
| > 20% | Kém |

### Ngưỡng trong MT (Machine Translation) — sử dụng TER làm tham chiếu

| Mức WER/TER | Diễn giải |
|-------------|-----------|
| < 30% | Rất tốt |
| **30–40%** | **Tốt ✅** |
| **40–50%** | **Chấp nhận được ⚠️** |
| > 50% | Cần cải thiện |

### Lưu ý quan trọng

WER **không phải metric lý tưởng cho MT** vì:
- Bản dịch tốt dùng từ đồng nghĩa sẽ bị WER cao
- Ngay cả bản dịch con người cũng có WER cao nếu dùng cách diễn đạt khác reference
- WER chỉ nên dùng bổ trợ, kết hợp với COMET hoặc BLEU

---

## 5. COMET (wmt22-comet-da)

**Thang điểm:** 0–1 (↑ tốt hơn), áp dụng cho model `Unbabel/wmt22-comet-da`

| Mức COMET | Diễn giải |
|-----------|-----------|
| < 0.70 | Kém, cần cải thiện đáng kể |
| 0.70–0.80 | Trung bình |
| 0.80–0.85 | Khá tốt |
| **0.85–0.90** | **Chất lượng chuyên nghiệp (professional-grade) ✅** |
| **> 0.90** | **Gần bằng con người (near-human quality) ✅✅** |

### Lưu ý quan trọng

- Các phiên bản COMET trước (wmt20-comet-da) sử dụng z-score, thang điểm khác hoàn toàn (có thể âm hoặc >1). Các ngưỡng trên **chỉ áp dụng cho wmt22-comet-da trở lên**.
- COMET được cộng đồng WMT khuyến nghị là metric chính cho đánh giá MT vì tương quan cao nhất với đánh giá con người (Kocmi et al., 2021; Freitag et al., 2022).

---

## Bảng Tổng Hợp

| Metric | Thang điểm | Ngưỡng Chấp Nhận | Ngưỡng Tốt | Ngưỡng Xuất Sắc | Nguồn chính |
|--------|-----------|-------------------|-------------|------------------|-------------|
| **BLEU** | 0–100 ↑ | > 30 | > 40 | > 50 | [1][2][3][4] |
| **spBLEU** | 0–100 ↑ | > 30 | > 40 | > 50 | [5][6] |
| **chrF** | 0–100 ↑ | > 50 | > 60 | > 70 | [7][8][9] |
| **chrF++** | 0–100 ↑ | > 50 | > 60 | > 70 | [7][8][10] |
| **WER** | 0–∞% ↓ | < 50% | < 40% | < 30% | [11][12][13][14] |
| **COMET** (wmt22) | 0–1 ↑ | > 0.80 | > 0.85 | > 0.90 | [15][16][17][18] |

---

## Danh Mục Tài Liệu Tham Khảo

| # | Tài liệu | URL |
|---|----------|-----|
| [1] | Google Cloud AutoML — BLEU interpretation guidelines | https://cloud.google.com/translate/automl/docs/evaluate#interpretation |
| [2] | Brenndoerfer, M. (2026). "BLEU Score: Evaluating Translation Quality with N-grams" | https://mbrenndoerfer.com/writing/bleu-score-machine-translation-evaluation-nlp |
| [3] | "Toward domain-specific MT and QE systems" (arXiv:2603.24955) | https://arxiv.org/pdf/2603.24955 |
| [4] | Traceloop. "Demystifying the BLEU Metric" | https://www.traceloop.com/blog/demystifying-the-bleu-metric |
| [5] | Goyal et al. (2022). "The FLORES-101 Evaluation Benchmark", TACL | https://aclanthology.org/2022.tacl-1.30.pdf |
| [6] | LLM Stats — Translation Set1→en spBLEU Leaderboard | https://llm-stats.com/benchmarks/translation-set1→en-spbleu |
| [7] | "Toward domain-specific MT and QE systems" (arXiv:2603.24955) — chrF section | https://arxiv.org/pdf/2603.24955 |
| [8] | "M3Kang" (arXiv:2601.16218) — chrF++ threshold selection | https://arxiv.org/pdf/2601.16218 |
| [9] | Popović, M. (2015). "chrF: character n-gram F-score for automatic MT evaluation", WMT-2015 | https://aclanthology.org/W15-3049 |
| [10] | Popović, M. (2017). "chrF++: words helping character n-grams", WMT-2017 | https://statmt.org/wmt17/pdf/WMT70.pdf |
| [11] | MetricGate (2026). "Word Error Rate Calculator" | https://metricgate.com/docs/word-error-rate/ |
| [12] | Wikipedia — "Word error rate" | https://en.wikipedia.org/wiki/Word_error_rate |
| [13] | "Toward domain-specific MT and QE systems" (arXiv:2603.24955) — TER section | https://arxiv.org/pdf/2603.24955 |
| [14] | "Quality expectations of machine translation" (arXiv:1803.08409) | https://arxiv.org/pdf/1803.08409 |
| [15] | Unbabel — COMET FAQ (official documentation) | https://unbabel.github.io/COMET/html/faqs.html |
| [16] | Unbabel/wmt22-comet-da — Hugging Face model card | https://huggingface.co/Unbabel/wmt22-comet-da |
| [17] | TokenMix (2026). "Best LLM for Translation 2026" | https://tokenmix.ai/blog/best-llm-for-translation |
| [18] | "Pitfalls and Outlooks in Using COMET" (arXiv:2408.15366) | https://arxiv.org/html/2408.15366v1 |

---

*Tài liệu tổng hợp ngày 22/06/2026. Các ngưỡng mang tính tham khảo và phụ thuộc vào cặp ngôn ngữ, domain, số lượng reference translations, và phương pháp tokenization.*
