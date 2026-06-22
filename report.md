# Báo Cáo Đánh Giá Mô Hình Dịch Máy
# Helsinki-NLP / opus-mt-vi-en
## Vietnamese → English Machine Translation Benchmark

---

**Ngày thực hiện:** 22/06/2026 | **Thiết bị:** CPU (AMD64, Windows 11) | **Framework:** PyTorch 2.12.0 + Transformers (HuggingFace)

---

## 1. Tổng Quan

### 1.1 Giới thiệu mô hình

| Thuộc tính | Giá trị |
|---|---|
| **Model ID** | `Helsinki-NLP/opus-mt-vi-en` |
| **Kiến trúc** | MarianMT (Transformer Encoder-Decoder) |
| **Tác vụ** | Neural Machine Translation (vi → en) |
| **Nguồn dữ liệu huấn luyện** | OPUS corpus (Tatoeba Challenge + CCAligned + WikiMatrix…) |
| **Tokenizer** | SentencePiece (vocab size 32k) |
| **Số tham số** | 72,177,152 (~72M) |
| **Kích thước mô hình** | 275.3 MB (float32 equivalent) |
| **License** | Apache 2.0 / CC-BY 4.0 |

### 1.2 Phạm vi đánh giá

Báo cáo này bao gồm:
- **Chất lượng dịch thuật:** BLEU, spBLEU, chrF, chrF++, WER, COMET
- **Phân tích theo domain:** 10 lĩnh vực × 10 câu/lĩnh vực
- **Hiệu năng vận hành:** Inference latency, throughput, model size, RAM usage, CPU load
- **Phân tích định tính:** Lỗi điển hình, điểm mạnh/yếu

---

## 2. Thiết Kế Benchmark

### 2.1 Cấu trúc tập dữ liệu

Benchmark được tự xây dựng gồm **100 cặp câu Việt–Anh** trải đều trên **10 domain**:

| Domain | Số câu | Mô tả |
|---|---|---|
| `news` | 10 | Tin tức thời sự, chính trị, kinh tế vĩ mô |
| `technology` | 10 | Công nghệ, AI, phần mềm, mạng 5G |
| `medical` | 10 | Y tế, dược phẩm, lâm sàng |
| `legal` | 10 | Pháp luật, hợp đồng, tố tụng |
| `education` | 10 | Giáo dục, học thuật, kỹ năng |
| `economy` | 10 | Kinh tế vi mô, tài chính, đầu tư |
| `environment` | 10 | Môi trường, biến đổi khí hậu |
| `culture` | 10 | Văn hóa, lịch sử, nghệ thuật Việt Nam |
| `daily_life` | 10 | Hội thoại và tình huống hàng ngày |
| `science` | 10 | Khoa học cơ bản: vật lý, sinh học, thiên văn |

**Tiêu chí thiết kế:**
- Câu có độ dài trung bình 10–20 từ tiếng Anh
- Bao gồm cả câu phức và câu đơn giản
- Đề cập đến các thuật ngữ chuyên ngành để kiểm tra OOV
- Reference translations do con người viết theo chuẩn WMT

### 2.2 Môi trường chạy

```
OS       : Windows 11 (10.0.26200)
Python   : 3.13.2 (Anaconda)
PyTorch  : 2.12.0+cpu  (CPU-only)
Device   : CPU — không có GPU/NPU
Batch size: 8 câu/batch
Warm-up  : 3 lần trước khi đo latency
```

> [!NOTE]
> Đánh giá chạy hoàn toàn trên **CPU**. Kết quả latency sẽ cải thiện đáng kể trên GPU/NPU.

---

## 3. Kết Quả Đánh Giá Chất Lượng

### 3.1 Tổng quan các metrics

| Metric | Score | Thang điểm | Đánh giá |
|---|---|---|---|
| **BLEU** | **37.08** | 0–100 (↑ tốt hơn) | ✅ Tốt (ngưỡng thông dụng: >30 = usable) |
| **spBLEU** | **36.17** | 0–100 (↑ tốt hơn) | ✅ Tốt |
| **chrF** | **62.11** | 0–100 (↑ tốt hơn) | ✅ Tốt |
| **chrF++** | **59.85** | 0–100 (↑ tốt hơn) | ✅ Tốt |
| **WER** | **49.24%** | 0–∞% (↓ tốt hơn) | ⚠️ Trung bình–Cao |
| **COMET** | **0.8740** | 0–1 (↑ tốt hơn) | ✅ Tốt (>0.85 là chấp nhận được) |

---

### 3.2 Chi tiết từng metric

#### 3.2.1 BLEU (Bilingual Evaluation Understudy)

```
BLEU = 37.08
N-gram precision:
  1-gram : 65.1%   ← từ đơn khớp tốt
  2-gram : 41.4%   ← bigram khớp khá
  3-gram : 30.4%   ← trigram khớp ổn
  4-gram : 23.7%   ← 4-gram giảm dần
Brevity Penalty (BP)  : 0.994
Length ratio (hyp/ref): 0.994 (1251 / 1258 tokens)
```

**Nhận xét:** BLEU 37.08 nằm trong ngưỡng "usable translation" theo chuẩn WMT. Độ dài hypothesis rất gần reference (BP ≈ 1.0), cho thấy mô hình không cắt ngắn hay phình câu. Tuy nhiên, 4-gram precision chỉ đạt 23.7% phản ánh cụm từ dài vẫn còn lệch với reference.

#### 3.2.2 spBLEU (SentencePiece BLEU — flores101 tokenizer)

```
spBLEU = 36.17
N-gram precision:
  1-gram : 64.8%
  2-gram : 42.6%
  3-gram : 31.2%
  4-gram : 24.4%
BP       : 0.950
```

**Nhận xét:** spBLEU sử dụng tokenization SentencePiece theo chuẩn Flores-101, phù hợp hơn cho đánh giá đa ngôn ngữ. Score thấp hơn BLEU khoảng 0.9 điểm do tokenization khác nhau (BP thấp hơn: 0.950 vs 0.994). Hai metric này nhất quán với nhau.

#### 3.2.3 chrF và chrF++

```
chrF   = 62.11   (character-level F-score, word_order=0)
chrF++ = 59.85   (character-level + word unigrams/bigrams, word_order=2)
```

**Nhận xét:** chrF đánh giá ở cấp độ ký tự, ít bị ảnh hưởng bởi tokenization. Score 62.11 cho thấy mô hình tái tạo được phần lớn chuỗi ký tự của reference. Chênh lệch chrF++ vs chrF (≈2.26 điểm) phản ánh mô hình đôi khi giữ đúng nội dung nhưng thứ tự từ chưa tối ưu.

#### 3.2.4 WER (Word Error Rate)

```
WER = 49.24%
```

**Nhận xét:** WER đo tỉ lệ insertion + deletion + substitution ở cấp từ so với reference. Giá trị 49.24% nghe có vẻ cao, nhưng cần lưu ý:
- WER vốn được thiết kế cho ASR, không phải MT — thậm chí bản dịch tốt của con người cũng có WER cao nếu dùng từ đồng nghĩa
- Một câu như "keep the basic interest rate intact" vs "keep the benchmark interest rate unchanged" → WER = 100% (khác hoàn toàn dù nghĩa tương đương)
- WER phù hợp hơn để phát hiện lỗi cấu trúc và sai nghĩa nghiêm trọng, nên dùng kết hợp với COMET

#### 3.2.5 COMET (wmt22-comet-da)

```
COMET Score = 0.8740
COMET Model : Unbabel/wmt22-comet-da
             (Direct Assessment model, dựa trên XLM-RoBERTa-large)
```

**Phân phối COMET score theo câu:**

| Dải score | Số câu | Tỉ lệ |
|---|---|---|
| ≥ 0.95 | 15 | 15% |
| 0.90 – 0.95 | 27 | 27% |
| 0.85 – 0.90 | 29 | 29% |
| 0.70 – 0.85 | 22 | 22% |
| < 0.70 | 7 | 7% |

**Nhận xét:** COMET 0.874 là chỉ số mạnh nhất trong bộ đánh giá, phản ánh rằng mô hình hiểu nghĩa tổng thể khá tốt dù surface form (n-gram) còn lệch. 71% câu đạt COMET ≥ 0.85. Chỉ 7 câu (<7%) bị đánh giá kém (<0.70), thường là lỗi dịch sai nghĩa hoặc bỏ nội dung quan trọng.

---

### 3.3 Phân tích theo Domain

| Domain | BLEU | chrF | Đánh giá |
|---|---|---|---|
| **technology** | **57.49** | **77.86** | 🥇 Tốt nhất — thuật ngữ kỹ thuật quốc tế phổ biến |
| **science** | **53.54** | **72.95** | 🥈 Rất tốt — danh từ riêng và khái niệm khoa học rõ ràng |
| **daily_life** | 44.94 | 66.79 | ✅ Tốt — ngữ pháp hàng ngày đơn giản |
| **news** | 39.56 | 65.45 | ✅ Tốt |
| **education** | 37.25 | 65.68 | ✅ Khá tốt |
| **medical** | 34.65 | 60.53 | ✅ Trung bình–khá |
| **environment** | 25.60 | 57.50 | ⚠️ Trung bình |
| **legal** | 25.84 | 52.77 | ⚠️ Trung bình — thuật ngữ pháp lý phức tạp |
| **culture** | 24.56 | 51.40 | ⚠️ Dưới trung bình — khái niệm đặc thù VN |
| **economy** | **17.39** | **53.52** | 🔴 Thấp nhất — thuật ngữ tài chính chuyên sâu |

**Nguyên nhân chênh lệch:**
- **Technology/Science cao:** Thuật ngữ quốc tế (AI, 5G, DNA, LIGO…) đồng nhất, mô hình có nhiều dữ liệu huấn luyện
- **Culture thấp:** Các khái niệm đặc thù Việt Nam ("Áo dài", "Phố cổ Hội An", "múa rối nước") dễ bị dịch sai
- **Economy thấp:** Thuật ngữ tài chính trừu tượng ("tài khóa thắt chặt", "thâm hụt ngân sách") gây khó khăn

---

## 4. Kết Quả Hiệu Năng (Performance Profiling)

### 4.1 Thống kê mô hình

| Chỉ số | Giá trị |
|---|---|
| **Số tham số** | 72,177,152 (~72M) |
| **Kích thước float32** | 275.3 MB |
| **RAM tăng thêm khi load model** | 277.8 MB |
| **VRAM sử dụng** | 0 MB (CPU-only) |

### 4.2 Inference Latency

> Đo trên 100 câu, batch size = 8, sau 3 lần warm-up

| Chỉ số | Giá trị |
|---|---|
| **Tổng thời gian dịch 100 câu** | 10.92 giây |
| **Latency trung bình (avg)** | 109.0 ms / câu |
| **Latency trung vị (median)** | 107.6 ms / câu |
| **Latency p95** | 127.7 ms / câu |
| **Throughput** | 9.16 câu / giây |
| **Tổng output tokens** | 2,357 tokens |

**Phân tích latency:**
- Median ≈ Mean → phân phối latency đồng đều, không có outlier lớn
- p95 (127.7ms) chỉ cao hơn median 18.6% → latency ổn định
- Tốc độ 9.16 câu/giây trên CPU là hợp lý cho mô hình 72M tham số

### 4.3 CPU Load

| Chỉ số | Giá trị |
|---|---|
| **CPU usage trung bình** | 83.35% |
| **CPU usage tối đa** | 93.4% |
| **CPU usage tối thiểu** | 0.0% (giai đoạn idle) |

**Nhận xét:** CPU bị tải nặng (~83% mean) trong suốt quá trình inference. Mô hình MarianMT sử dụng tối đa lõi CPU với PyTorch. Trong môi trường production, cần cân nhắc resource isolation.

### 4.4 RAM Usage

RAM tiêu thụ riêng cho **translation inference** chỉ khoảng **278–350 MB**.

---

## 5. Phân Tích Định Tính

### 5.1 Ví dụ dịch xuất sắc (COMET ≥ 0.97)

#### Ví dụ 1 — Bản dịch hoàn hảo [TECHNOLOGY]
| | |
|---|---|
| **Nguồn (VI)** | Các nhà nghiên cứu đã phát triển một thuật toán mới có thể phát hiện ung thư sớm hơn. |
| **Mô hình dịch** | Researchers have developed a new algorithm that can detect cancer earlier. |
| **Tham chiếu** | Researchers have developed a new algorithm that can detect cancer earlier. |
| **COMET** | 0.9769 ✅ |

#### Ví dụ 2 — Bản dịch hoàn hảo [SCIENCE]
| | |
|---|---|
| **Nguồn (VI)** | Hệ mặt trời của chúng ta bao gồm tám hành tinh quay quanh mặt trời. |
| **Mô hình dịch** | Our solar system consists of eight planets orbiting the sun. |
| **Tham chiếu** | Our solar system consists of eight planets orbiting the sun. |
| **COMET** | 0.9758 ✅ |

#### Ví dụ 3 — Bản dịch xuất sắc [NEWS]
| | |
|---|---|
| **Nguồn (VI)** | Hội nghị thượng đỉnh G20 sẽ được tổ chức tại New Delhi vào tháng tới. |
| **Mô hình dịch** | The G20 summit will be held in New Delhi next month. |
| **Tham chiếu** | The G20 summit will be held in New Delhi next month. |
| **COMET** | 0.9747 ✅ |

---

### 5.2 Lỗi điển hình

#### Lỗi 1 — Dịch sai số [ECONOMY]
| | |
|---|---|
| **Nguồn (VI)** | Tốc độ tăng trưởng GDP của Việt Nam được dự báo đạt **6,5** phần trăm trong năm nay. |
| **Mô hình dịch** | Vietnam's GDP growth rate is predicted to be **4.5** percent this year. |
| **Tham chiếu** | Vietnam's GDP growth rate is forecast to reach 6.5 percent this year. |
| **COMET** | 0.9440 |
| **Phân tích** | Lỗi nghiêm trọng về số (6,5 → 4.5). Dấu phẩy trong số tiếng Việt gây nhầm lẫn. |

#### Lỗi 2 — Thuật ngữ chuyên ngành sai [LEGAL]
| | |
|---|---|
| **Nguồn (VI)** | Quyền **sở hữu trí tuệ** cần được bảo vệ nghiêm ngặt để khuyến khích sáng tạo. |
| **Mô hình dịch** | **Mind ownership** needs strict protection to encourage creativity. |
| **Tham chiếu** | Intellectual property rights need to be strictly protected to encourage innovation. |
| **COMET** | 0.7404 |
| **Phân tích** | "Sở hữu trí tuệ" dịch thành "Mind ownership" thay vì "Intellectual property" — lỗi thuật ngữ pháp lý. |

#### Lỗi 3 — Từ chuyên ngành y tế [MEDICAL]
| | |
|---|---|
| **Nguồn (VI)** | **Tiêm vắc-xin** là biện pháp phòng ngừa hiệu quả nhất đối với nhiều bệnh truyền nhiễm. |
| **Mô hình dịch** | **Vast vaccine** is the most effective preventive measure for many infectious diseases. |
| **Tham chiếu** | Vaccination is the most effective preventive measure against many infectious diseases. |
| **COMET** | 0.8754 |
| **Phân tích** | "Tiêm vắc-xin" → "Vast vaccine" (đánh máy/hallucination). Nên là "Vaccination". |

#### Lỗi 4 — Danh từ riêng Việt Nam [CULTURE]
| | |
|---|---|
| **Nguồn (VI)** | **Phố cổ Hội An** là di sản văn hóa thế giới được UNESCO công nhận. |
| **Mô hình dịch** | **The Old Society of An** is the widely recognized cultural heritage of the world. |
| **Tham chiếu** | Hoi An Ancient Town is a UNESCO-recognized World Cultural Heritage site. |
| **COMET** | 0.5867 ❌ |
| **Phân tích** | Không nhận dạng được tên riêng "Hội An". "Phố cổ" dịch thành "Old Society" thay vì "Ancient Town". |

#### Lỗi 5 — Câu phức/trừu tượng [ECONOMY]
| | |
|---|---|
| **Nguồn (VI)** | **Lạm phát tăng cao** đang ảnh hưởng đến sức mua của người dân. |
| **Mô hình dịch** | **The rise and rise** are affecting people's buying power. |
| **Tham chiếu** | Rising inflation is affecting people's purchasing power. |
| **COMET** | 0.6712 ❌ |
| **Phân tích** | "Lạm phát tăng cao" → "The rise and rise" — mô hình hoàn toàn bỏ sót từ "lạm phát" (inflation). |

#### Lỗi 6 — Từ viết tắt [EDUCATION]
| | |
|---|---|
| **Nguồn (VI)** | Giáo dục **STEM** đang được chú trọng phát triển để đáp ứng nhu cầu nhân lực tương lai. |
| **Mô hình dịch** | **SMG** education is being promoted to meet future human resources needs. |
| **Tham chiếu** | STEM education is being emphasized for development to meet future workforce needs. |
| **COMET** | 0.7985 |
| **Phân tích** | Từ viết tắt "STEM" bị dịch nhầm thành "SMG" — lỗi tokenization với từ viết tắt chuyên ngành. |

---

### 5.3 Tổng hợp điểm mạnh và yếu

#### ✅ Điểm mạnh
1. **Fluency tốt:** Câu dịch tự nhiên, đúng ngữ pháp tiếng Anh trong hầu hết trường hợp
2. **Technical/Scientific terms:** Xử lý tốt thuật ngữ quốc tế (AI, DNA, 5G, LIGO, mRNA)
3. **Cấu trúc câu đơn giản:** Câu ngắn, chủ ngữ rõ ràng → dịch chính xác cao
4. **Danh từ riêng quốc tế:** G20, New Delhi, Amazon rainforest → không bị lỗi
5. **Tốc độ:** 9.16 câu/giây trên CPU — phù hợp cho ứng dụng không yêu cầu real-time

#### ⚠️ Điểm yếu
1. **Danh từ riêng Việt Nam:** "Hội An", "Áo dài", "Chăm Pa" thường bị dịch sai hoặc paraphrase lạ
2. **Thuật ngữ pháp lý/tài chính chuyên sâu:** "sở hữu trí tuệ", "tài khóa thắt chặt" → hallucination
3. **Số học với định dạng Việt Nam:** Dấu phẩy trong số (6,5%) gây nhầm lẫn
4. **Từ viết tắt:** STEM → SMG (lỗi tokenization)
5. **Câu dài phức hợp:** Mất thông tin hoặc paraphrase không chính xác
6. **Domain nặng về văn hóa bản địa:** Economy, Culture → BLEU thấp nhất

---

## 6. So Sánh Với Ngưỡng Tham Chiếu

### 6.1 Ngưỡng BLEU theo văn học

| Mức BLEU | Mô tả |
|---|---|
| < 10 | Gần như vô nghĩa |
| 10 – 20 | Rõ ý nhưng cần biên tập nhiều |
| 20 – 30 | Có thể hiểu được, cần biên tập |
| **30 – 40** | **Dịch tốt (understandable + good quality) ← opus-mt-vi-en nằm đây** |
| 40 – 50 | Dịch rất tốt, gần chất lượng con người |
| > 50 | Near-human quality |

### 6.2 So sánh với hệ thống thương mại (ước tính vi→en)

| Hệ thống | BLEU ước tính | Ghi chú |
|---|---|---|
| Google Translate | ~45–55 | SOTA commercial, nhiều data |
| DeepL | ~43–53 | Tốt nhất về fluency |
| **opus-mt-vi-en** | **37.08** | Open-source, 72M params, không cần API |
| mBART-50 | ~32–38 | Đa ngôn ngữ, cần fine-tune |
| NLLB-200 (600M) | ~40–48 | Meta, open-source nhưng lớn hơn |

> [!TIP]
> opus-mt-vi-en đạt **~70–80% chất lượng so với Google Translate** nhưng hoàn toàn **miễn phí, offline, nhẹ** (72M params). Đây là lựa chọn rất phù hợp cho ứng dụng embedded hoặc offline.

---

## 7. Kết Luận

### 7.1 Tóm tắt kết quả

```
┌─────────────────────────────────────────────────────┐
│   TỔNG KẾT ĐÁNH GIÁ opus-mt-vi-en (vi → en)        │
├─────────────────┬───────────┬─────────────────────── ┤
│ BLEU            │  37.08    │ ✅ Tốt                 │
│ spBLEU          │  36.17    │ ✅ Tốt                 │
│ chrF            │  62.11    │ ✅ Tốt                 │
│ chrF++          │  59.85    │ ✅ Tốt                 │
│ WER             │  49.24%   │ ⚠️ Trung bình (MT)     │
│ COMET           │   0.874   │ ✅ Tốt                 │
├─────────────────┼───────────┼────────────────────────┤
│ Tham số         │   72M     │ Nhỏ gọn                │
│ Model size      │ 275.3 MB  │ Deployable             │
│ Avg latency     │ 109 ms    │ Real-time trên CPU     │
│ Throughput      │ 9.16 sps  │ Batch ổn định          │
│ RAM model       │ ~278 MB   │ Nhẹ                    │
│ CPU load        │ 83%       │ Cao khi inference      │
└─────────────────┴───────────┴────────────────────────┘
```

### 7.2 Khuyến nghị sử dụng

| Tình huống | Phù hợp? |
|---|---|
| Dịch tin tức, báo cáo chung | ✅ Phù hợp tốt |
| Dịch tài liệu khoa học/kỹ thuật | ✅ Phù hợp tốt |
| Ứng dụng offline, embedded | ✅ Rất phù hợp (275MB) |
| Dịch văn bản pháp lý chính thức | ⚠️ Cần post-edit |
| Dịch nội dung văn hóa đặc thù VN | ⚠️ Kết quả không ổn định |
| Dịch tài liệu tài chính phức tạp | ❌ Cần hệ thống mạnh hơn |
| Real-time production với GPU | ✅ Với GPU: ~5–15ms/câu |

### 7.3 Khuyến nghị cải thiện

1. **Fine-tuning theo domain:** Với tập dữ liệu pháp lý/tài chính Việt Nam → cải thiện 5–15 BLEU
2. **Hậu xử lý (post-processing):** Rule-based correction cho số, từ viết tắt (STEM, MRI), danh từ riêng
3. **Beam search tuning:** Tăng `num_beams` (4→8) → cải thiện quality nhưng tăng latency
4. **Chuyển sang GPU:** Giảm latency 10–20x, giải phóng CPU
5. **Mô hình lớn hơn:** `Helsinki-NLP/opus-mt-tc-big-vi-en` hoặc NLLB-200 (1.3B) cho chất lượng cao hơn
6. **Ensemble/reranking:** Kết hợp với COMET để lọc bản dịch tốt nhất

---

## 8. Phụ Lục

### 8.1 Cấu hình chạy

```python
MODEL_NAME    = "Helsinki-NLP/opus-mt-vi-en"
BATCH_SIZE    = 8
MAX_NEW_TOKENS= 256
WARMUP_RUNS   = 3
DEVICE        = "cpu"
COMET_MODEL   = "Unbabel/wmt22-comet-da"
```

### 8.2 Các file kết quả

| File | Mô tả |
|---|---|
| `benchmark/vi_en_benchmark.json` | 100 cặp câu benchmark |
| `results/evaluation_results.json` | Kết quả đầy đủ (JSON) |
| `results/translations.jsonl` | Bản dịch từng câu + COMET score |
| `evaluate.py` | Script đánh giá chính |

### 8.3 Thư viện sử dụng

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| `transformers` | latest | Load model MarianMT |
| `sacrebleu` | latest | BLEU, spBLEU, chrF, chrF++ |
| `jiwer` | latest | WER |
| `unbabel-comet` | ≥2.2.0 | COMET scoring |
| `psutil` | latest | CPU/RAM monitoring |
| `torch` | 2.12.0+cpu | Inference engine |
| `sentencepiece` | latest | Tokenization |
| `sacremoses` | latest | Text normalization |

---
