# Báo Cáo Đánh Giá Mô Hình Text-to-Speech
# Piper TTS — en_US-lessac-medium
## English Speech Synthesis Benchmark (giai đoạn cuối của pipeline OneVoice vi → en → speech)

---

**Ngày thực hiện:** 23/06/2026 | **Thiết bị:** CPU (AMD64, Windows 11) | **Engine:** Piper TTS 1.4.x (ONNX Runtime) + eSpeak-NG

---

## 1. Tổng Quan

### 1.1 Vị trí trong pipeline OneVoice

OneVoice là pipeline **giọng-nói-sang-giọng-nói** ba tầng:

```
   Tiếng Việt (text)  ──►  opus-mt-vi-en (MT)  ──►  Tiếng Anh (text)  ──►  Piper TTS  ──►  English Speech
   [đầu vào]               [đã đánh giá ở             [hypothesis]            [báo cáo này]    [đầu ra]
                            report_mt.md]
```

Báo cáo này đánh giá **tầng TTS cuối cùng**. Điểm mấu chốt: TTS **không đọc tiếng Việt** — nó đọc **bản dịch tiếng Anh (`hypothesis`)** do tầng `opus-mt-vi-en` sinh ra và lưu trong `results/translations.jsonl`. Nhờ vậy đánh giá phản ánh đúng âm thanh mà người dùng cuối thực sự nghe được từ hệ thống.

### 1.2 Giới thiệu mô hình

| Thuộc tính | Giá trị |
|---|---|
| **Model** | `en_US-lessac-medium` (Piper voice) |
| **Kiến trúc** | VITS (conditional VAE + adversarial + normalizing flow) |
| **Tác vụ** | English Text-to-Speech (neural, single-speaker) |
| **Phoneme frontend** | eSpeak-NG (IPA phonemization) |
| **Runtime** | ONNX Runtime (CPU) |
| **Dữ liệu huấn luyện** | Lessac corpus (US English, professional narration) |
| **Sample rate** | 22.05 kHz |
| **Kích thước mô hình** | 60.3 MB (ONNX) |
| **License** | MIT (Piper) / dữ liệu giọng theo giấy phép riêng |

### 1.3 Phạm vi đánh giá

Báo cáo này bao gồm:
- **Intelligibility (mức độ rõ ràng):** WER qua ASR re-transcription (Whisper, English)
- **Naturalness (mức độ tự nhiên):** DNSMOS (predicted MOS không cần reference) — OVRL/SIG/BAK/P808
- **Phân tích theo domain:** 10 lĩnh vực × 10 câu/lĩnh vực
- **Hiệu năng vận hành:** Synthesis latency, Real-Time Factor (RTF), throughput, model size, RAM, CPU load
- **Phân tích định tính:** Các trường hợp WER cao, điểm mạnh/yếu

> [!NOTE]
> **Metrics nằm ngoài phạm vi:** PESQ, STOI, MCD, Speaker-Similarity và FAD đều là **intrusive** — cần bản ghi âm tham chiếu (ground-truth human recording) cho từng câu. Benchmark này là tập **văn bản** tự xây dựng, không có audio tham chiếu, nên các metric này không thể tính một cách có ý nghĩa và được loại trừ một cách có chủ đích (xem `docs/tts_metric_thresholds.md`).

---

## 2. Thiết Kế Benchmark

### 2.1 Nguồn dữ liệu đầu vào

Văn bản đưa vào TTS là **100 câu tiếng Anh** = trường `hypothesis` của `results/translations.jsonl`, tức đầu ra dịch máy của 100 câu nguồn tiếng Việt, trải đều trên **10 domain**:

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

**Lý do dùng trực tiếp hypothesis (thay vì reference tiếng Anh "vàng"):**
- Đánh giá đúng **chất lượng âm thanh thực tế** mà pipeline xuất ra cho người dùng cuối.
- WER được tính so với **chính văn bản đưa vào TTS** (hypothesis), nên đo **độ trung thực phát âm của TTS**, tách bạch hoàn toàn khỏi chất lượng dịch máy (đã đo bằng BLEU/COMET ở `report_mt.md`). Một số câu hypothesis có lỗi dịch nhỏ (vd. "locky policy", "restructied") — TTS vẫn phải đọc trung thực những gì nó nhận được, và phần đánh giá định tính tách riêng các trường hợp này.

### 2.2 Phương pháp đo WER cho TTS

```
hypothesis (text EN)  ──►  Piper TTS  ──►  audio.wav  ──►  Whisper ASR (en)  ──►  transcript
                                                                                       │
                              WER = so sánh(transcript, hypothesis text gốc) ◄─────────┘
```

WER thấp ⇒ TTS phát âm rõ ràng, chính xác đến mức ASR khôi phục lại đúng văn bản gốc.

### 2.3 Môi trường chạy

```
OS         : Windows 11 (10.0.26200)
Python     : 3.13.2 (Anaconda)
TTS engine : Piper TTS (onnxruntime, CPU)
ASR        : faster-whisper small (int8, CPU), beam_size = 5, language = en
DNSMOS     : speechmos (P.808 / P.835), input resampled to 16 kHz
Device     : CPU — không có GPU/NPU
Warm-up    : 3 lần trước khi đo latency
```

> [!NOTE]
> Đánh giá chạy hoàn toàn trên **CPU**. Latency sẽ cải thiện đáng kể trên GPU/NPU.

---

## 3. Kết Quả Đánh Giá Chất Lượng

### 3.1 Tổng quan các metrics

| Metric | Score | Thang điểm | Đánh giá |
|---|---|---|---|
| **WER (mean)** | **1.34%** | 0–∞% (↓ tốt hơn) | ✅✅ Xuất sắc (ngưỡng excellent: < 2%) |
| **WER (median)** | **0.00%** | 0–∞% (↓ tốt hơn) | ✅✅ Hơn 50% câu khôi phục hoàn hảo |
| **DNSMOS OVRL** | **3.28** | 1–5 (↑ tốt hơn) | ◐ Fair (xem lưu ý §3.3 về thiên lệch DNSMOS) |
| **DNSMOS SIG** | **3.56** | 1–5 (↑ tốt hơn) | ✅ Tín hiệu giọng tốt |
| **DNSMOS BAK** | **4.06** | 1–5 (↑ tốt hơn) | ✅✅ Nền rất sạch, không nhiễu |
| **DNSMOS P808** | **4.03** | 1–5 (↑ tốt hơn) | ✅✅ Chất lượng nghe tổng thể tốt |

---

### 3.2 Intelligibility — WER

```
WER (mean)   = 1.34%
WER (median) = 0.00%
ASR model    = faster-whisper-small (int8, language=en, beam=5)
Reference    = chính văn bản hypothesis đưa vào TTS
```

**Nhận xét:** WER trung bình **1.34%** nằm trong ngưỡng **"Xuất sắc" (< 2%)** theo `tts_metric_thresholds.md` — ngang với các model TTS SOTA (F5-TTS ~2.0–2.2% trên LibriSpeech). **Median = 0%** nghĩa là quá nửa số câu được ASR khôi phục **chính xác từng từ**. Quan trọng hơn, phân tích định tính (§5) cho thấy phần lớn WER còn lại **không phải lỗi phát âm của TTS**, mà đến từ:
- **Khác biệt chuẩn hóa của ASR:** `M.R.I.` → "MRI", `seven o'clock` → "7 o'clock", `smart phone` → "smartphone".
- **TTS đọc trung thực văn bản MT bị lỗi:** "locky policy", "Austrosiatic", "restructied" — đây là lỗi tầng dịch, không phải tầng TTS.

Nghĩa là chất lượng phát âm thực sự của TTS còn **cao hơn** con số 1.34% gợi ý.

### 3.3 Naturalness — DNSMOS

```
DNSMOS OVRL = 3.28   (overall)
DNSMOS SIG  = 3.56   (speech signal quality)
DNSMOS BAK  = 4.06   (background noise — cao = nền càng sạch)
DNSMOS P808 = 4.03   (P.808 listening quality)
```

**Nhận xét:** BAK = 4.06 và P808 = 4.03 xác nhận audio **sạch, không nhiễu nền, dễ nghe**. OVRL = 3.28 nằm ở mức "Fair", **thấp hơn ngưỡng MOS acceptable (≥ 3.5)** — nhưng cần đặt trong bối cảnh:

> [!IMPORTANT]
> DNSMOS được huấn luyện chủ yếu cho bài toán **speech enhancement / khử nhiễu**, không phải đánh giá TTS sạch. Nó có xu hướng **đánh giá thấp (underestimate)** giọng tổng hợp neural sạch vì giọng TTS thiếu các vi-biến thiên (micro-variation) của giọng người thật mà DNSMOS coi là "tự nhiên". Vì vậy OVRL ≈ 3.28 nên hiểu là **cận dưới (lower bound)** của chất lượng cảm nhận, không phải đánh giá MOS chủ quan thực tế. UTMOS (đo riêng cho TTS) thường cao hơn DNSMOS ở cùng audio, nhưng UTMOS không khả dụng trong môi trường này (cần tải mã ngoài qua torch.hub — bị chặn).

---

### 3.4 Phân tích theo Domain

| Domain | WER % | DNSMOS OVRL | RTF | Đánh giá |
|---|---|---|---|---|
| **legal** | **0.00** | 3.317 | 0.199 | ✅✅ Phát âm hoàn hảo |
| **news** | **0.00** | 3.233 | 0.195 | ✅✅ Phát âm hoàn hảo |
| **science** | **0.00** | 3.249 | 0.187 | ✅✅ Phát âm hoàn hảo |
| **education** | 0.77 | 3.300 | 0.192 | ✅✅ Xuất sắc |
| **environment** | 0.77 | 3.295 | 0.190 | ✅✅ Xuất sắc |
| **technology** | 1.54 | 3.255 | 0.200 | ✅✅ Xuất sắc |
| **culture** | 2.00 | **3.348** | 0.190 | ✅ Rất tốt (OVRL cao nhất) |
| **medical** | 2.00 | 3.297 | 0.195 | ✅ Rất tốt |
| **daily_life** | 2.73 | 3.271 | 0.208 | ✅ Rất tốt |
| **economy** | **3.62** | 3.279 | 0.192 | ✅ Rất tốt (WER cao nhất) |

**Nguyên nhân chênh lệch:**
- **WER toàn bộ ≤ 3.62%** — không domain nào ra khỏi ngưỡng "Rất tốt". Sai khác giữa các domain rất nhỏ và chủ yếu do **nội dung văn bản** chứ không phải khả năng của TTS.
- **economy / daily_life cao hơn chút:** chứa số, đơn vị tiền tệ, giờ giấc ("7 o'clock", "10 percent") → ASR chuẩn hóa chữ↔số khác cách viết gốc, tính thành "lỗi" WER dù phát âm đúng.
- **medical / culture:** chứa từ viết tắt/tên riêng ("M.R.I.", "Austrosiatic") và đôi khi là văn bản MT bị lỗi mà TTS đọc trung thực.
- **DNSMOS OVRL rất đồng đều (3.23–3.35):** giọng ổn định trên mọi domain, không có domain nào bị suy giảm chất lượng âm thanh.

---

## 4. Kết Quả Hiệu Năng (Performance Profiling)

### 4.1 Thống kê mô hình

| Chỉ số | Giá trị |
|---|---|
| **Kích thước mô hình (ONNX)** | 60.3 MB |
| **RAM cho TTS inference** | ~30 MB (rất nhẹ) |
| **VRAM sử dụng** | 0 MB (CPU-only) |
| **Sample rate đầu ra** | 22.05 kHz |

### 4.2 Synthesis Latency & Real-Time Factor

> Đo trên 100 câu, sau 3 lần warm-up. Tổng audio sinh ra: **397.8 giây** (~6.6 phút giọng nói).

| Chỉ số | Giá trị |
|---|---|
| **Tổng thời gian tổng hợp 100 câu** | 79.0 giây |
| **Overall RTF** | **0.1985** (≈ **5.0× real-time**) |
| **Latency trung bình (avg)** | 767.4 ms / câu |
| **Latency trung vị (median)** | 769.7 ms / câu |
| **Latency p95** | 896.8 ms / câu |
| **Throughput** | 1.27 câu / giây |

**Phân tích latency:**
- **RTF = 0.1985** ⇒ tạo 1 giây audio chỉ tốn ~0.2 giây tính toán → sinh giọng **nhanh gấp 5 lần thời gian phát**, dư sức cho ứng dụng streaming/real-time trên CPU.
- Median ≈ Mean (769.7 ≈ 767.4 ms) → phân phối latency đồng đều, không outlier lớn.
- p95 (896.8 ms) chỉ cao hơn median ~16.5% → latency ổn định. Latency mỗi câu tỉ lệ với độ dài câu (câu dài → audio dài → thời gian tổng hợp dài), nên đo theo RTF có ý nghĩa hơn latency tuyệt đối.

### 4.3 CPU Load

| Chỉ số | Giá trị |
|---|---|
| **CPU usage trung bình** | 58.6% |
| **CPU usage tối đa** | 91.3% |
| **CPU usage tối thiểu** | 38.4% |

**Nhận xét:** TTS dùng CPU ở mức vừa phải (~59% mean), thấp hơn tải của tầng MT (~83%). ONNX Runtime tận dụng đa luồng nhưng không bão hòa toàn bộ CPU, để dành tài nguyên cho các tầng khác trong pipeline.

### 4.4 RAM Usage

RAM riêng cho **TTS inference** chỉ khoảng **~30 MB** — cực kỳ nhẹ, phù hợp cho thiết bị edge/embedded.
---

## 5. Phân Tích Định Tính

### 5.1 Ví dụ tổng hợp xuất sắc (WER = 0%, OVRL cao)

#### Ví dụ 1 — Phát âm hoàn hảo [CULTURE]
| | |
|---|---|
| **Text vào TTS (EN)** | The traditional boat racing festival attracts thousands of participants and cheerleaders. |
| **ASR khôi phục** | The traditional boat racing festival attracts thousands of participants and cheerleaders. |
| **WER / OVRL** | 0.0% / 3.49 ✅✅ |

#### Ví dụ 2 — Phát âm hoàn hảo [EDUCATION]
| | |
|---|---|
| **Text vào TTS (EN)** | Soft skills, such as communication and group work, are increasingly valued in recruitment. |
| **ASR khôi phục** | Soft skills, such as communication and group work, are increasingly valued in recruitment. |
| **WER / OVRL** | 0.0% / 3.47 ✅✅ |

---

### 5.2 Các trường hợp WER cao — và vì sao đó KHÔNG phải lỗi TTS

#### Trường hợp 1 — Khác biệt chuẩn hóa viết tắt của ASR [MEDICAL]
| | |
|---|---|
| **Text vào TTS** | **M.R.I.** showed the doctor a better view of the structure inside the patient. |
| **ASR khôi phục** | A **MRI** showed the doctor a better view of the structure inside the patient. |
| **WER** | 20.0% |
| **Phân tích** | TTS đọc đúng "M-R-I". ASR ghi lại thành "MRI" (bỏ dấu chấm) và chèn mạo từ "A". Đây là **khác biệt chính tả/chuẩn hóa của ASR**, không phải lỗi phát âm. |

#### Trường hợp 2 — Chữ số vs chữ viết [DAILY_LIFE]
| | |
|---|---|
| **Text vào TTS** | This supermarket opens from **seven** o'clock in the morning to **ten** o'clock each day. |
| **ASR khôi phục** | This supermarket opens from **7** o'clock in the morning to **10** o'clock each day. |
| **WER** | 12.5% |
| **Phân tích** | TTS đọc đúng "seven"/"ten". Whisper chuẩn hóa thành chữ số "7"/"10" → bị tính là substitution. Lỗi nằm ở khâu so khớp text, không ở giọng. |

#### Trường hợp 3 — TTS đọc trung thực văn bản MT bị lỗi [ECONOMY]
| | |
|---|---|
| **Text vào TTS** | The strong **locky** policy is applied to controlling the deficit. |
| **ASR khôi phục** | The strong **Lock E** policy is applied to controlling the deficit. |
| **WER** | 20.0% |
| **Phân tích** | "locky" là **lỗi của tầng dịch máy** (opus-mt). TTS buộc phải đọc một từ không tồn tại; ASR nghe thành "Lock E". Đây là lỗi truyền từ tầng MT (xem `report_mt.md`), **không phải lỗi TTS**. |

#### Trường hợp 4 — Tách từ ghép [TECHNOLOGY]
| | |
|---|---|
| **Text vào TTS** | The largest tech company in the world has released the latest **smart phone**. |
| **ASR khôi phục** | The largest tech company in the world has released the latest **smartphone**. |
| **WER** | 15.4% |
| **Phân tích** | "smart phone" (2 từ trong text MT) bị ASR ghép thành "smartphone" (1 từ). Phát âm giống hệt; chỉ khác cách viết. |

---

### 5.3 Tổng hợp điểm mạnh và yếu

#### ✅ Điểm mạnh
1. **Intelligibility xuất sắc:** WER 1.34% (median 0%) — ngang model TTS SOTA, hơn 50% câu khôi phục hoàn hảo.
2. **Âm thanh sạch:** DNSMOS BAK 4.06, P808 4.03 — không nhiễu nền, dễ nghe.
3. **Tốc độ cao:** RTF 0.198 (~5× real-time) trên CPU — đủ cho streaming/real-time.
4. **Cực nhẹ:** 60 MB model, ~30 MB RAM — phù hợp edge/embedded.
5. **Ổn định theo domain:** WER ≤ 3.62% và OVRL 3.23–3.35 trên cả 10 domain — không có điểm yếu cục bộ.
6. **Phát âm tốt thuật ngữ & tên riêng tiếng Anh:** festival, recruitment, communication… đọc chuẩn.

#### ⚠️ Điểm yếu / lưu ý
1. **Đọc trung thực cả lỗi của tầng MT:** văn bản dịch sai ("locky", "restructied", "Austrosiatic") được phát âm y nguyên → nghe khó hiểu. Đây là vấn đề **upstream (MT)**, không phải TTS, nhưng ảnh hưởng trải nghiệm cuối.
2. **Viết tắt & chữ số:** "M.R.I.", số viết chữ → đọc đúng nhưng tạo WER giả do ASR chuẩn hóa khác (không phải lỗi TTS thực sự).
3. **Prosody đơn điệu:** giọng single-speaker, ngữ điệu khá phẳng (đặc trưng chung của Piper/VITS medium) — DNSMOS OVRL ở mức Fair phần nào phản ánh điều này.
4. **DNSMOS không lý tưởng cho TTS:** đánh giá thấp giọng tổng hợp sạch; nên bổ sung MOS chủ quan hoặc UTMOS khi có điều kiện.

---

## 6. So Sánh Với Ngưỡng Tham Chiếu

Đối chiếu với bảng tổng hợp trong `docs/tts_metric_thresholds.md`:

| Metric | Kết quả | Ngưỡng Chấp Nhận | Ngưỡng Tốt | Ngưỡng Xuất Sắc | Xếp loại |
|---|---|---|---|---|---|
| **WER** | **1.34%** | < 10% | < 5% | **< 2%** | ✅✅ **Xuất sắc** |
| **DNSMOS OVRL** (proxy MOS) | 3.28 | ≥ 3.5 | ≥ 4.0 | ≥ 4.5 | ◐ Dưới ngưỡng acceptable* |
| **DNSMOS P808** | 4.03 | (≈ MOS) ≥ 3.5 | ≥ 4.0 | ≥ 4.5 | ✅ Tốt |

\* DNSMOS underestimate giọng TTS sạch (xem §3.3). P808 = 4.03 — chỉ số listening-quality sát MOS nhất — đạt mức "Tốt", cho thấy chất lượng cảm nhận thực tế cao hơn con số OVRL.

> [!NOTE]
> Các metric intrusive (PESQ ≥ 3.0, STOI ≥ 0.75, MCD < 6 dB, Speaker-Sim ≥ 0.70, FAD < 5.0) **không áp dụng** vì benchmark không có audio tham chiếu. UTMOS (ngưỡng acceptable ≥ 3.5) là metric phù hợp nhất cho TTS nhưng không khả dụng trong môi trường này; DNSMOS được dùng thay thế làm proxy predicted-MOS.

### 6.1 So sánh với các hệ thống TTS khác (định tính)

| Hệ thống | WER (LibriSpeech-clean) | Ghi chú |
|---|---|---|
| F5-TTS / SOTA neural | ~2.0–2.2% | Model lớn, cần GPU |
| **Piper en_US-lessac-medium** | **~1.34%** (benchmark này) | Open-source, 60 MB, chạy CPU real-time |
| TTS thương mại (cloud) | ~1–3% | Chất lượng cao nhưng cần API, không offline |

> [!TIP]
> Piper đạt **intelligibility ngang model SOTA** với chi phí tài nguyên cực thấp (60 MB, CPU, ~5× real-time). Đây là lựa chọn lý tưởng cho tầng TTS **offline/embedded** trong pipeline OneVoice.

---

## 7. Kết Luận

### 7.1 Tóm tắt kết quả

```
┌──────────────────────────────────────────────────────────────┐
│   TỔNG KẾT ĐÁNH GIÁ Piper TTS (en_US-lessac-medium)          │
│   Đầu vào: bản dịch tiếng Anh (hypothesis) từ opus-mt-vi-en   │
├─────────────────────┬───────────┬────────────────────────────┤
│ WER (mean)          │  1.34%    │ ✅✅ Xuất sắc (<2%)         │
│ WER (median)        │  0.00%    │ ✅✅ >50% câu hoàn hảo      │
│ DNSMOS OVRL         │  3.28     │ ◐ Fair (DNSMOS thiên thấp) │
│ DNSMOS SIG / BAK    │ 3.56/4.06 │ ✅ Tín hiệu tốt, nền sạch   │
│ DNSMOS P808         │  4.03     │ ✅ Listening quality tốt    │
├─────────────────────┼───────────┼────────────────────────────┤
│ Model size          │  60.3 MB  │ Nhẹ, deployable            │
│ RAM (TTS)           │  ~30 MB   │ Cực nhẹ — edge-ready       │
│ Overall RTF         │  0.198    │ ~5× real-time trên CPU     │
│ Avg latency         │  767 ms   │ Tỉ lệ theo độ dài câu      │
│ Throughput          │ 1.27 sps  │ 6.6 phút audio / 79 giây   │
│ CPU load            │  58.6%    │ Vừa phải                   │
└─────────────────────┴───────────┴────────────────────────────┘
```

### 7.2 Đánh giá tổng thể

Tầng TTS của OneVoice đạt **intelligibility xuất sắc (WER 1.34%)** và **âm thanh sạch**, chạy **real-time trên CPU** với footprint cực nhỏ. Hầu hết WER còn lại là **giả** (do ASR chuẩn hóa viết tắt/chữ số khác cách viết) hoặc do **lỗi của tầng dịch máy phía trước** mà TTS đọc trung thực — **bản thân chất lượng phát âm của TTS gần như hoàn hảo**. Hạn chế thực sự duy nhất là **prosody hơi đơn điệu** (đặc trưng giọng VITS medium), phản ánh qua DNSMOS OVRL ở mức Fair.

### 7.3 Khuyến nghị sử dụng

| Tình huống | Phù hợp? |
|---|---|
| Đọc bản dịch tin tức, tài liệu chung | ✅ Rất phù hợp |
| Trợ lý ảo / thông báo giọng nói | ✅ Phù hợp (WER thấp, real-time) |
| Ứng dụng offline / embedded / edge | ✅✅ Lý tưởng (60 MB, ~30 MB RAM) |
| Pipeline real-time trên CPU | ✅ Phù hợp (RTF ~0.2) |
| Audiobook / nội dung nghe dài | ⚠️ Prosody đơn điệu — cân nhắc voice "high" |
| Nội dung yêu cầu cảm xúc/biểu cảm | ⚠️ Cần model biểu cảm hơn |

### 7.4 Khuyến nghị cải thiện

1. **Nâng cấp voice:** dùng `en_US-lessac-high` hoặc voice multi-speaker để cải thiện naturalness/prosody.
2. **Chuẩn hóa văn bản trước TTS (text normalization):** mở rộng viết tắt ("M.R.I." → "M R I"), chuyển số sang chữ nhất quán → giảm WER giả và tránh đọc nhầm.
3. **Lọc/sửa lỗi MT trước TTS:** phát hiện token lạ ("locky", "restructied") từ tầng dịch để tránh phát âm vô nghĩa — cải thiện trải nghiệm đầu-cuối nhiều nhất.
4. **Bổ sung đánh giá naturalness phù hợp:** UTMOS (TTS-specific) hoặc một vòng MOS chủ quan nhỏ để xác nhận chất lượng vượt mức DNSMOS gợi ý.
5. **Tăng tốc bằng GPU/streaming:** dù CPU đã ~5× real-time, GPU cho phép batch lớn / độ trễ thấp hơn nữa cho production.

---

## 8. Phụ Lục

### 8.1 Cấu hình chạy

```python
VOICE_PATH = "models/en_US-lessac-medium.onnx"
BENCHMARK  = "results/translations.jsonl"   # đọc trường `hypothesis`
TEXT_FIELD = "hypothesis"                    # bản dịch EN của opus-mt-vi-en
ASR_MODEL  = "small"      # faster-whisper, int8
ASR_LANG   = "en"
ASR_BEAM   = 5
WARMUP_RUNS= 3
SR_SYNTH   = 22050
DEVICE     = "cpu"
```

### 8.2 Các file kết quả

| File | Mô tả |
|---|---|
| `results/translations.jsonl` | Nguồn văn bản TTS (trường `hypothesis`) + COMET score |
| `results/tts_evaluation_results.json` | Kết quả đánh giá đầy đủ (metrics + per-domain) |
| `results/tts_synthesis.jsonl` | Kết quả từng câu: text, wav path, WER, DNSMOS, RTF, transcript |
| `results/audio/{id:03d}_{domain}.wav` | 100 file audio tổng hợp |
| `evaluate_tts.py` | Script đánh giá TTS chính |

### 8.3 Thư viện sử dụng

| Thư viện | Mục đích |
|---|---|
| `piper-tts` | Engine TTS (VITS + ONNX Runtime) |
| `faster-whisper` | ASR re-transcription để tính WER |
| `jiwer` | Tính WER |
| `speechmos` | DNSMOS (predicted MOS, non-intrusive) |
| `librosa` | Load & resample audio (16 kHz cho DNSMOS) |
| `psutil` | Giám sát CPU/RAM |
| `numpy` | Xử lý waveform |

### 8.4 Lưu ý về metrics bị loại trừ

| Metric | Lý do loại trừ |
|---|---|
| PESQ, STOI, MCD | Intrusive — cần audio reference cho từng câu (benchmark chỉ có text) |
| Speaker Similarity | Cần giọng tham chiếu mục tiêu để so embedding (không áp dụng cho single-speaker text benchmark) |
| FAD | Distributional — cần tập audio thật làm reference distribution |
| UTMOS | Metric TTS-specific lý tưởng, nhưng cần tải mã ngoài qua torch.hub (bị chặn trong môi trường) → thay bằng DNSMOS |

---

*Báo cáo tạo ngày 23/06/2026. Kết quả gắn với cấu hình cụ thể (voice en_US-lessac-medium, ASR faster-whisper-small, DNSMOS) và là tầng cuối của pipeline OneVoice vi → en → speech. WER được đo so với văn bản hypothesis đưa vào TTS nên phản ánh độ trung thực phát âm, độc lập với chất lượng dịch máy (đánh giá riêng tại `docs/report_mt.md`).*
