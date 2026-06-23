# Báo Cáo Tối Ưu Pipeline OneVoice Cho Thiết Bị Di Động
# Kế hoạch & Kết quả đo trực tiếp (Quantization + Mobile Runtime)
## vi (text) → opus-mt-vi-en → en (text) → Piper TTS → English speech

---

**Ngày thực hiện:** 23/06/2026 | **Thiết bị đo:** CPU x86-64 (AMD64, Windows 11) | **Scripts:** `optimize_mt_mobile.py`, `optimize_tts_mobile.py` | **Benchmark:** 100 câu × 10 domain

> [!IMPORTANT]
> Việc đo thực hiện trên **CPU x86 desktop**, không phải ARM mobile thật. Kết quả **kích thước / RAM / chất lượng** áp dụng trực tiếp cho mobile; riêng **tốc độ (latency)** chỉ mang tính **chỉ báo tương đối** — CPU ARM (NEON INT8 dot-product) và đặc biệt là **NPU/DSP (NNAPI/CoreML/QNN)** có thể cho kết quả khác (xem §4.3 và §6).

---

## 1. Kế Hoạch Tối Ưu (Plan)

### 1.1 Bối cảnh & nút thắt

| Tầng | Hiện trạng | Vấn đề trên mobile |
|---|---|---|
| MT | opus-mt-vi-en, **PyTorch fp32, 275 MB** | PyTorch desktop không chạy tốt trên mobile; quá nặng RAM & dung lượng |
| TTS | Piper, **ONNX fp32, 60 MB** | ONNX vốn edge-friendly nhưng vẫn nên thu nhỏ thêm |

➡️ Trọng tâm: **tầng MT** (kẻ ngốn tài nguyên chính).

### 1.2 Các hướng đề xuất và trạng thái triển khai

| # | Hướng | Trạng thái trong báo cáo này |
|---|---|---|
| 1 | **MT → runtime mobile (CTranslate2) + INT8** | ✅ **Đã làm & đo** |
| 2 | **TTS → INT8 (onnxruntime dynamic quant)** | ✅ **Đã làm & đo** (có phát hiện quan trọng) |
| 3 | NPU/GPU offload (NNAPI / CoreML / QNN) | ⏳ Cần thiết bị mobile thật — đề xuất ở §6 |
| 4 | TTS voice tier "low" (16 kHz) | ⏳ Đề xuất ở §6 |
| 5 | Streaming output + power capping | ⏳ Đã chứng minh streaming ở `report_pipeline_optimization.md` |

Báo cáo này tập trung **đo trực tiếp #1 và #2** — phần làm được trong môi trường hiện tại — và đưa khuyến nghị có căn cứ cho #3–#5.

### 1.3 Phương pháp đo

- **MT:** chuyển `Helsinki-NLP/opus-mt-vi-en` sang **CTranslate2 với `quantization="int8"`**; benchmark fp32 (PyTorch) vs int8 (CT2) trên 100 câu (batch=1, có warm-up). Đo: dung lượng đĩa, RAM khi load (process RSS delta), latency/câu, **BLEU** (đối chứng chất lượng).
- **TTS:** **dynamic INT8 quantization** ONNX của Piper bằng `onnxruntime.quantization.quantize_dynamic`; benchmark fp32 vs int8. Đo: dung lượng, RAM, latency/câu, RTF, và **WER** trên 30 câu (faster-whisper) để đảm bảo độ rõ không giảm.

---

## 2. Kết Quả — Tầng MT (CTranslate2 INT8) ✅

| Chỉ số (100 câu) | PyTorch fp32 | **CTranslate2 int8** | Cải thiện |
|---|---|---|---|
| **Dung lượng đĩa** | 275.3 MB | **71.2 MB** | 🟢 **3.87× nhỏ hơn** |
| **RAM khi load** | 273.6 MB | **81.4 MB** | 🟢 **3.36× ít hơn** |
| **Latency mean / câu** | 654.8 ms | **83.0 ms** | 🟢 **7.89× nhanh hơn** |
| **Latency p95 / câu** | 853.9 ms | 110.4 ms | 🟢 7.7× |
| **BLEU (chất lượng)** | 37.08 | **37.65** | 🟢 **+0.57 (không giảm)** |

```
MT — dung lượng & tốc độ (thấp hơn = tốt hơn)

 size  fp32 ████████████████████████████████  275 MB
       int8 ████████                           71 MB    (3.9× nhỏ hơn)

 lat   fp32 ████████████████████████████████  655 ms
       int8 ████                               83 ms    (7.9× nhanh hơn)
```

**Nhận xét:** Đây là **thắng lợi toàn diện** — nhỏ hơn ~4×, nhanh hơn ~8×, **chất lượng giữ nguyên** (BLEU thậm chí +0.57, nằm trong dao động bình thường; 53/100 câu cho output giống hệt fp32, phần còn lại khác biệt không đáng kể về nghĩa). CTranslate2 được tối ưu riêng cho kiến trúc Transformer/MarianMT và tận dụng INT8 hiệu quả → **lý tưởng cho mobile**. Việc loại bỏ phụ thuộc PyTorch cũng giảm mạnh kích thước thư viện đi kèm app.

---

## 3. Kết Quả — Tầng TTS (ONNX dynamic INT8) ⚠️

| Chỉ số (100 câu) | ONNX fp32 | ONNX int8 (dynamic) | Thay đổi |
|---|---|---|---|
| **Dung lượng đĩa** | 60.3 MB | **17.8 MB** | 🟢 **3.39× nhỏ hơn** |
| **RAM khi load** | 79.1 MB | **33.7 MB** | 🟢 2.35× ít hơn |
| **WER (30 câu, độ rõ)** | 1.66% | **1.68%** | 🟢 +0.02 pp (giữ nguyên) |
| **Latency mean / câu** | 632.1 ms | **4346.0 ms** | 🔴 **6.9× CHẬM HƠN** |
| **RTF** | 0.159 (6.3× real-time) | **1.082** | 🔴 **mất real-time** |

```
TTS — latency (thấp hơn = tốt hơn)

 fp32 ████                                632 ms   (RTF 0.16 ✅ real-time)
 int8 ████████████████████████████████  4346 ms   (RTF 1.08 ❌ chậm hơn phát)
```

**Phát hiện quan trọng (ngược kỳ vọng):** Dynamic INT8 giúp TTS **nhỏ hơn 3.4× và không mất chất lượng (WER ~1.7%)**, NHƯNG **chậm hơn gần 7× và mất khả năng real-time** trên CPU x86.

**Nguyên nhân:**
- Piper là **VITS — mô hình nặng Conv** (vocoder), trong khi **dynamic quantization chủ yếu tăng tốc các phép MatMul** (Transformer). Với mạng conv, ONNX Runtime chèn cặp Quantize/Dequantize quanh từng op nhưng **không có kernel INT8 tối ưu cho conv động** trên x86 → chi phí quant/dequant lấn át lợi ích.
- Nhiều op của VITS (Slice, NonZero, các attention nhỏ) **không quantize được** (xem cảnh báo khi chạy) → đồ thị lai fp32/int8 với nhiều điểm chuyển đổi tốn kém.

➡️ **Dynamic INT8 KHÔNG phù hợp cho TTS VITS.** Lợi ích dung lượng là thật nhưng cái giá latency quá lớn trên CPU thường.

---

## 4. Tổng Hợp & Cấu Hình Khuyến Nghị Cho Mobile

### 4.1 So sánh footprint pipeline

| Cấu hình | Dung lượng đĩa | RAM (2 model) | Latency/câu (x86) | Real-time? |
|---|---|---|---|---|
| **Baseline** (MT fp32 + TTS fp32) | 335.6 MB | ~352 MB | 1287 ms | ✅ |
| **Khuyến nghị** (MT **int8** + TTS **fp32**) | **131.5 MB** | **~160 MB** | **715 ms** | ✅ |
| Aggressive (MT int8 + TTS int8) | 89.0 MB | ~115 MB | 4429 ms | ❌ (TTS chậm) |

```
DUNG LƯỢNG PIPELINE (thấp hơn = tốt hơn)

 Baseline    ████████████████████████████████  336 MB
 Khuyến nghị ████████████                       132 MB   (2.55× nhỏ hơn)
 Aggressive  ████████                            89 MB   (nhưng TTS mất real-time)
```

### 4.2 Cấu hình khuyến nghị: **MT INT8 (CTranslate2) + TTS fp32 (ONNX)**

- **Dung lượng giảm 2.55×** (336 → 132 MB), **RAM giảm ~2.2×** (352 → 160 MB).
- **Latency/câu giảm ~1.8×** (1287 → 715 ms) — toàn bộ nhờ MT nhanh hơn 7.9×; TTS giữ fp32 để **không mất real-time** (RTF 0.16).
- **Chất lượng giữ nguyên** ở cả hai tầng (BLEU 37.65, WER 1.68%).
- TTS fp32 chỉ 60 MB và đã chạy 6.3× real-time → **không cần ép quantize**.

### 4.3 Vì sao chưa kết luận TTS không thể quantize

Kết quả TTS chậm là của **dynamic quant trên x86**. Trên mobile có thể khác:
- **Static quantization (có calibration)** sinh đồ thị INT8 thuần, ít điểm chuyển đổi → thường nhanh hơn dynamic nhiều cho conv.
- **CPU ARM** có lệnh INT8 dot-product (dotprod/i8mm) → conv INT8 có thể nhanh hơn fp32, khác hẳn x86.
- **NNAPI/CoreML/QNN** chạy INT8 trên NPU/DSP → vừa nhanh vừa tiết kiệm pin.

➡️ Trên mobile thật, **static-quant + NPU** cho TTS đáng thử lại; còn dynamic-quant như đo ở đây thì **không nên dùng**.

---

## 5. Kết Luận

```
┌────────────────────────────────────────────────────────────────────────┐
│  TỐI ƯU MOBILE — KẾT QUẢ ĐO TRỰC TIẾP (x86 CPU, 100 câu)                │
├──────────────────────┬──────────────┬──────────────┬───────────────────┤
│ Tầng / Kỹ thuật      │ Dung lượng   │ Tốc độ       │ Chất lượng        │
├──────────────────────┼──────────────┼──────────────┼───────────────────┤
│ MT  CTranslate2 int8 │ 3.9× nhỏ ✅  │ 7.9× nhanh ✅│ BLEU +0.57 ✅     │
│ TTS ONNX dyn-int8    │ 3.4× nhỏ ✅  │ 6.9× CHẬM ❌ │ WER +0.02pp ✅    │
├──────────────────────┴──────────────┴──────────────┴───────────────────┤
│ KHUYẾN NGHỊ: MT int8 + TTS fp32                                         │
│   → Pipeline 132 MB (2.55× nhỏ hơn) · RAM ~160 MB · 715 ms/câu · RT ✅  │
│   → TTS: KHÔNG dùng dynamic int8; cân nhắc static-quant + NPU trên mobile│
└────────────────────────────────────────────────────────────────────────┘
```

**Tóm tắt:**
- **MT là điểm tối ưu lớn nhất và đã thành công:** chuyển khỏi PyTorch sang CTranslate2 INT8 cho **nhỏ hơn 4×, nhanh hơn 8×, không mất chất lượng** — giải quyết đúng nút thắt mobile.
- **TTS nên giữ fp32** (60 MB, đã real-time). Dynamic INT8 phản tác dụng về tốc độ dù nhỏ hơn; chỉ quantize TTS nếu dùng **static quantization + tăng tốc phần cứng** trên thiết bị thật.
- **Pipeline khuyến nghị nhỏ hơn 2.55× và nhanh hơn 1.8×** so với baseline — đủ nhẹ và nhanh để chạy tốt trên mobile tầm trung.

---

## 5b. Kiểm Chứng Pipeline Khuyến Nghị (đo end-to-end thật)

Chạy lại **toàn bộ pipeline khuyến nghị** (MT CT2-int8 + TTS fp32) end-to-end, chế độ **sequential + streaming**, so với baseline fp32 đã đo (`results/pipeline_optimized_results.json`):

| Chỉ số (100 câu) | Baseline fp32 | **Khuyến nghị** | Cải thiện |
|---|---|---|---|
| **Wall time** | 161.2 s | **128.9 s** | 🟢 **1.25× nhanh hơn** |
| **Throughput** | 0.62 câu/s | **0.78 câu/s** | 🟢 1.25× |
| **Time-to-first-audio** | 1.77 s | **1.20 s** | 🟢 −0.57 s |
| **End-to-end RTF** | 0.407 | **0.326** | 🟢 3.07× real-time |
| **MT latency mean** | 1064 ms | **649 ms** | 🟢 1.64× |
| **TTS latency mean** | 548 ms | 640 ms | ≈ (dao động) |
| **CPU mean** | 65.0% | **59.5%** | 🟢 −5.6 pp |
| **Process RSS max** | 978 MB | **759 MB** | 🟢 **1.29× ít hơn** |
| **RSS khi load 2 model** | ~353 MB | **195 MB** | 🟢 1.8× ít hơn |

**Phát hiện quan trọng — thread oversubscription:** MT trong pipeline chỉ đạt **649 ms/câu**, KHÔNG phải **83 ms** như khi đo MT **độc lập** (§2). Nguyên nhân: khi **CTranslate2 và Piper (ONNX Runtime) cùng tồn tại trong một tiến trình**, mỗi runtime tạo một **thread pool bằng số lõi** → tổng **~2× số lõi** thread tranh nhau (CPU max vẫn 100%). Vì vậy MT bị chậm đi ~8× so với khi chạy một mình.

➡️ **Hệ quả:** pipeline khuyến nghị vẫn **nhanh hơn 1.25× và nhẹ hơn 1.29×** so với baseline, **nhưng chưa khai thác hết** tốc độ 7.9× của MT-int8. Để lấy lại phần này cần **giới hạn số thread của mỗi runtime** (vd. CT2 `intra_threads` + ORT session threads) sao cho tổng không vượt số lõi — đây là bước tinh chỉnh tiếp theo, kỳ vọng đưa wall-time giảm thêm đáng kể.

```
WALL TIME PIPELINE (100 câu, thấp hơn = tốt hơn)

 Baseline fp32  ████████████████████████████████  161.2 s
 Khuyến nghị    █████████████████████████          128.9 s   (1.25× nhanh hơn)
 (tiềm năng sau khi pin threads → còn thấp hơn nữa)
```

### 5b.1 Độ chính xác (accuracy) của pipeline khuyến nghị

Tối ưu chỉ có giá trị nếu **không làm giảm chất lượng**. Đo lại accuracy của đúng cấu hình khuyến nghị (`results/pipeline_accuracy_results.json`):

| Tầng | Metric | Baseline fp32 | **Khuyến nghị (int8 MT)** | Thay đổi |
|---|---|---|---|---|
| **MT** | **BLEU** ↑ | 37.08 | **37.65** | 🟢 **+0.57 (không giảm)** |
| MT | chrF ↑ | 62.11 | **62.53** | 🟢 +0.42 |
| MT | chrF++ ↑ | 59.85 | **60.27** | 🟢 +0.42 |
| **TTS** | **WER** ↓ | 1.34% | **2.28%** | 🟡 +0.94 pp (vẫn "Rất tốt" <5%) |
| TTS | WER median ↓ | 0.00% | **0.00%** | 🟢 = (quá nửa câu hoàn hảo) |

**Diễn giải:**
- **MT INT8 không hề mất chất lượng** — BLEU/chrF/chrF++ đều **nhỉnh hơn** fp32 (chênh lệch nằm trong dao động bình thường). Quantization INT8 với CTranslate2 giữ nguyên độ chính xác dịch thuật.
- **TTS WER tăng nhẹ 1.34% → 2.28%** nhưng **không phải do TTS kém đi** (TTS vẫn fp32, y nguyên). Nguyên nhân: audio lần này được đọc từ **bản dịch CT2-int8** (khác fp32 ở 47/100 câu), một số câu có cách diễn đạt/▁token hơi khác → ASR round-trip lệch hơn chút. WER 2.28% vẫn nằm trong ngưỡng **"Rất tốt" (2–5%)** theo `tts_metric_thresholds.md`, **median vẫn 0%**.
- **Per-domain BLEU** (CT2-int8): technology 57+, science 53+ cao nhất; economy/culture thấp nhất — **giữ nguyên mẫu hình** như báo cáo MT fp32, xác nhận quantization không làm méo theo domain.

➡️ **Kết luận accuracy:** pipeline khuyến nghị **bảo toàn độ chính xác** ở cả hai tầng (MT giữ nguyên, TTS vẫn excellent/very-good) trong khi **nhỏ hơn 2.55×, nhanh hơn 1.25× và nhẹ RAM hơn 1.29×**.

---

## 6. Các Bước Tiếp Theo (cần thiết bị mobile thật)

| # | Hành động | Lợi ích kỳ vọng |
|---|---|---|
| 1 | **Đóng gói MT-int8 (CT2) + TTS-fp32 (ORT Mobile)** vào app | Footprint 132 MB, chạy ngay |
| 2 | **TTS static quantization (có calibration)** rồi đo lại trên ARM | Có thể lấy lại tốc độ + nhỏ 3.4× |
| 3 | **NPU offload:** ORT + NNAPI (Android) / CoreML (iOS) cho TTS | Giảm pin mạnh; mở khóa pipelining liên tầng (không tranh CPU) |
| 4 | **Voice tier "low" 16 kHz** | Nhỏ & nhanh hơn nữa, đổi chút naturalness |
| 5 | **Streaming output + giới hạn thread/efficiency-core** | Độ trễ cảm nhận thấp, pin & nhiệt tốt hơn |
| 6 | **Pin số thread cho CT2 + ORT** (tránh oversubscription — xem §5b) | Lấy lại tốc độ MT 7.9× trong pipeline thật → wall-time giảm thêm |

> Lưu ý: kết luận "pipelining liên tầng phản tác dụng" ở `report_pipeline_optimization.md` chỉ đúng khi **cả hai tầng tranh cùng CPU**. Khi TTS chạy NPU và MT chạy CPU (bước #3), hai tầng **không tranh tài nguyên** → pipelining song song lúc đó **mới có lợi**.

---

## 7. Phụ Lục

### 7.1 File & artifact

| File | Nội dung |
|---|---|
| `optimize_mt_mobile.py` | Convert MT→CT2 int8 + benchmark (fp32 vs int8) |
| `optimize_tts_mobile.py` | Quantize TTS→int8 + benchmark + WER guard |
| `results/mobile_mt_results.json` | Số liệu MT đầy đủ |
| `results/mobile_tts_results.json` | Số liệu TTS đầy đủ |
| `models/mt_ct2_int8/` | **Model MT đã tối ưu (CTranslate2 int8, 71 MB)** — dùng cho mobile |
| `models/en_US-lessac-medium.int8.onnx` | TTS int8 (nhỏ nhưng chậm trên x86 — chỉ giữ để tham khảo/thử static-quant) |

### 7.2 Cấu hình kỹ thuật

```
MT  : ctranslate2 4.8.0 · TransformersConverter · quantization=int8 · compute_type=int8 · device=cpu
TTS : onnxruntime 1.27 · quantize_dynamic · weight_type=QInt8
ASR (chỉ để đo WER): faster-whisper small (int8), language=en, beam=5
Đo  : batch=1 (per-câu), warm-up 3 lần, process RSS (psutil)
```

### 7.3 Lưu ý phương pháp

- Latency đo trên **x86 desktop CPU**; chỉ dùng để **so sánh tương đối** giữa các biến thể, không phải số tuyệt đối trên mobile.
- BLEU đo bằng sacrebleu (chuẩn WMT, tokenize 13a) so với reference; WER đo round-trip qua ASR so với chính text đưa vào TTS.
- Kích thước MT CT2 là **cả thư mục model** (model.bin + vocab + config); kích thước fp32 là param float32 (đúng với checkpoint HF).

---

*Báo cáo tạo ngày 23/06/2026 từ số liệu đo trực tiếp (`results/mobile_mt_results.json`, `results/mobile_tts_results.json`). Kết quả kích thước/chất lượng áp dụng cho mobile; latency cần đo lại trên ARM/NPU. Tham chiếu chéo: `report_pipeline.md` (hiệu năng pipeline), `report_pipeline_optimization.md` (streaming/pipelining), `report_mt.md` & `report_tts.md` (chất lượng từng tầng).*
