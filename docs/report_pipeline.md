# Báo Cáo Tổng Hợp Pipeline OneVoice
# Vietnamese Speech-to-Speech: vi (text) → MT → en (text) → TTS → English Speech
## Đánh giá hiệu năng end-to-end (latency, CPU, RAM, throughput)

---

**Ngày thực hiện:** 23/06/2026 | **Thiết bị:** CPU (AMD64, Windows 11, không GPU) | **Benchmark:** 100 câu × 10 domain

---

## 1. Tổng Quan Pipeline

OneVoice ghép **2 mô hình** chạy tuần tự để biến văn bản tiếng Việt thành giọng nói tiếng Anh:

```
  ┌───────────────┐     ┌──────────────────────┐     ┌───────────────┐     ┌───────────────────────┐     ┌──────────────┐
  │ Tiếng Việt    │ ──► │  Tầng 1: MT          │ ──► │ Tiếng Anh     │ ──► │ Tầng 2: TTS           │ ──► │ English      │
  │ (text)        │     │  opus-mt-vi-en       │     │ (text)        │     │ Piper en_US-lessac-med│     │ Speech (wav) │
  └───────────────┘     └──────────────────────┘     └───────────────┘     └───────────────────────┘     └──────────────┘
                          MarianMT, 72M params           hypothesis            VITS, ONNX Runtime
                          PyTorch CPU                                          22.05 kHz
```

| | Tầng 1 — MT | Tầng 2 — TTS |
|---|---|---|
| **Model** | `Helsinki-NLP/opus-mt-vi-en` | `en_US-lessac-medium` (Piper) |
| **Kiến trúc** | MarianMT (Transformer enc-dec) | VITS (neural vocoder + flow) |
| **Runtime** | PyTorch 2.12 (CPU) | ONNX Runtime (CPU) |
| **Báo cáo chi tiết** | `docs/report_mt.md` | `docs/report_tts.md` |

> [!NOTE]
> Báo cáo này **tổng hợp hiệu năng vận hành** của cả pipeline. Chất lượng từng tầng (BLEU/COMET cho MT; WER/DNSMOS cho TTS) đã được phân tích riêng trong hai báo cáo trên. ASR (Whisper) và DNSMOS chỉ là **công cụ đo lường để chấm điểm**, **không thuộc pipeline production** — vì vậy chúng **không** được tính vào latency/CPU end-to-end dưới đây.

---

## 2. Phương Pháp Đo

- Cả hai tầng đo trên **cùng một máy, cùng CPU**, không GPU, sau warm-up.
- Pipeline production = **MT + TTS** chạy tuần tự (đầu ra MT là đầu vào TTS).
- Các con số end-to-end được suy ra bằng cách **ghép tuần tự** số liệu đo độc lập của từng tầng trên cùng 100 câu. Đây là mô hình **xử lý nối tiếp một câu một** (không pipelining/batching chéo tầng) — phản ánh kịch bản đơn giản và bảo thủ nhất.
- Số liệu gốc lấy từ `results/evaluation_results.json` (MT) và `results/tts_evaluation_results.json` (TTS).

---

## 3. Bảng So Sánh Hai Tầng

| Chỉ số (trên 100 câu) | Tầng 1 — MT | Tầng 2 — TTS | Ghi chú |
|---|---|---|---|
| **Tổng thời gian xử lý** | 10.92 s | 78.98 s | TTS chậm hơn ~7.2× |
| **Latency trung bình / câu** | 109.0 ms | 767.4 ms | |
| **Latency trung vị / câu** | 107.6 ms | 769.7 ms | |
| **Latency p95 / câu** | 127.7 ms | 896.8 ms | |
| **Throughput** | 9.16 câu/s | 1.27 câu/s | TTS là nút cổ chai |
| **CPU mean** | 83.35% | 58.62% | |
| **CPU max** | 93.4% | 91.3% | |
| **RAM model (delta khi load)** | ~278 MB | ~30 MB | |
| **Kích thước model (đĩa)** | 275.3 MB | 60.3 MB | |
| **Đầu ra** | 2,357 tokens | 397.8 s audio | |

**Quan sát chính:** TTS **chi phối hoàn toàn** ngân sách thời gian — chiếm **87.9%** tổng thời gian xử lý của pipeline, dù mô hình nhỏ hơn 4.5× về dung lượng. Lý do: TTS phải sinh dạng sóng 22 kHz mẫu-theo-mẫu, khối lượng tính toán tỉ lệ với **độ dài audio**, trong khi MT chỉ sinh chuỗi token ngắn.

---

## 4. Hiệu Năng End-to-End (MT + TTS)

### 4.1 Thời gian & Throughput

| Chỉ số end-to-end | Giá trị | Cách tính |
|---|---|---|
| **Tổng thời gian xử lý 100 câu** | **89.90 s** | 10.92 (MT) + 78.98 (TTS) |
| **Latency trung bình / câu** | **~876 ms** | 109.0 + 767.4 ms |
| **Latency p95 / câu (xấp xỉ)** | **~1.02 s** | 127.7 + 896.8 ms |
| **Throughput end-to-end** | **1.11 câu/s** | 100 / 89.90 |
| **Tổng audio sinh ra** | 397.8 s (~6.6 phút) | |
| **End-to-end RTF** | **0.226** (≈ **4.4× real-time**) | 89.90 / 397.8 |

```
PHÂN BỔ THỜI GIAN / 1 CÂU TRUNG BÌNH  (tổng ~876 ms)

 MT  ████ 109 ms (12.4%)
 TTS ████████████████████████████████ 767 ms (87.6%)
     └────────────────────────────────────────────────► time
```

**Diễn giải:**
- **Độ trễ ~0.88 giây/câu** từ lúc có câu tiếng Việt đến lúc có audio tiếng Anh hoàn chỉnh — **dưới 1 giây**, đủ nhanh cho hội thoại gần real-time.
- **End-to-end RTF = 0.226** nghĩa là pipeline sinh giọng nói **nhanh hơn 4.4× thời gian phát**. Hệ thống thừa năng lực để **streaming**: vừa phát audio câu trước vừa xử lý câu sau.
- **Nút cổ chai là TTS.** Mọi tối ưu tốc độ nên tập trung vào tầng TTS (xem §6).

### 4.2 CPU

| Chỉ số | Giá trị | Cách tính |
|---|---|---|
| **CPU mean (trọng số theo thời gian)** | **~61.6%** | (83.35×10.92 + 58.62×78.98) / 89.90 |
| **CPU max (toàn pipeline)** | **93.4%** | max(93.4, 91.3) — đỉnh ở pha MT |
| **Mẫu hình tải** | MT ngắn-nặng, TTS dài-vừa | |

**Diễn giải:** Pipeline tải CPU trung bình ~62%. MT tạo **đỉnh ngắn 83–93%** (Transformer tận dụng tối đa lõi trong ~11 giây), sau đó TTS giữ tải **ổn định ~59%** trong phần lớn thời gian. Vì hai tầng chạy nối tiếp nên **không cộng dồn tải đồng thời** — một máy CPU 4–8 lõi thừa sức chạy cả pipeline.

### 4.3 Bộ nhớ (RAM)

| Chỉ số | Giá trị |
|---|---|
| **RAM cho model MT** | ~278 MB |
| **RAM cho model TTS** | ~30 MB |
| **RAM cả hai model thường trú** | **~308 MB** |
| **Tổng dung lượng model (đĩa)** | **335.6 MB** (275.3 + 60.3) |

**Diễn giải:** Nếu giữ cả hai model trong bộ nhớ (để tránh load lại mỗi câu), pipeline chỉ tốn **~308 MB RAM** — rất nhẹ, chạy được trên thiết bị phổ thông hoặc edge tầm trung. Nếu RAM eo hẹp, có thể load lần lượt nhưng sẽ thêm độ trễ khởi tạo.

---

## 5. Tóm Tắt End-to-End

```
┌─────────────────────────────────────────────────────────────────────┐
│   PIPELINE OneVoice  —  vi(text) → MT → en(text) → TTS → speech      │
│   100 câu · 10 domain · CPU-only                                    │
├──────────────────────────┬──────────────┬───────────────────────────┤
│ Tổng thời gian (100 câu)  │   89.9 s     │ MT 12% + TTS 88%          │
│ Latency / câu (avg)       │   ~876 ms    │ < 1 giây ✅               │
│ Latency / câu (p95)       │   ~1.02 s    │                           │
│ Throughput end-to-end     │   1.11 sps   │ nút cổ chai = TTS         │
│ End-to-end RTF            │   0.226      │ ~4.4× real-time ✅        │
│ CPU mean (trọng số)       │   ~61.6%     │ đỉnh 93% ở pha MT         │
│ CPU max                   │   93.4%      │ không cộng dồn (nối tiếp) │
│ RAM (2 model thường trú)  │   ~308 MB    │ nhẹ ✅                    │
│ Dung lượng model (đĩa)    │   335.6 MB   │ deployable ✅             │
├──────────────────────────┴──────────────┴───────────────────────────┤
│ CHẤT LƯỢNG (từ báo cáo riêng):                                      │
│   MT  : BLEU 37.1 · COMET 0.874        (report_mt.md)               │
│   TTS : WER 1.34% · DNSMOS-P808 4.03   (report_tts.md)              │
└─────────────────────────────────────────────────────────────────────┘
```

**Kết luận hiệu năng:** Toàn bộ pipeline vi→en→speech chạy **dưới 1 giây/câu** và **nhanh hơn ~4.4× thời gian phát audio**, chỉ tốn **~308 MB RAM** và **~62% CPU trung bình** trên một máy **không có GPU**. Đây là cấu hình **đủ nhẹ để triển khai offline/edge** và **đủ nhanh cho ứng dụng hội thoại gần real-time**.

---

## 6. Phân Tích Nút Cổ Chai & Khuyến Nghị

### 6.1 TTS là nút cổ chai (88% thời gian)

| Hướng tối ưu | Tác động kỳ vọng |
|---|---|
| **Streaming TTS** (phát câu i trong khi tổng hợp câu i+1) | Ẩn gần hết độ trễ TTS → cảm nhận như chỉ còn latency MT (~109 ms) |
| **Chạy TTS trên GPU/NPU** | RTF giảm 5–15× → throughput tăng mạnh |
| **Voice "low" thay "medium"** | Nhanh hơn nhưng đánh đổi naturalness |
| **Batch nhiều câu** | Tăng throughput tổng (không giảm latency/câu) |

### 6.2 Tối ưu chung pipeline

1. **Giữ cả hai model thường trú** (warm) trong dịch vụ — tránh ~hàng giây load lại mỗi request.
2. **Pipelining liên tầng:** ngay khi MT dịch xong câu, đẩy sang TTS trong khi MT xử lý câu kế → tận dụng cả CPU (MT) lẫn tổng hợp (TTS) song song, nâng throughput thực tế vượt 1.11 sps.
3. **GPU cho cả hai tầng:** MT giảm còn ~5–15 ms/câu, TTS giảm RTF mạnh → pipeline có thể đạt **chục–trăm câu/giây**.
4. **Chuẩn hóa văn bản giữa hai tầng:** xử lý số/viết tắt ở đầu ra MT trước khi đưa vào TTS → tránh TTS đọc nhầm và giảm WER giả (xem `report_tts.md` §5).

---

## 7. Phụ Lục

### 7.1 Nguồn số liệu

| File | Tầng | Nội dung |
|---|---|---|
| `results/evaluation_results.json` | MT | Quality + performance MT |
| `results/translations.jsonl` | MT→TTS | Bản dịch (hypothesis) nối hai tầng |
| `results/tts_evaluation_results.json` | TTS | Quality + performance TTS |
| `results/tts_synthesis.jsonl` | TTS | Kết quả từng câu |

### 7.2 Giả định khi ghép end-to-end

- Hai tầng đo **độc lập trên cùng một máy/CPU**; số end-to-end là **tổng tuần tự** (mô hình nối tiếp một-câu-một, không pipelining).
- **Không tính** thời gian/CPU của ASR (Whisper) và DNSMOS — đây là công cụ đánh giá, không thuộc pipeline production.
- Latency p95 end-to-end là **xấp xỉ** (cộng p95 của hai tầng); giá trị thực tế có thể thấp hơn vì hai đỉnh p95 hiếm khi rơi vào cùng một câu.
- CPU max **không cộng dồn** vì hai tầng chạy nối tiếp, không đồng thời.

---

*Báo cáo tổng hợp ngày 23/06/2026. Số liệu hiệu năng gắn với cấu hình CPU-only cụ thể; chuyển sang GPU hoặc bật streaming/pipelining sẽ thay đổi đáng kể latency và throughput. Chi tiết chất lượng từng tầng xem `docs/report_mt.md` và `docs/report_tts.md`.*
