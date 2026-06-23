# Báo Cáo Tối Ưu Pipeline OneVoice — Tối ưu #1
# Streaming + Pipelining hai tầng (MT → TTS) — Đo trực tiếp
## vi (text) → opus-mt-vi-en → en (text) → Piper TTS → English speech

---

**Ngày thực hiện:** 23/06/2026 | **Thiết bị:** CPU (AMD64, Windows 11, không GPU) | **Script:** `run_pipeline.py` | **Benchmark:** 100 câu × 10 domain

---

## 1. Mục Tiêu

Triển khai **tối ưu #1** đã đề xuất ở `report_pipeline.md` §6 (streaming + pipelining liên tầng) và **đo trực tiếp** pipeline MT → TTS trên cùng một tiến trình — thay vì cộng số liệu hai tầng đo rời. Tối ưu #1 thực ra gồm **hai kỹ thuật khác nhau**, và báo cáo này tách bạch để đánh giá đúng từng cái:

| Kỹ thuật | Ý tưởng | Kỳ vọng |
|---|---|---|
| **1a. Streaming output** | Phát audio câu *i* ngay khi xong, không chờ cả lô | Giảm **độ trễ cảm nhận** (time-to-first-audio) |
| **1b. Cross-stage pipelining** | MT (producer) và TTS (consumer) chạy **song song** qua hàng đợi | Giảm **tổng thời gian** nhờ gối tầng |

## 2. Thiết Kế Thí Nghiệm

Đo **2 chế độ trên cùng máy, cùng run, cùng granularity 1 câu/lần** để cô lập đúng lợi ích của pipelining:

- **Mode A — SEQUENTIAL:** dịch xong 1 câu → tổng hợp 1 câu → câu kế (không gối tầng).
- **Mode B — PIPELINED:** MT và TTS chạy trên 2 thread song song, nối nhau bằng `queue.Queue` (TTS xử lý câu *i* trong khi MT dịch câu *i+1*).

Cả hai model **thường trú** trong bộ nhớ; có warm-up; CPU/RAM(tiến trình) đo bằng monitor 0.25s. ASR/DNSMOS **không** chạy (không thuộc pipeline production).

> [!IMPORTANT]
> Ở chế độ streaming, MT chạy **batch = 1** (từng câu) để có thể đẩy ngay sang TTS. Đây là điểm đánh đổi quan trọng: MT batch=1 chậm hơn nhiều so với MT batch=8 (~109 ms/câu trong `report_mt.md`). Vì vậy **tổng thời gian tuyệt đối ở đây cao hơn** con số ước tính 90 s trong `report_pipeline.md` (vốn dựa trên MT batched). Trọng tâm của báo cáo này là **so sánh tương đối Sequential vs Pipelined**, không phải tốc độ tuyệt đối.

---

## 3. Kết Quả Đo Trực Tiếp

### 3.1 Bảng so sánh

| Chỉ số (100 câu) | SEQUENTIAL | PIPELINED | Thay đổi |
|---|---|---|---|
| **Wall time** | **161.2 s** | **193.5 s** | 🔴 **+32.2 s (chậm hơn 20%)** |
| **Time-to-first-audio (TTFA)** | 1.77 s | 1.65 s | 🟢 −0.13 s |
| **Throughput** | 0.62 câu/s | 0.52 câu/s | 🔴 0.83× |
| **End-to-end RTF** | 0.407 | 0.487 | 🔴 cao hơn (chậm hơn) |
| **CPU mean** | 65.0% | 60.4% | −4.6 pp |
| **CPU max** | **100%** | **100%** | (đều bão hòa) |
| **MT latency mean/câu** | 1064 ms | **1927 ms** | 🔴 +81% (do tranh CPU) |
| **TTS latency mean/câu** | 548 ms | **736 ms** | 🔴 +34% (do tranh CPU) |
| **Process RSS max** | 978 MB | 1093 MB | +115 MB (thêm thread/queue) |

```
WALL TIME (100 câu) — thấp hơn = tốt hơn

 SEQUENTIAL  ████████████████████████████████          161.2 s
 PIPELINED   ████████████████████████████████████████  193.5 s   ◄ chậm hơn 20%
```

### 3.2 Phát hiện chính (kết quả ngược kỳ vọng)

**Pipelining liên tầng (1b) KHÔNG tăng tốc trên CPU — ngược lại còn chậm hơn 20%.**

Nguyên nhân đã được số liệu xác nhận:
- **Cả hai tầng đều là CPU-bound và đã chiếm tối đa lõi.** CPU max = **100% ở cả hai chế độ**. MT (PyTorch) và TTS (ONNX Runtime) mặc định đều dùng đa luồng vắt kiệt mọi lõi.
- Khi chạy song song, hai tầng **tranh cùng một tập lõi CPU** → context-switching và thrashing. Bằng chứng trực tiếp: **latency từng tầng phình lên** khi pipelined (MT 1064→1927 ms, TTS 548→736 ms). Không có lõi "rảnh" để gối tầng, nên song song chỉ tạo thêm chi phí điều phối.
- CPU mean **giảm** (65→60%) dù chậm hơn — đúng dấu hiệu của tranh chấp tài nguyên: thời gian trôi vào chờ/chuyển ngữ cảnh thay vì tính toán hữu ích.

**Streaming (1a) là lợi ích THẬT và đã đạt được**, nhưng nó nằm ở **độ trễ cảm nhận**, không phải tổng thời gian:
- **TTFA ≈ 1.6–1.8 s** ở cả hai chế độ: người dùng nghe được audio sau ~1.7 giây, **thay vì chờ trọn 161 s** cho cả lô rồi mới phát.
- Vì **end-to-end RTF ≈ 0.41 < 1** (sinh audio nhanh hơn ~2.5× thời gian phát), sau câu đầu tiên việc phát liên tục **không bao giờ bị "đói" dữ liệu** — câu kế luôn sẵn sàng trước khi câu trước phát xong. Đây mới là giá trị UX của #1.

---

## 4. Diễn Giải

| Câu hỏi | Trả lời (có số liệu) |
|---|---|
| Streaming có giảm độ trễ cảm nhận không? | ✅ Có. TTFA ~1.7 s thay vì chờ 161 s cho cả lô. Phát liên tục không nghẽn vì RTF 0.41 < 1. |
| Pipelining song song có giảm tổng thời gian không? | ❌ Không, trên CPU này còn **chậm hơn 20%** do hai tầng cùng bão hòa CPU (max 100%) → tranh lõi. |
| Tại sao trái với kỳ vọng pipelining? | Pipelining chỉ thắng khi các tầng dùng **tài nguyên khác nhau** (vd. 1 tầng GPU, 1 tầng CPU/IO) hoặc khi **còn lõi rảnh**. Ở đây cả hai đều CPU-bound, không có dư địa song song. |
| Streaming có miễn phí không? | Gần như. Đổi lại MT phải chạy batch nhỏ (batch=1) → mất throughput batching của MT (đánh đổi latency↔throughput). |

---

## 5. Kết Luận & Khuyến Nghị (cập nhật)

### 5.1 Kết luận

- **Áp dụng streaming output (1a):** ✅ **Nên làm.** Đây là phần thực sự cải thiện trải nghiệm — độ trễ tới-audio-đầu-tiên ~1.7 s và phát liên tục mượt mà nhờ RTF < 1. Không cần phần cứng mới.
- **Cross-stage pipelining bằng thread song song (1b):** ❌ **Không nên trên CPU bão hòa.** Đo thực tế cho thấy chậm hơn 20%. Chỉ bật khi có **lõi CPU dư** hoặc khi đưa một tầng sang **GPU** để giải phóng CPU cho tầng kia.

### 5.2 Khuyến nghị triển khai trên CPU-only

1. **Giữ hai tầng nối tiếp per-câu** (Sequential) cho hiệu năng tính toán tốt nhất, **nhưng stream audio ra loa/đầu phát ngay khi mỗi câu xong** → người dùng nghe sau ~1.7 s, phần còn lại phát nối tiếp không gián đoạn.
2. **Không chạy MT và TTS song song trên cùng CPU** — sẽ phản tác dụng (đã đo).
3. **Cân bằng granularity:** nếu cần throughput cao hơn cho lô lớn, dùng MT **batch vừa** (vd. 4–8 câu) rồi stream TTS theo câu trong lô — dung hòa giữa TTFA thấp và throughput MT cao.
4. **Để pipelining thực sự có ích → cần tài nguyên khác biệt:** đưa TTS (hoặc MT) lên GPU/NPU. Khi đó tầng GPU và tầng CPU **không tranh nhau**, pipelining sẽ cho speedup thật. Đây là bước tiếp theo đáng làm nếu có GPU.

### 5.3 Bảng tổng kết

```
┌──────────────────────────────────────────────────────────────────────┐
│  TỐI ƯU #1 — KẾT QUẢ ĐO TRỰC TIẾP (CPU-only, 100 câu)                 │
├───────────────────────────┬───────────────┬──────────────────────────┤
│ 1a. Streaming output      │ ✅ Hiệu quả   │ TTFA ~1.7s, phát liên tục │
│ 1b. Pipelining song song  │ ❌ Phản tác dụng│ chậm hơn 20% (tranh CPU) │
├───────────────────────────┴───────────────┴──────────────────────────┤
│ Wall  : Sequential 161.2s  vs  Pipelined 193.5s  → 0.83× (chậm hơn)   │
│ TTFA  : ~1.7s (cả hai)     → giá trị thật của streaming               │
│ CPU   : max 100% cả hai    → không còn lõi để gối tầng                │
│ RAM   : ~308–395 MB (2 model resident) · RSS đỉnh ~1 GB khi chạy      │
│ Khuyến nghị: STREAM output, KHÔNG chạy 2 tầng song song trên 1 CPU    │
│             Pipelining thật chỉ đáng khi 1 tầng chuyển sang GPU       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. Phụ Lục

### 6.1 File liên quan

| File | Nội dung |
|---|---|
| `run_pipeline.py` | Script đo: Sequential vs Pipelined, MT→TTS trên cùng tiến trình |
| `results/pipeline_results.json` | Số liệu đầy đủ hai chế độ + so sánh |
| `results/audio_pipeline/*.wav` | Audio sinh ra trong lúc đo |
| `docs/report_pipeline.md` | Báo cáo hiệu năng tổng hợp (ước tính từ 2 tầng rời) |
| `docs/report_mt.md`, `docs/report_tts.md` | Chất lượng + hiệu năng từng tầng |

### 6.2 Lưu ý phương pháp

- Wall time tuyệt đối ở đây **cao hơn** ước tính 90 s trong `report_pipeline.md` vì dùng **MT batch=1** (cần cho streaming), chậm hơn MT batch=8. So sánh có ý nghĩa là **tương đối Sequential vs Pipelined**, không phải con số tuyệt đối.
- Trên CPU, cả PyTorch (MT) lẫn ONNX Runtime (TTS) đều mặc định đa luồng và bão hòa lõi — đây chính là lý do pipelining song song không có dư địa. Trên máy nhiều lõi hơn / có GPU, kết luận 1b có thể đổi (cần đo lại).
- CPU đo là **system-wide**; RSS đo là **của tiến trình pipeline** (psutil RSS) — sát thực tế tiêu thụ bộ nhớ của OneVoice hơn so với RAM toàn hệ thống.

---

*Báo cáo tạo ngày 23/06/2026 từ số liệu đo trực tiếp `results/pipeline_results.json`. Kết luận gắn với cấu hình CPU-only cụ thể; việc thêm GPU/NPU hoặc thay đổi số lõi sẽ thay đổi cán cân giữa streaming và pipelining — nên đo lại khi đổi phần cứng.*
