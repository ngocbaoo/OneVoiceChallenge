# Ngưỡng Tham Chiếu Các Metrics Đánh Giá Text-to-Speech (TTS)

> Tổng hợp ngưỡng chấp nhận được (acceptable), tốt (good) và xuất sắc (excellent) cho các metrics phổ biến trong đánh giá chất lượng tổng hợp tiếng nói (TTS Evaluation), kèm trích dẫn nguồn.

---

## 1. MOS — Mean Opinion Score

**Loại:** Chủ quan (subjective) — đánh giá bởi con người

**Thang điểm:** 1–5 (↑ tốt hơn)

MOS là tiêu chuẩn vàng (gold standard) trong đánh giá TTS, trong đó người nghe đánh giá chất lượng giọng tổng hợp trên thang 1–5.

| Điểm MOS | Nhãn | Diễn giải |
|----------|------|-----------|
| 1.0 | Bad | Hoàn toàn không tự nhiên (completely unnatural) |
| 1.5 | Very Poor | Hầu như không nhận diện được (mostly unrecognizable speech) |
| 2.0 | Poor | Chủ yếu không tự nhiên (mostly unnatural) |
| 2.5 | Poor | Bắt đầu khá tự nhiên nhưng vẫn rõ ràng nhân tạo |
| **3.0** | **Fair** | **Ngang bằng giữa tự nhiên và không tự nhiên** |
| **3.5** | **Satisfactory** | **Nhìn chung tự nhiên, có vấn đề nhỏ (generally natural with minor issues) ✅** |
| **4.0** | **Good** | **Hầu hết tự nhiên, nhiều vấn đề nhỏ (mostly natural with multiple minor issues) ✅** |
| **4.5** | **Very Good** | **Hầu hết tự nhiên, chỉ 1 vấn đề nhỏ (mostly natural with a minor issue) ✅✅** |
| 5.0 | Excellent | Hoàn toàn tự nhiên (completely natural) |

### Ngưỡng theo use case

| Use case | Ngưỡng MOS tối thiểu | Nguồn |
|----------|----------------------|-------|
| Conversational agents / Virtual assistants | ≥ 3.5 | [2] |
| Tổng quát — chất lượng tốt, tự nhiên | ≥ 4.0 | [1][2][3] |
| Audiobook narration (nghe dài) | ≥ 4.5 | [1] |
| Giọng nói tự nhiên (natural-sounding) | > 4.0 | [3] |
| Chấp nhận được nhưng có artifacts | 3.5–4.0 | [3] |
| Cần cải thiện đáng kể | < 3.5 | [3] |


### Lưu ý quan trọng

- MOS phụ thuộc vào thiết kế test, số lượng evaluators, và cách chọn mẫu. MOS giữa các nghiên cứu khác nhau **không nên so sánh trực tiếp** [1][4].
- FonadaLabs (2026) chỉ ra rằng *"some of the most disliked TTS voices in real-world products have excellent MOS scores"* vì MOS không đo được prosody, rhythm, và emotional appropriateness trong nghe dài [7].

---

## 2. PESQ — Perceptual Evaluation of Speech Quality

**Loại:** Khách quan (objective), intrusive — cần reference signal

**Thang điểm:** −0.5 đến 4.5 (raw PESQ) hoặc 1.0–5.0 (MOS-LQO) (↑ tốt hơn)

PESQ được phát triển bởi ITU-T (P.862) để đánh giá chất lượng giọng nói trong viễn thông, mô phỏng thính giác con người.

| Mức PESQ (raw) | Mức MOS-LQO tương đương | Diễn giải |
|----------------|------------------------|-----------|
| < 1.0 | ~1.0 | Cực kỳ kém, rất hiếm gặp |
| 1.0–1.9 | 1.0–2.0 | Kém — suy giảm chất lượng nghiêm trọng (severe quality degradation) |
| 2.0–2.9 | 2.0–3.0 | Trung bình — có vấn đề rõ ràng nhưng vẫn hiểu được (noticeable quality issues) |
| **3.0–3.9** | **3.0–4.0** | **Tốt — chất lượng chấp nhận được, lỗi nhỏ (acceptable quality with minor imperfections) ✅** |
| **4.0–4.5** | **4.0–5.0** | **Xuất sắc — chất lượng cao, gần như giống gốc (high-quality, nearly identical to original) ✅✅** |

### Ngưỡng ứng dụng

| Ngữ cảnh | Ngưỡng PESQ | Nguồn |
|----------|-------------|-------|
| Mạng GSM — chất lượng tốt | ≥ 3.0 | [8] |
| TTS neural hiện đại — mục tiêu | ≥ 3.5 | [14] |
| Chất lượng cao (clean speech) | ≥ 4.0 | [8] |


### Lưu ý quan trọng

- PESQ được thiết kế cho viễn thông (narrowband/wideband speech), **không phải TTS**. Khi áp dụng cho TTS, PESQ không phản ánh tốt prosody hay emotional expressiveness [12].
- PESQ giá trị dưới 1.0 rất hiếm, chỉ xảy ra khi tín hiệu bị biến dạng cực kỳ nghiêm trọng [13].

---

## 3. STOI — Short-Time Objective Intelligibility

**Loại:** Khách quan (objective), intrusive — cần reference signal

**Thang điểm:** 0–1 (↑ tốt hơn), thực tế thường trong khoảng 0.4–1.0

STOI đo khả năng hiểu được (intelligibility) của giọng nói bằng cách phân tích tương quan giữa temporal envelopes của tín hiệu gốc và tín hiệu tổng hợp/suy giảm.

| Mức STOI | Diễn giải |
|----------|-----------|
| < 0.50 | Rất kém, gần như không hiểu được |
| 0.50–0.65 | Kém, hiểu được một phần |
| **0.65–0.80** | **Trung bình đến khá** |
| **0.80–0.90** | **Tốt — hiểu rõ ✅** |
| **> 0.90** | **Rất tốt đến xuất sắc ✅✅** |
| ~1.00 | Giọng sạch (clean speech), hoàn hảo |

### Lưu ý quan trọng

- STOI đo **intelligibility** (có hiểu được không), không phải **naturalness** (có tự nhiên không). Một giọng TTS có thể có STOI rất cao nhưng MOS trung bình.
- STOI được thiết kế cho speech enhancement/noise reduction, khi áp dụng cho TTS cần kết hợp với MOS hoặc PESQ.

---

## 4. MCD — Mel-Cepstral Distortion

**Loại:** Khách quan (objective), intrusive — cần reference signal

**Thang điểm:** 0–∞ dB (↓ tốt hơn)

MCD đo khoảng cách phổ (spectral distance) giữa MFCC của giọng tổng hợp và giọng tham chiếu, phản ánh mức độ giống nhau về đặc trưng âm học.

| Mức MCD | Diễn giải |
|---------|-----------|
| **< 4 dB** | **Tốt — chất lượng tổng hợp cao (good quality) ✅✅** |
| **4–6 dB** | **Chấp nhận được ✅** |
| **> 6 dB** | **Méo đáng kể (significant distortion) ⚠️** |
| > 8 dB | Kém — cần cải thiện |

### Lưu ý quan trọng

- MCD phụ thuộc vào số chiều MFCC (D, thường 13–40) và phương pháp alignment (DTW vs. forced alignment). **Kết quả giữa các nghiên cứu không so sánh trực tiếp** nếu dùng cấu hình khác nhau [20].
- Taubert & Sternkopf (2025) khuyến nghị **luôn báo cáo đầy đủ tham số** khi dùng MCD. Nguồn: [github.com/stefantaubert/mel-cepstral-distance](https://github.com/stefantaubert/mel-cepstral-distance)

---

## 5. WER — Word Error Rate (trong ngữ cảnh TTS)

**Loại:** Khách quan (objective), gián tiếp — dùng ASR để đánh giá intelligibility của TTS

**Thang điểm:** 0–∞% (↓ tốt hơn)

Trong TTS, WER được tính bằng cách đưa audio tổng hợp qua một ASR model (thường Whisper) rồi so sánh transcript với văn bản gốc. WER thấp = TTS phát âm rõ ràng, chính xác.

| Mức WER | Diễn giải trong TTS |
|---------|---------------------|
| **< 2%** | **Xuất sắc — gần bằng ground truth ✅✅** |
| **2–5%** | **Rất tốt ✅** |
| **5–10%** | **Tốt, chấp nhận được ✅** |
| 10–20% | Trung bình — có vấn đề phát âm |
| > 20% | Kém — nhiều lỗi phát âm, cần cải thiện |
| > 100% | Giọng tổng hợp không hiểu được |

### Lưu ý quan trọng

- WER trong TTS **phụ thuộc vào ASR model** được dùng để transcribe. Whisper large cho kết quả khác NeMo Conformer. Luôn ghi rõ ASR model khi báo cáo.
- Ngay cả ground truth audio cũng có WER > 0% khi transcribe bằng ASR [25].

---

## 6. Speaker Similarity — Cosine Similarity

**Loại:** Khách quan (objective) — đo bằng speaker encoder embeddings

**Thang điểm:** −1 đến 1, thực tế 0–1 (↑ tốt hơn)

Cosine similarity giữa speaker embeddings (thường từ ECAPA-TDNN, d-vector, hoặc Resemblyzer) của giọng tổng hợp và giọng mục tiêu, đánh giá mức độ giữ đặc trưng giọng nói.

| Mức Cosine Similarity | Diễn giải |
|-----------------------|-----------|
| < 0.60 | Kém — khác speaker rõ ràng |
| **0.60–0.75** | **Chấp nhận được (acceptable cho speaker verification) ✅** |
| **0.75–0.85** | **Tốt (strong resemblance) ✅** |
| **0.85–0.95** | **Rất tốt (good voice cloning quality) ✅✅** |
| **> 0.95** | **Xuất sắc (near-perfect voice cloning) ✅✅✅** |

### Lưu ý quan trọng

- Speaker similarity **phụ thuộc vào speaker encoder** (d-vector, x-vector, ECAPA-TDNN, Resemblyzer). Các encoder khác nhau cho score khác nhau — không so sánh trực tiếp giữa các hệ thống dùng encoder khác nhau.
- Ngưỡng 0.6 đủ để phân biệt same/different speaker trong speaker verification [29], nhưng cho voice cloning chất lượng cao cần ≥ 0.85 [28].

---

## 7. UTMOS — Predicted MOS

**Loại:** Khách quan (objective), non-intrusive — không cần reference signal

**Thang điểm:** 1.0–5.0 (↑ tốt hơn), dự đoán MOS từ waveform

UTMOS (UTokyo-SaruLab MOS) là mô hình neural dự đoán MOS mà không cần reference audio. Nó sử dụng ensemble learning trên SSL features (wav2vec 2.0, HuBERT) để xấp xỉ đánh giá con người.

| Mức UTMOS | Diễn giải |
|-----------|-----------|
| < 3.0 | Kém |
| 3.0–3.5 | Trung bình |
| **3.5–4.0** | **Khá tốt ✅** |
| **4.0–4.5** | **Tốt — tự nhiên ✅✅** |
| **> 4.5** | **Xuất sắc ✅✅✅** |

### Ngưỡng theo downstream task (NVIDIA NeMo)

| Downstream task | Ngưỡng UTMOS khuyến nghị | Nguồn |
|-----------------|--------------------------|-------|
| TTS training data curation | ≥ 4.0 | [31] |
| ASR training data | ≥ 3.5 | [31] |
| Permissive curation | ≥ 3.0 | [31] |

### Lưu ý quan trọng

- UTMOS **bão hòa (saturates)** ở chất lượng rất cao — không phân biệt tốt giữa các model top-tier [35].
- UTMOS được huấn luyện chủ yếu trên tiếng Anh, hiệu quả cho các ngôn ngữ khác có thể thấp hơn.
- Luôn kết hợp UTMOS với MOS chủ quan khi đánh giá cuối cùng.

---

## 8. FAD — Fréchet Audio Distance

**Loại:** Khách quan (objective), distributional — so sánh phân phối audio

**Thang điểm:** 0–∞ (↓ tốt hơn)

FAD đo khoảng cách Fréchet giữa phân phối embeddings (thường VGGish hoặc CLAP) của audio tổng hợp và audio thật, tương tự FID trong image generation.

| Mức FAD | Diễn giải |
|---------|-----------|
| **< 1.0** | **Rất tốt — phân phối gần giống audio thật ✅✅** |
| **1.0–5.0** | **Tốt đến chấp nhận được ✅** |
| 5.0–10.0 | Trung bình — có sự khác biệt rõ ràng |
| > 10.0 | Kém — distortion đáng kể |


### Lưu ý quan trọng

- FAD **không có ngưỡng tuyệt đối chuẩn hóa** vì giá trị phụ thuộc hoàn toàn vào embedding model (VGGish, CLAP, PANNs) và reference set. Ngưỡng trên chỉ là tham khảo tổng quát.
- FAD là metric **distributional** (cần nhiều samples), không dùng được cho đánh giá từng utterance đơn lẻ.
- Trong TTS, FAD ít phổ biến hơn MOS/PESQ/MCD nhưng đang được áp dụng ngày càng nhiều [38].

---

## Bảng Tổng Hợp

| Metric | Thang điểm | Loại | Ngưỡng Chấp Nhận | Ngưỡng Tốt | Ngưỡng Xuất Sắc | Nguồn chính |
|--------|-----------|------|-------------------|-------------|------------------|-------------|
| **MOS** | 1–5 ↑ | Chủ quan | ≥ 3.5 | ≥ 4.0 | ≥ 4.5 | [1][2][3][4][5] |
| **PESQ** | −0.5–4.5 ↑ | Khách quan, intrusive | ≥ 3.0 | ≥ 3.5 | ≥ 4.0 | [8][9][10][11] |
| **STOI** | 0–1 ↑ | Khách quan, intrusive | ≥ 0.75 | ≥ 0.85 | ≥ 0.92 | [12][13][14][15] |
| **MCD** | 0–∞ dB ↓ | Khách quan, intrusive | < 6 dB | < 4 dB | < 3 dB | [16][17][18] |
| **WER** | 0–∞% ↓ | Khách quan, gián tiếp | < 10% | < 5% | < 2% | [21][22][23][24][25] |
| **Speaker Sim.** | 0–1 ↑ | Khách quan | ≥ 0.70 | ≥ 0.85 | ≥ 0.95 | [26][27][28][29][30] |
| **UTMOS** | 1–5 ↑ | Khách quan, non-intrusive | ≥ 3.5 | ≥ 4.0 | ≥ 4.3 | [31][33][34][35] |
| **FAD** | 0–∞ ↓ | Khách quan, distributional | < 5.0 | < 2.0 | < 1.0 | [36][37][38] |

---

## Khuyến nghị Đánh giá TTS Toàn diện

Không có metric đơn lẻ nào đủ để đánh giá chất lượng TTS. Một pipeline đánh giá toàn diện nên kết hợp:

**Naturalness (tự nhiên):** MOS (chủ quan) + UTMOS (tự động) — hai góc nhìn bổ trợ nhau.

**Intelligibility (hiểu được):** WER qua ASR + STOI — đảm bảo nội dung phát âm chính xác.

**Fidelity (trung thực):** PESQ + MCD — đo mức độ giống reference ở tầng tín hiệu.

**Speaker Identity (giữ giọng):** Cosine Similarity — quan trọng cho voice cloning/multi-speaker TTS.

**Distributional Quality:** FAD — đánh giá tổng thể trên tập dữ liệu lớn.

---

## Danh Mục Tài Liệu Tham Khảo

| # | Tài liệu | URL |
|---|----------|-----|
| [1] | FutureBeeAI (2026). "MOS Score for TTS Models: What is Considered Good?" | https://www.futurebeeai.com/knowledge-hub/mos-score-tts-models |
| [2] | Zilliz (2025). "What are the standard evaluation metrics for TTS quality?" | https://zilliz.com/ai-faq/what-are-the-standard-evaluation-metrics-for-tts-quality |
| [3] | "The Thiomi Dataset" (arXiv:2603.29244) — MOS thresholds for African TTS | https://arxiv.org/pdf/2603.29244 |
| [4] | WellSaid Labs (2026). "Defining Naturalness as Primary Driver for Synthetic Voice Quality" | https://www.wellsaid.io/resources/blog/naturalness-primary-driver-synthetic-voice |
| [5] | BnTTS: Few-Shot Speaker Adaptation (arXiv:2502.05729) | https://arxiv.org/pdf/2502.05729 |
| [6] | "Data Processing for Optimizing Naturalness of Vietnamese TTS" (arXiv:2004.09607) | https://arxiv.org/pdf/2004.09607 |
| [7] | FonadaLabs (2026). "How to Measure Voice Naturalness in TTS (Why MOS Is Not Enough)" | https://fonadalabs.ai/blog/how-to-measure-voice-naturalness-in-text-to-speech-why-mos-score-is-not-enough |
| [8] | RF Wireless World. "PESQ Score: Evaluating Speech Quality in GSM Networks" | https://www.rfwireless-world.com/test-and-measurement/pesq-score-evaluating-speech-quality-in-gsm-networks |
| [9] | Rix et al. (2001). "PESQ: A New Method for Speech Quality Assessment", ITU-T P.862 | https://www.researchgate.net/publication/3908525 |
| [10] | US Patent 7327985. "Mapping objective voice quality metrics to a MOS domain" | https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7327985 |
| [11] | Kolbæk et al. (arXiv:1808.10620). "Single-Microphone Speech Enhancement Using Deep Learning" | https://arxiv.org/pdf/1808.10620 |
| [12] | Taal et al. (2011). "An Algorithm for Intelligibility Prediction of TF Weighted Noisy Speech" | https://www.researchgate.net/figure/Performance-of-STOI-compared-with-five-other-reference-objective-intelligibility-models_fig8_224219052 |
| [13] | MathWorks. "Measuring Speech Intelligibility with STOI and ViSQOL" | https://www.mathworks.com/help/audio/ug/measure-speech-intelligibility-and-perceived-audio-quality-with-stoi-and-visqol.html |
| [14] | CallShield — PESQ > 4.2, STOI > 0.94 on clean speech | https://www.researchgate.net/figure/Scatter-plots-between-STOI-and-the-speech-intelligibility-scores-from-three-different_fig5_224219052 |
| [15] | Picovoice (2024). "Measuring Speech Intelligibility" | https://picovoice.ai/blog/speech-intelligibility/ |
| [16] | "Towards Controllable Speech Synthesis" Survey (arXiv:2412.06602) — MCD thresholds | https://arxiv.org/pdf/2412.06602 |
| [17] | Kominek et al. (2008). "Synthesizer voice quality calibrated with MCD" | https://learnius.com/slp/9+Speech+Synthesis/1+Fundamental+Concepts/3+Evaluation/mel+cepstral+distortion+(MCD) |
| [18] | Zilliz (2025). "TTS quality metrics" — MCD reference | https://zilliz.com/ai-faq/what-are-the-standard-evaluation-metrics-for-tts-quality |
| [19] | "Kinship in Speech" (arXiv:2506.03884) — MCD scores for Indian languages | https://arxiv.org/pdf/2506.03884 |
| [20] | Taubert & Sternkopf (2025). mel-cepstral-distance Python library | https://github.com/stefantaubert/mel-cepstral-distance |
| [21] | Zilliz (2025). "TTS quality metrics" — WER reference | https://zilliz.com/ai-faq/what-are-the-standard-evaluation-metrics-for-tts-quality |
| [22] | NVIDIA Riva. "Evaluate a TTS Pipeline" | https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/tts-evaluate.html |
| [23] | Way With Words (2024). "Word Error Rate: Assessing Transcription Accuracy" | https://waywithwords.net/resource/word-error-rate-transcription-accuracy/ |
| [24] | F5-TTS (arXiv:2410.06885), Traceable TTS (arXiv:2507.03887) — SOTA WER scores | https://arxiv.org/pdf/2410.06885 |
| [25] | "Towards Responsible Evaluation for TTS" (arXiv:2510.06927) | https://arxiv.org/pdf/2510.06927 |
| [26] | Zilliz (2025). "TTS quality metrics" — Speaker similarity reference | https://zilliz.com/ai-faq/what-are-the-standard-evaluation-metrics-for-tts-quality |
| [27] | Zero-shot voice cloning thesis (Univ. Groningen, 2025) | https://campus-fryslan.studenttheses.ub.rug.nl/708/1/MAs5965055QYZhu.pdf |
| [28] | Nepali voice cloning (arXiv:2601.18694) — Speaker similarity thresholds | https://arxiv.org/pdf/2601.18694 |
| [29] | Speaker verification — cosine similarity threshold 0.6 | https://www.researchgate.net/figure/The-distribution-of-cosine-similarity-of-speaker-embeddings-in-three-conditions_fig3_360792905 |
| [30] | NeuTTS Air (Medium, 2025) — Cloning accuracy benchmarks | https://medium.com/data-science-in-your-pocket/neutts-air-revolutionizing-on-device-text-to-speech-with-instant-voice-cloning-df3aadebc5cc |
| [31] | NVIDIA NeMo Curator. "UTMOS Filter" — Threshold guidelines | https://docs.nvidia.com/nemo/curator/curate-audio/process-data/quality-filtering/utmos |
| [32] | Saeki et al. (2022). "UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022" | Referenced in [33][34] |
| [33] | F5-TTS (arXiv:2410.06885) — UTMOS benchmark scores | https://arxiv.org/pdf/2410.06885 |
| [34] | Traceable TTS (arXiv:2507.03887) — UTMOS scores | https://arxiv.org/pdf/2507.03887 |
| [35] | Emergent Mind (2025). "UTMOS Score" — Saturation analysis | https://www.emergentmind.com/topics/utmos |
| [36] | Kilgour et al. (2019). "Fréchet Audio Distance", Interspeech 2019 | https://www.isca-archive.org/interspeech_2019/kilgour19_interspeech.pdf |
| [37] | Gudmundsson et al. (2024). "Adapting FAD for Generative Music Evaluation" | https://arxiv.org/html/2311.01616v2 |
| [38] | "Adapting the FAD as an Objective TTS Metric" (IEEE, 2025) | https://ieeexplore.ieee.org/document/11264402/ |

---

*Các ngưỡng mang tính tham khảo và phụ thuộc vào ngôn ngữ, domain, speaker encoder, ASR model, và phương pháp tính toán cụ thể. Luôn ghi rõ cấu hình khi báo cáo kết quả.*
