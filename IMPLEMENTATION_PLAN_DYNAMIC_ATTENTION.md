# Implementation Plan: Dynamic Attention pada YOLOv26 PlantDoc

Dokumen ini menjadi rencana kerja sebelum dan selama penambahan *Dynamic Attention* ke baseline YOLOv26n. Fokus utamanya adalah memastikan perubahan arsitektur dapat diuji secara adil, direproduksi, dan benar-benar memberi manfaat pada citra PlantDoc yang memiliki background kompleks.

## 1. Keputusan teknis yang direkomendasikan

### Modul utama

Gunakan **Bi-Level Routing Attention (BRA)** sebagai implementasi Dynamic Attention pertama.

Alasannya:

- Routing bersifat *content-aware*: region relevan dipilih berdasarkan input, sehingga sesuai dengan hipotesis penyaringan tanah, bayangan, dan daun lain.
- Perhatian bersifat sparse melalui pemilihan top-k region, lebih masuk akal daripada global attention penuh pada feature map besar.
- Modul dapat dibuat *shape-preserving* sehingga tidak mengubah kontrak input detection head YOLOv26.
- Penempatannya dapat diuji satu per satu pada P3, P4, dan P5.

Referensi utama:

- [BiFormer: Vision Transformer with Bi-Level Routing Attention, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_BiFormer_Vision_Transformer_With_Bi-Level_Routing_Attention_CVPR_2023_paper.html)
- [Implementasi resmi BiFormer](https://github.com/rayleizhu/BiFormer)

### Titik integrasi awal

Mulai dari **P4 hasil Neck**, yaitu keluaran layer 19 pada konfigurasi YOLO26n lokal.

```text
Backbone -> FPN/PAN Neck -> P3 (layer 16, 64 channel)  ─┐
                         -> P4 (layer 19, 128 channel) -> BRA -> Detect
                         -> P5 (layer 22, 256 channel) ─┘
```

Pada input 640 × 640, P3/P4/P5 secara umum memiliki stride 8/16/32. P4 dipilih untuk eksperimen pertama karena menawarkan kompromi antara detail spasial, informasi semantik, penggunaan VRAM, dan latency.

Setelah P4 stabil, lakukan ablation berikut:

1. BRA pada P3 saja.
2. BRA pada P4 saja.
3. BRA pada P5 saja.
4. BRA pada P3 + P4.
5. BRA pada P3 + P4 + P5.

Jangan langsung memakai BRA pada semua level. Jika hasil meningkat, konfigurasi satu level diperlukan untuk membuktikan lokasi mana yang benar-benar berkontribusi.

### Alternatif tahap kedua

**Dynamic Head (DyHead)** dapat diuji setelah BRA selesai. DyHead memodelkan perhatian pada dimensi level, spasial, dan task sehingga menarik untuk lesi dengan ukuran sangat bervariasi. Namun, integrasinya lebih kompleks karena harus menerima beberapa level feature map sekaligus dan dapat membutuhkan deformable convolution.

Referensi: [Dynamic Head: Unifying Object Detection Heads with Attentions, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Dai_Dynamic_Head_Unifying_Object_Detection_Heads_With_Attentions_CVPR_2021_paper.html).

Environment saat ini sudah menyediakan `torchvision.ops.DeformConv2d`, tetapi ketersediaan operator saja belum menjamin kompatibilitas AMP, export, dan deployment. Karena itu DyHead ditempatkan sebagai fase lanjutan.

## 2. Kondisi proyek saat ini

### Environment terverifikasi

| Komponen      | Kondisi saat ini                              |
| ------------- | --------------------------------------------- |
| Python        | 3.13.3                                        |
| PyTorch       | 2.14.0+cu126                                  |
| Torchvision   | 0.29.0+cu126                                  |
| Ultralytics   | 8.4.143                                       |
| GPU           | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM |
| CUDA tersedia | Ya                                            |

### Dataset aktif

- 28 kelas.
- 1.795 gambar train.
- 386 gambar validation.
- 386 gambar test.
- 8.805 anotasi objek.
- Rasio kelas terbesar terhadap terkecil sekitar 76:1.

### Baseline yang tersedia

Hasil akhir validation pada eksperimen 50 epoch:

| Metrik       |  Nilai |
| ------------ | -----: |
| Precision    | 0,4988 |
| Recall       | 0,5775 |
| mAP@0.5      | 0,5475 |
| mAP@0.5:0.95 | 0,3842 |

Nilai tersebut berasal dari `runs/detect/yolov26_baseline_50e/results.csv`. Hasil ini belum cukup sebagai baseline eksperimen final karena environment belum dikunci dan hasil test belum disimpan sebagai artefak metrik terstruktur.

### Arsitektur YOLO26n lokal

Konfigurasi Ultralytics yang terpasang menghasilkan tiga fitur untuk detection head:

| Feature | Layer |   Channel YOLO26n | Fungsi utama                |
| ------- | ----: | ----------------: | --------------------------- |
| P3      |    16 |                64 | Detail dan objek/lesi kecil |
| P4      |    19 |               128 | Fitur skala menengah        |
| P5      |    22 |               256 | Konteks dan objek besar     |
| Detect  |    23 | menerima P3/P4/P5 | Regresi box dan klasifikasi |

YOLO26 sudah memiliki C2PSA di akhir backbone pada layer 10. Dynamic Attention tambahan harus ditempatkan dan diuji dengan hati-hati agar penelitian tidak hanya menambah blok attention yang redundan.

## 3. Yang wajib dilakukan sebelum menambah Dynamic Attention

Tahap ini merupakan prasyarat. Jangan mulai full training model attention sebelum semua butir P0 selesai.

### P0 — Reproducibility dan kebersihan eksperimen

- [ ] Buat `requirements.txt` atau lock file yang mencatat minimal Python, PyTorch, Torchvision, Ultralytics, PyYAML, NumPy, OpenCV, dan package profiling.
- [ ] Pindahkan atau salin skrip aktif dari `hasil lama/` ke direktori yang jelas seperti `scripts/`; folder lama tetap dipertahankan sebagai arsip.
- [ ] Perbaiki deskripsi skrip yang masih menyebut 30 kelas menjadi 28 kelas.
- [ ] Hilangkan default path checkpoint yang sudah tidak sesuai.
- [ ] Jangan menyimpan path absolut mesin lama dalam konfigurasi eksperimen baru.
- [ ] Pisahkan `runs/baseline`, `runs/attention`, `runs/ablation`, dan `runs/profile` atau gunakan nama eksperimen yang konsisten.
- [ ] Simpan seed, versi package, commit/source hash, hardware, hyperparameter, dan konfigurasi model pada setiap run.

Output tahap ini:

```text
requirements.txt
scripts/train.py
scripts/evaluate.py
scripts/profile.py
configs/experiments/baseline.yaml
```

### P0 — Audit dataset

- [ ] Pastikan setiap gambar memiliki file label pasangan dan sebaliknya.
- [ ] Validasi semua class ID berada pada rentang 0–27.
- [ ] Validasi koordinat YOLO berada pada rentang 0–1, width/height positif, dan bounding box tidak rusak.
- [ ] Cari duplikasi gambar atau near-duplicate lintas train/valid/test untuk mencegah data leakage.
- [ ] Pastikan hasil augmentasi dari gambar sumber yang sama tidak tersebar ke split berbeda.
- [ ] Rekam jumlah gambar, objek, dan distribusi kelas per split ke CSV/JSON.
- [ ] Hapus dan bangun ulang cache label setelah perubahan dataset.

Dataset tidak boleh diubah setelah eksperimen komparasi dimulai. Jika split berubah, baseline wajib dilatih ulang.

### P0 — Bekukan protokol baseline

- [ ] Tentukan satu protokol training yang akan dipakai oleh semua model: image size, epoch, batch, optimizer, learning rate, augmentasi, patience, AMP, worker, dan seed.
- [ ] Jalankan sanity check 1–3 epoch untuk memastikan pipeline bersih.
- [ ] Latih ulang baseline dengan source dan environment yang sudah dirapikan.
- [ ] Simpan metrik validation per epoch dan metrik per kelas.
- [ ] Simpan confusion matrix, prediction samples, parameter count, GFLOPs, peak VRAM, latency, dan FPS.
- [ ] Gunakan validation untuk pemilihan model. Jangan memakai test set untuk memilih lokasi attention atau hyperparameter.
- [ ] Jalankan final test hanya setelah desain model dibekukan.

Untuk penelitian yang lebih kuat, ulangi kandidat final dengan minimal tiga seed. Satu seed boleh dipakai saat screening awal untuk menghemat waktu.

### P0 — Tetapkan hipotesis dan kriteria keberhasilan

Hipotesis yang dapat diuji:

> BRA pada feature map hasil Neck akan meningkatkan kemampuan model membedakan area daun/lesi dari background kompleks, terutama pada ketepatan lokalisasi, tanpa menambah biaya inferensi secara berlebihan.

Kriteria keberhasilan yang disarankan untuk dibekukan sebelum eksperimen:

- Rata-rata mAP@0.5:0.95 validation dari tiga seed meningkat minimal 0,01 absolut dibanding baseline.
- Recall kelas minoritas tidak turun secara sistematis.
- Kenaikan latency tidak lebih dari 20% pada hardware dan batch yang sama.
- Kenaikan parameter tidak lebih dari 15% untuk model nano.
- Tidak ada NaN/Inf, error AMP, error reload checkpoint, atau kegagalan inferensi.
- Heatmap/attention map mendukung temuan kuantitatif; visualisasi bukan pengganti metrik.

Ambang tersebut adalah target eksperimen, bukan jaminan hasil.

### P1 — Siapkan static-attention control

Tambahkan satu model kontrol menggunakan CBAM atau SE pada titik yang sama dengan BRA. Tujuannya adalah menjawab apakah peningkatan berasal dari sifat dinamis BRA atau sekadar tambahan parameter/attention umum.

## 4. Strategi integrasi kode

### Prinsip implementasi

1. Jangan mengubah file di `.venv/Lib/site-packages/ultralytics` secara langsung.
2. Letakkan seluruh modul custom di repository.
3. Muat checkpoint pretrained baseline terlebih dahulu, lalu bungkus layer target dengan attention.
4. Pertahankan indeks layer YOLO agar bobot pretrained tidak kehilangan mapping.
5. Gunakan residual gate dengan nilai awal nol atau sangat kecil sehingga model awal ekuivalen atau sangat dekat dengan baseline.

Struktur yang disarankan:

```text
src/
└── plantdoc_attention/
    ├── __init__.py
    ├── modules/
    │   ├── bra.py
    │   ├── cbam_adapter.py
    │   └── residual_attention.py
    ├── integrations/
    │   └── yolo26.py
    └── metrics/
        └── experiment_report.py
configs/
└── experiments/
    ├── baseline.yaml
    ├── cbam_p4.yaml
    ├── bra_p3.yaml
    ├── bra_p4.yaml
    ├── bra_p5.yaml
    └── bra_p345.yaml
scripts/
├── train.py
├── evaluate.py
├── profile.py
└── visualize_attention.py
tests/
├── test_bra.py
├── test_yolo26_integration.py
└── test_checkpoint_reload.py
```

### Bentuk modul

Kontrak minimal BRA adapter:

```text
input : Tensor[B, C, H, W]
output: Tensor[B, C, H, W]
dtype : tetap mengikuti input
device: tetap mengikuti input
```

Blok disarankan memakai residual gating:

```text
y = x + gamma * BRA(Norm(x))
```

`gamma` diinisialisasi ke nol. Dengan cara ini, sebelum belajar, output wrapper sama dengan baseline. Ini mengurangi risiko merusak representasi pretrained pada awal fine-tuning dan menyediakan unit test yang tegas.

### Cara memasang tanpa menggeser layer

Alur integrasi yang direncanakan:

1. Muat `YOLO("yolo26n.pt")`.
2. Ambil module list dari model detection.
3. Bungkus layer target—awal: layer 19—dengan `ResidualAttentionWrapper`.
4. Wrapper menjalankan layer YOLO asli terlebih dahulu, lalu BRA.
5. Salin metadata graph Ultralytics yang diperlukan seperti `i`, `f`, dan atribut profiling ke wrapper.
6. Pastikan checkpoint custom dapat dimuat oleh script train, val, predict, dan export dengan package lokal tersedia.

Pendekatan wrapper dipilih agar bobot pretrained layer asli sudah termuat sebelum arsitektur dimodifikasi dan indeks layer berikutnya tidak berubah.

### Parameter awal BRA untuk screening

Parameter final harus ditentukan lewat profiling, tetapi titik awal yang aman adalah:

- Lokasi: P4.
- `topk`: 4.
- Jumlah region per sisi: 4 atau 5, disesuaikan agar H/W terbagi atau dipadding dengan benar.
- Attention heads: 4 untuk channel P4=128.
- Q/K dimension: 32 atau rasio 0,25 dari channel.
- Residual gate `gamma`: 0.
- Dropout attention: 0–0,1.

P3 membutuhkan perhatian khusus karena feature map lebih besar. Profiling memori wajib dilakukan sebelum full training P3.

## 5. Tahapan implementasi

### Fase 0 — Rapikan fondasi

Deliverable:

- Environment lock.
- Skrip train/evaluate/profile yang aktif.
- Audit dataset otomatis.
- Baseline rerun dan laporan benchmark.

Exit criteria:

- Baseline dapat dijalankan ulang dari perintah yang terdokumentasi.
- Hasil test tidak dipakai untuk tuning.
- Semua artefak benchmark tersimpan.

### Fase 1 — Implementasi BRA murni

Deliverable:

- Implementasi BRA lokal dengan referensi paper.
- Dokumentasi tensor shape dan padding.
- Unit test forward, backward, dtype, device, dan residual identity.

Exit criteria:

- Input dan output memiliki shape sama.
- Gradient mencapai seluruh parameter trainable.
- Berjalan pada CPU dan CUDA.
- Berjalan dengan AMP tanpa NaN/Inf.

### Fase 2 — Integrasi YOLO26 P4

Deliverable:

- Wrapper layer yang menjaga metadata graph Ultralytics.
- Model YOLO26n+BRA-P4.
- Script untuk membuat, menyimpan, memuat ulang, memvalidasi, dan memprediksi.

Exit criteria:

- Saat `gamma=0`, output model sebelum training mendekati output baseline dalam toleransi numerik.
- Satu batch forward/backward berhasil.
- Checkpoint dapat disimpan dan dimuat ulang.
- Model summary, parameter count, dan FLOPs dapat dihitung.

### Fase 3 — Smoke training

Urutan pengujian:

1. Overfit 4–16 gambar untuk membuktikan loss dapat turun.
2. Training 1 epoch untuk mendeteksi error dataloader, AMP, dan checkpoint.
3. Training 5 epoch untuk screening stabilitas dan penggunaan VRAM.
4. Full training hanya jika tiga langkah sebelumnya lolos.

Exit criteria:

- Loss finite dan cenderung turun.
- Tidak ada memory leak antarepoch.
- Peak VRAM masih aman pada GPU 6 GB.

### Fase 4 — Ablation study

Matriks minimum:

| ID | Model          | Lokasi                   | Tujuan                   |
| -- | -------------- | ------------------------ | ------------------------ |
| A  | YOLO26n        | Tanpa attention tambahan | Baseline                 |
| B  | YOLO26n + CBAM | P4                       | Static-attention control |
| C  | YOLO26n + BRA  | P4                       | Kandidat utama           |
| D  | YOLO26n + BRA  | P3                       | Uji lesi kecil           |
| E  | YOLO26n + BRA  | P5                       | Uji konteks global       |
| F  | YOLO26n + BRA  | P3+P4+P5                 | Uji multi-level          |

Semua konfigurasi harus memakai split, seed, epoch, augmentasi, batch, image size, pretrained weights, dan kebijakan early stopping yang sama.

Screening dapat memakai satu seed. Setelah kandidat dipilih dari validation, jalankan baseline, static control, dan kandidat final pada tiga seed.

### Fase 5 — Evaluasi final

Laporkan:

- Precision, recall, F1, mAP@0.5, dan mAP@0.5:0.95.
- AP dan recall per kelas.
- Hasil khusus kelas minoritas dan lesi kecil.
- Confusion matrix.
- Parameter, GFLOPs, ukuran checkpoint, peak VRAM.
- Latency preprocess/inference/postprocess dan FPS.
- Rata-rata serta standar deviasi antar-seed.
- Visualisasi true positive, false positive, dan false negative pada background kompleks.
- Heatmap baseline vs BRA pada gambar yang sama.

Gunakan validation untuk seluruh keputusan desain. Setelah desain dan checkpoint selection dibekukan, lakukan satu evaluasi final pada test split.

### Fase 6 — Export dan deployment check

Jika deployment termasuk ruang lingkup proyek:

- Uji export TorchScript dan ONNX.
- Bandingkan output PyTorch dan hasil export dengan toleransi numerik.
- Uji dynamic/static input shape sesuai target deployment.
- Dokumentasikan operator yang tidak didukung.
- Benchmark ulang latency pada perangkat target, bukan hanya laptop training.

BRA dengan operasi top-k/gather dinamis dapat menimbulkan kendala pada backend tertentu. Export harus dianggap sebagai acceptance test tersendiri, bukan diasumsikan otomatis berhasil.

## 6. Risiko dan mitigasi

| Risiko                                        | Dampak                                       | Mitigasi                                                                          |
| --------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------- |
| Attention redundan dengan C2PSA               | Parameter bertambah tanpa peningkatan        | Ablation lokasi dan static control; mulai P4, bukan P5 backbone.                  |
| Overfitting karena dataset kecil              | Validation naik sementara generalisasi buruk | Residual gate, regularisasi, early stopping, tiga seed, test hanya di akhir.      |
| Class imbalance menutupi kelas langka         | mAP agregat terlihat baik                    | Laporkan AP/recall per kelas dan macro summary.                                   |
| BRA P3 kehabisan VRAM                         | Training gagal atau batch terlalu kecil      | Mulai P4, profile P3, gunakan AMP/gradient accumulation bila perlu.               |
| Pretrained weights tidak termuat penuh        | Perbandingan tidak adil                      | Load baseline terlebih dahulu lalu wrap layer; verifikasi output saat gamma=0.    |
| Custom checkpoint gagal dimuat                | Val/predict/export tidak berjalan            | Import package lokal sebelum load dan buat test checkpoint round-trip.            |
| Perubahan versi Ultralytics merusak integrasi | Parser/metadata berubah                      | Pin versi 8.4.143 dan uji upgrade secara terpisah.                                |
| Test leakage                                  | Klaim performa bias                          | Validation untuk tuning; test hanya setelah desain dibekukan.                     |
| Implementasi paper tidak setara               | Klaim ilmiah lemah                           | Dokumentasikan perbedaan dari paper dan atribusi source; verifikasi lisensi kode. |

## 7. Urutan kerja yang disarankan

```text
Audit dataset
    ↓
Rapikan dan kunci environment
    ↓
Rerun + profile baseline
    ↓
Implementasi dan unit test BRA
    ↓
Integrasi BRA-P4 dengan residual gate
    ↓
Overfit mini-set → 1 epoch → 5 epoch
    ↓
Ablation lokasi dan static control
    ↓
Ulangi kandidat final pada 3 seed
    ↓
Bekukan desain
    ↓
Evaluasi test + visualisasi + deployment check
```

## 8. Definition of Done

Penambahan Dynamic Attention dinyatakan selesai apabila:

- Source custom tidak bergantung pada edit manual di `.venv`.
- Baseline dan seluruh kandidat dapat direproduksi dari config.
- Unit test forward/backward/AMP/checkpoint lulus.
- Tersedia bukti bahwa pretrained baseline dipertahankan sebelum fine-tuning.
- Ablation baseline, static attention, dan BRA selesai dengan protokol yang sama.
- Metrik accuracy dan efisiensi dilaporkan bersama, termasuk hasil per kelas.
- Kandidat final dievaluasi pada test hanya setelah desain dibekukan.
- Kesimpulan menyatakan hasil aktual, termasuk bila Dynamic Attention tidak memberi peningkatan yang signifikan.

## 9. Prioritas pengerjaan terdekat

Urutan implementasi berikutnya:

1. Buat environment lock dan skrip audit dataset.
2. Rapikan script training/evaluation dan jalankan baseline baru.
3. Tambahkan profiler parameter, FLOPs, VRAM, dan latency.
4. Implementasikan BRA beserta unit test.
5. Integrasikan BRA pada P4 menggunakan wrapper residual.
6. Jalankan smoke test sebelum full training.

Full training Dynamic Attention belum perlu dilakukan sebelum poin 1–5 lolos.
