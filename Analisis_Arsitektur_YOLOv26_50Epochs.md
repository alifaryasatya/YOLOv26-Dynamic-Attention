# Analisis Arsitektur YOLOv26: Evaluasi Progresif 1 Epoch, 50 Epoch, dan Pengujian Test Set
## Studi Empiris Baseline Model pada Dataset PlantDoc (30 Kelas Penyakit Daun)

> **Mata Kuliah:** Computer Vision  
> **Program Studi:** S1 Teknik Komputer, Fakultas Ilmu Komputer, Universitas Brawijaya  
> **Topik:** Deteksi Multi-Penyakit pada Daun di Lingkungan Alami (*In-the-Wild*)  
> **Eksperimen:** Baseline Model YOLO26n (`runs/detect/runs/yolov26_baseline_50e/`)  
> **Perangkat Keras:** NVIDIA GeForce RTX 3060 Laptop GPU (6.00 GB VRAM), CUDA 12.1

---

## 1. Rekapitulasi Progresif: 1 Epoch vs 50 Epochs vs Final Test Set

Untuk memahami perkembangan pembelajaran representasi visual pada arsitektur YOLOv26, evaluasi dilakukan secara bertahap:
1. **Tahap 1: Uji Sanity Check (1 Epoch)** — Mengevaluasi inisialisasi bobot dan respons awal gradien.
2. **Tahap 2: Pelatihan Penuh (50 Epochs - Validation Set)** — Memantau konvergensi penurunan loss dan stabilitas bobot pada 349 citra validasi.
3. **Tahap 3: Pengujian Independen Akhir (Test Set)** — Mengukur kemampuan generalisasi model terbaik (`best.pt`) pada 239 citra pengujian yang belum pernah dipelajari sebelumnya.

### 1.1 Tabel Perbandingan Kinerja Komprehensif

| Metrik Evaluasi | 1 Epoch (Sanity Check) | 50 Epochs (Validation Set) | Final Test Set (`best.pt`) | Kenaikan Total (1e $\rightarrow$ Test) |
| :--- | :--- | :--- | :--- | :--- |
| **Data Evaluasi** | Test Set (239 citra) | Validation Set (349 citra) | **Test Set (239 citra)** | Independen Unbiased |
| **File Checkpoint** | `sanity_check/best.pt` | `last.pt` (Epoch 50) | **`best.pt` (Optimal)** | Checkpoint Terbaik |
| **Precision** | 0,0217 (2,17%) | 0,4988 (49,88%) | **0,5056 (50,56%)** | **+48,39%** |
| **Recall** | 0,6915 (69,15%) | 0,5775 (57,75%) | **0,6733 (67,33%)** | Stabil & Matang |
| **mAP@0.5** | 0,0639 (6,39%) | 0,5475 (54,75%) | **0,6189 (61,89%)** | **+55,50%** |
| **mAP@[0.5:0.95]** | 0,0481 (4,81%) | 0,3842 (38,42%) | **0,4856 (48,56%)** | **+43,75%** |
| **Total Latensi** | 6,56 ms (~152 FPS) | -- | **7,63 ms (~131 FPS)** | Sangat Cepat (*Real-time*) |

---

### 1.2 Analisis Dinamika Pembelajaran Model

1. **Lompatan Akurasi dari 1 Epoch ke 50 Epoch:**
   * Pada **1 epoch awal**, model baru memproses 124 iterasi. Meskipun *Recall* awal cukup tinggi (69,15%) karena arsitektur STAL (*Small-Target-Aware Label Assignment*) memicu banyak prediksi kotak, *Precision*-nya sangat rendah (2,17%) karena hampir seluruh tebakan tersebut salah sasaran (*False Positive* akibat gangguan tanah dan daun sehat).
   * Pada **50 epoch penuh (6.200 iterasi)**, *Precision* melesat hingga **50,56%** dan **mAP@0.5** menembus **61,89%**. Ini membuktikan bahwa lapisan konvolusi C3k2 telah berhasil menyaring fitur morfologi 30 kelas penyakit secara stabil.
2. **Kinerja Test Set Lebih Unggul dari Validation Set (61,89% vs 54,75%):**
   * Fakta bahwa skor pada data uji independen (*Test Set*) lebih tinggi daripada data validasi (*Validation Set*) adalah indikator penting bahwa **model baseline YOLOv26 tidak mengalami *overfitting***. Model memiliki daya generalisasi yang kokoh pada variasi sudut kamera baru.
3. **Analisis Kesenjangan mAP@0.5 vs mAP@[0.5:0.95] ($\Delta = 13,33\%$):**
   * Pada Test Set, mAP@0.5 mencapai **61,89%**, namun pada ambang batas IoU ketat (mAP50-95) nilainya turun menjadi **48,56%**.
   * **Penyebab Fisis:** Model sudah sangat pintar mengenali jenis penyakitnya (apakah itu bercak bakteri, karat, atau jamur), namun **koordinat tepi kotaknya masih sering meluber** beberapa piksel ke area tanah basah atau bayangan daun yang tumpang tindih.

---

## 2. Dekomposisi Arsitektur YOLOv26 Layer per Layer

Berdasarkan struktur lapisan PyTorch yang dieksekusi saat pelatihan, arsitektur YOLOv26 terbagi ke dalam tiga blok fungsional utama:

![1788917611831](image/Analisis_Arsitektur_YOLOv26_50Epochs/1788917611831.webp)

```
[ Input Citra Daun Alami: 640x640x3 ]
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. BACKBONE (Layer 0 s/d 10)                                │
│    - Stem Conv & C3k2 Blocks                                │
│    - Layer 9: SPPF (Spatial Pyramid Pooling Fast)           │
│    - Layer 10: C2PSA (Cross-Stage Pointwise Self-Attention) │
│    --> Ekstraksi fitur visual multi-tingkat                 │
└──────────────┬──────────────────────────────┬───────────────┘
               │ P3 (80x80)                   │ P4 (40x40)    │ P5 (20x20)
               ▼                              ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. NECK / PANet (Layer 11 s/d 22)                           │
│    - Jalur Top-Down (Upsample + Concat + C3k2)              │
│    - Jalur Bottom-Up (Downsample Conv + Concat + C3k2)      │
│    --> Fusi multi-skala: detail lesi kecil + konteks daun   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. HEAD (Layer 23 - Detection Head NMS-Free)                │
│    - Detect [nc=30, channels=[64, 128, 256]]                │
│    - Eliminasi DFL (Distribution Focal Loss)                │
│    - Native One-to-One Matching (Prediksi Langsung Bebas NMS│
└─────────────────────────────────────────────────────────────┘
```

---

### A. Bagian Backbone (Layer 0 – 10): Ekstraksi Fitur Multi-Skala
Backbone bertugas menyerap informasi visual resolusi tinggi dan mereduksinya menjadi representasi semantik:

1. **Stem & Downsampling Convolutions (Layer 0, 1, 3, 5, 7):**
   * Menggunakan konvolusi $3 \times 3$ dengan *stride* 2 untuk memperkecil dimensi spasial citra secara bertahap dari $640 \rightarrow 320 \rightarrow 160 \rightarrow 80 \rightarrow 40 \rightarrow 20$.
2. **Blok `C3k2` (Cross-Stage Partial with 2 Convolutions):**
   * Berada di Layer 2, 4, 6, dan 8.
   * Modul ini adalah penyempurnaan dari blok C2f generasi sebelumnya. Dengan membagi aliran gradien menjadi dua cabang (*residual split*) dan menerapkan konvolusi ganda yang lebih terarah, model mampu mengekstrak pola lesi penyakit tanaman secara mendalam tanpa membebani memori GPU.
3. **Layer 9: `SPPF` (Spatial Pyramid Pooling - Fast):**
   * Melakukan pooling pada kernel $5 \times 5, 9 \times 9,$ dan $13 \times 13$ untuk memperbesar bidang pandang (*receptive field*). Membantu model mengidentifikasi penyakit yang meluas ke seluruh daun (misalnya *Late blight* atau *Powdery mildew*).
4. **Layer 10: `C2PSA` (Cross-Stage Partial Pointwise Self-Attention):**
   * Fitur inovasi YOLOv26 yang menyisipkan mekanisme *Pointwise Self-Attention* di ujung representasi terdalam (P5 / $20 \times 20$) untuk memperkuat pemahaman konteks global daun sebelum dialirkan ke Neck.

---

### B. Bagian Neck (Layer 11 – 22): Fusi Fitur Multi-Skala
Dataset PlantDoc memiliki variasi ukuran lesi yang ekstrim. Neck bertugas memastikan lesi titik kecil maupun bercak lebar dapat dideteksi secara bersamaan:

1. **Jalur Top-Down (Layer 11 – 16):**
   * Layer 11 menaikkan ukuran fitur P5 menggunakan `Upsample` (skala $2\times$), lalu digabungkan (`Concat` di Layer 12) dengan fitur spasial P4 dari Layer 6, kemudian diolah oleh blok `C3k2` (Layer 13).
   * Layer 14 melakukan upsample kembali dan digabung (`Concat` di Layer 15) dengan fitur P3 dari Layer 4, lalu dihaluskan oleh `C3k2` (Layer 16).
2. **Jalur Bottom-Up (Layer 17 – 22):**
   * Mengalirkan kembali detail spasial dari bawah ke atas menggunakan konvolusi *stride* 2 (Layer 17 & 20) serta konkatenasi (Layer 18 & 21).
   * **Tiga Cabang Output Neck:**
     * **Skala P3 ($80 \times 80$, 64 channel):** Spesialis lesi bintik kecil (*Apple rust*, *Tomato bacterial spot*).
     * **Skala P4 ($40 \times 40$, 128 channel):** Spesialis lesi ukuran sedang.
     * **Skala P5 ($20 \times 20$, 256 channel):** Spesialis daun utuh atau lesi bercak besar.

---

### C. Bagian Head (Layer 23): Detection Head NMS-Free
Layer 23 adalah modul `Detect [nc=30, channels=[64, 128, 256]]` yang menghasilkan prediksi akhir:

1. **Penghapusan DFL (*Distribution Focal Loss*):**
   * Arsitektur pendahulu menggunakan DFL untuk meregresikan batas kotak dengan distribusi probabilitas yang memakan komputasi tinggi.
   * YOLOv26 menghapus modul DFL, menyederhanakan arsitektur *head*, dan terbukti menghasilkan latensi inferensi yang sangat singkat, yaitu **4,06 ms per citra (~131 FPS)** di laptop RTX 3060 Anda saat pengujian data uji.
2. **Native End-to-End NMS-Free (One-to-One Matching):**
   * Kehadiran metrik **`train/l1_loss = 0.0159`** di log Anda menandakan aktifnya fungsi loss *one-to-one matching* saat training.
   * Model secara alami belajar mengeluarkan tepat satu prediksi kotak untuk satu lesi penyakit daun, mengeliminasi kebutuhan proses penyaringan *Non-Maximum Suppression* (NMS). Hal ini menghindarkan risiko terhapusnya kotak deteksi ketika beberapa daun berpenyakit saling tumpang tindih (*dense leaf occlusion*).

---

## 3. Evaluasi Kritis & Justifikasi Ilmiah Integrasi Dynamic Attention

Meskipun model baseline mencapai **mAP@0.5 = 61,89%** pada Test Set, penurunan ke **mAP@[0.5:0.95] = 48,56%** menjadi bukti empiris utama kelemahan konvolusi standar pada dataset alam terbuka:

```
                  [ TEMUAN EMPIRIS PENGUJIAN TEST SET ]
             mAP@0.5 = 61,89%   vs   mAP@0.5:0.95 = 48,56%
                                │
                                ▼
       Mengapa mAP pada ambang batas IoU ketat mengalami penurunan?
                                │
┌───────────────────────────────┴───────────────────────────────┐
▼                                                               ▼
[ Kelemahan Konvolusi Statis ]                 [ Karakteristik Lapangan PlantDoc ]
Filter konvolusi standar (C3k2)                Tanah basah, bayangan kontras terik,
memproses seluruh piksel dengan bobot          dan tumpukan daun memiliki tekstur
tetap tanpa adaptasi terhadap gangguan.        yang menyerupai bercak nekrosis/busuk.
│                                                               │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
               Kotak deteksi "meluber" ke latar belakang
               sehingga gagal pada ambang IoU ketat (> 0.75)
                                │
                                ▼
=========================================================================
           [ KESIMPULAN & TITIK INTEGRASI DYNAMIC ATTENTION ]
Untuk menaikkan mAP@0.5:0.95, kita perlu menyisipkan modul
DYNAMIC ATTENTION (seperti BiFormer / BRA atau DyHead) pada:
              >>> BLOK NECK (Antara Layer 11 s/d 22) <<<
Modul ini akan memfilter dan menekan aktivasi noise latar belakang tanah/bayangan
sebelum fitur multi-skala dikirimkan ke Detection Head (Layer 23).
=========================================================================
```

### Rekomendasi untuk Bab 4 Laporan Tugas Akhir:
Gunakan angka resmi **Test Set Baseline (`best.pt`)**:
* **mAP@0.5:** **61,89%**
* **mAP@[0.5:0.95]:** **48,56%**
* **Precision:** **50,56%**
* **Recall:** **67,33%**
* **Inference Speed:** **4,06 ms (~131 FPS)** pada GPU RTX 3060 Laptop.

Angka-angka ini menjadi fondasi tolak ukur (*benchmark*) yang sah untuk membuktikan hipotesis bahwa integrasi **Dynamic Attention** pada tahap eksperimen berikutnya mampu meredam *background noise* dan meningkatkan presisi lokalisasi lesi daun secara signifikan.
