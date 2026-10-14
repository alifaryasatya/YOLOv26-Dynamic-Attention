# Analisis BRA-P4 28 kelas vs baseline historis 30 kelas

Tanggal: 14 Oktober 2026.

**Status: perbandingan deskriptif historis, bukan ablation yang setara.** Baseline historis mempunyai skor validasi lebih tinggi, tetapi selisihnya tidak dapat diatribusikan hanya kepada penambahan Dynamic Attention karena jumlah kelas, mapping label, identitas split, dan protokol belum sepenuhnya setara.

## 1. Model dan sumber hasil

| Aspek | Baseline historis | BRA-P4 |
|---|---|---|
| Arsitektur | YOLO26n tanpa BRA | YOLO26n + BRA pada P4/layer 19 |
| Jumlah kelas checkpoint | 30 | 28 |
| Epoch selesai | 50 | 50 |
| Epoch dengan mAP50-95 tertinggi | 37 | 38 |
| Ukuran input / batch / seed | 640 / 16 / 0 | 640 / 16 / 0 |
| Optimizer dalam konfigurasi | `auto` | AdamW eksplisit, `lr0=0.000313` |
| Patience | 15 | 0, early stopping dinonaktifkan |
| Close mosaic | 10 | 10 |
| Parameter checkpoint sebelum fusion | 2.515.500 | 2.557.537 |

Sumber baseline:

- [results.csv](../runs/detect/yolov26_baseline_50e/results.csv).
- [args.yaml](../runs/detect/yolov26_baseline_50e/args.yaml).
- [best.pt](../runs/detect/yolov26_baseline_50e/weights/best.pt).

Sumber BRA:

- [results.csv](../runs/phase4_legacy_50e/bra_p4_50e_seed0/results.csv).
- [experiment_manifest.json](../runs/phase4_legacy_50e/bra_p4_50e_seed0/experiment_manifest.json).
- [best.pt](../runs/phase4_legacy_50e/bra_p4_50e_seed0/weights/best.pt).

Jumlah kelas diperiksa dari `model.names` checkpoint, bukan hanya judul dokumen atau nama file YAML. Baseline menyimpan referensi dataset pada komputer lama; kesamaan nama `plantdoc_data.yaml` tidak membuktikan isi dataset dan split identik dengan folder saat ini. `optimizer=auto` juga tidak berarti nilai `lr0` bawaan di YAML merupakan learning rate aktual yang dipakai.

## 2. Perbandingan checkpoint terbaik

Tabel ini memakai metrik validasi yang tersimpan pada masing-masing `best.pt`, yang sesuai dengan baris CSV epoch terbaik. **Ini bukan evaluasi ulang kedua model pada satu dataset yang sama.** Selisih adalah BRA dikurangi baseline dalam poin persentase.

| Metrik | Baseline 30 kelas, epoch 37 | BRA 28 kelas, epoch 38 | Selisih BRA |
|---|---:|---:|---:|
| Precision | 58,574% | 55,227% | -3,347 poin |
| Recall | 53,817% | 45,678% | -8,139 poin |
| mAP50 | 55,722% | 51,640% | -4,082 poin |
| mAP50-95 | 39,546% | 35,538% | -4,008 poin |

Secara angka historis, baseline lebih tinggi pada semua metrik. Selisih terbesar adalah recall. Namun, perbedaan ini tidak membuktikan bahwa BRA menyebabkan lebih banyak objek terlewat pada tugas yang identik: label dan populasi evaluasinya belum disamakan.

## 3. Perbandingan epoch terakhir

Epoch 50 tidak sama dengan checkpoint terbaik. Berikut angka yang tercatat pada epoch terakhir untuk kedua run:

| Metrik | Baseline 30 kelas | BRA 28 kelas | Selisih BRA |
|---|---:|---:|---:|
| Precision | 49,879% | 53,852% | +3,973 poin |
| Recall | 57,749% | 47,308% | -10,441 poin |
| mAP50 | 54,753% | 50,724% | -4,029 poin |
| mAP50-95 | 38,417% | 35,219% | -3,198 poin |

Precision BRA lebih tinggi pada epoch terakhir, tetapi recall dan kedua mAP tetap lebih rendah secara historis. Karena itu, kesimpulan dapat berbeda bila hanya melihat precision atau mencampur epoch terakhir satu model dengan checkpoint terbaik model lain. Tabel utama menggunakan best vs best secara konsisten.

## 4. Perbedaan mapping kelas

Indeks 0–14 memiliki nama kelas yang sama. Mulai indeks 15, mapping berubah:

| Nama kelas | ID baseline 30 kelas | ID BRA 28 kelas |
|---|---:|---:|
| Soyabean leaf | 15 | Tidak tersedia |
| Soybean leaf | 16 | 15 |
| Squash Powdery mildew leaf | 17 | 16 |
| Tomato leaf | 25 | 24 |
| Tomato two spotted spider mites leaf | 27 | Tidak tersedia |
| grape leaf black rot | 28 | 26 |
| grape leaf | 29 | 27 |

Keberadaan dua nama soybean pada baseline diverifikasi dari checkpoint. Ini tidak dengan sendirinya membuktikan bagaimana gambar/label historis digabung atau dibuang saat dataset menjadi 28 kelas; riwayat transformasi lengkap diperlukan untuk memastikan hal tersebut.

Mengevaluasi checkpoint 30 kelas langsung dengan label 28 kelas akan membandingkan ID yang salah setelah pergeseran tersebut. Contohnya, ID 16 berarti Soybean pada baseline tetapi Squash pada dataset BRA. **Jangan sekadar mengganti `nc`, mengedit `model.names`, atau menghapus dua nama untuk menyamakan model.** Bobot head dan anotasi perlu mapping yang benar; kelas yang hilang juga mengubah definisi tugas dan perhitungan agregat mAP.

## 5. Analisis training BRA

BRA menyelesaikan 50 epoch dan mencapai puncak mAP50-95 pada epoch 38. Pada epoch 50, mAP50-95 turun dari 35,538% menjadi 35,219%: penurunan 0,319 poin. Training berjalan, tetapi epoch tambahan setelah puncak tidak meningkatkan skor terbaik.

Gate residual BRA pada checkpoint terbaik sekitar `0.03714`, dibanding inisialisasi nol. Cabang attention telah dipelajari, tetapi gate nonzero bukan bukti bahwa modul meningkatkan generalisasi. Penurunan loss training yang tidak diikuti peningkatan mAP validasi konsisten dengan plateau; bukan diagnosis pasti bahwa attention salah atau training gagal.

Kode terkait:

- [bra.py](../src/plantdoc_attention/modules/bra.py): Bi-Level Routing Attention.
- [residual_attention.py](../src/plantdoc_attention/modules/residual_attention.py): normalisasi dan residual gate.
- [yolo26.py](../src/plantdoc_attention/integrations/yolo26.py): wrapper P4 dan trainer integrasi.
- [legacy_bra_p4_50e.yaml](../configs/experiments/legacy_bra_p4_50e.yaml): konfigurasi run yang dianalisis.

Grafik asli BRA tersedia di [hasil bra](../hasil%20bra/README.md). Grafik loss dan metrik per epoch bukan metrik evaluasi ulang FP32; sumbernya tetap dibedakan.

## 6. Biaya model dan keterbatasan perbandingan

BRA checkpoint memiliki 42.037 parameter lebih banyak daripada baseline historis. Selisih ini **bukan overhead BRA murni**, karena head baseline dan BRA mempunyai jumlah kelas berbeda. Pada pasangan 28 kelas yang setara, tambahan parameter BRA adalah 42.817.

Tidak ada benchmark latency historis 30 kelas yang disamakan dan dijalankan ulang dalam pekerjaan ini. Oleh karena itu, laporan ini tidak mengklaim BRA lebih cepat atau lambat dari baseline historis berdasarkan durasi training pada dua waktu/lingkungan berbeda.

Perbandingan AP per kelas historis juga belum disajikan: CSV training dan metrik agregat checkpoint tidak cukup untuk merekonstruksi AP setiap kelas. Nama kelas yang sama belum menjamin objek validasi atau split yang sama.

## 7. Hubungan dengan evaluasi 28 kelas yang konsisten

Laporan ini berbeda dari [analisis pasangan 28 kelas](LEGACY_50E_EVALUATION_ANALYSIS.md). Pada evaluasi ulang FP32/fusion dengan dataset dan protokol yang sama, baseline 28 kelas mendapat **36,19% mAP50-95**, sedangkan BRA mendapat **35,52%**. Selisih sekitar **-0,66 poin** lebih relevan untuk menilai konfigurasi BRA sekarang daripada selisih historis -4,008 poin.

Nilai 35,538% BRA di tabel utama adalah metrik validasi saat training, sedangkan 35,52% adalah evaluasi ulang standalone yang terkontrol. Kedua sumber tidak dicampur diam-diam. Skor standalone awal baseline 28 kelas sebesar 35,08% merupakan hasil yang tidak berhasil direproduksi dan tidak digunakan untuk menyimpulkan manfaat attention.

## 8. Kesimpulan dan langkah lanjutan

### Ringkasan F1 ketiga model

Nilai berikut dibaca dari legenda `all classes` pada gambar F1-Confidence hasil validasi akhir training masing-masing checkpoint terbaik. Angka F1 dibulatkan oleh plot ke dua desimal; confidence ke tiga desimal. Ini bukan F1 training per epoch atau F1 baru dari evaluasi ulang FP32.

| Model | Jumlah kelas | F1 terbaik pada kurva (dibulatkan) | Confidence threshold |
|---|---:|---:|---:|
| Baseline historis | 30 | 0,54 (54%) | 0,337 |
| Baseline dataset lama saat ini | 28 | 0,50 (50%) | 0,310 |
| YOLO26n + BRA-P4 | 28 | 0,48 (48%) | 0,369 |

Sumber: [kurva baseline 30 kelas](../runs/detect/yolov26_baseline_50e/BoxF1_curve.png), [kurva baseline 28 kelas](../runs/phase4_legacy_50e/baseline_50e_seed0/BoxF1_curve.png), dan [kurva BRA 28 kelas](../runs/phase4_legacy_50e/bra_p4_50e_seed0/BoxF1_curve.png).

Pada angka plot yang dibulatkan, BRA lebih rendah sekitar **2 poin** daripada baseline 28 kelas dan **6 poin** daripada baseline historis 30 kelas. Perbandingan 28 vs 28 lebih relevan; perbandingan dengan 30 kelas tetap bersifat historis. Ketiga angka adalah puncak kurva pada threshold masing-masing, bukan perbandingan pada satu threshold confidence tetap. Rata-rata F1 per kelas tidak harus sama dengan hasil rumus `2 * mean_precision * mean_recall / (mean_precision + mean_recall)`.

**Kesimpulan deskriptif:** hasil baseline historis 30 kelas lebih tinggi daripada BRA 28 kelas, dengan selisih best-checkpoint mAP50-95 sekitar 4,01 poin. Tidak sah menyebut seluruh selisih tersebut sebagai kerugian akibat Dynamic Attention.

Untuk klaim efek attention pada proyek saat ini, gunakan pasangan baseline/BRA 28 kelas yang identik dan evaluasi konsisten; lanjutkan dengan beberapa seed untuk menilai stabilitas. Jika tujuan penelitian benar-benar mempertahankan tugas historis 30 kelas, pulihkan YAML, anotasi, split, dan protokol asli dahulu, lalu latih BRA pada tugas 30 kelas tersebut.

Laporan ini tidak menjalankan training baru, tidak mengevaluasi checkpoint 30 kelas menggunakan mapping 28 kelas, dan tidak mengubah dataset atau artefak hasil yang sudah ada.
