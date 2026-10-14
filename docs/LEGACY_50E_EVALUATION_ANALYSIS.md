# Analisis baseline vs BRA-P4: dataset lama, 50 epoch

Tanggal analisis: 14 Oktober 2026. Dataset: `plantdoc_data.yaml`, 28 kelas.

## Hasil evaluasi awal

Kedua run menyelesaikan 50 epoch dan memiliki recipe training, seed, hash bobot awal, hash dataset, serta implementasi yang sama. Checkpoint yang dibandingkan adalah `best.pt`, bukan checkpoint historis 30 kelas.

| Metrik validasi | Baseline | BRA-P4 | Selisih BRA |
|---|---:|---:|---:|
| Precision | 52,99% | 55,10% | +2,12 poin |
| Recall | 48,08% | 45,57% | -2,50 poin |
| mAP50 | 50,91% | 51,60% | +0,69 poin |
| mAP50-95 | 35,08% | 35,52% | +0,44 poin |

Artefak awal: `artifacts/phase4_legacy_50e/comparison_phase4_legacy_50e/`.

## Hasil per kelas dan biaya

AP50-95 meningkat pada 12 kelas, menurun pada 15 kelas, dan tetap pada satu kelas. Peningkatan terbesar: Apple Scab Leaf +14,05 poin, Tomato leaf +6,28 poin. Penurunan terbesar: Bell_pepper leaf -4,60 poin dan Squash Powdery mildew leaf -3,04 poin. Rata-rata perubahan tanpa Apple Scab Leaf adalah -0,06 poin: keuntungan agregat belum menyeluruh.

Parameter bertambah 42.817 (+1,70%), GFLOPs +5,74%, waktu forward batch 1 FP16 13,76 menjadi 15,51 ms (+12,76%), dan peak VRAM training +10,40%. Forward benchmark tidak mencakup decode, resize, NMS, atau I/O. FPS validasi bukan benchmark deployment; waktu postprocessing kedua evaluasi awal sangat berbeda.

## Ketidaksesuaian skor yang ditelusuri

| Sumber skor mAP50-95 | Baseline | BRA-P4 |
|---|---:|---:|
| Validasi checkpoint terbaik saat training (epoch 38) | 36,24% | 35,54% |
| Evaluasi standalone awal | 35,08% | 35,52% |

Urutan model berbalik. Karena itu, kenaikan BRA pada evaluasi awal belum cukup untuk menyatakan bahwa attention lebih baik. Validasi training memakai model FP32 di bawah CUDA autocast dan input FP16; standalone memakai backend inference dengan fusion dan default precision berbeda. Penyebab spesifik perlu dibuktikan dengan evaluasi terkontrol, bukan diasumsikan.

## Evaluasi ulang terkontrol

Status: selesai. Semua evaluasi memakai split validasi; test dan training tidak dijalankan ulang. Hasil lama serta checkpoint tetap dipertahankan.

### Kontrol yang digunakan

Kedua model dievaluasi menggunakan `best.pt` pada 386 gambar validasi/1.343 objek yang sama. Dataset content SHA-256 dihitung ulang dan cocok dengan kedua manifest training: `be8a3b29a4a527244755b61a41ad03eb21860809f04976da69dea04650da9af8`. Hash kedua checkpoint juga tidak berubah.

Pengaturan utama: input 640, batch 16, CUDA 0, workers 2, seed 0, deterministic, `rect=True`, `conf=0.001`, `iou=0.7`, `max_det=300`, tanpa augmentasi. Jalur deteksi one-to-many/external NMS dipilih eksplisit (`nms=True`, runtime `head_end2end=False`). Pengaturan ini mengikuti head yang dipakai validasi training, bukan mengubah eksperimen menjadi head NMS-free.

Evaluasi utama menggunakan **FP32 dengan fusion aktif**. Pengaturan aktual dicatat, bukan hanya argumen yang diminta. Kedua model memiliki nol BatchNorm setelah fusion. Batas waktu wall-clock NMS dinonaktifkan pada pasangan akhir agar prediksi tidak dilewati hanya karena proses lambat; hasilnya identik dengan pasangan kontrol yang menggunakan batas waktu default. Perubahan NMS hanya berlaku sementara di proses diagnostik, tanpa mengedit library terpasang.

### Hasil utama yang konsisten

| Metrik | Baseline | BRA-P4 | Selisih BRA |
|---|---:|---:|---:|
| Precision | 53,38% | 55,10% | +1,72 poin |
| Recall | 49,70% | 45,57% | -4,12 poin |
| mAP50 | 52,43% | 51,60% | -0,83 poin |
| mAP50-95 | 36,19% | 35,52% | -0,66 poin |

Nilai mAP50-95 mentah adalah baseline `0.36187654379312717` dan BRA `0.355232042682279`. Pasangan FP32/fusion awal dan pasangan akhir tanpa deadline NMS menghasilkan angka yang sama. Pengulangan perintah evaluasi lama untuk baseline juga menghasilkan nilai baseline tersebut.

Artefak utama:

- [Baseline](../runs/evaluation/consistency_20261014_final_baseline_fp32_fusion_on/diagnostic_summary.json).
- [BRA-P4](../runs/evaluation/consistency_20261014_final_bra_p4_fp32_fusion_on/diagnostic_summary.json).
- [Pengulangan baseline dengan skrip/default lama](../runs/evaluation/consistency_original_defaults_20261014/metrics_summary.json).

Dengan baseline yang stabil, hitungan per kelas berubah menjadi **10 meningkat, 17 menurun, satu tetap**. AP Apple Scab Leaf baseline sebenarnya 42,63% pada evaluasi ulang, sedangkan BRA 38,91%: perubahan -3,72 poin, bukan kenaikan +14,05 poin dari laporan awal. Klaim keuntungan besar pada Apple Scab Leaf dari evaluasi awal tidak digunakan lagi.

Peningkatan AP50-95 terbesar yang masih bertahan: Tomato Early blight leaf +2,57 poin, Tomato leaf +2,32 poin, Apple rust leaf +2,16 poin. Penurunan terbesar: Squash Powdery mildew leaf -4,75 poin, Bell_pepper leaf -4,34 poin, Apple Scab Leaf -3,72 poin, dan Apple leaf -3,39 poin.

### Matriks diagnostik precision/fusion

| Model | Precision | Fusion | mAP50-95 |
|---|---|---|---:|
| Baseline | FP32 | Off | 36,1650% |
| Baseline | FP32 | On | 36,1877% |
| Baseline | FP16 | On | 36,1860% |
| Baseline | Autocast, bobot FP32/input FP16 | Off | 36,1824% |
| Baseline | Autocast, bobot FP32/input FP16 | On | 36,1860% |
| BRA | FP32 | On | 35,5232% |
| BRA | Autocast, bobot FP32/input FP16 | Off | 35,4817% |

Artefak tiap kasus ada di `runs/evaluation/consistency_20261014_c_*/diagnostic_summary.json`. Mode autocast tanpa fusion mendekati jalur forward validasi training, tetapi bukan pemulihan persis EMA FP32 historis. Validator tetap memakai checkpoint yang tersimpan.

Rentang semua kasus baseline hanya sekitar **0,023 poin**, jauh lebih kecil daripada anomali awal sekitar 1,11 poin terhadap evaluasi ulang FP32. Dengan demikian, perubahan precision/fusion yang diuji **bukan penjelasan memadai** untuk skor awal 35,08%.

## Penjelasan perubahan skor baseline: fakta dan batas diagnosis

1. **Bukan perubahan bobot atau dataset saat pemeriksaan ini.** Hash checkpoint dan hash isi dataset cocok dengan manifest training. Baseline yang dimaksud tetap model 28 kelas, bukan baseline historis 30 kelas.
2. **Skor standalone awal 35,08% merupakan hasil yang tidak dapat direproduksi dalam pemeriksaan ini.** Perintah lama dengan default yang sama sekarang menghasilkan 36,19%, tepat sama dengan evaluasi yang parameternya dipin. Karena itu, kesimpulan awal bahwa BRA unggul +0,44 poin harus dikoreksi.
3. **Precision/fusion hanya menjelaskan variasi kecil pada pengujian sekarang.** Menonaktifkan fusion atau memakai FP16/autocast tidak mengembalikan baseline ke 35,08%.
4. **Timeout NMS merupakan hipotesis, bukan penyebab yang sudah terbukti.** Kode lokal NMS mempunyai batas waktu `2 + 0.05 * batch_size` detik, yaitu 2,8 detik untuk batch 16; ketika batas itu terlewati, loop dapat berhenti sebelum semua gambar dalam batch diproses. Evaluasi awal baseline mencatat postprocessing 13,73 ms/gambar, jauh lebih tinggi daripada BRA 1,01 ms/gambar. Ini membuat gangguan runtime/postprocessing layak dicurigai, tetapi ringkasan lama tidak menyimpan log warning atau prediksi per gambar.
5. **Simulasi timeout sederhana tidak mereproduksi anomali persis.** Dengan sengaja mengosongkan prediksi 15 gambar terakhir dalam batch pertama, baseline menghasilkan 35,33%, bukan 35,08%. Simulasi ini hanya menunjukkan bagaimana prediksi terlewat dapat menurunkan skor; bukan bukti bahwa itulah yang terjadi pada evaluasi historis. Artefaknya jelas ditandai `simulated_first_batch_stop_after=1` dan tidak dipakai sebagai hasil model.
6. **Selisih kecil terhadap skor saat training masih ada.** Baseline training 36,2360% vs standalone stabil 36,1877% berbeda sekitar 0,048 poin. BRA training 35,5380% vs standalone 35,5232% berbeda sekitar 0,015 poin. Kode training menyimpan EMA checkpoint dalam FP16, sedangkan validasi per epoch memakai EMA FP32/autocast; precision, fusion, dan pembulatan bobot dapat menyumbang variasi kecil. Kontribusi masing-masing tidak dapat dipisahkan persis tanpa snapshot EMA FP32 epoch 38.

Kesimpulan diagnosis yang dapat dipertanggungjawabkan: **baseline tidak terbukti mengalami penurunan performa model sebesar 1,16 poin; skor evaluasi awalnya anomali yang tidak stabil. Penyebab historis spesifik belum dapat dipastikan dari artefak yang tersedia.** Tidak ada dasar untuk menyebut fusion, precision, kerusakan checkpoint, atau timeout sebagai penyebab pasti.

## Kesimpulan yang diperbarui

Pada evaluasi utama yang konsisten, baseline unggul **0,66 poin mAP50-95**, **0,83 poin mAP50**, dan **4,12 poin recall**. BRA hanya meningkatkan precision agregat sekitar 1,72 poin. Dengan biaya komputasi lebih tinggi dan AP menurun di 17 kelas, **BRA-P4 konfigurasi sekarang belum menunjukkan keuntungan keseluruhan pada seed ini**.

Prioritas selanjutnya bukan menambah epoch atau memakai hasil awal sebagai klaim peningkatan, melainkan memakai protokol evaluasi stabil ini untuk ablation/tuning yang terkontrol dan kemudian menguji beberapa seed. Penurunan kecil satu seed belum membuktikan bahwa semua desain Dynamic Attention tidak berguna.

## Mengulang evaluasi utama

Gunakan tag baru setiap kali agar hasil lama tidak ditimpa:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_legacy_evaluation.py --model baseline --precision fp32 --fusion on --tag my_repeat --disable-nms-time-limit
.\.venv\Scripts\python.exe scripts\diagnose_legacy_evaluation.py --model bra_p4 --precision fp32 --fusion on --tag my_repeat --disable-nms-time-limit
```

Untuk kontrol mendekati validasi training, gunakan `--precision amp --fusion off`, tetap dengan tag baru. Jangan memakai opsi `--simulate-first-batch-stop-after` untuk pelaporan performa model; opsi tersebut khusus eksperimen diagnostik.

## Batas interpretasi

Eksperimen hanya satu seed, sehingga belum membuktikan signifikansi statistik. Dataset lama memiliki kandidat kemiripan gambar antar split dari audit sebelumnya. Dataset tidak dibersihkan atau diubah pada evaluasi ini.
