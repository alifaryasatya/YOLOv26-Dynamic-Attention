# PlantDoc YOLO26 — baseline dan Dynamic Attention

Dataset aktif hanya `dataset/` melalui `plantdoc_data.yaml` (28 kelas). Train 1.795 gambar, validasi 386, test 386. Model merupakan prototipe riset, bukan alat diagnosis agronomi mandiri.

## Isi repository GitHub

Dataset, environment Python, checkpoint, folder `runs/`, dan `artifacts/` tidak diunggah; semuanya tetap lokal dan dikecualikan oleh `.gitignore`. Repository memuat source code, konfigurasi YAML, requirements, dokumentasi, dan grafik ringkasan yang dipilih di folder hasil. Setelah clone, siapkan dataset sesuai `plantdoc_data.yaml`, environment dari `requirements.txt`, dan checkpoint lokal yang diperlukan. Tes integrasi tertentu memerlukan checkpoint baseline 28 kelas. Tautan dokumentasi ke artefak lokal tidak tersedia di GitHub jika file tersebut diabaikan.

## Hasil yang dipertahankan

- Baseline 28 kelas: `runs/phase4_legacy_50e/baseline_50e_seed0/`.
- BRA-P4 28 kelas: `runs/phase4_legacy_50e/bra_p4_50e_seed0/`.
- Baseline historis 30 kelas: `runs/detect/yolov26_baseline_50e/`.
- [Grafik BRA](<hasil bra/README.md>).
- [Analisis evaluasi konsisten](docs/LEGACY_50E_EVALUATION_ANALYSIS.md).
- [Perbandingan historis dan F1 ketiga model](docs/BRA_VS_HISTORICAL_BASELINE_30_CLASSES.md).

Pada evaluasi FP32 yang konsisten, mAP50-95 baseline 36,19% dan BRA 35,52%. Hasil satu seed belum membuktikan signifikansi statistik. Baseline historis mempunyai mapping 30 kelas yang berbeda.

## Kode utama

- [bra.py](src/plantdoc_attention/modules/bra.py): Bi-Level Routing Attention.
- [residual_attention.py](src/plantdoc_attention/modules/residual_attention.py): normalisasi dan residual gate.
- [yolo26.py](src/plantdoc_attention/integrations/yolo26.py): wrapper P4 dan trainer.
- `scripts/train.py`, `evaluate.py`, `benchmark_model.py`: training, evaluasi dan profiling.
- `scripts/diagnose_legacy_evaluation.py`: kontrol precision/fusion/NMS.

Kode CBAM tetap tersedia sebagai kontrol attention opsional. Seluruh skrip/config aktif menggunakan dataset lama.

## Training 50 epoch

Training tetap dijalankan pengguna, satu per satu:

```powershell
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\experiments\legacy_baseline_50e.yaml
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\experiments\legacy_bra_p4_50e.yaml
```

Kedua run sudah ada: jangan menimpa checkpoint. Untuk training ulang gunakan nama baru. Recipe matched: COCO `yolo26n.pt`, seed 0, batch 16, input 640, AdamW, AMP, 50 epoch, patience 0, close_mosaic 10. Tambahkan `--dry-run` untuk validasi tanpa training.

## Evaluasi konsisten

Gunakan tag baru:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_legacy_evaluation.py --model baseline --precision fp32 --fusion on --tag my_repeat --disable-nms-time-limit
.\.venv\Scripts\python.exe scripts\diagnose_legacy_evaluation.py --model bra_p4 --precision fp32 --fusion on --tag my_repeat --disable-nms-time-limit
```

Pipeline evaluasi/profiling umum default ke pasangan dataset lama 50 epoch:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_phase4.py --tag legacy_new_report
```

Pipeline umum tidak sama dengan runner diagnostik yang mempin precision/fusion dan menonaktifkan deadline NMS. Sumber metrik harus dijelaskan.

## Audit, tes, inferensi

```powershell
.\.venv\Scripts\python.exe scripts\audit_dataset.py --data plantdoc_data.yaml
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe predict_sample.py --source dataset/valid/images --name legacy_samples_new
```

Default checkpoint inferensi adalah baseline 28 kelas; BRA dipilih dengan `--weights runs/phase4_legacy_50e/bra_p4_50e_seed0/weights/best.pt`.

Audit lama di `artifacts/audit/` menemukan kandidat kemiripan gambar antar split. Dataset dan split pengguna tidak diubah. Test hanya digunakan setelah desain dibekukan.

## PRD ringkas

Tujuan: deteksi penyakit daun yang reproducible dan pengukuran manfaat/biaya attention pada data serta protokol yang sama. Kriteria selesai: konfigurasi tercatat, tes forward/backward/AMP/checkpoint lolos, metrik per kelas tersedia, serta kesimpulan melaporkan hasil aktual termasuk bila attention tidak lebih baik.

Panduan: [Fase 1](docs/PHASE_1_BRA.md), [Fase 2](docs/PHASE_2_YOLO_P4.md), [Fase 3](docs/PHASE_3_TRAINING_GUIDE.md), [Fase 4](docs/PHASE_4_ABLATION_GUIDE.md). Dependency dikunci di `requirements.txt`; gunakan `check_setup.py` untuk memeriksa environment.
