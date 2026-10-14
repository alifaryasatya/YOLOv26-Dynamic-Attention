# Output grafik BRA-P4 — dataset lama, 50 epoch

Folder ini mengumpulkan delapan output asli dari `runs/phase4_legacy_50e/bra_p4_50e_seed0/`, dengan nama dan jenis grafik yang sama seperti template di `contoh hasil/`. Gambar disalin tanpa mengganti angka, mengubah kurva, atau merender ulang. Folder template tetap utuh.

Model: YOLO26n + Bi-Level Routing Attention pada P4/layer 19. Dataset: `plantdoc_data.yaml`, **28 kelas**. Training selesai 50 epoch. Template historis memiliki 30 kelas, sehingga jumlah label pada gambar BRA berbeda dan tidak dipaksakan mengikuti label template.

## Gambar

| File | Isi dan sumber |
|---|---|
| [results.png](results.png) | Loss dan metrik per epoch dari training BRA 50 epoch. |
| [BoxF1_curve.png](BoxF1_curve.png) | F1 terhadap confidence pada validasi checkpoint terbaik di akhir proses training. |
| [BoxPR_curve.png](BoxPR_curve.png) | Precision–Recall; legenda menunjukkan mAP50 BRA sekitar 0,516. |
| [BoxP_curve.png](BoxP_curve.png) | Precision terhadap confidence. |
| [BoxR_curve.png](BoxR_curve.png) | Recall terhadap confidence. |
| [confusion_matrix.png](confusion_matrix.png) | Confusion matrix dalam jumlah objek, termasuk background. |
| [confusion_matrix_normalized.png](confusion_matrix_normalized.png) | Confusion matrix ternormalisasi. |
| [labels.jpg](labels.jpg) | Distribusi anotasi kelas dan geometri bounding box pada data training; bukan prediksi model. |

## Catatan interpretasi

`results.png` memuat seluruh riwayat training, bukan hanya epoch terbaik. Kurva evaluasi dan confusion matrix berasal dari artefak validasi akhir training; jangan menyebutnya sebagai grafik baru dari evaluasi FP32 terkontrol. Tidak ada training atau evaluasi tambahan untuk menyalin paket gambar ini.

Analisis dan angka evaluasi ulang yang konsisten tersedia di [laporan evaluasi](../docs/LEGACY_50E_EVALUATION_ANALYSIS.md). Checkpoint asli tetap berada di [best.pt](../runs/phase4_legacy_50e/bra_p4_50e_seed0/weights/best.pt). Data angka riwayat tersedia di [results.csv](../runs/phase4_legacy_50e/bra_p4_50e_seed0/results.csv).
