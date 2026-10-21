# Simulasi Tanya Jawab Dosen Penguji

> Bahan latihan presentasi proyek YOLO26n + Bi-Level Routing Attention (BRA). Jawaban dirumuskan berdasarkan eksperimen dataset lama 28 kelas. Sampaikan dengan tenang, dan jangan mengklaim hasil yang belum dibuktikan.

## Pembukaan singkat

> “Proyek ini membandingkan YOLO26n baseline dengan YOLO26n yang ditambahkan Bi-Level Routing Attention pada fitur P4. Tujuannya adalah menguji apakah pemilihan wilayah fitur secara dinamis membantu deteksi penyakit daun pada dataset PlantDoc. Saya membandingkan kedua model pada dataset 28 kelas dengan protokol training yang sama. Pada evaluasi validasi yang konsisten, BRA belum meningkatkan mAP dibanding baseline, sehingga hasil ini menjadi temuan dan batasan eksperimen, bukan klaim bahwa attention pasti memperbaiki model.”

## Pertanyaan dasar tentang proyek

### 1. Apa masalah yang ingin diselesaikan proyek ini?

Dataset PlantDoc berisi gambar daun dalam kondisi lapangan dengan latar yang beragam, seperti bayangan, tanah, dan daun yang bertumpuk. Saya menguji apakah attention dinamis dapat membantu model memilih informasi fitur yang relevan untuk deteksi penyakit.

### 2. Apa tujuan penambahan BRA?

Tujuannya adalah menguji pengaruh Bi-Level Routing Attention pada fitur P4 YOLO26. BRA membagi fitur menjadi wilayah, menghitung relevansi antarwilayah, lalu memilih sejumlah wilayah untuk diperhatikan. Harapannya, fitur lesi yang relevan mendapat konteks yang lebih baik. Namun, hasil eksperimen saya menunjukkan konfigurasi BRA yang diuji belum mengungguli baseline.

### 3. Dataset apa yang digunakan?

Eksperimen utama memakai dataset lama yang dikonfigurasi di `plantdoc_data.yaml`, dengan 28 kelas: 1.795 gambar training, 386 validasi, dan 386 test. Perbandingan baseline dengan BRA yang saya laporkan menggunakan split validasi yang sama.

### 4. Mengapa tidak membandingkan langsung dengan baseline 30 kelas?

Karena baseline 30 kelas dan eksperimen sekarang 28 kelas memiliki ruang label yang berbeda. Mapping kelas berubah mulai indeks 15, jadi skor keduanya tidak menggambarkan tugas yang sama. Perbandingan yang lebih adil adalah baseline dan BRA 28 kelas yang memakai dataset serta protokol yang sama.

## Pertanyaan tentang arsitektur dan implementasi

### 5. Bagaimana alur arsitektur baseline YOLO26?

Secara umum, backbone mengekstrak fitur dari gambar, neck menggabungkan fitur pada beberapa skala, lalu detection head memprediksi kotak dan kelas. Pada graph YOLO26 yang dipakai integrasi ini, keluaran untuk head berasal dari layer 16, 19, dan 22, yang mewakili fitur P3, P4, dan P5. Arsitektur dasarnya dimuat dari `yolo26n.pt`; `scripts/train.py` mengatur training, bukan menulis ulang semua layer YOLO26.

**Tunjukkan di kode/dokumen:**

- [Diagram arsitektur baseline](ARSITEKTUR_BASELINE_DAN_BRA.md#L15) dan uraian komponen backbone, neck, serta head di dokumen yang sama, bagian 3.
- [Config baseline](../configs/experiments/baseline.yaml#L9) memilih `yolo26n.pt` dan [dataset](../configs/experiments/baseline.yaml#L10). [Loader model](../scripts/train.py#L45) menyelesaikan checkpoint dan [baris 104](../scripts/train.py#L104) memuatnya melalui Ultralytics.
- Integrasi memverifikasi graph 24 layer dan input head `[16, 19, 22]` di [yolo26.py](../src/plantdoc_attention/integrations/yolo26.py#L68).

**Catatan saat presentasi:** definisi lengkap backbone/neck/head YOLO26 berasal dari arsitektur Ultralytics yang dimuat oleh `yolo26n.pt`, bukan file model YAML buatan repo ini. Di repo, bukti nomor layer yang dipakai adalah diagram arsitektur dan validasi graph integrasi.

### 6. BRA dipasang di bagian mana? Mengapa di P4?

BRA dipasang setelah blok C3k2 P4 pada layer 19 di neck, sebelum fiturnya diteruskan ke detection head. P4 dipilih sebagai titik awal karena merupakan skala menengah dan terhubung langsung ke salah satu input head. Pemilihan ini adalah keputusan desain eksperimen, bukan bukti bahwa P4 merupakan lokasi terbaik.

**Tunjukkan di kode:**

- [Config BRA](../configs/experiments/legacy_bra_p4_50e.yaml#L5) mengaktifkan jenis BRA, dan [baris 7](../configs/experiments/legacy_bra_p4_50e.yaml#L7) menetapkan `layer: 19`.
- [Pemeriksaan dan pemasangan wrapper](../src/plantdoc_attention/integrations/yolo26.py#L63): baris 66 memastikan target layer 19; baris 76–83 memvalidasi bahwa itu C3k2 P4 dengan 128 channel; baris 79 memeriksa koneksi head `[16, 19, 22]`; baris 84–90 memasang wrapper.
- [Forward wrapper](../src/plantdoc_attention/integrations/yolo26.py#L52) menjalankan C3k2 asli kemudian residual attention.

**Kalimat saat menunjuk kode:** “Di sini layer 19 ditetapkan oleh config. Fungsi `attach_attention_p4` memastikan struktur dan channel P4 sesuai sebelum mengganti layer itu dengan wrapper.”

### 7. Apa yang berubah dari baseline ketika BRA ditambahkan?

Backbone dan head tetap memakai YOLO26. Pada layer P4, blok C3k2 asli tetap dijalankan, kemudian keluarannya masuk ke cabang residual BRA. Dengan demikian, fitur yang diteruskan sebagai P4 sudah mendapat pembaruan dari attention.

**Tunjukkan di kode:** pada [scripts/train.py](../scripts/train.py#L104), model dasar dimuat. Baris 105–106 memilih trainer khusus hanya jika config memiliki `attention`; [baris 113](../scripts/train.py#L113) memulai training. Di [wrapper](../src/plantdoc_attention/integrations/yolo26.py#L52), baris 53 menunjukkan urutan `self.original(x)` lalu `self.residual(...)`. Untuk baseline, config tidak memiliki blok `attention`, sehingga trainer attention tidak dipilih.

### 8. Bisa jelaskan rumus residual BRA?

Rumusnya:

```text
y = x + gamma × BRA(LayerNorm(x))
```

`x` adalah fitur P4 asli. `gamma` dimulai dari nol, sehingga pada awal training cabang attention belum mengubah keluaran. Nilai `gamma` kemudian dapat dipelajari untuk mengatur kekuatan kontribusi BRA.

**Tunjukkan di kode:** [ResidualAttention](../src/plantdoc_attention/modules/residual_attention.py#L31) mendefinisikan modul; baris 43–45 membuat BRA, normalisasi, dan parameter `gamma`; baris 47–49 berisi perhitungan forward serta penjumlahan residual. Nilai awal nol berasal dari [config BRA, baris 15](../configs/experiments/legacy_bra_p4_50e.yaml#L15).

### 9. Apa arti “dinamis” pada BRA?

Pemilihan wilayah bergantung pada isi fitur gambar, sehingga wilayah yang dipilih dapat berbeda untuk input yang berbeda. Dalam implementasi ini, fitur diproyeksikan menjadi Q, K, dan V, kemudian wilayah relevan dipilih menggunakan top-k. Keputusan top-k bersifat diskret dan indeks routing dihitung dari representasi yang di-detach; operasi attention pada token terpilih tetap dipelajari melalui proses training.

**Tunjukkan di kode:** kelas dimulai di [bra.py](../src/plantdoc_attention/modules/bra.py#L38). Baris 94 membentuk Q/K/V; baris 105–108 menghitung skor routing lalu memilih indeks top-k; baris 115–121 menghitung attention token terpilih; baris 122–125 menyusun kembali fitur, menambahkan konteks lokal, dan memproyeksikan output.

### 10. Parameter BRA apa yang digunakan?

Untuk eksperimen BRA-P4, dimensi fitur adalah 128, dimensi Q/K 32, jumlah head 4, grid wilayah maksimal 4×4, dan `topk=4`. Artinya, setiap wilayah memilih hingga empat wilayah kunci dari grid yang tersedia. Implementasi juga memakai depthwise convolution berukuran 3 untuk konteks lokal.

**Tunjukkan di kode:** seluruh nilai run tercantum di [legacy_bra_p4_50e.yaml](../configs/experiments/legacy_bra_p4_50e.yaml#L5) baris 5–15. Default/tipe parameter juga didefinisikan dalam [BRAConfig](../src/plantdoc_attention/integrations/yolo26.py#L16). Validasi bahwa `dim` dan `qk_dim` dapat dibagi rata ke jumlah head berada di [bra.py](../src/plantdoc_attention/modules/bra.py#L64); parameter konvolusi QKV dan konteks lokal dibuat di baris 75–79.

### 11. Di mana kode BRA diimplementasikan?

Operasi attention inti berada di `src/plantdoc_attention/modules/bra.py`, khususnya kelas `BiLevelRoutingAttention`. Rumus residual dan normalisasi berada di `src/plantdoc_attention/modules/residual_attention.py`. Pemasangan ke graph YOLO26 dilakukan di `src/plantdoc_attention/integrations/yolo26.py`, sedangkan config eksperimennya berada di `configs/experiments/legacy_bra_p4_50e.yaml`.

**Urutan file yang bisa Anda buka saat ditanya:**

1. [Config BRA](../configs/experiments/legacy_bra_p4_50e.yaml#L5) — menunjukkan bahwa BRA aktif pada layer 19 dan parameter eksperimen.
2. [Pemilihan trainer](../scripts/train.py#L104) — menunjukkan config attention mengaktifkan trainer khusus.
3. [Integrasi ke graph](../src/plantdoc_attention/integrations/yolo26.py#L106) — `get_model()` membangun graph dasar lalu memanggil `attach_attention_p4()` pada baris 126.
4. [Wrapper P4](../src/plantdoc_attention/integrations/yolo26.py#L38) — menyimpan blok asli dan menambahkan residual attention di forward.
5. [Residual gate](../src/plantdoc_attention/modules/residual_attention.py#L31) — menunjukkan rumus gabungan fitur.
6. [BRA inti](../src/plantdoc_attention/modules/bra.py#L38) — menunjukkan operasi routing dan attention.

### Urutan demo singkat di editor

Saat penguji meminta bukti implementasi, buka file dalam urutan berikut: `legacy_bra_p4_50e.yaml` → `scripts/train.py` → `integrations/yolo26.py` → `residual_attention.py` → `bra.py`. Dengan urutan ini, Anda memperlihatkan alur dari pemilihan konfigurasi, aktivasi trainer, pemasangan ke layer 19, rumus residual, sampai operasi attention inti.

## Pertanyaan tentang eksperimen

### 12. Bagaimana Anda menjaga perbandingan baseline dan BRA tetap adil?

Keduanya menggunakan bobot awal `yolo26n.pt`, dataset 28 kelas yang sama, seed 0, ukuran input 640, batch 16, AdamW, AMP, dan 50 epoch. Config BRA mewarisi config baseline, lalu menambahkan parameter attention. Dengan begitu, perubahan utama pada eksperimen adalah adanya BRA di P4.

### 13. Mengapa memakai bobot pretrained?

Pretraining memberi model fitur awal yang sudah dipelajari dari dataset besar, sehingga training dapat difokuskan pada penyesuaian terhadap PlantDoc. Baseline dan BRA sama-sama dimulai dari bobot YOLO26n yang sama agar pembandingannya konsisten.

### 14. Bagaimana mengevaluasi model?

Hasil utama berasal dari `best.pt` masing-masing model pada split validasi yang sama, yaitu 386 gambar dan 1.343 objek. Evaluasi ulang utama menggunakan FP32, fusion aktif, ukuran input 640, dan parameter evaluasi yang dikontrol. Saya melaporkan sumber metriknya sebagai evaluasi validasi, bukan hasil test.

### 15. Mengapa hasil evaluasi awal berbeda dari hasil evaluasi ulang?

Evaluasi awal baseline menghasilkan mAP50-95 35,08%, tetapi angka itu tidak berhasil direproduksi. Evaluasi ulang yang terkontrol dan pengulangan dengan skrip lama menghasilkan sekitar 36,19%. Hash dataset dan checkpoint cocok dengan manifest training. Saya belum dapat memastikan penyebab historis perbedaan tersebut, jadi saya memakai hasil evaluasi ulang konsisten dan tidak menyatakan timeout NMS atau precision sebagai penyebab pasti.

## Pertanyaan kritis tentang hasil

### 16. Apa hasil utama perbandingan baseline dan BRA?

Pada evaluasi validasi konsisten, baseline memperoleh precision 53,38%, recall 49,70%, mAP50 52,43%, dan mAP50-95 36,19%. BRA memperoleh precision 55,10%, recall 45,57%, mAP50 51,60%, dan mAP50-95 35,52%. Jadi BRA menaikkan precision sekitar 1,72 poin persentase, tetapi mAP50-95 turun sekitar 0,66 poin dan recall turun sekitar 4,12 poin.

### 17. Kalau precision BRA naik, mengapa Anda menyimpulkan hasilnya belum lebih baik?

Karena precision saja tidak menggambarkan seluruh kualitas deteksi. Recall dan kedua metrik mAP turun. Artinya, pada konfigurasi ini peningkatan ketepatan prediksi yang dibuat belum mengimbangi penurunan kemampuan menemukan objek dan kualitas deteksi agregat.

### 18. Apakah BRA meningkatkan hasil pada semua kelas?

Tidak. Dibanding baseline, AP50-95 meningkat pada 10 kelas, menurun pada 17 kelas, dan tetap pada satu kelas. Jadi manfaatnya tidak merata dan hasil agregatnya masih di bawah baseline.

### 19. Apakah hasil ini membuktikan BRA tidak berguna?

Tidak. Hasil ini hanya menunjukkan bahwa konfigurasi BRA-P4 yang saya uji belum memberi peningkatan pada satu seed dan protokol ini. Hasil tersebut tidak membuktikan semua varian BRA atau semua lokasi attention akan gagal. Pengujian beberapa seed dan ablation lokasi diperlukan untuk kesimpulan yang lebih kuat.

### 20. Apakah hasilnya signifikan secara statistik?

Saya belum bisa menyatakan signifikansi statistik karena eksperimen utama baru memakai satu seed. Selisih yang terlihat adalah hasil pada run tersebut. Untuk menilai stabilitas, eksperimen perlu diulang dengan beberapa seed dan dilaporkan bersama variasi hasilnya.

### 21. Apa biaya komputasi tambahan BRA?

Pada pasangan 28 kelas, BRA menambah sekitar 42.817 parameter atau 1,70%, GFLOPs meningkat sekitar 5,74%, dan peak VRAM training meningkat sekitar 10,40%. Waktu forward batch 1 FP16 yang diukur naik dari sekitar 13,76 menjadi 15,51 milidetik. Pengukuran forward ini tidak mencakup seluruh proses seperti I/O dan postprocessing, jadi saya tidak menyebutnya sebagai waktu end-to-end deployment.

## Pertanyaan tentang keterbatasan dan langkah lanjut

### 22. Apa keterbatasan utama penelitian ini?

Pertama, hasil utama baru menggunakan satu seed. Kedua, evaluasi yang saya pakai adalah validasi, belum hasil test final. Ketiga, audit sebelumnya menemukan kandidat gambar mirip antar split, sehingga kemungkinan kebocoran data perlu diperiksa lebih lanjut. Keempat, konfigurasi BRA yang diuji hanya P4 dengan satu set parameter.

### 23. Mengapa belum memakai hasil test sebagai hasil utama?

Split test sebaiknya dipakai setelah desain dan pemilihan model dibekukan. Menggunakan test berulang kali untuk memilih konfigurasi dapat membuat keputusan desain ikut menyesuaikan diri dengan test. Karena itu, hasil yang saya presentasikan sekarang adalah hasil validasi.

### 24. Apa eksperimen selanjutnya yang Anda usulkan?

Saya akan mempertahankan baseline sebagai pembanding, menguji beberapa seed, dan mengevaluasi konfigurasi BRA secara terkontrol. Saya juga akan meninjau kandidat kemiripan gambar antar split sebelum membuat klaim generalisasi. Jika BRA tetap tidak meningkatkan hasil dengan biaya tambahan yang terukur, kesimpulan yang tepat adalah konfigurasi tersebut belum layak menggantikan baseline.

## Pertanyaan jebakan

### “Bukankah kenaikan precision berarti BRA berhasil?”

“Belum tentu. Precision meningkat, tetapi recall dan mAP menurun. Jadi kesimpulan harus mempertimbangkan beberapa metrik dan tujuan deteksi, bukan satu angka saja.”

### “Kenapa tidak menyebut hasil BRA lebih baik berdasarkan evaluasi awal?”

“Karena skor awal baseline tidak berhasil direproduksi. Saya memperbarui kesimpulan berdasarkan evaluasi ulang yang terkontrol; pada evaluasi itu mAP50-95 baseline lebih tinggi 0,66 poin.”

### “Apakah model Anda sudah benar-benar NMS-free saat evaluasi?”

“Evaluasi utama dikontrol agar mengikuti jalur deteksi yang digunakan pada validasi training, yaitu jalur one-to-many dengan external NMS. Jadi saya tidak mengklaim angka evaluasi tersebut sebagai benchmark mode native NMS-free.”

### “Kalau hasilnya lebih buruk, apa kontribusi proyek Anda?”

“Kontribusinya adalah implementasi BRA yang terintegrasi pada YOLO26 dan evaluasi perbandingan yang terkontrol. Hasil negatif tetap informatif: pada dataset, lokasi P4, dan konfigurasi ini, BRA menambah biaya tetapi belum meningkatkan mAP. Itu membantu mempersempit desain yang layak diteliti berikutnya.”

## Angka yang perlu diingat

- Dataset eksperimen utama: **28 kelas**.
- Training: **50 epoch**, seed **0**, input **640**, batch **16**.
- Pemasangan BRA: **P4, layer 19**.
- mAP50-95 validasi konsisten: **baseline 36,19%**, **BRA 35,52%**.
- Selisih BRA terhadap baseline: **−0,66 poin mAP50-95**.
- Kesimpulan: **BRA konfigurasi sekarang belum mengungguli baseline; satu seed belum cukup untuk klaim statistik**.

## Referensi kode dan laporan

- [Analisis evaluasi 50 epoch](LEGACY_50E_EVALUATION_ANALYSIS.md)
- [Analisis baseline historis 30 kelas vs BRA 28 kelas](BRA_VS_HISTORICAL_BASELINE_30_CLASSES.md)
- [Config BRA 50 epoch](../configs/experiments/legacy_bra_p4_50e.yaml)
- [Implementasi integrasi YOLO26](../src/plantdoc_attention/integrations/yolo26.py)
- [Modul BRA](../src/plantdoc_attention/modules/bra.py)
- [Residual attention](../src/plantdoc_attention/modules/residual_attention.py)
