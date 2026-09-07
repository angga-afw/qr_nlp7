# qr_nlp7

## Ringkasan Proyek

Aplikasi ini merupakan sistem asistensi medis berbasis Streamlit yang menggabungkan:

- QR-based patient access
- rekam medis elektronik pasien
- AI chat untuk SOAP note
- retrieval-augmented generation (RAG) untuk konteks klinis
- NEWS2 triage untuk penilaian cepat

Aplikasi ini dibuat untuk membantu proses pemeriksaan awal pasien dengan memadukan data profil pasien, riwayat rekam medis sebelumnya, dan panduan klinis yang relevan sebelum model AI menghasilkan output medis.

## Cara Kerja RAG di Aplikasi Ini

RAG (Retrieval-Augmented Generation) bekerja dengan cara mencari konteks yang paling relevan dari basis pengetahuan sebelum model AI menjawab pertanyaan atau menghasilkan laporan medis.

### 1. Input pengguna masuk ke sistem
Ketika pasien mengirimkan keluhan seperti:

- "Saya pusing sejak kemarin"
- "Saya sesak nafas, saturasi turun"
- "Saya sudah pernah punya riwayat darah tinggi"

keluhan tersebut masuk ke fungsi `process_narrative()` di `app.py`.

### 2. Sistem mengambil konteks relevan
Dalam fungsi tersebut, aplikasi menjalankan dua retrieval terhadap sumber pengetahuan yang berbeda:

#### a. Riwayat pasien dari rekam medis lama
Fungsi `retrieve_patient_history()` mengambil data dari `medical_encounters.csv`. Data pertemuan lama diproses menjadi teks gabungan seperti:

- tanggal
- subjective
- objective
- assessment
- plan

Setelah itu, teks tersebut diubah menjadi embedding vektor menggunakan model embedding Google. Query dari pasien juga dibuat embedding. Lalu dihitung similarity score dengan cosine similarity untuk memilih riwayat yang paling relevan.

#### b. Panduan klinis dari knowledge base
Fungsi `retrieve_guidelines()` membaca file `clinical_guidelines.txt`, memecah isi dokumen menjadi chunk, lalu membuat embedding untuk masing-masing chunk. Query dari pasien juga di-encode menjadi embedding, lalu dicari chunk yang paling mirip.

### 3. Keterkaitan dengan profil pasien
Selain retrieval, aplikasi juga menyisipkan data profil aktif pasien dari `medical_registry_v3.csv`, misalnya:

- nama
- tanggal lahir
- jenis kelamin
- alergi
- riwayat penyakit kronis
- tekanan darah terakhir
- saturasi oksigen terakhir

Semua informasi ini dimasukkan ke dalam prompt yang akan dikirim ke model AI.

### 4. Model AI menerima konteks + pertanyaan
Setelah konteks relevan terkumpul, prompt dibuat seperti:

- profil pasien
- riwayat medis relevan
- panduan klinis relevan
- narasi keluhan pasien

Lalu model AI diberi perintah untuk menghasilkan output JSON yang berisi:

- SOAP note
- update profil medis
- tanda vital untuk triage NEWS2

### 5. Output digunakan di UI
Setelah AI menghasilkan JSON, aplikasi menyimpan hasil ke session state dan menampilkannya di antarmuka:

- preview SOAP di chat
- referensi RAG terkait
- data profil pasien yang berpotensi diperbarui
- NEWS2 triage default berdasarkan pasien

## Fungsi Utama RAG dalam Kode

Bagian utama yang mengimplementasikan RAG ada di `app.py`:

- `get_embedding()`
- `cosine_similarity_val()`
- `load_guideline_chunks()`
- `get_cached_guidelines()`
- `retrieve_guidelines()`
- `retrieve_patient_history()`
- `process_narrative()`

### Contoh alur kerja program

1. User mengirim narasi ke chat
2. `process_narrative()` dipanggil
3. `retrieve_patient_history()` mencari rekam medis pasien yang relevan
4. `retrieve_guidelines()` mencari panduan klinis yang relevan
5. hasil retrieval dimasukkan ke prompt
6. Google Generative AI menghasilkan JSON SOAP + triage
7. hasil ditampilkan ke UI dan bisa disimpan ke CSV

## Keuntungan RAG dalam sistem ini

- lebih relevan dibanding model yang hanya menjawab dari pengetahuan umum
- memanfaatkan riwayat pasien yang sudah ada
- menghubungkan keluhan pasien dengan panduan klinis yang berlaku
- membantu AI mengekstraksi parameter seperti SpO2, BPS, RR, HR, AVPU untuk NEWS2
- mengurangi kemungkinan jawaban yang terlalu umum atau tidak sesuai konteks medis pasien

## Struktur Data yang Digunakan

- `medical_registry_v3.csv`: data profil pasien
- `medical_encounters.csv`: riwayat pertemuan medis sebelumnya
- `clinical_guidelines.txt`: knowledge base klinis
- `model_comparison_results.csv`: log hasil evaluasi model

## Persentase Penggunaan Informasi: Umum vs RAG

Pada menu admin terdapat metrik yang menampilkan persentase penggunaan informasi dari dua sumber utama:

- `Informasi Umum (General)`
- `Informasi dari RAG`

### Arti dari kedua metrik

#### 1. Informasi Umum (General)
Metrik ini menunjukkan persentase kasus di mana model menjawab menggunakan pengetahuan umum model itu sendiri, tanpa menemukan konteks yang cukup relevan dari knowledge base atau riwayat pasien.

Artinya, pada kasus tersebut:

- tidak ada hasil retrieval dari `medical_encounters.csv` yang relevan
- tidak ada chunk dari `clinical_guidelines.txt` yang dianggap relevan
- model lebih banyak bergantung pada pengetahuan bawaan atau pola bahasa umum

#### 2. Informasi dari RAG
Metrik ini menunjukkan persentase kasus di mana model mendapat konteks yang relevan dari:

- riwayat pasien lama dari `medical_encounters.csv`, atau
- panduan klinis dari `clinical_guidelines.txt`

Jika hasil retrieval menemukan kesesuaian yang cukup, maka log akan menandai sumber sebagai `RAG`.

### Kenapa bisa 100% Informasi Umum?

Nilai 100% Informasi Umum bukan berarti sistem gagal, tetapi berarti pada semua percobaan yang tercatat, retrieval tidak menghasilkan hasil yang cukup relevan. Dalam implementasi saat ini, retrieval hanya dianggap aktif jika similarity score melebihi nilai tertentu (`> 0.35`), seperti yang diterapkan di fungsi `retrieve_guidelines()` dan `retrieve_patient_history()` di `app.py`.

Jadi, jika query terlalu umum atau tidak cocok dengan data riwayat/panduan yang tersedia, maka:

- `Knowledge_Source = "General"`
- `RAG_Used = 0`
- persentasenya akan muncul sebagai 100% General

Contoh query yang kemungkinan menghasilkan General:

- "saya pusing"
- "saya merasa tidak enak"
- "saya sakit"

Ketika query seperti itu terlalu umum, mesin pencarian semantik mungkin tidak menemukan konteks yang cukup kuat untuk dinyatakan sebagai RAG.

### Contoh penggunaan yang akan menaikkan persentase RAG

RAG akan lebih sering aktif jika query mengandung informasi yang spesifik, misalnya:

- "saya punya riwayat hipertensi, pusing, dan tekanan darah 140/89"
- "saya pernah rawat inap karena demam tifoid, sekarang lemas dan saturasi 88"
- "kaki saya nyeri dan demam setelah cedera, sebelumnya pernah diobati"

Dengan konteks yang lebih spesifik, similarity score antar query dan dokumen/riwayat biasanya lebih tinggi, sehingga retrieval dapat aktif dan persentase RAG meningkat.

### Metrik tambahan yang tersedia di admin
Pada admin juga terdapat sub-metrik:

- `RAG History`: persentase kasus yang memanfaatkan riwayat pasien lama
- `RAG Guidelines`: persentase kasus yang memanfaatkan panduan klinis
- `RAG Both`: persentase kasus yang memakai keduanya

Artinya, admin tidak hanya melihat apakah informasi bersumber dari RAG atau General, tetapi juga dari sumber RAG mana yang dominan.

## Catatan penting

RAG di aplikasi ini bukan sekadar penyimpanan dokumen. Ia bekerja sebagai sistem pencarian semantik dan konteks yang relevan sebelum model membuat keputusan atau rangkuman klinis. Dengan cara ini, keputusan AI lebih berbasis pada data pasien dan panduan klinis yang spesifik, bukan hanya pola bahasa umum.

---

Dokumentasi ini dibuat agar alur kerja RAG pada sistem ini mudah dipahami dan dapat dikembangkan lebih lanjut di masa depan.

Untuk aplikasi Anda, evaluasi sebaiknya mencakup 6 kelompok utama:

Ekstraksi informasi klinis

Apakah model benar membaca RR, SpO2, tekanan darah, HR, dan AVPU.
Ukur: accuracy, precision, recall, F1-score.
Untuk angka vital, gunakan MAE atau persentase nilai yang diekstraksi tepat.
SOAP note

Kesesuaian bagian S, O, A, dan P dengan jawaban dokter.
Nilai: ketepatan, kelengkapan, relevansi, dan konsistensi.
Sebaiknya dinilai oleh minimal satu atau beberapa dokter menggunakan skala 1 sampai 5.
Klasifikasi triage/NEWS2

Apakah status seperti Alert, Urgent, atau Emergency benar.
Fokus utama: recall/sensitivity kondisi gawat dan jumlah false negative.
Buat confusion matrix.
Evaluasi RAG

Apakah riwayat pasien dan panduan klinis yang diambil relevan.
Apakah jawaban model benar-benar didukung oleh sumber tersebut.
Ukur: precision@k, recall@k, atau penilaian relevansi oleh ahli.
Perbandingan model
Bandingkan model dengan input dan kasus yang sama berdasarkan:

F1-score ekstraksi
sensitivitas deteksi kondisi gawat
skor penilaian dokter
hallucination/error rate
latency
biaya per permintaan
Keandalan dan keamanan

Respons ketika data vital tidak lengkap.
Respons terhadap typo dan bahasa informal.
Error API atau output JSON tidak valid.
Perlindungan data pasien.
Apakah model memberikan rujukan yang tepat saat kondisi darurat.
Prioritas minimum untuk penelitian Anda:

Komponen	Metrik utama
Ekstraksi tanda vital	Precision, recall, F1
NEWS2/triage	Sensitivity, specificity, false negative
SOAP note	Skor validasi dokter
RAG	Relevansi retrieval dan groundedness
Model	Perbandingan F1, safety, latency
Sistem	Error rate dan waktu respons
Anda perlu membuat dataset evaluasi dengan kolom seperti:

Input_Narrative, Ground_Truth_S, Ground_Truth_O, Ground_Truth_A, Ground_Truth_P, Ground_Truth_SpO2, Ground_Truth_BP, Ground_Truth_Triage, dan Doctor_Validation.

Catatan penting: jangan membagi data evaluasi berdasarkan output model. Tetapkan jawaban benar terlebih dahulu oleh dokter, lalu bandingkan output setiap model terhadap jawaban tersebut.