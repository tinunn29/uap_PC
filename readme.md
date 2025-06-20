# 🚧 Sistem Deteksi Jalan Berlubang - Setup Guide

## 📋 Deskripsi Proyek

Sistem deteksi jalan berlubang menggunakan **GLCM (Gray Level Co-Occurrence Matrix)** untuk ekstraksi fitur tekstur dan **KNN (K-Nearest Neighbors)** untuk klasifikasi. Sistem ini dapat menganalisis gambar jalan dan menentukan apakah jalan tersebut berlubang atau normal.

## 🎯 Tujuan Pembelajaran

Melalui proyek ini, mahasiswa akan mempelajari:

1. **Computer Vision**: Pemrosesan dan analisis gambar digital
2. **Feature Extraction**: Ekstraksi fitur tekstur menggunakan GLCM
3. **Machine Learning**: Klasifikasi menggunakan algoritma KNN
4. **Web Development**: Membuat aplikasi web dengan Flask
5. **Data Science Pipeline**: Alur kerja dari data hingga deployment

## 🛠️ Persiapan Environment

### 1. Install Python dan Virtual Environment
```bash
# Install Python 3.8+ (jika belum ada)
# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
# Install semua library yang diperlukan
pip install -r requirements.txt
```

### 3. Struktur Folder Proyek
Buat struktur folder seperti berikut:
```
deteksi_jalan_berlubang/
├── app.py
├── requirements.txt
├── dataset/
│   ├── berlubang/          # Gambar jalan berlubang
│   └── normal/             # Gambar jalan normal
├── static/
│   └── uploads/            # Tempat gambar yang diupload user
└── templates/
    ├── index.html
    ├── result.html
    └── info.html
```

## 📸 Persiapan Dataset

### 1. Kumpulkan Gambar
- **Gambar Jalan Berlubang**: 20-50 gambar jalan dengan lubang
- **Gambar Jalan Normal**: 20-50 gambar jalan yang baik
- Format: JPG, PNG, JPEG
- Resolusi: Bebas (sistem akan resize otomatis)

### 2. Organisasi Dataset
```bash
# Buat folder dataset
mkdir dataset
mkdir dataset/berlubang
mkdir dataset/normal

# Pindahkan gambar ke folder yang sesuai
# dataset/berlubang/ → gambar jalan berlubang
# dataset/normal/ → gambar jalan normal
```

### 3. Tips Pengumpulan Data
- Gunakan gambar dari berbagai kondisi pencahayaan
- Variasikan sudut pengambilan gambar
- Pastikan kualitas gambar cukup jelas
- Hindari gambar yang terlalu blur atau gelap

## 🚀 Menjalankan Aplikasi

### 1. Jalankan Server Flask
```bash
python app.py
```

### 2. Akses Aplikasi
Buka browser dan kunjungi: `http://localhost:5000`

### 3. Output yang Diharapkan
```
============================================================
SISTEM DETEKSI JALAN BERLUBANG
Menggunakan GLCM + KNN
============================================================
Loading dataset...
Loading images from berlubang...
  ✓ Processed: lubang1.jpg
  ✓ Processed: lubang2.jpg
  ...
Loading images from normal...
  ✓ Processed: normal1.jpg
  ✓ Processed: normal2.jpg
  ...
Dataset loaded: 45 samples
Classes: ['berlubang' 'normal']
Model trained successfully!
Training samples: 36
Test accuracy: 0.89
------------------------------------------------------------
Starting Flask application...
Access the application at: http://localhost:5000
============================================================
```

## 🧠 Penjelasan Konsep

### 1. GLCM (Gray Level Co-Occurrence Matrix)

GLCM adalah matriks yang menunjukkan seberapa sering kombinasi intensitas pixel muncul dalam gambar.

**Fitur yang diekstraksi:**
- **Contrast**: Mengukur variasi intensitas lokal
- **Dissimilarity**: Mengukur ketidakmiripan pixel bertetangga
- **Homogeneity**: Mengukur keseragaman tekstur
- **Energy**: Mengukur keteraturan distribusi intensitas
- **Correlation**: Mengukur korelasi linear pixel
- **ASM**: Angular Second Moment, ukuran keteraturan

### 2. K-Nearest Neighbors (KNN)

Algoritma klasifikasi yang menggunakan k tetangga terdekat untuk menentukan kelas.

**Cara kerja:**
1. Hitung jarak antara data baru dengan semua data training
2. Pilih k tetangga terdekat
3. Ambil mayoritas label dari k tetangga tersebut
4. Gunakan sebagai prediksi

## 🔬 Eksperimen dan Pengembangan

### 1. Eksperimen Parameter GLCM
```python
# Coba variasi parameter GLCM
distances = [1, 2, 3]  # Jarak pixel
angles = [0, 45, 90, 135]  # Sudut dalam derajat
levels = [64, 128, 256]  # Jumlah level gray
```

### 2. Tuning Parameter KNN
```python
# Coba nilai k yang berbeda
for k in [3, 5, 7, 9]:
    model = KNeighborsClassifier(n_neighbors=k)
    # Evaluasi performa
```

### 3. Evaluasi Model
```python
from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
```

## 📊 Analisis Hasil

### 1. Interpretasi Fitur GLCM
- **Jalan Berlubang**: Biasanya memiliki contrast dan dissimilarity tinggi
- **Jalan Normal**: Biasanya memiliki homogeneity dan energy tinggi

### 2. Evaluasi Performa
- **Accuracy**: Persentase prediksi yang benar
- **Precision**: Ketepatan prediksi positif
- **Recall**: Kemampuan mendeteksi kelas positif
- **F1-Score**: Harmonic mean precision dan recall

## 🚨 Troubleshooting

### 1. Error saat Install Dependencies
```bash
# Jika ada error dengan OpenCV
pip install opencv-python-headless

# Jika ada error dengan scikit-image
pip install --upgrade setuptools
pip install scikit-image
```

### 2. Dataset Tidak Terdeteksi
- Pastikan struktur folder benar
- Cek nama folder: `berlubang` dan `normal`
- Pastikan gambar dalam format yang didukung

### 3. Model Accuracy Rendah
- Tambah lebih banyak data training
- Coba parameter GLCM yang berbeda
- Pertimbangkan preprocessing gambar

## 📚 Pengembangan Lanjutan

### 1. Fitur Tambahan
- [ ] Visualisasi GLCM matrix
- [ ] Real-time detection via webcam
- [ ] Batch processing multiple images
- [ ] Export hasil ke CSV/PDF

### 2. Algoritma Alternatif
- [ ] SVM (Support Vector Machine)
- [ ] Random Forest
- [ ] Deep Learning dengan CNN

### 3. Fitur Tekstur Lain
- [ ] LBP (Local Binary Pattern)
- [ ] Gabor Filter
- [ ] Wavelet Transform

## 🎓 Tugas Praktikum

### Tugas 1: Basic Implementation
1. Setup environment dan install dependencies
2. Siapkan dataset minimal 20 gambar
3. Jalankan sistem dan test dengan 5 gambar berbeda
4. Dokumentasikan hasil dan akurasi

### Tugas 2: Eksperimen Parameter
1. Coba 3 nilai k yang berbeda pada KNN
2. Coba 2 kombinasi parameter GLCM berbeda
3. Bandingkan hasil dan buat analisis
4. Buat grafik perbandingan akurasi

### Tugas 3: Pengembangan Fitur
1. Tambah fitur untuk menampilkan confidence score
2. Implementasi visualisasi GLCM matrix
3. Tambah validasi input gambar yang lebih robust
4. Buat dokumentasi API endpoints

## 📖 Referensi

1. Haralick, R. M. (1979). Statistical and structural approaches to texture
2. Scikit-image documentation: https://scikit-image.org/
3. Scikit-learn KNN: https://scikit-learn.org/stable/modules/neighbors.html
4. Flask documentation: https://flask.palletsprojects.com/

## 💡 Tips untuk Mahasiswa

1. **Pahami Konsep**: Jangan hanya copy-paste, pahami setiap baris kode
2. **Eksperimen**: Coba parameter berbeda dan amati hasilnya
3. **Dokumentasi**: Catat setiap eksperimen dan hasilnya
4. **Visualisasi**: Gunakan matplotlib untuk visualisasi data dan hasil
5. **Kolaborasi**: Diskusikan dengan teman untuk pemahaman yang lebih baik

Selamat belajar dan bereksperimen! 🎉