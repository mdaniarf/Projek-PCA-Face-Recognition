import os
import cv2
import shutil
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="PCA Face Recognition System", layout="wide")

# ==============================================================================
# 1. KONSTANTA & CONFIGURATION
# ==============================================================================
IMG_SIZE = (100, 100)  # Ukuran standar sesuai Langkah 2 PPT
RAW_DIR = "dataset"     # Menyesuaikan folder root dataset Anda
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# ==============================================================================
# 2. FUNGSI UTAMA BACKEND (PCA, PREPROCESSING, & METRIK)
# ==============================================================================
def split_faces_dataset(source_dir, train_dir, test_dir, split_ratio=0.8):
    """Membagi dataset per folder identitas menjadi data latih 80% dan data uji 20%"""
    if not os.path.exists(source_dir):
        return False
        
    # Reset folder lama jika ada untuk mencegah duplikasi data
    if os.path.exists(train_dir): shutil.rmtree(train_dir)
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for person_folder in os.listdir(source_dir):
        person_path = os.path.join(source_dir, person_folder)
        if not os.path.isdir(person_path): 
            continue
        
        os.makedirs(os.path.join(train_dir, person_folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, person_folder), exist_ok=True)
        
        # Mengambil semua file gambar di dalam folder (misal: 001A02.jpg, 001A43a.jpg)
        images = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.seed(42)  # Lock seed agar pembagian data konsisten
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        for img in train_images:
            shutil.copy(os.path.join(person_path, img), os.path.join(train_dir, person_folder, img))
        for img in test_images:
            shutil.copy(os.path.join(person_path, img), os.path.join(test_dir, person_folder, img))
    return True

def load_and_preprocess_image(image_path):
    """Langkah 2 PPT: Preprocessing Gambar (Grayscale, Resize, Normalisasi, Flatten)"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gambar tidak ditemukan: {image_path}")
    
    # 1. Mengubah gambar menjadi grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Melakukan resize ke ukuran yang sama (100x100)
    resized = cv2.resize(gray, IMG_SIZE)
    
    # 3. Melakukan normalisasi nilai piksel (0-1)
    normalized = resized / 255.0
    
    # 4. Melakukan flatten (Matriks 100x100 -> Vektor 10.000 elemen)
    return normalized.flatten()

def load_dataset_from_folder(dir_path):
    """Langkah 3 PPT: Membentuk Matriks Data X"""
    X = []
    labels = []
    if not os.path.exists(dir_path):
        return np.array(X), np.array(labels)
        
    for person_name in os.listdir(dir_path):
        person_folder = os.path.join(dir_path, person_name)
        if not os.path.isdir(person_folder): 
            continue
        
        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(person_folder, filename)
                try:
                    vector = load_and_preprocess_image(img_path)
                    X.append(vector)
                    labels.append(person_name)  # Menggunakan nama folder (misal: '001') sebagai label identitas
                except:
                    continue
    return np.array(X), np.array(labels)

def calculate_metrics(face_1_pca, face_2_pca):
    """Langkah 6 PPT: Menghitung kemiripan menggunakan 2 metode sekaligus"""
    # Metode A: Euclidean Distance
    euclidean_dist = np.linalg.norm(face_1_pca - face_2_pca)
    
    # Metode B: Cosine Similarity
    cos_sim = cosine_similarity(face_1_pca.reshape(1, -1), face_2_pca.reshape(1, -1))[0][0]
    
    return euclidean_dist, cos_sim

def evaluate_accuracy(X_train_pca, train_labels, X_test_pca, test_labels, method='cosine', threshold=0.75):
    """Mengukur akurasi sistem pada data uji (Target sukses > 50%)"""
    if len(X_test_pca) == 0:
        return 0.0
    correct = 0
    total = len(X_test_pca)
    
    for i, test_vec in enumerate(X_test_pca):
        if method == 'cosine':
            # Cari nilai cosine similarity tertinggi terhadap database latih
            similarities = cosine_similarity(test_vec.reshape(1, -1), X_train_pca)[0]
            best_idx = np.argmax(similarities)
            pred_label = train_labels[best_idx] if similarities[best_idx] >= threshold else "Tidak dikenal"
        else:
            # Cari jarak euclidean terpendek terhadap database latih
            distances = np.linalg.norm(X_train_pca - test_vec, axis=1)
            best_idx = np.argmin(distances)
            pred_label = train_labels[best_idx] if distances[best_idx] <= threshold else "Tidak dikenal"
            
        if pred_label == test_labels[i]:
            correct += 1
            
    return (correct / total) * 100

# ==============================================================================
# 3. INTERFACES & UI/UX (STREAMLIT)
# ==============================================================================
st.title("🧮 Sistem Deteksi Kemiripan Wajah Berbasis PCA/SVD")
st.write("Aplikasi GUI interaktif untuk mereduksi dimensi data wajah menjadi ruang Eigenfaces serta menguji kemiripannya.")

# Sidebar - Konfigurasi Parameter
st.sidebar.header("⚙️ Konfigurasi Model PCA")
n_components = st.sidebar.slider("Jumlah Komponen Utama ($k$)", 5, 100, 50)  # Default 50 sesuai dokumen
euclidean_threshold = st.sidebar.slider("Threshold Jarak Euclidean", 1.0, 50.0, 15.0)
cosine_threshold = st.sidebar.slider("Threshold Cosine Similarity", 0.0, 1.0, 0.75)

# Cek keberadaan dataset awal
dataset_ready = False
if os.path.exists(TRAIN_DIR) and len(os.listdir(TRAIN_DIR)) > 0:
    dataset_ready = True
else:
    st.info(f"ℹ️ Folder data latih kosong atau belum terbagi. Mencoba mendeteksi dataset mentah di folder `{RAW_DIR}`...")
    if os.path.exists(RAW_DIR) and len(os.listdir(RAW_DIR)) > 0:
        with st.spinner("Membagi dataset secara otomatis (80% Latih, 20% Uji)..."):
            success = split_faces_dataset(RAW_DIR, TRAIN_DIR, TEST_DIR)
            if success:
                st.success("✅ Dataset berhasil dibagi dengan proporsi 80:20!")
                dataset_ready = True
                st.rerun()
            else:
                st.error("❌ Gagal memproses pembagian dataset.")
    else:
        st.error(f"⚠️ Folder `{RAW_DIR}/` tidak ditemukan atau kosong! Pastikan Anda menaruh data dengan struktur berikut:")
        st.code(f"""
Struktur direktori proyek Anda:
face-recognition-pca/
├── app.py
└── {RAW_DIR}/
    ├── 001/
    │   ├── 001A02.jpg
    │   ├── 001A43a.jpg
    │   └── 001A43b.jpg
    └── 002/
        └── ...
        """)
        st.stop()

# Load dan Cache data wajah agar performa komputasi ringan
@st.cache_data
def get_cached_dataset():
    X_tr, y_tr = load_dataset_from_folder(TRAIN_DIR)
    X_te, y_te = load_dataset_from_folder(TEST_DIR)
    return X_tr, y_tr, X_te, y_te

X_train, y_train, X_test, y_test = get_cached_dataset()

# Training Core Model PCA (Langkah 5 PPT: Menggunakan SVD secara internal)
pca = PCA(n_components=n_components, svd_solver='full')
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Membuat Tab Tampilan UI/UX
tab1, tab2, tab3 = st.tabs(["📊 Analisis Data Terbuka (EDA)", "📸 Pengujian Kemiripan Gambar", "📈 Evaluasi Akurasi Sistem"])

# ------------------------------------------------------------------------------
# TAB 1: EDA (EXPLORATORY DATA ANALYSIS)
# ------------------------------------------------------------------------------
with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Informasi Dimensi Matriks Data")
        df_info = pd.DataFrame({
            "Deskripsi Metrik Data": [
                "Jumlah Gambar Data Latih ($m$)", 
                "Jumlah Gambar Data Uji", 
                "Dimensi Fitur Piksel Asli ($n$)", 
                "Dimensi Fitur Tereduksi setelah PCA ($k$)"
            ],
            "Nilai Ukuran": [X_train.shape[0], X_test.shape[0], X_train.shape[1], n_components]
        })
        st.table(df_info)
        st.info(f"💡 **Langkah 3 PPT Terpenuhi:** Ukuran awal matriks data $X$ adalah **{X_train.shape[0]} × {X_train.shape[1]}**.")
        
    with col2:
        st.subheader("Distribusi Jumlah Foto Per Identitas Folder (Data Latih)")
        unique_labels, counts = np.unique(y_train, return_counts=True)
        chart_data = pd.DataFrame({"Jumlah Sampel Foto": counts}, index=unique_labels)
        st.bar_chart(chart_data)

    st.write("---")
    st.subheader("Visualisasi Pola Matriks SVD (Wajah Rata-rata & Eigenfaces)")
    st.write("Sesuai konsep Eigenfaces (Langkah 4 & 5 PPT), gambar di bawah merupakan hasil visualisasi kolom-kolom komponen utama matriks $V$.")
    
    # Mengambil mean face (Langkah 4 PPT: Centering Data)
    mean_face = pca.mean_.reshape(100, 100)
    
    col_m, col_e = st.columns([1, 4])
    with col_m:
        st.image(mean_face, caption="Mean Face (Rata-rata)
