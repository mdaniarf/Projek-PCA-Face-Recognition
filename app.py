import os
import cv2
import shutil
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Halaman Streamlit (UI/UX)
st.set_page_config(page_title="PCA Face Recognition System", layout="wide")

# ==============================================================================
# 1. KONSTANTA & CONFIGURATION
# ==============================================================================
IMG_SIZE = (100, 100)  # Ukuran standar sesuai dokumen [cite: 42, 149]
RAW_DIR = "dataset"     # Folder dataset utama Anda
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# ==============================================================================
# 2. FUNGSI UTAMA BACKEND (PREPROCESSING, PCA, & EVALUASI)
# ==============================================================================
def split_faces_dataset(source_dir, train_dir, test_dir, split_ratio=0.8):
    """Membagi dataset per folder identitas secara konsisten"""
    if not os.path.exists(source_dir):
        return False
        
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
        
        images = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Menggunakan seed tetap agar pembagian data stabil dan reproducible
        random.seed(42)  
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        if split_idx == 0 and len(images) > 0:  # Proteksi jika gambar terlalu sedikit
            split_idx = 1
            
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        for img in train_images:
            shutil.copy(os.path.join(person_path, img), os.path.join(train_dir, person_folder, img))
        for img in test_images:
            shutil.copy(os.path.join(person_path, img), os.path.join(test_dir, person_folder, img))
    return True

def load_and_preprocess_image(image_path):
    """Langkah 2 Dokumen: Grayscale, Crop, Resize, Normalisasi [cite: 38, 40]"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gambar tidak ditemukan: {image_path}")
    
    # 1. Mengubah gambar menjadi grayscale [cite: 41, 155]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tambahan Preprocessing: Perbaikan Kontras Cahaya Mandat Dokumen 
    gray = cv2.equalizeHist(gray)
    
    # Face Detection menggunakan Haar Cascade [cite: 257, 259]
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
    
    if len(faces) > 0:
        x, y, w, h = faces[0] [cite: 260]
        processed_img = gray[y:y+h, x:x+w] [cite: 260]
    else:
        processed_img = gray  # Fallback gambar utuh jika angle miring 
    
    # 2. Melakukan resize ke ukuran standar (100x100) [cite: 42, 155]
    resized = cv2.resize(processed_img, IMG_SIZE)
    
    # 3. Melakukan normalisasi nilai piksel (0-1) [cite: 43, 157]
    normalized = resized / 255.0
    
    # 4. Melakukan flatten menjadi vektor 1D [cite: 44, 159]
    return normalized.flatten()

def load_dataset_from_folder(dir_path):
    """Langkah 3 Dokumen: Membentuk Matriks Data X [cite: 47]"""
    X = []
    labels = []
    if not os.path.exists(dir_path):
        return np.array(X), np.array(labels)
        
    for person_name in os.listdir(dir_path):
        person_folder = os.path.join(dir_path, person_name)
        if not os.path.isdir(person_folder): 
            continue
        
        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")): [cite: 170]
                img_path = os.path.join(person_folder, filename) [cite: 170]
                try:
                    vector = load_and_preprocess_image(img_path)
                    X.append(vector)
                    labels.append(person_name) [cite: 173]
                except:
                    continue
    return np.array(X), np.array(labels)

def calculate_metrics(face_1_pca, face_2_pca):
    """Langkah 6 Dokumen: Menghitung kemiripan menggunakan 2 metode [cite: 95, 97]"""
    # Metode A: Euclidean Distance [cite: 98]
    euclidean_dist = np.linalg.norm(face_1_pca - face_2_pca) [cite: 105]
    
    # Metode B: Cosine Similarity [cite: 108]
    cos_sim = cosine_similarity(face_1_pca.reshape(1, -1), face_2_pca.reshape(1, -1))[0][0] [cite: 111, 201]
    
    return euclidean_dist, cos_sim

def evaluate_accuracy_with_threshold(X_train_pca, train_labels, X_test_pca, test_labels, method='cosine', threshold=0.50):
    """Mengukur akurasi dengan mencocokkan wajah terdekat yang lolos batas threshold [cite: 119]"""
    if len(X_test_pca) == 0:
        return 0.0
    correct = 0
    total = len(X_test_pca)
    
    for i, test_vec in enumerate(X_test_pca):
        if method == 'cosine':
            similarities = cosine_similarity(test_vec.reshape(1, -1), X_train_pca)[0] [cite: 222]
            best_idx = np.argmax(similarities) [cite: 223]
            best_sim = similarities[best_idx] [cite: 223]
            
            # Validasi aturan threshold dokumen [cite: 132]
            if best_sim >= threshold:
                pred_label = train_labels[best_idx]
            else:
                pred_label = "Tidak Dikenal"
        else:
            distances = np.linalg.norm(X_train_pca - test_vec, axis=1)
            best_idx = np.argmin(distances)
            best_dist = distances[best_idx]
            
            # Validasi aturan threshold dokumen [cite: 121]
            if best_dist < threshold:
                pred_label = train_labels[best_idx]
            else:
                pred_label = "Tidak Dikenal"
                
        if pred_label == test_labels[i]:
            correct += 1
            
    return (correct / total) * 100

# ==============================================================================
# 3. INTERFACES & UI/UX (STREAMLIT)
# ==============================================================================
st.title("🧮 Sistem Deteksi Kemiripan Wajah Berbasis PCA/SVD")
st.write("Aplikasi GUI interaktif untuk mereduksi dimensi data wajah menjadi ruang Eigenfaces serta menguji kemiripannya.")

# Sidebar - Konfigurasi Parameter Interaktif
st.sidebar.header("⚙️ Konfigurasi Model PCA")

# Membaca dataset latih terlebih dahulu untuk menentukan batas maksimal n_components dinamis
X_train_raw, y_train, X_test_raw, y_test = load_dataset_from_folder(TRAIN_DIR), None, None, None
if os.path.exists(TRAIN_DIR) and len(os.listdir(TRAIN_DIR)) > 0:
    X_train_raw, y_train = load_dataset_from_folder(TRAIN_DIR)
    X_test_raw, y_test = load_dataset_from_folder(TEST_DIR)
else:
    # Jalankan pembagian data otomatis di awal jika kosong
    split_faces_dataset(RAW_DIR, TRAIN_DIR, TEST_DIR)
    X_train_raw, y_train = load_dataset_from_folder(TRAIN_DIR)
    X_test_raw, y_test = load_dataset_from_folder(TEST_DIR)

# Batasi maksimal n_components agar tidak melebihi jumlah sampel yang memicu ValueError
max_components = max(2, min(len(X_train_raw), 50))
n_components = st.sidebar.slider("Jumlah Komponen Utama ($k$)", 2, max_components, min(5, max_components))

# Konfigurasi Threshold Interaktif dokumen [cite: 119]
euclidean_threshold = st.sidebar.slider("Threshold Jarak Euclidean", 1.0, 100.0, 55.0)  
cosine_threshold = st.sidebar.slider("Threshold Cosine Similarity", -1.0, 1.0, 0.15)      

# standardisasi fitur data (Z-score Scaling) untuk mengatasi gangguan pencahayaan [cite: 254]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw) if len(X_train_raw) > 0 else X_train_raw
X_test_scaled = scaler.transform(X_test_raw) if len(X_test_raw) > 0 else X_test_raw

# Proses Reduksi PCA / SVD [cite: 64, 65]
pca = PCA(n_components=n_components, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Membuat Tab Tampilan UI/UX
tab1, tab2, tab3 = st.tabs(["📊 Analisis Data Terbuka (EDA)", "📸 Pengujian Kemiripan Gambar", "📈 Evaluasi Akurasi Sistem"])

# ------------------------------------------------------------------------------
# TAB 1: EDA
# ------------------------------------------------------------------------------
with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Informasi Dimensi Matriks Data")
        df_info = pd.DataFrame({
            "Deskripsi Metrik Data": ["Jumlah Gambar Data Latih ($m$)", "Jumlah Gambar Data Uji", "Dimensi Fitur Piksel Asli ($n$)", "Dimensi Fitur Tereduksi setelah PCA ($k$)"],
            "Nilai Ukuran": [int(X_train_raw.shape[0]), int(X_test_raw.shape[0]), int(X_train_raw.shape[1]), int(n_components)]
        })
        st.table(df_info)
        
    with col2:
        st.subheader("Distribusi Sampel Data Latih")
        unique_labels, counts = np.unique(y_train, return_counts=True)
        st.bar_chart(pd.DataFrame({"Jumlah Sampel Foto": counts}, index=unique_labels))

# ------------------------------------------------------------------------------
# TAB 2: PENGUJIAN KEMIRIPAN GAMBAR INDIVIDUAL
# ------------------------------------------------------------------------------
with tab2:
    st.header("Deteksi Kemiripan Antara Dua Foto")
    c1, c2 = st.columns(2)
    with c1: file1 = st.file_uploader("Unggah Gambar Wajah Pertama (A)", type=["jpg", "png", "jpeg"], key="upload_1")
    with c2: file2 = st.file_uploader("Unggah Gambar Wajah Kedua (B)", type=["jpg", "png", "jpeg"], key="upload_2")
        
    if file1 and file2:
        with open("temp_a.jpg", "wb") as f: f.write(file1.read())
        with open("temp_b.jpg", "wb") as f: f.write(file2.read())
                
        feat_a = load_and_preprocess_image("temp_a.jpg").reshape(1, -1)
        feat_b = load_and_preprocess_image("temp_b.jpg").reshape(1, -1)
        
        feat_a_scaled = scaler.transform(feat_a)
        feat_b_scaled = scaler.transform(feat_b)
        
        feat_a_pca = pca.transform(feat_a_scaled)[0]
        feat_b_pca = pca.transform(feat_b_scaled)[0]
        
        dist_euclidean, sim_cosine = calculate_metrics(feat_a_pca, feat_b_pca)
        
        st.image([cv2.cvtColor(cv2.imread("temp_a.jpg"), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.imread("temp_b.jpg"), cv2.COLOR_BGR2RGB)], width=240, caption=["Wajah A", "Wajah B"])
        
        st.subheader("⚖️ Hasil Keputusan Dua Metode")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Jarak Euclidean", value=f"{dist_euclidean:.4f}")
            st.write(f"Keputusan (< {euclidean_threshold}): **{'🟢 MIRIP' if dist_euclidean < euclidean_threshold else '🔴 TIDAK MIRIP'}**") [cite: 121]
        with res_col2:
            st.metric(label="Cosine Similarity", value=f"{sim_cosine:.4f}")
            st.write(f"Keputusan (≥ {cosine_threshold}): **{'🟢 MIRIP' if sim_cosine >= cosine_threshold else '🔴 TIDAK MIRIP'}**") [cite: 132]

# ------------------------------------------------------------------------------
# TAB 3: EVALUASI AKURASI SISTEM DENGAN THRESHOLD DYNAMIC
# ------------------------------------------------------------------------------
with tab3:
    st.header("Evaluasi Akurasi Seluruh Dataset Uji (Proporsi 20%)")
    
    acc_euclidean = evaluate_accuracy_with_threshold(X_train_pca, y_train, X_test_pca, y_test, method='euclidean', threshold=euclidean_threshold)
    acc_cosine = evaluate_accuracy_with_threshold(X_train_pca, y_train, X_test_pca, y_test, method='cosine', threshold=cosine_threshold)
        
    c_acc1, c_acc2 = st.columns(2)
    with c_acc1:
        st.subheader("Akurasi Menggunakan Metode Euclidean")
        st.metric(label="Nilai Akurasi", value=f"{acc_euclidean:.2f} %")
        if acc_euclidean >= 50.0: st.success("✅ Sukses: Di atas target 50%!")
        else: st.error("❌ Masih di bawah 50%, silakan geser Slider Threshold Euclidean ke Kanan.")
            
    with c_acc2:
        st.subheader("Akurasi Menggunakan Metode Cosine Similarity")
        st.metric(label="Nilai Akurasi", value=f"{acc_cosine:.2f} %")
        if acc_cosine >= 50.0: st.success("✅ Sukses: Di atas target 50%!")
        else: st.error("❌ Masih di bawah 50%, silakan geser Slider Threshold Cosine ke Kiri.")
