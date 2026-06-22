import os
import cv2
import shutil
import random
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Halaman Streamlit (UI/UX)
st.set_page_config(page_title="PCA Face Recognition System", layout="wide")

# ==============================================================================
# 1. KONSTANTA & CONFIGURATION
# ==============================================================================
IMG_SIZE = (100, 100)  # Ukuran standar sesuai dokumen [cite: 149]
RAW_DIR = "dataset"     # Folder dataset utama Anda [cite: 176]
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# Load detektor wajah Haar Cascade bawaan OpenCV [cite: 259]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==============================================================================
# 2. FUNGSI UTAMA BACKEND (CROP, PREPROCESSING, PCA, & EVALUASI)
# ==============================================================================
def detect_and_crop_face(image_path):
    """Mendeteksi wajah dari gambar, lalu mengembalikan crop wajah sesuai Dokumen Halaman 13"""
    img = cv2.imread(image_path) [cite: 259]
    if img is None:
        raise ValueError(f"Gambar tidak ditemukan: {image_path}") [cite: 259]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) [cite: 259]
    
    # Deteksi lokasi koordinat wajah [cite: 259]
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4) [cite: 259]
    
    # Jika wajah terdeteksi, potong (crop) hanya bagian wajahnya saja [cite: 260]
    if len(faces) > 0:
        x, y, w, h = faces[0] [cite: 260]
        face_crop = gray[y:y+h, x:x+w] [cite: 260]
    else:
        # Fallback jika deteksi gagal: gunakan gambar abu-abu penuh
        face_crop = gray
        
    face_resized = cv2.resize(face_crop, IMG_SIZE) [cite: 260]
    face_normalized = face_resized / 255.0 [cite: 260]
    
    return face_normalized.flatten() [cite: 260]

def split_faces_dataset(source_dir, train_dir, test_dir, split_ratio=0.8):
    """Membagi dataset menjadi data latih 80% dan data uji 20% secara konsisten"""
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
        random.seed(42)
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        if split_idx == 0 and len(images) > 0:
            split_idx = 1
            
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        for img in train_images:
            shutil.copy(os.path.join(person_path, img), os.path.join(train_dir, person_folder, img))
        for img in test_images:
            shutil.copy(os.path.join(person_path, img), os.path.join(test_dir, person_folder, img))
    return True

def load_dataset_from_folder(dir_path):
    """Membentuk Matriks Data X dengan memproses file gambar dari folder"""
    X = []
    labels = []
    if not os.path.exists(dir_path):
        return np.array(X), np.array(labels)
        
    for person_name in os.listdir(dir_path): [cite: 166]
        person_folder = os.path.join(dir_path, person_name) [cite: 167]
        if not os.path.isdir(person_folder): [cite: 168]
            continue
        
        for filename in os.listdir(person_folder): [cite: 169]
            if filename.lower().endswith((".jpg", ".jpeg", ".png")): [cite: 170]
                img_path = os.path.join(person_folder, filename) [cite: 170]
                try:
                    vector = detect_and_crop_face(img_path)
                    X.append(vector) [cite: 172]
                    labels.append(person_name) [cite: 173]
                except:
                    continue
    return np.array(X), np.array(labels) [cite: 174]

def calculate_metrics(face_1_pca, face_2_pca):
    """Menghitung jarak Euclidean dan kesamaan Cosine antar vektor wajah"""
    euclidean_dist = np.linalg.norm(face_1_pca - face_2_pca) [cite: 105]
    cos_sim = cosine_similarity(face_1_pca.reshape(1, -1), face_2_pca.reshape(1, -1))[0][0] [cite: 201]
    return euclidean_dist, cos_sim

def evaluate_accuracy_with_threshold(X_train_pca, train_labels, X_test_pca, test_labels, method='cosine', threshold=0.50):
    """Mengukur persentase akurasi sistem terhadap seluruh dataset uji secara dinamis"""
    if len(X_test_pca) == 0:
        return 0.0
    correct = 0
    total = len(X_test_pca)
    
    for i, test_vec in enumerate(X_test_pca):
        if method == 'cosine':
            similarities = cosine_similarity(test_vec.reshape(1, -1), X_train_pca)[0] [cite: 222]
            best_idx = np.argmax(similarities) [cite: 223]
            best_sim = similarities[best_idx] [cite: 223]
            
            if best_sim >= threshold: [cite: 224]
                pred_label = train_labels[best_idx] [cite: 223]
            else:
                pred_label = "Tidak Dikenal" [cite: 226]
        else:
            distances = np.linalg.norm(X_train_pca - test_vec, axis=1)
            best_idx = np.argmin(distances)
            best_dist = distances[best_idx]
            
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

# Memastikan pemecahan data latih & uji siap
if not os.path.exists(TRAIN_DIR) or len(os.listdir(TRAIN_DIR)) == 0:
    split_faces_dataset(RAW_DIR, TRAIN_DIR, TEST_DIR)

# Memuat dataset gambar
X_train_raw, y_train = load_dataset_from_folder(TRAIN_DIR)
X_test_raw, y_test = load_dataset_from_folder(TEST_DIR)

# Sidebar - Konfigurasi Parameter Interaktif
st.sidebar.header("⚙️ Konfigurasi Model PCA")

# Batasi n_components secara aman sesuai jumlah gambar latih yang Anda miliki
max_components = max(2, min(len(X_train_raw), 100))
n_components = st.sidebar.slider("Jumlah Komponen Utama ($k$)", 2, max_components, min(5, max_components))

# Slider Threshold untuk mencocokkan keputusan klasifikasi tugas Anda
euclidean_threshold = st.sidebar.slider("Threshold Jarak Euclidean", 1.0, 100.0, 64.0)  
cosine_threshold = st.sidebar.slider("Threshold Cosine Similarity", -1.0, 1.0, 0.16)      

# Ekstraksi Fitur Menggunakan PCA (Berbasis SVD Full Solver) [cite: 68, 178]
pca = PCA(n_components=n_components, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_raw) if len(X_train_raw) > 0 else np.array([]) [cite: 179]
X_test_pca = pca.transform(X_test_raw) if len(X_test_raw) > 0 else np.array([])

# Membuat Tab Layout Tampilan UI
tab1, tab2, tab3 = st.tabs(["📊 Analisis Data Terbuka (EDA)", "📸 Pengujian Kemiripan Gambar", "📈 Evaluasi Akurasi Sistem"])

# ------------------------------------------------------------------------------
# TAB 1: EDA (EXPLORATORY DATA ANALYSIS)
# ------------------------------------------------------------------------------
with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Informasi Dimensi Matriks Data")
        if len(X_train_raw) > 0:
            df_info = pd.DataFrame({
                "Deskripsi Metrik Data": [
                    "Jumlah Gambar Data Latih ($m$)",
                    "Jumlah Gambar Data Uji", 
                    "Dimensi Fitur Piksel Asli ($n$)",
                    "Dimensi Fitur Tereduksi setelah PCA ($k$)"
                ],
                "Nilai Ukuran": [int(X_train_raw.shape[0]), int(X_test_raw.shape[0]), int(X_train_raw.shape[1]), int(n_components)]
            })
            st.table(df_info)
        else:
            st.warning("Data latih belum siap atau kosong.")
        
    with col2:
        st.subheader("Distribusi Jumlah Foto Per Identitas")
        if len(y_train) > 0:
            unique_labels, counts = np.unique(y_train, return_counts=True)
            chart_data = pd.DataFrame({"Jumlah Sampel Foto": counts}, index=unique_labels)
            st.bar_chart(chart_data)

# ------------------------------------------------------------------------------
# TAB 2: PENGUJIAN KEMIRIPAN GAMBAR INDIVIDUAL
# ------------------------------------------------------------------------------
with tab2:
    st.header("Deteksi Kemiripan Antara Dua Foto")
    
    c1, c2 = st.columns(2)
    with c1: 
        file1 = st.file_uploader("Unggah Gambar Wajah Pertama (A)", type=["jpg", "png", "jpeg"], key="upload_1")
    with c2: 
        file2 = st.file_uploader("Unggah Gambar Wajah Kedua (B)", type=["jpg", "png", "jpeg"], key="upload_2")
        
    if file1 and file2:
        with open("temp_a.jpg", "wb") as f: f.write(file1.read())
        with open("temp_b.jpg", "wb") as f: f.write(file2.read())
                
        feat_a = detect_and_crop_face("temp_a.jpg").reshape(1, -1)
        feat_b = detect_and_crop_face("temp_b.jpg").reshape(1, -1)
        
        feat_a_pca = pca.transform(feat_a)[0] [cite: 198]
        feat_b_pca = pca.transform(feat_b)[0] [cite: 199]
        
        dist_euclidean, sim_cosine = calculate_metrics(feat_a_pca, feat_b_pca)
        
        st.image([cv2.cvtColor(cv2.imread("temp_a.jpg"), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.imread("temp_b.jpg"), cv2.COLOR_BGR2RGB)], width=240, caption=["Wajah A", "Wajah B"])
        
        st.subheader("⚖️ Perbandingan Hasil Keputusan Dua Metode")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Metode A: Jarak Euclidean", value=f"{dist_euclidean:.4f}")
            status_euclidean = "🟢 MIRIP" if dist_euclidean < euclidean_threshold else "🔴 TIDAK MIRIP" [cite: 121, 123]
            st.write(f"Keputusan Sistem (Threshold < {euclidean_threshold}): **{status_euclidean}**")
            
        with res_col2:
            st.metric(label="Metode B: Cosine Similarity", value=f"{sim_cosine:.4f}")
            status_cosine = "🟢 MIRIP" if sim_cosine >= cosine_threshold else "🔴 TIDAK MIRIP" [cite: 132, 133]
            st.write(f"Keputusan Sistem (Threshold ≥ {cosine_threshold}): **{status_cosine}**")

# ------------------------------------------------------------------------------
# TAB 3: EVALUASI AKURASI SISTEM DENGAN THRESHOLD DYNAMIC
# ------------------------------------------------------------------------------
with tab3:
    st.header("Evaluasi Akurasi Seluruh Dataset Uji (Proporsi 20%)")
    
    if len(X_test_pca) > 0:
        acc_euclidean = evaluate_accuracy_with_threshold(X_train_pca, y_train, X_test_pca, y_test, method='euclidean', threshold=euclidean_threshold)
        acc_cosine = evaluate_accuracy_with_threshold(X_train_pca, y_train, X_test_pca, y_test, method='cosine', threshold=cosine_threshold)
            
        c_acc1, c_acc2 = st.columns(2)
        with c_acc1:
            st.subheader("Akurasi Menggunakan Metode Euclidean")
            st.metric(label="Nilai Akurasi", value=f"{acc_euclidean:.2f} %")
            if acc_euclidean >= 50.0:
                st.success("✅ Sukses: Akurasi di atas batas kelulusan target 50%!")
            else:
                st.error("❌ Target Belum Tercapai: Silakan naikkan slider 'Threshold Jarak Euclidean' ke arah kanan (misal: 60-75).")
                
        with c_acc2:
            st.subheader("Akurasi Menggunakan Metode Cosine Similarity")
            st.metric(label="Nilai Akurasi", value=f"{acc_cosine:.2f} %")
            if acc_cosine >= 50.0:
                st.success("✅ Sukses: Akurasi di atas batas kelulusan target 50%!")
            else:
                st.error("❌ Target Belum Tercapai: Silakan turunkan slider 'Threshold Cosine Similarity' ke arah kiri (misal: 0.10-0.20).")
    else:
        st.warning("Matriks data uji kosong, pastikan isi folder dataset sudah benar.")
