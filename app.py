import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pca_model import load_dataset_from_folder, calculate_metrics, evaluate_accuracy
from sklearn.decomposition import PCA
import os
import cv2

st.set_page_config(page_title="PCA Face Recognition System", layout="wide")

st.title("🧮 Sistem Deteksi Kemiripan Wajah Berbasis PCA/SVD")
st.write("Aplikasi untuk mereduksi dimensi data wajah menjadi ruang Eigenfaces dan mendeteksi kemiripan wajah[cite: 2, 9, 10].")

# Pilihan path dataset
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# Validasi Folder Data
if not os.path.exists(TRAIN_DIR) or len(os.listdir(TRAIN_DIR)) == 0:
    st.error("⚠️ Silakan kumpulkan dataset di folder `data/train` dan `data/test` terlebih dahulu!")
    st.stop()

# Load Data & Cache agar performa cepat
@st.cache_data
def get_cached_data():
    X_train, y_train = load_dataset_from_folder(TRAIN_DIR)
    X_test, y_test = load_dataset_from_folder(TEST_DIR)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = get_cached_data()

# ==========================================
# SIDEBAR: Pengaturan Parameter & Komponen Utama
# ==========================================
st.sidebar.header("⚙️ Konfigurasi Model PCA")
n_components = st.sidebar.slider("Jumlah Komponen Utama ($k$)", 5, 100, 50) # Default 50 sesuai dokumen [cite: 76, 178]
euclidean_threshold = st.sidebar.slider("Threshold Euclidean Distance", 1.0, 50.0, 15.0) # [cite: 127]
cosine_threshold = st.sidebar.slider("Threshold Cosine Similarity", 0.0, 1.0, 0.75)

# Training Model PCA
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train) # [cite: 179]
X_test_pca = pca.transform(X_test)

# ==========================================
# TAB 1: EDA (Exploratory Data Analysis) & Eigenfaces
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Analisis Data (EDA)", "📸 Pengujian Kemiripan Wajah", "📈 Evaluasi & Akurasi"])

with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Informasi Dimensi Data")
        df_info = pd.DataFrame({
            "Metrik Data": ["Jumlah Sampel Latih (m)", "Jumlah Sampel Uji", "Fitur Asli (Piksel n)", "Fitur Tereduksi (k)"],
            "Nilai": [X_train.shape[0], X_test.shape[0], X_train.shape[1], n_components] # [cite: 52, 53, 180, 181]
        })
        st.table(df_info)
        
    with col2:
        st.subheader("Distribusi Kelas Gambar Latih")
        unique_labels, counts = np.unique(y_train, return_counts=True)
        st.bar_chart(pd.DataFrame({"Jumlah Foto": counts}, index=unique_labels))

    st.write("---")
    st.subheader("Visualisasi Hubungan SVD: Wajah Rata-rata & Eigenfaces")
    
    # Visualisasi Mean Face [cite: 62]
    mean_face = pca.mean_.reshape(100, 100)
    
    col_m, col_e = st.columns([1, 3])
    with col_m:
        st.image(mean_face, caption="Mean Face (Wajah Rata-rata)", use_container_width=True, clamp=True, channels="GRAY")
        
    with col_e:
        # Tampilkan 5 Eigenfaces pertama [cite: 71, 249]
        fig, axes = plt.subplots(1, 5, figsize=(10, 3))
        for i in range(5):
            if i < len(pca.components_):
                eigenface = pca.components_[i].reshape(100, 100)
                axes[i].imshow(eigenface, cmap='gray')
                axes[i].set_title(f"Eigen #{i+1}")
                axes[i].axis('off')
        st.pyplot(fig)
        
    # Cumulative Explained Variance
    st.subheader("Cumulative Explained Variance Ratio")
    cum_variance = np.cumsum(pca.explained_variance_ratio_) # [cite: 182]
    st.line_chart(cum_variance)
    st.info(f"Total informasi (variance) yang dipertahankan dengan {n_components} komponen: **{cum_variance[-1]*100:.2f}%** [cite: 9, 182]")

# ==========================================
# TAB 2: Pengujian Kemiripan Wajah (2 Metode Wajib + Uji Ukuran Foto)
# ==========================================
with tab2:
    st.header("Deteksi Kemiripan Antara Dua Foto Wajah")
    st.write("Silakan unggah 2 gambar acak untuk membandingkan kemiripannya. Sistem akan secara otomatis melakukan resizing (menangani foto besar maupun foto kecil)[cite: 39, 42].")
    
    c1, c2 = st.columns(2)
    with c1:
        file1 = st.file_uploader("Unggah Gambar Wajah A", type=["jpg", "png", "jpeg"], key="img1")
    with c2:
        file2 = st.file_uploader("Unggah Gambar Wajah B", type=["jpg", "png", "jpeg"], key="img2")
        
    if file1 and file2:
        # Simpan sementara file unggahan untuk diproses cv2
        for f, name in zip([file1, file2], ["temp_a.jpg", "temp_b.jpg"]):
            with open(name, "wb") as buffer:
                buffer.write(f.read())
                
        # Menjawab poin 4: Menguji foto kecil vs foto besar
        img_a_raw = cv2.imread("temp_a.jpg")
        img_b_raw = cv2.imread("temp_b.jpg")
        
        st.warning(f"📐 **Ukuran Awal Gambar**: Wajah A berukuran `{img_a_raw.shape[:2]}` | Wajah B berukuran `{img_b_raw.shape[:2]}`. Sistem akan menyeragamkannya menjadi `(100, 100)` melalui tahap Preprocessing[cite: 39, 42].")
        
        # Ekstraksi Fitur via Model PCA
        feat_a = load_and_preprocess_image("temp_a.jpg").reshape(1, -1)
        feat_b = load_and_preprocess_image("temp_b.jpg").reshape(1, -1)
        
        feat_a_pca = pca.transform(feat_a)[0] # [cite: 198]
        feat_b_pca = pca.transform(feat_b)[0] # [cite: 199]
        
        # Hitung Menggunakan 2 Metode Berbeda [cite: 95, 97]
        dist_euclidean, sim_cosine = calculate_metrics(feat_a_pca, feat_b_pca)
        
        # Tampilkan Hasil Visualisasi Berdampingan
        col_img1, col_img2 = st.columns(2)
        col_img1.image(cv2.cvtColor(img_a_raw, cv2.COLOR_BGR2RGB), caption="Wajah A (Asli)", width=250)
        col_img2.image(cv2.cvtColor(img_b_raw, cv2.COLOR_BGR2RGB), caption="Wajah B (Asli)", width=250)
        
        # Tampilkan Hasil Perhitungan Kemiripan
        st.subheader("📊 Hasil Perbandingan")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Metode A: Jarak Euclidean", value=f"{dist_euclidean:.4f}") # [cite: 98]
            status_euclidean = "🟢 MIRIP" if dist_euclidean < euclidean_threshold else "🔴 TIDAK MIRIP" # [cite: 121, 123]
            st.write(f"Keputusan (Threshold < {euclidean_threshold}): **{status_euclidean}** [cite: 119, 120]")
            
        with res_col2:
            st.metric(label="Metode B: Cosine Similarity", value=f"{sim_cosine:.4f}") # [cite: 108]
            status_cosine = "🟢 MIRIP" if sim_cosine >= cosine_threshold else "🔴 TIDAK MIRIP" # [cite: 132, 133]
            st.write(f"Keputusan (Threshold ≥ {cosine_threshold}): **{status_cosine}** [cite: 131]")

# ==========================================
# TAB 3: Pengujian Akurasi Sistem (Target Sukses > 50%)
# ==========================================
with tab3:
    st.header("Evaluasi Akurasi Seluruh Dataset Uji (20%)")
    st.write("Sistem melakukan uji validasi silang otomatis membandingkan performa akurasi antara metode Euclidean & Cosine Similarity.")
    
    acc_euclidean = evaluate_accuracy(X_train_pca, y_train, X_test_pca, y_test, method='euclidean', threshold=euclidean_threshold)
    acc_cosine = evaluate_accuracy(X_train_pca, y_train, X_test_pca, y_test, method='cosine', threshold=cosine_threshold)
    
    c_acc1, c_acc2 = st.columns(2)
    with c_acc1:
        st.subheader("Akurasi Sistem (Metode Euclidean)")
        st.header(f"{acc_euclidean:.2f} %")
        if acc_euclidean > 50.0:
            st.success("✅ Target sukses terpenuhi (>50%)")
        else:
            st.error("❌ Akurasi di bawah 50%, atur ulang parameter slider di sidebar.")
            
    with c_acc2:
        st.subheader("Akurasi Sistem (Metode Cosine Similarity)")
        st.header(f"{acc_cosine:.2f} %")
        if acc_cosine > 50.0:
            st.success("✅ Target sukses terpenuhi (>50%)")
        else:
            st.error("❌ Akurasi di bawah 50%, atur ulang parameter slider di sidebar.")
