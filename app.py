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

# Konfigurasi Halaman Streamlit (UI/UX)
st.set_page_config(page_title="PCA Face Recognition System", layout="wide")

# ==============================================================================
# 1. KONSTANTA & CONFIGURATION
# ==============================================================================
IMG_SIZE = (100, 100)  # Ukuran standar sesuai Langkah 2 dokumen
RAW_DIR = "dataset"     # Folder dataset utama Anda
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# ==============================================================================
# 2. FUNGSI UTAMA BACKEND (PREPROCESSING, PCA, & EVALUASI)
# ==============================================================================
def split_faces_dataset(source_dir, train_dir, test_dir, split_ratio=0.8):
    """Membagi dataset per folder identitas menjadi data latih 80% dan data uji 20%"""
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
        random.seed(42)  # Lock seed agar pembagian data konsisten
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

def load_and_preprocess_image(image_path):
    """Langkah 2 Dokumen: Mengubah ke grayscale, resize, normalisasi, dan flatten"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gambar tidak ditemukan: {image_path}")
    
    # 1. Mengubah gambar menjadi grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Melakukan resize ke ukuran standar (100x100)
    resized = cv2.resize(gray, IMG_SIZE)
    
    # 3. Melakukan normalisasi nilai piksel (0-1)
    normalized = resized / 255.0
    
    # 4. Melakukan flatten menjadi vektor 1D
    return normalized.flatten()

def load_dataset_from_folder(dir_path):
    """Langkah 3 Dokumen: Membentuk Matriks Data X"""
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
                    labels.append(person_name)
                except:
                    continue
    return np.array(X), np.array(labels)

def calculate_metrics(face_1_pca, face_2_pca):
    """Menghitung kemiripan menggunakan 2 metode wajib dokumen"""
    # Metode A: Euclidean Distance
    euclidean_dist = np.linalg.norm(face_1_pca - face_2_pca)
    
    # Metode B: Cosine Similarity
    cos_sim = cosine_similarity(face_1_pca.reshape(1, -1), face_2_pca.reshape(1, -1))[0][0]
    
    return euclidean_dist, cos_sim

def evaluate_accuracy_with_threshold(X_train_pca, train_labels, X_test_pca, test_labels, method='cosine', threshold=0.50):
    """Mengukur akurasi dengan mempertimbangkan nilai ambang batas (Threshold) secara dinamis"""
    if len(X_test_pca) == 0:
        return 0.0
    correct = 0
    total = len(X_test_pca)
    
    for i, test_vec in enumerate(X_test_pca):
