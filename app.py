import streamlit as st
import cv2
import numpy as np
import os
import re

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(page_title="Face Recognition PCA")

IMG_SIZE = (100, 100)
DATASET_PATH = "dataset"

# =====================================================
# AGE GROUP
# =====================================================

def age_group(age):

    if age <= 12:
        return "Child"

    elif age <= 19:
        return "Teen"

    elif age <= 35:
        return "Young Adult"

    elif age <= 55:
        return "Adult"

    else:
        return "Senior"


# =====================================================
# GET AGE
# =====================================================

def get_age(filename):

    match = re.search(r"A(\d+)", filename)

    if match:
        return int(match.group(1))

    return None


# =====================================================
# PREPROCESS
# =====================================================

def preprocess_image(path):

    img = cv2.imread(path)

    if img is None:
        return None

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        IMG_SIZE
    )

    gray = gray / 255.0

    return gray.flatten()


# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset(dataset_path):

    X = []
    y = []

    if not os.path.exists(dataset_path):
        st.error(
            f"Folder dataset tidak ditemukan: {dataset_path}"
        )
        return np.array([]), np.array([])

    folders = os.listdir(dataset_path)

    for person in folders:

        folder_path = os.path.join(
            dataset_path,
            person
        )

        if not os.path.isdir(folder_path):
            continue

        files = os.listdir(folder_path)

        for file in files:

            if not file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            age = get_age(file)

            if age is None:
                continue

            image_path = os.path.join(
                folder_path,
                file
            )

            vector = preprocess_image(
                image_path
            )

            if vector is None:
                continue

            X.append(vector)
            y.append(age_group(age))

    return np.array(X), np.array(y)


# =====================================================
# TRAIN MODEL
# =====================================================

@st.cache_resource
def train_model():

    X, y = load_dataset(DATASET_PATH)

    if len(X) == 0:
        return None

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    min_class_count = counts.min()

    if min_class_count < 2:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

    else:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    n_components = min(
        30,
        X_train.shape[0] - 1,
        X_train.shape[1]
    )

    pca = PCA(
        n_components=n_components
    )

    X_train_pca = pca.fit_transform(
        X_train
    )

    X_test_pca = pca.transform(
        X_test
    )

    knn = KNeighborsClassifier(
        n_neighbors=3
    )

    knn.fit(
        X_train_pca,
        y_train
    )

    pred = knn.predict(
        X_test_pca
    )

    acc = accuracy_score(
        y_test,
        pred
    )

    return (
        pca,
        knn,
        acc
    )


# =====================================================
# LOAD MODEL
# =====================================================

model_result = train_model()

# =====================================================
# UI
# =====================================================

st.title(
    "Face Recognition Based On Age (PCA)"
)

if model_result is None:

    st.error(
        "Dataset gagal dibaca atau kosong."
    )

    st.stop()

pca, knn, acc = model_result

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Home",
        "Prediksi Usia",
        "Face Similarity"
    ]
)

# =====================================================
# HOME
# =====================================================

if menu == "Home":

    X, y = load_dataset(DATASET_PATH)

    st.subheader("Informasi Dataset")

    st.write(
        "Total Gambar:",
        len(X)
    )

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for u, c in zip(unique, counts):
        st.write(f"{u}: {c}")

    st.subheader("Akurasi")

    st.success(
        f"{acc*100:.2f}%"
    )

# =====================================================
# PREDIKSI USIA
# =====================================================

elif menu == "Prediksi Usia":

    uploaded = st.file_uploader(
        "Upload Foto",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(uploaded)

        st.image(
            image,
            width=250
        )

        img = np.array(image)

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

        gray = cv2.resize(
            gray,
            IMG_SIZE
        )

        gray = gray / 255.0

        vector = gray.flatten()

        vector = vector.reshape(
            1,
            -1
        )

        vector_pca = pca.transform(
            vector
        )

        prediction = knn.predict(
            vector_pca
        )[0]

        st.success(
            f"Prediksi: {prediction}"
        )

# =====================================================
# FACE SIMILARITY
# =====================================================

elif menu == "Face Similarity":

    file1 = st.file_uploader(
        "Upload Foto Pertama",
        type=["jpg", "jpeg", "png"],
        key="1"
    )

    file2 = st.file_uploader(
        "Upload Foto Kedua",
        type=["jpg", "jpeg", "png"],
        key="2"
    )

    if file1 and file2:

        img1 = np.array(
            Image.open(file1)
        )

        img2 = np.array(
            Image.open(file2)
        )

        gray1 = cv2.cvtColor(
            img1,
            cv2.COLOR_RGB2GRAY
        )

        gray2 = cv2.cvtColor(
            img2,
            cv2.COLOR_RGB2GRAY
        )

        gray1 = cv2.resize(
            gray1,
            IMG_SIZE
        )

        gray2 = cv2.resize(
            gray2,
            IMG_SIZE
        )

        vec1 = gray1.flatten() / 255.0
        vec2 = gray2.flatten() / 255.0

        vec1_pca = pca.transform(
            vec1.reshape(1, -1)
        )

        vec2_pca = pca.transform(
            vec2.reshape(1, -1)
        )

        cosine = cosine_similarity(
            vec1_pca,
            vec2_pca
        )[0][0]

        distance = euclidean(
            vec1_pca[0],
            vec2_pca[0]
        )

        st.write(
            f"Cosine Similarity: {cosine:.4f}"
        )

        st.write(
            f"Euclidean Distance: {distance:.4f}"
        )

        if cosine > 0.8:
            st.success("Mirip")
        else:
            st.error("Tidak Mirip")
