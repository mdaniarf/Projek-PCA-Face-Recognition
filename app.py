import streamlit as st
import cv2
import numpy as np
import os
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

st.set_page_config(
    page_title="Age-Invariant Face Recognition",
    layout="wide"
)

DATASET_PATH = "dataset"
IMG_SIZE = (128, 128)

# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess_image(path):

    img = cv2.imread(path)

    if img is None:
        return None

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    gray = cv2.resize(
        gray,
        IMG_SIZE
    )

    gray = gray / 255.0

    return gray.flatten()

# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset():

    X = []
    y = []

    if not os.path.exists(DATASET_PATH):
        return np.array([]), np.array([])

    persons = sorted(
        os.listdir(DATASET_PATH)
    )

    for person in persons:

        folder_path = os.path.join(
            DATASET_PATH,
            person
        )

        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):

            if not file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
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

            # label = identitas orang
            y.append(person)

    return np.array(X), np.array(y)

# =====================================================
# TRAIN MODEL
# =====================================================

@st.cache_resource
def train_model():

    X, y = load_dataset()

    if len(X) == 0:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    n_components = min(
        80,
        X_train.shape[0] - 1
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
        n_neighbors=1
    )

    knn.fit(
        X_train_pca,
        y_train
    )

    prediction = knn.predict(
        X_test_pca
    )

    acc = accuracy_score(
        y_test,
        prediction
    )

    return (
        pca,
        knn,
        acc,
        X,
        y
    )

# =====================================================
# LOAD MODEL
# =====================================================

model = train_model()

if model is None:

    st.error(
        "Dataset tidak ditemukan atau kosong"
    )

    st.stop()

pca, knn, acc, X_all, y_all = model

# =====================================================
# SIDEBAR
# =====================================================

menu = st.sidebar.radio(
    "Menu",
    [
        "Home",
        "EDA",
        "Recognize Person",
        "Face Similarity"
    ]
)

# =====================================================
# HOME
# =====================================================

if menu == "Home":

    st.title(
        "Age-Invariant Face Recognition Using PCA"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Images",
            len(X_all)
        )

    with col2:

        st.metric(
            "Accuracy",
            f"{acc*100:.2f}%"
        )

# =====================================================
# EDA
# =====================================================

elif menu == "EDA":

    st.title("Exploratory Data Analysis")

    unique, counts = np.unique(
        y_all,
        return_counts=True
    )

    st.subheader(
        "Jumlah Foto per Individu"
    )

    for person, count in zip(
        unique,
        counts
    ):

        st.write(
            f"Individu {person} : {count} foto"
        )

# =====================================================
# RECOGNIZE PERSON
# =====================================================

elif menu == "Recognize Person":

    st.title(
        "Recognize Person"
    )

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(uploaded)

        st.image(
            image,
            width=300
        )

        img = np.array(image)

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

        gray = cv2.equalizeHist(
            gray
        )

        gray = cv2.resize(
            gray,
            IMG_SIZE
        )

        gray = gray / 255.0

        vector = gray.flatten()

        vector_pca = pca.transform(
            vector.reshape(1, -1)
        )

        prediction = knn.predict(
            vector_pca
        )[0]

        distance, _ = knn.kneighbors(
            vector_pca,
            n_neighbors=1
        )

        confidence = max(
            0,
            100 - distance[0][0] * 10
        )

        st.success(
            f"Identitas Terdeteksi : {prediction}"
        )

        st.info(
            f"Confidence : {confidence:.2f}%"
        )

# =====================================================
# FACE SIMILARITY
# =====================================================

elif menu == "Face Similarity":

    st.title(
        "Face Similarity"
    )

    col1, col2 = st.columns(2)

    with col1:

        file1 = st.file_uploader(
            "Foto Pertama",
            type=["jpg", "jpeg", "png"],
            key="file1"
        )

    with col2:

        file2 = st.file_uploader(
            "Foto Kedua",
            type=["jpg", "jpeg", "png"],
            key="file2"
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

        gray1 = cv2.equalizeHist(
            gray1
        )

        gray2 = cv2.equalizeHist(
            gray2
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

        st.metric(
            "Cosine Similarity",
            f"{cosine:.4f}"
        )

        st.metric(
            "Euclidean Distance",
            f"{distance:.4f}"
        )

        if cosine > 0.80:

            st.success(
                "Kedua wajah kemungkinan orang yang sama"
            )

        else:

            st.error(
                "Kedua wajah kemungkinan berbeda"
            )
