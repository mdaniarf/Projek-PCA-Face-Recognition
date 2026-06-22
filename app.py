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

IMG_SIZE = (100,100)
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
# EXTRACT AGE
# =====================================================

def get_age(filename):

    match = re.search(r"A(\d+)", filename)

    if match:
        return int(match.group(1))

    return None

# =====================================================
# LOAD IMAGE
# =====================================================

def preprocess_image(path):

    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, IMG_SIZE)

    gray = gray / 255.0

    return gray.flatten()

# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset(dataset_path):

    X = []
    y = []

    for person in os.listdir(dataset_path):

        folder = os.path.join(dataset_path, person)

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):

            if file.endswith((".jpg",".jpeg",".png")):

                age = get_age(file)

                if age is None:
                    continue

                group = age_group(age)

                path = os.path.join(folder,file)

                vector = preprocess_image(path)

                X.append(vector)

                y.append(group)

    return np.array(X), np.array(y)

# =====================================================
# TRAIN MODEL
# =====================================================

@st.cache_resource
def train_model():

    X,y = load_dataset(DATASET_PATH)

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pca = PCA(n_components=50)

    X_train_pca = pca.fit_transform(X_train)

    X_test_pca = pca.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train_pca,y_train)

    pred = knn.predict(X_test_pca)

    acc = accuracy_score(y_test,pred)

    return pca,knn,acc,X_train_pca,y_train

# =====================================================
# LOAD MODEL
# =====================================================

pca,knn,acc,X_train_pca,y_train = train_model()

# =====================================================
# UI
# =====================================================

st.title("Face Recognition Based On Age")

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

    st.subheader("Informasi Model")

    st.write(f"Akurasi : {acc*100:.2f}%")

# =====================================================
# PREDIKSI USIA
# =====================================================

elif menu == "Prediksi Usia":

    uploaded = st.file_uploader(
        "Upload Foto",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        image = Image.open(uploaded)

        st.image(image,width=250)

        img = np.array(image)

        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        gray = cv2.resize(gray,IMG_SIZE)

        gray = gray/255.0

        vector = gray.flatten()

        vector = vector.reshape(1,-1)

        vector_pca = pca.transform(vector)

        prediction = knn.predict(vector_pca)[0]

        st.success(
            f"Prediksi Usia : {prediction}"
        )

# =====================================================
# FACE SIMILARITY
# =====================================================

elif menu == "Face Similarity":

    col1,col2 = st.columns(2)

    with col1:

        file1 = st.file_uploader(
            "Foto 1",
            type=["jpg","jpeg","png"]
        )

    with col2:

        file2 = st.file_uploader(
            "Foto 2",
            type=["jpg","jpeg","png"]
        )

    if file1 and file2:

        img1 = np.array(Image.open(file1))
        img2 = np.array(Image.open(file2))

        gray1 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)

        gray1 = cv2.resize(gray1,IMG_SIZE)
        gray2 = cv2.resize(gray2,IMG_SIZE)

        vec1 = gray1.flatten()/255.0
        vec2 = gray2.flatten()/255.0

        vec1 = pca.transform(vec1.reshape(1,-1))
        vec2 = pca.transform(vec2.reshape(1,-1))

        cosine = cosine_similarity(
            vec1,
            vec2
        )[0][0]

        distance = euclidean(
            vec1[0],
            vec2[0]
        )

        st.subheader("Hasil Similarity")

        st.write(
            f"Cosine Similarity : {cosine:.4f}"
        )

        st.write(
            f"Euclidean Distance : {distance:.4f}"
        )

        if cosine > 0.80:

            st.success("Wajah Mirip")

        else:

            st.error("Wajah Tidak Mirip")