"""
Normal vs Abnormal Brain MRI Classification

Pipeline:
1. Load Brain MRI dataset
2. Normal = no tumor
3. Abnormal = glioma + meningioma + pituitary
4. Preprocessing: grayscale + resizing + Gaussian blur
5. Edge detection: Canny
6. Feature extraction: HOG
7. SVM classifier
8. Accuracy, Precision, Sensitivity, Specificity, F1-score
9. Confusion matrix
10. Sample pipeline visualization
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# SETTINGS
# ============================================================

DATASET = "dataset"

TRAINING_FOLDER = os.path.join(DATASET, "Training")
TESTING_FOLDER = os.path.join(DATASET, "Testing")

IMG_SIZE = (128, 128)


# ============================================================
# CLASS LABELS
# ============================================================

# Normal = 0
# Abnormal = 1

class_labels = {
    "notumor": 0,
    "glioma": 1,
    "meningioma": 1,
    "pituitary": 1
}


# ============================================================
# FUNCTION TO EXTRACT FEATURES
# ============================================================

def extract_features(image_path):

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None, None, None, None

    # Resize image
    img = cv2.resize(img, IMG_SIZE)

    # --------------------------------------------------------
    # 1. Gaussian Blur
    # --------------------------------------------------------

    preprocessed = cv2.GaussianBlur(
        img,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # 2. Canny Edge Detection
    # --------------------------------------------------------

    edges = cv2.Canny(
        preprocessed,
        50,
        150
    )

    # --------------------------------------------------------
    # 3. HOG Feature Extraction
    # --------------------------------------------------------

    features = hog(
        edges,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features, img, preprocessed, edges


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(folder):

    X = []
    y = []

    display_images = []

    class_counts = {
        "notumor": 0,
        "glioma": 0,
        "meningioma": 0,
        "pituitary": 0
    }

    # Go through every class folder
    for class_name in class_labels:

        class_folder = os.path.join(
            folder,
            class_name
        )

        if not os.path.exists(class_folder):

            print("Folder not found:", class_folder)
            continue

        # Read every image
        for filename in os.listdir(class_folder):

            if not filename.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                continue

            image_path = os.path.join(
                class_folder,
                filename
            )

            features, img, preprocessed, edges = extract_features(
                image_path
            )

            if features is None:
                continue

            # Store features
            X.append(features)

            # Store binary label
            label = class_labels[class_name]

            y.append(label)

            class_counts[class_name] += 1

            # Store a few examples for visualization
            if len(display_images) < 4:

                display_images.append(
                    (
                        img,
                        preprocessed,
                        edges,
                        label
                    )
                )

    return (
        np.array(X),
        np.array(y),
        display_images,
        class_counts
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\n========================================")
print("LOADING TRAINING DATA")
print("========================================")

X_train, y_train, _, train_counts = load_dataset(
    TRAINING_FOLDER
)

print("\nTraining dataset:")

for class_name, count in train_counts.items():

    if class_labels[class_name] == 0:
        print(f"{class_name:12} -> Normal   : {count}")

    else:
        print(f"{class_name:12} -> Abnormal : {count}")


print("\nTotal training images:", len(X_train))

print("Feature vector size:", X_train.shape[1])


# ============================================================
# LOAD TESTING DATA
# ============================================================

print("\n========================================")
print("LOADING TESTING DATA")
print("========================================")

X_test, y_test, display_images, test_counts = load_dataset(
    TESTING_FOLDER
)

print("\nTesting dataset:")

for class_name, count in test_counts.items():

    if class_labels[class_name] == 0:
        print(f"{class_name:12} -> Normal   : {count}")

    else:
        print(f"{class_name:12} -> Abnormal : {count}")


print("\nTotal testing images:", len(X_test))


# ============================================================
# TRAIN SVM
# ============================================================

print("\n========================================")
print("TRAINING SVM")
print("========================================")

model = SVC(
    kernel="rbf",
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("SVM training completed.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

sensitivity = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# Specificity
specificity = tn / (tn + fp) if (tn + fp) else 0


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("PERFORMANCE RESULTS")
print("========================================")

print(
    f"Accuracy    : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision   : {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Sensitivity : {sensitivity:.4f} "
    f"({sensitivity * 100:.2f}%)"
)

print(
    f"Specificity : {specificity:.4f} "
    f"({specificity * 100:.2f}%)"
)

print(
    f"F1 Score    : {f1:.4f} "
    f"({f1 * 100:.2f}%)"
)


print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Abnormal"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(5, 4)
)

plt.imshow(
    cm,
    cmap="Blues"
)

plt.title(
    "Normal vs Abnormal - Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    [0, 1],
    ["Normal", "Abnormal"]
)

plt.yticks(
    [0, 1],
    ["Normal", "Abnormal"]
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14
        )


plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=150
)

plt.close()

# ============================================================
# SELECT 2 NORMAL + 2 ABNORMAL IMAGES
# ============================================================

print("\n========================================")
print("SELECTING SAMPLE IMAGES")
print("========================================")

sample_images = []

# ------------------------------------------------------------
# 2 NORMAL IMAGES
# ------------------------------------------------------------

normal_folder = os.path.join(
    TRAINING_FOLDER,
    "notumor"
)

normal_count = 0

for filename in os.listdir(normal_folder):

    if not filename.lower().endswith(
        (".png", ".jpg", ".jpeg")
    ):
        continue

    image_path = os.path.join(
        normal_folder,
        filename
    )

    features, img, preprocessed, edges = extract_features(
        image_path
    )

    if features is None:
        continue

    prediction = model.predict(
        features.reshape(1, -1)
    )[0]

    sample_images.append(
        (
            img,
            preprocessed,
            edges,
            features,
            0,
            prediction,
            "Normal"
        )
    )

    normal_count += 1

    if normal_count == 2:
        break


# ------------------------------------------------------------
# 2 ABNORMAL IMAGES
# ------------------------------------------------------------

abnormal_classes = [
    "glioma",
    "meningioma"
]

for class_name in abnormal_classes:

    abnormal_folder = os.path.join(
        TRAINING_FOLDER,
        class_name
    )

    for filename in os.listdir(abnormal_folder):

        if not filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            continue

        image_path = os.path.join(
            abnormal_folder,
            filename
        )

        features, img, preprocessed, edges = extract_features(
            image_path
        )

        if features is None:
            continue

        prediction = model.predict(
            features.reshape(1, -1)
        )[0]

        sample_images.append(
            (
                img,
                preprocessed,
                edges,
                features,
                1,
                prediction,
                class_name
            )
        )

        break


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("SAMPLE IMAGE PREDICTIONS")
print("========================================")

for i, sample in enumerate(sample_images):

    actual = sample[4]
    prediction = sample[5]
    class_name = sample[6]

    actual_name = (
        "Normal"
        if actual == 0
        else "Abnormal"
    )

    predicted_name = (
        "Normal"
        if prediction == 0
        else "Abnormal"
    )

    print(
        f"Image {i + 1}: "
        f"Actual = {actual_name}, "
        f"Predicted = {predicted_name}, "
        f"Source = {class_name}"
    )


# ============================================================
# CREATE VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    4,
    4,
    figsize=(14, 11)
)


for col, sample in enumerate(sample_images):

    img = sample[0]
    preprocessed = sample[1]
    edges = sample[2]
    actual = sample[4]
    prediction = sample[5]
    class_name = sample[6]

    actual_name = (
        "Normal"
        if actual == 0
        else "Abnormal"
    )

    predicted_name = (
        "Normal"
        if prediction == 0
        else "Abnormal"
    )


    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    axes[0, col].imshow(
        img,
        cmap="gray"
    )

    axes[0, col].set_title(
        f"Original\n"
        f"Actual: {actual_name}\n"
        f"Class: {class_name}"
    )


    # --------------------------------------------------------
    # GAUSSIAN BLUR
    # --------------------------------------------------------

    axes[1, col].imshow(
        preprocessed,
        cmap="gray"
    )

    axes[1, col].set_title(
        "Gaussian Blur"
    )


    # --------------------------------------------------------
    # CANNY
    # --------------------------------------------------------

    axes[2, col].imshow(
        edges,
        cmap="gray"
    )

    axes[2, col].set_title(
        "Canny Edges"
    )


    # --------------------------------------------------------
    # HOG
    # --------------------------------------------------------

    _, hog_image = hog(
        edges,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True
    )

    axes[3, col].imshow(
        hog_image,
        cmap="gray"
    )

    axes[3, col].set_title(
        f"HOG Features\n"
        f"Predicted: {predicted_name}"
    )


# Remove axes

for ax in axes.ravel():

    ax.axis("off")


plt.tight_layout()


# Save output

output_file = os.path.abspath(
    "sample_predictions.png"
)

plt.savefig(
    output_file,
    dpi=150,
    bbox_inches="tight"
)


print("\nSample visualization saved at:")

print(output_file)


plt.show()


