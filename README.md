# Normal-Abnormal Image Classification Using Brain MRI

## Overview

This project implements a binary image classification system that classifies brain MRI images into two categories: **Normal** and **Abnormal**.

The system uses traditional computer vision techniques for image preprocessing and feature extraction, followed by a Support Vector Machine (SVM) classifier to distinguish MRI images with no tumor from those containing a brain tumor.

The project combines Gaussian Blur, Canny Edge Detection, Histogram of Oriented Gradients (HOG), and an RBF-kernel SVM to build an image classification pipeline.

**Note:** This is an educational computer vision project and is not intended for clinical diagnosis or medical decision-making.

## Objectives

* Develop a computer vision pipeline for brain MRI image classification.
* Apply image preprocessing to improve feature extraction.
* Extract visual features using Histogram of Oriented Gradients.
* Train an SVM classifier using an RBF kernel.
* Evaluate classification performance using standard metrics.
* Visualize the classification pipeline and confusion matrix.

## Dataset

The project uses a Brain Tumor MRI Dataset containing four original categories:

* Glioma
* Meningioma
* Pituitary tumor
* No tumor

For binary classification, the original categories are mapped as follows:

| Original Category    | Binary Label |
| -------------------- | ------------ |
| No Tumor (`notumor`) | Normal       |
| Glioma               | Abnormal     |
| Meningioma           | Abnormal     |
| Pituitary Tumor      | Abnormal     |

### Dataset Distribution

| Dataset Split | Number of Images |
| ------------- | ---------------: |
| Training      |            5,712 |
| Testing       |            1,311 |
| Total         |            7,023 |

The dataset is organized into `Training` and `Testing` directories, with separate subdirectories for each original category.

## Methodology

The classification pipeline consists of the following stages.

### 1. Image Preprocessing

* Convert images to grayscale.
* Resize images to 128 × 128 pixels.
* Apply Gaussian Blur to reduce noise.

### 2. Edge Detection

Canny Edge Detection is applied to identify significant image boundaries and structural patterns.

### 3. Feature Extraction

Histogram of Oriented Gradients (HOG) extracts features representing local intensity gradients and edge orientations.

The resulting feature representation contains 8,100 features per image.

### 4. Classification

A Support Vector Machine (SVM) with an RBF kernel is used to classify the extracted features into Normal and Abnormal categories.

### 5. Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Sensitivity (Recall)
* Specificity
* F1-score
* Confusion Matrix

## Results

The following results were obtained during the reported evaluation of the implemented pipeline.

| Evaluation Metric    |  Score |
| -------------------- | -----: |
| Accuracy             | 97.25% |
| Precision            | 98.55% |
| Sensitivity (Recall) | 97.46% |
| Specificity          | 96.79% |
| F1-score             | 98.00% |

### Confusion Matrix

| Actual / Predicted | Normal | Abnormal |
| ------------------ | -----: | -------: |
| Normal             |    392 |       13 |
| Abnormal           |     23 |      883 |

The results indicate that the implemented classifier performed well on the evaluated dataset. However, performance on this dataset does not establish clinical reliability or generalization to other hospitals, scanners, or patient populations.

## Technologies Used

* Python
* OpenCV
* NumPy
* scikit-image
* scikit-learn
* Matplotlib

## Project Structure

```text
Normal-Abnormal-Image-Classification/
│
├── main.py
├── requirements.txt
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── pipeline_output.png
├── confusion_matrix.png
└── README.md
```

The directory structure above describes the dataset organization and project artifacts. Actual folder names and included files may differ depending on how the repository is packaged.

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/abhaycosmicstorm/Normal-Abnormal-Image-Classification.git
```

### 2. Navigate to the Project Directory

```bash
cd Normal-Abnormal-Image-Classification
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python main.py
```

Ensure that the dataset is available in the directory structure expected by the script before running it.

## Future Improvements

* Compare HOG features with alternative feature extraction techniques.
* Evaluate additional classifiers, including Random Forest.
* Investigate class imbalance and threshold selection.
* Add a user interface for uploading MRI images.
* Test generalization on independent datasets.
* Investigate validation strategies that reduce the risk of data leakage.

## Author

**Abhay Raj**

GitHub: https://github.com/abhaycosmicstorm

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical device and must not be used to diagnose, exclude, or make treatment decisions about brain tumors.
