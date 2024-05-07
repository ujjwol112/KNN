# kNN Classification

This repository contains Python scripts for implementing k-nearest neighbors (kNN) classification algorithms on two different datasets: **"harth_S028.csv"** and **"voice.csv"**. Each dataset represents a distinct classification task with unique features and target variables.

## Introduction

The kNN algorithm is a simple yet powerful machine learning method used for classification and regression tasks. It relies on the idea that data points with similar features tend to belong to the same class. This repository explores the application of kNN classification on human activity recognition data and voice data for gender prediction.

## Datasets

### 1. harth_S028.csv
- **Description**: This dataset contains sensor data collected from a wearable device worn by individuals engaged in various physical activities.
- **Features**: The dataset includes multiple sensor readings (e.g., accelerometer, gyroscope) along with a timestamp.
- **Target Variable**: The target variable represents the activity label, indicating the type of physical activity being performed (e.g., walking, running, sitting).

The HARTH dataset contains recordings of 22 participants wearing two 3-axial Axivity AX3 accelerometers for around 2 hours in a free-living setting. One sensor was attached to the right front thigh and the other to the lower back. The provided sampling rate is 50Hz. Video recordings of a chest-mounted camera were used to annotate the performed activities frame-by-frame.

Each subject's recordings are provided in a separate .csv file. One such .csv file contains the following columns:

- timestamp: date and time of recorded sample
- back_x: acceleration of back sensor in x-direction (down) in the unit g
- back_y: acceleration of back sensor in y-direction (left) in the unit g
- back_z: acceleration of back sensor in z-direction (forward) in the unit g
- thigh_x: acceleration of thigh sensor in x-direction (down) in the unit g
- thigh_y: acceleration of thigh sensor in y-direction (right) in the unit g
- thigh_z: acceleration of thigh sensor in z-direction (backward) in the unit g
- label: annotated activity code
 
The dataset contains the following annotated activities with the corresponding coding:
 1: walking
 2: running
 3: shuffling
 4: stairs (ascending)
 5: stairs (descending)
 6: standing
 7: sitting
 8: lying
 13: cycling (sit)
 14: cycling (stand)
 130: cycling (sit, inactive)
 140: cycling (stand, inactive)

### 2. voice.csv
- **Description**: This dataset consists of acoustic features extracted from voice recordings, aiming to predict the gender of the speaker based on voice characteristics.
- **Features**: The features include various acoustic properties such as mean fundamental frequency, duration of speech signals, and spectral properties.
- **Target Variable**: The target variable represents the gender of the speaker (male or female).

This dataset is based on speech and voice acoustics, this database was developed to classify voices as male or female. 3,168 recorded speech samples from male and female speakers make up the collection. Using the seewave and tuneR packages in R, acoustic analysis is used to preprocess the speech samples.

0 Hz to 280 Hz in frequency range was examined (human vocal range).

Column Description:
- meanfreq: mean frequency (in kHz)
- sd: standard deviation of frequency
- median: median frequency (in kHz)
- Q25: first quantile (in kHz)
- Q75: third quantile (in kHz)
- IQR: interquantile range (in kHz)
- skew: skewness (see note in specprop description)
- kurt: kurtosis (see note in specprop description)
- sp.ent: spectral entropy
- sfm: spectral flatness
- mode: mode frequency
- centroid: frequency centroid (see specprop)
- peakf: peak frequency (frequency with highest energy)
- meanfun: average of fundamental frequency measured across acoustic signal
- minfun: minimum fundamental frequency measured across acoustic signal
- maxfun: maximum fundamental frequency measured across acoustic signal
- meandom: average of dominant frequency measured across acoustic signal
- mindom: minimum of dominant frequency measured across acoustic signal
- maxdom: maximum of dominant frequency measured across acoustic signal
- dfrange: range of dominant frequency measured across acoustic signal
- modindx: modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
- label: male or female

## Code Overview

### 1. harth_S028.csv Analysis

- **Description**: This script performs kNN classification on the harth_S028 dataset after preprocessing steps including feature standardization, correlation analysis, and customization of distance weights. It evaluates the model using confusion matrices, classification reports, and accuracy scores. Additionally, it explores feature-target correlations and optimizes the model using k-fold cross-validation.

**Dataset: HAR (Human Activity Recognition)**
- The "harth_S028.csv" dataset contains features extracted from smartphone sensors for recognizing different human activities.
- Features include accelerometer and gyroscope readings from various smartphone sensors.
- Target labels represent different human activities such as walking, running, shuffling, etc.

**Description:**
- This code implements a K-Nearest Neighbors (KNN) classifier to predict human activities based on sensor data.
- It preprocesses the dataset, scales the features using StandardScaler, and splits the data into training and testing sets.
- The classifier is trained using default hyperparameters and evaluated using confusion matrix, classification report, and accuracy score.
- Feature correlation and importance analysis are performed to identify significant features for activity recognition.
- Weighted KNN and feature selection techniques are applied to improve model performance.
- Hyperparameter tuning using k-fold cross-validation helps find the optimal value of k for the KNN classifier.

**Results and Insights:**
- The baseline KNN model achieves a certain level of accuracy in recognizing human activities.
- Feature correlation and importance analysis provide insights into the significance of sensor readings for activity recognition.
- Weighted KNN and feature selection techniques may further enhance model performance by focusing on relevant features.
- Hyperparameter tuning improves the model's accuracy by finding the optimal number of neighbors for classification.
- Using cosine distance as the distance metric may provide better results in certain cases.
- Mutual Information scores reveal the importance of features in classifying human activities.
- Correlation analysis provides insights into the relationship between features and the target variable.

### 2. voice.csv Analysis

- **Description**: This script applies kNN classification on the voice dataset for gender prediction. Similar to the previous script, it conducts preprocessing steps, evaluates the model performance, and explores feature correlations. It also identifies the optimal value of k through k-fold cross-validation and visualizes the results using confusion matrices.

**Dataset: Voice Dataset**
- The "voice.csv" dataset contains acoustic features extracted from voice recordings for gender recognition.
- Features include mean frequency, standard deviation of frequency, and other acoustic properties.
- The target variable represents the gender of the speaker (male or female).

**Description:**
- This code implements a KNN classifier to predict gender based on voice features.
- The dataset is preprocessed by replacing gender labels with binary values and standardizing the features.
- It splits the data into training and testing sets and trains a KNN classifier with default hyperparameters.
- The classifier's performance is evaluated using confusion matrix, classification report, and accuracy score.
- Feature correlation analysis helps identify important features for gender recognition.
- Weighted KNN and feature selection techniques are applied to improve model performance.
- Hyperparameter tuning using k-fold cross-validation helps find the optimal value of k for the KNN classifier.

**Results and Insights:**
- Gender Recognition by Voice using K-Nearest Neighbors (kNN)
- The KNN classifier achieves a certain level of accuracy in predicting gender based on voice features.
- Feature correlation analysis provides insights into the acoustic properties most correlated with gender.
- Weighted KNN and feature selection techniques may enhance model performance by focusing on relevant acoustic features.
- Hyperparameter tuning improves the model's accuracy by finding the optimal number of neighbors for classification.
- Selecting top correlated features and tuning hyperparameters can further improve model performance.
- The optimal value of k is determined using k-fold cross-validation, resulting in improved accuracy.

## Results and Insights

- Both scripts provide detailed analyses of the datasets and model performance.
- Key metrics such as accuracy, precision, recall, and F1-score are reported for model evaluation.
- Correlation matrices and feature-target correlations offer insights into the relevance of features for classification tasks.
- Optimization techniques such as customizing distance weights and selecting the optimal value of k enhance the performance of the kNN classifiers.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/ujjwol112/kNN.git
```

2. Navigate to the directory containing the desired analysis script:

```bash
cd kNN
```
## Dependencies

- Python 3.x
- pandas
- scikit-learn
- seaborn
- matplotlib
- numpy
