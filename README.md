# IMDB_customer_churn_prediction

> **Added comprehensive documentation for the IMDB customer churn prediction project, including project overview, dataset information, objectives, technology stack, workflow, model performance, and future enhancements.**



## Project Overview

Customer churn prediction is a critical business analytics problem in **telecommunications, banking, insurance, SaaS, and subscription-based platforms**. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project demonstrates an **industry-oriented implementation of Machine Learning and Deep Learning techniques** to identify customers who are likely to discontinue a service.

This repository presents a **complete end-to-end churn prediction pipeline**, including:

* Data preprocessing
* Feature engineering
* Multiple Machine Learning algorithms
* Deep Neural Network implementation using **TensorFlow**
* Performance comparison using standard evaluation metrics
* Visualization and business interpretation of results

The implementation is designed in a **professional, production-style workflow** suitable for **B.Tech, MCA, Data Analytics, and AI/ML portfolio projects**.



# Dataset Information

| Attribute             | Value                 |
| --------------------- | --------------------- |
| **Dataset**           | Telco Customer Churn  |
| **Domain**            | Telecommunications    |
| **Problem Type**      | Binary Classification |
| **Target Variable**   | Churn                 |
| **Total Records**     | **7043**              |
| **Features**          | **33**                |
| **Duplicate Records** | **0**                 |

### Target Classes

* **Yes** → Customer left the service
* **No** → Customer retained the service


# Project Objectives

The primary objectives of this project are:

* Predict customer churn using historical customer data.
* Compare **Machine Learning vs Deep Learning performance**.
* Analyze the trade-off between **accuracy, recall, ROC-AUC, and computational cost**.
* Build a reusable TensorFlow-based binary classification workflow.
* Demonstrate practical expertise in **ML, DL, preprocessing, evaluation, and model comparison**.


# Technology Stack

## Programming Language

* **Python 3.x**

## Libraries Used

### Data Processing

* `pandas`
* `numpy`

### Visualization

* `matplotlib`
* `seaborn`

### Machine Learning

* `scikit-learn`

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Support Vector Machine

### Deep Learning

* `tensorflow`
* `keras`


# Project Structure

```text
IMDB_customer_churn_prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   └── Practical_03_Customer_Churn_Comparison.ipynb
│
├── reports/
│   └── model_comparison_results.csv
│
├── images/
│   ├── accuracy_curve.png
│   └── confusion_matrix.png
│
├── requirements.txt
└── README.md
```


# Workflow

## 1. Data Preprocessing

The following preprocessing steps were performed:

* Handling missing values
* Removing duplicate records
* Encoding categorical variables using **Label Encoding**
* Converting `TotalCharges` to numeric format
* Feature scaling using **StandardScaler**
* Splitting data into **training (80%)** and **testing (20%)** sets



# Machine Learning Models

The following supervised learning algorithms were implemented:

| Model                      | Purpose                                       |
| -------------------------- | --------------------------------------------- |
| **Logistic Regression**    | Baseline linear classifier                    |
| **Decision Tree**          | Rule-based interpretable model                |
| **Random Forest**          | Ensemble learning for improved generalization |
| **Support Vector Machine** | Non-linear margin-based classifier            |



# Deep Learning Model

A **Deep Neural Network (DNN)** was developed using **TensorFlow/Keras**.

## Network Architecture

```text
Input Layer (20 Features)
        ↓
Dense Layer (64 neurons, ReLU)
        ↓
Dropout Layer (0.3)
        ↓
Dense Layer (32 neurons, ReLU)
        ↓
Output Layer (1 neuron, Sigmoid)
```

## Model Summary

```text
Layer (type)                Output Shape      Param #
====================================================
Dense (64)                  (None, 64)          1344
Dropout (0.3)               (None, 64)             0
Dense (32)                  (None, 32)          2080
Dense (1)                   (None, 1)             33
====================================================
Total Parameters: 3,457
```

This architecture is lightweight yet effective for **binary customer churn prediction tasks**.


# Training Performance

The model was trained for **20 epochs** using the **Adam optimizer** and **binary crossentropy loss**.

## Accuracy Curve

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/65233265-2be4-49c1-87b9-4d64d8752ff4" />


### Observation

* Training accuracy improved from **71.3% → 80.8%**
* Validation accuracy remained stable around **81–82%**
* The gap between training and validation accuracy is small, indicating **minimal overfitting**



# Confusion Matrix

<img width="513" height="470" alt="image" src="https://github.com/user-attachments/assets/22df44e5-f097-4405-a9df-80d3a7a11dba" />

## Interpretation

| Metric              | Value |
| ------------------- | ----- |
| **True Negatives**  | 905   |
| **False Positives** | 107   |
| **False Negatives** | 172   |
| **True Positives**  | 223   |

The model performs well in identifying **non-churn customers**, while churn detection can be further improved by techniques such as:

* Class balancing
* SMOTE oversampling
* Threshold tuning
* Cost-sensitive learning



# Model Performance Comparison

| Model                      | Accuracy   | Precision  | Recall     | F1-Score   | ROC-AUC    |
| -------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **Decision Tree**          | **0.8095** | 0.7471     | 0.4861     | 0.5890     | 0.8494     |
| **Logistic Regression**    | 0.8074     | 0.6879     | **0.5747** | **0.6262** | **0.8615** |
| **Random Forest**          | 0.8074     | 0.7039     | 0.5418     | 0.6123     | 0.8491     |
| **Support Vector Machine** | 0.8017     | **0.7479** | 0.4430     | 0.5564     | 0.8265     |
| **Deep Neural Network**    | 0.8017     | 0.6758     | 0.5646     | 0.6152     | 0.8570     |



# Computational Performance

| Model                  | Training Time (s) | Prediction Time (s) |
| ---------------------- | ----------------- | ------------------- |
| Decision Tree          | **0.0220**        | **0.0020**          |
| Logistic Regression    | 0.0958            | 0.0028              |
| Random Forest          | 0.7606            | 0.0230              |
| Support Vector Machine | 4.3020            | 0.2059              |
| Deep Neural Network    | 10.5891           | 0.3220              |



# Key Findings

## Best Accuracy

**Decision Tree** achieved the highest accuracy:

```text
Accuracy = 80.95%
```

## Best Recall

**Logistic Regression** produced the highest recall:

```text
Recall = 57.47%
```

This is important because **recall is often the most critical metric in churn prediction**, as failing to identify a potential churn customer can result in revenue loss.

## Best ROC-AUC

**Logistic Regression** achieved the best overall discrimination capability:

```text
ROC-AUC = 86.15%
```

## Deep Learning Analysis

The Deep Neural Network delivered **competitive performance** with:

* **Accuracy:** 80.17%
* **F1-Score:** 61.52%
* **ROC-AUC:** 85.70%

Although the DNN did not outperform the best ML model on this structured dataset, it demonstrated **strong generalization ability** and provides a scalable foundation for larger datasets and feature-rich environments.



# Business Insights

From the experimental analysis:

* Customers with **shorter tenure** are more likely to churn.
* **Higher monthly charges** show a stronger association with churn behavior.
* Traditional ML models remain highly effective for **tabular business datasets**.
* Deep Learning becomes more beneficial when:

  * dataset size is significantly larger,
  * feature interactions are highly complex,
  * unstructured data (text, images, logs) is involved.



# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/IMDB_customer_churn_prediction.git

cd IMDB_customer_churn_prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```


# Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/Practical_03_Customer_Churn_Comparison.ipynb
```

Run all cells sequentially.


# Sample Output

After execution, the project generates:

* Preprocessed dataset
* Classification reports
* Confusion matrix
* Accuracy comparison graph
* `model_comparison_results.csv`



# Skills Demonstrated

This project demonstrates practical expertise in:

## Machine Learning

* Supervised Learning
* Classification Algorithms
* Ensemble Methods
* Model Evaluation
* ROC-AUC Analysis

## Deep Learning

* TensorFlow/Keras
* Neural Network Architecture Design
* Dropout Regularization
* Binary Classification
* Training & Validation Monitoring

## Data Analytics

* Data Cleaning
* Feature Encoding
* Feature Scaling
* Exploratory Data Analysis
* Business Interpretation of Predictive Models


# Future Enhancements

Potential improvements include:

* **SMOTE** for class imbalance handling
* **Hyperparameter tuning** using GridSearchCV
* **XGBoost / LightGBM / CatBoost**
* **EarlyStopping** and **ModelCheckpoint**
* Deployment using **Flask** or **FastAPI**
* Real-time churn prediction dashboard using **Power BI** or **Streamlit**


# Conclusion

This project presents a **comprehensive comparative study of Machine Learning and Deep Learning approaches for customer churn prediction**.

### Final Conclusion

* **Decision Tree** achieved the **highest accuracy**.
* **Logistic Regression** provided the **best recall and ROC-AUC**, making it the most suitable model for **customer retention strategies**.
* **Deep Neural Network** achieved **competitive performance** and demonstrated a robust TensorFlow-based workflow for binary classification.

The project highlights an important industry insight:

> **For structured tabular business data, well-tuned classical Machine Learning models often perform as well as or better than Deep Learning models, while requiring significantly lower computational resources.**

This repository reflects a **professional AI/ML engineering workflow** involving **data preprocessing, predictive modeling, TensorFlow-based deep learning, performance evaluation, visualization, and business-focused interpretation**, making it suitable for **academic submission, portfolio projects, internships, and entry-level AI/ML roles**.


# Author

**Govind Kumar**
**expert.govindjha@gmail.com**

**AI/ML & Data Analytics Enthusiast**

* Python | SQL | TensorFlow | Scikit-learn | Deep Learning | Business Analytics

If you found this project useful, consider **starring the repository** and connecting for collaboration on **AI, Machine Learning, Deep Learning, and Data Analytics projects**.
