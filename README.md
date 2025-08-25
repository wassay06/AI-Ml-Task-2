# For task 2

# End-to-End ML Pipeline for Telco Customer Churn Prediction

## 🎯 Objective

[cite_start]The primary objective of this project was to build a reusable and production-ready machine learning pipeline for predicting customer churn using the Telco Churn Dataset[cite: 29]. [cite_start]This task was completed as part of the AI/ML Engineering Internship at Developers Hub Corporation[cite: 5, 27].

## 🧠 Methodology / Approach

[cite_start]An end-to-end machine learning pipeline was constructed using `scikit-learn`'s Pipeline API, which encapsulates all data preprocessing and modeling steps into a single, cohesive object[cite: 25, 31].

1.  **Data Loading and Preprocessing**: The Telco Churn dataset was loaded and preprocessed. The `TotalCharges` column was cleaned, and the `Churn` target variable was converted to a binary format.
2.  **Feature Engineering**: Categorical and numerical features were identified. A `ColumnTransformer` was used to apply a `StandardScaler` to numerical data and a `OneHotEncoder` to categorical data, ensuring the model receives appropriately scaled and encoded features.
3.  **Model Training and Hyperparameter Tuning**: A `scikit-learn` `Pipeline` was created to chain the preprocessor with a classifier. [cite_start]`GridSearchCV` was then used to perform an exhaustive search for the best hyperparameters across two different models: **Logistic Regression** and **Random Forest Classifier**[cite: 32, 33]. This ensures the selection of the most optimal model configuration.
4.  [cite_start]**Pipeline Export**: The best-performing model, along with all its preprocessing steps, was exported using `joblib`[cite: 34]. [cite_start]This practice allows the complete pipeline to be saved and later loaded for making predictions on new, unseen data, which is a crucial aspect of building production-ready systems[cite: 38, 39].

## 📊 Key Results / Observations

* The `GridSearchCV` identified **Logistic Regression** as the best-performing model for this task.
* The best cross-validation accuracy achieved during training was approximately **81%**.
* The final model's performance on the unseen test set was an accuracy of **79%**, demonstrating its generalization ability.
* The entire trained pipeline is saved as `telco_churn_pipeline.joblib`, ready for deployment.

---
### **Skills Gained**

This task provided hands-on experience with:
* [cite_start]ML pipeline construction [cite: 36]
* [cite_start]Hyperparameter tuning with `GridSearchCV` [cite: 37]
* [cite_start]Model export and reusability [cite: 38]
* [cite_start]Production-readiness practices [cite: 39]


# For TASK 5

# Task 5: Auto Tagging Support Tickets Using LLM

## Objective
The goal of this project is to develop an automated system for tagging support tickets using Large Language Models (LLMs). The system employs zero-shot classification and fine-tuning techniques to assign relevant tags (e.g., Technical issue, Billing inquiry) to free-text support ticket descriptions. The solution leverages Hugging Face Transformers and includes model evaluation and deployment readiness.

## Dataset
- **Source**: Customer Support Ticket Dataset (downloaded from Kaggle: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset, file: `customer_support_tickets.csv`).
- **Description**: The dataset contains approximately 4,850 rows with columns including `Ticket Description` (free-text) and `Ticket Type` (label). A subset (up to 500 rows) is used for faster processing.
  - **Training Set**: 80% of the subset (e.g., 400 rows).
  - **Test Set**: 20% of the subset (e.g., 100 rows).
- **Preprocessing**: Text data is tokenized, and labels are mapped to integer IDs for classification.

## Approach
- **Models**:
  - **Zero-Shot Classification**: Utilizes `facebook/bart-large-mnli` to predict tags without training, providing a baseline.
  - **Fine-Tuning**: Employs `distilbert-base-uncased` for supervised fine-tuning on the labeled dataset.
- **Techniques**:
  - Zero-shot classification uses a pre-trained model with candidate labels.
  - Fine-tuning involves training the model for 2-3 epochs with a learning rate of 2e-5 and batch size of 4.
  - Evaluation includes Accuracy and weighted F1-Score for both methods.
- **Output**: The system predicts the top 3 probable tags for each ticket, with scores where applicable.
- **Tools**: Python, Hugging Face Transformers, `datasets`, `scikit-learn`, `pandas`, and `joblib` for model saving.

## Results
- **Zero-Shot Classification**:
  - Accuracy: 1.0000 (based on limited dummy data or perfect match; may vary with full dataset).
  - F1-Score: 1.0000 (may indicate overfitting or small test set; update with real dataset results).
- **Fine-Tuned Model**:
  - Accuracy: 0.0000 (Current issue detected; likely due to label mapping or training convergence failure).
  - F1-Score: 0.0000 (Requires investigation; see Troubleshooting).
- **Note**: The fine-tuned model’s zero accuracy suggests a potential issue with label mapping, test set compatibility, or training configuration. Further debugging is recommended (see Troubleshooting).

## How to Run
### Prerequisites
- Python 3.8 or higher.
- Required libraries: `transformers`, `datasets`, `scikit-learn`, `pandas`, `torch`, `joblib`.
