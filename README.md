# Breast Cancer Classifier

a simple benchmarking experiment for breast cancer diagnosis using the **Wisconsin Breast Cancer dataset**.  

## What I did
- Preprocessed the dataset by encoding the diagnosis labels (`M = 1` for malignant, `B = 0` for benign).
- Trained and tested **Logistic Regression** and **Support Vector Machine (SVM)** models.
- Evaluated the models using accuracy, confusion matrix, and classification reports.

## Results
Both models gave around **96% accuracy**, showing they perform well on this dataset.

Ah! Got it ✅ you want a **short practical “how to run” section** for readers. Here’s an updated version of your README with that included:

- **Dataset**: [Breast Cancer Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)  
- **Workflow**: Data preprocessing → Train/test split → Model training → Evaluation  

## How to Run
1. Clone this repository:
   ```bash
   git clone (https://github.com/yab-g4u/classifier.git)
   cd classifier

2. Install dependencies:

   ```bash
   pip install -r requirements.txt

   ```
3. Download the Kaggle dataset and place the CSV file in this folder.
4. Run the classifier:

   ```bash
   python logistic_svm.py
   ```
5. Check the console output for **accuracy, confusion matrix, and classification reports**.

## Results

Both models gave around **96% accuracy**, showing they perform well on this dataset.

