# 📧 Email Spam Detection Using Support Vector Machine (SVM)

This project builds a machine learning model to classify emails as **spam** or **ham (not spam)** using a Support Vector Machine (SVM) classifier. The model is trained and evaluated on a labeled dataset of email messages, with a focus on **hyperparameter tuning**, **cross-validation**, and **performance evaluation**.

---

## 🧠 Problem Statement

Spam emails are unsolicited messages that clutter inboxes and may contain harmful content. An efficient spam detection system helps users automatically filter such emails. This project applies **SVM** to build a robust spam classifier.

---

## 📂 Dataset

- The dataset consists of labeled emails (`spam` or `ham`).
- Each email is represented using numerical features (preprocessed).
- CSV Format: Each row is an email with its features and a label column `spam` (1 for spam, 0 for ham).

---

## ⚙️ Technologies Used

- Python
- pandas, numpy
- scikit-learn (SVC, GridSearchCV, KFold)
- matplotlib (for visualization)

---

## 🔍 Key Steps

1. **Data Loading and Preprocessing**
   - Load dataset from CSV
   - Separate features and target (`spam`)
   - Scale features using `sklearn.preprocessing.scale`

2. **Train-Test Split**
   - Stratified split to preserve spam/ham ratio

3. **Model Training**
   - Use `SVC` with RBF kernel
   - Train initial model on training data

4. **Hyperparameter Tuning**
   - Tune `C` and `gamma` using `GridSearchCV` with 5-fold cross-validation
   - Best parameters: `C = 100`, `gamma = 0.0001`, `kernel = rbf`

5. **Evaluation**
   - Evaluate model on test set
   - Metrics: Accuracy, Precision, Recall (Sensitivity)
   - Plot train/test accuracy for each combination of hyperparameters

---

## 📈 Final Model Performance

| Metric       | Score     |
|--------------|-----------|
| Accuracy     | 92.75%    |
| Precision    | 93.35%    |
| Recall       | 88.77%m

✅ The model demonstrates strong performance, with high precision (low false positives) and high recall (low false negatives), making it suitable for real-world spam detection.

---

## 📊 Visualizations

- Accuracy trends across different combinations of `C` and `gamma`
- Train vs test accuracy plots for each gamma value

---

## 📌 Conclusion

The SVM-based spam detection model efficiently classifies emails with high accuracy and minimal misclassification. With proper tuning, SVM proves to be a powerful algorithm for binary classification problems like spam detection.


## 🙌 Acknowledgements

- Dataset: Public spam dataset (e.g., UCI ML Repository or similar)
- Tools: scikit-learn, matplotlib

---

## 📁 Project Structure

```bash
📦spam-detection-svm
 ┣ 📜Spam (1).csv
 ┣ 📜notebook.ipynb
 ┗ 📜README.md
