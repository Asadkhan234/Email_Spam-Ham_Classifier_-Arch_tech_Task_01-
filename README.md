📨 Spam Email Classifier

A machine learning project to automatically detect spam emails using Logistic Regression and TF-IDF vectorization.

🚀 Features

Binary classification: Spam vs Not Spam (Ham)

Clean and professional ML pipeline

TF-IDF vectorization for text features

Label encoding for output labels

Accuracy evaluation on test data

Predict new emails with confidence score

🛠 Workflow

Data Cleaning – Prepare email text for ML

Label Encoding – Convert labels (spam / ham) to numeric (1 / 0)

Train/Test Split – 80% training, 20% testing

TF-IDF Vectorization – Fit on training data, transform test/user input

Train Model – Logistic Regression

Evaluate Accuracy – Test on unseen data

User Input Prediction – Take email text, predict spam or ham

📊 Model Evaluation

Accuracy measured on test set

Optional: Confusion matrix and classification report for precision, recall, and F1-score

Avoids data leakage by fitting vectorizer only on training data

💻 How to Use

Run the model training script

Enter new email text when prompted

Get prediction: Spam or Not Spam

Optional: Show confidence percentage

🧰 Tech & Libraries

Python

Pandas

Scikit-learn (Logistic Regression, TF-IDF, LabelEncoder, train_test_split)

NumPy
