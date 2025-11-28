import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import ollama  
import joblib

# Function to embed sentences
def embed(sentence):
    response = ollama.embed("nomic-embed-text", "classification: " + sentence)
    return response['embeddings'][0]

# Load the dataset
df_train = pd.read_csv('./dataset/bigger_dataset.csv',sep=";") 
df_test = pd.read_csv('./dataset/test_set.csv',sep=";") 


embeddings_train = np.array([embed(sentence) for sentence in df_train['sentence']])
embeddings_test = np.array([embed(sentence) for sentence in df_test['sentence']])


n_components = 32

# Apply PCA to reduce dimensions
pca = PCA(n_components=n_components)
X_train = pca.fit_transform(embeddings_train)
X_test = pca.transform(embeddings_test)

y_train = df_train['label']
y_test = df_test['label']




# Train an SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)  # You can change the kernel to 'rbf' or others
svm_classifier.fit(X_train, y_train)

# Predict using the SVM classifier
y_pred_svm = svm_classifier.predict(X_test)

# Evaluate the SVM classifier
print("SVM Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_svm, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_svm, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_svm, average='weighted'):.4f}")
print(classification_report(y_test, y_pred_svm))

# Train a Logistic Regression classifier
logistic_classifier = LogisticRegression(max_iter=1000, random_state=42)  # Increase max_iter if needed
logistic_classifier.fit(X_train, y_train) # by deafult is regularized with L2 penalty

# Predict using the Logistic Regression classifier
y_pred_logistic = logistic_classifier.predict(X_test)

# Evaluate the Logistic Regression classifier
print("\nLogistic Regression Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logistic):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_logistic, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_logistic, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_logistic, average='weighted'):.4f}")
print(classification_report(y_test, y_pred_logistic))


joblib.dump(pca, './models/pca_model.pkl')
joblib.dump(svm_classifier, './models/svm_model.pkl')
joblib.dump(logistic_classifier, './models/logistic_model.pkl')