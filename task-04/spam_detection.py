import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_pred = nb.predict(X_test_tfidf)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, y_train)
lr_pred = lr.predict(X_test_tfidf)
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_tfidf, y_train)
svm_pred = svm.predict(X_test_tfidf)
print("Naive Bayes")
print("Accuracy:", round(accuracy_score(y_test, nb_pred) * 100, 2), "%")
print(classification_report(y_test, nb_pred, target_names=['Ham', 'Spam']))
print("Logistic Regression")
print("Accuracy:", round(accuracy_score(y_test, lr_pred) * 100, 2), "%")
print(classification_report(y_test, lr_pred, target_names=['Ham', 'Spam']))
print("Support Vector Machine")
print("Accuracy:", round(accuracy_score(y_test, svm_pred) * 100, 2), "%")
print(classification_report(y_test, svm_pred, target_names=['Ham', 'Spam']))
models = {
    'Naive Bayes': accuracy_score(y_test, nb_pred),
    'Logistic Regression': accuracy_score(y_test, lr_pred),
    'Support Vector Machine': accuracy_score(y_test, svm_pred)
}
best = max(models, key=models.get)
print("Best Model:", best, "with accuracy", round(models[best] * 100, 2), "%")
sample = ["Congratulations! You won a free iPhone. Click here to claim now.",
          "Hey, are we still meeting for lunch today?"]
sample_tfidf = tfidf.transform(sample)
print("\nSample Predictions:")
for msg, pred in zip(sample, svm.predict(sample_tfidf)):
    print(f"'{msg}' --> {'Spam' if pred == 1 else 'Ham'}")