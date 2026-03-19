import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
def load_labeled_data(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(":::")
            if len(parts) >= 4:
                records.append({
                    "id": parts[0].strip(),
                    "title": parts[1].strip(),
                    "genre": parts[2].strip(),
                    "description": parts[3].strip()
                })
    return pd.DataFrame(records)
def load_test_data(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(":::")
            if len(parts) >= 3:
                records.append({
                    "id": parts[0].strip(),
                    "title": parts[1].strip(),
                    "description": parts[2].strip()
                })
    return pd.DataFrame(records)
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text
train = load_labeled_data("train_data.txt")
print("Total training records:", len(train))
train["description"] = train["description"].apply(preprocess)
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train = tfidf.fit_transform(train["description"])
y_train = train["genre"]
nb = MultinomialNB()
nb.fit(X_train, y_train)
test = load_test_data("test_data.txt")
test["description"] = test["description"].apply(preprocess)
X_test = tfidf.transform(test["description"])
test["genre"] = nb.predict(X_test)
print("\nSample Output:")
print(test[["title", "genre"]].head(10))
actual = load_labeled_data("test_data_solution.txt")
acc = accuracy_score(actual["genre"], test["genre"])
print("\nAccuracy:", round(acc * 100, 2), "%")