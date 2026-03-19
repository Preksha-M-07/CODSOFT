import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
fraud_train = pd.read_csv("fraudTrain.csv")
fraud_test = pd.read_csv("fraudTest.csv")
fraud_train = fraud_train.sample(frac=0.05, random_state=10)
fraud_test = fraud_test.sample(frac=0.05, random_state=10)
cols_to_remove = ['Unnamed: 0','trans_date_trans_time','cc_num','first','last',
                  'street','city','state','dob','trans_num']
fraud_train.drop(columns=cols_to_remove, inplace=True, errors='ignore')
fraud_test.drop(columns=cols_to_remove, inplace=True, errors='ignore')
fraud_train = pd.get_dummies(fraud_train)
fraud_test = pd.get_dummies(fraud_test)
fraud_train, fraud_test = fraud_train.align(fraud_test, join='left', axis=1, fill_value=0)
features = [col for col in fraud_train.columns if col != 'is_fraud']
X_tr = fraud_train[features]
y_tr = fraud_train['is_fraud']
X_te = fraud_test[features]
y_te = fraud_test['is_fraud']
clf = RandomForestClassifier(n_estimators=5, max_depth=5, n_jobs=-1, random_state=10)
clf.fit(X_tr, y_tr)
result = clf.predict(X_te)
print("Accuracy:", round(accuracy_score(y_te, result) * 100, 2), "%")